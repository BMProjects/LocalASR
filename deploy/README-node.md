# LocalASR backends

Two machines, one job each.

```
麦克风 → 桌面（VAD、时间轴、保真校验）
            ├─ 音频 ──LAN──►  Orin      Qwen3-ASR-1.7B Q8_0   常驻
            └─ 文本 ─localhost─►  RTX 3050  Qwen3.5-4B Q4_K_M    常驻
```

Splitting them solved two problems at once. The Orin's 8 GB could hold ASR *or* a good
refiner, never both, so every refinement paid a ~35 s model swap; and the 2B that did fit
alongside needed two rounds of prompt work to become usable at all. Put the refiner on
the desktop's 4 GB card — which was otherwise idle, since ASR moved to the Orin — and
both machines hold one warm model.

Measured end to end, warm, five consecutive runs:

| | median | max |
|---|---:|---:|
| ASR (Orin, over LAN) | 0.90 s | 1.08 s |
| refinement (3050, localhost) | 0.92 s | 1.18 s |
| **total** | **1.82 s** | |

Against ~66 s for the same work with sequential loading on one board. The network carries
a few hundred characters of text; it is not the cost.

The desktop keeps the fidelity validator. Whatever a refiner returns — Unsloth Studio,
llama-server, anything speaking the same protocol — is checked against the original here,
so no backend can put unverified text in front of the user.

## Configuration

```toml
# ~/.config/localasr/config.toml
node_url  = "http://asr-node.local:8090"   # audio only
node_token = "..."

refiner_url   = "http://127.0.0.1:8091"     # text only
refiner_model = "Qwen3.5-4B-UD-Q4_K_XL"

local_asr_fallback = false
```

Both backends are reached by URL, and neither is started by the desktop. Changing which
model answers, or which machine answers, is a line in this file — the application has no
opinion about either, and holds no weights of its own.

`local_asr_fallback = false` matters. With the refiner resident on the 4 GB card, an
unreachable Orin must not cause a local ASR model to start beside it — that runs the card
out of memory silently, mid-sentence. The node being down is reported as exactly that.

## The refiner host (text only)

`localasr-refiner` is the process that holds the refinement model. It is deliberately
separate from the desktop: a model whose lifetime is a window dies when the window does,
lives only on the machine running the GUI, and is configured by application settings
rather than by whoever owns the hardware.

```bash
localasr-refiner                    # the configured model on 127.0.0.1:8091
localasr-refiner --model local-x    # a specific catalog or imported id
```

It imports nothing from the frontends, the capture stack or the apps — there is a test
for exactly that — so it starts on a box with a GPU and no audio hardware or display.

**By default nothing is installed.** The desktop spawns this module when it starts and
terminates it when it exits, so a working setup needs no unit file, no `systemctl`, and
no entry in `~/.config/systemd/user`. Leave `refiner_url` unset and that is what happens.

The application still holds no model-loading code: it spawns a process by name and
speaks HTTP to it. The port is fixed (8091) rather than ephemeral, which is what lets the
same module be run any of three ways without the application knowing the difference.

If you would rather the model outlive the window — several sessions a day, or a shared
machine — install the unit instead and point `refiner_url` at it:

```bash
cp deploy/localasr-refiner.service ~/.config/systemd/user/   # optional
systemctl --user enable --now localasr-refiner
```

Then `refiner_url = "http://127.0.0.1:8091"` in config.toml stops the desktop spawning
its own. If the port is already answering when the desktop starts, it uses what is there
and does not start a second copy — and does not stop it on exit either, since it is not
its to stop.

Serving it to another machine needs a token; binding to anything but loopback without
one is refused. llama-server has no access control of its own, and an open
`/v1/chat/completions` is an open door into that machine's memory budget — one that
fails silently, since everything keeps working, slowly.

```bash
localasr-refiner --bind 0.0.0.0 --token "$(openssl rand -hex 16)"
```

### Using a model another tool already downloaded

Weights managed by something else — Unsloth Studio, an existing HF cache — can be linked
instead of copied, which is what this machine does:

```bash
localasr models import ~/.cache/huggingface/.../Qwen3.5-4B-UD-Q4_K_XL.gguf \
    --link --kind llm --name qwen3_5-4b-unsloth
```

`--kind llm` is not optional in practice: without it the file is registered as an ASR
model and never appears as a refiner. `--link` keeps one copy on disk; removing the
import later unlinks it and never touches the original.

## Both models follow the session

Neither model is resident when the application is not running, and neither needs a
command to make that true. They get there differently, because they are in different
places:

| | how it starts | how it stops |
|---|---|---|
| refiner (local) | the app spawns `localasr.refine.host` | the child is terminated |
| ASR (Orin) | the app asks the node to load | the app asks it to release |

There is no remote process to spawn and no signal to send, so on the far side the
lifetime is bound at the layer where the cost actually is. Measured on the Orin:

| | MemAvailable |
|---|---:|
| model resident | 2625 MiB |
| model released | 5427 MiB |
| the node service itself | 42–61 MiB RSS |

The model is 2799 MiB of it; the service is under 100. So residency follows the session
by default, and the service is left alone — it costs almost nothing idle, survives a
network blip mid-sentence, and keeps the node usable by anything else on the LAN.

**Quitting hands the model back**, whether or not this session was the one that loaded
it. An earlier version released only what it had loaded itself, reasoning that unloading
somebody else's model is rude — sound in the abstract, and it meant the feature never
fired. The node loads on demand, so the model becomes resident the first time anyone
transcribes; from then on every launch found it warm, adopted it, and released nothing.

Set `node_release_on_exit = false` for a node that genuinely has more than one client,
where the last one to quit should not unload a model somebody else is mid-sentence with.
The node has no notion of sessions, so "already resident" cannot tell "somebody is using
this" from "I left it there yesterday" — only the deployment knows which it is.

### Having the app start the node too

Optional, and only worth it if you want the board completely idle between sessions:

```toml
node_ssh = "user@asr-node.local"     # needs key-based SSH
node_service = "localasr-node"      # the unit name, if it differs
```

With this set, an unresponsive node is started through `systemctl --user start` over SSH
and stopped again on exit. systemd on the far side, deliberately, rather than a command
held open through an SSH pipe: a pipe would take the node down with any network blip, and
would move the remote paths, interpreter and environment into this machine's config —
all of which the unit already holds. The unit can then be left installed but **not
enabled**, so nothing starts at boot.

## The Orin node (ASR only)

# LocalASR compute node (Jetson Orin Nano Super)

## Why headless

The node is a compute role: it has no display and no microphone. On 8 GB of *unified*
memory the desktop session is not free — measured on this machine:

| | MemAvailable |
|---|---:|
| graphical.target (GNOME + Xorg + NX) | 4180 MiB |
| multi-user.target | **5875 MiB** |

That 1.7 GiB is the difference between holding one model and holding two, which in turn
is the difference between a refinement taking ~2 s and taking ~35 s, because taking
turns means reloading a multi-gigabyte model on every switch.

## Bringing the desktop back when you want it

Headless is the default, not a one-way door. NX is a separate service and still starts
on demand:

```bash
sudo systemctl isolate graphical.target     # now, until reboot
sudo systemctl set-default graphical.target # permanently, takes effect next boot
```

Going back to headless is the same two commands with `multi-user.target`. SSH is
unaffected either way — `sshd` belongs to multi-user.target, so it survives both.

## Deploying

```bash
LOCALASR_NODE_TOKEN=$(openssl rand -hex 16) ./deploy/orin-node.sh user@asr-node.local
```

The node runs as a systemd **user** service (`~/.config/systemd/user/localasr-node.service`)
rather than from an SSH command. Backgrounding over SSH was tried and does not survive
the connection closing, whatever combination of `nohup`, `setsid` and redirection is
used; systemd also restarts it on failure and gives it a stop timeout long enough that a
model load is never interrupted half-way, which would otherwise leave a llama-server
holding memory nothing will reclaim.

The token lives in `~/.config/localasr-node.env`, mode 600, not in the unit file — unit
files are world-readable.

## The refiner (RTX 3050)

Started and stopped from the frontend, not by a service unit. `refiner_model_id` in
config.toml names a catalog model and the desktop runs llama-server for it itself, which
is what makes 「启动」/「卸载」 mean anything: the point is holding the model across a
whole session and releasing the card afterwards, and a unit that comes up at boot and
stays up cannot do that.

Uses the same GGUF the Orin was measured against — same hash, same prompt, comparable
validator results. 2899 MiB of the 3050's 4096, with a desktop session running.

Vulkan compiles its shaders on the first request: ~28 s once, then a 0.9 s median. Warm
it before judging any latency number.

Set `refiner_url` instead to point at a server somebody else runs — Unsloth Studio, a
llama-server under systemd. It then becomes read-only from the frontend, and the buttons
say so rather than doing nothing. If you use Unsloth Studio, start it with
`--disable-tools`: its search, Python and terminal tools run with the invoking user's
rights, and refinement needs none of them.

## `--no-dev` belongs on every uv command

The dev dependency group pulls `localasr[capture,gui,node]` so that `uv run pytest`
works on a development machine. A bare `uv run` on the node re-resolves with it and
quietly installs PySide6, onnxruntime, soxr and sounddevice onto a headless Jetson —
observed taking the venv from 74 MB to 802 MB. `uv sync --no-dev` alone does not
prevent this; the flag has to be on the `uv run` too.
