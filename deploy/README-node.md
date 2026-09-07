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

The desktop keeps both texts. Whatever a refiner returns — LM Studio, llama-server,
anything speaking the same protocol — is shown beside the transcript it came from rather
than in place of it, so no backend can quietly replace what was said.

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

Weights managed by something else — LM Studio, an existing HF cache — can be linked
instead of copied, which is what this machine does:

```bash
localasr models import \
    ~/.lmstudio/models/unsloth/Qwen3.5-4B-MTP-GGUF/Qwen3.5-4B-UD-Q4_K_XL.gguf \
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
output. 2899 MiB of the 3050's 4096, with a desktop session running.

Vulkan compiles its shaders on the first request: ~28 s once, then a 0.9 s median. Warm
it before judging any latency number.

Set `refiner_url` instead to point at a server somebody else runs — LM Studio, a
llama-server under systemd. It then becomes read-only from the frontend, and the buttons
say so rather than doing nothing.

### LM Studio

`refiner_url = "http://127.0.0.1:1234"` — no `/v1`, the client appends it. Its defaults
(`~/.lmstudio/.internal/http-server-config.json`) are the right ones: port 1234,
`networkInterface: 127.0.0.1`, and `justInTimeModelLoading`, which loads the weights on
the first request. `autoStartOnLaunch` is off, so the server still has to be started once
per run — `lms server start`, or Developer → Start Server.

llmster is a service, not a window. It is its own 151 MB binary under
`~/.lmstudio/llmster/`, links no X11, Wayland or GTK library, opens no display file
descriptor, and starts with `DISPLAY`, `WAYLAND_DISPLAY`, `XDG_SESSION_TYPE` and
`XDG_RUNTIME_DIR` all unset. It holds the listening socket itself (`llmster` is the
process `ss` names on 127.0.0.1:1234). Idle cost: ~450 MiB RSS and 14 MiB of VRAM once
the model is unloaded.

LM Studio is the one external refiner the 「启动」/「卸载」 buttons still work for, because
it exposes residency as part of its API rather than as an implementation detail:

    GET  /api/v1/models          every downloaded model, and its `loaded_instances`
    POST /api/v1/models/load     {"model": "<key>"}
    POST /api/v1/models/unload   {"instance_id": "<id>"}

Which is the same shape as this node's own load/release, so it is driven by the same kind
of object. Nothing has to be configured for it: `GET /api/v1/models` is what identifies an
LM Studio, and llama-server answers 404 there. Two settings exist only for the cases the
probe cannot decide:

- `refiner_model` doubles as the model **key** here (`unsloth/Qwen3.5-4B-MTP-GGUF`), for
  when LM Studio has more than one LLM. Left at its default it is a label, not a key, and
  is not sent as one.
- `refiner_release_on_exit = false` for an LM Studio whose own chat window is in use —
  otherwise closing this window takes its model with it.

JIT loading means the model buttons are about latency, not correctness: refinement works
either way, but nothing except this application will unload the model afterwards, and a
4B Q4 holds ~2.9 GB of a 4 GB card until something does.

The HTTP front end is a different matter — it is off unless `autoStartOnLaunch` is set,
so "not started" is the state most sessions begin in. When the URL is loopback, on the
port LM Studio's own config names, and `lms` is on this machine, 「启动」 runs LM Studio's
documented recipe before loading anything:

    lms daemon up        # ExecStartPre, in their systemd unit
    lms server start     # ExecStart

and 「卸载」 winds back exactly what it started, in reverse: unload the model, `lms server
stop`, `lms daemon down`. Each step is conditional on this session having caused it. A
daemon that was already up is somebody else's — possibly with somebody else's model in
it — and `lms daemon down` ends every client's session, not just this one, so it is asked
first (`lms daemon status --json`, whose exit code is 0 either way and whose body carries
the answer).

Cold to cold, measured, nothing else running:

    start_refiner()   5.8 s   daemon up + server start + model load (3.1 s)
    refine            2.4 s   到出 -> 导出
    shutdown          0.8 s   unload + server stop + daemon down
    afterwards                {"status":"not-running"}, 14 MiB VRAM

**That needs the headless daemon, not just the CLI.** The desktop app installs `lms`, but
`lms daemon up` then wakes the GUI application, which does not accept `--run-as-service`
and times out after ~60 s. Install llmster and the same command starts a headless service
instead:

    curl -fsSL https://lmstudio.ai/install.sh | bash

(`ldconfig` must be on `PATH` — on Debian it lives in `/usr/sbin`, which a user shell does
not always include, and the installer stops with a clear message if it is missing.)

Measured afterwards, no GUI anywhere:

    lms daemon up                    2.3 s      llmster v0.0.23+1
    lms daemon down                  0.2 s
    lms server start                 ~1 s       port 1234 (a no-op when
                                                autoStartOnLaunch already did it)
    model load, cold                 3.4 s      qwen3.5-4b-mtp
    refinement, warm                 2.4 s      到出 -> 导出
    unload on exit                   0.5 s      3734 MiB -> 14 MiB

If llmster is absent the error says so and names both fixes — installing it, or opening
the desktop app and letting its `autoStartOnLaunch` bring the server up. A bare
"timed out" would be a dead end.

A URL that does not answer is not evidence that it is not LM Studio, and treating it as
such is what once left both buttons grey for the rest of a session with nothing able to
re-enable them. Only a server that answers *and is something else* is ruled out.

## `--no-dev` belongs on every uv command

The dev dependency group pulls `localasr[capture,gui,node]` so that `uv run pytest`
works on a development machine. A bare `uv run` on the node re-resolves with it and
quietly installs PySide6, onnxruntime, soxr and sounddevice onto a headless Jetson —
observed taking the venv from 74 MB to 802 MB. `uv sync --no-dev` alone does not
prevent this; the flag has to be on the `uv run` too.
