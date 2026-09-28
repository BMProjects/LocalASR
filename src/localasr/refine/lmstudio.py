"""LM Studio's own server, driven the way the recognition node is.

There are two ways to put a refinement model in front of this application, and until now
only one of them could be turned on and off from it. `refine.host` spawns llama-server as
a child, so the button that starts it is a `Popen`. An external `refiner_url` was the
other way, and it was read-only by definition: a URL is not a process, and there is no
signal to send to somebody else's server.

LM Studio breaks that assumption, because its server exposes model residency as part of
the API rather than as an implementation detail:

    GET  /api/v1/models          every downloaded model, and its `loaded_instances`
    POST /api/v1/models/load     {"model": "<key>"}       → {"instance_id", "status"}
    POST /api/v1/models/unload   {"instance_id": "<id>"}  → {"instance_id"}

That is the same shape as the compute node's own load/release endpoints, which is why
this class is the same shape as `NodeCompanion`: `answers`, what is resident, `start`
for the session, `stop` afterwards. Nothing new is invented; a second server that can
answer the same questions gets the same object.

**Loading is about latency, not correctness.** LM Studio's `justInTimeModelLoading` is on
by default, so a refinement request loads the model by itself. What it cannot do is load
it *before* the user is waiting for it, and it will not unload it when this application
exits — a 4B Q4 holds ~2.9 GB until something says otherwise. Both halves are the point.

Docs: https://lmstudio.ai/docs/api/rest-api (load, unload, list). Default port 1234.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import httpx

DEFAULT_URL = "http://127.0.0.1:1234"
"""LM Studio's server port, and loopback unless the user changes `networkInterface`."""

DEFAULT_PORT = 1234
SERVER_CONFIG = "~/.lmstudio/.internal/http-server-config.json"
"""Where LM Studio records the port it serves on. Read rather than assumed, because a
user who moved it would otherwise get a start button that starts the wrong thing."""

CLI = "~/.lmstudio/bin/lms"
"""Where the installer puts `lms`. `PATH` is tried first — this is the fallback for a
desktop launcher, which does not inherit a login shell's exports."""

PROBE_TIMEOUT = 3.0
LOAD_TIMEOUT = 300.0
"""A cold load reads several gigabytes off disk before it answers."""

SERVER_SETTLE = 3.0
"""How long to give `autoStartOnLaunch` to bring the server up with the daemon before
concluding it will not, and starting it ourselves."""

SERVICE_TIMEOUT = 120.0
"""`lms daemon up` waits ~60 s on its own before giving up."""

LOOPBACK = ("127.0.0.1", "localhost", "::1", "")

LOAD_CONFIG = {"context_length": 4096, "parallel": 1, "eval_batch_size": 512}
"""What the refiner is loaded with, instead of LM Studio's per-model defaults.

Those defaults are 8192 context x 4 parallel slots x 2048-token batches, and on a 4 GB
card they leave ~280 MiB for everything else. Loading succeeds, then the first prompt's
matmul needs more than that: CUDA OOM, the engine process dies, the request comes back
400, the model is gone, and the next load does it all again. Measured on the RTX 3050:
3790 MiB with the defaults, 3484 MiB with these, and a 750-character refinement that
crashed before now runs in 4.3 s.

4096 is what `refine.client` budgets for (DEFAULT_CONTEXT); a dictation client sends one
request at a time, so one slot. `parallel` is not in the documented load schema but is
honoured — `echo_load_config` reports it back.
"""


class LMStudioError(RuntimeError):
    """The server answered, and the answer was no."""


def _headers(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def find_cli() -> Path | None:
    """`lms`, LM Studio's own CLI, if this machine has it."""
    found = shutil.which("lms")
    if found:
        return Path(found)
    fallback = Path(CLI).expanduser()
    return fallback if os.access(fallback, os.X_OK) else None


def configured_port() -> int:
    """The port LM Studio serves on, from its own config file."""
    try:
        raw = json.loads(Path(SERVER_CONFIG).expanduser().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return DEFAULT_PORT
    port = raw.get("port")
    return port if isinstance(port, int) else DEFAULT_PORT


def daemon_running() -> bool | None:
    """Whether llmster is already up. `None` when `lms` cannot answer.

    `lms daemon status --json` prints `{"status": "running"|"not-running"}` and exits 0
    either way, so the exit code says nothing and the body says everything.
    """
    cli = find_cli()
    if cli is None:
        return None
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [str(cli), "daemon", "status", "--json"],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT * 4,
        )
        return json.loads(result.stdout).get("status") == "running"
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def probe(url: str, token: str | None = None) -> bool | None:
    """Whether `url` is LM Studio. `None` means it did not answer, which is not a no.

    Every server this application can talk to serves `/v1/chat/completions`; only LM
    Studio also serves `/api/v1/models` returning a `models` list. llama-server answers
    404 there — an unambiguous no, and one fewer setting for the user to get wrong.

    The third case is the one that matters here. A server that is simply not started yet
    is exactly what the start button exists for, and reading its silence as "not LM
    Studio" is what left those buttons grey with nothing able to turn them on again.
    """
    try:
        with httpx.Client(timeout=PROBE_TIMEOUT) as client:
            response = client.get(f"{url.rstrip('/')}/api/v1/models", headers=_headers(token))
    except httpx.HTTPError:
        return None
    try:
        return response.status_code == 200 and isinstance(response.json().get("models"), list)
    except ValueError:
        return False


def speaks_lmstudio(url: str, token: str | None = None) -> bool:
    """A running LM Studio at `url`. Silence counts as no."""
    return probe(url, token) is True


def startable_here(url: str) -> bool:
    """Whether a silent `url` is an LM Studio this machine can start.

    Three conditions, all necessary. The host must be loopback, because `lms` starts the
    server on this machine and nowhere else. The port must be the one LM Studio is
    configured to serve on, or `lms server start` would bring up something the URL does
    not point at. And `lms` must exist.
    """
    parsed = urlparse(url if "//" in url else f"//{url}")
    if parsed.hostname not in LOOPBACK:
        return False
    if (parsed.port or DEFAULT_PORT) != configured_port():
        return False
    return find_cli() is not None


@dataclass
class LMStudioCompanion:
    """Hold LM Studio's refinement model for the session, and hand it back afterwards."""

    url: str = DEFAULT_URL
    token: str | None = None
    model: str | None = None
    """The model key, e.g. ``unsloth/Qwen3.5-4B-MTP-GGUF``. When unset, the one LLM this
    LM Studio has is not a guess; several are ambiguous and say so."""

    load_config: dict[str, int] = field(default_factory=lambda: dict(LOAD_CONFIG))
    autostart: bool = True
    """Bring the server up with `lms` when it is not answering.

    The recognition node has the same option under `node_ssh`, for the same reason and
    at the same cost: without it a companion can only talk to a service somebody else
    started, and "starts with the app" is half true. `lms` is the local equivalent of
    that ssh call — LM Studio's own CLI, doing what its documented systemd unit does.
    """

    release_on_exit: bool = True
    """Unload on exit even if this session did not load it.

    The same deployment fact `NodeCompanion` carries, for the same reason: the server has
    no notion of sessions, so "already resident" cannot distinguish "somebody else is
    using this" from "I left it there yesterday". Turn it off when LM Studio is genuinely
    shared — with its own chat window, for instance."""

    _instance: str | None = field(default=None, init=False, repr=False)
    """The instance `start` left loaded or adopted. Nothing else is ever unloaded."""

    _loaded: bool = field(default=False, init=False, repr=False)
    """Whether *we* made it resident — the only thing unloaded when `release_on_exit`
    is off."""

    _started_server: bool = field(default=False, init=False, repr=False)
    """Whether *we* started the server. Only then is it ours to stop."""

    _started_daemon: bool = field(default=False, init=False, repr=False)
    """Whether *we* started llmster itself. Same rule one layer down, and it matters
    more here: `lms daemon down` ends every client's session, not just ours."""

    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    """`start` and `stop` run on worker threads — the startup thread, the panel's
    button — and two concurrent starts meant two loads racing, one of which restarted
    the server under the other's in-flight request."""

    def __post_init__(self) -> None:
        self.url = self.url.rstrip("/")

    def _get(self, path: str) -> dict:
        with httpx.Client(timeout=PROBE_TIMEOUT) as client:
            response = client.get(f"{self.url}{path}", headers=_headers(self.token))
        response.raise_for_status()
        return response.json()

    def answers(self) -> bool:
        try:
            self._get("/api/v1/models")
        except (httpx.HTTPError, ValueError):
            return False
        return True

    def catalog(self) -> list[dict]:
        """Every model LM Studio has downloaded, LLMs and embedding models alike."""
        return self._get("/api/v1/models").get("models", [])

    def instances(self) -> dict[str, dict]:
        """Loaded instance id → the config it was loaded with."""
        return {
            instance["id"]: instance.get("config") or {}
            for entry in self.catalog()
            for instance in entry.get("loaded_instances", [])
            if "id" in instance
        }

    def fits(self, config: dict) -> bool:
        """Whether an instance loaded with `config` is no heavier than ours.

        Anything larger is the configuration that runs out of memory on the first
        prompt, so adopting it only postpones the failure to the moment the user asks
        for a refinement.
        """
        return all(
            config.get(name, 0) <= limit for name, limit in self.load_config.items()
        )

    def resident(self) -> tuple[str, ...]:
        """Instance ids currently loaded. Empty is the normal state before a session."""
        return tuple(
            instance["id"]
            for entry in self.catalog()
            for instance in entry.get("loaded_instances", [])
            if "id" in instance
        )

    def key(self) -> str:
        """Which model to load.

        `model` when the user named one. Otherwise the same rule `refine.host` applies to
        this machine's own catalog: exactly one candidate is an answer, none and several
        are questions — asked here rather than resolved into whichever came first.
        """
        if self.model:
            return self.model
        llms = [entry for entry in self.catalog() if entry.get("type") == "llm"]
        if len(llms) == 1:
            return llms[0]["key"]
        if not llms:
            raise LMStudioError(f"{self.url} 上没有可用于整理的模型，请先在 LM Studio 里下载一个")
        names = "、".join(entry.get("key", "?") for entry in llms)
        raise LMStudioError(f"LM Studio 上有多个模型（{names}），请用 refiner_model 指定一个")

    def start(self) -> str:
        """Load the model now, so the first refinement is not the one that waits.

        Returns what happened, in words the status panel can show.
        """
        with self._lock:
            return self._start()

    def _start(self) -> str:
        if not self.answers():
            if not (self.autostart and startable_here(self.url)):
                raise LMStudioError(
                    f"LM Studio 服务 {self.url} 无响应。请在 LM Studio 里启动本地服务器"
                    "（Developer → Start Server，或 `lms server start`）。"
                )
            self._start_server()
        key = self.key()
        loaded = self.instances()
        if key in loaded:
            if self.fits(loaded[key]) or not self.release_on_exit:
                # Found it warm. Adopted, not loaded: whether it is ours to unload
                # afterwards is what `release_on_exit` answers.
                self._instance = key
                return f"{key}（LM Studio 上已加载）"
            # Loaded with LM Studio's defaults — by its own JIT, by hand, by an older
            # version of this module. That is the instance that dies on the first
            # prompt, so it is replaced rather than adopted.
            self._unload(key)

        body: dict[str, object] = {"model": key, **self.load_config}
        try:
            with httpx.Client(timeout=LOAD_TIMEOUT) as client:
                response = client.post(
                    f"{self.url}/api/v1/models/load", json=body, headers=_headers(self.token)
                )
            if response.status_code >= 400:
                raise LMStudioError(f"LM Studio 拒绝加载 {key}：{_reason(response)}")
            payload = response.json()
        except httpx.HTTPError as exc:
            raise LMStudioError(f"加载 {key} 失败：{exc}") from exc

        self._instance = payload.get("instance_id", key)
        self._loaded = True
        seconds = payload.get("load_time_seconds")
        return f"{self._instance}（{seconds:.1f}s）" if seconds else str(self._instance)

    def stop(self) -> None:
        """Unload, so the card is not still holding a refiner after the window closes."""
        with self._lock:
            self._stop()

    def _unload(self, instance: str) -> None:
        with httpx.Client(timeout=LOAD_TIMEOUT) as client:
            client.post(
                f"{self.url}/api/v1/models/unload",
                json={"instance_id": instance},
                headers=_headers(self.token),
            )

    def _stop(self) -> None:
        instance, loaded = self._instance, self._loaded
        self._instance, self._loaded = None, False
        if instance is not None and (self.release_on_exit or loaded):
            # Unreachable at exit means its memory is its own problem now. Raising here
            # would only stop the application from closing.
            with contextlib.suppress(httpx.HTTPError):
                self._unload(instance)
        if self._started_server:
            self._started_server = False
            self._run_cli("server", "stop", check=False)
        if self._started_daemon:
            # Only a daemon this session started. `lms daemon down` ends every client's
            # session, not just this one, so adopting one and then shutting it down would
            # take somebody else's loaded model with it — the same mistake the node
            # companion is written to avoid, one layer further down.
            self._started_daemon = False
            self._run_cli("daemon", "down", check=False)

    def _start_server(self) -> None:
        """`lms daemon up` then `lms server start`, which is LM Studio's own recipe.

        Verbatim from the unit in its Linux startup docs — `daemon up` as ExecStartPre,
        `server start` as ExecStart. The daemon is the process that holds models; the
        server is the HTTP front end this application speaks to, and it is off by default
        (`autoStartOnLaunch: false`), so both steps are needed from cold.
        """
        if daemon_running():
            # Already up — somebody else's, so not ours to shut down afterwards.
            self._start_http_server()
            return
        try:
            self._run_cli("daemon", "up")
        except LMStudioError as exc:
            # `lms daemon up` wakes whatever `app-install-location.json` points at. On a
            # machine with only the desktop app and its CLI — no headless llmster — that
            # is a GUI application, and it times out after ~60 s with nothing started.
            # Observed exactly that here, so say which of the two fixes applies.
            raise LMStudioError(
                f"{exc}\n无法从命令行启动 LM Studio 服务。两个办法："
                "装上无界面守护进程 `curl -fsSL https://lmstudio.ai/install.sh | bash`，"
                "或者手动打开 LM Studio 桌面版（它的 autoStartOnLaunch 会带起服务器）。"
            ) from exc
        self._started_daemon = True
        self._start_http_server()

    def _start_http_server(self) -> None:
        """The HTTP front end, and the wait for it to answer.

        Separate from the daemon because the two have different owners: llmster may
        already be somebody else's, while the front end is cheap to start and stop.

        Asked first, and only started if silent. `lms server start` against a running
        server is not a no-op, as this once assumed: it stops and restarts it, and a load
        in flight at that moment dies with "Model load request cancelled by client
        disconnect" — observed at startup, where `autoStartOnLaunch` has usually brought
        the server up with the daemon a moment earlier.
        """
        if self._wait_until_answering(SERVER_SETTLE):
            return
        self._run_cli("server", "start")
        self._started_server = True
        if not self._wait_until_answering(SERVICE_TIMEOUT):
            raise LMStudioError(
                f"lms 已执行，但 {self.url} 在 {SERVICE_TIMEOUT:.0f}s 内没有就绪"
            )

    def _wait_until_answering(self, seconds: float) -> bool:
        end = time.monotonic() + seconds
        while True:
            if self.answers():
                return True
            if time.monotonic() >= end:
                return False
            time.sleep(0.5)

    def _run_cli(self, *args: str, check: bool = True) -> None:
        cli = find_cli()
        if cli is None:
            raise LMStudioError("找不到 lms（LM Studio 的命令行工具）")
        try:
            result = subprocess.run(  # noqa: S603 - fixed argv, no shell
                [str(cli), *args], capture_output=True, text=True, timeout=SERVICE_TIMEOUT
            )
        except (OSError, subprocess.SubprocessError) as exc:
            if not check:
                return
            raise LMStudioError(f"lms {' '.join(args)} 无法执行：{exc}") from exc
        if check and result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()[:200]
            raise LMStudioError(f"lms {' '.join(args)} 失败：{detail}")


def _reason(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip()[:200] or f"HTTP {response.status_code}"
    for key in ("error", "message", "detail"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, dict) and isinstance(value.get("message"), str):
            return value["message"]
    return f"HTTP {response.status_code}"
