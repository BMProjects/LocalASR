"""Lifecycle for the llama-server process that holds the loaded model.

STOPPED ──► STARTING ──► READY ──► STOPPING ──► STOPPED
                └──────► FAILED
"""

from __future__ import annotations

import enum
import os
import shutil
import signal
import socket
import subprocess
import time
from contextlib import closing
from pathlib import Path
from typing import IO

from localasr.core.engine.client import EngineError, TranscriptionClient
from localasr.registry.manager import (
    ModelKind,
    ModelSpec,
    is_downloaded,
    local_paths,
    model_dir,
    verified_builds,
)

STARTUP_TIMEOUT = 180.0
"""Generous: the first launch of a Vulkan build compiles shaders before serving."""

PORT_RETRIES = 3
"""A port free at selection can be taken before the server binds it."""


class SupervisorError(RuntimeError):
    pass


class State(enum.Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"
    FAILED = "failed"


def find_llama_server() -> Path:
    """Locate the llama-server binary.

    `LOCALASR_LLAMA_DIR` wins, then a vendored copy beside the repo, then PATH.
    """
    override = os.environ.get("LOCALASR_LLAMA_DIR")
    if override:
        candidate = Path(override).expanduser() / "llama-server"
        if candidate.is_file():
            return candidate
        raise SupervisorError(f"LOCALASR_LLAMA_DIR set but {candidate} is not a file")

    vendor = Path(__file__).resolve().parents[4] / "vendor"
    if vendor.is_dir():
        found = sorted(vendor.glob("*/llama-server"))
        if found:
            return found[-1]

    on_path = shutil.which("llama-server")
    if on_path:
        return Path(on_path)

    raise SupervisorError(
        "llama-server not found; set LOCALASR_LLAMA_DIR to a llama.cpp build directory"
    )


def installed_build(binary: Path) -> str | None:
    """The build id reported by `llama-server --version`, if it can be read."""
    try:
        proc = subprocess.run(
            [str(binary), "--version"],
            capture_output=True,
            timeout=15,
            env={**os.environ, "LD_LIBRARY_PATH": str(binary.parent)},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = (proc.stdout + proc.stderr).decode(errors="replace")
    for line in text.splitlines():
        if line.startswith("version:"):
            return line.split()[1].strip()
    return None


def _normalize_build(build: str | None) -> str | None:
    """Releases are tagged `b10333`; `--version` reports the bare `10333`."""
    return build.lstrip("b") if build else None


def check_build(binary: Path, runtime: str = "llama-server") -> str | None:
    """Warn text when the installed build is not one that has been exercised.

    A list rather than a single pin, because the desktop and the Jetson run different
    binaries by necessity — x86_64 Vulkan against a locally built aarch64 CUDA one — and
    a pin that only ever matches one of them warns on every start of the other, which
    trains people to ignore it.
    """
    expected = verified_builds(runtime)
    actual = installed_build(binary)
    if not expected or actual is None:
        return None
    if _normalize_build(actual) in {_normalize_build(e) for e in expected}:
        return None
    return (
        f"llama-server build {actual} is not among the verified {expected}; "
        "audio support is upstream-experimental, so decoding behaviour may differ"
    )


_OOM_MARKERS = (
    "out of memory",
    "failed to allocate",
    "cudaMalloc failed",
    "ErrorOutOfDeviceMemory",
    "unable to allocate backend buffer",
)


def _why_it_died(log_path: Path | None) -> str:
    """Turn an exit code into something a person can act on.

    "exited with code 1" is true and useless. On a 4 GB card the overwhelmingly likely
    cause is that something else is already holding the memory — another llama-server
    left running, a browser, a previous session that was not released — and saying so
    turns a dead end into an obvious next step.
    """
    if log_path is None or not log_path.is_file():
        return "未捕获日志"
    try:
        tail = log_path.read_text(errors="replace").splitlines()[-40:]
    except OSError:
        return "无法读取日志"

    for line in reversed(tail):
        if any(marker.lower() in line.lower() for marker in _OOM_MARKERS):
            return f"显存或内存不足（{line.strip()[:120]}）"
    errors = [line.strip() for line in tail if " E " in line or "error" in line.lower()]
    return errors[-1][:160] if errors else "日志中没有明确原因"


def _free_port() -> int:
    with closing(socket.socket()) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class EngineSupervisor:
    """Starts llama-server, waits for readiness, and shuts it down again.

    The context is deliberately small: llama-server's default allocation (4 slots of
    18688 tokens) costs about 2 GB of VRAM more than a single 4096-token slot, which is
    the difference between fitting and not fitting a 4 GB card.
    """

    def __init__(
        self,
        spec: ModelSpec,
        *,
        port: int | None = None,
        device: str | None = None,
        n_gpu_layers: int = 99,
        bind: str = "127.0.0.1",
        api_key: str | None = None,
    ) -> None:
        if spec.runtime != "llama-server":
            raise SupervisorError(f"unsupported runtime {spec.runtime!r} for {spec.model_id}")
        self.spec = spec
        self._requested_port = port
        self.port = port or 0
        self.device = device or os.environ.get("LOCALASR_DEVICE")
        self.n_gpu_layers = n_gpu_layers
        self.bind = bind
        self.api_key = api_key
        self.state = State.STOPPED
        self.warning: str | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._log: IO[bytes] | None = None

    @property
    def base_url(self) -> str:
        # The loopback address even when bound to 0.0.0.0: this is the URL *we* use to
        # reach it, and 0.0.0.0 is not one.
        host = "127.0.0.1" if self.bind in ("0.0.0.0", "::") else self.bind
        return f"http://{host}:{self.port}"

    def _command(self, model_path: Path, mmproj_path: Path | None) -> list[str]:
        cmd = [
            str(find_llama_server()),
            "-m",
            str(model_path),
            "-c",
            str(self.spec.ctx_size),
            "--parallel",
            "1",
            "-ngl",
            str(self.n_gpu_layers),
            "--host",
            self.bind,
            "--port",
            str(self.port),
        ]
        if self.api_key:
            cmd += ["--api-key", self.api_key]
        if mmproj_path is not None:
            cmd += ["--mmproj", str(mmproj_path)]
        if self.spec.kind is ModelKind.LLM:
            # Chat templates are only applied with --jinja, and without it Qwen3's
            # thinking blocks cannot be switched off through chat_template_kwargs.
            cmd += ["--jinja"]
        if self.device:
            cmd += ["--device", self.device]
        return cmd

    def start(self, log_path: Path | None = None) -> TranscriptionClient:
        """Launch the server and return a client once it answers /health."""
        if self.state is not State.STOPPED:
            raise SupervisorError(f"cannot start from state {self.state.value}")

        if not is_downloaded(self.spec):
            raise SupervisorError(
                f"model is not downloaded: {model_dir(self.spec)}; "
                f"run `localasr models pull {self.spec.model_id}` explicitly"
            )
        binary = find_llama_server()
        self.warning = check_build(binary)

        last_error: Exception | None = None
        for _ in range(PORT_RETRIES):
            self.port = self._requested_port or _free_port()
            try:
                return self._launch(binary, log_path)
            except EngineError:
                # A server that runs but cannot transcribe will not start differently
                # on another port. Clean up and surface the reason unchanged.
                self._cleanup()
                self.state = State.FAILED
                raise
            except BaseException as exc:
                # Any other failure must not leave the child process behind; a leaked
                # llama-server keeps the GPU allocated for the rest of the session.
                self._cleanup()
                if not isinstance(exc, SupervisorError):
                    self.state = State.FAILED
                    raise
                last_error = exc
                if self._requested_port is not None:
                    break
        self.state = State.FAILED
        raise SupervisorError(f"llama-server failed to start: {last_error}")

    def _launch(self, binary: Path, log_path: Path | None) -> TranscriptionClient:
        self.state = State.STARTING
        model_path, mmproj_path = local_paths(self.spec)

        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            filter(None, [str(binary.parent), env.get("LD_LIBRARY_PATH", "")])
        )

        self._log = log_path.open("wb") if log_path else None
        stdout = self._log if self._log else subprocess.DEVNULL
        self._process = subprocess.Popen(
            self._command(model_path, mmproj_path),
            stdout=stdout,
            stderr=subprocess.STDOUT,
            env=env,
        )

        client = TranscriptionClient(self.base_url)
        deadline = time.monotonic() + STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                code = self._process.returncode
                where = f"; see {log_path}" if log_path else ""
                client.close()
                raise SupervisorError(
                    f"llama-server exited with code {code}: {_why_it_died(log_path)}"
                    f"{where}"
                )
            if client.is_ready():
                # Only an ASR server must prove it has an audio modality. A refiner is
                # a text model; asking it for audio would reject a perfectly good one.
                if self.spec.kind is ModelKind.ASR:
                    try:
                        client.capabilities().require_audio()
                    except EngineError:
                        client.close()
                        raise
                self.state = State.READY
                return client
            time.sleep(0.5)

        client.close()
        raise SupervisorError(f"llama-server did not become ready within {STARTUP_TIMEOUT}s")

    def _cleanup(self) -> None:
        """Terminate the process and close the log handle, whatever state we are in."""
        if self._process is not None:
            if self._process.poll() is None:
                self._process.send_signal(signal.SIGTERM)
                try:
                    self._process.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()
            self._process = None
        if self._log is not None:
            self._log.close()
            self._log = None

    def stop(self) -> None:
        """Stop the server and wait for the process to actually exit.

        Model switching depends on this being synchronous: starting a second server
        before the first releases its VRAM would exceed the card on a 4 GB machine.
        """
        if self.state is State.STOPPED:
            return
        self.state = State.STOPPING
        self._cleanup()
        self.state = State.STOPPED

    def __enter__(self) -> TranscriptionClient:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
