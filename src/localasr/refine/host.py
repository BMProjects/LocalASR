"""`localasr-refiner`: a process whose only job is to hold the refinement model.

The desktop can start a refiner itself, and for a long time that was the only way. It
works, but it ties the model's lifetime to a window: the model dies when the application
exits, it lives on whatever machine the application runs on, and "which model, loaded
how" is answered by application settings rather than by whoever owns the hardware.

This module is the other half. It brings the model up as a plain OpenAI-compatible
server on a **fixed port**, and then the application is just a client — point
`refiner_url` at it and the desktop never launches a model at all. Nothing here imports
the frontend, the capture stack or the apps: it is startable on a machine that has no
audio hardware and no display, which is the point of separating it.

It is deliberately not a new abstraction over `EngineSupervisor` — it *is* an
EngineSupervisor, given a stable address and a signal handler. The protocol boundary
already existed; what was missing was something on the far side of it that the
application does not own.

    localasr-refiner                       # the configured refiner on 127.0.0.1:8091
    localasr-refiner --model local-x       # a specific catalog or imported id
    localasr-refiner --bind 0.0.0.0 --token secret   # for another machine to use

Then, in config.toml:

    refiner_url = "http://127.0.0.1:8091"
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import httpx

from localasr.core.engine.supervisor import STARTUP_TIMEOUT, EngineSupervisor, SupervisorError
from localasr.registry import manager
from localasr.registry.manager import ModelKind

DEFAULT_PORT = 8091
"""Next to the compute node's 8090, and fixed rather than chosen at random.

A refiner the application launches can use an ephemeral port, because the launcher is
also the caller. One that outlives the application cannot: `refiner_url` is a line in a
config file, and a line that has to be rewritten after every restart is not a
configuration.
"""

LOOPBACK = ("127.0.0.1", "localhost", "::1")


@dataclass(frozen=True, slots=True)
class HostConfig:
    model_id: str | None = None
    port: int = DEFAULT_PORT
    bind: str = "127.0.0.1"
    token: str | None = None
    device: str | None = None

    @property
    def exposed(self) -> bool:
        return self.bind not in LOOPBACK

    def resolve(self) -> object:
        """The model to serve.

        `--model` first, then the desktop's `refiner_model_id` — which only answers when
        the application still manages a model on this machine. Once it stops (the point
        of this host existing) that setting is empty, and the first version of this code
        then fell through to the catalog default: a model whose weights had been deleted,
        while a perfectly good imported one sat beside it. Two settings with different
        owners had been conflated into one.

        So the last resort is what this machine actually has. Exactly one downloaded
        refiner is not a guess; more than one is genuinely ambiguous and says so.
        """
        if self.model_id is not None:
            return manager.get_model(self.model_id, ModelKind.LLM)

        from localasr.context import Settings

        configured = Settings.load().refiner_model_id
        if configured:
            return manager.get_model(configured, ModelKind.LLM)

        present = [
            spec
            for spec in manager.models_of_kind(ModelKind.LLM)
            if manager.is_downloaded(spec)
        ]
        if len(present) == 1:
            return present[0]
        if not present:
            raise manager.RegistryError(
                "no refinement model is present on this machine. Import one with "
                "`localasr models import <file.gguf> --kind llm --link`, or download "
                "one with `localasr models pull`."
            )
        names = ", ".join(spec.model_id for spec in present)
        raise manager.RegistryError(
            f"several refinement models are present ({names}); choose one with --model"
        )


class Companion:
    """The host, run as a child of the application instead of by systemd.

    Same module, same protocol, shorter life: it starts when the application starts and
    is terminated when it exits, so nothing has to be registered with the init system to
    use it. That is the whole difference — an installed unit buys a model that outlives
    the window, and costs a file in `~/.config/systemd/user` and a service to remember.

    What it deliberately does *not* do is give the application back its old job. The
    parent spawns a process by name and then speaks HTTP to it; which model that process
    loads, with which flags, on which port, is decided inside the module. Pointing
    `refiner_url` at a systemd-managed or remote instance later means not spawning this
    one — no other code changes.
    """

    def __init__(
        self,
        *,
        port: int = DEFAULT_PORT,
        model_id: str | None = None,
        device: str | None = None,
    ) -> None:
        self.port = port
        self.model_id = model_id
        self.device = device
        self._process: subprocess.Popen[bytes] | None = None
        self._log: IO[bytes] | None = None
        self._owned = False
        """Whether we started what is on the port. Stopping something we merely found is
        not ours to do."""

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _answers(self) -> bool:
        try:
            with httpx.Client(timeout=2.0) as client:
                return client.get(f"{self.base_url}/health").status_code == 200
        except httpx.HTTPError:
            return False

    def start(self, timeout: float = STARTUP_TIMEOUT) -> str:
        """Spawn the module and return once it answers, or raise saying why not."""
        if self.running:
            return self.base_url

        # Somebody is already serving this port — a systemd unit, or a previous session
        # that outlived its parent. Use it rather than starting a second copy of a
        # multi-gigabyte model that could not bind anyway. `_owned` stays False, so
        # stopping this companion does not stop a server it did not start.
        if self._answers():
            self._owned = False
            return self.base_url
        self._owned = True

        command = [sys.executable, "-m", "localasr.refine.host", "--port", str(self.port)]
        if self.model_id:
            command += ["--model", self.model_id]
        if self.device:
            command += ["--device", self.device]

        # Its own file rather than the parent's stdout: a desktop launched from a menu
        # has nowhere for that to go, and "it did not start" with no output is the least
        # actionable failure there is.
        log_path = manager.data_dir() / "refiner-host.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = log_path.open("wb")
        self._process = subprocess.Popen(command, stdout=self._log, stderr=subprocess.STDOUT)

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            # Liveness before health, and both before believing either. A child that
            # has already exited must never be reported as started because something
            # else happens to answer on its port.
            if self._process.poll() is not None:
                detail = _tail(log_path)
                self._close_log()
                self._process = None
                raise SupervisorError(f"整理服务启动失败：{detail}")
            if self._answers():
                return self.base_url
            time.sleep(0.5)

        self.stop()
        raise SupervisorError(f"整理服务在 {timeout:.0f}s 内没有就绪，详见 {log_path}")

    def stop(self) -> None:
        """Terminate the module and wait for it to release the model.

        SIGTERM, not kill: the module's handler is what stops llama-server, and killing
        the parent would leave the grandchild holding the memory with nothing left to
        reclaim it.
        """
        if not self._owned:
            self._process = None
            self._close_log()
            return
        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        self._close_log()

    def _close_log(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = None


def _tail(path: Path, lines: int = 6) -> str:
    try:
        return " / ".join(path.read_text(errors="replace").splitlines()[-lines:]) or "无输出"
    except OSError:
        return "无法读取日志"


def check(config: HostConfig) -> None:
    """Refuse an unauthenticated model on the network.

    Same rule the compute node applies to itself. An open `/v1/chat/completions` is not
    only somebody else's free inference: it is an open door into this machine's memory
    budget, and the failure is silent — everything keeps working, slowly, for reasons
    nothing on this host explains.
    """
    if config.exposed and not config.token:
        raise SystemExit(
            f"refusing to serve on {config.bind} without a token.\n"
            "Pass --token, or set LOCALASR_REFINER_TOKEN, or bind to 127.0.0.1."
        )


def serve(config: HostConfig) -> int:
    """Start the model and stay up until asked to stop. Returns a process exit code."""
    check(config)
    try:
        spec = config.resolve()
    except manager.RegistryError as exc:
        print(f"cannot serve: {exc}", file=sys.stderr)
        return 2

    if not manager.is_downloaded(spec):
        print(
            f"{spec.model_id} is not present at {manager.model_dir(spec)}.\n"
            f"Run `localasr models pull {spec.model_id}`, or import a local file with "
            "`localasr models import <file.gguf> --kind llm --link`.",
            file=sys.stderr,
        )
        return 2

    supervisor = EngineSupervisor(
        spec,
        port=config.port,
        bind=config.bind,
        api_key=config.token,
        device=config.device,
    )
    log = manager.data_dir() / "refiner-host.log"
    try:
        supervisor.start(log_path=log).close()
    except SupervisorError as exc:
        print(f"failed to start {spec.model_id}: {exc}", file=sys.stderr)
        return 1

    if supervisor.warning:
        print(f"warning: {supervisor.warning}", file=sys.stderr)
    print(f"serving {spec.model_id} on http://{config.bind}:{config.port}")
    print(f'set  refiner_url = "{supervisor.base_url}"  in config.toml')
    if config.token:
        print("clients must send it as  refiner_token")
    sys.stdout.flush()

    # Wait rather than poll. The whole process exists to keep one child alive, and a
    # sleep loop would only add latency to the one thing it has to do — shut down
    # cleanly, so the model's memory is actually released.
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    try:
        stop.wait()
    finally:
        print("\nreleasing the model…", file=sys.stderr)
        supervisor.stop()
    return 0


def parse_args(argv: list[str] | None = None) -> HostConfig:
    parser = argparse.ArgumentParser(
        prog="localasr-refiner",
        description="Serve the refinement model as an OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("LOCALASR_REFINER_MODEL") or None,
        help="Catalog or imported model id. Defaults to the only one present.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("LOCALASR_REFINER_PORT", DEFAULT_PORT)),
    )
    parser.add_argument(
        "--bind",
        default=os.environ.get("LOCALASR_REFINER_BIND", "127.0.0.1"),
        help="127.0.0.1 by default; anything else requires --token.",
    )
    parser.add_argument("--token", default=os.environ.get("LOCALASR_REFINER_TOKEN"))
    parser.add_argument("--device", default=None, help="Passed through to llama-server.")
    args = parser.parse_args(argv)
    return HostConfig(
        model_id=args.model,
        port=args.port,
        bind=args.bind,
        token=args.token,
        device=args.device,
    )


def main(argv: list[str] | None = None) -> int:
    return serve(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
