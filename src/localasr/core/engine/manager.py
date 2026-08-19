"""A persistent engine shared by every application.

Loading a model costs seconds, which is invisible for a batch subtitle job and fatal
for push-to-talk dictation. The manager keeps one server alive across requests and
releases the GPU once nobody has used it for a while — on a 4 GB laptop, holding
VRAM indefinitely is not an option either.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from localasr.core.engine import lock
from localasr.core.engine.client import TranscriptionClient
from localasr.core.engine.supervisor import EngineSupervisor, State, SupervisorError
from localasr.registry.manager import ModelSpec

DEFAULT_IDLE_TIMEOUT = 120.0


class EngineManager:
    """Owns the engine process. Thread-safe; `acquire()` is reentrant across callers.

    `idle_timeout` of 0 disables unloading, which is what a one-shot CLI run wants.
    """

    def __init__(
        self,
        spec: ModelSpec,
        *,
        idle_timeout: float = DEFAULT_IDLE_TIMEOUT,
        log_path: Path | None = None,
        device: str | None = None,
        share: bool = True,
        node_url: str | None = None,
        node_token: str | None = None,
        local_fallback: bool = True,
    ) -> None:
        self.spec = spec
        self.idle_timeout = idle_timeout
        self.log_path = log_path
        self.device = device
        self.share = share
        self.node_url = node_url
        self.node_token = node_token
        self.local_fallback = local_fallback
        self.fallback_reason: str | None = None
        """Why a configured node is not being used. None means it is, or none is set."""

        self._lock = threading.RLock()
        self._supervisor: EngineSupervisor | None = None
        self._client: TranscriptionClient | None = None
        self._attached = False
        self._last_used = 0.0
        self._in_flight = 0
        self._reaper: threading.Thread | None = None
        self._stop_reaper = threading.Event()

    @property
    def attached(self) -> bool:
        """True when using a server another process started, which we must not stop."""
        with self._lock:
            return self._attached

    @property
    def loaded(self) -> bool:
        """True when a client is bound to a live server.

        Not the supervisor state: when attached to a server another process started
        there is no supervisor here, but the model is very much loaded.
        """
        with self._lock:
            return self._client is not None

    @property
    def state(self) -> State:
        with self._lock:
            return self._supervisor.state if self._supervisor else State.STOPPED

    @property
    def warning(self) -> str | None:
        with self._lock:
            return self._supervisor.warning if self._supervisor else None

    def acquire(self) -> TranscriptionClient:
        """Return a ready client, attaching to a shared server or starting one."""
        with self._lock:
            if self._client is None:
                self._client = self._attach() or self._launch()
            self._in_flight += 1
            self._last_used = time.monotonic()
            return self._client

    def _attach(self) -> TranscriptionClient | None:
        """Reuse a server another process is already running, if it fits our needs.

        A different model is not reusable, and stopping someone else's engine to load
        ours would break whatever they are doing — so this declines instead.
        """
        if self.node_url:
            remote = self._attach_node()
            if remote is not None:
                return remote
            if not self.local_fallback:
                # With a refiner resident on this machine's 4 GB card, starting a local
                # ASR model beside it runs the card out of memory — silently, and
                # mid-sentence. Saying the node is down is the better failure.
                raise SupervisorError(
                    f"远程识别节点不可用：{self.fallback_reason}。"
                    "本机回退已关闭（config.toml 中 local_asr_fallback = false），"
                    "因为本机显卡正驻留整理模型。"
                )
            self.fallback_reason += "；已改用本机引擎"
            # Otherwise fall through to the local engine. A node that is off, rebooting
            # or out of memory should degrade to slower local recognition, not to no
            # recognition: the person is mid-sentence and cannot act on the outage.
        if not self.share:
            return None
        record = lock.read()
        if record is None:
            return None
        if record.model_id != self.spec.model_id or record.revision != self.spec.revision:
            return None
        client = TranscriptionClient(record.base_url)
        if not client.is_ready():
            client.close()
            lock.clear(owner_pid=record.pid)
            return None
        self._attached = True
        return client

    def _attach_node(self) -> TranscriptionClient | None:
        """Bind to a remote LocalASR node, or explain why not.

        Marked attached, so nothing here will ever try to stop it: the node belongs to
        another machine and is very likely serving someone else too.
        """
        client = TranscriptionClient(self.node_url, token=self.node_token)
        try:
            ready = client.is_ready()
        except Exception as exc:  # noqa: BLE001 - any transport failure means fall back
            client.close()
            self.fallback_reason = f"节点 {self.node_url} 无法连接（{exc}）"
            return None
        if not ready:
            client.close()
            self.fallback_reason = f"节点 {self.node_url} 未就绪"
            return None
        self.fallback_reason = None
        self._attached = True
        return client

    def _launch(self) -> TranscriptionClient:
        supervisor = EngineSupervisor(self.spec, device=self.device)
        client = supervisor.start(self.log_path)
        self._supervisor = supervisor
        self._attached = False
        if self.share:
            lock.write(
                lock.EngineRecord(
                    pid=os.getpid(),
                    base_url=supervisor.base_url,
                    model_id=self.spec.model_id,
                    revision=self.spec.revision,
                )
            )
        self._start_reaper()
        return client

    def hold(self) -> None:
        """Keep the engine loaded until `release_hold()`, however idle it looks.

        A live capture session only touches the engine when an utterance completes, so
        between sentences it looks idle and the reaper unloads it — mid-recording. The
        next sentence then pays the full model load, which is exactly what preparing
        before opening the device was meant to avoid.
        """
        self.acquire()

    def release_hold(self) -> None:
        self.release()

    def release(self) -> None:
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
            self._last_used = time.monotonic()

    def switch(self, spec: ModelSpec) -> None:
        """Load a different model, releasing the current one first.

        Sequential by necessity: two servers resident at once would not fit.
        """
        with self._lock:
            if spec.model_id == self.spec.model_id and self._client is not None:
                return
            self.shutdown()
            self.spec = spec

    def shutdown(self) -> None:
        with self._lock:
            self._stop_reaper.set()
            if self._client is not None:
                self._client.close()
                self._client = None
            if self._supervisor is not None:
                self._supervisor.stop()
                self._supervisor = None
                if self.share:
                    lock.clear(owner_pid=os.getpid())
            self._attached = False
            self._in_flight = 0

    def _start_reaper(self) -> None:
        if self.idle_timeout <= 0:
            return
        self._stop_reaper.clear()
        self._reaper = threading.Thread(target=self._reap_when_idle, daemon=True)
        self._reaper.start()

    def _reap_when_idle(self) -> None:
        while not self._stop_reaper.wait(min(self.idle_timeout, 5.0)):
            with self._lock:
                if self._client is None:
                    return
                idle = time.monotonic() - self._last_used
                if self._in_flight == 0 and idle >= self.idle_timeout:
                    self.shutdown()
                    return

    def __enter__(self) -> EngineManager:
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()
