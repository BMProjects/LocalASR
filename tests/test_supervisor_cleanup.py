"""A failed start must never leave the child process or the log handle behind.

A leaked llama-server holds its VRAM for the rest of the session, which on a 4 GB card
means nothing else can load.
"""

import subprocess

import pytest

from localasr.core.engine import supervisor
from localasr.core.engine.client import CapabilityError
from localasr.core.engine.supervisor import EngineSupervisor, State, SupervisorError
from localasr.registry import manager


class FakeProcess:
    def __init__(self, exit_code=None):
        self.returncode = exit_code
        self.signals = []
        self.killed = False
        self._exit_code = exit_code

    def poll(self):
        return self._exit_code

    def send_signal(self, sig):
        self.signals.append(sig)
        self._exit_code = 0
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode or 0

    def kill(self):
        self.killed = True
        self._exit_code = -9


@pytest.fixture
def patched(monkeypatch, tmp_path):
    binary = tmp_path / "llama-server"
    binary.write_text("")
    monkeypatch.setattr(supervisor, "find_llama_server", lambda: binary)
    monkeypatch.setattr(supervisor, "check_build", lambda *_args, **_kw: None)
    monkeypatch.setattr(supervisor, "is_downloaded", lambda _spec: True)
    monkeypatch.setattr(
        supervisor, "local_paths", lambda _spec: (tmp_path / "m.gguf", tmp_path / "p.gguf")
    )
    return binary


def _supervisor() -> EngineSupervisor:
    return EngineSupervisor(manager.get_model())


def test_process_exiting_immediately_is_reported_and_reaped(patched, monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_kw: FakeProcess(exit_code=1))
    sup = _supervisor()
    with pytest.raises(SupervisorError, match="exited with code"):
        sup.start()
    assert sup.state is State.FAILED
    assert sup._process is None


def test_a_text_only_model_fails_fast_without_leaking_the_process(patched, monkeypatch):
    """Retrying on another port cannot help here, so it must not be attempted."""
    process = FakeProcess()
    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_kw: process)

    class Client:
        def __init__(self, *_a, **_kw):
            pass

        def is_ready(self):
            return True

        def capabilities(self):
            raise CapabilityError("text-only model")

        def close(self):
            pass

    monkeypatch.setattr(supervisor, "TranscriptionClient", Client)

    sup = _supervisor()
    with pytest.raises(CapabilityError):
        sup.start()
    assert sup.state is State.FAILED
    assert sup._process is None
    assert process.signals or process.killed


def test_the_log_handle_is_closed_when_start_fails(patched, monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_kw: FakeProcess(exit_code=1))
    log = tmp_path / "server.log"

    sup = _supervisor()
    with pytest.raises(SupervisorError):
        sup.start(log_path=log)
    assert sup._log is None


def test_stop_is_idempotent(patched, monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_kw: FakeProcess(exit_code=1))
    sup = _supervisor()
    with pytest.raises(SupervisorError):
        sup.start()
    sup.stop()
    sup.stop()
    assert sup.state is State.STOPPED
