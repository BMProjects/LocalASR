"""Runtime pinning checks that do not need a GPU."""

import pytest

from localasr.core.engine import supervisor
from localasr.registry import manager


def test_release_tag_and_reported_version_compare_equal():
    """Releases are tagged `b10333`; `llama-server --version` reports `10333`."""
    assert supervisor._normalize_build("b10333") == supervisor._normalize_build("10333")


def test_a_genuinely_different_build_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, "installed_build", lambda _binary: "10999")
    warning = supervisor.check_build(tmp_path / "llama-server")
    assert warning and "10999" in warning


def test_the_pinned_build_produces_no_warning(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, "installed_build", lambda _binary: "b10333")
    assert supervisor.check_build(tmp_path / "llama-server") is None


def test_an_unreadable_version_does_not_warn(tmp_path, monkeypatch):
    """A build we cannot identify is not evidence of a mismatch."""
    monkeypatch.setattr(supervisor, "installed_build", lambda _binary: None)
    assert supervisor.check_build(tmp_path / "llama-server") is None


def test_supervisor_starts_in_the_stopped_state():
    from localasr.registry import manager

    sup = supervisor.EngineSupervisor(manager.get_model())
    assert sup.state is supervisor.State.STOPPED


def test_stopping_a_stopped_supervisor_is_a_no_op():
    from localasr.registry import manager

    sup = supervisor.EngineSupervisor(manager.get_model())
    sup.stop()
    assert sup.state is supervisor.State.STOPPED


def test_start_never_downloads_a_missing_model(monkeypatch):
    monkeypatch.setattr(supervisor, "is_downloaded", lambda _spec: False)
    sup = supervisor.EngineSupervisor(manager.get_model())

    with pytest.raises(supervisor.SupervisorError, match="explicitly"):
        sup.start()


def test_a_held_engine_is_not_reaped_while_idle():
    """`hold()` is the difference between an engine that survives a silent stretch of
    recording and one that unloads under it."""
    import time

    from localasr.core.engine.manager import EngineManager
    from localasr.registry import manager as registry

    engine = EngineManager(registry.get_model(), idle_timeout=0.2, share=False)
    engine._client = object()  # pretend a server is loaded, without starting one
    engine._last_used = time.monotonic()

    engine.hold()
    engine._last_used = time.monotonic() - 10  # long past the timeout
    with engine._lock:
        idle = time.monotonic() - engine._last_used
        would_reap = engine._in_flight == 0 and idle >= engine.idle_timeout
    assert not would_reap, "a held engine was eligible for unloading"

    engine.release_hold()
    # `release` refreshes the idle clock, so the engine becomes reapable only once the
    # timeout elapses again — not instantly.
    engine._last_used = time.monotonic() - 10
    with engine._lock:
        idle = time.monotonic() - engine._last_used
        would_reap = engine._in_flight == 0 and idle >= engine.idle_timeout
    assert would_reap, "releasing the hold did not make the engine reapable again"
    engine._client = None
