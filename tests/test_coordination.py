"""Sharing one engine between three applications.

A single EngineManager guarantees one model process per process. These cover the two
things it does not: which workloads may run together, and what stops a second *process*
from starting its own llama-server.
"""

import os

import pytest

from localasr.apps.bus import EventBus
from localasr.apps.coordinator import Activity, ActivityConflict, ActivityCoordinator
from localasr.apps.events import SessionStopped
from localasr.core.engine import lock

# --- activity rules ----------------------------------------------------------


def test_two_subtitle_jobs_cannot_run_together():
    coordinator = ActivityCoordinator()
    coordinator.acquire(Activity.SUBTITLE)
    with pytest.raises(ActivityConflict):
        coordinator.acquire(Activity.SUBTITLE)


def test_a_meeting_cannot_start_during_a_subtitle_job():
    coordinator = ActivityCoordinator()
    coordinator.acquire(Activity.SUBTITLE)
    with pytest.raises(ActivityConflict) as excinfo:
        coordinator.acquire(Activity.MEETING)
    assert excinfo.value.blocking == (Activity.SUBTITLE,)


def test_dictation_during_a_subtitle_job_is_refused_but_can_be_forced():
    """The user has to be told what they are interrupting; they are not stopped from
    doing it."""
    coordinator = ActivityCoordinator()
    coordinator.acquire(Activity.SUBTITLE)

    with pytest.raises(ActivityConflict):
        coordinator.acquire(Activity.DICTATION)
    token = coordinator.acquire(Activity.DICTATION, force=True)
    assert Activity.DICTATION in coordinator.active()
    coordinator.release(token)


def test_force_never_lets_two_exclusive_workloads_coexist():
    coordinator = ActivityCoordinator()
    coordinator.acquire(Activity.MEETING)
    with pytest.raises(ActivityConflict):
        coordinator.acquire(Activity.SUBTITLE, force=True)


def test_dictation_cannot_preempt_a_model_download_even_with_force():
    coordinator = ActivityCoordinator()
    coordinator.acquire(Activity.MODEL)

    with pytest.raises(ActivityConflict):
        coordinator.acquire(Activity.DICTATION, force=True)


def test_dictation_alone_needs_no_force():
    coordinator = ActivityCoordinator()
    assert coordinator.acquire(Activity.DICTATION)


def test_releasing_frees_the_slot():
    coordinator = ActivityCoordinator()
    token = coordinator.acquire(Activity.SUBTITLE)
    coordinator.release(token)
    assert coordinator.acquire(Activity.SUBTITLE)


def test_model_switching_requires_an_idle_machine():
    coordinator = ActivityCoordinator()
    assert coordinator.can_switch_model()
    token = coordinator.acquire(Activity.DICTATION)
    assert not coordinator.can_switch_model()
    coordinator.release(token)
    assert coordinator.can_switch_model()


# --- cross-process engine record ---------------------------------------------


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALASR_RUNTIME_DIR", str(tmp_path))
    return tmp_path


def test_a_live_record_is_returned(runtime):
    lock.write(lock.EngineRecord(os.getpid(), "http://127.0.0.1:9", "m", "rev"))
    record = lock.read()
    assert record and record.base_url == "http://127.0.0.1:9"


def test_a_record_from_a_dead_process_is_reclaimed(runtime):
    """A crash leaves the file behind; attaching to it would hang on a dead server."""
    lock.write(lock.EngineRecord(2**30, "http://127.0.0.1:9", "m", "rev"))
    assert lock.read() is None
    assert not lock.record_path().exists()


def test_no_record_reads_as_none(runtime):
    assert lock.read() is None


def test_a_corrupt_record_reads_as_none(runtime):
    path = lock.record_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json", encoding="utf-8")
    assert lock.read() is None


def test_clear_only_removes_your_own_record(runtime):
    lock.write(lock.EngineRecord(os.getpid(), "http://127.0.0.1:9", "m", "rev"))
    lock.clear(owner_pid=os.getpid() + 1)
    assert lock.read() is not None
    lock.clear(owner_pid=os.getpid())
    assert lock.read() is None


# --- event bus ---------------------------------------------------------------


def test_bus_delivers_in_order():
    bus = EventBus()
    for index in range(5):
        bus.publish(SessionStopped(utterances=index))
    assert [e.utterances for e in bus.drain()] == [0, 1, 2, 3, 4]


def test_bus_counts_what_it_had_to_drop_rather_than_blocking():
    """A full bus must not stall a capture thread, but the loss has to be visible."""
    bus = EventBus(capacity=2)
    for index in range(5):
        bus.publish(SessionStopped(utterances=index))
    assert len(bus.drain()) == 2
    assert bus.dropped == 3


def test_bus_wakeup_fires_on_publish():
    bus = EventBus()
    woken = []
    bus.set_wakeup(lambda: woken.append(1))
    bus.publish(SessionStopped(utterances=0))
    assert woken


def test_bus_drain_is_empty_when_nothing_published():
    assert EventBus().drain() == []
