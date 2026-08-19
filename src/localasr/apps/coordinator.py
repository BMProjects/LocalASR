"""Which workloads may run at the same time.

A shared engine guarantees one model process; it does not stop three applications from
hitting it at once. A subtitle job saturating the single llama-server slot makes
dictation — the one workload where latency is the whole point — wait behind it.

This is not a scheduler. It is a small set of rules that makes the conflict explicit
so a controller can ask the user, rather than producing a mysteriously slow dictation.
"""

from __future__ import annotations

import enum
import threading
from dataclasses import dataclass


class Activity(enum.Enum):
    SUBTITLE = "subtitle"
    DICTATION = "dictation"
    MEETING = "meeting"
    MODEL = "model download"


#: Activities that cannot coexist. Dictation is absent: it is allowed to preempt, but
#: only after the caller has acknowledged what it is interrupting.
_EXCLUSIVE = {Activity.SUBTITLE, Activity.MEETING, Activity.MODEL}


class ActivityConflict(RuntimeError):
    """Raised instead of silently queueing or silently proceeding."""

    def __init__(self, requested: Activity, blocking: tuple[Activity, ...]) -> None:
        names = ", ".join(a.value for a in blocking)
        super().__init__(f"cannot start {requested.value} while {names} is running")
        self.requested = requested
        self.blocking = blocking


@dataclass(frozen=True, slots=True)
class ActivityToken:
    activity: Activity
    serial: int


class ActivityCoordinator:
    """Tracks active workloads. Thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active: dict[int, Activity] = {}
        self._serial = 0

    def active(self) -> tuple[Activity, ...]:
        with self._lock:
            return tuple(self._active.values())

    def is_active(self, activity: Activity) -> bool:
        with self._lock:
            return activity in self._active.values()

    def conflicts_with(self, activity: Activity) -> tuple[Activity, ...]:
        """What would block `activity` right now.

        Two exclusive workloads never coexist. Dictation is blocked by nothing outright,
        but reports the exclusive workload it would contend with so the caller can offer
        to pause or stop it.
        """
        with self._lock:
            running = tuple(self._active.values())
        if activity in _EXCLUSIVE:
            return tuple(a for a in running if a in _EXCLUSIVE or a is activity)
        return tuple(a for a in running if a in _EXCLUSIVE)

    def acquire(self, activity: Activity, *, force: bool = False) -> ActivityToken:
        """Claim `activity`, or raise ActivityConflict describing what is in the way.

        `force` is for dictation once the user has accepted the contention; it never
        lets two exclusive workloads run together.
        """
        with self._lock:
            blocking = self.conflicts_with(activity)
            can_preempt = (
                force and activity is Activity.DICTATION and Activity.MODEL not in blocking
            )
            if blocking and not can_preempt:
                raise ActivityConflict(activity, blocking)
            self._serial += 1
            token = ActivityToken(activity=activity, serial=self._serial)
            self._active[token.serial] = activity
            return token

    def release(self, token: ActivityToken) -> None:
        with self._lock:
            self._active.pop(token.serial, None)

    def can_switch_model(self) -> bool:
        """Model switching unloads the engine, so it needs an idle machine."""
        with self._lock:
            return not self._active
