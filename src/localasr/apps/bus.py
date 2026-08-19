"""Serialising events from worker threads to a single consumer.

Capture threads and the transcription worker both publish; Qt widgets may only be
touched from the GUI thread. This queue is the hand-off point, and it is deliberately
free of Qt so `apps/` stays importable without a display.

A Qt frontend owns an `EventBus`, wakes on its queue, and re-emits each event as a
signal on the GUI thread. A CLI frontend drains the same queue in its main loop.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from queue import Empty, Full, Queue

from localasr.apps.events import Event

BUS_CAPACITY = 2048


class EventBus:
    """Thread-safe fan-in. `publish` never blocks a worker for long."""

    def __init__(self, capacity: int = BUS_CAPACITY) -> None:
        self._queue: Queue[Event] = Queue(maxsize=capacity)
        self._dropped = 0
        self._lock = threading.Lock()
        self._wakeup: Callable[[], None] | None = None

    @property
    def dropped(self) -> int:
        """Events lost because the consumer fell behind; surfaced, never hidden."""
        with self._lock:
            return self._dropped

    def set_wakeup(self, callback: Callable[[], None] | None) -> None:
        """Called after each publish so a GUI loop can schedule a drain."""
        self._wakeup = callback

    def publish(self, event: Event) -> None:
        try:
            self._queue.put_nowait(event)
        except Full:
            with self._lock:
                self._dropped += 1
            return
        if self._wakeup is not None:
            self._wakeup()

    def drain(self, limit: int = 256) -> list[Event]:
        """Take up to `limit` pending events. Never blocks."""
        events: list[Event] = []
        for _ in range(limit):
            try:
                events.append(self._queue.get_nowait())
            except Empty:
                break
        return events

    def wait(self, timeout: float = 0.1) -> Iterator[Event]:
        """Block briefly for one event, then yield everything queued behind it."""
        try:
            yield self._queue.get(timeout=timeout)
        except Empty:
            return
        yield from self.drain()
