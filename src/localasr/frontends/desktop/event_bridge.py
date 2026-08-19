"""Delivering worker-thread events onto the Qt thread.

Capture threads and the transcription worker publish to an `EventBus`; Qt widgets may
only be touched from the GUI thread. This is the only place the two meet.

    worker threads -> EventBus (thread-safe queue) -> Qt signal -> controllers/views

The bus wake-up uses a queued signal emission rather than calling into widgets, so the
actual delivery always happens on the GUI thread even though `publish` was called from
elsewhere.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Qt, QTimer, Signal

from localasr.apps.bus import EventBus
from localasr.apps.events import Event


class QtEventBridge(QObject):
    """Re-emits bus events as a Qt signal on the GUI thread."""

    event = Signal(object)
    dropped = Signal(int)

    def __init__(self, bus: EventBus, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._bus = bus
        self._last_dropped = 0

        # A queued connection is what moves the work onto the GUI thread: `_wake` runs
        # on whichever thread published, but the slot it triggers does not.
        self._pump = _Pump()
        self._pump.wake.connect(self._drain, Qt.ConnectionType.QueuedConnection)
        bus.set_wakeup(self._pump.wake.emit)

        # A slow safety net for events published before the bridge was connected, and
        # for anything a coalesced wake-up left behind.
        self._timer = QTimer(self)
        self._timer.setInterval(200)
        self._timer.timeout.connect(self._drain)
        self._timer.start()

    def _drain(self) -> None:
        for item in self._bus.drain():
            self.event.emit(item)
        dropped = self._bus.dropped
        if dropped != self._last_dropped:
            self._last_dropped = dropped
            self.dropped.emit(dropped)

    def stop(self) -> None:
        self._timer.stop()
        self._bus.set_wakeup(None)


class _Pump(QObject):
    wake = Signal()


def describe(event: Event) -> str:
    """One-line rendering, used by the status bar and the log view."""
    return f"{type(event).__name__}: {event}"
