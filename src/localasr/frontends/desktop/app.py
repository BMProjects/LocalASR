"""The desktop host: one process, one engine, three application windows."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import PySide6
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QSystemTrayIcon

from localasr.apps.dictation import DictationController, DictationOptions
from localasr.context import AppContext
from localasr.frontends.desktop.dictation_window import DictationWindow
from localasr.frontends.desktop.event_bridge import QtEventBridge
from localasr.frontends.desktop.meeting_window import MeetingWindow
from localasr.frontends.desktop.single_instance import (
    QUIT,
    AlreadyRunning,
    SingleInstance,
    activate,
)
from localasr.frontends.desktop.subtitle_window import SubtitleWindow
from localasr.frontends.desktop.theme import apply_theme
from localasr.frontends.desktop.tray import DictationTray


def _copy_via_qt(text: str) -> bool:
    clipboard = QApplication.clipboard()
    if clipboard is None:
        return False
    clipboard.setText(text)
    return True


class _BackendStarter(QThread):
    """Brings both model backends up without holding up the window.

    Neither is required to record — recognition falls back or reports, and refinement is
    a later step — so a backend that does not come up is reported and nothing more. The
    application is a dictation tool first, and must not put a failed model between the
    user and the record button.
    """

    def __init__(self, context: AppContext) -> None:
        super().__init__()
        self._context = context

    def run(self) -> None:
        if self._context.node_url:
            try:
                self._context.start_node()
            except Exception as exc:  # noqa: BLE001 - never take the host down for this
                print(f"识别节点未能就绪：{exc}", file=sys.stderr)
        if self._context.refiner_managed:
            try:
                self._context.start_refiner()
            except Exception as exc:  # noqa: BLE001
                print(f"整理服务未能启动：{exc}", file=sys.stderr)


class _Activation(QObject):
    """Moves an activation request onto the GUI thread.

    `SingleInstance` serves requests on its own thread. Calling `host.show()` from
    there constructs QWidgets outside the GUI thread, which Qt does not support: the
    host wedges, stops acknowledging, and every later launch finds an address held by
    something that never answers. Same hop as QtEventBridge, for the same reason.
    """

    requested = Signal(str)

    def __init__(self, host: DesktopHost) -> None:
        super().__init__()
        self._host = host
        self.requested.connect(self._deliver, Qt.ConnectionType.QueuedConnection)

    @Slot(str)
    def _deliver(self, message: str) -> None:
        self._host.show(message)


def fix_input_method(environ: dict[str, str] | None = None) -> str | None:
    """Let Wayland's own input-method protocol take over when the plugin is missing.

    PySide6 ships its own Qt and its own plugin directory, and that directory has no
    fcitx plugin — only compose, ibus and the virtual keyboard. A desktop that sets
    `QT_IM_MODULE=fcitx` system-wide therefore points this application at something it
    does not have: Qt builds no input context at all and every CJK keystroke is dropped,
    silently, in every text field.

    Borrowing the system plugin is not an option — it is built against a different Qt
    (6.8 here against PySide6's 6.11) and loading it across that boundary is a crash
    waiting to happen. Clearing the variable instead makes Qt use the Wayland text-input
    protocol, which fcitx5 already serves through its wayland launcher.

    Returns the value that was removed, or None when nothing needed doing. Only touches
    Wayland sessions: on X11 the variable is how input methods are found at all.
    """
    env = os.environ if environ is None else environ
    module = env.get("QT_IM_MODULE", "")
    if not module or not env.get("WAYLAND_DISPLAY"):
        return None

    plugins = Path(PySide6.__file__).parent / "Qt/plugins/platforminputcontexts"
    if any(plugins.glob(f"*{module}*")):
        return None  # the bundled Qt can honour it after all

    env.pop("QT_IM_MODULE", None)
    return module


class DesktopHost:
    """Owns the shared context and the three windows.

    The windows are created lazily and hidden rather than destroyed, so switching
    between applications never rebuilds state or touches the engine.
    """

    def __init__(self, app: QApplication) -> None:
        self.app = app
        self.context = AppContext()
        self.bridge = QtEventBridge(self.context.bus)
        self.dictation_controller = DictationController(
            self.context,
            # Qt always has a clipboard, so a desktop session can never end up with no
            # way to hand over text it just recognised.
            DictationOptions(clipboard_fallback=_copy_via_qt),
        )
        self._dictation: DictationWindow | None = None
        self._subtitle: SubtitleWindow | None = None
        self._meeting: MeetingWindow | None = None
        self.tray = DictationTray(self.context, self.bridge, self, self.dictation_controller)
        self._backends: _BackendStarter | None = None
        if not QSystemTrayIcon.isSystemTrayAvailable():
            # Not fatal: the windows carry their own controls and exits. Saying so
            # beats leaving the user hunting for an icon that will never appear.
            print(
                "no system tray in this session; use the windows (Ctrl+Q quits)",
                file=sys.stderr,
            )

    def start_backends(self) -> None:
        """Bring both model backends up alongside the window, off the GUI thread.

        Called from `main`, not from the constructor. Building a host must not load a
        model as a side effect — every test that constructed one started a real
        llama-server, which is how a 2.8 GB model ended up resident during a unit run.

        Loading the model takes seconds, and blocking here would mean the application
        appears to hang on launch — for a feature the user may not touch this session.
        Nothing waits on it: 整理 stays greyed until it answers, which is what the button
        state already does for every other reason it might not be ready.

        Skipped entirely when `refiner_url` points somewhere: that is the setting that
        says somebody else owns the model.
        """
        self._backends = _BackendStarter(self.context)
        self._backends.start()

    def show(self, which: str) -> str:
        """Raise the requested application, and always end with a visible window.

        An unknown name falls back to the subtitle window rather than doing nothing:
        a launcher that appears to do nothing is indistinguishable from a broken
        install, and the only surface left would be a tray icon.
        """
        if which == QUIT:
            self.app.quit()
            return QUIT

        factory = {
            "dictation": self.dictation_window,
            "subtitle": self.subtitle_window,
            "meeting": self.meeting_window,
        }.get(which)
        if factory is None:
            factory = self.subtitle_window
            which = "subtitle"

        target = factory()
        target.show()
        target.raise_()
        target.activateWindow()
        return which

    def dictation_window(self) -> DictationWindow:
        if self._dictation is None:
            self._dictation = DictationWindow(self.context, self.bridge, self.dictation_controller)
        return self._dictation

    def toggle_dictation(self) -> None:
        self.dictation_window().toggle()

    def cancel_dictation(self) -> None:
        self.dictation_window().cancel()

    def subtitle_window(self) -> SubtitleWindow:
        if self._subtitle is None:
            self._subtitle = SubtitleWindow(self.context, self.bridge)
        return self._subtitle

    def meeting_window(self) -> MeetingWindow:
        if self._meeting is None:
            self._meeting = MeetingWindow(self.context, self.bridge)
        return self._meeting

    def shutdown(self) -> None:
        # First, so the icon disappears the moment quitting starts rather than after
        # the engine has been torn down — that wait is seconds long, and an icon that
        # still responds to nothing is how a quit looks like it failed.
        self.tray.hide()
        if self._dictation is not None:
            self._dictation.cancel_if_running()
        elif self.dictation_controller.recording:
            self.dictation_controller.cancel()
        if self._meeting is not None:
            self._meeting.stop_if_running()
        self.bridge.stop()
        # Before shutting the context down, or a start still in flight wins the race:
        # `stop_refiner` would find nothing to stop, the spawn would finish afterwards,
        # and a llama-server holding the card would outlive the application with nothing
        # left that knows about it.
        if self._backends is not None:
            self._backends.wait(300_000)
            self._backends = None
        self.context.shutdown()


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    requested = argv[0] if argv else "subtitle"

    # Only step aside for a host that answers. One that accepts the connection but
    # never acknowledges — hung, or left over from a build that does not know the
    # requested window — would otherwise swallow every launch, and the application
    # would simply stop starting with nothing on screen to explain why.
    if activate(requested):
        return 0

    if requested == QUIT:
        print("no running LocalASR desktop to quit", file=sys.stderr)
        return 1

    fix_input_method()
    app = QApplication(sys.argv[:1])
    app.setApplicationName("LocalASR")
    app.setWindowIcon(QIcon.fromTheme("localasr"))
    # Windows quit the host themselves once nothing is running; see ShellWindow.
    app.setQuitOnLastWindowClosed(False)
    apply_theme(app)

    host = DesktopHost(app)
    activation = _Activation(host)
    instance = SingleInstance()
    try:
        instance.acquire(activation.requested.emit)
    except AlreadyRunning:
        host.shutdown()
        if activate(requested):
            return 0  # lost a start-up race; the winner shows the window
        # The address is held by something that will not answer — typically a host
        # from an earlier build still running. Exiting quietly here is what makes the
        # application look like it stopped launching, so say what is wrong instead.
        print(
            "another process holds the LocalASR desktop address but does not respond.\n"
            "It is probably an older instance. Stop it and try again:\n"
            "  pkill -f localasr-desktop",
            file=sys.stderr,
        )
        return 1

    host.show(requested)
    host.start_backends()
    app.aboutToQuit.connect(host.shutdown)
    app.aboutToQuit.connect(instance.release)
    try:
        return app.exec()
    finally:
        host.shutdown()
        instance.release()


if __name__ == "__main__":
    raise SystemExit(main())
