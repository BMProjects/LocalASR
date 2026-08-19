"""Shared window behaviour: how any LocalASR window is closed and quit.

Quitting must never depend on the tray. A KDE session can be configured without a
StatusNotifier host, the icon can be collapsed into an overflow the user does not find,
and `setQuitOnLastWindowClosed(False)` means closing every window otherwise leaves a
process with no visible surface and no way out.

So every window carries the same three exits: Ctrl+Q, a 退出 action, and closing the
last window. Work in progress is the only thing that turns a close into a hide, and
then it says so.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtGui import QCloseEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget


class ShellWindow(QWidget):
    """Base for the three application windows.

    Subclasses implement `busy_reason()` to report work that a close would interrupt.
    """

    def install_exits(self) -> None:
        """Wire Ctrl+Q and Ctrl+W. Call once the widgets exist."""
        quit_shortcut = QShortcut(QKeySequence.StandardKey.Quit, self)
        quit_shortcut.activated.connect(self.quit_application)
        close_shortcut = QShortcut(QKeySequence.StandardKey.Close, self)
        close_shortcut.activated.connect(self.close)

    def busy_reason(self) -> str | None:
        """What a close would interrupt, or None when the window is idle."""
        return None

    def quit_application(self) -> None:
        """Exit the whole host, confirming first if any window is busy."""
        reason = _first_busy_reason()
        if reason and not _confirm(self, reason):
            return
        app = QApplication.instance()
        if app is not None:
            app.quit()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        """Close means quit once nothing is left running or open.

        Hiding the last window would leave an invisible process holding the GPU.
        """
        reason = self.busy_reason()
        if reason:
            event.ignore()
            self.hide()
            _notify_hidden(self, reason)
            return

        event.accept()
        if not _other_windows_visible(self) and not _first_busy_reason():
            app = QApplication.instance()
            if app is not None:
                app.quit()


def _visible_shells() -> list[ShellWindow]:
    app = QApplication.instance()
    if app is None:
        return []
    return [w for w in app.topLevelWidgets() if isinstance(w, ShellWindow) and w.isVisible()]


def _other_windows_visible(current: QWidget) -> bool:
    return any(window is not current for window in _visible_shells())


def _first_busy_reason() -> str | None:
    app = QApplication.instance()
    if app is None:
        return None
    for window in app.topLevelWidgets():
        if isinstance(window, ShellWindow):
            reason = window.busy_reason()
            if reason:
                return reason
    return None


def _confirm(parent: QWidget, reason: str) -> bool:
    answer = QMessageBox.question(
        parent,
        "仍有任务在进行",
        f"{reason}\n\n现在退出会中断它，确定退出吗？",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    return answer == QMessageBox.StandardButton.Yes


def _notify_hidden(window: QWidget, reason: str) -> None:
    QMessageBox.information(
        window,
        "已最小化到托盘",
        f"{reason}\n\n窗口已隐藏，任务继续进行。要立即退出请按 Ctrl+Q。",
    )


def confirm_quit(parent: QWidget, on_quit: Callable[[], None]) -> None:
    """Quit path for surfaces that are not ShellWindows, such as the tray menu."""
    reason = _first_busy_reason()
    if reason and not _confirm(parent, reason):
        return
    on_quit()
