"""Tray shortcuts and at-a-glance state for dictation."""

from __future__ import annotations

from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QMenu, QSystemTrayIcon

from localasr.apps.dictation import FAILED, IDLE, RECORDING, TRANSCRIBING, DictationController
from localasr.apps.events import DictationStateChanged, TextEmitted
from localasr.context import AppContext

_COLOURS = {
    IDLE: "#7f8c8d",
    RECORDING: "#e74c3c",
    TRANSCRIBING: "#f39c12",
    FAILED: "#c0392b",
}
_LABELS = {
    IDLE: "空闲",
    RECORDING: "录音中",
    TRANSCRIBING: "识别中",
    FAILED: "失败",
}


def _dot(colour: str, size: int = 32) -> QIcon:
    """A coloured dot, so the tray works without shipping icon assets."""
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor(colour))
    painter.setPen(QColor(colour))
    painter.drawEllipse(4, 4, size - 8, size - 8)
    painter.end()
    return QIcon(pixmap)


class DictationTray:
    def __init__(
        self,
        context: AppContext,
        bridge,  # noqa: ANN001 - Qt signal owner
        host,  # noqa: ANN001 - DesktopHost; avoiding an import cycle
        controller: DictationController,
    ) -> None:
        self.context = context
        self.host = host
        self.controller = controller

        self.icon = QSystemTrayIcon(_dot(_COLOURS[IDLE]))
        self.icon.setToolTip("LocalASR — 空闲")

        menu = QMenu()
        # Every action is parented to the menu. `QMenu.addAction` does not take
        # ownership, so an action held only by a local name is garbage-collected the
        # moment this constructor returns and its entry vanishes from the menu. That is
        # what removed 退出 — the two actions that survived were the two kept as
        # attributes — and it left the tray with no way to quit at all.
        open_action = QAction("打开语音输入…", menu)
        open_action.triggered.connect(lambda: host.show("dictation"))
        menu.addAction(open_action)

        self.toggle_action = QAction("开始听写", menu)
        self.toggle_action.triggered.connect(host.toggle_dictation)
        menu.addAction(self.toggle_action)
        self.cancel_action = QAction("取消本次听写", menu)
        self.cancel_action.setEnabled(False)
        self.cancel_action.triggered.connect(host.cancel_dictation)
        menu.addAction(self.cancel_action)
        menu.addSeparator()

        subtitle_action = QAction("字幕生成…", menu)
        subtitle_action.triggered.connect(lambda: host.show("subtitle"))
        menu.addAction(subtitle_action)

        meeting_action = QAction("会议记录…", menu)
        meeting_action.triggered.connect(lambda: host.show("meeting"))
        menu.addAction(meeting_action)
        menu.addSeparator()

        self.quit_action = QAction("退出 LocalASR", menu)
        self.quit_action.triggered.connect(self._quit)
        menu.addAction(self.quit_action)

        self.menu = menu
        self.icon.setContextMenu(menu)
        self.icon.activated.connect(self._activated)
        self.icon.show()

        bridge.event.connect(self._on_event)

    def announce(self) -> None:
        """Confirm readiness without requiring the control window to remain open."""
        from localasr.platform import text_output

        self.icon.showMessage(
            "LocalASR 语音输入已就绪",
            f"{text_output.summary()}\n可从语音输入窗口、托盘或全局快捷键开始听写",
            QSystemTrayIcon.MessageIcon.Information,
            5000,
        )

    def _activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.host.show("dictation")

    def hide(self) -> None:
        """Take the icon out of the status area before the process goes away.

        An icon left registered when its process exits is worse than no icon: it stays
        in the tray, and clicking it reaches nothing. Hiding it explicitly means the
        status area never shows a LocalASR that is not running.
        """
        self.icon.setContextMenu(None)
        self.icon.hide()

    def _quit(self) -> None:
        from PySide6.QtWidgets import QApplication

        from localasr.frontends.desktop.shell import confirm_quit

        app = QApplication.instance()
        # A parentless modal dialog is unreliable on Wayland — it can open behind
        # everything or not at all, which turns "quit" into a click that does nothing.
        # Confirmation belongs on a real window whenever one is on screen.
        parent = next(
            (w for w in app.topLevelWidgets() if w.isVisible() and w.isWindow()), None
        )
        confirm_quit(parent, app.quit)

    def _on_event(self, event: object) -> None:
        if isinstance(event, DictationStateChanged):
            self.icon.setIcon(_dot(_COLOURS.get(event.state, _COLOURS[IDLE])))
            label = _LABELS.get(event.state, event.state)
            suffix = f"：{event.detail}" if event.detail else ""
            self.icon.setToolTip(f"LocalASR — {label}{suffix}")
            self.toggle_action.setText("停止听写" if event.state == RECORDING else "开始听写")
            self.toggle_action.setEnabled(event.state != TRANSCRIBING)
            self.cancel_action.setEnabled(event.state == RECORDING)
            if event.state == FAILED and event.detail:
                self.icon.showMessage("听写失败", event.detail, QSystemTrayIcon.MessageIcon.Warning)
        elif isinstance(event, TextEmitted) and event.method == "clipboard":
            self.icon.showMessage(
                "已复制到剪贴板",
                "无法直接输入，请按 Ctrl+V 粘贴",
                QSystemTrayIcon.MessageIcon.Information,
            )
