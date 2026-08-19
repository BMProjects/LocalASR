"""The tray menu, and the fact that quitting has to be reachable from it.

The bug this file exists for: `QMenu.addAction` does not take ownership, so actions
held only by a local name were collected as soon as the constructor returned. Four of
the seven entries disappeared — including 退出 — leaving a tray icon with no way to
quit the application it represented.
"""

from __future__ import annotations

import gc
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6", reason="desktop frontend requires the [gui] extra")

from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from localasr.apps.dictation import RECORDING, DictationController  # noqa: E402
from localasr.apps.events import DictationStateChanged  # noqa: E402
from localasr.context import AppContext  # noqa: E402
from localasr.frontends.desktop.event_bridge import QtEventBridge  # noqa: E402
from localasr.frontends.desktop.theme import apply_theme  # noqa: E402
from localasr.frontends.desktop.tray import DictationTray  # noqa: E402

EXPECTED_ENTRIES = [
    "打开语音输入…",
    "开始听写",
    "取消本次听写",
    "字幕生成…",
    "会议记录…",
    "退出 LocalASR",
]


class FakeHost:
    def __init__(self) -> None:
        self.shown: list[str] = []
        self.toggled = 0
        self.cancelled = 0

    def show(self, which: str) -> None:
        self.shown.append(which)

    def toggle_dictation(self) -> None:
        self.toggled += 1

    def cancel_dictation(self) -> None:
        self.cancelled += 1


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    apply_theme(app)
    yield app


@pytest.fixture(autouse=True)
def no_real_dialogs(monkeypatch):
    """Keep the quit path hermetic.

    `confirm_quit` asks every top-level ShellWindow whether it is busy, so windows left
    behind by other test modules can make it open a genuine modal dialog — which, with
    no one to click it, hangs the run. Both halves are neutralised here and re-patched
    by the tests that are actually about them.
    """
    monkeypatch.setattr("localasr.frontends.desktop.shell._first_busy_reason", lambda: None)
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
    )


@pytest.fixture
def tray(qt_app):
    context = AppContext()
    bridge = QtEventBridge(context.bus)
    host = FakeHost()
    icon = DictationTray(context, bridge, host, DictationController(context))
    # The collection is the point: it is what used to destroy most of the menu.
    gc.collect()
    yield icon, host
    bridge.stop()


def _labels(menu) -> list[str]:  # noqa: ANN001
    return [action.text() for action in menu.actions() if not action.isSeparator()]


def test_every_menu_entry_survives_garbage_collection(tray) -> None:
    icon, _host = tray
    assert _labels(icon.menu) == EXPECTED_ENTRIES


def test_the_tray_can_quit_the_application(tray, monkeypatch) -> None:
    """Without this entry the tray is a status light with no exit, and closing the
    windows does not necessarily end the process."""
    icon, _host = tray
    quits = []
    monkeypatch.setattr(QApplication.instance(), "quit", lambda: quits.append(True))

    quit_action = next(a for a in icon.menu.actions() if a.text() == "退出 LocalASR")
    assert quit_action.isEnabled()
    quit_action.trigger()

    assert quits == [True]


def test_quitting_while_busy_asks_first(tray, monkeypatch) -> None:
    icon, _host = tray
    quits = []
    monkeypatch.setattr(QApplication.instance(), "quit", lambda: quits.append(True))
    monkeypatch.setattr(
        "localasr.frontends.desktop.shell._first_busy_reason", lambda: "正在录音"
    )
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.No)
    )

    next(a for a in icon.menu.actions() if a.text() == "退出 LocalASR").trigger()
    assert quits == [], "answering No must not quit"

    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
    )
    next(a for a in icon.menu.actions() if a.text() == "退出 LocalASR").trigger()
    assert quits == [True]


def test_the_other_entries_reach_the_host(tray) -> None:
    icon, host = tray
    for label in ("打开语音输入…", "字幕生成…"):
        next(a for a in icon.menu.actions() if a.text() == label).trigger()
    assert host.shown == ["dictation", "subtitle"]

    next(a for a in icon.menu.actions() if a.text() == "开始听写").trigger()
    assert host.toggled == 1


def test_hiding_removes_the_icon_from_the_status_area(tray) -> None:
    """An icon still registered after its process exits stays in the tray and answers
    nothing, which looks exactly like a quit that failed."""
    icon, _host = tray
    icon.hide()
    assert not icon.icon.isVisible()


def test_the_recording_state_relabels_the_toggle(tray) -> None:
    icon, _host = tray
    icon._on_event(DictationStateChanged(RECORDING, None))
    assert icon.toggle_action.text() == "停止听写"
    assert icon.cancel_action.isEnabled()
