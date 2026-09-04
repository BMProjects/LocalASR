"""Refinement as the dictation window exposes it.

The property that matters most here is negative: when a check fails, the pane must show
the original and say why. A tidy-up the user cannot distinguish from the transcript is
the failure mode the whole validator exists to prevent.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6", reason="desktop frontend requires the [gui] extra")

from PySide6.QtWidgets import QApplication  # noqa: E402

from localasr.apps.dictation import DictationController  # noqa: E402
from localasr.context import AppContext  # noqa: E402
from localasr.frontends.desktop.dictation_window import DictationWindow  # noqa: E402
from localasr.frontends.desktop.event_bridge import QtEventBridge  # noqa: E402
from localasr.frontends.desktop.theme import apply_theme  # noqa: E402
from localasr.refine.types import (
    Note,  # noqa: E402
    RefinementMode,
    RefinementRequest,
    RefinementResult,
)

RAW = "嗯我们下周一交三个报告"


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    apply_theme(app)
    yield app


@pytest.fixture
def window(qt_app):
    context = AppContext()
    bridge = QtEventBridge(context.bus)
    win = DictationWindow(context, bridge, DictationController(context))
    yield win
    bridge.stop()


def _accepted(raw: str, refined: str) -> RefinementResult:
    return RefinementResult(
        raw_text=raw, refined_text=refined, mode=RefinementMode.CONSERVATIVE
    )


def test_refining_is_unavailable_without_a_node(window) -> None:  # noqa: ANN001
    """A second llama-server will not fit beside ASR on a 4 GB card, so the honest
    state is 'absent and explained', not a button that fails when pressed."""
    window.last_text.setPlainText(RAW)

    assert not window.context.can_refine
    assert not window.refine_button.isEnabled()
    assert "未配置整理服务" in window.refine_button.toolTip()


def test_a_noted_refinement_is_shown_with_the_note(window) -> None:  # noqa: ANN001
    """The reverse of what this used to assert. Withholding the refinement and putting
    the transcript in its place is what made three reports read as "the model just
    repeated my text" — and there is nobody to protect, since both panes are on screen
    and the user copies deliberately. The note points; it does not withhold.
    """
    noted = RefinementResult(
        raw_text=RAW,
        refined_text="我们下周五交五个报告。",
        mode=RefinementMode.CONSERVATIVE,
        notes=(Note("number_new", "结果里的数字原文中没有：['5']"),),
    )
    window._refined(noted)

    assert window.refined_text.toPlainText() == "我们下周五交五个报告。"
    assert "['5']" in window.refine_status.text(), "the number change is pointed at"


def test_an_accepted_refinement_is_shown_beside_the_original(window) -> None:  # noqa: ANN001
    window.last_text.setPlainText(RAW)
    window._refined(_accepted(RAW, "我们下周一交三个报告。"))

    assert window.last_text.toPlainText() == RAW, "the original stays put"
    assert window.refined_text.toPlainText() == "我们下周一交三个报告。"
    assert window.refined_text.isVisibleTo(window)
    # 复制 now takes the refinement; the transcript stays reachable beside it.
    assert window.copy_button.isEnabled() and window.copy_raw_button.isEnabled()
    assert window.save_refined_button.isEnabled()


def test_the_refined_pane_is_read_only(window) -> None:  # noqa: ANN001
    """Editing it would produce a third version with no provenance at all."""
    assert window.refined_text.isReadOnly()
    assert not window.last_text.isReadOnly()


def test_an_unavailable_service_explains_itself(window) -> None:  # noqa: ANN001
    result = window.context.refine(RefinementRequest(RAW))

    assert not result.ok
    assert result.text == RAW
    window._refined(result)
    assert "未配置计算节点" in window.refine_status.text()


def test_refining_is_blocked_while_recording(window, monkeypatch) -> None:  # noqa: ANN001
    """Mid-recording it would tidy half a sentence and, on a node in sequential mode,
    evict the ASR model still in use."""
    monkeypatch.setattr(type(window.context), "can_refine", property(lambda self: True))
    monkeypatch.setattr(type(window.controller), "recording", property(lambda self: True))
    window.last_text.setPlainText(RAW)
    assert not window.refine_button.isEnabled(), "must not refine mid-recording"

    # Same text, same node, but idle: now it is allowed.
    monkeypatch.setattr(type(window.controller), "recording", property(lambda self: False))
    window._result_changed()
    assert window.refine_button.isEnabled()


def test_clearing_removes_both_panes(window) -> None:  # noqa: ANN001
    window.last_text.setPlainText(RAW)
    window._refined(_accepted(RAW, "我们下周一交三个报告。"))
    window._clear_result()

    assert window.last_text.toPlainText() == ""
    assert window.refined_text.toPlainText() == ""
    # The pane stays on screen; clearing empties it rather than removing the comparison.
    assert window.refined_text.isVisibleTo(window)
    assert not window.save_refined_button.isEnabled()


def test_a_crashing_refinement_still_reports_something(qt_app) -> None:
    """A worker thread that ends without emitting leaves the button live and the screen
    blank — the user cannot tell that from the click having been ignored."""
    from localasr.frontends.desktop.dictation_window import _RefineThread

    context = AppContext()
    thread = _RefineThread(context, RefinementRequest(RAW))
    received = []
    thread.done.connect(received.append)

    def explode(_request):  # noqa: ANN001, ANN202
        raise RuntimeError("boom")

    context.refine = explode
    thread.run()

    assert len(received) == 1
    assert not received[0].ok
    assert received[0].text == RAW
    assert "整理失败" in received[0].failure


def test_saving_writes_the_transcript_alongside_the_refinement(  # noqa: ANN001
    window, tmp_path, monkeypatch
) -> None:
    """A refinement without the transcript beside it is a claim with its evidence thrown
    away — and under a custom instruction it is a rewrite whose fidelity was never
    proven."""
    from PySide6.QtWidgets import QFileDialog

    target = tmp_path / "out.md"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(target), ""))
    )
    window.instruction.setPlainText("提取成待办列表")
    window.last_text.setPlainText(RAW)
    window._refined(_accepted(RAW, "- 交三个报告\n- 准备演示"))
    window._save_refined()

    written = target.read_text(encoding="utf-8")
    assert "- 交三个报告" in written
    assert RAW in written, "the original must travel with the rewrite"
    assert "提取成待办列表" in written, "and so must what was asked for"
    assert "已保存到" in window.refine_status.text()


def test_saving_plain_text_writes_only_the_refinement(window, tmp_path, monkeypatch) -> None:  # noqa: ANN001
    from PySide6.QtWidgets import QFileDialog

    target = tmp_path / "out.txt"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(target), ""))
    )
    window._refined(_accepted(RAW, "整理后的文字。"))
    window._save_refined()

    assert target.read_text(encoding="utf-8") == "整理后的文字。"


def test_the_two_pane_layout_is_visible_before_any_refinement(window) -> None:  # noqa: ANN001
    """The right pane used to be hidden until a refinement succeeded, so opening the
    window showed one full-width box and a greyed button — nothing on screen said the
    side-by-side comparison was the point of the design."""
    window.show()

    assert window.refined_text.isVisibleTo(window), "the comparison must be visible empty"
    assert window.refined_text.toPlainText() == ""
    assert window.refined_text.placeholderText(), "an empty pane must explain itself"
    assert window.instruction.isVisibleTo(window)
    assert window.refine_button.isVisibleTo(window)
    # Present but disabled, rather than absent: the actions are part of the shape.
    for button in (window.copy_raw_button, window.save_refined_button):
        assert button.isVisibleTo(window)
        assert not button.isEnabled()


def test_every_disabled_state_of_the_refine_button_explains_itself(window, monkeypatch) -> None:  # noqa: ANN001
    """A greyed button with no explanation is indistinguishable from a broken one.

    Each precondition is set explicitly rather than inherited from whatever config the
    machine happens to carry, so the three messages are checked, not one of them twice.
    """
    monkeypatch.setattr(type(window.context), "can_refine", property(lambda self: False))
    window._result_changed()
    assert not window.refine_button.isEnabled()
    assert "未配置整理服务" in window.refine_button.toolTip()

    monkeypatch.setattr(type(window.context), "can_refine", property(lambda self: True))
    window.last_text.clear()
    window._result_changed()
    assert not window.refine_button.isEnabled()
    assert "先识别出文字" in window.refine_button.toolTip()

    monkeypatch.setattr(type(window.controller), "recording", property(lambda self: True))
    window.last_text.setPlainText(RAW)
    window._result_changed()
    assert not window.refine_button.isEnabled()
    assert "录音或识别进行中" in window.refine_button.toolTip()
