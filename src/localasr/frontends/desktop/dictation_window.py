"""A small, visible control surface for push-to-talk dictation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QThread, QTimer, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from localasr.apps.coordinator import Activity
from localasr.apps.dictation import (
    FAILED,
    IDLE,
    PREPARING,
    RECORDING,
    TRANSCRIBING,
    DictationController,
)
from localasr.apps.events import (
    AudioDropped,
    DictationStateChanged,
    FinalTranscript,
    SegmentDropped,
    TextEmitted,
)
from localasr.context import AppContext
from localasr.frontends.desktop.backend_panel import BackendPanel
from localasr.frontends.desktop.backend_panel import setup_panel as make_setup_panel
from localasr.frontends.desktop.device_panel import DevicePanel
from localasr.frontends.desktop.model_panel import ModelPanel
from localasr.frontends.desktop.shell import ShellWindow
from localasr.frontends.desktop.theme import set_role, set_tone
from localasr.platform import text_output
from localasr.refine.types import (
    FidelityIssue,
    RefinementMode,
    RefinementRequest,
    RefinementResult,
)

_STATE_COPY = {
    IDLE: ("就绪", "neutral", "点击开始，说完后再次点击；识别结果会保留在窗口中。"),
    PREPARING: ("加载模型", "active", "正在检查模型并启动识别引擎，首次启动约需数秒…"),
    RECORDING: ("正在聆听", "danger", "请自然说话；每次停顿后会追加已完成的识别片段。"),
    TRANSCRIBING: ("正在识别", "active", "正在整理并发送文字，请稍候…"),
    FAILED: ("需要处理", "warning", "听写没有完成，请查看下方提示后重试。"),
}


class _ControllerThread(QThread):
    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, operation: Callable[[], str | None]) -> None:
        super().__init__()
        self._operation = operation

    def run(self) -> None:
        try:
            self.completed.emit(self._operation() or "")
        except Exception as exc:  # noqa: BLE001 - keep the desktop host alive
            self.failed.emit(str(exc))




class _RefineThread(QThread):
    """Refinement is a network round trip that may first swap models on the node."""

    done = Signal(object)

    def __init__(self, context: AppContext, request: RefinementRequest) -> None:
        super().__init__()
        self._context = context
        self._request = request

    def run(self) -> None:
        try:
            self.done.emit(self._context.refine(self._request))
        except Exception as exc:  # noqa: BLE001 - a thread that dies silently is worse
            # `refine` is documented as never raising, but a thread that ends without
            # emitting leaves the button re-enabled and nothing on screen, which is
            # indistinguishable from the request having been ignored. Belt and braces.
            self.done.emit(
                RefinementResult.rejected(
                    self._request,
                    refined_text="",
                    issues=(FidelityIssue("unavailable", f"整理失败：{exc}"),),
                )
            )


class DictationWindow(ShellWindow):
    """Status, last result and mouse controls for the tray-based dictation app."""

    def __init__(
        self,
        context: AppContext,
        bridge,  # noqa: ANN001 - Qt signal owner
        controller: DictationController,
    ) -> None:
        super().__init__()
        self.context = context
        self.controller = controller
        self._worker: _ControllerThread | None = None
        self._abort_requested = False
        self._rendered = 0
        self._operation_error: str | None = None
        self._completion_detail: str | None = None
        self._worker_reason = "正在识别"
        self._starting = False
        self._hidden_for_delivery = False
        self._refiner: _RefineThread | None = None
        self._state = IDLE
        """The last state shown. Enablement is computed from it in one place, so that
        presentation and availability cannot drift apart — which they did: starting a
        refinement disabled 开始识别 and finishing one never re-enabled it."""

        self.setObjectName("appSurface")
        self.setWindowTitle("LocalASR — 语音输入")
        # Both values clear the layout's own minimum (845 px), which is dominated by the
        # transcript's 240 px floor plus the refinement row and the two pane headings.
        # Below that minimum the wrapped guidance labels are clipped rather than
        # reflowed.
        self.resize(1000, 920)
        self.setMinimumSize(620, 850)

        title = QLabel("语音输入")
        title.setObjectName("pageTitle")
        self.state_badge = QLabel()
        self.state_badge.setObjectName("statusPill")

        # The subtitle used to sit here saying "按一次开始录音，再按一次完成识别"; the
        # status line below the button says the same thing and updates as the state
        # changes, so keeping both cost a row to repeat something already on screen.
        heading = QHBoxLayout()
        heading.addWidget(title, 1)
        heading.addWidget(self.state_badge, 0)
        panel = make_setup_panel(context, bridge, self)
        self.backend_panel = panel if isinstance(panel, BackendPanel) else None
        self.model_panel = panel if isinstance(panel, ModelPanel) else None

        self.device_panel = DevicePanel(context)

        status_card = QFrame()
        status_card.setObjectName("card")
        status_layout = QVBoxLayout(status_card)
        status_layout.setContentsMargins(16, 12, 16, 12)
        status_layout.setSpacing(9)
        # No "当前状态" header: the pill in the title bar already names the state, and a
        # section title for a card holding one button is a row spent on nothing.
        self.status_message = QLabel()
        self.status_message.setWordWrap(True)
        self.status_message.setProperty("muted", True)
        self.toggle_button = QPushButton("开始识别")
        # Not full width. A stretched button is a large target but a weak signal — it
        # looks the same whichever action it is about to perform. Colour carries that
        # instead: blue to start, red to stop, switched in `_show_state`.
        self.toggle_button.setMinimumWidth(148)
        set_role(self.toggle_button, "primary")
        self.auto_deliver = QCheckBox("暂停后自动输入到原窗口")
        self.capture_system = QCheckBox("同时识别系统声音")
        self.capture_system.setToolTip(
            "默认关闭：听写记录的是你说的话，开启后播放中的视频旁白也会被写进来"
        )
        self.auto_deliver.setToolTip(
            "开启后，完成识别时会暂时隐藏 LocalASR，并把文字发送到此前使用的应用"
        )
        # The action and its two options share one row: the button no longer spans the
        # card, so the space beside it is exactly where the options belong.
        action = QHBoxLayout()
        action.setSpacing(18)
        action.addWidget(self.toggle_button)
        action.addStretch(1)
        action.addWidget(self.auto_deliver)
        action.addWidget(self.capture_system)
        status_layout.addLayout(action)
        status_layout.addWidget(self.status_message)

        result_card = QFrame()
        result_card.setObjectName("card")
        result_layout = QVBoxLayout(result_card)
        result_layout.setContentsMargins(16, 12, 16, 14)
        result_layout.setSpacing(9)
        result_heading = QHBoxLayout()
        self.result_title = QLabel("实时识别结果")
        self.result_title.setObjectName("sectionTitle")
        self.copy_button = QPushButton("复制")
        self.copy_button.setEnabled(False)
        # Takes the refinement when there is one. It is the button people reach for
        # after asking for a tidy-up, and handing back the untidied version is not what
        # that gesture means.
        self.clear_button = QPushButton("清空")
        self.clear_button.setEnabled(False)
        result_heading.addWidget(self.result_title)
        result_heading.addStretch(1)  # buttons are inserted after the title
        result_heading.addWidget(self.copy_button)
        # 复制原文 belongs beside 复制, not across the row from it: they are the same
        # action on the two panes, and separating them made the pair read as unrelated.
        self.copy_raw_button = QPushButton("复制原文")
        self.copy_raw_button.setEnabled(False)
        self.copy_raw_button.setToolTip("复制左侧的原始转写，不含整理结果")
        result_heading.addWidget(self.copy_raw_button)
        result_heading.addWidget(self.clear_button)
        self.last_text = QTextEdit()
        # Editable: the point of dictation is to fix the one word it got wrong before
        # sending the text on. Copy, cut, paste, delete and undo come with QTextEdit.
        self.last_text.setReadOnly(False)
        self.last_text.setUndoRedoEnabled(True)
        self.last_text.setPlaceholderText(
            "开始识别后，每说完一句（停顿约 0.6 秒）就会自动追加到这里。"
        )
        # This is what the window is for, so it gets the floor and all the slack. The
        # setup cards above it are fixed-height by construction, so every pixel gained
        # by compacting them and every pixel of a resize lands here.
        self.last_text.setMinimumHeight(240)

        # The refined pane sits beside the original rather than replacing it. A tidy-up
        # is a proposal, not a correction: the only way to judge it is to read both, and
        # the raw transcript stays the thing that was actually said.
        self.refined_text = QTextEdit()
        self.refined_text.setReadOnly(True)
        self.refined_text.setPlaceholderText(
            "整理结果会出现在这里，原文始终保留在左侧。\n\n"
            "通不过校验时会显示原文并说明原因——一个看起来合理的错数字，"
            "比明显未经润色的文字危险得多。"
        )
        self.refined_text.setMinimumHeight(240)

        # Both panes are visible from the start, each under its own heading. Hiding the
        # right one until a refinement succeeded meant the window could not explain its
        # own shape: on opening you saw a single full-width box and a greyed button, and
        # nothing said the two-pane comparison was the point.
        raw_title = QLabel("原始转写")
        raw_title.setObjectName("sectionTitle")
        refined_title = QLabel("整理结果")
        refined_title.setObjectName("sectionTitle")
        titles = QHBoxLayout()
        titles.setSpacing(10)
        titles.addWidget(raw_title, 1)
        titles.addWidget(refined_title, 1)

        panes = QHBoxLayout()
        panes.setSpacing(10)
        panes.addWidget(self.last_text, 1)
        panes.addWidget(self.refined_text, 1)

        # Saving writes both layers, not just the tidied one. A refinement without the
        # transcript beside it is a claim with its evidence thrown away — and under a
        # custom instruction it is a rewrite whose fidelity was never proven.
        self.save_refined_button = QPushButton("另存整理结果…")
        self.save_refined_button.setEnabled(False)
        self.save_refined_button.setToolTip("整理完成后可用；保存时会一并写入原始转写")
        self.save_refined_button.clicked.connect(self._save_refined)
        result_heading.insertWidget(1, self.save_refined_button)

        # The instruction sits next to the button that uses it, not in a settings dialog:
        # it is the thing most likely to change between one recording and the next.
        self.instruction = QLineEdit(self.context.settings.refine_instruction)
        self.instruction.setPlaceholderText(
            "整理要求（留空 = 只加标点、删口头填充，逐字保留）"
        )
        self.instruction.setToolTip(
            "例如：提取成待办列表 / 写成会议纪要 / 改写成正式邮件。\n"
            "留空时使用保守清理，可以证明没有增删实义内容；\n"
            "填入要求后模型会改写，只能做风险筛查，请与左侧原文核对。"
        )
        self.instruction.setClearButtonEnabled(True)
        self.refine_button = QPushButton("整理文本")
        self.refine_button.setEnabled(False)
        self.refine_button.setMinimumWidth(110)

        refine_row = QHBoxLayout()
        refine_row.setSpacing(8)
        refine_row.addWidget(self.instruction, 1)
        refine_row.addWidget(self.refine_button)

        self.refine_status = QLabel()
        self.refine_status.setWordWrap(True)
        self.refine_status.setProperty("muted", True)
        self.refine_status.hide()

        result_layout.addLayout(result_heading)
        result_layout.addLayout(refine_row)
        result_layout.addLayout(titles)
        result_layout.addLayout(panes, 1)
        result_layout.addWidget(self.refine_status)

        self.delivery = QLabel(self._delivery_summary())
        self.delivery.setWordWrap(True)
        self.delivery.setProperty("muted", True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(11)
        layout.addLayout(heading)
        layout.addWidget(panel)
        layout.addWidget(self.device_panel)
        layout.addWidget(status_card)
        layout.addWidget(result_card, 1)
        layout.addWidget(self.delivery)

        self.toggle_button.clicked.connect(
            lambda: self.toggle(deliver=self.auto_deliver.isChecked())
        )
        self.refine_button.clicked.connect(self._refine)
        self.instruction.editingFinished.connect(self._save_instruction)
        self.instruction.returnPressed.connect(self._refine)
        self.copy_raw_button.clicked.connect(self._copy_raw)
        self.copy_button.clicked.connect(self._copy_last)
        self.clear_button.clicked.connect(self._clear_result)
        self.last_text.textChanged.connect(self._result_changed)
        # The refined pane drives its own buttons. They used to be set once, inside the
        # handler that produced a result, which meant nothing afterwards could correct
        # them — including 清空, which had to remember to switch them off by hand.
        self.refined_text.textChanged.connect(self._result_changed)
        bridge.event.connect(self._on_event)
        self.install_exits()
        self._show_state(IDLE)
        # Buttons and their tooltips are derived state; without this they start blank
        # and only become explanatory after the first keystroke.
        self._result_changed()

    def toggle(self, *, deliver: bool = True) -> None:
        """Start recognising, pause it, or abort a preparation still in flight."""
        if self._starting:
            self.cancel()
            return
        if self._worker is not None:
            return
        if self.controller.recording:

            def operation() -> str:
                return self.controller.finish(deliver=deliver)

            # Only hide when the user explicitly asked for automatic delivery. The
            # default keeps the window visible, so stopping cannot look like a crash
            # and the result remains available even without ydotool/clipboard.
            if deliver and self.isActiveWindow():
                self.hide()
                self._hidden_for_delivery = True
                QTimer.singleShot(
                    200,
                    lambda: self._run_background(
                        operation,
                        "没有检测到可识别的语音，请靠近麦克风后重试。",
                        "正在完成识别",
                    ),
                )
            else:
                self._run_background(
                    operation,
                    "没有检测到可识别的语音，请靠近麦克风后重试。",
                    "正在完成识别",
                )
            return
        self._prepare_and_start()

    def _prepare_and_start(self) -> None:
        blocking = self.context.coordinator.conflicts_with(Activity.DICTATION)
        force = False
        if blocking:
            names = "、".join(item.value for item in blocking)
            answer = QMessageBox.question(
                self,
                "识别引擎正在使用",
                f"{names} 正在使用识别引擎。继续后听写可能需要等待，是否继续？",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            force = True

        def operation() -> str:
            # `start` prepares the engine before opening the device; see the controller.
            self.controller.options = replace(
                self.controller.options, capture_system=self.capture_system.isChecked()
            )
            self.controller.start(force=force)
            if self._abort_requested:
                # The user pressed cancel while the engine was loading. Honour it now
                # rather than leaving a live microphone they already asked to stop.
                self.controller.cancel()
            return ""

        self._starting = True
        self._abort_requested = False
        self._run_background(operation, reason="正在准备识别引擎")
        self._show_state(PREPARING)

    def cancel(self) -> None:
        if self._starting:
            self._abort_requested = True
            self.toggle_button.setEnabled(False)
            self.status_message.setText("已请求取消；引擎准备结束后立即停止。")
            return
        if self._worker is None and self.controller.recording:
            self._run_background(self.controller.cancel, "已取消本次听写。", "正在取消")

    def cancel_if_running(self) -> None:
        self.device_panel.wait_for_probe()
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(15_000)
        elif self.controller.recording:
            self.controller.cancel()

    def _run_background(
        self,
        operation: Callable[[], str | None],
        empty_detail: str | None = None,
        reason: str = "正在识别",
    ) -> None:
        self._operation_error = None
        self._completion_detail = empty_detail
        self._worker_reason = reason
        self._worker = _ControllerThread(operation)
        self._worker.completed.connect(self._operation_done)
        self._worker.failed.connect(self._operation_failed)
        self._worker.finished.connect(self._thread_finished)
        self._worker.start()
        self.toggle_button.setEnabled(False)
        self.auto_deliver.setEnabled(False)
        self.capture_system.setEnabled(False)

    def _operation_done(self, text: str) -> None:
        if self._starting and self.controller.recording:
            # The box is never emptied automatically. It may hold text the user has
            # already corrected, or is still copying from, and starting another round
            # of dictation is not a request to throw that away — 清空 is.
            # The counter still resets, because the controller begins a new session
            # with an empty segment list and the next segment is again index 0.
            self._rendered = 0
        if text:
            self._append_new_segments()
            self._completion_detail = "识别完成；文字可直接在下方编辑、复制或剪切。"

    def _operation_failed(self, message: str) -> None:
        self._operation_error = message
        self.controller.state = FAILED
        self.context.bus.publish(DictationStateChanged(FAILED, message))
        self._show_state(FAILED, message)

    def _thread_finished(self) -> None:
        self._worker = None
        self._starting = False
        self.auto_deliver.setEnabled(True)
        self.capture_system.setEnabled(True)
        # 整理 and 清空 are also gated on no worker running, and this is the only moment
        # that becomes true. The text itself arrived earlier, from _operation_done, back
        # when this thread was still alive — so without asking again here they stay grey
        # for the whole session and dictation can never be refined.
        self._result_changed()
        if self._hidden_for_delivery:
            # The window hid itself so the text would go to the previous application.
            # Leaving it hidden looks exactly like the app having crashed on stop.
            self._hidden_for_delivery = False
            self.show()
            self.raise_()
        if self._operation_error is not None:
            self._show_state(FAILED, self._operation_error)
        elif self.controller.state == FAILED:
            self._show_state(FAILED)
        else:
            self._show_state(self.controller.state, self._completion_detail)

    def _append_new_segments(self) -> None:
        """Add only the sentences not shown yet, at the end.

        Replacing the whole box with the controller's text would be simpler and would
        discard the user's edits every time another sentence arrived.
        """
        settled = self.controller.segments
        fresh = [segment.text for segment in settled[self._rendered:] if segment.text]
        self._rendered = len(settled)
        if not fresh:
            return
        cursor = self.last_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        separator = (
            "" if not self.last_text.toPlainText() else self.controller.options.join_separator
        )
        cursor.insertText(separator + "".join(fresh))
        self.last_text.setTextCursor(cursor)
        self.last_text.ensureCursorVisible()

    def _result_changed(self) -> None:
        """Recompute every control from the window's state, in one place.

        There are three mutually exclusive activities here — 识别中, 整理中, idle — and two
        bodies of text: the transcript on the left, which the user and the recogniser both
        write to, and the refinement on the right, which is derived from it. Three rules
        follow, and every control below is one of them:

        * **Reading is always safe.** Copying or saving cannot corrupt anything, so those
          are gated only on there being something to copy.
        * **Changing the transcript waits for a refinement of it.** 清空 or a new round of
          dictation while the model is working leaves the right pane describing text that
          is no longer on the left — under a heading that tells the user to compare them.
        * **Starting an activity waits for the other one.** Including 整理 itself.

        These used to be decided per button, and disagreed: 复制 ignored everything, 清空
        watched the worker, 整理 watched the worker and recording, and none of them watched
        whether a refinement was already running — so appending a segment mid-refinement
        re-enabled 整理文本, which then did nothing when pressed, because `_refine` refuses
        a second one. A button that is lit and inert is worse than one that is grey.
        """
        has_raw = bool(self.last_text.toPlainText().strip())
        has_refined = bool(self.refined_text.toPlainText().strip())
        # One expression for "is recognition happening", used by every control below.
        # The toggle used to answer it from `self._state` while everything else asked the
        # controller, which is two sources of truth for one question — the shape of bug
        # this method exists to prevent.
        recognising = (
            self._worker is not None
            or self.controller.recording
            or self._state in {RECORDING, TRANSCRIBING}
        )
        refining = self._refiner is not None

        self.copy_button.setEnabled(has_raw or has_refined)
        self.copy_button.setToolTip(
            "复制整理结果（原文用「复制原文」）" if has_refined else "复制原始转写"
        )
        self.copy_raw_button.setEnabled(has_raw)
        self.save_refined_button.setEnabled(has_refined)

        # 开始识别 belongs here too, not in `_show_state`. Split across the two, the two
        # drifted: a refinement starting went through both and a refinement finishing
        # through only this one, so the record button stayed grey for the rest of the
        # session. PREPARING is the exception — the same button is the way out of a slow
        # first load, so it stays live.
        # PREPARING is the exception: the same button is the only way out of a slow first
        # load, so it stays live while everything else waits.
        preparing = self._state == PREPARING and not self._abort_requested
        self.toggle_button.setEnabled(preparing or (not self._blocking() and not refining))
        self.device_panel.set_enabled(not recognising and not refining)

        self.clear_button.setEnabled(has_raw and not recognising and not refining)
        self.refine_button.setEnabled(
            has_raw and not recognising and not refining and self.context.can_refine
        )
        # The instruction is an input to a refinement, so while one is running, editing it
        # can only mislead: the request has already been sent under the old text.
        self.instruction.setEnabled(not refining)

        # A greyed button with no explanation is indistinguishable from a broken one.
        if not self.context.can_refine:
            self.refine_button.setToolTip("未配置整理服务：在 config.toml 设置 refiner_model_id")
        elif refining:
            self.refine_button.setToolTip("正在整理，完成后可再次整理")
        elif not has_raw:
            self.refine_button.setToolTip("先识别出文字，或直接在左侧粘贴要整理的内容")
        elif recognising:
            self.refine_button.setToolTip("录音或识别进行中，结束后再整理")
        else:
            self.refine_button.setToolTip("按当前要求整理左侧文字，结果显示在右侧")
        self.clear_button.setToolTip(
            "整理进行中不能清空原文，否则右侧结果将无从核对"
            if refining
            else "清空左右两栏"
        )

    def _blocking(self) -> bool:
        """Work that pressing 开始识别 could not interrupt or usefully queue behind."""
        return self._state == TRANSCRIBING or self._worker is not None

    def _clear_result(self) -> None:
        self.last_text.clear()
        self._rendered = len(self.controller.segments)
        self.refined_text.clear()
        self.refine_status.hide()

    def _refine(self) -> None:
        """Send the current transcript for tidying, on a worker thread."""
        raw = self.last_text.toPlainText().strip()
        if not raw or self._refiner is not None:
            return
        instruction = self.instruction.text().strip()
        self.refine_status.setText(
            f"正在按「{instruction}」整理…" if instruction else "正在整理…"
        )
        self.refine_status.show()

        # No source_segment_ids here. The left pane is editable and survives across
        # rounds of dictation, so its text is no longer in one-to-one correspondence
        # with the controller's segments — claiming a mapping that may be wrong is worse
        # than claiming none. Journals, where the mapping is real, still record them.
        self._save_instruction()
        request = self.context.refinement_request(raw)
        self._refiner = _RefineThread(self.context, request)
        self._refiner.done.connect(self._refined)
        self._refiner.finished.connect(self._refine_finished)
        self._refiner.start()
        self._show_state(self.controller.state)

    def _refined(self, result: object) -> None:
        self.refined_text.setPlainText(result.text)

        if result.accepted and result.mode is not RefinementMode.CONSERVATIVE:
            # A custom instruction rewrites; the subsequence proof does not hold and
            # saying "已完成" alone would imply a guarantee that is not there.
            warnings = "；".join(i.detail for i in result.warnings)
            self.refine_status.setText(
                "已按你的要求改写——这一模式无法证明内容未被改动，请与左侧原文核对。"
                + (f" 注意：{warnings}" if warnings else "")
            )
        elif result.accepted:
            # No count of what was removed. The model used to report that itself, which
            # is exactly the number not to trust: in the one real refinement measured on
            # the Orin it dropped 「我们」 and would not have counted it. An unverified
            # tally reads as assurance, and the only real check here is a person reading
            # both panes.
            warnings = "；".join(i.detail for i in result.warnings)
            note = "整理完成，请与左侧原文核对。"
            self.refine_status.setText(f"{note}{' 注意：' + warnings if warnings else ''}")
        else:
            # Never show the model's version when a check failed. The pane holds the
            # original, and the reason is stated rather than left as silence.
            reasons = "；".join(i.detail for i in result.errors)
            self.refine_status.setText(f"整理结果未通过校验，右侧沿用原文。原因：{reasons}")

    def _refine_finished(self) -> None:
        self._refiner = None
        self._result_changed()

    def _save_refined(self) -> None:
        """Write the refinement out as its own file, with the transcript it came from."""
        refined = self.refined_text.toPlainText()
        if not refined.strip():
            return
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        default = str(Path.home() / f"localasr-{stamp}.md")
        path_text, _ = QFileDialog.getSaveFileName(
            self, "另存整理结果", default, "Markdown (*.md);;纯文本 (*.txt)"
        )
        if not path_text:
            return

        path = Path(path_text)
        instruction = self.instruction.text().strip()
        if path.suffix.lower() == ".txt":
            body = refined
        else:
            body = (
                f"# 整理结果\n\n"
                f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}\n"
                f"- 整理要求：{instruction or '（保守清理：只加标点、删口头填充）'}\n\n"
                f"{refined}\n\n---\n\n## 原始转写\n\n"
                f"{self.last_text.toPlainText()}\n"
            )
        try:
            path.write_text(body, encoding="utf-8")
        except OSError as exc:
            self.refine_status.setText(f"保存失败：{exc}")
            return
        self.refine_status.setText(f"已保存到 {path}")

    def _save_instruction(self) -> None:
        """Persist the standing instruction; it is the same one most of the time."""
        text = self.instruction.text().strip()
        if text == self.context.settings.refine_instruction:
            return
        self.context.settings.refine_instruction = text
        try:
            self.context.settings.save()
        except OSError as exc:
            self.refine_status.setText(f"整理要求本次有效，但保存失败：{exc}")
            self.refine_status.show()

    def _copy_raw(self) -> None:
        """The transcript exactly as it was recognised."""
        self._copy(self.last_text.toPlainText(), "原文")

    def _copy_last(self) -> None:
        """Whichever of the two the user most likely wants.

        The refinement when one exists: asking for a tidy-up and then being handed the
        untidied text back is the one outcome nobody wants from this button. Which of
        the two it took is said out loud, because a button that changes meaning halfway
        through a session must not do it silently.
        """
        refined = self.refined_text.toPlainText()
        if refined.strip():
            self._copy(refined, "整理结果")
        else:
            self._copy(self.last_text.toPlainText(), "原文")

    def _copy(self, text: str, what: str) -> None:
        from PySide6.QtWidgets import QApplication

        if not text.strip():
            return
        QApplication.clipboard().setText(text)
        self.status_message.setText(f"已复制{what}到剪贴板。")







    def _on_event(self, event: object) -> None:
        if isinstance(event, DictationStateChanged):
            detail = event.detail
            if event.state == RECORDING and self.controller.current_text:
                detail = "已显示最新识别片段；仍在录音，可以继续说话。"
            self._show_state(event.state, detail)
        elif isinstance(event, FinalTranscript) and self.controller.recording:
            self._append_new_segments()
        elif isinstance(event, SegmentDropped):
            self.status_message.setText(
                f"有一段语音未被采用（{event.reason}）；如果那是你说的话，请重说一次。"
            )
        elif isinstance(event, AudioDropped):
            self.status_message.setText(
                f"丢失了 {event.seconds:.2f} 秒音频（{event.reason}）。"
            )
        elif isinstance(event, TextEmitted):
            # Do not overwrite: the box already holds every sentence, possibly edited.
            self._append_new_segments()
            if event.method == "clipboard":
                self.status_message.setText("文字已复制到剪贴板，请在目标应用中按 Ctrl+V。")

    def _show_state(self, state: str, detail: str | None = None) -> None:
        """What the window says it is doing. Availability follows, via _result_changed."""
        self._state = state
        label, tone, guidance = _STATE_COPY.get(state, (state, "neutral", ""))
        self.state_badge.setText(label)
        set_tone(self.state_badge, tone)
        self.status_message.setText(detail or guidance)
        preparing = state == PREPARING and not self._abort_requested
        if preparing:
            # Nothing to pause yet; the same button is the way out of a slow first
            # launch, which otherwise leaves the window inert until the model loads.
            self.toggle_button.setText("取消准备")
        else:
            self.toggle_button.setText("暂停识别" if state == RECORDING else "开始识别")
        # Red whenever pressing it stops something, blue when it starts something. The
        # label alone is easy to misread at a glance while you are talking.
        stopping = preparing or state == RECORDING
        set_role(self.toggle_button, "stop" if stopping else "primary")
        # Enablement is not decided here. This method is reached only when the state
        # changes, and buttons also have to answer to things that change without one —
        # a refinement ending, text being edited. One function owns that; this one owns
        # what the window *says*.
        self._result_changed()

    @staticmethod
    def _delivery_summary() -> str:
        methods = text_output.available_methods()
        typers = [method for method in methods if method != "clipboard"]
        if typers:
            return f"发送方式：{typers[0]}（失败时自动复制到剪贴板）"
        if "clipboard" in methods:
            return "发送方式：剪贴板。识别后请在目标应用中按 Ctrl+V。"
        return "尚无可用的文字发送方式；请先运行 localasr doctor 查看修复建议。"

    def busy_reason(self) -> str | None:
        model_reason = self.model_panel.busy_reason() if self.model_panel else None
        if model_reason:
            return model_reason
        if self.controller.recording:
            return "正在录音"
        if self._worker is not None and self._worker.isRunning():
            return self._worker_reason
        if self.device_panel.probing:
            return "正在测试麦克风"
        return None
