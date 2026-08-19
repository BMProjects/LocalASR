"""Meeting window: live transcript, source health and recoverable export."""

from __future__ import annotations

import html
from pathlib import Path

from PySide6.QtCore import QElapsedTimer, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from localasr.apps.coordinator import ActivityConflict
from localasr.apps.events import (
    AudioDropped,
    FinalTranscript,
    SegmentDropped,
    SessionFailed,
)
from localasr.apps.meeting import (
    MeetingController,
    MeetingOptions,
    MeetingSession,
    default_journal_path,
)
from localasr.capture.microphone import CaptureError
from localasr.context import AppContext
from localasr.formats.subtitles import render
from localasr.frontends.desktop.backend_panel import BackendPanel
from localasr.frontends.desktop.backend_panel import setup_panel as make_setup_panel
from localasr.frontends.desktop.device_panel import DevicePanel
from localasr.frontends.desktop.model_panel import ModelPanel
from localasr.frontends.desktop.shell import ShellWindow
from localasr.frontends.desktop.theme import set_role, set_tone

LABELS = {"mic": "我", "system": "对方"}
EXPORT_FORMATS = ("md", "txt", "srt", "json")


class _StopThread(QThread):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, controller: MeetingController) -> None:
        super().__init__()
        self._controller = controller

    def run(self) -> None:
        try:
            self.completed.emit(self._controller.stop())
        except Exception as exc:  # noqa: BLE001 - surface failures without freezing/crashing Qt
            self.failed.emit(str(exc))


class MeetingWindow(ShellWindow):
    def __init__(self, context: AppContext, bridge) -> None:  # noqa: ANN001
        super().__init__()
        self.context = context
        self.setObjectName("appSurface")
        self.setWindowTitle("LocalASR — 会议记录")
        self.resize(820, 820)
        self.setMinimumSize(700, 790)

        self._controller: MeetingController | None = None
        self._stop_thread: _StopThread | None = None
        self._journal: Path | None = None
        self._last_session: MeetingSession | None = None
        self._elapsed = QElapsedTimer()
        self._dropped = 0
        self._dropped_segments = 0

        title = QLabel("会议记录")
        title.setObjectName("pageTitle")
        subtitle = QLabel("实时记录麦克风与会议播放声音，原始日志持续落盘")
        subtitle.setObjectName("pageSubtitle")
        self.state_badge = QLabel("未开始")
        self.state_badge.setObjectName("statusPill")
        set_tone(self.state_badge, "neutral")
        self.elapsed_label = QLabel("00:00:00")
        self.elapsed_label.setObjectName("sectionTitle")
        header_text = QVBoxLayout()
        header_text.setSpacing(3)
        header_text.addWidget(title)
        header_text.addWidget(subtitle)
        header = QHBoxLayout()
        header.addLayout(header_text, 1)
        header.addWidget(self.elapsed_label)
        header.addSpacing(10)
        header.addWidget(self.state_badge)
        panel = make_setup_panel(context, bridge, self)
        self.backend_panel = panel if isinstance(panel, BackendPanel) else None
        self.model_panel = panel if isinstance(panel, ModelPanel) else None

        capture_card = QFrame()
        capture_card.setObjectName("card")
        capture_layout = QVBoxLayout(capture_card)
        capture_layout.setContentsMargins(18, 14, 18, 14)
        capture_top = QHBoxLayout()
        capture_title = QLabel("录制来源")
        capture_title.setObjectName("sectionTitle")
        self.device_panel = DevicePanel(context)
        self.system_box = QCheckBox("包含系统声音（在线会议中的对方）")
        self.system_box.setChecked(True)
        capture_top.addWidget(capture_title)
        capture_top.addStretch(1)
        capture_top.addWidget(self.system_box)
        self.notice = QLabel(
            "「我」= 麦克风，「对方」= 系统声音。这是音源标记，不是说话人分离；"
            "同一房间内多人通过麦克风发言时都会标记为「我」。"
        )
        self.notice.setWordWrap(True)
        self.notice.setProperty("muted", True)
        capture_layout.addLayout(capture_top)
        capture_layout.addWidget(self.notice)

        transcript_card = QFrame()
        transcript_card.setObjectName("card")
        transcript_layout = QVBoxLayout(transcript_card)
        transcript_layout.setContentsMargins(14, 13, 14, 14)
        transcript_heading = QHBoxLayout()
        transcript_title = QLabel("实时转写")
        transcript_title.setObjectName("sectionTitle")
        self.segment_count = QLabel("0 条")
        self.segment_count.setProperty("muted", True)
        transcript_heading.addWidget(transcript_title)
        transcript_heading.addStretch(1)
        transcript_heading.addWidget(self.segment_count)
        self.transcript = QTextEdit()
        self.transcript.setReadOnly(True)
        self.transcript.setPlaceholderText("开始会议后，识别完成的发言会按时间顺序显示在这里。")
        transcript_layout.addLayout(transcript_heading)
        transcript_layout.addWidget(self.transcript, 1)

        self.status = QLabel("点击“开始记录”创建可恢复的会议日志")
        self.status.setWordWrap(True)
        self.status.setProperty("muted", True)
        self.start_button = QPushButton("开始记录")
        set_role(self.start_button, "primary")
        self.pause_button = QPushButton("暂停")
        self.stop_button = QPushButton("停止记录")
        set_role(self.stop_button, "danger")
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        live_controls = QHBoxLayout()
        live_controls.addWidget(self.status, 1)
        live_controls.addWidget(self.pause_button)
        live_controls.addWidget(self.stop_button)
        live_controls.addWidget(self.start_button)

        export_card = QFrame()
        export_card.setObjectName("card")
        export_layout = QHBoxLayout(export_card)
        export_layout.setContentsMargins(18, 12, 18, 12)
        export_title = QLabel("停止后导出")
        export_title.setObjectName("sectionTitle")
        self.export_format = QComboBox()
        self.export_format.addItems(EXPORT_FORMATS)
        self.export_button = QPushButton("另存为…")
        self.export_button.setEnabled(False)
        self.open_journal_button = QPushButton("打开日志目录")
        self.open_journal_button.setEnabled(False)
        export_layout.addWidget(export_title)
        export_layout.addStretch(1)
        export_layout.addWidget(QLabel("格式"))
        export_layout.addWidget(self.export_format)
        export_layout.addWidget(self.export_button)
        export_layout.addWidget(self.open_journal_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(14)
        layout.addLayout(header)
        layout.addWidget(panel)
        layout.addWidget(self.device_panel)
        layout.addWidget(capture_card)
        layout.addWidget(transcript_card, 1)
        layout.addLayout(live_controls)
        layout.addWidget(export_card)

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._update_elapsed)
        self.start_button.clicked.connect(self._start)
        self.pause_button.clicked.connect(self._toggle_pause)
        self.stop_button.clicked.connect(self._stop)
        self.export_button.clicked.connect(self._export)
        self.open_journal_button.clicked.connect(self._open_journal_dir)
        bridge.event.connect(self._on_event)
        self.install_exits()

    def _start(self) -> None:
        if self._controller is not None or self._stop_thread is not None:
            return
        try:
            self.context.require_model_available()
        except RuntimeError as exc:
            self.status.setText(str(exc))
            self._set_state("需要准备模型", "warning")
            return
        self._journal = default_journal_path()
        controller = MeetingController(
            self.context,
            MeetingOptions(
                language=self.context.settings.language,
                mic_device=self.device_panel.selected,
                capture_system=self.system_box.isChecked(),
            ),
        )
        try:
            session = controller.start(self._journal)
        except (CaptureError, ActivityConflict, RuntimeError) as exc:
            self.status.setText(f"无法开始：{exc}")
            self._set_state("需要处理", "warning")
            return

        self._controller = controller
        self._last_session = None
        self._dropped = 0
        self._dropped_segments = 0
        self.transcript.clear()
        self.segment_count.setText("0 条")
        warning = "；".join(session.warnings)
        if warning:
            self.status.setText(f"已降级运行：{warning}")
        else:
            self.status.setText(f"日志实时保存到 {self._journal}")
        self._elapsed.start()
        self._timer.start()
        self._set_state("录制中", "danger")
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.system_box.setEnabled(False)
        self.device_panel.set_enabled(False)
        self.export_button.setEnabled(False)

    def _toggle_pause(self) -> None:
        if self._controller is None:
            return
        if self._controller.paused:
            self._controller.resume()
            self.pause_button.setText("暂停")
            self.status.setText("已继续录制，日志仍在实时保存")
            self._set_state("录制中", "danger")
        else:
            self._controller.pause()
            self.pause_button.setText("继续")
            self.status.setText("已暂停；恢复后时间轴继续沿用本次会议")
            self._set_state("已暂停", "warning")

    def _stop(self) -> None:
        if self._controller is None or self._stop_thread is not None:
            return
        self._set_state("正在收尾", "active")
        self.status.setText("正在处理最后一段语音并关闭日志…")
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self._stop_thread = _StopThread(self._controller)
        self._stop_thread.completed.connect(self._stopped)
        self._stop_thread.failed.connect(self._stop_failed)
        self._stop_thread.finished.connect(self._stop_thread_finished)
        self._stop_thread.start()

    def _stopped(self, session: MeetingSession | None) -> None:
        self._controller = None
        self._timer.stop()
        self._last_session = session
        self.start_button.setEnabled(True)
        self.pause_button.setText("暂停")
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.system_box.setEnabled(True)
        self.device_panel.set_enabled(True)
        self.open_journal_button.setEnabled(self._journal is not None)
        has_segments = bool(session and session.segments)
        self.export_button.setEnabled(has_segments)
        if has_segments:
            self.status.setText(
                f"记录完成，共 {len(session.segments)} 条；原始日志已安全保存，可选择格式导出"
            )
            self._set_state("已完成", "success")
        else:
            self.status.setText("记录已停止，但没有检测到可转写的语音")
            self._set_state("无语音", "neutral")

    def _stop_failed(self, message: str) -> None:
        self._controller = None
        self._timer.stop()
        self.start_button.setEnabled(True)
        self.system_box.setEnabled(True)
        self.device_panel.set_enabled(True)
        self.status.setText(f"停止时发生错误：{message}；请保留原始日志用于恢复")
        self._set_state("需要处理", "warning")
        self.open_journal_button.setEnabled(self._journal is not None)

    def _stop_thread_finished(self) -> None:
        self._stop_thread = None

    def _export(self) -> None:
        session = self._last_session
        if session is None or not session.segments:
            return
        fmt = self.export_format.currentText()
        target, _ = QFileDialog.getSaveFileName(
            self,
            "导出会议记录",
            str(session.path.with_suffix(f".{fmt}")),
            f"{fmt.upper()} (*.{fmt})",
        )
        if not target:
            return
        try:
            Path(target).write_text(render(session.transcript(), fmt), encoding="utf-8")
        except OSError as exc:
            self.status.setText(f"导出失败：{exc}")
            self._set_state("需要处理", "warning")
            return
        self.status.setText(f"已导出到 {target}")
        self._set_state("已导出", "success")

    def _open_journal_dir(self) -> None:
        if self._journal is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._journal.parent)))

    def _update_elapsed(self) -> None:
        if not self._elapsed.isValid():
            return
        seconds = self._elapsed.elapsed() // 1000
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.elapsed_label.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")

    def _set_state(self, text: str, tone: str) -> None:
        self.state_badge.setText(text)
        set_tone(self.state_badge, tone)

    def stop_if_running(self) -> None:
        self.device_panel.wait_for_probe()
        if self._stop_thread is not None and self._stop_thread.isRunning():
            self._stop_thread.wait(15_000)
        elif self._controller is not None:
            self._controller.stop()
            self._controller = None

    def _on_event(self, event: object) -> None:
        # FinalTranscript is shared with dictation; only consume it while this meeting
        # owns an active controller.
        if self._controller is None:
            return
        if isinstance(event, FinalTranscript):
            segment = event.segment
            meeting = self._controller.meeting
            # Dictation may be explicitly forced while a meeting is active. Both use
            # FinalTranscript, so only render utterances recorded by this controller.
            owned = meeting is not None and (
                segment in meeting.segments
                if segment.utterance_id is None
                else any(saved.utterance_id == segment.utterance_id for saved in meeting.segments)
            )
            if not owned:
                return
            who = LABELS.get(segment.source or "", "语音")
            colour = "#1558a0" if segment.source == "system" else "#126344"
            self.transcript.append(
                f'<p><span style="color:{colour};font-weight:650">{who}</span> '
                f'<span style="color:#7a8798">{segment.start:7.1f}s</span> '
                f"{html.escape(segment.text)}</p>"
            )
            count = len(meeting.segments)
            self.segment_count.setText(f"{count} 条")
        elif isinstance(event, AudioDropped):
            self._dropped += 1
            self.status.setText(
                f"检测到 {self._dropped} 次音频中断；"
                f"最近丢失 {event.seconds:.2f}s（{event.source}）"
            )
            self._set_state("录制中 · 有警告", "warning")
        elif isinstance(event, SegmentDropped):
            self._dropped_segments += 1
            self.transcript.append(
                f'<p style="color:#8a6d3b"><i>跳过一段无法采用的语音（{event.reason}）</i></p>'
            )
            self._set_state("录制中 · 有跳过", "warning")
        elif isinstance(event, SessionFailed):
            self.status.setText(f"采集或识别错误：{event.error}")
            self._set_state("需要处理", "warning")

    def busy_reason(self) -> str | None:
        if self.device_panel.probing:
            return "正在测试麦克风"
        model_reason = self.model_panel.busy_reason() if self.model_panel else None
        if model_reason:
            return model_reason
        if self._controller is not None:
            return "会议正在录制中"
        if self._stop_thread is not None and self._stop_thread.isRunning():
            return "会议正在收尾"
        return None
