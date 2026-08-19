"""Subtitle window: a small batch queue with explicit output decisions."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
)

from localasr.apps.coordinator import ActivityConflict
from localasr.apps.events import (
    JobCancelled,
    JobFailed,
    JobFinished,
    JobStarted,
    SegmentDropped,
    SegmentResumed,
    SegmentTranscribed,
)
from localasr.apps.subtitle import SubtitleController, SubtitleOptions, expand_media
from localasr.context import AppContext
from localasr.formats.subtitles import WRITERS
from localasr.frontends.desktop.backend_panel import BackendPanel
from localasr.frontends.desktop.backend_panel import setup_panel as make_setup_panel
from localasr.frontends.desktop.model_panel import ModelPanel
from localasr.frontends.desktop.shell import ShellWindow
from localasr.frontends.desktop.theme import set_role, set_tone


class _JobThread(QThread):
    """Run a batch off the GUI thread; progress arrives through the event bus."""

    finished_with = Signal(list)
    failed = Signal(str)

    def __init__(self, controller: SubtitleController, media: list[Path], publish) -> None:  # noqa: ANN001
        super().__init__()
        self._controller = controller
        self._media = media
        self._publish = publish

    def run(self) -> None:
        try:
            self.finished_with.emit(self._controller.run(self._media, listener=self._publish))
        except ActivityConflict as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001 - surface, never crash the GUI
            self.failed.emit(str(exc))


class SubtitleWindow(ShellWindow):
    def __init__(self, context: AppContext, bridge) -> None:  # noqa: ANN001
        super().__init__()
        self.context = context
        self.setObjectName("appSurface")
        self.setWindowTitle("LocalASR — 字幕生成")
        self.resize(760, 700)
        self.setMinimumSize(650, 640)
        self.setAcceptDrops(True)

        self._files: list[Path] = []
        self._items: dict[Path, QListWidgetItem] = {}
        self._thread: _JobThread | None = None
        self._controller: SubtitleController | None = None
        self._output_dir: Path | None = None
        self._last_output: Path | None = None
        self._dropped = 0

        title = QLabel("字幕生成")
        title.setObjectName("pageTitle")
        subtitle = QLabel("添加音视频文件，LocalASR 会按顺序生成字幕")
        subtitle.setObjectName("pageSubtitle")
        self.queue_badge = QLabel("0 个文件")
        self.queue_badge.setObjectName("statusPill")
        set_tone(self.queue_badge, "neutral")
        heading_text = QVBoxLayout()
        heading_text.setSpacing(3)
        heading_text.addWidget(title)
        heading_text.addWidget(subtitle)
        heading = QHBoxLayout()
        heading.addLayout(heading_text, 1)
        heading.addWidget(self.queue_badge)
        panel = make_setup_panel(context, bridge, self)
        self.backend_panel = panel if isinstance(panel, BackendPanel) else None
        self.model_panel = panel if isinstance(panel, ModelPanel) else None

        drop_card = QFrame()
        drop_card.setObjectName("dropZone")
        drop_layout = QVBoxLayout(drop_card)
        drop_layout.setContentsMargins(12, 12, 12, 12)
        toolbar = QHBoxLayout()
        queue_title = QLabel("待处理文件")
        queue_title.setObjectName("sectionTitle")
        self.add_button = QPushButton("添加文件…")
        self.folder_button = QPushButton("添加文件夹…")
        self.remove_button = QPushButton("移除所选")
        self.clear_button = QPushButton("清空")
        self.remove_button.setEnabled(False)
        self.clear_button.setEnabled(False)
        toolbar.addWidget(queue_title)
        toolbar.addStretch(1)
        toolbar.addWidget(self.add_button)
        toolbar.addWidget(self.folder_button)
        toolbar.addWidget(self.remove_button)
        toolbar.addWidget(self.clear_button)

        self.list = QListWidget()
        self.list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.empty = QLabel("把音频或视频拖到这里\n也可以一次添加整个文件夹")
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty.setProperty("muted", True)
        self.stack = QStackedWidget()
        self.stack.addWidget(self.empty)
        self.stack.addWidget(self.list)
        drop_layout.addLayout(toolbar)
        drop_layout.addWidget(self.stack, 1)

        settings_card = QFrame()
        settings_card.setObjectName("card")
        settings_layout = QVBoxLayout(settings_card)
        settings_layout.setContentsMargins(18, 14, 18, 14)
        settings_title = QLabel("输出设置")
        settings_title.setObjectName("sectionTitle")
        options = QHBoxLayout()
        self.format_box = QComboBox()
        self.format_box.addItems(sorted(WRITERS))
        self.format_box.setCurrentText("srt")
        self.output_label = QLabel("与源文件相同")
        self.output_label.setProperty("muted", True)
        self.output_label.setToolTip("每个字幕文件保存在对应音视频旁边")
        self.output_button = QPushButton("更改目录…")
        self.output_reset_button = QPushButton("恢复原目录")
        self.output_reset_button.setEnabled(False)
        options.addWidget(QLabel("格式"))
        options.addWidget(self.format_box)
        options.addSpacing(14)
        options.addWidget(QLabel("保存到"))
        options.addWidget(self.output_label, 1)
        options.addWidget(self.output_button)
        options.addWidget(self.output_reset_button)
        safeguards = QHBoxLayout()
        self.resume_box = QCheckBox("继续未完成的任务")
        self.resume_box.setChecked(True)
        self.overwrite_box = QCheckBox("覆盖同名字幕")
        self.overwrite_box.setToolTip("默认不覆盖：已有字幕会安全跳过")
        safeguards.addWidget(self.resume_box)
        safeguards.addWidget(self.overwrite_box)
        safeguards.addStretch(1)
        settings_layout.addWidget(settings_title)
        settings_layout.addLayout(options)
        settings_layout.addLayout(safeguards)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.status = QLabel("等待添加文件")
        self.status.setWordWrap(True)
        self.status.setProperty("muted", True)
        self.open_output_button = QPushButton("打开输出目录")
        self.open_output_button.setEnabled(False)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setEnabled(False)
        set_role(self.cancel_button, "danger")
        self.start_button = QPushButton("开始生成字幕")
        self.start_button.setEnabled(False)
        set_role(self.start_button, "primary")
        actions = QHBoxLayout()
        actions.addWidget(self.status, 1)
        actions.addWidget(self.open_output_button)
        actions.addWidget(self.cancel_button)
        actions.addWidget(self.start_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(15)
        layout.addLayout(heading)
        layout.addWidget(panel)
        layout.addWidget(drop_card, 1)
        layout.addWidget(settings_card)
        layout.addWidget(self.progress)
        layout.addLayout(actions)

        self.add_button.clicked.connect(self._choose_files)
        self.folder_button.clicked.connect(self._choose_folder)
        self.remove_button.clicked.connect(self._remove_selected)
        self.clear_button.clicked.connect(self._clear)
        self.list.itemSelectionChanged.connect(self._selection_changed)
        self.output_button.clicked.connect(self._choose_output_dir)
        self.output_reset_button.clicked.connect(self._reset_output_dir)
        self.open_output_button.clicked.connect(self._open_output_dir)
        self.start_button.clicked.connect(self._start)
        self.install_exits()
        self.cancel_button.clicked.connect(self._cancel)
        bridge.event.connect(self._on_event)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802 - Qt API
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802 - Qt API
        self._add([Path(url.toLocalFile()) for url in event.mimeData().urls()])
        event.acceptProposedAction()

    def _choose_files(self) -> None:
        names, _ = QFileDialog.getOpenFileNames(
            self,
            "选择音视频文件",
            filter=(
                "音视频文件 (*.wav *.mp3 *.flac *.m4a *.ogg *.opus "
                "*.mp4 *.mkv *.mov *.webm);;所有文件 (*)"
            ),
        )
        self._add([Path(name) for name in names])

    def _choose_folder(self) -> None:
        name = QFileDialog.getExistingDirectory(self, "选择包含音视频的文件夹")
        if name:
            self._add([Path(name)])

    def _add(self, paths: list[Path]) -> None:
        added = 0
        invalid: list[str] = []
        for path in paths:
            try:
                media = expand_media([path])
            except FileNotFoundError:
                invalid.append(path.name or str(path))
                continue
            for item_path in media:
                if item_path in self._items:
                    continue
                self._files.append(item_path)
                item = QListWidgetItem(f"{item_path.name}\n{item_path.parent}")
                item.setData(Qt.ItemDataRole.UserRole, str(item_path))
                item.setToolTip(str(item_path))
                self.list.addItem(item)
                self._items[item_path] = item
                added += 1
        if invalid:
            self.status.setText(f"未找到：{', '.join(invalid[:3])}")
        elif added == 0 and paths:
            self.status.setText("没有发现新的音视频文件")
        else:
            self.status.setText(f"已添加 {added} 个文件")
        self._update_queue()

    def _remove_selected(self) -> None:
        selected = list(self.list.selectedItems())
        for item in selected:
            path = Path(item.data(Qt.ItemDataRole.UserRole))
            self._items.pop(path, None)
            if path in self._files:
                self._files.remove(path)
            self.list.takeItem(self.list.row(item))
        self._update_queue()

    def _clear(self) -> None:
        self._files.clear()
        self._items.clear()
        self.list.clear()
        self.status.setText("等待添加文件")
        self.progress.setValue(0)
        self._update_queue()

    def _selection_changed(self) -> None:
        self.remove_button.setEnabled(bool(self.list.selectedItems()) and self._thread is None)

    def _update_queue(self) -> None:
        count = len(self._files)
        self.queue_badge.setText(f"{count} 个文件")
        set_tone(self.queue_badge, "active" if count else "neutral")
        self.stack.setCurrentWidget(self.list if count else self.empty)
        self.start_button.setEnabled(bool(count) and self._thread is None)
        self.clear_button.setEnabled(bool(count) and self._thread is None)
        self._selection_changed()

    def _choose_output_dir(self) -> None:
        name = QFileDialog.getExistingDirectory(self, "选择字幕输出目录")
        if not name:
            return
        self._output_dir = Path(name)
        self.output_label.setText(str(self._output_dir))
        self.output_label.setToolTip(str(self._output_dir))
        self.output_reset_button.setEnabled(True)

    def _reset_output_dir(self) -> None:
        self._output_dir = None
        self.output_label.setText("与源文件相同")
        self.output_label.setToolTip("每个字幕文件保存在对应音视频旁边")
        self.output_reset_button.setEnabled(False)

    def _start(self) -> None:
        if not self._files or self._thread is not None:
            return
        self._controller = SubtitleController(
            self.context,
            SubtitleOptions(
                fmt=self.format_box.currentText(),
                language=self.context.settings.language,
                output_dir=self._output_dir,
                overwrite=self.overwrite_box.isChecked(),
                resume=self.resume_box.isChecked(),
            ),
        )
        self._thread = _JobThread(self._controller, list(self._files), self.context.bus.publish)
        self._thread.finished_with.connect(self._done)
        self._thread.failed.connect(self._failed)
        self._thread.finished.connect(self._thread_finished)
        self._thread.start()
        self._set_running(True)
        self.status.setText("正在准备音频和识别引擎…")

    def _cancel(self) -> None:
        if self._controller is not None:
            self._controller.cancel()
            self.cancel_button.setEnabled(False)
            self.status.setText("正在取消；当前语音片段完成后停止…")

    def _done(self, results: list) -> None:
        written = [
            result
            for result in results
            if result.output and not result.error and not result.skipped
        ]
        skipped = [result for result in results if result.skipped]
        failed = [result for result in results if result.error]
        for result in results:
            item = self._items.get(result.media)
            if item is None:
                continue
            marker = "完成"
            if result.skipped:
                marker = "已跳过（输出存在）"
            elif result.error:
                marker = f"失败：{result.error}"
            item.setText(f"{result.media.name}  ·  {marker}\n{result.media.parent}")
            if result.output and not result.error:
                self._last_output = result.output
        self.status.setText(
            f"处理完成：生成 {len(written)}，跳过 {len(skipped)}，失败 {len(failed)}"
        )
        self.progress.setValue(self.progress.maximum())
        self.open_output_button.setEnabled(self._last_output is not None)
        self._reset_job()

    def _failed(self, message: str) -> None:
        self.status.setText(f"无法开始任务：{message}")
        self._reset_job()

    def _reset_job(self) -> None:
        self._controller = None
        self._set_running(False)

    def _thread_finished(self) -> None:
        self._thread = None
        self._update_queue()

    def _set_running(self, running: bool) -> None:
        for widget in (
            self.add_button,
            self.folder_button,
            self.format_box,
            self.output_button,
            self.resume_box,
            self.overwrite_box,
        ):
            widget.setEnabled(not running)
        self.start_button.setEnabled(not running and bool(self._files))
        self.clear_button.setEnabled(not running and bool(self._files))
        self.output_reset_button.setEnabled(not running and self._output_dir is not None)
        self.cancel_button.setEnabled(running)
        self.remove_button.setEnabled(not running and bool(self.list.selectedItems()))

    def _open_output_dir(self) -> None:
        if self._last_output is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_output.parent)))

    def _on_event(self, event: object) -> None:
        if isinstance(event, JobStarted):
            self._dropped = 0
            self.progress.setRange(0, event.total_segments or 0)
            self.progress.setValue(0)
            if event.media is not None:
                self.status.setText(f"正在处理 {event.media.name}")
        elif isinstance(event, SegmentTranscribed | SegmentResumed):
            self.progress.setValue(event.index)
            action = "已恢复" if isinstance(event, SegmentResumed) else "已识别"
            self.status.setText(
                f"{action} {event.index}/{self.progress.maximum()}：{event.segment.text[:60]}"
            )
        elif isinstance(event, SegmentDropped):
            # Same defect class the dictation window had: a segment that vanishes with
            # no explanation is indistinguishable from audio the pipeline lost.
            self._dropped += 1
            self.status.setText(
                f"已跳过 {self._dropped} 段无法采用的语音；最近一处在 "
                f"{event.span.start:.1f}s（{event.reason}）"
            )
        elif isinstance(event, JobFailed):
            self.status.setText(f"失败：{event.error}")
        elif isinstance(event, JobCancelled):
            self.status.setText(f"已取消；{event.completed} 段已完成的结果保留在断点日志中")
        elif isinstance(event, JobFinished):
            self.progress.setValue(self.progress.maximum())

    def busy_reason(self) -> str | None:
        model_reason = self.model_panel.busy_reason() if self.model_panel else None
        if model_reason:
            return model_reason
        if self._thread is not None and self._thread.isRunning():
            return "字幕任务正在运行"
        return None
