"""Compact model selection shared by all three desktop applications."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from localasr.apps.events import ModelChanged, ModelDownloaded
from localasr.context import AppContext
from localasr.frontends.desktop.model_state import DiskState, Intent, ModelFacts, plan
from localasr.frontends.desktop.theme import set_role
from localasr.registry import imported, manager
from localasr.registry.manager import ModelSpec


def _gib(size: int) -> str:
    return f"{size / (1024**3):.2f} GiB"


def _model_label(spec: ModelSpec) -> str:
    size = sum(file.size for file in spec.files)
    return f"{spec.model_id}  ·  {_gib(size)}"


class _ModelActionThread(QThread):
    completed = Signal()
    failed = Signal(str)
    progressed = Signal(int)

    def __init__(self, operation: Callable[[Callable[[str, int, int], None]], None]) -> None:
        super().__init__()
        self._operation = operation

    def run(self) -> None:
        try:
            self._operation(self._progress)
        except Exception as exc:  # noqa: BLE001 - report errors without killing Qt
            self.failed.emit(str(exc))
        else:
            self.completed.emit()

    def _progress(self, _name: str, completed: int, total: int) -> None:
        self.progressed.emit(round(completed * 100 / total) if total else 0)


class ModelPanel(QFrame):
    """Catalog-only model chooser; selection and download are always separate actions."""

    def __init__(self, context: AppContext, bridge, parent: QWidget | None = None) -> None:  # noqa: ANN001
        super().__init__(parent)
        self.context = context
        self._thread: _ModelActionThread | None = None
        self.setObjectName("card")

        title = QLabel("识别模型")
        title.setObjectName("sectionTitle")
        self.model_box = QComboBox()
        self.model_box.setMinimumContentsLength(18)
        self.model_box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.primary_button = QPushButton("使用此模型")
        self.import_button = QPushButton("导入模型…")
        self.status = QLabel()
        # One line, elided, with the full text on hover. Wrapped, this label ran to
        # three lines of model path and took more of the window than the transcript.
        self.status.setWordWrap(False)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.status.setProperty("muted", True)
        self.status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.hide()

        # One row. Choosing a model is setup, done once; it should not cost the same
        # vertical space as the results the window exists to show.
        chooser = QHBoxLayout()
        chooser.setSpacing(8)
        chooser.addWidget(title)
        chooser.addWidget(self.model_box, 1)
        chooser.addWidget(self.primary_button)
        chooser.addWidget(self.import_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(6)
        layout.addLayout(chooser)
        layout.addWidget(self.status)
        layout.addWidget(self.progress)

        # ASR only. The catalog also carries the refinement model, which is a
        # different role with its own server and must never appear in the chooser that
        # decides what transcribes audio.
        for spec in manager.models_of_kind(manager.ModelKind.ASR):
            self.model_box.addItem(_model_label(spec), spec.model_id)
        self.model_box.currentIndexChanged.connect(self.refresh)
        self.primary_button.clicked.connect(self._primary)
        self.import_button.clicked.connect(self._import)
        bridge.event.connect(self._on_event)
        self._select_current()
        self.refresh()

    @property
    def selected_spec(self) -> ModelSpec:
        return manager.get_model(self.model_box.currentData())

    def _select_current(self) -> None:
        current = self.context.spec.model_id
        index = self.model_box.findData(current)
        if index >= 0:
            self.model_box.setCurrentIndex(index)

    def facts(self) -> ModelFacts:
        """Observe the world; `model_state.plan` decides what that permits."""
        spec = self.selected_spec
        # `is_downloaded` is the authority on completeness everywhere else; missing
        # bytes only distinguish "nothing here" from "interrupted part-way".
        total = sum(file.size for file in spec.files)
        if manager.is_downloaded(spec):
            disk = DiskState.PRESENT
        elif manager.missing_bytes(spec) < total:
            disk = DiskState.PARTIAL
        else:
            disk = DiskState.ABSENT

        active = self.context.coordinator.active()
        return ModelFacts(
            model_id=spec.model_id,
            disk=disk,
            is_current=spec.model_id == self.context.spec.model_id,
            is_loaded=self.context.engine_loaded_model == spec.model_id,
            is_imported=imported.is_imported(spec.model_id),
            workload_running="、".join(item.value for item in active) if active else None,
            operation_running=self._thread is not None,
        )

    def refresh(self) -> None:
        spec = self.selected_spec
        state = plan(self.facts())
        size = sum(file.size for file in spec.files)
        # The path is reference material, not something to read while dictating, so it
        # goes on hover; the line itself stays short enough to never wrap.
        self.status.setText(f"{state.summary} · {_gib(size)} · revision {spec.revision[:8]}")
        self.status.setToolTip(f"{manager.model_dir(spec)}")
        self.model_box.setEnabled(state.chooser_enabled)
        # The accent colour is only worth anything while it is scarce. Reserve it for
        # the cases that genuinely block recognition — a model that is missing or is not
        # the current one. Loading is not one of them: pressing 开始识别 prepares the
        # engine by itself, so 加载到显存 is a convenience, and 释放显存 is maintenance.
        blocking = state.primary.intent in {Intent.DOWNLOAD_AND_USE, Intent.USE}
        set_role(self.primary_button, "primary" if blocking else "")
        for button, action in (
            (self.primary_button, state.primary),
            (self.import_button, state.import_model),
        ):
            button.setText(action.text)
            button.setEnabled(action.enabled)
            button.setToolTip(action.tip)

    def _primary(self) -> None:
        """Move the selected model one step towards being ready, or release it."""
        if self._thread is not None:
            return
        intent = plan(self.facts()).primary.intent
        if intent is Intent.NONE:
            return
        if intent is Intent.RELEASE:
            self._run(lambda: self.context.unload_engine(), "正在释放显存…")
        elif intent is Intent.LOAD:
            self._run(self.context.load_engine, "正在加载模型到显存…")
        elif intent is Intent.USE:
            self._run(self._use_selected, "正在切换并加载模型…")
        else:
            self._confirm_download()

    def _use_selected(self) -> None:
        self.context.switch_model(self.selected_spec.model_id)
        self.context.load_engine()

    def _run(self, action: Callable[[], None], message: str) -> None:
        def operation(_progress: Callable[[str, int, int], None]) -> None:
            action()

        self.status.setText(message)
        self._start(operation)

    def _confirm_download(self) -> None:
        spec = self.selected_spec
        size = sum(file.size for file in spec.files)
        needed = manager.missing_bytes(spec)
        answer = QMessageBox.question(
            self,
            "确认下载模型",
            f"还需下载 {_gib(needed)}（完整模型 {_gib(size)}）到：\n"
            f"{manager.model_dir(spec)}\n\n"
            "仅下载 catalog 锁定的 revision，完成后会校验大小与 SHA-256，"
            "并设为当前模型。是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        def operation(progress: Callable[[str, int, int], None]) -> None:
            completed_by_file: dict[str, int] = {}

            def aggregate(name: str, completed: int, _total: int) -> None:
                completed_by_file[name] = completed
                progress("all", sum(completed_by_file.values()), size)

            self.context.download_model(spec.model_id, aggregate)
            # Downloading was only ever a means to using it.
            self.context.switch_model(spec.model_id)
            self.context.load_engine()

        self.status.setText(f"正在下载 {spec.model_id}…")
        self.progress.setValue(0)
        self.progress.show()
        self._start(operation)

    def _import(self) -> None:
        if self._thread is not None:
            return
        model_file, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件（.gguf）", str(imported.store_root() or Path.home()),
            "GGUF 模型 (*.gguf)"
        )
        if not model_file:
            return
        mmproj_file, _ = QFileDialog.getOpenFileName(
            self,
            "选择配套的 mmproj 文件（音频模型必需，可取消）",
            str(Path(model_file).parent),
            "GGUF 模型 (*.gguf)",
        )
        request = imported.ImportRequest(
            model_path=Path(model_file),
            mmproj_path=Path(mmproj_file) if mmproj_file else None,
        )

        def operation(progress: Callable[[str, int, int], None]) -> None:
            spec = imported.import_model(request, progress)
            self._imported_id = spec.model_id

        self._imported_id = None
        self.status.setText("正在校验并复制模型文件…")
        self._start(operation)



    def _reload_models(self) -> None:
        """Rebuild the list after an import or a removal changed what exists."""
        wanted = self.model_box.currentData()
        self.model_box.blockSignals(True)
        self.model_box.clear()
        # ASR only. The catalog also carries the refinement model, which is a
        # different role with its own server and must never appear in the chooser that
        # decides what transcribes audio.
        for spec in manager.models_of_kind(manager.ModelKind.ASR):
            self.model_box.addItem(_model_label(spec), spec.model_id)
        index = self.model_box.findData(wanted)
        self.model_box.setCurrentIndex(index if index >= 0 else 0)
        self.model_box.blockSignals(False)


    def _start(self, operation: Callable[[Callable[[str, int, int], None]], None]) -> None:
        self._thread = _ModelActionThread(operation)
        self._thread.progressed.connect(self.progress.setValue)
        self._thread.completed.connect(self._done)
        self._thread.failed.connect(self._failed)
        self._thread.finished.connect(self._finished)
        self._thread.start()
        self.refresh()

    def _done(self) -> None:
        if getattr(self, "_imported_id", None):
            self._reload_models()
            index = self.model_box.findData(self._imported_id)
            if index >= 0:
                self.model_box.setCurrentIndex(index)
            self._imported_id = None
        elif getattr(self, "_removed_id", None):
            self._reload_models()
            self._removed_id = None
        self.status.setText("操作完成。")

    def _failed(self, message: str) -> None:
        self.status.setText(f"模型操作失败：{message}")

    def _finished(self) -> None:
        self._thread = None
        self.progress.hide()
        self._select_current()
        self.refresh()

    def _on_event(self, event: object) -> None:
        if isinstance(event, (ModelChanged, ModelDownloaded)):
            self._select_current()
            self.refresh()

    def busy_reason(self) -> str | None:
        if self._thread is not None and self._thread.isRunning():
            return "模型正在下载或切换"
        return None
