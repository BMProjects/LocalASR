"""Choosing and testing the input device.

Dictation and meetings need the same thing — pick a microphone, confirm it is actually
picking something up, remember the choice — so it lives in one widget rather than being
written twice and drifting.

Testing before recording matters more than it looks: an input that is muted, or a
Bluetooth headset that switched profiles, produces a perfectly successful recording of
nothing, and the failure only surfaces as an empty transcript minutes later.
"""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from localasr.capture.microphone import CaptureError, InputLevel, list_devices, probe_input
from localasr.capture.naming import device_labels
from localasr.context import AppContext

PROBE_SECONDS = 1.0


class _ProbeThread(QThread):
    measured = Signal(object)
    failed = Signal(str)

    def __init__(self, device: str | int | None) -> None:
        super().__init__()
        self._device = device

    def run(self) -> None:
        try:
            self.measured.emit(probe_input(self._device, duration=PROBE_SECONDS))
        except (CaptureError, ValueError) as exc:
            self.failed.emit(str(exc))


class DevicePanel(QFrame):
    """Input chooser with a one-second level test. Selection is persisted on change."""

    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.context = context
        self._probe: _ProbeThread | None = None
        self.setObjectName("card")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(6)
        title = QLabel("输入设备")
        title.setObjectName("sectionTitle")

        self.device_box = QComboBox()
        self.device_box.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.device_box.setMinimumContentsLength(18)
        self.refresh_button = QPushButton("刷新")
        self.test_button = QPushButton("测试输入")
        # The title shares the row with the chooser. Picking a microphone happens once
        # per session at most; a full-width section header for it is space the
        # transcript needs more.
        controls = QHBoxLayout()
        controls.setSpacing(8)
        controls.addWidget(title)
        controls.addWidget(self.device_box, 1)
        controls.addWidget(self.refresh_button)
        controls.addWidget(self.test_button)

        self.status = QLabel("选择麦克风后，可先测试一秒确认是否有输入。")
        self.status.setWordWrap(False)
        self.status.setProperty("muted", True)
        self.status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

        layout.addLayout(controls)
        layout.addWidget(self.status)

        self.refresh_button.clicked.connect(self.refresh)
        self.test_button.clicked.connect(self._test)
        self.refresh()
        self.device_box.currentIndexChanged.connect(self._changed)

    @property
    def selected(self) -> str | int | None:
        return self.device_box.currentData()

    def set_enabled(self, enabled: bool) -> None:
        """Lock the panel while a session owns the device."""
        busy = self._probe is not None
        for widget in (self.device_box, self.refresh_button, self.test_button):
            widget.setEnabled(enabled and not busy)

    def refresh(self) -> None:
        selected = self.context.settings.input_device
        self.device_box.blockSignals(True)
        self.device_box.clear()
        self.device_box.addItem("系统默认输入设备", None)
        try:
            devices = [device for device in list_devices() if not device.is_monitor]
        except CaptureError as exc:
            self.status.setText(f"无法读取音频设备：{exc}")
            self.device_box.blockSignals(False)
            return

        index = 0
        found = selected is None
        # Shown under the names the system settings use; the raw PortAudio PCM names
        # ("sof-hda-dsp: - (hw:0,6)") match nothing the user has seen elsewhere.
        for device, label in zip(devices, device_labels(devices), strict=True):
            # Persist the PortAudio name rather than its numeric index: indices can move
            # after plugging in a USB/Bluetooth device, while hw:* names stay stable.
            self.device_box.addItem(label, device.name)
            if selected in {str(device.index), device.name} or (
                isinstance(selected, str) and selected in device.name
            ):
                index = self.device_box.count() - 1
                found = True
        if selected is not None and not found:
            self.device_box.addItem(f"已保存但当前不可用：{selected}", selected)
            index = self.device_box.count() - 1
        self.device_box.setCurrentIndex(index)
        self.device_box.blockSignals(False)
        self.status.setText(f"发现 {len(devices)} 个输入端点；选择后自动保存。")

    def _changed(self) -> None:
        self.context.settings.input_device = self.selected
        try:
            path = self.context.settings.save()
        except OSError as exc:
            self.status.setText(f"设备已用于本次运行，但保存失败：{exc}")
            return
        self.status.setText(f"已选择 {self.device_box.currentText()}；配置保存到 {path}")

    def _test(self) -> None:
        if self._probe is not None or self.context.coordinator.active():
            self.status.setText("请先停止正在进行的录音、会议或字幕任务。")
            return
        self._probe = _ProbeThread(self.selected)
        self._probe.measured.connect(self._measured)
        self._probe.failed.connect(lambda message: self.status.setText(f"测试失败：{message}"))
        self._probe.finished.connect(self._finished)
        self._probe.start()
        self.status.setText("正在采集 1 秒，请对着麦克风说话…")
        self.set_enabled(False)

    def _measured(self, level: InputLevel) -> None:
        if level.dbfs > -50:
            verdict = "输入正常"
        elif level.dbfs > -65:
            verdict = "信号较弱"
        else:
            verdict = "几乎没有输入"
        overflow = "，采集发生溢出" if level.overflowed else ""
        self.status.setText(f"{verdict}：{level.dbfs:.1f} dBFS，峰值 {level.peak:.4f}{overflow}")

    def _finished(self) -> None:
        self._probe = None
        self.set_enabled(True)

    @property
    def probing(self) -> bool:
        """True while the one-second level test holds the device."""
        return self._probe is not None and self._probe.isRunning()

    def wait_for_probe(self, timeout: int = 3000) -> None:
        """Let a shutting-down host drain the probe thread."""
        if self._probe is not None and self._probe.isRunning():
            self._probe.wait(timeout)
