"""What the two backends are doing, for a desktop that no longer runs the models.

The model panel this replaces offered 「加载到显存」 and 「释放显存」. With recognition
on the Orin and refinement on the local card, both labels describe something that is not
happening: there is no model in this machine's VRAM to load or release, and pressing the
button reaches a server on another host. Showing which backends answer, and what each is
holding, is the thing that is actually true.

Status is polled rather than pushed. A node can go away without telling anyone — the
board reboots, the network drops — and a panel that only updates when asked would show a
healthy tick over a machine that stopped answering ten minutes ago.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import httpx
from PySide6.QtCore import QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from localasr.context import AppContext
from localasr.frontends.desktop.theme import set_tone
from localasr.registry import imported
from localasr.registry.manager import ModelKind

POLL_SECONDS = 20
"""Often enough that a dead node is noticed before the user hits record, rare enough that
it costs nothing. Both probes are sub-millisecond on a LAN."""

PROBE_TIMEOUT = 3.0


@dataclass(frozen=True, slots=True)
class BackendStatus:
    label: str
    detail: str
    tone: str


def _probe_asr(context: AppContext) -> BackendStatus:
    url = context.node_url
    if not url:
        return BackendStatus("本机", "未配置识别节点，使用本机引擎", "neutral")
    try:
        with httpx.Client(timeout=PROBE_TIMEOUT) as client:
            headers = {}
            token = context.settings.node_token
            if token:
                headers["Authorization"] = f"Bearer {token}"
            if client.get(f"{url}/health").status_code != 200:
                return BackendStatus("无响应", f"{url} 未回应健康检查", "danger")
            ready = client.get(f"{url}/readyz", headers=headers)
        loaded = ready.json().get("loaded", {}) if ready.status_code in (200, 503) else {}
        model = loaded.get("asr")
        if model:
            return BackendStatus("就绪", f"{url} · {model} 已驻留", "success")
        # 503 here is not a fault: the node loads on demand and simply has not been
        # asked yet. Saying "will load on first use" beats a red light on a good node.
        return BackendStatus("待命", f"{url} · 首次识别时加载模型", "active")
    except httpx.HTTPError as exc:
        return BackendStatus("不可达", f"{url}（{type(exc).__name__}）", "danger")


def _probe_refiner(context: AppContext) -> BackendStatus:
    if context.refiner_managed:
        # Ours to start and stop, so "not running" is a state rather than a fault — it
        # is what the machine looks like before a session and after one.
        if not context.refiner_loaded:
            return BackendStatus(
                "未启动", f"{context.refiner_spec.model_id} · 点「启动」加载到显存", "neutral"
            )
        return BackendStatus(
            "就绪", f"{context.refiner_base_url} · {context.refiner_spec.model_id}", "success"
        )

    url = context.settings.refiner_url
    if not url:
        if context.node_url:
            return BackendStatus("经识别节点", "整理请求走识别节点", "neutral")
        return BackendStatus(
            "未配置", "设置 refiner_url 或 refiner_model_id 后才能整理文本", "warning"
        )
    try:
        with httpx.Client(timeout=PROBE_TIMEOUT) as client:
            if client.get(f"{url}/health").status_code != 200:
                return BackendStatus("无响应", f"{url} 未回应健康检查", "danger")
        return BackendStatus("就绪", f"{url} · {context.settings.refiner_model}", "success")
    except httpx.HTTPError as exc:
        return BackendStatus("不可达", f"{url}（{type(exc).__name__}）", "danger")


class _NodeActionThread(QThread):
    """Load or release on the recognition node. A cold load is tens of seconds."""

    done = Signal(str)

    def __init__(self, url: str, token: str | None, action: str) -> None:
        super().__init__()
        self._url = url
        self._token = token
        self._action = action

    def run(self) -> None:
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        verb = "加载" if self._action == "load" else "卸载"
        try:
            with httpx.Client(timeout=180.0) as client:
                response = client.post(
                    f"{self._url}/api/v1/models/{self._action}",
                    json={"kind": "asr"},
                    headers=headers,
                )
                body = response.json()
                if response.status_code == 409:
                    self.done.emit(f"{verb}失败：{body.get('error', '节点拒绝')}")
                    return
                response.raise_for_status()
        except httpx.HTTPError as exc:
            self.done.emit(f"{verb}失败：{exc}")
            return
        free = (body.get("memory_mb") or {}).get("available")
        suffix = f"，节点可用 {free} MiB" if free else ""
        if self._action == "load":
            loaded = ", ".join(body.get("loaded", {}).values()) or "无"
            self.done.emit(f"识别模型已就绪：{loaded}{suffix}")
        else:
            released = body.get("released") or []
            self.done.emit(
                f"已卸载 {'、'.join(released)}{suffix}" if released
                else "节点上没有已驻留的模型"
            )


class _RefinerActionThread(QThread):
    """Start or stop the refinement server this machine owns."""

    done = Signal(str)

    def __init__(self, context: AppContext, action: str) -> None:
        super().__init__()
        self._context = context
        self._action = action

    def run(self) -> None:
        try:
            if self._action == "load":
                self._context.start_refiner()
                self.done.emit(f"整理模型已就绪：{self._context.refiner_spec.model_id}")
            else:
                self._context.stop_refiner()
                self.done.emit("整理模型已卸载，显存已释放")
        except Exception as exc:  # noqa: BLE001 - report, never take the host down
            verb = "启动" if self._action == "load" else "停止"
            self.done.emit(f"整理服务{verb}失败：{exc}")


class _ImportThread(QThread):
    """Copying a multi-gigabyte file, and hashing it afterwards."""

    done = Signal(str)

    def __init__(self, request: imported.ImportRequest) -> None:
        super().__init__()
        self._request = request

    def run(self) -> None:
        try:
            spec = imported.import_model(self._request)
        except Exception as exc:  # noqa: BLE001 - report, never take the host down
            self.done.emit(f"导入失败：{exc}")
            return
        self.done.emit(f"已导入 {spec.model_id}；重启整理服务后可用")


class _ProbeThread(QThread):
    """Probes off the GUI thread: a node that has gone away answers by timing out, and
    three seconds of frozen interface is worse than a stale label."""

    done = Signal(object, object)

    def __init__(self, context: AppContext) -> None:
        super().__init__()
        self._context = context

    def run(self) -> None:
        try:
            self.done.emit(_probe_asr(self._context), _probe_refiner(self._context))
        except Exception as exc:  # noqa: BLE001 - a status panel must not kill the host
            failed = BackendStatus("检查失败", str(exc), "danger")
            self.done.emit(failed, failed)


class BackendPanel(QFrame):
    """Live status of the recognition node and the refinement backend."""

    summary = Signal(str)
    """One line describing both backends, for whoever is showing this panel folded."""

    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.context = context
        self._probe: _ProbeThread | None = None
        self._release_thread: QThread | None = None
        self._import_thread: _ImportThread | None = None
        self.setObjectName("card")

        layout = QGridLayout(self)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(6)

        self.asr_pill = QLabel("检查中")
        self.asr_pill.setObjectName("statusPill")
        self.asr_detail = QLabel("正在检查识别节点…")
        self.refiner_pill = QLabel("检查中")
        self.refiner_pill.setObjectName("statusPill")
        self.refiner_detail = QLabel("正在检查整理服务…")
        for detail in (self.asr_detail, self.refiner_detail):
            detail.setProperty("muted", True)
            detail.setWordWrap(False)
            # A URL plus a model name is wider than the window's declared minimum, and a
            # status line must never be the thing that decides how narrow a window may
            # be. It elides; the full text is on hover.
            detail.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

        self.refresh_button = QPushButton("重新检查")
        self.refresh_button.clicked.connect(self.refresh)
        # Unloading acts on the recognition node, because that is the only model this
        # machine can reach an API for. The refiner is a plain llama-server with no
        # release endpoint — stopping it is a service operation, not a button here.
        # Two buttons per backend, because the point is holding both models across a
        # whole session: warm them before starting, release them when finished. Leaving
        # it to the first request means paying a cold load after the user has already
        # begun to speak.
        self.asr_load_button = QPushButton("启动")
        self.asr_load_button.setToolTip("让识别节点现在就加载模型，避免第一句话等几十秒")
        self.asr_load_button.clicked.connect(lambda: self._node_action("load"))
        self.release_button = QPushButton("卸载")
        self.release_button.setToolTip(
            "让识别节点释放已驻留的模型，把内存还给那台机器。\n下一次识别会重新加载。"
        )
        self.release_button.clicked.connect(lambda: self._node_action("release"))

        self.refiner_load_button = QPushButton("启动")
        self.refiner_unload_button = QPushButton("卸载")
        self.refiner_load_button.clicked.connect(lambda: self._refiner_action("load"))
        self.refiner_unload_button.clicked.connect(lambda: self._refiner_action("release"))
        if not context.refiner_managed:
            for button in (self.refiner_load_button, self.refiner_unload_button):
                button.setEnabled(False)
                button.setToolTip(
                    "整理服务由外部管理（config.toml 设置了 refiner_url），"
                    "启停请用 systemctl 或启动它的方式。"
                )
        # Import lands in this machine's catalog, which is where the refiner reads from.
        self.import_button = QPushButton("导入模型…")
        self.import_button.setToolTip(
            "把本地的 .gguf 复制进模型目录并登记。\n"
            "识别模型需要在识别节点上导入，这里导入的模型供本机整理服务使用。"
        )
        self.import_button.clicked.connect(self._import)

        self.action_status = QLabel()
        self.action_status.setProperty("muted", True)
        self.action_status.setWordWrap(False)
        self.action_status.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self.action_status.hide()

        layout.addWidget(QLabel("语音识别"), 0, 0)
        layout.addWidget(self.asr_pill, 0, 1)
        layout.addWidget(self.asr_detail, 0, 2)
        layout.addWidget(self.asr_load_button, 0, 3)
        layout.addWidget(self.release_button, 0, 4)
        layout.addWidget(QLabel("文本整理"), 1, 0)
        layout.addWidget(self.refiner_pill, 1, 1)
        layout.addWidget(self.refiner_detail, 1, 2)
        layout.addWidget(self.refiner_load_button, 1, 3)
        layout.addWidget(self.refiner_unload_button, 1, 4)
        layout.addWidget(self.action_status, 2, 0, 1, 3)
        layout.addWidget(self.refresh_button, 2, 3)
        layout.addWidget(self.import_button, 2, 4)
        layout.setColumnStretch(2, 1)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(POLL_SECONDS * 1000)
        self.refresh()

    @property
    def probing(self) -> bool:
        return self._probe is not None and self._probe.isRunning()

    @property
    def busy(self) -> bool:
        return any(t is not None and t.isRunning() for t in (self._release, self._import))

    def _node_action(self, action: str) -> None:
        if self._release_thread is not None or not self.context.node_url:
            return
        if action == "release" and self.context.coordinator.active():
            # Evicting mid-recording fails the very request that is using the model.
            self._report("请先停止正在进行的识别或会议。")
            return
        self._set_actions_enabled(False)
        self._report("正在加载识别模型，首次约需 25–40 秒…" if action == "load"
                     else "正在请求节点卸载模型…")
        self._release_thread = _NodeActionThread(
            self.context.node_url, self.context.settings.node_token, action
        )
        self._release_thread.done.connect(self._report)
        self._release_thread.finished.connect(self._action_finished)
        self._release_thread.start()

    def _refiner_action(self, action: str) -> None:
        if self._release_thread is not None or not self.context.refiner_managed:
            return
        if action == "release" and self.context.coordinator.active():
            self._report("请先停止正在进行的识别或会议。")
            return
        self._set_actions_enabled(False)
        self._report("正在启动整理服务…" if action == "load" else "正在释放整理模型…")
        self._release_thread = _RefinerActionThread(self.context, action)
        self._release_thread.done.connect(self._report)
        self._release_thread.finished.connect(self._action_finished)
        self._release_thread.start()

    def _import(self) -> None:
        if self._import_thread is not None:
            return
        # Opened where the weights actually are. A store path is several levels deep and
        # partly hashes, so starting at $HOME made the common case the longest one.
        start = imported.store_root() or Path.home()
        model_file, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件（.gguf）", str(start), "GGUF 模型 (*.gguf)"
        )
        if not model_file:
            return
        # The role decides which list the model joins and how it is launched, and it
        # cannot be guessed from the filename. Every import used to become an ASR model,
        # so a refiner could be imported and then never appear as one.
        role = QMessageBox(self)
        role.setWindowTitle("模型角色")
        role.setText("这个模型用于哪一项？")
        asr_choice = role.addButton("语音识别", QMessageBox.ButtonRole.AcceptRole)
        llm_choice = role.addButton("文本整理", QMessageBox.ButtonRole.AcceptRole)
        role.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        role.exec()
        if role.clickedButton() not in (asr_choice, llm_choice):
            return
        kind = ModelKind.ASR if role.clickedButton() is asr_choice else ModelKind.LLM

        mmproj_file = ""
        if kind is ModelKind.ASR:
            # Only asked for where it is needed. Refinement never sees audio, and the
            # dialog used to appear for it anyway with "整理模型不需要" in the title.
            mmproj_file, _ = QFileDialog.getOpenFileName(
                self,
                "选择配套的 mmproj 文件（音频模型必需，可取消）",
                str(Path(model_file).parent),
                "GGUF 模型 (*.gguf)",
            )

        # Linking is offered rather than assumed: a copy keeps working after the user
        # tidies up their downloads. But a file inside another tool's model store is not
        # a download waiting to be tidied — that tool is what keeps it — so there the
        # default flips and the reason is named instead of hypothesised.
        owner = imported.in_store(Path(model_file))
        link = (
            QMessageBox.question(
                self,
                "复制还是链接？",
                f"「{Path(model_file).name}」约 "
                f"{Path(model_file).stat().st_size / 1024**3:.1f} GB。\n\n"
                "链接不占额外磁盘，但原文件被移动或删除后模型将无法加载。\n"
                + (
                    f"这个文件在 {owner} 的模型库里，由它管理，建议链接。"
                    if owner
                    else "如果这个文件由别的工具管理（例如 LM Studio），链接更合适。"
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes if owner else QMessageBox.StandardButton.No,
            )
            == QMessageBox.StandardButton.Yes
        )

        self._set_actions_enabled(False)
        self._report("正在链接并校验模型文件…" if link else "正在复制并校验模型文件…")
        self._import_thread = _ImportThread(
            imported.ImportRequest(
                model_path=Path(model_file),
                mmproj_path=Path(mmproj_file) if mmproj_file else None,
                kind=kind,
                link=link,
            )
        )
        self._import_thread.done.connect(self._report)
        self._import_thread.finished.connect(self._action_finished)
        self._import_thread.start()

    def _report(self, message: str) -> None:
        self.action_status.setText(message)
        self.action_status.setToolTip(message)
        self.action_status.show()

    def _set_actions_enabled(self, enabled: bool) -> None:
        for button in (self.asr_load_button, self.release_button, self.import_button):
            button.setEnabled(enabled)
        if self.context.refiner_managed:
            self.refiner_load_button.setEnabled(enabled)
            self.refiner_unload_button.setEnabled(enabled)

    def _action_finished(self) -> None:
        self._release_thread = None
        self._import_thread = None
        self._set_actions_enabled(True)
        # Deferred, not immediate. A probe started here races the one already in flight
        # from the periodic timer, and `refresh` declines while that is running — so the
        # panel would keep showing the state from before the release.
        QTimer.singleShot(200, self.refresh)

    def refresh(self) -> None:
        if self._probe is not None:
            return
        self.refresh_button.setEnabled(False)
        self._probe = _ProbeThread(self.context)
        self._probe.done.connect(self._show)
        self._probe.finished.connect(self._finished)
        self._probe.start()

    def _show(self, asr: BackendStatus, refiner: BackendStatus) -> None:
        # Broadcast for a collapsed header: folding setup away must not fold away the
        # answer to "is it working".
        self.summary.emit(f"识别 {asr.label} · 整理 {refiner.label}")
        for pill, detail, status in (
            (self.asr_pill, self.asr_detail, asr),
            (self.refiner_pill, self.refiner_detail, refiner),
        ):
            pill.setText(status.label)
            set_tone(pill, status.tone)
            detail.setText(status.detail)
            detail.setToolTip(status.detail)

    def _finished(self) -> None:
        self._probe = None
        self.refresh_button.setEnabled(True)

    def wait_for_probe(self, timeout: int = 4000) -> None:
        """Let a shutting-down host drain the panel's threads.

        A QThread destroyed while running takes the process with it, and the periodic
        probe means one is almost always in flight.
        """
        for thread in (self._probe, self._release_thread, self._import_thread):
            if thread is not None and thread.isRunning():
                thread.wait(timeout)

    def closeEvent(self, event) -> None:  # noqa: ANN001, N802 - Qt API
        self._timer.stop()
        self.wait_for_probe()
        super().closeEvent(event)


def setup_panel(context: AppContext, bridge, parent: QWidget | None = None) -> QWidget:  # noqa: ANN001
    """The panel that belongs at the top of a window, given where the models live.

    With a recognition node configured there is nothing in this machine's VRAM to load
    or release, so offering those buttons would describe work happening on another host.
    Without one, the local model chooser is exactly right. All three windows make the
    same choice, so they make it here.
    """
    from localasr.frontends.desktop.model_panel import ModelPanel

    if context.node_url:
        return BackendPanel(context, parent)
    return ModelPanel(context, bridge, parent)
