"""Desktop host wiring.

Rendered offscreen, so this checks the parts that can be wrong without a display:
that events reach widgets on the Qt thread, that a second launch does not become a
second process, and that the windows construct at all.
"""

import os
import threading
import time
from datetime import datetime

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6", reason="desktop frontend requires the [gui] extra")

from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtGui import QShortcut  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from localasr.apps.bus import EventBus  # noqa: E402
from localasr.apps.dictation import (  # noqa: E402
    RECORDING,
    DictationController,
    DictationOptions,  # noqa: E402
)
from localasr.apps.events import (  # noqa: E402
    DictationStateChanged,
    FinalTranscript,
    SessionStopped,
    TextEmitted,
)
from localasr.capture.microphone import DeviceInfo  # noqa: E402
from localasr.context import AppContext  # noqa: E402
from localasr.core.types import Segment, Span  # noqa: E402
from localasr.frontends.desktop import theme  # noqa: E402
from localasr.frontends.desktop.dictation_window import DictationWindow  # noqa: E402
from localasr.frontends.desktop.event_bridge import QtEventBridge  # noqa: E402
from localasr.frontends.desktop.single_instance import (  # noqa: E402
    AlreadyRunning,
    SingleInstance,
    activate,
)
from localasr.frontends.desktop.theme import apply_theme  # noqa: E402
from localasr.registry import manager  # noqa: E402


def _contrast(foreground: str, background: str) -> float:
    """WCAG relative-luminance contrast ratio between two hex colours."""

    def luminance(value: str) -> float:
        channels = [int(value[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    light, dark = sorted((luminance(foreground), luminance(background)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    apply_theme(app)
    yield app


def _pump(times: int = 20) -> None:
    for _ in range(times):
        QCoreApplication.processEvents()


def test_events_published_from_a_worker_thread_reach_the_qt_thread(qt_app):
    """Widgets may only be touched from the GUI thread; the bridge is what makes a
    background publish safe."""
    bus = EventBus()
    bridge = QtEventBridge(bus)
    received = []
    threads = []

    def record(event):
        received.append(event)
        threads.append(threading.get_ident())

    bridge.event.connect(record)

    def publish() -> None:
        bus.publish(SessionStopped(utterances=7))

    worker = threading.Thread(target=publish)
    worker.start()
    worker.join()
    _pump()

    assert [e.utterances for e in received] == [7]
    assert threads[0] == threading.get_ident()
    bridge.stop()


def test_bridge_reports_dropped_events(qt_app):
    bus = EventBus(capacity=1)
    bridge = QtEventBridge(bus)
    dropped = []
    bridge.dropped.connect(dropped.append)

    for index in range(4):
        bus.publish(SessionStopped(utterances=index))
    _pump()

    assert dropped and dropped[-1] == 3
    bridge.stop()


def test_a_second_instance_cannot_take_the_address():
    """Three windows must not become three processes, each with its own llama-server."""
    address = "\0localasr-test-instance"
    first = SingleInstance(address)
    first.acquire(lambda _message: None)
    try:
        with pytest.raises(AlreadyRunning):
            SingleInstance(address).acquire(lambda _message: None)
    finally:
        first.release()


def test_activation_hands_the_request_to_the_running_instance():
    address = "\0localasr-test-activate"
    received = threading.Event()
    seen = []

    instance = SingleInstance(address)
    instance.acquire(lambda message: (seen.append(message), received.set()))
    try:
        assert activate("meeting", address)
        assert received.wait(2.0)
        assert seen == ["meeting"]
    finally:
        instance.release()


def test_activation_returns_false_when_nothing_is_running():
    assert not activate("subtitle", "\0localasr-test-absent")


def test_windows_construct_and_share_one_context(qt_app):
    from localasr.frontends.desktop.meeting_window import MeetingWindow
    from localasr.frontends.desktop.subtitle_window import SubtitleWindow

    context = AppContext()
    bridge = QtEventBridge(context.bus)
    subtitle = SubtitleWindow(context, bridge)
    meeting = MeetingWindow(context, bridge)
    dictation = DictationWindow(context, bridge, DictationController(context))

    assert subtitle.context is meeting.context is dictation.context
    assert "音源标记" in meeting.notice.text(), "the source-label caveat must be visible"
    assert dictation.toggle_button.text() == "开始识别"
    assert subtitle.model_panel.model_box.count() == 2
    assert meeting.model_panel.model_box.count() == 2
    assert dictation.model_panel.model_box.count() == 2
    bridge.stop()


def test_model_selection_never_downloads_or_switches_implicitly(qt_app, monkeypatch):
    from localasr.frontends.desktop.model_panel import ModelPanel

    calls = []
    specs = manager.list_models()
    context = AppContext()
    bridge = QtEventBridge(context.bus)
    panel = ModelPanel(context, bridge)
    monkeypatch.setattr(context, "download_model", lambda *_args: calls.append("download"))
    monkeypatch.setattr(context, "switch_model", lambda *_args: calls.append("switch"))

    other = next(spec for spec in specs if spec.model_id != context.spec.model_id)
    panel.model_box.setCurrentIndex(panel.model_box.findData(other.model_id))

    assert calls == []
    assert context.spec.model_id != other.model_id
    bridge.stop()


def test_downloaded_model_can_be_selected_and_persisted(qt_app, tmp_path, monkeypatch):
    from localasr.frontends.desktop.model_panel import ModelPanel

    config = tmp_path / "config.toml"
    monkeypatch.setenv("LOCALASR_CONFIG", str(config))
    monkeypatch.setattr(manager, "is_downloaded", lambda _spec: True)
    context = AppContext()
    bridge = QtEventBridge(context.bus)
    panel = ModelPanel(context, bridge)
    other = next(spec for spec in manager.list_models() if spec.model_id != context.spec.model_id)
    panel.model_box.setCurrentIndex(panel.model_box.findData(other.model_id))

    # Selecting and loading are one action now; the load itself is stubbed so the test
    # needs no GPU.
    monkeypatch.setattr(context, "load_engine", lambda: None)
    assert panel.primary_button.text() == "使用此模型"
    panel._primary()
    worker = panel._thread
    assert worker is not None
    assert worker.wait(2000)
    _pump()

    assert context.settings.model_id == other.model_id
    assert AppContext().settings.model_id == other.model_id
    assert panel.primary_button.text() in {"加载到显存", "释放显存"}
    bridge.stop()


def test_model_download_requires_an_explicit_confirmed_click(qt_app, monkeypatch):
    from localasr.frontends.desktop.model_panel import ModelPanel

    downloaded = set()
    calls = []
    monkeypatch.setattr(manager, "is_downloaded", lambda spec: spec.model_id in downloaded)
    # Honest stub: a model that was never fetched is missing all of its bytes, which
    # is what distinguishes 「下载并使用」 from 「继续下载并使用」.
    monkeypatch.setattr(
        manager,
        "missing_bytes",
        lambda spec: 0 if spec.model_id in downloaded else sum(f.size for f in spec.files),
    )
    monkeypatch.setattr(
        "localasr.frontends.desktop.model_panel.QMessageBox.question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )
    context = AppContext()

    def download(model_id, progress):
        calls.append(model_id)
        progress("model", 1024, 1024)
        downloaded.add(model_id)

    monkeypatch.setattr(AppContext, "load_engine", lambda _self: None)

    monkeypatch.setattr(context, "download_model", download)
    bridge = QtEventBridge(context.bus)
    panel = ModelPanel(context, bridge)
    selected = panel.selected_spec

    assert calls == [], "constructing or selecting the model must never download it"
    assert panel.primary_button.text() == "下载并使用", "the label must say it downloads"
    panel._primary()
    worker = panel._thread
    assert worker is not None
    assert worker.wait(2000)
    _pump()

    assert calls == [selected.model_id]
    bridge.stop()


def test_subtitle_queue_defaults_to_safe_non_overwrite(qt_app, tmp_path):
    from localasr.frontends.desktop.subtitle_window import SubtitleWindow

    media = tmp_path / "sample.wav"
    media.touch()
    context = AppContext()
    bridge = QtEventBridge(context.bus)
    window = SubtitleWindow(context, bridge)

    window._add([media])

    assert window.list.count() == 1
    assert window.start_button.isEnabled()
    assert not window.overwrite_box.isChecked()
    assert window.stack.currentWidget() is window.list

    window.list.selectAll()
    window._remove_selected()
    assert window.list.count() == 0
    assert window.stack.currentWidget() is window.empty
    bridge.stop()


def test_dictation_window_exposes_state_and_preserves_last_text(qt_app):
    context = AppContext()
    bridge = QtEventBridge(context.bus)
    window = DictationWindow(context, bridge, DictationController(context))

    assert not window.auto_deliver.isChecked()
    window._on_event(DictationStateChanged(RECORDING))
    assert window.state_badge.text() == "正在聆听"
    assert window.toggle_button.text() == "暂停识别"
    assert window.toggle_button.isEnabled()

    # TextEmitted must not overwrite the box: by then it already holds every sentence,
    # possibly with the user's corrections.
    window.last_text.setPlainText("我改过的文字")
    window._on_event(TextEmitted("测试文字", "clipboard"))
    assert window.last_text.toPlainText() == "我改过的文字"
    assert window.copy_button.isEnabled()
    assert "Ctrl+V" in window.status_message.text()
    bridge.stop()


def test_dictation_device_selection_is_visible_and_persisted(qt_app, tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    monkeypatch.setenv("LOCALASR_CONFIG", str(config))
    monkeypatch.setattr(
        "localasr.frontends.desktop.device_panel.list_devices",
        lambda: [
            DeviceInfo(4, "sof-hda-dsp (hw:0,6)", 2, 48_000.0, False),
            DeviceInfo(5, "sof-hda-dsp (hw:0,7)", 2, 16_000.0, False),
        ],
    )
    context = AppContext()
    bridge = QtEventBridge(context.bus)
    window = DictationWindow(context, bridge, DictationController(context))

    assert window.result_title.text() == "实时识别结果"
    assert window.device_panel.device_box.count() == 3
    window.device_panel.device_box.setCurrentIndex(2)

    assert context.settings.input_device == "sof-hda-dsp (hw:0,7)"
    assert "16 kHz" in window.device_panel.device_box.currentText()
    assert config.is_file()
    assert AppContext().settings.input_device == "sof-hda-dsp (hw:0,7)"
    bridge.stop()


def test_dictation_window_shows_each_settled_utterance_while_recording(qt_app):
    first = Segment(span=Span(0.0, 1.0), text="第一句。", source="mic", utterance_id="a")
    second = Segment(span=Span(1.0, 2.0), text="第二句。", source="mic", utterance_id="b")

    class RecordingController:
        recording = True
        current_text = "第一句。第二句。"
        state = RECORDING
        segments = (first, second)
        options = DictationOptions()

    context = AppContext()
    bridge = QtEventBridge(context.bus)
    window = DictationWindow(context, bridge, RecordingController())

    window._on_event(FinalTranscript(second))

    assert window.last_text.toPlainText() == "第一句。第二句。"
    assert window.copy_button.isEnabled()
    bridge.stop()


def test_stopping_visible_dictation_keeps_window_and_shows_result(qt_app):
    class RecordingController:
        recording = True
        current_text = ""
        state = RECORDING
        segments = ()
        options = DictationOptions()
        deliveries = []

        def finish(self, *, deliver=True):
            self.deliveries.append(deliver)
            self.recording = False
            # The real controller settles a segment; the view renders those, not the
            # returned string.
            self.segments = (Segment(span=Span(0.0, 1.0), text="识别完成"),)
            self.current_text = "识别完成"
            self.state = "idle"
            return self.current_text

    context = AppContext()
    bridge = QtEventBridge(context.bus)
    controller = RecordingController()
    window = DictationWindow(context, bridge, controller)
    window.show()
    _pump()

    window.toggle(deliver=False)
    worker = window._worker
    assert worker is not None
    assert worker.wait(1000)
    _pump()

    assert window.isVisible(), "stopping looked like the application crashed"
    assert window.last_text.toPlainText() == "识别完成"
    assert controller.deliveries == [False]
    assert window._worker is None
    bridge.stop()


def test_refining_becomes_available_once_recognition_has_finished(qt_app):
    """The 整理 button was permanently grey for anyone who used it as intended.

    Its enabled state depends on `_worker is None`, but the only thing that recomputed
    it was the text box changing — and the text arrives from `_operation_done`, while
    that worker is still alive. By the time the worker cleared, nothing asked again, so
    speech could never enable the button. Pasting text by hand could, which is why this
    looked like it worked when it was tested that way.
    """

    class RecordingController:
        recording = True
        current_text = ""
        state = RECORDING
        segments = ()
        options = DictationOptions()

        def finish(self, *, deliver=True):
            self.recording = False
            self.segments = (Segment(span=Span(0.0, 1.0), text="识别完成"),)
            self.current_text = "识别完成"
            self.state = "idle"
            return self.current_text

    context = AppContext()
    context.settings.node_url = "http://asr-node.local:8090"
    context.settings.refiner_model_id = "qwen3_5-4b-refiner-q4"
    assert context.can_refine
    bridge = QtEventBridge(context.bus)
    window = DictationWindow(context, bridge, RecordingController())

    window.toggle(deliver=False)
    assert window._worker is not None and window._worker.wait(1000)
    _pump()

    assert window.last_text.toPlainText() == "识别完成"
    assert window.refine_button.isEnabled(), "there is text and nothing is running"
    assert window.clear_button.isEnabled(), "the same stale-state bug hid this one too"
    bridge.stop()


def test_dictation_prepares_engine_before_opening_microphone(qt_app, monkeypatch):
    calls = []

    class IdleController:
        recording = False
        current_text = ""
        state = "idle"
        segments = ()
        options = DictationOptions()

        def __init__(self, context):
            self._context = context

        def start(self, *, force=False):
            # Mirrors the real controller: preparing is part of starting, so a caller
            # that only calls `start` still gets the engine loaded first.
            self._context.prepare_engine()
            calls.append(("start", force))
            self.recording = True
            self.state = RECORDING

    context = AppContext()
    monkeypatch.setattr(context, "prepare_engine", lambda: calls.append(("prepare", False)))
    bridge = QtEventBridge(context.bus)
    controller = IdleController(context)
    window = DictationWindow(context, bridge, controller)

    window.toggle(deliver=False)
    worker = window._worker
    assert worker is not None
    assert worker.wait(1000)
    _pump()

    assert calls == [("prepare", False), ("start", False)]
    assert controller.recording
    assert window._worker is None
    bridge.stop()


def test_dictation_does_not_open_microphone_when_model_is_missing(qt_app, monkeypatch):
    starts = []

    class IdleController:
        recording = False
        current_text = ""
        state = "idle"
        segments = ()
        options = DictationOptions()

        def __init__(self, context):
            self._context = context

        def start(self, *, force=False):
            self._context.prepare_engine()   # raises before any device is opened
            starts.append(force)

    context = AppContext()

    def missing_model():
        raise RuntimeError("识别模型尚未准备好")

    monkeypatch.setattr(context, "prepare_engine", missing_model)
    bridge = QtEventBridge(context.bus)
    controller = IdleController(context)
    window = DictationWindow(context, bridge, controller)

    window.toggle(deliver=False)
    worker = window._worker
    assert worker is not None
    assert worker.wait(1000)
    _pump()

    assert starts == []
    assert window._worker is None
    assert "模型尚未准备好" in window.status_message.text()
    assert window.state_badge.text() == "需要处理"
    bridge.stop()


def test_meeting_window_ignores_transcripts_not_owned_by_its_controller(qt_app, tmp_path):
    from localasr.apps.meeting import MeetingController, MeetingSession
    from localasr.frontends.desktop.meeting_window import MeetingWindow

    context = AppContext()
    bridge = QtEventBridge(context.bus)
    window = MeetingWindow(context, bridge)
    controller = MeetingController(context)
    controller.meeting = MeetingSession(tmp_path / "meeting.jsonl", datetime.now(), ("mic",))
    window._controller = controller
    unrelated = FinalTranscript(
        Segment(span=Span(0.0, 1.0), text="不属于会议", utterance_id="other")
    )
    owned_segment = Segment(
        span=Span(1.0, 2.0), text="会议内容", source="mic", utterance_id="meeting"
    )
    controller.meeting.segments.append(owned_segment)

    window._on_event(unrelated)
    window._on_event(FinalTranscript(owned_segment))

    assert "不属于会议" not in window.transcript.toPlainText()
    assert "会议内容" in window.transcript.toPlainText()
    bridge.stop()


@pytest.mark.parametrize("factory", ["subtitle", "meeting", "dictation"])
def test_a_window_never_opens_smaller_than_its_own_layout(qt_app, factory):
    """A default or minimum size below the layout minimum does not reflow the wrapped
    guidance labels — it clips them, so the window opens already hiding the text that
    explains what to do."""
    from localasr.apps.dictation import DictationController
    from localasr.frontends.desktop.dictation_window import DictationWindow
    from localasr.frontends.desktop.meeting_window import MeetingWindow
    from localasr.frontends.desktop.subtitle_window import SubtitleWindow

    context = AppContext()
    bridge = QtEventBridge(context.bus)
    window = {
        "subtitle": lambda: SubtitleWindow(context, bridge),
        "meeting": lambda: MeetingWindow(context, bridge),
        "dictation": lambda: DictationWindow(context, bridge, DictationController(context)),
    }[factory]()
    window.show()
    _pump()

    floor = window.layout().minimumSize()
    assert window.height() >= floor.height(), f"{factory} opens below its layout minimum"
    assert window.width() >= floor.width(), f"{factory} opens below its layout minimum"
    assert window.minimumHeight() >= floor.height(), f"{factory} can be dragged below it"
    assert window.minimumWidth() >= floor.width(), f"{factory} can be dragged below it"
    bridge.stop()


# --- launching and quitting --------------------------------------------------


def test_an_unacknowledged_host_does_not_swallow_the_launch():
    """A host that accepts the connection but never answers — hung, or from an older
    build — must not make the application silently stop starting."""
    import socket
    import threading

    address = "\0localasr-test-mute"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(address)
    server.listen(1)
    stop = threading.Event()

    def accept_and_ignore():
        server.settimeout(0.5)
        while not stop.is_set():
            try:
                conn, _ = server.accept()
            except (TimeoutError, OSError):
                continue
            conn.recv(64)  # read the request, answer nothing
            stop.wait(1.0)
            conn.close()

    thread = threading.Thread(target=accept_and_ignore, daemon=True)
    thread.start()
    try:
        assert not activate("subtitle", address, timeout=0.5)
    finally:
        stop.set()
        server.close()


def test_activation_is_confirmed_by_a_live_host():
    address = "\0localasr-test-ack"
    instance = SingleInstance(address)
    instance.acquire(lambda _message: None)
    try:
        assert activate("subtitle", address)
    finally:
        instance.release()


def test_an_unknown_window_name_still_opens_something(qt_app):
    """A launcher that appears to do nothing is indistinguishable from a broken
    install — and would leave only a tray icon."""
    from localasr.frontends.desktop.app import DesktopHost

    host = DesktopHost(qt_app)
    try:
        assert host.show("nonsense") == "subtitle"
        assert host.subtitle_window().isVisible()
    finally:
        host.shutdown()


@pytest.mark.parametrize("which", ["subtitle", "meeting", "dictation"])
def test_every_launch_argument_produces_a_visible_window(qt_app, which):
    from localasr.frontends.desktop.app import DesktopHost

    host = DesktopHost(qt_app)
    try:
        assert host.show(which) == which
        _pump()
        visible = [w for w in qt_app.topLevelWidgets() if w.isVisible()]
        assert visible, f"{which} left nothing on screen"
    finally:
        host.shutdown()


@pytest.mark.parametrize("factory", ["subtitle", "meeting", "dictation"])
def test_every_window_can_quit_without_the_tray(qt_app, factory):
    """The tray may be collapsed, or absent from the session entirely; it must never
    be the only way out."""
    from PySide6.QtGui import QKeySequence

    from localasr.apps.dictation import DictationController
    from localasr.frontends.desktop.dictation_window import DictationWindow
    from localasr.frontends.desktop.meeting_window import MeetingWindow
    from localasr.frontends.desktop.subtitle_window import SubtitleWindow

    context = AppContext()
    bridge = QtEventBridge(context.bus)
    window = {
        "subtitle": lambda: SubtitleWindow(context, bridge),
        "meeting": lambda: MeetingWindow(context, bridge),
        "dictation": lambda: DictationWindow(context, bridge, DictationController(context)),
    }[factory]()

    shortcuts = {s.key().toString() for s in window.findChildren(QShortcut)}
    assert QKeySequence(QKeySequence.StandardKey.Quit).toString() in shortcuts
    assert window.busy_reason() is None
    bridge.stop()


@pytest.mark.parametrize("factory", ["subtitle", "meeting", "dictation"])
def test_closing_an_idle_window_is_accepted(qt_app, factory):
    """Closing must not silently hide, leaving an invisible process holding the GPU."""
    from PySide6.QtGui import QCloseEvent

    from localasr.apps.dictation import DictationController
    from localasr.frontends.desktop.dictation_window import DictationWindow
    from localasr.frontends.desktop.meeting_window import MeetingWindow
    from localasr.frontends.desktop.subtitle_window import SubtitleWindow

    context = AppContext()
    bridge = QtEventBridge(context.bus)
    window = {
        "subtitle": lambda: SubtitleWindow(context, bridge),
        "meeting": lambda: MeetingWindow(context, bridge),
        "dictation": lambda: DictationWindow(context, bridge, DictationController(context)),
    }[factory]()
    window.show()
    _pump()

    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted()
    bridge.stop()


def test_a_busy_window_hides_instead_of_closing(qt_app, monkeypatch):
    """Work in progress is the one reason a close becomes a hide — and it says so."""
    from PySide6.QtGui import QCloseEvent
    from PySide6.QtWidgets import QMessageBox

    from localasr.frontends.desktop.subtitle_window import SubtitleWindow

    context = AppContext()
    bridge = QtEventBridge(context.bus)
    window = SubtitleWindow(context, bridge)
    monkeypatch.setattr(window, "busy_reason", lambda: "字幕任务正在运行")
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
    window.show()
    _pump()

    event = QCloseEvent()
    window.closeEvent(event)
    assert not event.isAccepted()
    assert not window.isVisible()
    bridge.stop()


def test_quit_request_reaches_the_running_host(qt_app):
    from localasr.frontends.desktop.single_instance import QUIT, request_quit

    address = "\0localasr-test-quit"
    seen = []
    instance = SingleInstance(address)
    instance.acquire(seen.append)
    try:
        assert request_quit(address)
        for _ in range(40):
            if seen:
                break
            _pump()
            time.sleep(0.01)
        assert seen == [QUIT]
    finally:
        instance.release()


def test_activation_from_the_socket_thread_runs_on_the_gui_thread(qt_app):
    """SingleInstance serves on its own thread. Constructing QWidgets there is not
    supported by Qt: the host wedges, stops acknowledging, and every later launch
    finds an address held by something that never answers."""
    import threading

    from localasr.frontends.desktop.app import DesktopHost, _Activation

    host = DesktopHost(qt_app)
    seen: list[int] = []
    original = host.show

    def record(message):
        seen.append(threading.get_ident())
        return original(message)

    host.show = record
    activation = _Activation(host)
    try:
        worker = threading.Thread(target=lambda: activation.requested.emit("meeting"))
        worker.start()
        worker.join()
        for _ in range(50):
            if seen:
                break
            _pump()
            time.sleep(0.01)
        assert seen, "activation never reached the host"
        assert seen[0] == threading.get_ident(), "window built off the GUI thread"
    finally:
        host.show = original
        host.shutdown()


def test_the_dictation_window_comes_back_after_delivering(qt_app, monkeypatch):
    """It hides so the text lands in the previous application. Staying hidden looks
    exactly like the app having crashed when the user pressed stop."""
    from localasr.apps.dictation import DictationController
    from localasr.frontends.desktop.dictation_window import DictationWindow

    context = AppContext()
    bridge = QtEventBridge(context.bus)
    window = DictationWindow(context, bridge, DictationController(context))
    window.show()
    _pump()

    window._hidden_for_delivery = True
    window.hide()
    _pump()
    assert not window.isVisible()

    window._thread_finished()
    _pump()
    assert window.isVisible(), "window stayed hidden after dictation finished"
    bridge.stop()


def test_the_toggle_button_keeps_offering_to_stop_while_recording(qt_app):
    """If the label flips back to 开始 mid-recording, the next press starts a second
    capture stream instead of stopping."""
    from localasr.apps.dictation import RECORDING, DictationController
    from localasr.frontends.desktop.dictation_window import DictationWindow

    context = AppContext()
    bridge = QtEventBridge(context.bus)
    window = DictationWindow(context, bridge, DictationController(context))
    window._show_state(RECORDING, "识别到的中间结果")
    assert "暂停" in window.toggle_button.text()
    assert window.toggle_button.isEnabled()
    bridge.stop()


def test_engine_loading_is_shown_as_loading_not_as_recognising(qt_app):
    from localasr.apps.dictation import PREPARING, DictationController
    from localasr.frontends.desktop.dictation_window import DictationWindow

    context = AppContext()
    bridge = QtEventBridge(context.bus)
    window = DictationWindow(context, bridge, DictationController(context))
    window._show_state(PREPARING)
    assert "加载" in window.state_badge.text()
    assert "识别" not in window.state_badge.text()
    bridge.stop()


def test_cancel_stays_available_while_the_engine_loads(qt_app):
    """Loading runs on a worker, so the generic busy rule disabled cancel for its whole
    duration, leaving Ctrl+Q as the only way out of a slow first launch."""
    from localasr.apps.dictation import PREPARING, DictationController
    from localasr.frontends.desktop.dictation_window import DictationWindow

    context = AppContext()
    bridge = QtEventBridge(context.bus)
    window = DictationWindow(context, bridge, DictationController(context))
    window._starting = True
    window._show_state(PREPARING)
    assert window.toggle_button.isEnabled()
    assert window.toggle_button.text() == "取消准备"

    window.cancel()
    assert window._abort_requested
    assert not window.toggle_button.isEnabled()
    bridge.stop()


def test_interim_text_reaches_the_live_view_during_recording(qt_app):
    """Each recognised sentence is shown while the microphone is still open."""
    from localasr.apps.dictation import RECORDING, DictationController
    from localasr.apps.events import FinalTranscript
    from localasr.core.types import Segment, Span
    from localasr.frontends.desktop.dictation_window import DictationWindow

    context = AppContext()
    bridge = QtEventBridge(context.bus)
    controller = DictationController(context)
    window = DictationWindow(context, bridge, controller)

    controller._segments = [Segment(span=Span(0.0, 1.0), text="第一句")]
    object.__setattr__(controller, "_session", object())  # pretend a session is open
    try:
        window._on_event(FinalTranscript(controller._segments[0]))
        assert window.last_text.toPlainText() == "第一句"
        assert window.copy_button.isEnabled()

        controller._segments.append(Segment(span=Span(1.5, 2.5), text="第二句"))
        window._on_event(FinalTranscript(controller._segments[1]))
        assert window.last_text.toPlainText() == "第一句第二句"
        window._show_state(RECORDING, controller.current_text)
        assert window.toggle_button.text() == "暂停识别"
    finally:
        controller._session = None
        bridge.stop()




def test_delivery_falls_back_to_the_qt_clipboard(qt_app, monkeypatch):
    """Correctly recognised text must never be discarded because a command-line tool
    is missing — a running Qt application always has a clipboard."""
    from localasr.frontends.desktop.app import _copy_via_qt
    from localasr.platform import text_output

    monkeypatch.setattr(text_output, "available_methods", lambda: [])
    monkeypatch.setattr(text_output, "copy_to_clipboard", lambda _text: False)

    delivery = text_output.deliver("识别出来的一句话", fallback=_copy_via_qt)
    assert delivery.method == "clipboard"
    assert qt_app.clipboard().text() == "识别出来的一句话"


def test_delivery_still_raises_when_there_is_no_fallback_at_all(monkeypatch):
    from localasr.platform import text_output

    monkeypatch.setattr(text_output, "available_methods", lambda: [])
    monkeypatch.setattr(text_output, "copy_to_clipboard", lambda _text: False)
    with pytest.raises(text_output.TextOutputError):
        text_output.deliver("x")


def test_the_dictation_window_has_a_single_two_state_control(qt_app):
    """One control: start, or pause. Anything else is a third option the user did not
    ask for."""
    from PySide6.QtWidgets import QPushButton

    from localasr.apps.dictation import DictationController
    from localasr.frontends.desktop.dictation_window import DictationWindow

    context = AppContext()
    bridge = QtEventBridge(context.bus)
    window = DictationWindow(context, bridge, DictationController(context))

    assert not hasattr(window, "cancel_button")
    status_buttons = [
        b.text()
        for b in window.findChildren(QPushButton)
        if b.text() in {"开始识别", "暂停识别", "取消准备"}
    ]
    assert status_buttons == ["开始识别"]
    bridge.stop()


def test_the_result_area_is_editable(qt_app):
    """Fixing the one word it got wrong before sending the text on is the point."""
    from localasr.apps.dictation import DictationController
    from localasr.frontends.desktop.dictation_window import DictationWindow

    context = AppContext()
    bridge = QtEventBridge(context.bus)
    window = DictationWindow(context, bridge, DictationController(context))

    assert not window.last_text.isReadOnly()
    assert window.last_text.isUndoRedoEnabled()
    window.last_text.setPlainText("可以编辑")
    assert window.copy_button.isEnabled()
    assert window.clear_button.isEnabled()
    bridge.stop()


def test_new_sentences_are_appended_without_discarding_edits(qt_app):
    """Replacing the box with the controller's text would wipe the user's corrections
    every time another sentence arrived."""
    from localasr.apps.dictation import DictationOptions
    from localasr.apps.events import FinalTranscript
    from localasr.core.types import Segment, Span
    from localasr.frontends.desktop.dictation_window import DictationWindow

    first = Segment(span=Span(0.0, 1.0), text="第一句", utterance_id="a")
    second = Segment(span=Span(1.0, 2.0), text="第二句", utterance_id="b")

    class Recording:
        recording = True
        state = RECORDING
        current_text = "第一句第二句"
        options = DictationOptions()
        segments = (first,)

    controller = Recording()
    context = AppContext()
    bridge = QtEventBridge(context.bus)
    window = DictationWindow(context, bridge, controller)

    window._on_event(FinalTranscript(first))
    assert window.last_text.toPlainText() == "第一句"

    window.last_text.setPlainText("第一句（已改）")
    controller.segments = (first, second)
    window._on_event(FinalTranscript(second))

    assert window.last_text.toPlainText() == "第一句（已改）第二句"
    bridge.stop()


def test_clearing_the_result_does_not_replay_old_sentences(qt_app):
    from localasr.apps.dictation import DictationOptions
    from localasr.apps.events import FinalTranscript
    from localasr.core.types import Segment, Span
    from localasr.frontends.desktop.dictation_window import DictationWindow

    settled = Segment(span=Span(0.0, 1.0), text="旧的一句", utterance_id="a")

    class Recording:
        recording = True
        state = RECORDING
        current_text = "旧的一句"
        options = DictationOptions()
        segments = (settled,)

    context = AppContext()
    bridge = QtEventBridge(context.bus)
    window = DictationWindow(context, bridge, Recording())

    window._on_event(FinalTranscript(settled))
    window._clear_result()
    window._on_event(FinalTranscript(settled))

    assert window.last_text.toPlainText() == ""
    bridge.stop()


def test_the_model_stays_resident_until_the_application_exits(qt_app, tmp_path, monkeypatch):
    """Reloading costs seconds and the user did not ask for it. Isolated from the
    ambient config: a value saved before the default changed would mask the default."""
    monkeypatch.setenv("LOCALASR_CONFIG", str(tmp_path / "config.toml"))
    context = AppContext()
    assert context.settings.idle_timeout == 0
    assert context.engine.idle_timeout == 0


def test_a_saved_setting_still_wins_over_the_default(tmp_path, monkeypatch):
    """Changing a default does nothing for a config already on disk — which is how a
    stale 120 s kept unloading the model after the default became 0."""
    config = tmp_path / "config.toml"
    config.write_text("idle_timeout = 45.0\n", encoding="utf-8")
    monkeypatch.setenv("LOCALASR_CONFIG", str(config))
    assert AppContext().settings.idle_timeout == 45.0


# --- parity across the three applications ------------------------------------


def test_dictation_does_not_capture_system_audio_by_default(qt_app):
    """Dictation records what you say; a video's narration must not land in it."""
    from localasr.apps.dictation import DictationController
    from localasr.frontends.desktop.dictation_window import DictationWindow

    context = AppContext()
    bridge = QtEventBridge(context.bus)
    window = DictationWindow(context, bridge, DictationController(context))

    assert not window.capture_system.isChecked()
    assert DictationOptions().capture_system is False
    bridge.stop()


def test_meeting_captures_system_audio_by_default(qt_app):
    """The other party arrives only through the system monitor."""
    from localasr.apps.meeting import MeetingOptions
    from localasr.frontends.desktop.meeting_window import MeetingWindow

    context = AppContext()
    bridge = QtEventBridge(context.bus)
    window = MeetingWindow(context, bridge)

    assert window.system_box.isChecked()
    assert MeetingOptions().capture_system is True
    bridge.stop()


@pytest.mark.parametrize("factory", ["dictation", "meeting"])
def test_both_live_windows_offer_device_selection(qt_app, factory):
    """Written once and shared: an untested, muted input records a perfect nothing."""
    from localasr.apps.dictation import DictationController
    from localasr.frontends.desktop.device_panel import DevicePanel
    from localasr.frontends.desktop.meeting_window import MeetingWindow

    context = AppContext()
    bridge = QtEventBridge(context.bus)
    window = {
        "dictation": lambda: DictationWindow(context, bridge, DictationController(context)),
        "meeting": lambda: MeetingWindow(context, bridge),
    }[factory]()

    assert isinstance(window.device_panel, DevicePanel)
    assert window.device_panel.test_button.text() == "测试输入"
    bridge.stop()


@pytest.mark.parametrize("factory", ["subtitle", "meeting"])
def test_every_window_reports_dropped_segments(qt_app, factory):
    """The defect fixed in dictation applied to all three: a segment that vanishes with
    no explanation looks like audio the pipeline lost."""
    from localasr.apps.events import SegmentDropped
    from localasr.core.types import Span
    from localasr.frontends.desktop.meeting_window import MeetingWindow
    from localasr.frontends.desktop.subtitle_window import SubtitleWindow

    context = AppContext()
    bridge = QtEventBridge(context.bus)
    window = {
        "subtitle": lambda: SubtitleWindow(context, bridge),
        "meeting": lambda: MeetingWindow(context, bridge),
    }[factory]()

    if factory == "meeting":
        # A meeting only renders events while it owns a recording; otherwise it would
        # show drops belonging to dictation.
        window._controller = object()

    before = window.status.text() if factory == "subtitle" else window.transcript.toPlainText()
    window._on_event(SegmentDropped(1, Span(3.0, 4.0), "噪声", "repetition loop"))
    after = window.status.text() if factory == "subtitle" else window.transcript.toPlainText()

    assert after != before, f"{factory} dropped a segment without saying so"
    assert "跳过" in after
    if factory == "meeting":
        window._controller = None
    bridge.stop()


def test_the_subtitle_window_reports_failure_and_cancellation(qt_app):
    from localasr.apps.events import JobCancelled, JobFailed
    from localasr.frontends.desktop.subtitle_window import SubtitleWindow

    context = AppContext()
    bridge = QtEventBridge(context.bus)
    window = SubtitleWindow(context, bridge)

    window._on_event(JobFailed("解码失败"))
    assert "解码失败" in window.status.text()

    window._on_event(JobCancelled(completed=3))
    assert "3" in window.status.text()
    bridge.stop()


def test_the_theme_is_dark_and_every_foreground_stays_readable():
    """The combo box was unreadable because its popup is a separate view that the
    QComboBox rules never reached, so it kept the platform's light-theme text on the
    dark surface. Contrast is checked rather than merely asserting the rule exists."""
    surfaces = (theme.BG, theme.SURFACE, theme.SURFACE_HIGH)
    for surface in surfaces:
        assert _contrast(theme.TEXT, surface) >= 4.5, f"body text unreadable on {surface}"
    assert _contrast(theme.TEXT_MUTED, theme.SURFACE) >= 4.5
    assert _contrast(theme.ACCENT_TEXT, theme.ACCENT) >= 4.5, "primary button label"

    # The popup rule itself, since no amount of colour arithmetic proves it is applied.
    assert "QComboBox QAbstractItemView" in theme.APP_STYLESHEET
    for widget in ("QCheckBox::indicator", "QMenu", "QMessageBox", "QScrollBar:vertical"):
        assert widget in theme.APP_STYLESHEET, f"{widget} left to the platform palette"


def test_the_palette_backs_up_the_stylesheet(qt_app):
    """Native pieces (file dialogs, some item views) read the palette, not the sheet."""
    from PySide6.QtGui import QPalette

    palette = qt_app.palette()
    for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Base):
        assert palette.color(role).lightness() < 128, f"{role} is not dark"
    assert palette.color(QPalette.ColorRole.Text).lightness() > 128


@pytest.mark.parametrize("factory", ["dictation", "meeting", "subtitle"])
def test_windows_render_under_the_dark_theme(qt_app, monkeypatch, factory):
    """Each window is built and polished so a stylesheet syntax error surfaces here."""
    monkeypatch.setattr(
        "localasr.frontends.desktop.device_panel.list_devices",
        lambda: [DeviceInfo(0, "sof-hda-dsp: - (hw:0,6)", 2, 48000.0, is_monitor=False)],
    )
    from localasr.frontends.desktop.meeting_window import MeetingWindow
    from localasr.frontends.desktop.subtitle_window import SubtitleWindow

    context = AppContext()
    bridge = QtEventBridge(context.bus)
    window = {
        "dictation": lambda: DictationWindow(context, bridge, DictationController(context)),
        "meeting": lambda: MeetingWindow(context, bridge),
        "subtitle": lambda: SubtitleWindow(context, bridge),
    }[factory]()
    window.setStyleSheet(theme.APP_STYLESHEET)
    window.ensurePolished()
    _pump(3)
    assert window.styleSheet()
    bridge.stop()


def test_the_transcript_dominates_the_dictation_window(qt_app, monkeypatch):
    """The window exists to show recognised text. Model and microphone are chosen once;
    the transcript is read continuously, so it gets the area, not the setup cards."""
    monkeypatch.setattr(
        "localasr.frontends.desktop.device_panel.list_devices",
        lambda: [DeviceInfo(0, "sof-hda-dsp: - (hw:0,6)", 2, 48000.0, is_monitor=False)],
    )
    context = AppContext()
    bridge = QtEventBridge(context.bus)
    window = DictationWindow(context, bridge, DictationController(context))
    window.show()
    _pump()

    assert window.last_text.height() >= 0.4 * window.height(), (
        "the transcript lost its share of the window to setup controls"
    )

    setup = (window.model_panel.height(), window.device_panel.height())
    before = window.last_text.height()
    window.resize(window.width(), window.height() + 240)
    _pump()

    # Setup cards are fixed height, so growing the window can only feed the transcript.
    assert (window.model_panel.height(), window.device_panel.height()) == setup
    assert window.last_text.height() == before + 240
    bridge.stop()


def test_setup_panels_stay_single_line(qt_app, monkeypatch):
    """The model path once wrapped to three lines and took more room than the results.
    It belongs in a tooltip, and these labels must never wrap again."""
    monkeypatch.setattr(
        "localasr.frontends.desktop.device_panel.list_devices",
        lambda: [DeviceInfo(0, "sof-hda-dsp: - (hw:0,6)", 2, 48000.0, is_monitor=False)],
    )
    context = AppContext()
    bridge = QtEventBridge(context.bus)
    window = DictationWindow(context, bridge, DictationController(context))

    assert not window.model_panel.status.wordWrap()
    assert not window.device_panel.status.wordWrap()
    assert str(manager.model_dir(window.model_panel.selected_spec)) in (
        window.model_panel.status.toolTip()
    ), "the full path must still be reachable, just not in the layout"
    assert str(manager.model_dir(window.model_panel.selected_spec)) not in (
        window.model_panel.status.text()
    )
    bridge.stop()


def test_the_toggle_button_changes_colour_with_what_it_will_do(qt_app, monkeypatch):
    """The same button starts and stops. Reading its label mid-sentence is harder than
    seeing its colour, so blue means start and red means stop."""
    from localasr.apps.dictation import IDLE, PREPARING, RECORDING, TRANSCRIBING

    monkeypatch.setattr(
        "localasr.frontends.desktop.device_panel.list_devices",
        lambda: [DeviceInfo(0, "sof-hda-dsp: - (hw:0,6)", 2, 48000.0, is_monitor=False)],
    )
    context = AppContext()
    bridge = QtEventBridge(context.bus)
    window = DictationWindow(context, bridge, DictationController(context))

    for state, expected in (
        (IDLE, "primary"),
        (RECORDING, "stop"),
        (PREPARING, "stop"),
        (TRANSCRIBING, "primary"),
    ):
        window._show_state(state)
        assert window.toggle_button.property("role") == expected, f"wrong colour for {state}"

    # Not stretched: a full-width button is a big target but says nothing about which
    # of the two actions it is about to perform.
    window.show()
    _pump()
    assert window.toggle_button.width() < window.width() // 2
    bridge.stop()


def test_the_stop_colour_is_readable_and_distinct():
    assert _contrast(theme.STOP_TEXT, theme.STOP) >= 4.5
    assert theme.STOP != theme.ACCENT
    assert 'QPushButton[role="stop"]' in theme.APP_STYLESHEET


def test_only_one_control_wears_the_accent_once_the_model_is_ready(qt_app, monkeypatch):
    """The accent is only a signal while it is scarce. With the model in place, the
    recognition button owns it and the model panel's button steps back."""
    monkeypatch.setattr(manager, "is_downloaded", lambda _spec: True)
    monkeypatch.setattr(
        "localasr.frontends.desktop.device_panel.list_devices",
        lambda: [DeviceInfo(0, "sof-hda-dsp: - (hw:0,6)", 2, 48000.0, is_monitor=False)],
    )
    context = AppContext()
    bridge = QtEventBridge(context.bus)
    window = DictationWindow(context, bridge, DictationController(context))

    assert window.toggle_button.property("role") == "primary"
    assert window.model_panel.primary_button.property("role") != "primary"

    # A model that is not downloaded does block recognition, so there it keeps the accent.
    monkeypatch.setattr(manager, "is_downloaded", lambda _spec: False)
    monkeypatch.setattr(manager, "missing_bytes", lambda spec: sum(f.size for f in spec.files))
    window.model_panel.refresh()
    assert window.model_panel.primary_button.property("role") == "primary"
    bridge.stop()


def test_starting_again_keeps_what_is_already_in_the_box(qt_app):
    """Only 清空 empties the transcript. A second round of dictation may be adding to
    notes the user has already corrected, and starting it is not a request to discard
    them."""
    from localasr.apps.dictation import IDLE
    from localasr.core.types import Segment, Span

    kept = Segment(span=Span(0.0, 1.0), text="上一轮的句子", utterance_id="a")

    class Controller:
        recording = True
        state = RECORDING
        current_text = ""
        options = DictationOptions()
        segments = (kept,)

    controller = Controller()
    context = AppContext()
    bridge = QtEventBridge(context.bus)
    window = DictationWindow(context, bridge, controller)

    window._on_event(FinalTranscript(kept))
    window.last_text.setPlainText("上一轮的句子（已手动修正）")

    # A new session begins: the controller starts over with an empty segment list.
    controller.segments = ()
    window._starting = True
    window._operation_done("")

    assert window.last_text.toPlainText() == "上一轮的句子（已手动修正）"
    assert window._rendered == 0, "the next segment must still render as the first one"

    fresh = Segment(span=Span(0.0, 1.0), text="新一轮的句子", utterance_id="b")
    controller.segments = (fresh,)
    window._on_event(FinalTranscript(fresh))
    assert "上一轮的句子（已手动修正）" in window.last_text.toPlainText()
    assert "新一轮的句子" in window.last_text.toPlainText()

    # Clearing stays available and is the only thing that empties it.
    window._show_state(IDLE)
    window._clear_result()
    assert window.last_text.toPlainText() == ""
    bridge.stop()


# --- split backends: which setup panel a window shows -------------------------


@pytest.mark.parametrize("factory", ["dictation", "meeting", "subtitle"])
def test_a_remote_node_replaces_the_local_model_panel(qt_app, monkeypatch, factory):
    """With recognition on another machine there is nothing in this VRAM to load or
    release, so 「加载到显存」 would name work happening on a different host."""
    from localasr.frontends.desktop.backend_panel import BackendPanel
    from localasr.frontends.desktop.meeting_window import MeetingWindow
    from localasr.frontends.desktop.subtitle_window import SubtitleWindow

    monkeypatch.setattr(
        "localasr.frontends.desktop.device_panel.list_devices",
        lambda: [DeviceInfo(0, "sof-hda-dsp: - (hw:0,6)", 2, 48000.0, is_monitor=False)],
    )
    monkeypatch.setattr(
        "localasr.frontends.desktop.backend_panel._probe_asr",
        lambda _c: __import__(
            "localasr.frontends.desktop.backend_panel", fromlist=["BackendStatus"]
        ).BackendStatus("就绪", "orin", "success"),
    )
    monkeypatch.setattr(
        "localasr.frontends.desktop.backend_panel._probe_refiner",
        lambda _c: __import__(
            "localasr.frontends.desktop.backend_panel", fromlist=["BackendStatus"]
        ).BackendStatus("就绪", "local", "success"),
    )

    context = AppContext()
    context.settings.node_url = "http://asr-node.local:8090"
    bridge = QtEventBridge(context.bus)
    window = {
        "dictation": lambda: DictationWindow(context, bridge, DictationController(context)),
        "meeting": lambda: MeetingWindow(context, bridge),
        "subtitle": lambda: SubtitleWindow(context, bridge),
    }[factory]()

    assert isinstance(window.backend_panel, BackendPanel)
    assert window.model_panel is None, "the local chooser must not be shown"
    assert window.busy_reason() is None, "a missing model panel must not break this"
    window.backend_panel.wait_for_probe()
    bridge.stop()


@pytest.mark.parametrize("factory", ["dictation", "meeting", "subtitle"])
def test_without_a_node_the_local_model_panel_stays(qt_app, monkeypatch, factory):
    """Nothing changes for a single-machine install."""
    from localasr.frontends.desktop.meeting_window import MeetingWindow
    from localasr.frontends.desktop.model_panel import ModelPanel
    from localasr.frontends.desktop.subtitle_window import SubtitleWindow

    monkeypatch.setattr(
        "localasr.frontends.desktop.device_panel.list_devices",
        lambda: [DeviceInfo(0, "sof-hda-dsp: - (hw:0,6)", 2, 48000.0, is_monitor=False)],
    )
    context = AppContext()
    context.settings.node_url = None
    monkeypatch.delenv("LOCALASR_NODE_URL", raising=False)
    bridge = QtEventBridge(context.bus)
    window = {
        "dictation": lambda: DictationWindow(context, bridge, DictationController(context)),
        "meeting": lambda: MeetingWindow(context, bridge),
        "subtitle": lambda: SubtitleWindow(context, bridge),
    }[factory]()

    assert isinstance(window.model_panel, ModelPanel)
    assert window.backend_panel is None
    bridge.stop()


# --- model management from the backend panel ----------------------------------


def _backend_panel(qt_app, monkeypatch):  # noqa: ANN001, ANN202
    from localasr.frontends.desktop import backend_panel as module

    ready = module.BackendStatus("就绪", "ok", "success")
    monkeypatch.setattr(module, "_probe_asr", lambda _c: ready)
    monkeypatch.setattr(module, "_probe_refiner", lambda _c: ready)
    context = AppContext()
    context.settings.node_url = "http://asr-node.local:8090"
    context.settings.node_token = "tok"
    panel = module.BackendPanel(context)
    panel.wait_for_probe()
    return panel, module


def test_releasing_asks_the_node_and_reports_what_was_freed(qt_app, monkeypatch):
    """The only model this machine can reach an API for lives on the node; releasing it
    is how the board's memory is handed back without stopping the service."""
    import httpx

    panel, module = _backend_panel(qt_app, monkeypatch)
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(
            200, json={"released": ["asr"], "loaded": {}, "memory_mb": {"available": 5837}}
        )

    real_client = httpx.Client

    def patched(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(module.httpx, "Client", patched)
    panel._node_action("release")
    panel.wait_for_probe(10_000)
    _pump()

    assert seen["url"].endswith("/api/v1/models/release")
    assert seen["auth"] == "Bearer tok", "the node refuses an unauthenticated release"
    assert "已卸载 asr" in panel.action_status.text()
    assert "5837" in panel.action_status.text()


def test_releasing_is_refused_while_a_workload_is_running(qt_app, monkeypatch):
    """Evicting the model mid-recording would fail the very request using it."""
    from localasr.apps.coordinator import Activity

    panel, _ = _backend_panel(qt_app, monkeypatch)
    token = panel.context.coordinator.acquire(Activity.DICTATION)
    try:
        panel._node_action("release")
        assert panel._release_thread is None, "must not have started"
        assert "请先停止" in panel.action_status.text()
    finally:
        panel.context.coordinator.release(token)


def test_the_panel_offers_both_management_actions(qt_app, monkeypatch):
    panel, _ = _backend_panel(qt_app, monkeypatch)
    # Two per backend: warm both before a session, release both after. Leaving it to
    # the first request means paying a cold load after the user has begun speaking.
    assert panel.asr_load_button.text() == "启动"
    assert panel.release_button.text() == "卸载"
    assert panel.refiner_load_button.text() == "启动"
    assert panel.refiner_unload_button.text() == "卸载"
    assert panel.import_button.text() == "导入模型…"
    # Where each acts is not obvious from the label, so it is on the tooltip.
    assert "识别节点" in panel.release_button.toolTip()
    assert "本机整理服务" in panel.import_button.toolTip()


def test_a_failed_import_is_reported_not_raised(qt_app, monkeypatch, tmp_path):
    """Copying gigabytes can fail for a dozen reasons; none of them should take the
    desktop host down with it."""
    from localasr.frontends.desktop import backend_panel as module
    from localasr.registry import imported

    panel, _ = _backend_panel(qt_app, monkeypatch)
    monkeypatch.setattr(
        imported, "import_model", lambda _r: (_ for _ in ()).throw(OSError("disk full"))
    )
    thread = module._ImportThread(imported.ImportRequest(model_path=tmp_path / "m.gguf"))
    received = []
    thread.done.connect(received.append)
    thread.run()

    assert received and "导入失败" in received[0] and "disk full" in received[0]


def test_a_locally_owned_refiner_shows_not_started_rather_than_broken(qt_app, monkeypatch):
    """It is ours to start and stop, so "not running" is what the machine looks like
    before a session — a state, not a fault."""
    from localasr.frontends.desktop import backend_panel as module

    context = AppContext()
    context.settings.node_url = "http://asr-node.local:8090"
    context.settings.refiner_url = None
    context.settings.refiner_model_id = "qwen3_5-4b-refiner-q4"
    assert context.refiner_managed

    status = module._probe_refiner(context)
    assert status.label == "未启动"
    assert status.tone == "neutral", "an unstarted model is not an error"
    assert "qwen3_5-4b-refiner-q4" in status.detail


# --- what each button is allowed to do, and when ------------------------------


def _idle_window(qt_app):  # noqa: ANN001, ANN202
    """A window with a transcript, a refinement, and nothing running."""

    class IdleController:
        recording = False
        current_text = ""
        state = "idle"
        segments = ()
        options = DictationOptions()

    context = AppContext()
    context.settings.node_url = "http://asr-node.local:8090"
    context.settings.refiner_model_id = "qwen3_5-4b-refiner-q4"
    bridge = QtEventBridge(context.bus)
    window = DictationWindow(context, bridge, IdleController())
    window.last_text.setPlainText("下周一交三个报告")
    window.refined_text.setPlainText("下周一交三个报告。")
    return window, bridge


def test_everything_is_available_when_nothing_is_running(qt_app):
    window, bridge = _idle_window(qt_app)

    assert window.copy_button.isEnabled()
    assert window.clear_button.isEnabled()
    assert window.refine_button.isEnabled()
    assert window.copy_raw_button.isEnabled()
    assert window.save_refined_button.isEnabled()
    assert window.instruction.isEnabled()
    bridge.stop()


def test_a_refinement_in_flight_protects_the_text_it_is_working_from(qt_app):
    """The right pane is headed 整理结果 and the status line tells the user to compare it
    with the left. Emptying the left pane, or dictating more into it, while the model is
    still working makes that instruction impossible to follow.

    Every one of these was permitted: 清空 only watched the recognition worker, 开始识别
    only watched recognition state, and 整理文本 watched neither — so appending a segment
    mid-refinement lit it up again, and pressing it did nothing at all.
    """
    window, bridge = _idle_window(qt_app)
    window._refiner = object()  # a refinement is in flight
    window._result_changed()
    window._show_state("idle")

    assert not window.clear_button.isEnabled(), "would strand the right pane"
    assert not window.toggle_button.isEnabled(), "would append to text being refined"
    assert not window.refine_button.isEnabled(), "a second refinement is refused anyway"
    assert not window.instruction.isEnabled(), "editing it cannot affect the sent request"

    # Reading harms nothing, and the user may well want the text while they wait.
    assert window.copy_button.isEnabled()
    assert window.copy_raw_button.isEnabled()
    window._refiner = None
    bridge.stop()


def test_recognising_blocks_the_same_things_refining_does(qt_app):
    """Both are activities that own the transcript; they had different rules before."""
    window, bridge = _idle_window(qt_app)
    window.controller.recording = True
    window._result_changed()

    assert not window.clear_button.isEnabled()
    assert not window.refine_button.isEnabled()
    assert window.copy_button.isEnabled(), "reading is safe in every state"
    bridge.stop()


def test_the_refined_pane_drives_its_own_buttons(qt_app):
    """They used to be set once, by the handler that produced a result, so nothing that
    happened afterwards could correct them."""
    window, bridge = _idle_window(qt_app)
    assert window.save_refined_button.isEnabled()
    assert "整理结果" in window.copy_button.toolTip(), "复制 follows the pane too"

    window.refined_text.clear()
    assert not window.save_refined_button.isEnabled()
    assert "原始转写" in window.copy_button.toolTip()

    window.refined_text.setPlainText("再次整理的结果")
    assert window.save_refined_button.isEnabled()
    assert "整理结果" in window.copy_button.toolTip()
    bridge.stop()


def test_a_lit_refine_button_can_always_be_pressed(qt_app):
    """The invariant behind the table: `_refine` refuses a second refinement outright, so
    any state where the button is enabled and that refusal would fire is a lie."""
    window, bridge = _idle_window(qt_app)

    for refining in (False, True):
        for recording in (False, True):
            window._refiner = object() if refining else None
            window.controller.recording = recording
            window._result_changed()
            if window.refine_button.isEnabled():
                assert not refining and not recording, (
                    f"enabled while refining={refining} recording={recording}, "
                    "but _refine would return immediately"
                )
    window._refiner = None
    bridge.stop()


# --- typing Chinese into the application at all ------------------------------


def test_a_missing_input_method_plugin_is_stepped_around() -> None:
    """The bug this exists for: no Chinese could be typed into any field.

    The desktop sets QT_IM_MODULE=fcitx system-wide, but PySide6 ships its own Qt with
    its own plugin directory and no fcitx plugin in it. Qt then built no input context
    at all and dropped every CJK keystroke silently. Clearing the variable hands the job
    to the Wayland text-input protocol, which fcitx5 already serves.
    """
    from localasr.frontends.desktop.app import fix_input_method

    env = {"QT_IM_MODULE": "fcitx", "WAYLAND_DISPLAY": "wayland-0"}
    assert fix_input_method(env) == "fcitx"
    assert "QT_IM_MODULE" not in env


def test_a_plugin_that_is_present_is_left_alone() -> None:
    """PySide6 does ship ibus. Clearing a variable Qt can honour would break a working
    setup to fix one that is not broken."""
    from localasr.frontends.desktop.app import fix_input_method

    env = {"QT_IM_MODULE": "ibus", "WAYLAND_DISPLAY": "wayland-0"}
    assert fix_input_method(env) is None
    assert env["QT_IM_MODULE"] == "ibus"


def test_x11_is_never_touched() -> None:
    """Under X11 there is no text-input protocol to fall back to: QT_IM_MODULE is how
    an input method is found at all, so removing it would cause the very bug it fixes."""
    from localasr.frontends.desktop.app import fix_input_method

    env = {"QT_IM_MODULE": "fcitx", "DISPLAY": ":0"}
    assert fix_input_method(env) is None
    assert env["QT_IM_MODULE"] == "fcitx"


def test_nothing_to_do_when_no_input_method_is_configured() -> None:
    from localasr.frontends.desktop.app import fix_input_method

    assert fix_input_method({"WAYLAND_DISPLAY": "wayland-0"}) is None


def test_recording_is_available_again_once_a_refinement_ends(qt_app):
    """The mirror of an earlier bug, and introduced by its fix.

    Enablement lived in two functions: `_result_changed` owned 清空/整理/复制, and
    `_show_state` owned 开始识别. Starting a refinement called both; finishing one called
    only the first, so the record button stayed grey for the rest of the session with
    nothing left that would ever recompute it. Clearing the text afterwards looked like
    the cause, because it was the last thing the user touched.
    """
    from localasr.refine.types import RefinementMode, RefinementResult

    context = AppContext()
    context.settings.node_url = "http://asr-node.local:8090"
    bridge = QtEventBridge(context.bus)
    window = DictationWindow(context, bridge, DictationController(context))
    window.last_text.setPlainText("嗯那个我们下周一要交三个报告")

    window._refiner = object()
    window._result_changed()
    window._show_state(window.controller.state)
    assert not window.toggle_button.isEnabled(), "recording during a refinement is refused"

    window._refined(
        RefinementResult(
            raw_text="嗯那个我们下周一要交三个报告",
            refined_text="我们下周一要交三个报告。",
            mode=RefinementMode.CONSERVATIVE,
            source_segment_ids=(),
        )
    )
    window._refine_finished()

    assert window.toggle_button.isEnabled(), "nothing is running; recording must be possible"
    window._clear_result()
    assert window.toggle_button.isEnabled(), "clearing text has nothing to do with recording"
    bridge.stop()


def _clipboard_text() -> str:
    return QApplication.clipboard().text()


def test_copy_takes_the_refined_text_once_there_is_one(qt_app):
    """复制 is the action people reach for after refining, and it was handing back the
    raw transcript — the thing they had just asked to have tidied up."""
    context = AppContext()
    bridge = QtEventBridge(context.bus)
    window = DictationWindow(context, bridge, DictationController(context))

    window.last_text.setPlainText("嗯那个我们下周一要交三个报告")
    window.copy_button.click()
    assert _clipboard_text() == "嗯那个我们下周一要交三个报告", "no refinement yet"

    window.refined_text.setPlainText("我们下周一要交三个报告。")
    window.copy_button.click()
    assert _clipboard_text() == "我们下周一要交三个报告。"
    bridge.stop()


def test_the_original_stays_reachable_after_a_refinement(qt_app):
    """The two panes exist because the transcript is the evidence. Making 复制 prefer the
    refinement is only safe while the original is still one click away."""
    context = AppContext()
    bridge = QtEventBridge(context.bus)
    window = DictationWindow(context, bridge, DictationController(context))

    window.last_text.setPlainText("原始的逐字转写")
    window.refined_text.setPlainText("整理过的版本。")
    window.copy_raw_button.click()

    assert _clipboard_text() == "原始的逐字转写"
    bridge.stop()


def test_copying_says_which_of_the_two_it_took(qt_app):
    """Otherwise the button silently changes meaning halfway through a session."""
    context = AppContext()
    bridge = QtEventBridge(context.bus)
    window = DictationWindow(context, bridge, DictationController(context))

    window.last_text.setPlainText("逐字转写")
    window.copy_button.click()
    assert "原文" in window.status_message.text()

    window.refined_text.setPlainText("整理结果。")
    window.copy_button.click()
    assert "整理" in window.status_message.text()
    bridge.stop()
