"""The three application controllers, against a stub engine and a fake microphone."""

import json
import threading
import time

import numpy as np
import pytest

from localasr.apps.coordinator import Activity, ActivityConflict
from localasr.apps.dictation import IDLE, RECORDING, DictationController, DictationOptions
from localasr.apps.meeting import (
    MeetingController,
    MeetingOptions,
    default_journal_path,
    load_journal,
)
from localasr.apps.subtitle import SubtitleController, SubtitleOptions, expand_media
from localasr.context import AppContext
from localasr.core.engine.client import TranscriptionResult
from localasr.core.types import SAMPLE_RATE, AudioBlock
from localasr.platform import text_output


class StubClient:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, *, language=None, **_kwargs):
        self.calls.append(audio.duration)
        return TranscriptionResult(text="识别结果", language="Chinese")


class StubEngine:
    def __init__(self, spec):
        self.client = StubClient()
        self.spec = spec
        self.warning = None

    def acquire(self):
        return self.client

    def release(self):
        pass

    def hold(self):
        self.held = getattr(self, "held", 0) + 1

    def release_hold(self):
        self.held = getattr(self, "held", 0) - 1

    def shutdown(self):
        pass


@pytest.fixture
def context(monkeypatch):
    ctx = AppContext()
    engine = StubEngine(ctx.spec)
    monkeypatch.setattr(type(ctx), "engine", property(lambda _self: engine))
    # The engine is a stub, so no weights are needed — but the availability check reads
    # the disk, and these tests quietly depended on a 2.4 GB download being present.
    # They passed on the machine that had it and would fail on a fresh checkout, which
    # is not a property of the controller logic they are about.
    monkeypatch.setattr(type(ctx), "require_model_available", lambda _self: None)
    return ctx


def _no_overlap(*_args, **_kwargs):
    """Keep tests off the real microphone; overlap detection is tested on its own."""
    from localasr.capture.overlap import OverlapResult

    return OverlapResult(False, 0.0, 0.0, "两路内容不同（测试替身）")


class FakeSource:
    """Replays a file as timed blocks, with the MicrophoneSource interface.

    `stop()` ends the stream but, like the real device, hands over what was already
    captured rather than truncating mid-buffer. `exhausted` lets a test wait for the
    replay deterministically instead of sleeping.
    """

    def __init__(self, path, source="mic", **_kwargs):
        self.path = path
        self.source = source
        self.device = None
        self.warning = None
        self._stopped = False
        self.exhausted = threading.Event()

    def open(self):
        return None

    def close(self):
        self._stopped = True

    def stop(self):
        self._stopped = True

    def __iter__(self):
        from localasr.capture.microphone import blocks_from_file

        try:
            for block in blocks_from_file(self.path, source=self.source, block_seconds=0.2):
                if self._stopped:
                    return
                yield block
        finally:
            self.exhausted.set()

    def wait(self, timeout: float = 10.0) -> None:
        assert self.exhausted.wait(timeout), "replay did not finish"


# --- subtitle ----------------------------------------------------------------


def test_expand_media_walks_directories_and_ignores_non_media(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.mp4").write_bytes(b"")

    found = expand_media([tmp_path])
    assert [p.name for p in found] == ["a.wav", "b.mp4"]


def test_subtitle_writes_one_file_per_input(context, english_speech, chinese_speech, tmp_path):
    controller = SubtitleController(
        context, SubtitleOptions(fmt="srt", output_dir=tmp_path, overwrite=True)
    )
    results = controller.run([english_speech, chinese_speech])

    assert len(results) == 2
    assert all(r.output and r.output.exists() for r in results)
    assert (tmp_path / f"{english_speech.stem}.srt").read_text(encoding="utf-8")


def test_subtitle_skips_existing_output_unless_overwriting(context, english_speech, tmp_path):
    options = SubtitleOptions(fmt="srt", output_dir=tmp_path)
    (tmp_path / f"{english_speech.stem}.srt").write_text("existing", encoding="utf-8")

    results = SubtitleController(context, options).run([english_speech])
    assert results[0].skipped
    assert (tmp_path / f"{english_speech.stem}.srt").read_text(encoding="utf-8") == "existing"


def test_subtitle_removes_the_journal_once_the_file_is_written(context, english_speech, tmp_path):
    controller = SubtitleController(
        context, SubtitleOptions(fmt="srt", output_dir=tmp_path, overwrite=True)
    )
    controller.run([english_speech])
    assert not controller.journal_for(english_speech).exists()


def test_one_failing_file_does_not_abandon_the_batch(context, english_speech, tmp_path):
    missing = tmp_path / "broken.wav"
    missing.write_bytes(b"not audio")
    controller = SubtitleController(
        context, SubtitleOptions(fmt="srt", output_dir=tmp_path, overwrite=True)
    )
    results = controller.run([missing, english_speech])

    assert results[0].error
    assert results[1].output and results[1].output.exists()


def test_subtitle_holds_the_exclusive_slot_only_while_running(context, english_speech, tmp_path):
    controller = SubtitleController(
        context, SubtitleOptions(fmt="srt", output_dir=tmp_path, overwrite=True)
    )
    controller.run([english_speech])
    assert context.coordinator.active() == ()


def test_an_invalid_format_is_rejected_up_front(context):
    with pytest.raises(ValueError, match="unknown format"):
        SubtitleController(context, SubtitleOptions(fmt="nope"))


# --- dictation ---------------------------------------------------------------


def test_dictation_records_transcribes_and_returns_text(context, chinese_speech, monkeypatch):
    sources = []

    def make(**_kwargs):
        source = FakeSource(chinese_speech, source="mic")
        sources.append(source)
        return source

    monkeypatch.setattr("localasr.apps.sources.MicrophoneSource", make)
    controller = DictationController(
        context, DictationOptions(deliver=False), listener=lambda _e: None
    )
    controller.start()
    assert controller.state == RECORDING
    sources[0].wait()
    text = controller.finish()

    assert text == "识别结果"
    assert controller.state == IDLE


def test_visible_dictation_can_finish_without_injecting_text(context, chinese_speech, monkeypatch):
    sources = []
    deliveries = []

    def make(**_kwargs):
        source = FakeSource(chinese_speech, source="mic")
        sources.append(source)
        return source

    monkeypatch.setattr("localasr.apps.sources.MicrophoneSource", make)
    monkeypatch.setattr(
        "localasr.apps.dictation.text_output.deliver",
        lambda text, **_kwargs: deliveries.append(text),
    )
    controller = DictationController(context, DictationOptions(deliver=True))
    controller.start()
    sources[0].wait()

    assert controller.finish(deliver=False) == "识别结果"
    assert deliveries == []
    assert controller.state == IDLE


def test_dictation_releases_its_slot_after_finishing(context, chinese_speech, monkeypatch):
    monkeypatch.setattr(
        "localasr.apps.sources.MicrophoneSource",
        lambda **kwargs: FakeSource(chinese_speech),
    )
    controller = DictationController(context, DictationOptions(deliver=False))
    controller.start()
    controller.finish()
    assert context.coordinator.active() == ()


def test_dictation_during_a_subtitle_job_reports_the_conflict(context, chinese_speech, monkeypatch):
    monkeypatch.setattr(
        "localasr.apps.sources.MicrophoneSource",
        lambda **kwargs: FakeSource(chinese_speech),
    )
    context.coordinator.acquire(Activity.SUBTITLE)
    controller = DictationController(context, DictationOptions(deliver=False))

    with pytest.raises(ActivityConflict):
        controller.start()
    controller.start(force=True)
    controller.finish()


def test_cancelling_dictation_discards_the_recording(context, chinese_speech, monkeypatch):
    monkeypatch.setattr(
        "localasr.apps.sources.MicrophoneSource",
        lambda **kwargs: FakeSource(chinese_speech),
    )
    controller = DictationController(context, DictationOptions(deliver=False))
    controller.start()
    controller.cancel()

    assert controller.state == IDLE
    assert context.coordinator.active() == ()


def test_repeated_dictation_cycles_leave_no_state_behind(context, chinese_speech, monkeypatch):
    """Continuous triggering must not leak slots, threads or text between cycles."""
    sources = []

    def make(**_kwargs):
        source = FakeSource(chinese_speech)
        sources.append(source)
        return source

    monkeypatch.setattr("localasr.apps.sources.MicrophoneSource", make)
    controller = DictationController(context, DictationOptions(deliver=False))
    for _ in range(5):
        controller.start()
        sources[-1].wait()
        assert controller.finish() == "识别结果"
        assert context.coordinator.active() == ()
    assert controller.state == IDLE


# --- meeting -----------------------------------------------------------------


def test_meeting_records_both_sources_on_one_timeline(
    context, chinese_speech, tmp_path, monkeypatch
):
    def fake_source(device=None, source="mic", **_kwargs):
        return FakeSource(chinese_speech, source=source)

    monkeypatch.setattr("localasr.apps.sources.MicrophoneSource", fake_source)
    monkeypatch.setattr("localasr.capture.overlap.probe", _no_overlap)
    monkeypatch.setattr(
        "localasr.capture.pulse.open_system_source",
        lambda _hint: (FakeSource(chinese_speech, source="system"), None),
    )

    journal = tmp_path / "m.jsonl"
    controller = MeetingController(context, MeetingOptions(), listener=lambda _e: None)
    session = controller.start(journal)
    time.sleep(0.5)
    finished = controller.stop()

    assert set(session.sources) == {"mic", "system"}
    assert finished.segments
    assert {s.source for s in finished.segments} <= {"mic", "system"}


def test_meeting_falls_back_to_microphone_only(context, chinese_speech, tmp_path, monkeypatch):
    """A missing monitor degrades the session; half a recording beats none."""

    def fake_source(device=None, source="mic", **_kwargs):
        return FakeSource(chinese_speech, source=source)

    monkeypatch.setattr("localasr.apps.sources.MicrophoneSource", fake_source)
    monkeypatch.setattr("localasr.capture.overlap.probe", _no_overlap)
    monkeypatch.setattr(
        "localasr.capture.pulse.open_system_source",
        lambda _hint: (None, "system audio unavailable (no monitor); recording microphone only"),
    )

    controller = MeetingController(context, MeetingOptions(), listener=lambda _e: None)
    session = controller.start(tmp_path / "m.jsonl")
    controller.stop()

    assert session.sources == ("mic",)
    assert any("microphone only" in w for w in session.warnings)


def test_meeting_journal_is_written_incrementally(context, chinese_speech, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "localasr.apps.sources.MicrophoneSource",
        lambda device=None, source="mic", **_k: FakeSource(chinese_speech, source=source),
    )
    monkeypatch.setattr("localasr.capture.overlap.probe", _no_overlap)
    journal = tmp_path / "m.jsonl"
    controller = MeetingController(
        context, MeetingOptions(capture_system=False), listener=lambda _e: None
    )
    controller.start(journal)
    time.sleep(0.5)
    controller.stop()

    rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["schema"] == 1
    assert any("text" in row for row in rows[1:])


def test_a_journal_left_by_a_crash_can_still_be_exported(tmp_path):
    journal = tmp_path / "crashed.jsonl"
    journal.write_text(
        json.dumps({"schema": 1, "started_at": "2026-08-10T10:00:00", "sources": ["mic"]})
        + "\n"
        + json.dumps({"start": 0.0, "end": 1.0, "source": "mic", "text": "第一句"})
        + "\n"
        + '{"start": 1.5, "end":',  # torn write
        encoding="utf-8",
    )
    session = load_journal(journal)
    assert [s.text for s in session.segments] == ["第一句"]


def test_default_journal_path_is_timestamped(tmp_path):
    first = default_journal_path(tmp_path)
    assert first.suffix == ".jsonl"
    assert first.parent == tmp_path


# --- text delivery -----------------------------------------------------------


def test_delivery_falls_back_to_the_clipboard_when_typing_is_unavailable(monkeypatch):
    """Text on the clipboard is recoverable; text that vanished is not."""
    monkeypatch.setattr(text_output, "available_methods", lambda: ["clipboard"])
    monkeypatch.setattr(text_output, "copy_to_clipboard", lambda _text: True)

    delivery = text_output.deliver("你好")
    assert delivery.method == "clipboard"
    assert not delivery.pasted
    assert "Ctrl+V" in delivery.detail


def test_delivery_prefers_ydotool_when_available(monkeypatch):
    monkeypatch.setattr(text_output, "available_methods", lambda: ["ydotool", "clipboard"])
    monkeypatch.setattr(text_output, "_type_with_ydotool", lambda _text: True)

    assert text_output.deliver("你好").method == "ydotool"


def test_a_failing_typer_falls_through_to_the_next_method(monkeypatch):
    monkeypatch.setattr(text_output, "available_methods", lambda: ["ydotool", "clipboard"])
    monkeypatch.setattr(text_output, "_type_with_ydotool", lambda _text: False)
    monkeypatch.setattr(text_output, "copy_to_clipboard", lambda _text: True)

    assert text_output.deliver("你好").method == "clipboard"


def test_delivery_raises_only_when_even_the_clipboard_is_gone(monkeypatch):
    monkeypatch.setattr(text_output, "available_methods", lambda: [])
    monkeypatch.setattr(text_output, "copy_to_clipboard", lambda _text: False)

    with pytest.raises(text_output.TextOutputError, match="ydotool"):
        text_output.deliver("你好")


def test_empty_text_is_not_delivered(monkeypatch):
    assert text_output.deliver("").method == "none"


def test_summary_says_dictation_cannot_deliver_on_a_bare_system(monkeypatch):
    monkeypatch.setattr(text_output, "available_methods", lambda: [])
    assert "localasr doctor" in text_output.summary()


def test_blocks_from_file_stamps_a_monotonic_sequence(chinese_speech):
    from localasr.capture.microphone import blocks_from_file

    blocks = list(blocks_from_file(chinese_speech, block_seconds=0.5))
    assert [b.sequence for b in blocks] == list(range(len(blocks)))
    assert all(isinstance(b, AudioBlock) for b in blocks)
    assert all(b.samples.dtype == np.float32 for b in blocks)
    stamps = [b.captured_at for b in blocks]
    assert stamps == sorted(stamps)
    assert blocks[0].sample_rate == SAMPLE_RATE


def test_an_utterance_finishing_mid_recording_does_not_end_the_recording(
    context, chinese_speech, monkeypatch
):
    """`recording` follows the capture session, not the display label.

    Deriving it from `state` made it flip to false the moment any utterance completed,
    while the microphone was still open — so the toggle offered to *start* again and
    the next press opened a second stream on the same device, which aborts the process
    inside PortAudio with no Python traceback.
    """
    sources = []

    def make(**_kwargs):
        source = FakeSource(chinese_speech)
        sources.append(source)
        return source

    monkeypatch.setattr("localasr.apps.sources.MicrophoneSource", make)
    controller = DictationController(context, DictationOptions(deliver=False))
    controller.start()
    sources[0].wait()
    time.sleep(0.3)  # let a FinalTranscript land while still recording

    assert controller.recording, "recording ended because an utterance completed"
    assert controller.current_text == "识别结果"
    controller.finish()
    assert not controller.recording


def test_starting_twice_never_opens_a_second_capture_stream(context, chinese_speech, monkeypatch):
    sources = []

    def make(**_kwargs):
        source = FakeSource(chinese_speech)
        sources.append(source)
        return source

    monkeypatch.setattr("localasr.apps.sources.MicrophoneSource", make)
    controller = DictationController(context, DictationOptions(deliver=False))
    controller.start()
    controller.start()
    controller.start()
    try:
        assert len(sources) == 1
        assert len(context.coordinator.active()) == 1
    finally:
        controller.finish()


def test_toggle_stops_after_an_utterance_has_already_completed(
    context, chinese_speech, monkeypatch
):
    """The exact user sequence: speak a sentence, then press stop."""
    sources = []

    def make(**_kwargs):
        source = FakeSource(chinese_speech)
        sources.append(source)
        return source

    monkeypatch.setattr("localasr.apps.sources.MicrophoneSource", make)
    controller = DictationController(context, DictationOptions(deliver=False))
    assert controller.toggle() is None
    sources[0].wait()
    time.sleep(0.3)

    assert controller.toggle() == "识别结果"
    assert len(sources) == 1, "stop started a second recording instead of finishing"
    assert context.coordinator.active() == ()


def test_start_prepares_the_engine_before_opening_the_device(context, chinese_speech, monkeypatch):
    """Every caller gets this, not just the GUI. The CLI and the global hotkey call
    `start()` directly; without it the model loads lazily during the first
    transcription — while the user is already talking — and a missing model surfaces
    mid-session instead of before it."""
    order = []

    def prepare():
        order.append("prepare")

    def make(**_kwargs):
        order.append("open-device")
        return FakeSource(chinese_speech)

    monkeypatch.setattr(context, "hold_engine", prepare)
    monkeypatch.setattr("localasr.apps.sources.MicrophoneSource", make)

    controller = DictationController(context, DictationOptions(deliver=False))
    controller.start()
    try:
        assert order == ["prepare", "open-device"]
    finally:
        controller.finish()


def test_a_failed_preparation_never_opens_the_microphone(context, monkeypatch):
    opened = []

    def make(**_kwargs):
        opened.append(1)
        raise AssertionError("device opened despite the engine being unavailable")

    monkeypatch.setattr(
        context, "hold_engine", lambda: (_ for _ in ()).throw(RuntimeError("模型尚未准备好"))
    )
    monkeypatch.setattr("localasr.apps.sources.MicrophoneSource", make)

    controller = DictationController(context, DictationOptions(deliver=False))
    with pytest.raises(RuntimeError, match="模型"):
        controller.start()

    assert opened == []
    assert not controller.recording
    assert context.coordinator.active() == (), "the activity slot leaked on failure"


def test_preparation_is_reported_as_its_own_state(context, chinese_speech, monkeypatch):
    """Loading the model is not recognising anything; showing it as 「正在识别」 tells
    the user the wrong thing about what the wait is for."""
    from localasr.apps.dictation import PREPARING
    from localasr.apps.events import DictationStateChanged

    seen = []
    monkeypatch.setattr(context, "hold_engine", lambda: None)
    monkeypatch.setattr(
        "localasr.apps.sources.MicrophoneSource", lambda **_k: FakeSource(chinese_speech)
    )
    controller = DictationController(
        context,
        DictationOptions(deliver=False),
        listener=lambda e: seen.append(e) if isinstance(e, DictationStateChanged) else None,
    )
    controller.start()
    try:
        states = [e.state for e in seen]
        assert states[0] == PREPARING
        assert states.index(PREPARING) < states.index(RECORDING)
    finally:
        controller.finish()


def test_settled_text_is_published_while_still_recording(context, chinese_speech, monkeypatch):
    """The live view renders this: each sentence appears as it is recognised, rather
    than being withheld until the user stops."""
    from localasr.apps.events import DictationStateChanged

    updates = []
    monkeypatch.setattr(context, "hold_engine", lambda: None)
    sources = []

    def make(**_kwargs):
        source = FakeSource(chinese_speech)
        sources.append(source)
        return source

    monkeypatch.setattr("localasr.apps.sources.MicrophoneSource", make)
    controller = DictationController(
        context,
        DictationOptions(deliver=False),
        listener=lambda e: (
            updates.append(e.detail)
            if isinstance(e, DictationStateChanged) and e.state == RECORDING and e.detail
            else None
        ),
    )
    controller.start()
    sources[0].wait()
    time.sleep(0.4)
    try:
        assert any(u == "识别结果" for u in updates), "no interim text was published"
        assert controller.recording
    finally:
        controller.finish()


def test_the_engine_is_held_for_the_whole_dictation_session(context, chinese_speech, monkeypatch):
    """A live session only touches the engine when an utterance completes, so between
    sentences it looks idle. Without a lease the reaper unloads it mid-recording and
    the next sentence pays a full model load — the very thing preparing avoided."""
    events = []
    monkeypatch.setattr(context, "hold_engine", lambda: events.append("hold"))
    monkeypatch.setattr(context, "release_engine", lambda: events.append("release"))
    monkeypatch.setattr(
        "localasr.apps.sources.MicrophoneSource", lambda **_k: FakeSource(chinese_speech)
    )

    controller = DictationController(context, DictationOptions(deliver=False))
    controller.start()
    assert events == ["hold"], "engine was not held while recording"
    controller.finish()
    assert events == ["hold", "release"]


def test_cancelling_also_releases_the_engine(context, chinese_speech, monkeypatch):
    events = []
    monkeypatch.setattr(context, "hold_engine", lambda: events.append("hold"))
    monkeypatch.setattr(context, "release_engine", lambda: events.append("release"))
    monkeypatch.setattr(
        "localasr.apps.sources.MicrophoneSource", lambda **_k: FakeSource(chinese_speech)
    )

    controller = DictationController(context, DictationOptions(deliver=False))
    controller.start()
    controller.cancel()
    assert events == ["hold", "release"]


def test_a_device_failure_does_not_leak_the_engine_lease(context, monkeypatch):
    from localasr.capture.microphone import CaptureError

    events = []
    monkeypatch.setattr(context, "hold_engine", lambda: events.append("hold"))
    monkeypatch.setattr(context, "release_engine", lambda: events.append("release"))

    def broken(**_kwargs):
        raise CaptureError("no microphone")

    monkeypatch.setattr("localasr.apps.sources.MicrophoneSource", broken)
    controller = DictationController(context, DictationOptions(deliver=False))
    with pytest.raises(CaptureError):
        controller.start()

    assert events == ["hold", "release"], "the lease outlived the failed session"
    assert context.coordinator.active() == ()
