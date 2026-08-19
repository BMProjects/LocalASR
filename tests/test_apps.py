"""Application-layer behaviour, exercised against a stub engine.

The offline job and the live session share domain objects but not execution flow, so
each is tested for what only it must do: ordering, resumption and cancellation offline;
utterance boundaries, source tagging and shutdown ordering live.
"""

import json

import pytest

from localasr.apps.events import (
    FinalTranscript,
    JobCancelled,
    JobFinished,
    JobStarted,
    SegmentDropped,
    SegmentResumed,
    SegmentTranscribed,
    SessionStarted,
    SessionStopped,
)
from localasr.apps.live import LiveOptions, LiveSession, SessionError
from localasr.apps.offline import JobCancelledError, OfflineJob, OfflineOptions
from localasr.capture.microphone import blocks_from_file
from localasr.core.audio.decode import decode_file, to_wav_bytes
from localasr.core.audio.vad import SileroVad, VadConfig
from localasr.core.engine.client import TranscriptionResult
from localasr.core.types import SAMPLE_RATE, Audio
from localasr.registry import manager


class StubClient:
    """Returns canned text and records what it was asked to transcribe."""

    def __init__(self, texts=None):
        self.texts = list(texts) if texts else None
        self.calls = []

    def transcribe(self, audio, *, language=None, **_kwargs):
        self.calls.append((audio.duration, language))
        if self.texts:
            return TranscriptionResult(text=self.texts.pop(0), language="Chinese")
        return TranscriptionResult(text=f"seg{len(self.calls)}", language="Chinese")


class StubEngine:
    def __init__(self, client, spec=None):
        self.client = client
        self.spec = spec or manager.get_model()
        self.acquired = 0
        self.released = 0
        self.warning = None

    def acquire(self):
        self.acquired += 1
        return self.client

    def release(self):
        self.released += 1

    def hold(self):
        self.held = getattr(self, "held", 0) + 1

    def release_hold(self):
        self.held = getattr(self, "held", 0) - 1


@pytest.fixture
def vad():
    return SileroVad(manager.vad_path())


@pytest.fixture
def speech_file(english_speech):
    """Real speech: Silero is trained on it, and synthetic noise is correctly
    classified as non-speech, which would make these tests assert on empty input."""
    return english_speech


def _blocks(path):
    return blocks_from_file(path, block_seconds=0.25)


# --- offline -----------------------------------------------------------------


def test_offline_job_emits_start_progress_and_finish(speech_file, vad):
    events = []
    OfflineJob(speech_file, StubEngine(StubClient()), vad).run(listener=events.append)

    assert isinstance(events[0], JobStarted)
    assert isinstance(events[-1], JobFinished)
    assert any(isinstance(e, SegmentTranscribed) for e in events)


def test_offline_job_always_releases_the_engine(speech_file, vad):
    engine = StubEngine(StubClient())
    OfflineJob(speech_file, engine, vad).run()
    assert engine.acquired == engine.released == 1


def test_offline_job_releases_the_engine_when_transcription_fails(speech_file, vad):
    class Exploding(StubClient):
        def transcribe(self, audio, **kwargs):
            raise RuntimeError("engine died")

    engine = StubEngine(Exploding())
    with pytest.raises(RuntimeError):
        OfflineJob(speech_file, engine, vad).run()
    assert engine.released == 1


def test_offline_job_reports_dropped_segments_rather_than_hiding_them(speech_file, vad):
    engine = StubEngine(StubClient(texts=["好的好的好的好的好的好的", "正常的一句话"]))
    events = []
    transcript = OfflineJob(speech_file, engine, vad).run(listener=events.append)

    dropped = [e for e in events if isinstance(e, SegmentDropped)]
    assert len(dropped) == 1
    assert "repetition" in dropped[0].reason
    assert all("好的好的" not in s.text for s in transcript.segments)


def test_offline_job_forwards_the_forced_language(speech_file, vad):
    client = StubClient()
    OfflineJob(speech_file, StubEngine(client), vad, OfflineOptions(language="zh")).run()
    assert all(language == "zh" for _duration, language in client.calls)


def test_offline_segments_are_ordered_in_time(speech_file, vad):
    transcript = OfflineJob(speech_file, StubEngine(StubClient()), vad).run()
    starts = [s.start for s in transcript.segments]
    assert starts == sorted(starts)


def test_cancelling_stops_the_job_and_reports_what_was_done(speech_file, vad):
    engine = StubEngine(StubClient())
    job = OfflineJob(speech_file, engine, vad)
    events = []

    class CancelAfterFirst(StubClient):
        def transcribe(self, audio, **kwargs):
            result = super().transcribe(audio, **kwargs)
            job.cancel()
            return result

    engine.client = CancelAfterFirst()
    with pytest.raises(JobCancelledError):
        job.run(listener=events.append)

    cancelled = [e for e in events if isinstance(e, JobCancelled)]
    assert cancelled and cancelled[0].completed == 1
    assert engine.released == 1


# --- journal / resume --------------------------------------------------------


def test_a_second_run_resumes_instead_of_duplicating(speech_file, vad, tmp_path):
    """Appending alone is not resumption: without the header check, re-running would
    append a second copy of every segment."""
    journal = tmp_path / "j.jsonl"
    engine = StubEngine(StubClient())
    first = OfflineJob(speech_file, engine, vad).run(journal_path=journal)

    engine2 = StubEngine(StubClient())
    events = []
    second = OfflineJob(speech_file, engine2, vad).run(listener=events.append, journal_path=journal)

    assert [s.text for s in second.segments] == [s.text for s in first.segments]
    assert engine2.client.calls == []
    assert len([e for e in events if isinstance(e, SegmentResumed)]) == len(first.segments)


def test_journal_is_discarded_when_the_language_changes(speech_file, vad, tmp_path):
    journal = tmp_path / "j.jsonl"
    OfflineJob(speech_file, StubEngine(StubClient()), vad).run(journal_path=journal)

    engine = StubEngine(StubClient())
    OfflineJob(speech_file, engine, vad, OfflineOptions(language="zh")).run(journal_path=journal)
    assert engine.client.calls, "a different language must re-transcribe, not resume"


def test_journal_is_discarded_when_the_media_changes(speech_file, chinese_speech, vad, tmp_path):
    journal = tmp_path / "j.jsonl"
    OfflineJob(speech_file, StubEngine(StubClient()), vad).run(journal_path=journal)

    engine = StubEngine(StubClient())
    OfflineJob(chinese_speech, engine, vad).run(journal_path=journal)
    assert engine.client.calls, "a different file must re-transcribe, not resume"


def test_resume_can_be_disabled(speech_file, vad, tmp_path):
    journal = tmp_path / "j.jsonl"
    OfflineJob(speech_file, StubEngine(StubClient()), vad).run(journal_path=journal)

    engine = StubEngine(StubClient())
    OfflineJob(speech_file, engine, vad, OfflineOptions(resume=False)).run(journal_path=journal)
    assert engine.client.calls


def test_journal_rows_match_the_transcript(speech_file, vad, tmp_path):
    journal = tmp_path / "j.jsonl"
    transcript = OfflineJob(speech_file, StubEngine(StubClient()), vad).run(journal_path=journal)

    from localasr.apps.journal import SCHEMA_VERSION

    rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["schema"] == SCHEMA_VERSION
    assert len(rows) - 1 == len(transcript.segments)
    assert rows[1]["text"] == transcript.segments[0].text
    # Rows are typed now that refinements share the file.
    assert rows[1]["type"] == "segment"


# --- live --------------------------------------------------------------------


def test_live_session_tags_each_utterance_with_its_source(speech_file):
    session = LiveSession(
        StubEngine(StubClient()), lambda: SileroVad(manager.vad_path()), LiveOptions()
    )
    events = []
    session.start(listener=events.append, sources=("system",))
    session.feed(_blocks(speech_file), tag="system")
    session.stop()

    finals = [e for e in events if isinstance(e, FinalTranscript)]
    assert finals
    assert all(e.segment.source == "system" for e in finals)


def test_live_session_emits_session_lifecycle_events(speech_file):
    session = LiveSession(StubEngine(StubClient()), lambda: SileroVad(manager.vad_path()))
    events = []
    session.start(listener=events.append)
    session.feed(_blocks(speech_file))
    count = session.stop()

    assert isinstance(events[0], SessionStarted)
    stopped = [e for e in events if isinstance(e, SessionStopped)]
    assert stopped and stopped[0].utterances == count


def test_live_utterances_carry_a_stable_unique_id(speech_file):
    """Finals must be matchable to their partials by id, not by span."""
    session = LiveSession(StubEngine(StubClient()), lambda: SileroVad(manager.vad_path()))
    events = []
    session.start(listener=events.append)
    session.feed(_blocks(speech_file))
    session.stop()

    ids = [e.segment.utterance_id for e in events if isinstance(e, FinalTranscript)]
    assert all(ids)
    assert len(set(ids)) == len(ids)


def test_live_session_utterances_advance_in_time(speech_file):
    session = LiveSession(StubEngine(StubClient()), lambda: SileroVad(manager.vad_path()))
    events = []
    session.start(listener=events.append)
    session.feed(_blocks(speech_file))
    session.stop()

    starts = [e.segment.start for e in events if isinstance(e, FinalTranscript)]
    assert starts == sorted(starts)


def test_background_feed_is_joined_before_the_worker_stops(speech_file):
    """The capture thread must finish first: an utterance queued after the sentinel
    would never be read."""
    session = LiveSession(StubEngine(StubClient()), lambda: SileroVad(manager.vad_path()))
    seen = []
    session.start(on_utterance=seen.append)
    session.feed_in_background(_blocks(speech_file))
    count = session.stop()

    assert count > 0
    assert count == len(seen)


def test_live_session_flush_closes_an_utterance_still_open_at_the_end(chinese_speech, tmp_path):
    """Speech running to the last sample must still be transcribed — exactly the
    push-to-talk case, where the key is released while the user is still talking."""
    audio = decode_file(chinese_speech)
    truncated = tmp_path / "held.wav"
    truncated.write_bytes(to_wav_bytes(Audio(samples=audio.samples[: int(4.5 * SAMPLE_RATE)])))

    session = LiveSession(
        StubEngine(StubClient()), lambda: SileroVad(manager.vad_path(), VadConfig())
    )
    events = []
    session.start(listener=events.append)
    session.feed(_blocks(truncated))
    session.stop()

    assert [e for e in events if isinstance(e, FinalTranscript)]


def test_starting_a_session_twice_is_refused(speech_file):
    session = LiveSession(StubEngine(StubClient()), lambda: SileroVad(manager.vad_path()))
    session.start()
    with pytest.raises(SessionError):
        session.start()
    session.stop()


def test_stopping_a_session_that_never_started_is_a_no_op():
    session = LiveSession(StubEngine(StubClient()), lambda: SileroVad(manager.vad_path()))
    assert session.stop() == 0


def test_transcription_request_times_out_before_live_shutdown():
    from localasr.apps.live import STOP_TIMEOUT
    from localasr.core.engine.client import TRANSCRIPTION_TIMEOUT

    assert TRANSCRIPTION_TIMEOUT < STOP_TIMEOUT


def test_an_utterance_is_recorded_before_it_is_announced(speech_file):
    """Subscribers identify their own utterances by looking them up in the controller's
    record (a meeting must ignore a forced dictation). Announcing first would make that
    lookup a race, dropping lines from the live view when the subscriber wins it."""
    recorded: list[str] = []
    seen_when_announced: list[bool] = []

    def on_utterance(segment):
        recorded.append(segment.utterance_id)

    def listener(event):
        if isinstance(event, FinalTranscript):
            seen_when_announced.append(event.segment.utterance_id in recorded)

    session = LiveSession(StubEngine(StubClient()), lambda: SileroVad(manager.vad_path()))
    session.start(listener=listener, on_utterance=on_utterance)
    session.feed(_blocks(speech_file))
    session.stop()

    assert seen_when_announced
    assert all(seen_when_announced)


# --- nothing disappears silently ---------------------------------------------


def test_a_rejected_utterance_is_reported_not_swallowed(speech_file):
    """A sentence that vanishes with no explanation is indistinguishable from audio
    the pipeline lost — which is how a filter turns into a bug report."""
    from localasr.apps.events import SegmentDropped

    session = LiveSession(
        StubEngine(StubClient(texts=["好的好的好的好的好的好的"] * 8)),
        lambda: SileroVad(manager.vad_path()),
    )
    events = []
    session.start(listener=events.append)
    session.feed(_blocks(speech_file))
    session.stop()

    dropped = [e for e in events if isinstance(e, SegmentDropped)]
    assert dropped, "the repetition filter dropped an utterance without saying so"
    assert "repetition" in dropped[0].reason
    assert dropped[0].span.duration > 0


def test_capture_overruns_are_reported(speech_file):
    """The device dropped audio before we ever saw it; only its own counter knows."""
    from localasr.apps.events import AudioDropped

    class Overrunning:
        """Reports a growing overrun count, as MicrophoneSource does under load."""

        overruns = 0

        def __iter__(self):
            for index, block in enumerate(_blocks(speech_file)):
                if index == 3:
                    Overrunning.overruns += 5
                yield block

    session = LiveSession(StubEngine(StubClient()), lambda: SileroVad(manager.vad_path()))
    events = []
    session.start(listener=events.append)
    session.feed(Overrunning())
    session.stop()

    drops = [e for e in events if isinstance(e, AudioDropped) and "overran" in e.reason]
    assert drops, "a capture overrun was never surfaced"
    assert drops[0].seconds > 0


def test_short_answers_survive_the_live_preset():
    """「对」/「嗯」/"yes" are whole utterances in dictation; 0.25 s dropped them."""
    from localasr.core.audio.vad import VadConfig

    assert VadConfig.for_live().min_speech < 0.25
    assert VadConfig.for_live().min_speech >= 0.1
