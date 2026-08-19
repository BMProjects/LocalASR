"""Events published by jobs and sessions.

Frontends subscribe to these; they do not drive audio, models or subprocesses. Keeping
the vocabulary here means an application can gain a GUI without the GUI gaining
knowledge of the engine — these are plain dataclasses, so `apps/` never imports Qt.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from localasr.core.types import Segment, SourceTag, Span, Transcript


@dataclass(frozen=True, slots=True)
class JobStarted:
    total_segments: int | None
    duration: float | None
    media: Path | None = None


@dataclass(frozen=True, slots=True)
class SegmentTranscribed:
    """A finished segment of an offline job."""

    index: int
    total: int
    segment: Segment


@dataclass(frozen=True, slots=True)
class SegmentDropped:
    """A segment the post-processing rejected, reported so it is not silent."""

    index: int
    span: Span
    text: str
    reason: str


@dataclass(frozen=True, slots=True)
class SegmentResumed:
    """A segment restored from a previous run's journal rather than re-transcribed."""

    index: int
    segment: Segment


@dataclass(frozen=True, slots=True)
class JobFinished:
    transcript: Transcript
    media: Path | None = None


@dataclass(frozen=True, slots=True)
class JobCancelled:
    completed: int
    media: Path | None = None


@dataclass(frozen=True, slots=True)
class JobFailed:
    error: str
    media: Path | None = None


@dataclass(frozen=True, slots=True)
class SessionStarted:
    sources: tuple[SourceTag, ...]


@dataclass(frozen=True, slots=True)
class SessionStopping:
    pass


@dataclass(frozen=True, slots=True)
class SessionStopped:
    utterances: int


@dataclass(frozen=True, slots=True)
class SessionCancelled:
    reason: str


@dataclass(frozen=True, slots=True)
class SpeechStarted:
    """The live segmenter opened an utterance; drives the recording indicator."""

    at: float
    source: SourceTag


@dataclass(frozen=True, slots=True)
class SpeechEnded:
    at: float
    source: SourceTag
    utterance_id: str


@dataclass(frozen=True, slots=True)
class AudioDropped:
    """Audio that never reached the segmenter.

    Reported rather than absorbed: silently losing a buffer shifts every later
    timestamp, which is indistinguishable from the speaker simply pausing.
    """

    at: float
    seconds: float
    source: SourceTag
    reason: str


@dataclass(frozen=True, slots=True)
class PartialTranscript:
    """Provisional text for an utterance still in progress.

    Nothing emits this yet: the current engine transcribes whole utterances only. It
    exists so a low-latency first pass can be added without changing subscribers —
    subscribers key on `utterance_id`, which the final result repeats.
    """

    utterance_id: str
    span: Span
    text: str
    source: SourceTag


@dataclass(frozen=True, slots=True)
class FinalTranscript:
    """Settled text for one utterance; replaces any partial with the same id."""

    segment: Segment

    @property
    def utterance_id(self) -> str | None:
        return self.segment.utterance_id


@dataclass(frozen=True, slots=True)
class DictationStateChanged:
    """Idle / recording / transcribing / failed, for the tray indicator."""

    state: str
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class TextEmitted:
    """Dictation produced text and delivered it (or failed to)."""

    text: str
    method: str


@dataclass(frozen=True, slots=True)
class SessionFailed:
    error: str


@dataclass(frozen=True, slots=True)
class ModelChanged:
    model_id: str
    revision: str


@dataclass(frozen=True, slots=True)
class ModelDownloaded:
    model_id: str
    revision: str


Event = (
    JobStarted
    | SegmentTranscribed
    | SegmentDropped
    | SegmentResumed
    | JobFinished
    | JobCancelled
    | JobFailed
    | SessionStarted
    | SessionStopping
    | SessionStopped
    | SessionCancelled
    | SpeechStarted
    | SpeechEnded
    | AudioDropped
    | PartialTranscript
    | FinalTranscript
    | DictationStateChanged
    | TextEmitted
    | SessionFailed
    | ModelChanged
    | ModelDownloaded
)
