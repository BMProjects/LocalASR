"""Core data types shared by every stage of the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

SAMPLE_RATE = 16_000
"""Every stage downstream of decoding operates at this rate."""

Waveform = NDArray[np.float32]
"""Mono PCM, shape (n_samples,), dtype float32, nominally in [-1.0, 1.0]."""

SourceTag = Literal["mic", "system"]
"""Which capture path a segment came from; drives the 我/对方 labelling."""


@dataclass(frozen=True, slots=True)
class Audio:
    """A decoded mono waveform."""

    samples: Waveform
    sample_rate: int = SAMPLE_RATE

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate


@dataclass(frozen=True, slots=True)
class AudioBlock:
    """A captured buffer stamped on a clock shared by every source in a session.

    Two capture threads never start at the same instant, so timing a block by counting
    the samples each source has produced makes the microphone and system timelines
    incomparable — which is exactly what a meeting recording needs them to be. The
    monotonic stamp is the shared reference; `sequence` exposes gaps that the stamp
    alone would silently absorb.
    """

    samples: Waveform
    captured_at: float
    sequence: int
    source: SourceTag = "mic"
    sample_rate: int = SAMPLE_RATE

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate


@dataclass(frozen=True, slots=True)
class Span:
    """A half-open time interval [start, end) in seconds from the start of the stream."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class Word:
    """A word (Latin scripts) or character (CJK) with its alignment."""

    text: str
    start: float
    end: float


@dataclass(slots=True)
class Segment:
    """One VAD span plus whatever the engine transcribed for it.

    `utterance_id` is stable for the life of an utterance, so a final result can
    replace the partial it supersedes. Matching on `span` cannot do that: the span
    keeps changing while the utterance is still open.
    """

    span: Span
    text: str
    source: SourceTag | None = None
    words: list[Word] = field(default_factory=list)
    utterance_id: str | None = None

    @property
    def start(self) -> float:
        return self.span.start

    @property
    def end(self) -> float:
        return self.span.end


@dataclass(slots=True)
class Transcript:
    """The full result for one audio input."""

    segments: list[Segment]
    duration: float
    language: str | None = None

    @property
    def text(self) -> str:
        """One utterance per line; avoids guessing whether the language needs word spacing."""
        return "\n".join(s.text for s in self.segments)
