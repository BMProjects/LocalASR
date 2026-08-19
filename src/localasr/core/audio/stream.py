"""Incremental segmentation for live capture.

Offline segmentation can look at the whole probability array before deciding where
utterances end. A live session cannot: it must commit to a boundary as soon as the
speaker pauses, and it must already hold the audio from *before* the VAD triggered, or
every utterance loses its first consonant. Those are different semantics, not a
parameterisation of the offline path, so this is a separate state machine.

Timing comes from the block stamps rather than from a sample count, so an utterance's
span is comparable across capture sources even when the sources started at different
instants or one of them dropped a buffer.
"""

from __future__ import annotations

import uuid
from collections import deque
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np

from localasr.core.audio.vad import WINDOW_SAMPLES, SileroVad, VadConfig
from localasr.core.types import SAMPLE_RATE, Audio, AudioBlock, SourceTag, Span, Waveform

DRIFT_TOLERANCE = 0.25
"""How far the capture offset may grow beyond its best-case value before the audio is
treated as lost. Compared against a running minimum, not against zero — see
`_detect_gap`."""


@dataclass(frozen=True, slots=True)
class Utterance:
    """One finalized stretch of speech, timed against the session epoch."""

    audio: Audio
    span: Span
    source: SourceTag = "mic"
    utterance_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass(frozen=True, slots=True)
class Gap:
    """Audio the segmenter never received, inferred from the block stamps."""

    at: float
    seconds: float
    source: SourceTag
    reason: str


class StreamingSegmenter:
    """Feed it AudioBlocks; it hands back utterances as they complete.

    `pre_roll` is the amount of audio kept before the trigger point. Silero needs a few
    windows above threshold to fire, so without it every utterance starts clipped.
    """

    def __init__(
        self,
        vad: SileroVad,
        config: VadConfig | None = None,
        *,
        pre_roll: float = 0.30,
        epoch: float | None = None,
    ) -> None:
        self._vad = vad
        self.config = config or vad.config
        self._pre_roll_windows = max(1, int(pre_roll * SAMPLE_RATE / WINDOW_SAMPLES))
        self.epoch = epoch

        self._history: deque[tuple[Waveform, float]] = deque(maxlen=self._pre_roll_windows)
        self._buffer: list[Waveform] = []
        self._triggered = False
        self._start_time = 0.0
        self._silence_samples = 0
        self._samples_seen = 0
        self._clock = 0.0
        self._carry = np.zeros(0, dtype=np.float32)
        self._carry_time = 0.0
        self._next_sequence: int | None = None
        self._base_offset: float | None = None
        self.source: SourceTag = "mic"

    @property
    def is_speaking(self) -> bool:
        """True while an utterance is open; drives the recording indicator."""
        return self._triggered

    @property
    def position(self) -> float:
        """Session time of the most recently consumed audio."""
        return self._clock

    def reset(self) -> None:
        self._vad.reset()
        self._history.clear()
        self._buffer.clear()
        self._triggered = False
        self._start_time = 0.0
        self._silence_samples = 0
        self._samples_seen = 0
        self._clock = 0.0
        self._carry = np.zeros(0, dtype=np.float32)
        self._next_sequence = None
        self._base_offset = None

    def push_block(self, block: AudioBlock) -> tuple[list[Utterance], Gap | None]:
        """Consume one captured block.

        Returns the utterances that completed within it and, when the stamps show more
        elapsed time than samples received, the gap that explains the difference.
        """
        if self.epoch is None:
            self.epoch = block.captured_at
        self.source = block.source

        gap = self._detect_gap(block)
        if gap is not None:
            # The stream is no longer contiguous; close whatever was open rather than
            # splicing across missing audio.
            self._carry = np.zeros(0, dtype=np.float32)

        block_time = block.captured_at - self.epoch
        if not self._carry.size:
            self._carry_time = block_time
        self._carry = np.concatenate((self._carry, block.samples))
        self._samples_seen += len(block.samples)

        utterances: list[Utterance] = []
        usable = len(self._carry) - (len(self._carry) % WINDOW_SAMPLES)
        for index in range(0, usable, WINDOW_SAMPLES):
            window = self._carry[index : index + WINDOW_SAMPLES]
            at = self._carry_time + index / SAMPLE_RATE
            done = self._push_window(window, at)
            if done is not None:
                utterances.append(done)
        self._carry_time += usable / SAMPLE_RATE
        self._carry = self._carry[usable:]
        return utterances, gap

    def _detect_gap(self, block: AudioBlock) -> Gap | None:
        """Distinguish audio that was lost from audio that merely arrived late.

        A block is stamped when its callback runs, which is always after the samples
        were captured — PortAudio's buffering alone puts this in the hundreds of
        milliseconds. That offset is latency, not loss, and comparing the stamp against
        a zero baseline reports it as a gap on every single block.

        What identifies real loss is the offset *growing*: latency stays near its
        best-case value, whereas a dropped buffer permanently shifts the stream. So the
        baseline is the smallest offset seen so far, and only excess over it is a gap.
        """
        expected_sequence = self._next_sequence
        self._next_sequence = block.sequence + 1
        if expected_sequence is None or self.epoch is None:
            return None

        received = self._samples_seen / SAMPLE_RATE
        offset = (block.captured_at - self.epoch) - received
        if self._base_offset is None or offset < self._base_offset:
            self._base_offset = offset
        excess = offset - self._base_offset

        if block.sequence != expected_sequence:
            return Gap(
                at=received,
                seconds=max(0.0, excess),
                source=block.source,
                reason=f"sequence jumped {expected_sequence} -> {block.sequence}",
            )
        if excess > DRIFT_TOLERANCE:
            # Absorb the step so a single loss is reported once, not on every block after.
            self._base_offset = offset
            return Gap(at=received, seconds=excess, source=block.source, reason="clock drift")
        return None

    def _push_window(self, window: Waveform, at: float) -> Utterance | None:
        cfg = self.config
        prob = self._vad.push(window)
        self._clock = at + len(window) / SAMPLE_RATE

        if not self._triggered:
            self._history.append((window, at))
            if prob >= cfg.threshold:
                self._triggered = True
                self._silence_samples = 0
                held = list(self._history)
                self._buffer = [chunk for chunk, _ in held]
                self._start_time = held[0][1]
                self._history.clear()
            return None

        self._buffer.append(window)

        if prob < cfg.resolved_neg_threshold():
            self._silence_samples += len(window)
            # The bar for "this pause ended the utterance" drops as the buffer grows,
            # so continuous speech is still cut at a pause rather than mid-word.
            buffered = self._clock - self._start_time
            if self._silence_samples >= cfg.silence_for(buffered) * SAMPLE_RATE:
                return self._finalize(trailing_silence=self._silence_samples)
        else:
            self._silence_samples = 0

        if (self._clock - self._start_time) >= cfg.max_speech:
            return self._finalize(trailing_silence=0)
        return None

    def flush(self) -> Utterance | None:
        """Close an open utterance, e.g. when the user releases the dictation key."""
        if not self._triggered:
            return None
        return self._finalize(trailing_silence=self._silence_samples)

    def _finalize(self, *, trailing_silence: int) -> Utterance | None:
        samples = np.concatenate(self._buffer) if self._buffer else np.zeros(0, dtype=np.float32)
        # Keep a little of the trailing silence: the decoder handles a short tail better
        # than a hard cut on the final phoneme.
        keep = max(0, len(samples) - max(0, trailing_silence - int(0.2 * SAMPLE_RATE)))
        samples = samples[:keep]

        start = self._start_time
        self._buffer = []
        self._triggered = False
        self._silence_samples = 0
        self._history.clear()

        duration = len(samples) / SAMPLE_RATE
        if duration < self.config.min_speech:
            return None
        return Utterance(
            audio=Audio(samples=samples.astype(np.float32), sample_rate=SAMPLE_RATE),
            span=Span(start, start + duration),
            source=self.source,
        )


def windows(samples: Waveform, size: int = WINDOW_SAMPLES) -> list[Waveform]:
    """Split a buffer into VAD-sized windows, keeping any short trailing one."""
    return [samples[i : i + size] for i in range(0, len(samples), size)]


def scan_probabilities(chunks: Iterable[Waveform], vad: SileroVad) -> Waveform:
    """Score a whole stream, retaining only the probabilities.

    One float per 32 ms is about 450 kB per hour of audio, so a long recording can be
    segmented with the offline state machine without ever holding its waveform.
    """
    vad.reset()
    probs: list[float] = []
    carry = np.zeros(0, dtype=np.float32)
    for chunk in chunks:
        carry = np.concatenate((carry, chunk)) if carry.size else chunk
        usable = len(carry) - (len(carry) % WINDOW_SAMPLES)
        for window in windows(carry[:usable]):
            probs.append(vad.push(window))
        carry = carry[usable:]
    if carry.size:
        probs.append(vad.push(carry))
    return np.asarray(probs, dtype=np.float32)


def extract_spans(
    chunks: Iterable[Waveform], spans: Sequence[Span], sample_rate: int = SAMPLE_RATE
) -> Iterator[tuple[Span, Audio]]:
    """Yield each span's audio from a forward-only stream.

    `spans` must be sorted and non-overlapping, which is what the segmenters produce.
    Memory is bounded by the longest span rather than by the length of the stream.
    """
    pending = deque(spans)
    if not pending:
        return

    position = 0
    collecting: list[Waveform] = []
    start = int(pending[0].start * sample_rate)
    end = int(pending[0].end * sample_rate)

    for chunk in chunks:
        chunk_start = position
        position += len(chunk)
        while pending and start < position:
            lo = max(0, start - chunk_start)
            hi = min(len(chunk), end - chunk_start)
            if hi > lo:
                collecting.append(chunk[lo:hi])
            if end <= position:
                span = pending.popleft()
                samples = (
                    np.concatenate(collecting) if collecting else np.zeros(0, dtype=np.float32)
                )
                yield span, Audio(samples=samples, sample_rate=sample_rate)
                collecting = []
                if not pending:
                    return
                start = int(pending[0].start * sample_rate)
                end = int(pending[0].end * sample_rate)
            else:
                break

    if pending and collecting:
        samples = np.concatenate(collecting)
        yield pending[0], Audio(samples=samples, sample_rate=sample_rate)
