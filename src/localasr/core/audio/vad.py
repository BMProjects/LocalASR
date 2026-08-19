"""Speech segmentation with Silero VAD.

Runs the ONNX graph directly rather than through the ``silero-vad`` package, which
pulls in PyTorch — keeping torch out of the default dependency tree. The
post-processing is a port of the upstream ``get_speech_timestamps`` state machine
(silero-vad @ 76e3dc4), because naive per-window thresholding cuts speech in ways the
official hysteresis does not.

Segment boundaries produced here are the pipeline's only source of timestamps, since
the transcription endpoint returns text without timing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from localasr.core.types import SAMPLE_RATE, Audio, Span, Waveform

WINDOW_SAMPLES = 512
"""Silero advances 512 samples (32 ms) per step at 16 kHz. The graph rejects any other
window size at this rate."""

CONTEXT_SAMPLES = 64
"""Each step is fed the previous 64 samples prepended to its window.

The graph declares `input` as shape [None, None], so feeding a bare 512-sample window
is accepted silently and yields near-zero probabilities for every window instead of an
error. The context is mandatory.
"""

_STATE_SHAPE = (2, 1, 128)


@dataclass(frozen=True, slots=True)
class VadConfig:
    """Segmentation tuning, in seconds.

    `threshold` / `neg_threshold` form the upstream hysteresis pair: speech starts at
    the upper bound and only ends after the probability stays under the lower one.

    `min_silence` and `speech_pad` deliberately exceed Silero's defaults (0.10 / 0.03),
    which target speech detection. Transcription wants whole utterances: a 100 ms pause
    is a comma, not an utterance boundary, and splitting there hands the engine
    fragments that decode worse than the full clause.

    These defaults are the offline ones, tuned for subtitles. `for_live()` trades some
    of that for latency; see its docstring for the measurements behind the numbers.
    """

    threshold: float = 0.5
    neg_threshold: float | None = None
    min_speech: float = 0.25
    min_silence: float = 0.60
    speech_pad: float = 0.15
    max_speech: float = 30.0
    min_silence_at_max_speech: float = 0.098

    min_silence_floor: float = 0.10
    """Shortest gap still treated as a pause once an utterance has run long.

    Measured on real speech, the gaps below ~0.07 s are stop-consonant closures rather
    than pauses; breaths sit at 0.13–0.21 s and intra-sentence pauses at 0.29–0.35 s.
    0.10 s is also Silero's own `min_silence_duration_ms` default — upstream's line
    between "silence" and "articulation".
    """

    silence_hold: float = 8.0
    """An utterance shorter than this keeps the full `min_silence`.

    Ordinary sentences must not be split at their commas. Decaying from zero would put
    the threshold at 0.31 s by five seconds, already inside the measured 0.29–0.35 s
    band of intra-sentence pauses.
    """

    silence_horizon: float = 30.0
    """By this much buffered speech the threshold has reached the floor."""

    silence_decay: float = 2.5
    """Curvature of the decay. 1.0 is linear; above 1.0 the threshold barely moves at
    first and falls steeply near the horizon.

    Linear decay spends its middle stretch in the worst possible range — low enough to
    cut at a hesitation, high enough that the buffer still grows. Holding near the full
    threshold and then giving way quickly keeps ordinary speech intact for as long as
    possible, and concentrates the compromise into the last few seconds before the
    horizon, where the buffer genuinely has to be closed.
    """

    def silence_for(self, buffered: float) -> float:
        """How long a pause must last to end an utterance already `buffered` s long.

        A fixed threshold has to choose between splitting ordinary sentences at their
        commas and letting continuous speech grow without bound. Decaying it does
        neither: a short utterance still needs a real sentence-final pause, while a long
        one will settle for a breath — and because the cut is still made *at a pause*,
        it never lands inside a word the way a hard length cap does.
        """
        if buffered <= self.silence_hold:
            return self.min_silence
        if buffered >= self.silence_horizon:
            return self.min_silence_floor
        span = self.silence_horizon - self.silence_hold
        progress = (buffered - self.silence_hold) / span
        return self.min_silence + (self.min_silence_floor - self.min_silence) * (
            progress**self.silence_decay
        )

    @classmethod
    def for_live(cls) -> VadConfig:
        """Settings for dictation and meetings, where waiting is the dominant cost.

        `min_silence` is what the user actually waits: nothing is transcribed until the
        pause has lasted this long. Measured on this machine:

        * Pauses in the sample recordings fall into three bands: articulation gaps at
          0.13–0.32 s, a hesitation — thinking mid-sentence — at 0.54 s, and genuine
          sentence ends at 0.99–1.09 s. 0.70 s sits in the gap between the last two, so
          it rides through someone pausing to think and still closes the utterance as
          soon as they actually finish a sentence.
        * The earlier 0.35 s was set from the articulation band alone and cut people off
          mid-thought. Raising it costs exactly the difference in wait — about 350 ms
          more before text appears — which is the price of not being interrupted.
        * Fragmenting costs punctuation, not accuracy: the same audio split three ways
          produced the same words with commas turned into full stops. For dictation,
          which is edited anyway, that is a fair trade; subtitles keep the defaults.
        * Splitting more often is nearly free here — a request costs 103 ms fixed plus
          68 ms per second of audio, against an engine running at ~10x realtime.

        `min_speech` is lower than the offline default too. Short answers — 「对」,
        「嗯」, "yes" — are whole utterances in dictation, and dropping them below
        0.25 s loses real words silently. Noise that survives the VAD is caught later
        by the empty-transcript check instead.

        There is no length cap. A cap cuts wherever it lands, which for continuous
        speech means inside a word; `silence_for` instead lowers the bar for what counts
        as a pause as the buffer grows, so the cut still happens at a pause. With these
        numbers the threshold is still 0.68 s after 13 s of unbroken speech, only drops
        under the 0.54 s hesitation band around 21 s, and reaches 0.10 s at 30 s.
        """
        return cls(min_silence=0.70, max_speech=math.inf, min_speech=0.15)

    def resolved_neg_threshold(self) -> float:
        if self.neg_threshold is not None:
            return self.neg_threshold
        return max(self.threshold - 0.15, 0.01)


class SileroVad:
    """Stateful model wrapper; `probabilities()` resets internally per call."""

    def __init__(self, model_path: str | Path, config: VadConfig | None = None) -> None:
        # Imported here, not at module scope. `VadConfig` is a plain dataclass that the
        # application modules import, and through them the CLI — so a top-level
        # `import onnxruntime` made `localasr models pull` require the inference runtime.
        # Downloading a file has nothing to do with running a VAD, and on the compute
        # node, which deliberately has no audio stack, it failed outright.
        import onnxruntime as ort

        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"silero VAD model not found: {model_path}")

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.config = config or VadConfig()
        self._state = np.zeros(_STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)

    def reset(self) -> None:
        self._state = np.zeros(_STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)

    def push(self, window: Waveform) -> float:
        """Score one window, advancing the model state.

        `window` shorter than WINDOW_SAMPLES is zero-padded, matching upstream, so the
        trailing partial window of a stream is still scored instead of dropped.
        """
        if len(window) < WINDOW_SAMPLES:
            window = np.pad(window, (0, WINDOW_SAMPLES - len(window)))
        frame = np.concatenate((self._context, window))
        out, self._state = self._session.run(
            ["output", "stateN"],
            {
                "input": frame.reshape(1, -1),
                "state": self._state,
                "sr": np.array(SAMPLE_RATE, dtype=np.int64),
            },
        )
        self._context = frame[-CONTEXT_SAMPLES:]
        return float(out[0][0])

    def probabilities(self, samples: Waveform) -> Waveform:
        """Per-window speech probability, shape (n_windows,), dtype float32.

        The final partial window is padded rather than dropped, so speech in the last
        32 ms of the audio is still detected.
        """
        self.reset()
        n_windows = int(np.ceil(len(samples) / WINDOW_SAMPLES))
        probs = np.empty(n_windows, dtype=np.float32)
        for i in range(n_windows):
            probs[i] = self.push(samples[i * WINDOW_SAMPLES : (i + 1) * WINDOW_SAMPLES])
        return probs

    def segment(self, audio: Audio) -> list[Span]:
        """Locate speech spans in `audio`, in seconds relative to its start."""
        if audio.sample_rate != SAMPLE_RATE:
            raise ValueError(f"VAD expects {SAMPLE_RATE} Hz, got {audio.sample_rate}")
        probs = self.probabilities(audio.samples)
        return segment_probabilities(probs, len(audio.samples), self.config)


def segment_probabilities(
    probs: Waveform, total_samples: int, config: VadConfig | None = None
) -> list[Span]:
    """Turn per-window speech probabilities into padded speech spans.

    A port of upstream ``get_speech_timestamps``: hysteresis on entry/exit, silence
    tolerated up to `min_silence`, and an over-long span cut at its longest candidate
    silence rather than at an arbitrary offset. Kept as a pure function so the whole
    state machine is testable without running the model.
    """
    cfg = config or VadConfig()
    neg_threshold = cfg.resolved_neg_threshold()

    min_speech_samples = cfg.min_speech * SAMPLE_RATE
    speech_pad_samples = int(cfg.speech_pad * SAMPLE_RATE)
    min_silence_at_max = cfg.min_silence_at_max_speech * SAMPLE_RATE
    max_speech_samples = cfg.max_speech * SAMPLE_RATE - WINDOW_SAMPLES - 2 * speech_pad_samples

    speeches: list[dict[str, int]] = []
    current: dict[str, int] = {}
    triggered = False
    temp_end = 0
    prev_end = 0
    next_start = 0
    possible_ends: list[tuple[int, int]] = []

    for i, prob in enumerate(probs):
        cur = WINDOW_SAMPLES * i

        if prob >= cfg.threshold and temp_end:
            silence = cur - temp_end
            if silence > min_silence_at_max:
                possible_ends.append((temp_end, silence))
            temp_end = 0
            if next_start < prev_end:
                next_start = cur

        if prob >= cfg.threshold and not triggered:
            triggered = True
            current = {"start": cur}
            continue

        if triggered and (cur - current["start"]) > max_speech_samples:
            if possible_ends:
                prev_end, gap = max(possible_ends, key=lambda item: item[1])
                current["end"] = prev_end
                speeches.append(current)
                current = {}
                next_start = prev_end + gap
                if next_start < prev_end + cur:
                    current = {"start": next_start}
                else:
                    triggered = False
            else:
                current["end"] = cur
                speeches.append(current)
                current = {}
                triggered = False
            prev_end = next_start = temp_end = 0
            possible_ends = []
            if not triggered:
                continue

        if prob < neg_threshold and triggered:
            if not temp_end:
                temp_end = cur
            buffered = (cur - current["start"]) / SAMPLE_RATE
            if (cur - temp_end) < cfg.silence_for(buffered) * SAMPLE_RATE:
                continue
            current["end"] = temp_end
            if (current["end"] - current["start"]) > min_speech_samples:
                speeches.append(current)
            current = {}
            prev_end = next_start = temp_end = 0
            triggered = False
            possible_ends = []

    if current and (total_samples - current["start"]) > min_speech_samples:
        current["end"] = total_samples
        speeches.append(current)

    return _apply_padding(speeches, speech_pad_samples, total_samples)


def _apply_padding(speeches: list[dict[str, int]], pad: int, total_samples: int) -> list[Span]:
    """Widen spans by `pad`, splitting a gap too small for both neighbours.

    Ends are clamped to the real audio length, so the zero padding used to score the
    trailing window never shows up as timestamps past the end of the file.
    """
    for i, speech in enumerate(speeches):
        if i == 0:
            speech["start"] = max(0, speech["start"] - pad)
        if i != len(speeches) - 1:
            gap = speeches[i + 1]["start"] - speech["end"]
            if gap < 2 * pad:
                speech["end"] += gap // 2
                speeches[i + 1]["start"] = max(0, speeches[i + 1]["start"] - gap // 2)
            else:
                speech["end"] = min(total_samples, speech["end"] + pad)
                speeches[i + 1]["start"] = max(0, speeches[i + 1]["start"] - pad)
        else:
            speech["end"] = min(total_samples, speech["end"] + pad)

    return [Span(s["start"] / SAMPLE_RATE, s["end"] / SAMPLE_RATE) for s in speeches]


def slice_audio(audio: Audio, span: Span) -> Audio:
    """Extract `span` from `audio`; the returned samples are a view, not a copy."""
    start = max(0, int(span.start * audio.sample_rate))
    end = min(len(audio.samples), int(span.end * audio.sample_rate))
    return Audio(samples=audio.samples[start:end], sample_rate=audio.sample_rate)
