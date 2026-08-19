"""Detecting that the microphone and the system monitor carry the same sound.

Measured on this machine: the same playback through both paths correlates at 0.795
with a +452 ms Bluetooth delay; unrelated audio sits near 0.33. All pure-function —
nothing here opens a device.
"""

import numpy as np

from localasr.capture.overlap import CORRELATION_THRESHOLD, compare, envelope, rms
from localasr.core.types import SAMPLE_RATE


def speech_like(seconds: float, *, rate: float = 3.0, phase: float = 0.0, seed: int = 0):
    """Noise under a syllable-rate loudness contour — what the envelope keys on.

    `rate` and `phase` are what make two signals genuinely unrelated: sharing a
    contour would make any two clips correlate regardless of their content.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * SAMPLE_RATE)
    carrier = rng.standard_normal(n).astype(np.float32) * 0.2
    t = np.arange(n) / SAMPLE_RATE
    # A periodic contour makes the lag ambiguous — a delay of nearly one period
    # correlates just as well as no delay at all. Real speech is not periodic, so the
    # contour is a smoothed random walk driven by `rate`.
    steps = rng.standard_normal(max(2, int(seconds * rate)))
    contour = np.interp(t, np.linspace(0, seconds, len(steps)), steps)
    contour = (contour - contour.min()) / (np.ptp(contour) + 1e-9)
    contour = (0.1 + 0.9 * np.roll(contour, int(phase * SAMPLE_RATE))).astype(np.float32)
    return (carrier * contour).astype(np.float32)


def test_the_same_sound_through_two_paths_is_detected():
    """Different gain and a delay — exactly the real conditions."""
    source = speech_like(3.0)
    mic = (np.roll(source, int(0.45 * SAMPLE_RATE)) * 0.05).astype(np.float32)
    result = compare(mic, source)
    assert result.overlapping
    assert result.correlation >= CORRELATION_THRESHOLD


def test_unrelated_audio_is_not_flagged():
    """Different syllable rhythm and phase: two people talking about different things."""
    first = speech_like(3.0, rate=2.3, phase=0.0, seed=1)
    second = speech_like(3.0, rate=5.7, phase=1.1, seed=2)
    result = compare(first, second)
    assert not result.overlapping
    assert result.conclusive


def test_two_silences_are_reported_as_inconclusive():
    """Correlating two silences yields a meaningless number — which is how a first
    attempt at this reported 'different' while the duplication was happening."""
    quiet = np.zeros(int(3.0 * SAMPLE_RATE), dtype=np.float32)
    result = compare(quiet, quiet)
    assert not result.overlapping
    assert not result.conclusive
    assert "太安静" in result.reason


def test_one_silent_source_is_inconclusive_not_different():
    result = compare(np.zeros(int(3.0 * SAMPLE_RATE), dtype=np.float32), speech_like(3.0))
    assert not result.conclusive


def test_a_sample_too_short_to_judge_says_so():
    tiny = np.ones(100, dtype=np.float32)
    result = compare(tiny, tiny)
    assert not result.conclusive
    assert "太短" in result.reason


def test_the_envelope_ignores_gain():
    loud = speech_like(2.0)
    quiet = (loud * 0.01).astype(np.float32)
    assert np.allclose(envelope(loud), envelope(quiet), atol=1e-6)


def test_rms_of_silence_is_zero():
    assert rms(np.zeros(1000, dtype=np.float32)) == 0.0


def test_the_reported_lag_has_the_right_magnitude():
    """The lag is reported so a warning can name the acoustic delay; its sign depends
    on which stream is passed first, so only the magnitude is pinned."""
    source = speech_like(3.0)
    delayed = np.roll(source, int(0.30 * SAMPLE_RATE)).astype(np.float32)
    result = compare(delayed, source)
    assert result.overlapping
    assert abs(abs(result.lag) - 0.30) < 0.06
