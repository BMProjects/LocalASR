"""Resampling capture audio to the 16 kHz the VAD and the engine both require."""

from __future__ import annotations

import numpy as np
import pytest

from localasr.capture.resample import Resampler, pick_source_rate
from localasr.core.types import SAMPLE_RATE


def _tone(frequency: float, seconds: float, rate: int) -> np.ndarray:
    t = np.arange(int(seconds * rate), dtype=np.float64) / rate
    return np.sin(2 * np.pi * frequency * t).astype(np.float32)


def _dominant(signal: np.ndarray, rate: int) -> float:
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
    return float(np.fft.rfftfreq(len(signal), 1 / rate)[int(np.argmax(spectrum))])


def test_equal_rates_pass_through_untouched() -> None:
    resampler = Resampler(SAMPLE_RATE, SAMPLE_RATE)
    assert resampler.passthrough
    block = _tone(440, 0.1, SAMPLE_RATE)
    assert np.array_equal(resampler(block), block)


def test_speech_band_tone_keeps_its_frequency_and_amplitude() -> None:
    resampler = Resampler(48_000, SAMPLE_RATE)
    out = resampler(_tone(1_000, 1.0, 48_000))
    assert abs(_dominant(out, SAMPLE_RATE) - 1_000) < 5
    # A tone inside the passband must not be attenuated on the way through.
    assert 0.9 < float(np.max(np.abs(out[SAMPLE_RATE // 4 :]))) < 1.1


def test_content_above_the_new_nyquist_is_filtered_not_aliased() -> None:
    """Dropping every third sample would fold 10 kHz down to 6 kHz, landing noise in the
    middle of the speech band. That is the failure this filter exists to prevent."""
    resampler = Resampler(48_000, SAMPLE_RATE)
    out = resampler(_tone(10_000, 1.0, 48_000))

    steady = out[SAMPLE_RATE // 4 :]
    assert float(np.max(np.abs(steady))) < 0.01, "out-of-band tone survived"

    naive = _tone(10_000, 1.0, 48_000)[::3]
    assert float(np.max(np.abs(naive[SAMPLE_RATE // 4 :]))) > 0.9, "decimation aliases"


def test_block_by_block_matches_one_shot() -> None:
    """Capture arrives in 100 ms pieces. A resampler that forgets its tail between them
    puts a discontinuity at every boundary — ten clicks a second."""
    signal = _tone(700, 1.0, 48_000) + 0.3 * _tone(2_500, 1.0, 48_000)

    whole = Resampler(48_000, SAMPLE_RATE)(signal)

    streaming = Resampler(48_000, SAMPLE_RATE)
    block = int(0.1 * 48_000)
    pieces = [streaming(signal[i : i + block]) for i in range(0, len(signal), block)]
    incremental = np.concatenate(pieces)

    assert len(incremental) == len(whole)
    assert np.max(np.abs(incremental - whole)) < 1e-6


def _feed(resampler: Resampler, signal: np.ndarray, sizes: tuple[int, ...]) -> int:
    produced = 0
    cursor = 0
    index = 0
    while cursor < len(signal):
        size = sizes[index % len(sizes)]
        chunk = signal[cursor : cursor + size]
        cursor += size
        produced += len(resampler(chunk, last=cursor >= len(signal)))
        index += 1
    return produced


def test_uneven_block_sizes_do_not_drift() -> None:
    """PortAudio hands over short and uneven blocks. Output must track the exact rate
    ratio rather than gaining or losing samples the longer the stream runs."""
    sizes = (317, 1024, 4800, 91, 6000)
    for seconds in (2.0, 8.0):
        streaming = Resampler(48_000, SAMPLE_RATE)
        signal = _tone(500, seconds, 48_000)
        produced = _feed(streaming, signal, sizes)
        assert abs(produced - len(signal) // 3) <= 1, f"drifted over {seconds}s"


def test_non_integer_ratio_is_supported() -> None:
    """44.1 kHz devices exist and 44100/16000 is not an integer ratio."""
    resampler = Resampler(44_100, SAMPLE_RATE)
    out = resampler(_tone(1_000, 1.0, 44_100), last=True)
    assert abs(len(out) - SAMPLE_RATE) <= 1
    assert abs(_dominant(out, SAMPLE_RATE) - 1_000) < 5


def test_empty_block_is_harmless() -> None:
    resampler = Resampler(48_000, SAMPLE_RATE)
    assert resampler(np.zeros(0, dtype=np.float32)).size == 0


def test_reset_forgets_the_previous_stream() -> None:
    resampler = Resampler(48_000, SAMPLE_RATE)
    first = resampler(_tone(1_000, 0.5, 48_000))
    resampler.reset()
    second = resampler(_tone(1_000, 0.5, 48_000))
    assert np.allclose(first, second)


def test_rejects_nonsense_rates() -> None:
    with pytest.raises(ValueError):
        Resampler(0, SAMPLE_RATE)


def test_pick_source_rate_prefers_no_conversion() -> None:
    assert pick_source_rate(lambda rate: rate in {16_000, 48_000}, 48_000.0) == SAMPLE_RATE


def test_pick_source_rate_falls_back_to_the_device_native_rate() -> None:
    # The Digital Microphone case: 48 kHz only, which is why demanding 16 kHz failed.
    assert pick_source_rate(lambda rate: rate == 48_000, 48_000.0) == 48_000


def test_pick_source_rate_reports_a_device_it_cannot_open() -> None:
    with pytest.raises(ValueError):
        pick_source_rate(lambda _rate: False, 48_000.0)
