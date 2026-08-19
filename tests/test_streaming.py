"""Streaming decode and span extraction: the path that keeps a long recording from
being held in memory all at once."""

import numpy as np
import pytest

from localasr.core.audio.decode import DecodeError, decode_file, iter_pcm, to_wav_bytes
from localasr.core.audio.stream import extract_spans, windows
from localasr.core.types import SAMPLE_RATE, Audio, Span


@pytest.fixture
def tone(tmp_path):
    """A 6-second ramp, so any sample's value identifies its position exactly."""
    samples = np.linspace(-1.0, 1.0, 6 * SAMPLE_RATE, dtype=np.float32)
    path = tmp_path / "tone.wav"
    path.write_bytes(to_wav_bytes(Audio(samples=samples)))
    return path


def test_streaming_yields_the_same_audio_as_a_full_decode(tone):
    whole = decode_file(tone)
    streamed = np.concatenate(list(iter_pcm(tone, chunk_seconds=0.7)))
    assert len(streamed) == len(whole.samples)
    assert np.allclose(streamed, whole.samples, atol=1e-4)


def test_streaming_chunks_stay_within_the_requested_size(tone):
    chunks = list(iter_pcm(tone, chunk_seconds=0.5))
    assert len(chunks) > 1
    assert all(len(chunk) <= int(0.5 * SAMPLE_RATE) for chunk in chunks)


def test_streaming_a_missing_file_raises_decode_error(tmp_path):
    with pytest.raises(DecodeError):
        list(iter_pcm(tmp_path / "nope.wav"))


def test_streaming_a_non_media_file_raises_decode_error(tmp_path):
    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"not audio at all")
    with pytest.raises(DecodeError):
        list(iter_pcm(junk))


def _chunked(samples, size):
    return [samples[i : i + size] for i in range(0, len(samples), size)]


def test_extract_spans_recovers_the_right_audio_regardless_of_chunk_boundaries():
    samples = np.arange(SAMPLE_RATE * 4, dtype=np.float32)
    spans = [Span(0.5, 1.0), Span(2.0, 2.25)]

    for chunk_size in (1000, SAMPLE_RATE // 3, SAMPLE_RATE * 2):
        extracted = list(extract_spans(_chunked(samples, chunk_size), spans))
        assert [span for span, _ in extracted] == spans
        for span, audio in extracted:
            expected = samples[int(span.start * SAMPLE_RATE) : int(span.end * SAMPLE_RATE)]
            assert np.array_equal(audio.samples, expected)


def test_extract_spans_handles_a_span_spanning_many_chunks():
    samples = np.arange(SAMPLE_RATE * 3, dtype=np.float32)
    spans = [Span(0.1, 2.9)]
    ((_, audio),) = list(extract_spans(_chunked(samples, 512), spans))
    assert len(audio.samples) == int(2.9 * SAMPLE_RATE) - int(0.1 * SAMPLE_RATE)


def test_extract_spans_with_no_spans_yields_nothing():
    assert list(extract_spans([np.zeros(100, dtype=np.float32)], [])) == []


def test_windows_keeps_a_short_trailing_window():
    parts = windows(np.zeros(1100, dtype=np.float32))
    assert [len(p) for p in parts] == [512, 512, 76]
