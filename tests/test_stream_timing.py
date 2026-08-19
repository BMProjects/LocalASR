"""Timing of the live segmenter.

A meeting's two capture threads never start at the same instant. If each stream timed
itself by counting its own samples, the microphone and system timelines would drift
apart and the transcript order would be wrong. These pin the behaviour that prevents it.
"""

import numpy as np
import pytest

from localasr.core.audio.stream import StreamingSegmenter, windows
from localasr.core.audio.vad import WINDOW_SAMPLES, SileroVad
from localasr.core.types import SAMPLE_RATE, AudioBlock
from localasr.registry import manager


@pytest.fixture
def segmenter():
    return StreamingSegmenter(SileroVad(manager.vad_path()), epoch=1000.0)


def block(at: float, sequence: int, seconds: float = 0.1, source="mic") -> AudioBlock:
    return AudioBlock(
        samples=np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32),
        captured_at=at,
        sequence=sequence,
        source=source,
    )


def test_position_is_measured_from_the_session_epoch_not_from_the_first_block(segmenter):
    """A source that starts two seconds late must report two seconds, not zero."""
    segmenter.push_block(block(at=1002.0, sequence=0))
    assert segmenter.position == pytest.approx(2.1, abs=0.05)


def test_two_sources_starting_apart_stay_on_one_timeline():
    vad_a = SileroVad(manager.vad_path())
    vad_b = SileroVad(manager.vad_path())
    epoch = 500.0
    first = StreamingSegmenter(vad_a, epoch=epoch)
    second = StreamingSegmenter(vad_b, epoch=epoch)

    first.push_block(block(at=epoch + 0.0, sequence=0))
    second.push_block(block(at=epoch + 3.0, sequence=0))

    assert second.position - first.position == pytest.approx(3.0, abs=0.05)


def test_a_sequence_gap_is_reported(segmenter):
    segmenter.push_block(block(at=1000.0, sequence=0))
    _, gap = segmenter.push_block(block(at=1000.5, sequence=7))
    assert gap is not None
    assert "sequence jumped" in gap.reason


def test_contiguous_blocks_report_no_gap(segmenter):
    segmenter.push_block(block(at=1000.0, sequence=0))
    _, gap = segmenter.push_block(block(at=1000.1, sequence=1))
    assert gap is None


def test_clock_drift_beyond_tolerance_is_reported(segmenter):
    """In-order blocks whose stamps outrun the audio they carry mean lost buffers.

    Measured against the established baseline: the first block that can be compared
    defines it, because at that point latency and loss are indistinguishable.
    """
    for index in range(3):
        segmenter.push_block(block(at=1000.0 + index * 0.1, sequence=index))
    _, gap = segmenter.push_block(block(at=1000.3 + 2.0, sequence=3))
    assert gap is not None
    assert gap.reason == "clock drift"
    assert gap.seconds > 1.0


def test_small_jitter_is_not_reported_as_a_gap(segmenter):
    segmenter.push_block(block(at=1000.0, sequence=0))
    _, gap = segmenter.push_block(block(at=1000.15, sequence=1))
    assert gap is None


def test_partial_windows_are_carried_across_blocks(segmenter):
    """A device block size is not a multiple of the VAD window; the remainder must be
    kept, not dropped."""
    odd = WINDOW_SAMPLES + 100
    for index in range(4):
        segmenter.push_block(
            AudioBlock(
                samples=np.zeros(odd, dtype=np.float32),
                captured_at=1000.0 + index * odd / SAMPLE_RATE,
                sequence=index,
            )
        )
    consumed = segmenter.position * SAMPLE_RATE
    assert consumed >= 4 * odd - WINDOW_SAMPLES


def test_utterances_carry_the_block_source(chinese_speech):
    from localasr.capture.microphone import blocks_from_file

    seg = StreamingSegmenter(SileroVad(manager.vad_path()))
    found = []
    for audio_block in blocks_from_file(chinese_speech, source="system", block_seconds=0.2):
        utterances, _ = seg.push_block(audio_block)
        found.extend(utterances)
    trailing = seg.flush()
    if trailing:
        found.append(trailing)

    assert found
    assert all(u.source == "system" for u in found)


def test_each_utterance_gets_a_distinct_id(chinese_speech, english_speech):
    from localasr.capture.microphone import blocks_from_file

    ids = set()
    for path in (chinese_speech, english_speech):
        seg = StreamingSegmenter(SileroVad(manager.vad_path()))
        for audio_block in blocks_from_file(path, block_seconds=0.2):
            utterances, _ = seg.push_block(audio_block)
            ids.update(u.utterance_id for u in utterances)
        trailing = seg.flush()
        if trailing:
            ids.add(trailing.utterance_id)
    assert len(ids) >= 2


def test_windows_keeps_a_short_trailing_window():
    parts = windows(np.zeros(1100, dtype=np.float32))
    assert [len(p) for p in parts] == [512, 512, 76]


def test_constant_capture_latency_is_not_reported_as_drift(segmenter):
    """Blocks are stamped when their callback runs, so every stamp trails the audio by
    the device latency. That offset is constant and must not be reported on every
    block — the first real run produced a warning per 100 ms buffer."""
    latency = 0.28
    for index in range(30):
        _, gap = segmenter.push_block(
            AudioBlock(
                samples=np.zeros(1600, dtype=np.float32),
                captured_at=1000.0 + latency + index * 0.1,
                sequence=index,
            )
        )
        assert gap is None, f"latency reported as drift on block {index}"


def test_a_real_loss_on_top_of_latency_is_still_reported(segmenter):
    latency = 0.28
    for index in range(5):
        segmenter.push_block(
            AudioBlock(
                samples=np.zeros(1600, dtype=np.float32),
                captured_at=1000.0 + latency + index * 0.1,
                sequence=index,
            )
        )
    _, gap = segmenter.push_block(
        AudioBlock(
            samples=np.zeros(1600, dtype=np.float32),
            captured_at=1000.0 + latency + 5 * 0.1 + 1.5,
            sequence=5,
        )
    )
    assert gap is not None
    assert gap.seconds == pytest.approx(1.5, abs=0.05)


def test_a_single_loss_is_reported_once_not_forever(segmenter):
    """Absorbing the step matters: otherwise one dropped buffer warns on every block
    for the rest of the session."""
    for index in range(3):
        segmenter.push_block(
            AudioBlock(
                samples=np.zeros(1600, dtype=np.float32),
                captured_at=1000.0 + index * 0.1,
                sequence=index,
            )
        )
    segmenter.push_block(
        AudioBlock(
            samples=np.zeros(1600, dtype=np.float32),
            captured_at=1000.0 + 3 * 0.1 + 2.0,
            sequence=3,
        )
    )
    gaps = []
    for index in range(4, 12):
        _, gap = segmenter.push_block(
            AudioBlock(
                samples=np.zeros(1600, dtype=np.float32),
                captured_at=1000.0 + index * 0.1 + 2.0,
                sequence=index,
            )
        )
        gaps.append(gap)
    assert all(gap is None for gap in gaps)
