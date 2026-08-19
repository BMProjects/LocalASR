"""Segmentation decides every timestamp we emit, so the state machine is tested as a
pure function over probabilities, without running the model."""

import numpy as np

from localasr.core.audio.vad import (
    WINDOW_SAMPLES,
    VadConfig,
    segment_probabilities,
    slice_audio,
)
from localasr.core.types import SAMPLE_RATE, Audio, Span

WINDOWS_PER_SECOND = SAMPLE_RATE / WINDOW_SAMPLES  # 31.25


def probs_from(pattern: list[tuple[float, float]]) -> np.ndarray:
    """Build a probability array from (probability, seconds) runs."""
    out: list[float] = []
    for value, seconds in pattern:
        out.extend([value] * int(round(seconds * WINDOWS_PER_SECOND)))
    return np.asarray(out, dtype=np.float32)


def samples_for(probs: np.ndarray) -> int:
    return len(probs) * WINDOW_SAMPLES


def test_finds_a_single_utterance_between_silences():
    probs = probs_from([(0.0, 1.0), (0.9, 2.0), (0.0, 2.0)])
    spans = segment_probabilities(probs, samples_for(probs), VadConfig())
    assert len(spans) == 1
    assert 0.7 < spans[0].start < 1.05
    assert 2.9 < spans[0].end < 3.3


def test_short_pause_does_not_split_an_utterance():
    """A 300 ms gap is a comma; min_silence is 600 ms."""
    probs = probs_from([(0.0, 0.5), (0.9, 1.5), (0.0, 0.3), (0.9, 1.5), (0.0, 1.0)])
    spans = segment_probabilities(probs, samples_for(probs), VadConfig())
    assert len(spans) == 1


def test_long_pause_splits_an_utterance():
    probs = probs_from([(0.0, 0.5), (0.9, 1.5), (0.0, 1.2), (0.9, 1.5), (0.0, 1.0)])
    spans = segment_probabilities(probs, samples_for(probs), VadConfig())
    assert len(spans) == 2


def test_hysteresis_keeps_speech_open_between_thresholds():
    """Probability dipping to 0.4 is below `threshold` but above `neg_threshold`,
    so it must not end the utterance — plain > 0.5 thresholding would split here."""
    probs = probs_from([(0.0, 0.5), (0.9, 1.0), (0.4, 1.0), (0.9, 1.0), (0.0, 1.0)])
    spans = segment_probabilities(probs, samples_for(probs), VadConfig())
    assert len(spans) == 1


def test_speech_running_to_the_end_of_audio_is_closed_and_clamped():
    """The trailing window is zero-padded for scoring; timestamps must not exceed
    the real audio length."""
    probs = probs_from([(0.0, 0.5), (0.9, 2.0)])
    total = samples_for(probs)
    spans = segment_probabilities(probs, total, VadConfig())
    assert len(spans) == 1
    assert spans[-1].end <= total / SAMPLE_RATE


def test_no_span_extends_past_the_audio_after_padding():
    probs = probs_from([(0.9, 3.0)])
    total = samples_for(probs)
    for span in segment_probabilities(probs, total, VadConfig()):
        assert span.end <= total / SAMPLE_RATE
        assert span.start >= 0.0


def test_silence_only_input_yields_nothing():
    probs = probs_from([(0.0, 5.0)])
    assert segment_probabilities(probs, samples_for(probs), VadConfig()) == []


def test_audio_shorter_than_one_window_does_not_crash():
    probs = np.asarray([0.9], dtype=np.float32)
    assert isinstance(segment_probabilities(probs, 200, VadConfig()), list)


def test_empty_probabilities_yield_nothing():
    assert segment_probabilities(np.zeros(0, dtype=np.float32), 0, VadConfig()) == []


def test_blip_shorter_than_min_speech_is_discarded():
    probs = probs_from([(0.0, 1.0), (0.9, 0.1), (0.0, 2.0)])
    assert segment_probabilities(probs, samples_for(probs), VadConfig()) == []


def test_continuous_speech_is_split_at_max_speech():
    probs = probs_from([(0.9, 75.0)])
    spans = segment_probabilities(probs, samples_for(probs), VadConfig(max_speech=30.0))
    assert len(spans) >= 2
    assert all(span.duration <= 32.0 for span in spans)


def test_max_speech_cut_prefers_the_longest_internal_silence():
    """With a 0.2 s lull at 20 s and a 0.15 s lull at 25 s, the cut goes to the
    longer one rather than to an arbitrary offset."""
    probs = probs_from([(0.9, 20.0), (0.0, 0.2), (0.9, 4.8), (0.0, 0.15), (0.9, 20.0)])
    spans = segment_probabilities(probs, samples_for(probs), VadConfig(max_speech=30.0))
    assert len(spans) >= 2
    assert 19.0 < spans[0].end < 22.0


def test_spans_are_ordered_and_non_overlapping():
    probs = probs_from(
        [(0.0, 0.5), (0.9, 1.0), (0.0, 1.0), (0.9, 1.0), (0.0, 1.0), (0.9, 1.0), (0.0, 0.5)]
    )
    spans = segment_probabilities(probs, samples_for(probs), VadConfig())
    assert len(spans) >= 2
    for earlier, later in zip(spans, spans[1:], strict=False):
        assert earlier.end <= later.start


def test_custom_neg_threshold_overrides_the_derived_default():
    assert VadConfig(threshold=0.5).resolved_neg_threshold() == 0.35
    assert VadConfig(threshold=0.5, neg_threshold=0.2).resolved_neg_threshold() == 0.2


def test_slice_audio_clamps_to_the_available_samples():
    audio = Audio(samples=np.ones(SAMPLE_RATE, dtype=np.float32))
    sliced = slice_audio(audio, Span(0.5, 5.0))
    assert len(sliced.samples) == SAMPLE_RATE // 2


# --- live vs offline presets -------------------------------------------------


def test_the_live_preset_rides_through_a_hesitation():
    """`min_silence` is literally what the user waits before seeing anything, but
    waiting less than a thinking pause means being interrupted mid-thought.

    The sample recordings put hesitations at 0.54 s and real sentence ends at
    0.99-1.09 s, so the threshold has to sit between the two bands.
    """
    live = VadConfig.for_live()
    assert live.min_silence > 0.54, "would cut someone off while they think"
    assert live.min_silence < 0.99, "would sit through a finished sentence"


def test_the_live_preset_has_no_length_cap():
    """A cap cuts wherever it lands, which for continuous speech is inside a word."""
    import math

    assert VadConfig.for_live().max_speech == math.inf


def test_the_live_preset_still_clears_real_intra_sentence_pauses():
    """Measured on real Mandarin speech, pauses inside one sentence clustered at
    0.29–0.32 s. A threshold at or below those splits a sentence into fragments."""
    assert VadConfig.for_live().min_silence > 0.32


def test_a_sentence_with_natural_pauses_stays_whole_under_the_live_preset():
    probs = probs_from([(0.0, 0.5), (0.9, 1.5), (0.0, 0.30), (0.9, 1.5), (0.0, 1.0)])
    spans = segment_probabilities(probs, samples_for(probs), VadConfig.for_live())
    assert len(spans) == 1, "a 0.30 s pause inside a sentence must not end the utterance"


def test_a_real_sentence_boundary_still_splits_under_the_live_preset():
    # 1.0 s is where genuine sentence ends were measured in the sample recordings.
    probs = probs_from([(0.0, 0.5), (0.9, 1.5), (0.0, 1.0), (0.9, 1.5), (0.0, 1.0)])
    spans = segment_probabilities(probs, samples_for(probs), VadConfig.for_live())
    assert len(spans) == 2


# --- decaying silence threshold ----------------------------------------------


def test_a_short_utterance_needs_a_real_sentence_pause():
    cfg = VadConfig.for_live()
    assert cfg.silence_for(0.0) == cfg.min_silence
    assert cfg.silence_for(cfg.silence_hold) == cfg.min_silence


def test_the_threshold_decays_once_the_buffer_grows():
    cfg = VadConfig.for_live()
    assert cfg.silence_for(12.0) < cfg.min_silence
    assert cfg.silence_for(20.0) < cfg.silence_for(12.0)


def test_the_threshold_reaches_the_floor_by_the_horizon_and_stays():
    cfg = VadConfig.for_live()
    assert cfg.silence_for(cfg.silence_horizon) == cfg.min_silence_floor
    assert cfg.silence_for(cfg.silence_horizon * 3) == cfg.min_silence_floor


def test_the_floor_is_a_pause_not_a_consonant_closure():
    """Measured gaps below ~0.07 s are stop-consonant closures; cutting there would
    split a word."""
    assert VadConfig.for_live().min_silence_floor >= 0.08


def test_the_threshold_gives_way_only_near_the_horizon():
    """An ordinary utterance keeps the full threshold; only one approaching the horizon
    settles for a hesitation, and then for an intra-sentence pause."""
    cfg = VadConfig.for_live()
    assert cfg.silence_for(5.0) == cfg.min_silence
    assert cfg.silence_for(15.0) > 0.54, "still above the hesitation band at 15 s"
    assert cfg.silence_for(24.0) < 0.54, "gives way to a hesitation near the horizon"
    assert cfg.silence_for(28.0) < 0.32, "and to an intra-sentence pause at the end"


def test_the_decay_holds_high_then_gives_way_quickly():
    """Linear decay spends its middle stretch in the worst range: low enough to cut at
    a hesitation, high enough that the buffer still grows."""
    cfg = VadConfig.for_live()
    span = cfg.silence_horizon - cfg.silence_hold
    midpoint = cfg.silence_for(cfg.silence_hold + span / 2)

    dropped_in_first_half = cfg.min_silence - midpoint
    dropped_in_second_half = midpoint - cfg.min_silence_floor
    assert dropped_in_second_half > 3 * dropped_in_first_half

    # Still monotonic: a longer buffer never demands a longer pause.
    values = [cfg.silence_for(t) for t in range(0, 35)]
    assert all(later <= earlier for earlier, later in zip(values, values[1:], strict=False))


def test_long_speech_is_cut_at_a_pause_the_decay_has_reached():
    """The same pause is ignored early and accepted once the buffer is long enough.

    A 0.35 s gap clears the threshold only once the decay has run most of its course,
    and the state machine measures a gap one 32 ms window short of its true length
    (upstream's loop structure), so the margin has to clear both.
    """
    cfg = VadConfig.for_live()
    early = probs_from([(0.9, 3.0), (0.0, 0.35), (0.9, 3.0), (0.0, 1.5)])
    assert len(segment_probabilities(early, samples_for(early), cfg)) == 1

    late = probs_from([(0.9, 28.0), (0.0, 0.35), (0.9, 3.0), (0.0, 1.5)])
    assert len(segment_probabilities(late, samples_for(late), cfg)) == 2


def test_the_measured_gap_lags_the_configured_threshold_by_one_window():
    """Documented, not fixed: matching upstream's loop is worth 32 ms of bias."""
    cfg = VadConfig.for_live()
    just_under = probs_from([(0.9, 2.0), (0.0, 0.36), (0.9, 2.0), (0.0, 1.5)])
    assert len(segment_probabilities(just_under, samples_for(just_under), cfg)) == 1


def test_speech_with_no_pause_at_all_is_never_cut():
    """The honest consequence of never cutting mid-word: with nothing to cut at, the
    utterance keeps growing. Real speech always breathes; a synthetic tone does not."""
    probs = probs_from([(0.9, 40.0)])
    spans = segment_probabilities(probs, samples_for(probs), VadConfig.for_live())
    assert len(spans) == 1
