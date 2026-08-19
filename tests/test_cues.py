"""Cue composition: the step between "what was recognised" and "what a reader sees"."""

from localasr.core.types import Segment, Span, Transcript
from localasr.formats.cues import (
    CueStyle,
    compose,
    display_width,
    is_cjk,
    merge_short,
    wrap,
)


def seg(start: float, end: float, text: str, source: str | None = None) -> Segment:
    return Segment(span=Span(start, end), text=text, source=source)


def test_cjk_reading_speed_applies_to_mixed_chinese_english():
    assert is_cjk("今天天气很好")
    assert is_cjk("今天天气很好，appointment 在 3 点")
    assert not is_cjk("And so, my fellow Americans")


def test_display_width_counts_cjk_as_two_columns():
    assert display_width("abc") == 3
    assert display_width("今天") == 4
    assert display_width("今天ok") == 6


def test_short_text_is_one_line():
    assert wrap("短句", 40) == ["短句"]


def test_wrapping_prefers_punctuation():
    lines = wrap("今天天气很好，我们一起去公园散步吧。", 24)
    assert len(lines) == 2
    assert lines[0].endswith("，")


def test_wrapping_hard_breaks_cjk_without_punctuation():
    lines = wrap("一" * 40, 40)
    assert all(display_width(line) <= 40 for line in lines)
    assert "".join(lines) == "一" * 40


def test_wrapping_breaks_latin_on_word_boundaries():
    lines = wrap("ask not what your country can do for you", 20)
    assert all(display_width(line) <= 20 for line in lines)
    assert " ".join(lines) == "ask not what your country can do for you"


def test_wrapping_never_splits_a_latin_word_inside_mixed_text():
    """The mixed case is why wrapping is width-based rather than script-based."""
    text = "我们的 appointment 安排在下午三点，请准时参加会议"
    lines = wrap(text, 20)
    assert any("appointment" in line for line in lines)
    assert "".join(line.replace(" ", "") for line in lines) == text.replace(" ", "")


def test_a_very_short_cue_is_extended_to_the_minimum_duration():
    transcript = Transcript(segments=[seg(0.0, 0.3, "好的")], duration=5.0)
    cue = compose(transcript)[0]
    assert cue.end - cue.start >= 1.0


def test_a_dense_cue_is_extended_to_respect_reading_speed():
    """18 CJK characters at 9 chars/s needs about 2 s, not the 0.5 s it was spoken in."""
    dense = "今天天气很好我们一起去公园散步吧啊"
    transcript = Transcript(segments=[seg(0.0, 0.5, dense)], duration=10.0)
    cue = compose(transcript)[0]
    assert cue.end - cue.start >= 1.8


def test_extending_a_cue_never_overruns_the_next_one():
    transcript = Transcript(
        segments=[seg(0.0, 0.3, "第一句话说得很快"), seg(1.0, 3.0, "第二句")], duration=5.0
    )
    cues = compose(transcript)
    assert cues[0].end <= cues[1].start


def test_extending_a_cue_never_moves_its_start():
    transcript = Transcript(segments=[seg(2.0, 2.2, "好")], duration=5.0)
    assert compose(transcript)[0].start == 2.0


def test_short_neighbouring_segments_are_merged():
    style = CueStyle()
    merged = merge_short([seg(0.0, 0.8, "那么"), seg(1.0, 2.5, "我们开始吧")], style)
    assert len(merged) == 1
    assert merged[0].text == "那么我们开始吧"


def test_merge_does_not_cross_a_sentence_boundary():
    merged = merge_short([seg(0.0, 0.8, "好的。"), seg(1.0, 2.5, "下一个问题")], CueStyle())
    assert len(merged) == 2


def test_merge_does_not_cross_a_long_gap():
    merged = merge_short([seg(0.0, 0.8, "那么"), seg(5.0, 6.5, "我们开始")], CueStyle())
    assert len(merged) == 2


def test_merge_does_not_join_different_speakers():
    segments = [seg(0.0, 0.8, "那么", source="mic"), seg(1.0, 2.5, "我们开始", source="system")]
    assert len(merge_short(segments, CueStyle())) == 2


def test_latin_merge_inserts_a_space():
    merged = merge_short([seg(0.0, 0.8, "ask not"), seg(1.0, 2.5, "what your country")], CueStyle())
    assert merged[0].text == "ask not what your country"


def test_no_cue_exceeds_the_line_limit():
    long_text = "这是一段很长的话没有任何标点符号所以只能硬切分成很多行来显示给观众看清楚"
    transcript = Transcript(segments=[seg(0.0, 10.0, long_text)], duration=10.0)
    cue = compose(transcript)[0]
    assert len(cue.lines) <= CueStyle().max_lines


def test_no_text_is_lost_when_rebalancing_onto_two_lines():
    long_text = "这是一段很长的话没有任何标点符号所以只能硬切分成很多行来显示给观众看清楚"
    transcript = Transcript(segments=[seg(0.0, 10.0, long_text)], duration=10.0)
    cue = compose(transcript)[0]
    assert "".join(cue.lines) == long_text


def test_empty_segments_produce_no_cues():
    transcript = Transcript(segments=[seg(0.0, 1.0, "  ")], duration=1.0)
    assert compose(transcript) == []
