import json

from localasr.core.postprocess.filters import is_hallucination, is_repetitive
from localasr.core.types import Segment, Span, Transcript
from localasr.formats.subtitles import render


def _transcript() -> Transcript:
    return Transcript(
        segments=[
            Segment(span=Span(0.202, 2.39), text="And so, my fellow Americans."),
            Segment(span=Span(3.146, 4.534), text="今天天气很好。"),
        ],
        duration=11.0,
        language="English",
    )


def test_srt_uses_comma_millis_and_one_based_index():
    out = render(_transcript(), "srt")
    assert "1\n00:00:00,202 --> 00:00:02,390\nAnd so, my fellow Americans.\n" in out
    assert "2\n00:00:03,146 --> 00:00:04,534\n" in out


def test_vtt_has_header_and_dot_millis():
    out = render(_transcript(), "vtt")
    assert out.startswith("WEBVTT")
    assert "00:00:00.202 --> 00:00:02.390" in out


def test_clock_rolls_over_into_hours():
    long = Transcript(segments=[Segment(span=Span(3661.5, 3662.0), text="x")], duration=4000.0)
    assert "01:01:01,500" in render(long, "srt")


def test_json_keeps_unicode_and_rounds_times():
    payload = json.loads(render(_transcript(), "json"))
    assert payload["segments"][1]["text"] == "今天天气很好。"
    assert payload["language"] == "English"
    assert payload["segments"][0]["start"] == 0.202


def test_txt_puts_one_utterance_per_line():
    assert render(_transcript(), "txt").splitlines() == [
        "And so, my fellow Americans.",
        "今天天气很好。",
    ]


def test_repetition_detects_stuck_decode():
    assert is_repetitive("好的好的好的好的好的好的")
    assert is_repetitive("yes yes yes yes yes yes")


def test_repetition_leaves_normal_speech_alone():
    assert not is_repetitive("今天天气很好，我们去公园散步吧。")
    assert not is_repetitive("And so, my fellow Americans, ask not what your country can do.")


def test_hallucination_drops_empty_and_impossibly_dense_text():
    assert is_hallucination("", 2.0)
    assert is_hallucination("   ", 2.0)
    assert is_hallucination("字" * 200, 2.0)


def test_hallucination_keeps_plausible_speech():
    assert not is_hallucination("今天天气很好。", 2.0)
