"""The `language X<asr_text>` prefix is the one piece of engine output we must not
pass through verbatim (ggml-org/llama.cpp#26749)."""

from localasr.core.engine.client import parse_asr_payload


def test_strips_language_prefix_and_reports_language():
    raw = "language English<asr_text>And so, my fellow Americans."
    result = parse_asr_payload(raw)
    assert result.text == "And so, my fellow Americans."
    assert result.language == "English"


def test_handles_chinese_transcript():
    result = parse_asr_payload("language Chinese<asr_text>今天天气很好。")
    assert result.text == "今天天气很好。"
    assert result.language == "Chinese"


def test_passes_through_when_marker_absent():
    result = parse_asr_payload("  a bare transcript  ")
    assert result.text == "a bare transcript"
    assert result.language is None


def test_keeps_text_when_prefix_is_not_a_language_tag():
    result = parse_asr_payload("something else<asr_text>the words")
    assert result.text == "the words"
    assert result.language is None


def test_empty_transcript_after_marker():
    result = parse_asr_payload("language English<asr_text>")
    assert result.text == ""
    assert result.language == "English"
