"""Guards against the failure mode specific to LLM-based ASR: confident invention.

Qwen3-ASR decodes autoregressively, so a span containing only breath or room tone can
still produce fluent text, and a stuck decode can repeat one phrase to the token limit.
Neither is detectable from the text alone — both need the span's duration as context.
"""

from __future__ import annotations

MAX_CHARS_PER_SECOND = 25.0
"""Well above natural speech (Mandarin ~5 char/s, English ~15 char/s), so this only
fires on output that could not correspond to the audio."""

MIN_REPEAT_LENGTH = 4
MAX_REPEAT_RATIO = 0.6


def is_repetitive(text: str) -> bool:
    """True when one short unit repeats over most of `text`.

    Detects the stuck-decode loop ("好的好的好的…") without flagging ordinary
    repetition inside a normal sentence.
    """
    stripped = "".join(text.split())
    if len(stripped) < MIN_REPEAT_LENGTH * 3:
        return False

    for unit in range(1, len(stripped) // 3 + 1):
        pattern = stripped[:unit]
        repeats = 1
        while stripped[repeats * unit : (repeats + 1) * unit] == pattern:
            repeats += 1
        if repeats * unit >= len(stripped) * MAX_REPEAT_RATIO and repeats >= 3:
            return True
    return False


def rejection_reason(text: str, duration: float) -> str | None:
    """Why `text` should be dropped for a span of `duration`, or None to keep it.

    Returning the reason rather than a bool lets callers report what was discarded;
    a silently dropped segment looks identical to a segment that was never there.
    """
    stripped = text.strip()
    if not stripped:
        return "empty"
    if duration > 0 and len(stripped) / duration > MAX_CHARS_PER_SECOND:
        return f"implausible density ({len(stripped) / duration:.0f} chars/s)"
    if is_repetitive(stripped):
        return "repetition loop"
    return None


def is_hallucination(text: str, duration: float) -> bool:
    """True when `text` should be dropped rather than emitted for a span."""
    return rejection_reason(text, duration) is not None
