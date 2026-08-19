"""Turn transcript segments into readable subtitle cues.

A VAD span is an utterance boundary, not a subtitle. Emitting one cue per span gives
cues that flash for 400 ms, cues that hold a wall of text for 30 seconds, and no line
breaks at all. This stage imposes the constraints a reader actually needs.

What it deliberately does not do: split a segment's *timing*. Without word-level
alignment, the only way to time a sub-part of a segment is to apportion by text length,
which invents timings that look precise and are not. So long segments are wrapped onto
multiple lines within one cue, and splitting into separate cues waits for
`Segment.words` to be populated by a forced aligner.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from localasr.core.types import Segment, Span, Transcript

_CJK = re.compile(r"[　-鿿豈-﫿＀-￯]")
_SENTENCE_END = "。！？!?…"
_CLAUSE_END = "，、；：,;:"


@dataclass(frozen=True, slots=True)
class CueStyle:
    """Reading constraints, in display columns and seconds.

    Line length is budgeted in columns rather than characters because a CJK glyph
    occupies two. One budget then covers Chinese, English and the mixed Chinese-English
    that Qwen3-ASR is chosen for — classifying a line as one script or the other breaks
    exactly on the mixed case, where a CJK rule hard-breaks Latin words mid-token.

    40 columns is 20 CJK characters or 40 Latin ones, the usual subtitle range.
    """

    max_columns: int = 40
    max_lines: int = 2
    min_duration: float = 1.0
    max_cps_cjk: float = 9.0
    max_cps_latin: float = 21.0
    merge_gap: float = 0.4
    min_merge_duration: float = 1.5


@dataclass(frozen=True, slots=True)
class Cue:
    span: Span
    lines: list[str]

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    @property
    def start(self) -> float:
        return self.span.start

    @property
    def end(self) -> float:
        return self.span.end


def is_cjk_char(char: str) -> bool:
    return bool(_CJK.match(char))


def is_cjk(text: str) -> bool:
    """True when the text contains enough CJK to be read at CJK reading speed.

    Used only to pick the characters-per-second budget, not to pick a wrapping
    strategy: mixed Chinese-English is common enough that wrapping handles both at
    once.
    """
    dense = [c for c in text if not c.isspace()]
    if not dense:
        return False
    return sum(is_cjk_char(c) for c in dense) / len(dense) >= 0.2


def display_width(text: str) -> int:
    """Width in terminal-style columns; a CJK glyph occupies two."""
    return sum(2 if is_cjk_char(c) else 1 for c in text)


def _tokenize(text: str) -> list[str]:
    """Split into units that must not be broken apart.

    Each CJK character is its own unit; a run of Latin characters is one unit, so a
    word is never split. Trailing punctuation stays attached to its unit so a line
    never begins with a comma.
    """
    tokens: list[str] = []
    buffer = ""
    for char in text:
        if is_cjk_char(char) or char.isspace():
            if buffer:
                tokens.append(buffer)
                buffer = ""
            if not char.isspace():
                tokens.append(char)
            elif tokens:
                tokens[-1] += " "
        else:
            buffer += char
    if buffer:
        tokens.append(buffer)
    return tokens


def wrap(text: str, max_columns: int) -> list[str]:
    """Break `text` into lines of at most `max_columns` display columns.

    Breaks after CJK characters, at spaces, and after punctuation; never inside a
    Latin word. A single token wider than the budget is emitted on its own line
    rather than truncated.
    """
    text = text.strip()
    if not text:
        return []
    if display_width(text) <= max_columns:
        return [text]

    tokens = _tokenize(text)
    lines: list[str] = []
    index = 0

    while index < len(tokens):
        taken = _fill(tokens, index, max_columns)
        cut = _best_break(tokens, index, index + taken, max_columns)
        lines.append("".join(tokens[index:cut]).strip())
        index = cut
    return [line for line in lines if line]


def _fill(tokens: list[str], start: int, max_columns: int) -> int:
    """How many tokens fit on one line from `start`; always at least one."""
    width = 0
    count = 0
    for token in tokens[start:]:
        token_width = display_width(token.rstrip())
        if count and width + token_width > max_columns:
            break
        width += display_width(token)
        count += 1
    return max(1, count)


def _best_break(tokens: list[str], start: int, end: int, max_columns: int) -> int:
    """Prefer ending a line on punctuation, provided the line stays reasonably full.

    Looking back over the greedily filled line beats breaking eagerly at the first
    punctuation: an early comma would otherwise leave a two-character line.
    """
    if end >= len(tokens):
        return end
    width = 0
    best = end
    for offset, token in enumerate(tokens[start:end], start=1):
        width += display_width(token)
        ends_clause = token.rstrip().endswith(tuple(_SENTENCE_END + _CLAUSE_END))
        if ends_clause and width >= max_columns * 0.5:
            best = start + offset
    return best


def merge_short(segments: list[Segment], style: CueStyle) -> list[Segment]:
    """Join neighbouring segments that are too brief to read on their own.

    Only merges across a small gap, and never past a sentence-final punctuation mark,
    so a merged cue still corresponds to contiguous speech.
    """
    if not segments:
        return []

    merged = [segments[0]]
    for segment in segments[1:]:
        previous = merged[-1]
        gap = segment.start - previous.end
        joined_duration = segment.end - previous.start
        too_short = previous.span.duration < style.min_merge_duration
        ends_sentence = previous.text.rstrip().endswith(tuple(_SENTENCE_END))

        if (
            too_short
            and not ends_sentence
            and gap <= style.merge_gap
            and joined_duration <= style.min_merge_duration * 4
            and previous.source == segment.source
        ):
            separator = "" if is_cjk(previous.text) else " "
            merged[-1] = Segment(
                span=Span(previous.start, segment.end),
                text=f"{previous.text}{separator}{segment.text}".strip(),
                source=previous.source,
                words=[*previous.words, *segment.words],
            )
        else:
            merged.append(segment)
    return merged


def compose(transcript: Transcript, style: CueStyle | None = None) -> list[Cue]:
    """Build display-ready cues from a transcript."""
    style = style or CueStyle()
    cues: list[Cue] = []

    for segment in merge_short(list(transcript.segments), style):
        text = segment.text.strip()
        if not text:
            continue

        lines = wrap(text, style.max_columns)
        if len(lines) > style.max_lines:
            # Rebalance onto the allowed number of lines rather than dropping text:
            # a wider line is readable, missing dialogue is not.
            widened = max(style.max_columns, -(-display_width(text) // style.max_lines))
            lines = wrap(text, widened)
            while len(lines) > style.max_lines:
                widened = int(widened * 1.3) + 1
                lines = wrap(text, widened)

        cues.append(Cue(span=_readable_span(segment, text, is_cjk(text), style), lines=lines))

    return _resolve_overlaps(cues)


def _readable_span(segment: Segment, text: str, cjk: bool, style: CueStyle) -> Span:
    """Extend a cue that is on screen too briefly for its length.

    Only the end moves, and only later: shifting the start would desynchronise the cue
    from the speech that produced it.
    """
    max_cps = style.max_cps_cjk if cjk else style.max_cps_latin
    needed = max(style.min_duration, len(text) / max_cps)
    if segment.span.duration >= needed:
        return segment.span
    return Span(segment.start, segment.start + needed)


def _resolve_overlaps(cues: list[Cue]) -> list[Cue]:
    """Trim any cue whose extended end runs into the next cue's start."""
    for i in range(len(cues) - 1):
        if cues[i].end > cues[i + 1].start:
            cues[i] = Cue(span=Span(cues[i].start, cues[i + 1].start), lines=cues[i].lines)
    return cues
