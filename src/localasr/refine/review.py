"""One advisory check on a refinement: did the numbers change?

This module used to be `fidelity.py`, and it used to reject. It was built on a premise
that turned out to be wrong for this application: that a substitution is the dangerous
thing a tidy-up can do. For a *speech* transcript the opposite is true — recognition
routinely returns a homophone, and inferring the intended word from a wrong one that
sounds like it is the single most valuable thing the refinement model does. The old
subsequence proof forbade exactly that, and forbade it by construction, so the feature's
main purpose read as its primary threat.

The rest of the machinery — rejecting, and falling back to showing the transcript —
answered a question this interface never asks. Both panes are on screen the whole time,
the transcript is editable, the caret is the user's, and nothing is delivered anywhere
without them pressing something. There is no unsupervised path for bad output to escape
down, so there is nothing for a gate to protect. What the gate did instead was throw away
good rewrites and leave the original sitting in the refined pane, which read as the model
having ignored the request — three separate bug reports came from that.

So: nothing is rejected here. What survives is the one check a reader genuinely skims
past. A substituted word reads oddly and gets noticed; 三十五万 and 三十六万 do not, and
both parse fluently. It costs nothing to point at, and it blocks nothing.
"""

from __future__ import annotations

import re
import unicodedata

from localasr.refine.types import Note

_CJK_DIGITS = "零〇一二三四五六七八九十百千万亿两"
_MEASURE_WORDS = (
    "个人次天年月日时点分秒元块角毛米厘公斤克吨度倍号条张台部件套页章节步项种类"
)
_DIGIT_RUN = re.compile(r"\d+(?:[.,:：/]\d+)*%?")
_CJK_RUN = re.compile(f"[{_CJK_DIGITS}]+")


def significant_numbers(text: str) -> frozenset[str]:
    """Numbers whose change would change what was said.

    Western digits always count. A lone CJK numeral does not: 一 occurs inside ordinary
    words (一般, 一起, 十分), and treating those as quantities would point at every
    rewrite that happened to rephrase one. A CJK numeral counts when it is long enough to
    be a real quantity, or when a measure word follows it.
    """
    normalised = unicodedata.normalize("NFKC", text)
    found = set(_DIGIT_RUN.findall(normalised))
    for match in _CJK_RUN.finditer(normalised):
        run = match.group()
        following = normalised[match.end() : match.end() + 1]
        if len(run) >= 2 or following in _MEASURE_WORDS:
            found.add(run)
    return frozenset(found)


def review(raw: str, refined: str) -> tuple[Note, ...]:
    """Things worth a second look. Never a reason to withhold the refinement."""
    before, after = significant_numbers(raw), significant_numbers(refined)
    notes: list[Note] = []
    if gone := before - after:
        notes.append(Note("number_gone", f"原文里的数字没有出现在结果中：{sorted(gone)}"))
    if new := after - before:
        notes.append(Note("number_new", f"结果里的数字原文中没有：{sorted(new)}"))
    return tuple(notes)
