"""Checks run on a model's output before anyone is allowed to see it.

Two very different strengths live here, and conflating them would be the dangerous
mistake.

**Conservative cleaning is provable.** If the model may only add punctuation and delete
filler, then its output — with punctuation removed — must be a *subsequence* of the
input. That single property rules out every insertion and every substitution at once,
which is exactly the class of failure that matters: a fabricated number, a flipped
negation, a "corrected" product name. No amount of prompt engineering establishes that;
a subsequence check does.

**Custom instructions and prompt drafting are only risk-screened.** Both reorder and
rewrite by design, so subsequence checking cannot apply, and what remains — comparing
sets of numbers, negations and identifiers — does *not* prove meaning was preserved.
It cannot see which number
attaches to which object, what a negation now scopes over, or that a figure quoted twice
is now quoted once. It catches careless corruption; it does not certify fidelity. That
is why anything produced by those modes has to be read next to the original, and why the
interface says which guarantee is in force rather than showing the same tick for both.

Everything here is pure and offline. The tests use a fake model precisely because the
checks must hold for *any* output, including adversarial ones.
"""

from __future__ import annotations

import re
import unicodedata

from localasr.refine.types import FidelityIssue, RefinementMode, Severity

# Negation carries meaning inversion, which is the most damaging thing a "tidy-up" can
# silently do. Both scripts, since dictation here is routinely mixed.
_NEGATIONS_ZH = ("不", "没", "无", "非", "未", "别", "勿", "莫", "甭", "否")
_NEGATIONS_EN = (
    "not", "no", "never", "none", "without", "cannot", "cant", "dont", "doesnt",
    "didnt", "wont", "isnt", "arent", "wasnt", "werent", "shouldnt", "couldnt", "nor",
)

_NOT_ACTUALLY_NEGATION = (
    # 别 as "distinguish/particular", never as 别去
    "特别", "区别", "差别", "级别", "类别", "性别", "个别", "分别", "识别", "鉴别",
    "派别", "告别", "别人", "别的", "另别",
    # 非 as "extraordinary/Africa/right-and-wrong"
    "非常", "非洲", "是非", "非凡",
    # 未 as "future"
    "未来",
    # 否 as "whether"
    "是否", "能否", "可否", "与否",
    # 莫 as a name or fixed phrase
    "莫名", "莫过于", "莫斯科",
    # 没 read mò: submerge, not "not have"
    "淹没", "埋没", "出没", "没收",
)
"""Words that contain a negation character without negating anything.

The check below asks whether a negation appeared or vanished, and it does so by looking
for single characters. That is too blunt on its own: 「特别」 is not 「别去」, 「非常」 is
not 「非」, 「是否符合」 asks a question rather than denying anything. Scanning raw text
made every one of those read as an invented negation — and `negation_invented` is an
error in every mode, so a perfectly good rewrite was discarded and the user was shown
their original text back with no visible reason.

Observed: 「现有的方案…尤其是键盘的」 refined to 「请重新梳理现有方案，特别是键盘键位设定」
was rejected for inventing 「别」. Words this common decide whether the feature works at
all, so this is a correctness fix, not a relaxation — the strictness for real negations
is unchanged.
"""

_CJK_DIGITS = "零〇一二三四五六七八九十百千万亿两"
_MEASURE_WORDS = (
    "个人次天年月日时点分秒元块角毛米厘公斤克吨度倍号条张台部件套页章节步项种类"
)

_ASCII_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9._/:\\-]*")
_DIGIT_RUN = re.compile(r"\d+(?:[.,:：/]\d+)*%?")
_CJK_RUN = re.compile(f"[{_CJK_DIGITS}]+")

CONSERVATIVE_MIN_RETENTION = 0.4
"""Below this fraction of the original characters, conservative cleaning has stopped
cleaning and started summarising. Filler and duplicates are a minority of any real
utterance; losing more than half of it means something else happened.

This floor is also the *only* guard against over-deletion, and it is a blunt one. The
subsequence check proves nothing was inserted or substituted; it says nothing about what
was removed. Observed on a real Qwen3.5-4B refinement: 「嗯那个我们下周一要交三个报告」
came back as 「下周一要交三个报告」 — correct filler removal, but 「我们」 went with it,
which is a meaning change that passed at 65% retention. Reading the two panes side by
side is what catches that, which is why the interface shows both rather than replacing
one with the other."""


def _strip_marks(text: str) -> str:
    """Drop punctuation, symbols and whitespace, keeping only content characters.

    Punctuation is the one thing conservative refinement is meant to add, so it has to
    be invisible to the comparison.
    """
    return "".join(
        ch for ch in unicodedata.normalize("NFKC", text)
        if unicodedata.category(ch)[0] not in "PZCS"
    )


def is_subsequence(needle: str, haystack: str) -> bool:
    """True when every character of `needle` appears in `haystack`, in order."""
    iterator = iter(haystack)
    return all(ch in iterator for ch in needle)


def _significant_numbers(text: str) -> frozenset[str]:
    """Numbers whose disappearance would change what was said.

    Western digits always count. A lone CJK numeral does not: 一 occurs inside ordinary
    words (一般, 一起, 十分) and treating those as quantities makes the validator reject
    perfectly good cleaning. A CJK numeral counts when it is long enough to be a real
    quantity, or when a measure word follows it.
    """
    normalised = unicodedata.normalize("NFKC", text)
    found = set(_DIGIT_RUN.findall(normalised))
    for match in _CJK_RUN.finditer(normalised):
        run = match.group()
        following = normalised[match.end() : match.end() + 1]
        if len(run) >= 2 or following in _MEASURE_WORDS:
            found.add(run)
    return frozenset(found)


def _negations(text: str) -> frozenset[str]:
    normalised = unicodedata.normalize("NFKC", text).lower()
    for compound in _NOT_ACTUALLY_NEGATION:
        normalised = normalised.replace(compound, "")
    found = {mark for mark in _NEGATIONS_ZH if mark in normalised}
    words = set(re.findall(r"[a-z']+", normalised))
    stripped = {word.replace("'", "") for word in words}
    found |= {word for word in _NEGATIONS_EN if word in stripped}
    return frozenset(found)


def _ascii_tokens(text: str) -> frozenset[str]:
    """Acronyms, file names, commands and URLs — things a model loves to 'correct'."""
    normalised = unicodedata.normalize("NFKC", text)
    return frozenset(
        token.lower().rstrip(".,:/-")
        for token in _ASCII_TOKEN.findall(normalised)
        if len(token) > 1
    )


def validate(raw: str, refined: str, mode: RefinementMode) -> tuple[FidelityIssue, ...]:
    """Check a refinement against its source. Empty result means it may be shown.

    For `CONSERVATIVE` this is close to a proof. For `PROMPT` it is a screen: passing
    means nothing obviously wrong was found, not that the meaning is intact.
    """
    issues: list[FidelityIssue] = []
    # Asking for a rewrite is asking for things to be left out; a summary that keeps
    # every quantity is not a summary. Fabrication is the failure that stays fatal in
    # every mode, so omission drops to a warning once the user has requested one.
    rewriting = mode is not RefinementMode.CONSERVATIVE
    lost = Severity.WARNING if rewriting else Severity.ERROR
    raw_body = _strip_marks(raw)
    refined_body = _strip_marks(refined)

    if not refined_body:
        return (FidelityIssue("empty", "模型返回了空结果"),)

    if mode is RefinementMode.CONSERVATIVE:
        if not is_subsequence(refined_body, raw_body):
            issues.append(
                FidelityIssue(
                    "not_a_subsequence",
                    "保守清理只允许删除和加标点，但结果包含原文没有的内容",
                )
            )
        retention = len(refined_body) / len(raw_body) if raw_body else 1.0
        if retention < CONSERVATIVE_MIN_RETENTION:
            issues.append(
                FidelityIssue(
                    "over_deletion",
                    f"只保留了原文 {retention:.0%} 的字符，像是摘要而不是清理",
                )
            )

    raw_numbers, refined_numbers = _significant_numbers(raw), _significant_numbers(refined)
    if missing := raw_numbers - refined_numbers:
        issues.append(
            FidelityIssue("number_lost", f"原文中的数字消失了：{sorted(missing)}", lost)
        )
    if added := refined_numbers - raw_numbers:
        issues.append(FidelityIssue("number_invented", f"出现了原文没有的数字：{sorted(added)}"))

    raw_negations, refined_negations = _negations(raw), _negations(refined)
    if missing := raw_negations - refined_negations:
        issues.append(
            FidelityIssue(
                "negation_lost", f"否定词消失，语义可能反转：{sorted(missing)}", lost
            )
        )
    if added := refined_negations - raw_negations:
        issues.append(
            FidelityIssue("negation_invented", f"出现了原文没有的否定词：{sorted(added)}")
        )

    raw_tokens, refined_tokens = _ascii_tokens(raw), _ascii_tokens(refined)
    if missing := raw_tokens - refined_tokens:
        issues.append(
            FidelityIssue("token_lost", f"英文术语/文件名/命令消失：{sorted(missing)}", lost)
        )
    if added := refined_tokens - raw_tokens:
        # A warning, not an error: structured drafting legitimately adds section labels,
        # and refusing those would make the mode unusable.
        issues.append(
            FidelityIssue(
                "token_added",
                f"出现了原文没有的英文词，请核对：{sorted(added)}",
                Severity.WARNING,
            )
        )

    return tuple(issues)
