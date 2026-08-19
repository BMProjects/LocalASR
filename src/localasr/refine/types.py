"""Domain objects for transcript refinement.

The rule the whole module exists to enforce: **the raw transcript is evidence and is
never overwritten.** A refinement is a separate artefact that points back at the
utterances it came from, carries the model and template that produced it, and can always
be discarded in favour of the original.

That matters because a language model asked to "tidy this up" will, unprompted, supply
causal links that were never spoken, correct numbers it thinks are wrong, normalise
proper nouns, and merge two statements that only sound alike. A prompt cannot prevent
this; only keeping the original and checking the output against it can.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RefinementMode(StrEnum):
    """What the model is permitted to change.

    Two explicit contracts rather than one vague "AI 优化", because the checks that can
    be enforced afterwards differ completely between them.
    """

    CONSERVATIVE = "conservative"
    """Punctuation, paragraphing, filler removal, adjacent duplicate removal. Nothing
    else: order is preserved and no word is substituted. This is verifiable — the output
    must be a subsequence of the input."""

    CUSTOM = "custom"
    """Whatever the user asked for, in their own words.

    Summarise, extract action items, rewrite as a brief — all legitimate, none of them
    a subsequence of the transcript, so the proof that holds for conservative cleaning
    does not apply here and cannot be made to. Only risk screening remains, and the
    interface has to say so: the original stays on screen and the user decides.
    """

    PROMPT = "prompt"
    """Restructure the speech into a task/background/constraints brief. Reordering is
    expected, so subsequence checking does not apply and only the fact-level invariants
    can be enforced."""


class Severity(StrEnum):
    ERROR = "error"
    """The refinement is unusable and the raw transcript is shown instead. Never
    "show it anyway with a warning" — a plausible-looking wrong number is worse than
    visibly unpolished text."""

    WARNING = "warning"
    """Surfaced to the user, refinement still offered."""


@dataclass(frozen=True, slots=True)
class FidelityIssue:
    kind: str
    detail: str
    severity: Severity = Severity.ERROR

    def __str__(self) -> str:
        return f"[{self.severity.value}] {self.kind}: {self.detail}"


@dataclass(frozen=True, slots=True)
class RefinementRequest:
    """One unit of work: this text, under these rules.

    `source_segment_ids` are the utterance ids the raw text was assembled from. They are
    what lets a refined paragraph be traced back to the audio, and what lets a journal
    replay only the refinements that never completed.
    """

    raw_text: str
    source_segment_ids: tuple[str, ...] = ()
    mode: RefinementMode = RefinementMode.CONSERVATIVE
    instruction: str = ""
    """What the user asked for, verbatim, when `mode` is CUSTOM. Ignored otherwise —
    conservative cleaning has one fixed contract and taking instructions would void it."""

    def __post_init__(self) -> None:
        if not self.raw_text.strip():
            raise ValueError("refinement needs non-empty raw text")


@dataclass(frozen=True, slots=True)
class RefinementResult:
    """A refinement plus everything needed to distrust it.

    `accepted` is False when validation rejected the model's output. The object is still
    returned, carrying `issues`, so the interface can explain why the user is looking at
    the raw text — silence would read as the feature being broken.
    """

    raw_text: str
    refined_text: str
    mode: RefinementMode
    source_segment_ids: tuple[str, ...] = ()
    issues: tuple[FidelityIssue, ...] = ()
    model_id: str = ""
    model_revision: str = ""
    template_revision: str = ""
    accepted: bool = True

    @property
    def errors(self) -> tuple[FidelityIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity is Severity.ERROR)

    @property
    def warnings(self) -> tuple[FidelityIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity is Severity.WARNING)

    @property
    def text(self) -> str:
        """What to show. Falls back to the raw transcript whenever validation failed."""
        return self.refined_text if self.accepted else self.raw_text

    @classmethod
    def rejected(
        cls,
        request: RefinementRequest,
        refined_text: str,
        issues: tuple[FidelityIssue, ...],
        **provenance: str,
    ) -> RefinementResult:
        return cls(
            raw_text=request.raw_text,
            refined_text=refined_text,
            mode=request.mode,
            source_segment_ids=request.source_segment_ids,
            issues=issues,
            accepted=False,
            **provenance,
        )

