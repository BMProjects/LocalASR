"""Domain objects for transcript refinement.

The rule the module exists to enforce: **the raw transcript is evidence and is never
overwritten.** A refinement is a separate artefact that points back at the utterances it
came from and carries the model and template that produced it, so the two can always be
read against each other.

Kept, not gated. There used to be a validator here that could refuse a refinement and
show the transcript in its place. It was removed: substitution is what makes refinement
worth having on speech — recognition returns homophones, and inferring the intended word
from a wrong one that sounds like it is the main job — while the interface keeps both
texts on screen for a person who is watching the whole time anyway. See `review.py`.
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
    """Punctuation, paragraphing, filler removal, and correcting words recognition
    clearly misheard. Meaning and order are kept; nothing is summarised or restructured.

    Substitution is allowed on purpose. A transcript is full of homophones, and refusing
    to fix them was refusing the most useful thing this mode can do."""

    CUSTOM = "custom"
    """Whatever the user asked for, in their own words.

    Summarise, extract action items, rewrite as a brief. The original stays on screen
    beside it and the user decides.
    """

    PROMPT = "prompt"
    """Restructure the speech into a task/background/constraints brief."""


@dataclass(frozen=True, slots=True)
class Note:
    """Something worth a second look. Never a reason to withhold the refinement."""

    kind: str
    detail: str

    def __str__(self) -> str:
        return f"{self.kind}: {self.detail}"


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
    """A refinement, the transcript it came from, and what produced it.

    `failure` separates the two things that used to be conflated under "not accepted":
    a refinement that did not happen at all — no refiner configured, the request failed —
    from one that happened and might be imperfect. Only the first has nothing to show, and
    only the first falls back to the transcript.
    """

    raw_text: str
    refined_text: str
    mode: RefinementMode
    source_segment_ids: tuple[str, ...] = ()
    notes: tuple[Note, ...] = ()
    """Advisory, always. Nothing here withholds the refinement."""

    model_id: str = ""
    model_revision: str = ""
    template_revision: str = ""
    failure: str | None = None
    """Why no refinement exists, when none does."""

    @property
    def ok(self) -> bool:
        return self.failure is None

    @property
    def text(self) -> str:
        """What to show: the refinement, or the transcript when there is no refinement."""
        return self.raw_text if self.failure else self.refined_text

    @classmethod
    def failed(cls, request: RefinementRequest, reason: str, **provenance: str) -> RefinementResult:
        return cls(
            raw_text=request.raw_text,
            refined_text="",
            mode=request.mode,
            source_segment_ids=request.source_segment_ids,
            failure=reason,
            **provenance,
        )
