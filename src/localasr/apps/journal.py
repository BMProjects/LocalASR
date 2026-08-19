"""Resumable per-file transcription journals.

Appending finished segments to a JSONL file preserves work across an interrupt, but on
its own it does not survive a *re-run*: the second attempt appends its own copy of every
segment. To resume rather than duplicate, the journal has to record what it is a journal
*of* — which file, which model, which settings — and be discarded when any of those
change.

The fingerprint is size plus mtime plus the first and last megabyte, not a full hash: a
two-hour recording should not be re-read end to end just to decide whether to resume.

Since refinement arrived there are two kinds of row, and — more importantly — two levels
of reuse. Transcribing is expensive and refining is cheap, so a journal written by a
different *refiner* must still yield its segments: changing the tidy-up model is no
reason to re-decode an hour of audio. Only the refinements are discarded.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from localasr.core.types import Segment, Span
from localasr.refine.serde import refinement_from_dict, refinement_to_dict
from localasr.refine.types import RefinementResult

SCHEMA_VERSION = 2
"""Bumped when refinement rows were added. A version 1 journal is not resumed: its rows
predate the raw/refined split, and guessing which layer its `text` belonged to would be
worse than transcribing again."""

_SAMPLE_BYTES = 1 << 20

SEGMENT = "segment"
REFINEMENT = "refinement"


def fingerprint(path: Path) -> str:
    """Cheap identity for a media file."""
    stat = path.stat()
    digest = hashlib.sha256()
    digest.update(f"{stat.st_size}:{int(stat.st_mtime)}".encode())
    with path.open("rb") as handle:
        digest.update(handle.read(_SAMPLE_BYTES))
        if stat.st_size > _SAMPLE_BYTES * 2:
            handle.seek(-_SAMPLE_BYTES, 2)
            digest.update(handle.read(_SAMPLE_BYTES))
    return digest.hexdigest()[:32]


@dataclass(frozen=True, slots=True)
class JournalHeader:
    schema: int
    source: str
    source_fingerprint: str
    model_id: str
    model_revision: str
    language: str | None
    refiner_id: str = ""
    refiner_revision: str = ""
    template_revision: str = ""
    refinement_mode: str = ""

    def matches(self, other: JournalHeader) -> bool:
        """Whether the *segments* may be reused.

        Deliberately blind to the refinement fields: those decide whether the tidy-up
        can be reused, not whether the audio has to be decoded again.
        """
        return (
            self.schema == other.schema
            and self.source_fingerprint == other.source_fingerprint
            and self.model_id == other.model_id
            and self.model_revision == other.model_revision
            and self.language == other.language
        )

    def refinements_match(self, other: JournalHeader) -> bool:
        """Whether the stored refinements may be reused as well.

        The template revision counts: text tidied under different instructions is not
        comparable with text tidied under the current ones, and a journal that cannot
        tell them apart cannot honestly be replayed.
        """
        return (
            self.refiner_id == other.refiner_id
            and self.refiner_revision == other.refiner_revision
            and self.template_revision == other.template_revision
            and self.refinement_mode == other.refinement_mode
        )


@dataclass(slots=True)
class ResumeState:
    """What a previous run left behind that this one can pick up."""

    segments: list[Segment] = field(default_factory=list)
    refinements: list[RefinementResult] = field(default_factory=list)

    def pending_refinement_ids(self) -> tuple[str, ...]:
        """Utterances transcribed but never refined — the only work a restart redoes."""
        done = {sid for result in self.refinements for sid in result.source_segment_ids}
        return tuple(
            segment.utterance_id
            for segment in self.segments
            if segment.utterance_id and segment.utterance_id not in done
        )



class Journal:
    """Append-only record of one media file's segments and their refinements."""

    def __init__(self, path: Path, header: JournalHeader) -> None:
        self.path = Path(path)
        self.header = header
        self._handle = None

    def resume_state(self, *, reuse_refinements: bool = True) -> ResumeState:
        """Everything from a previous run this one may keep.

        Returns nothing when the journal is missing, from another schema, or for
        different inputs — in which case it is truncated on open rather than appended to.
        """
        state = ResumeState()
        if not self.path.is_file():
            return state
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return state
        if not lines:
            return state

        try:
            stored = JournalHeader(**json.loads(lines[0]))
        except (ValueError, TypeError):
            return state
        if not stored.matches(self.header):
            return state
        keep_refinements = reuse_refinements and stored.refinements_match(self.header)

        for line in lines[1:]:
            try:
                row = json.loads(line)
                # Rows written before the discriminator existed are segments.
                kind = row.get("type", SEGMENT)
                if kind == SEGMENT:
                    state.segments.append(
                        Segment(
                            span=Span(row["start"], row["end"]),
                            text=row["text"],
                            source=row.get("source"),
                            utterance_id=row.get("utterance_id"),
                        )
                    )
                elif kind == REFINEMENT and keep_refinements:
                    state.refinements.append(refinement_from_dict(row))
            except (ValueError, KeyError):
                # A partially written final line is expected after a hard interrupt.
                break
        return state

    def resumable(self) -> list[Segment]:
        """Segments only, for callers that do not refine."""
        return self.resume_state().segments

    def open(self, resume: bool) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if resume and self.path.is_file():
            self._handle = self.path.open("a", encoding="utf-8")
            return
        self._handle = self.path.open("w", encoding="utf-8")
        self._write(
            {
                "schema": self.header.schema,
                "source": self.header.source,
                "source_fingerprint": self.header.source_fingerprint,
                "model_id": self.header.model_id,
                "model_revision": self.header.model_revision,
                "language": self.header.language,
                "refiner_id": self.header.refiner_id,
                "refiner_revision": self.header.refiner_revision,
                "template_revision": self.header.template_revision,
                "refinement_mode": self.header.refinement_mode,
            }
        )

    def append(self, segment: Segment) -> None:
        self._write(
            {
                "type": SEGMENT,
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "text": segment.text,
                **({"source": segment.source} if segment.source else {}),
                **({"utterance_id": segment.utterance_id} if segment.utterance_id else {}),
            }
        )

    def append_refinement(self, result: RefinementResult) -> None:
        """Record a refinement, accepted or not.

        Rejected ones are written too. They are the evidence that the tidy-up was tried
        and refused, which is what stops a restart from attempting it again and again,
        and what lets the rejection be explained rather than silently reappearing as
        unrefined text.
        """
        self._write(refinement_to_dict(result))

    def _write(self, payload: dict) -> None:
        if self._handle is None:
            raise RuntimeError("journal is not open")
        self._handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> Journal:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
