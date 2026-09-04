"""The one JSON shape of a refinement, for everyone who writes one down.

A `RefinementResult` crosses three boundaries — the node's HTTP response, the desktop
client reading it, and both journals — and it used to be hand-converted at each. Three
copies of a format is three places to forget a field, and the day one grows a key the
others do not is the day an export quietly loses it.

`result_from_dict` is total: any payload at all produces a result rather than an
exception. That is what lets the callers promise never to fail on a refinement, since a
malformed answer from a model server is a thing that happens and is never worth taking
the dictation down for.
"""

from __future__ import annotations

from localasr.refine.types import Note, RefinementMode, RefinementResult

ROW_TYPE = "refinement"


def result_to_dict(result: RefinementResult) -> dict:
    """Serialise one refinement, whether or not it produced anything."""
    return {
        "failure": result.failure,
        "mode": result.mode.value,
        "source_segment_ids": list(result.source_segment_ids),
        "raw_text": result.raw_text,
        "refined_text": result.refined_text,
        "notes": [{"kind": n.kind, "detail": n.detail} for n in result.notes],
        "model_id": result.model_id,
        "model_revision": result.model_revision,
        "template_revision": result.template_revision,
    }


def _notes_from(raw: object) -> tuple[Note, ...]:
    """Read the note list, discarding anything that is not one."""
    if not isinstance(raw, list):
        return ()
    return tuple(
        Note(kind=str(item.get("kind", "")), detail=str(item.get("detail", "")))
        for item in raw
        if isinstance(item, dict)
    )


def result_from_dict(payload: object, *, raw_text: str | None = None) -> RefinementResult:
    """Rebuild a refinement from any payload, never raising.

    `raw_text` overrides whatever the payload claims. The caller holds the text the user
    is actually looking at, and that — not a server's echo of it — is what a refinement
    that never happened has to fall back to.
    """
    body = payload if isinstance(payload, dict) else {}
    try:
        mode = RefinementMode(body.get("mode", RefinementMode.CONSERVATIVE.value))
    except ValueError:
        mode = RefinementMode.CONSERVATIVE

    ids = body.get("source_segment_ids")
    return RefinementResult(
        raw_text=raw_text if raw_text is not None else str(body.get("raw_text", "")),
        refined_text=str(body.get("refined_text", "")),
        mode=mode,
        source_segment_ids=tuple(str(x) for x in ids) if isinstance(ids, list) else (),
        notes=_notes_from(body.get("notes")),
        model_id=str(body.get("model_id", "")),
        model_revision=str(body.get("model_revision", "")),
        template_revision=str(body.get("template_revision", "")),
        failure=(str(failure) if (failure := body.get("failure")) else None),
    )


def refinement_to_dict(result: RefinementResult) -> dict:
    """A journal row: the shared shape plus what a journal needs to replay it."""
    return {
        "type": ROW_TYPE,
        "status": "completed" if result.ok else "failed",
        **result_to_dict(result),
    }


def refinement_from_dict(row: dict) -> RefinementResult:
    return result_from_dict(row)
