"""What a journal lets a restart keep, and what it makes it redo.

The rule worth protecting: transcription is expensive and refinement is cheap, so
changing the refiner must never cost a re-decode. Everything here is about that
asymmetry.
"""

from __future__ import annotations

import json

import pytest

from localasr.apps.journal import SCHEMA_VERSION, Journal, JournalHeader
from localasr.core.types import Segment, Span
from localasr.refine.types import (
    FidelityIssue,
    RefinementMode,
    RefinementResult,
    Severity,
)


def _header(**overrides) -> JournalHeader:  # noqa: ANN003
    base = {
        "schema": SCHEMA_VERSION,
        "source": "meeting.wav",
        "source_fingerprint": "abc123",
        "model_id": "qwen3-asr-1_7b-q8",
        "model_revision": "36a67868",
        "language": "zh",
        "refiner_id": "qwen3_5-2b-refiner-q4",
        "refiner_revision": "d7f544ee",
        "template_revision": "2026-08-11.1",
        "refinement_mode": "conservative",
    }
    return JournalHeader(**{**base, **overrides})


def _segment(index: int) -> Segment:
    return Segment(
        span=Span(index * 2.0, index * 2.0 + 1.5),
        text=f"第{index}句",
        utterance_id=f"u{index}",
    )


def _refinement(*ids: str, accepted: bool = True) -> RefinementResult:
    return RefinementResult(
        raw_text="嗯第0句",
        refined_text="第0句。",
        mode=RefinementMode.CONSERVATIVE,
        source_segment_ids=ids,
        issues=() if accepted else (FidelityIssue("number_lost", "数字消失"),),
        model_id="qwen3_5-2b-refiner-q4",
        model_revision="d7f544ee",
        template_revision="2026-08-11.1",
        accepted=accepted,
    )


def _written(tmp_path, header: JournalHeader):  # noqa: ANN001, ANN202
    path = tmp_path / "j.jsonl"
    with Journal(path, header) as journal:
        journal.open(resume=False)
        journal.append(_segment(0))
        journal.append(_segment(1))
        journal.append_refinement(_refinement("u0"))
    return path


def test_segments_and_refinements_both_survive_a_restart(tmp_path) -> None:  # noqa: ANN001
    path = _written(tmp_path, _header())
    state = Journal(path, _header()).resume_state()

    assert [s.text for s in state.segments] == ["第0句", "第1句"]
    assert len(state.refinements) == 1
    assert state.refinements[0].accepted


def test_only_unrefined_utterances_are_redone(tmp_path) -> None:  # noqa: ANN001
    """The whole point of recording refinements: a restart picks up where it stopped."""
    path = _written(tmp_path, _header())
    state = Journal(path, _header()).resume_state()

    assert state.pending_refinement_ids() == ("u1",)


def test_a_changed_refiner_keeps_the_segments_and_drops_the_refinements(tmp_path) -> None:  # noqa: ANN001
    """Swapping the tidy-up model must not cost an hour of re-decoding."""
    path = _written(tmp_path, _header())
    state = Journal(path, _header(refiner_id="something-else")).resume_state()

    assert len(state.segments) == 2, "transcription is expensive and still valid"
    assert state.refinements == []
    assert state.pending_refinement_ids() == ("u0", "u1")


def test_a_changed_prompt_template_also_drops_the_refinements(tmp_path) -> None:  # noqa: ANN001
    """Text tidied under different instructions is not comparable with this run's."""
    path = _written(tmp_path, _header())
    state = Journal(path, _header(template_revision="2030-01-01.9")).resume_state()

    assert len(state.segments) == 2
    assert state.refinements == []


def test_a_changed_mode_drops_the_refinements(tmp_path) -> None:  # noqa: ANN001
    path = _written(tmp_path, _header())
    state = Journal(path, _header(refinement_mode="prompt")).resume_state()
    assert state.refinements == [] and len(state.segments) == 2


def test_a_changed_asr_model_drops_everything(tmp_path) -> None:  # noqa: ANN001
    path = _written(tmp_path, _header())
    state = Journal(path, _header(model_id="qwen3-asr-0_6b-q8")).resume_state()

    assert state.segments == [] and state.refinements == []


def test_a_different_source_file_drops_everything(tmp_path) -> None:  # noqa: ANN001
    path = _written(tmp_path, _header())
    state = Journal(path, _header(source_fingerprint="different")).resume_state()
    assert state.segments == []


def test_a_version_one_journal_is_not_resumed(tmp_path) -> None:  # noqa: ANN001
    """Its rows predate the raw/refined split; guessing which layer `text` was is worse
    than transcribing again."""
    path = tmp_path / "old.jsonl"
    path.write_text(
        json.dumps(
            {
                "schema": 1,
                "source": "meeting.wav",
                "source_fingerprint": "abc123",
                "model_id": "qwen3-asr-1_7b-q8",
                "model_revision": "36a67868",
                "language": "zh",
            }
        )
        + "\n"
        + json.dumps({"start": 0.0, "end": 1.0, "text": "旧的一句"})
        + "\n",
        encoding="utf-8",
    )
    assert Journal(path, _header()).resume_state().segments == []


def test_a_rejected_refinement_is_recorded_so_it_is_not_retried_forever(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "j.jsonl"
    with Journal(path, _header()) as journal:
        journal.open(resume=False)
        journal.append(_segment(0))
        journal.append_refinement(_refinement("u0", accepted=False))

    state = Journal(path, _header()).resume_state()
    stored = state.refinements[0]

    assert not stored.accepted
    assert stored.text == stored.raw_text, "a rejected refinement still shows the original"
    assert stored.errors[0].kind == "number_lost"
    assert stored.errors[0].severity is Severity.ERROR
    assert state.pending_refinement_ids() == (), "tried and refused is not pending"


def test_a_truncated_final_line_is_tolerated(tmp_path) -> None:  # noqa: ANN001
    """Expected after a hard interrupt; it must cost the last row, not the journal."""
    path = _written(tmp_path, _header())
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"type": "segment", "start": 9.0, "en')

    state = Journal(path, _header()).resume_state()
    assert len(state.segments) == 2


def test_refinement_rows_carry_their_provenance(tmp_path) -> None:  # noqa: ANN001
    path = _written(tmp_path, _header())
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    refinement = next(row for row in rows if row.get("type") == "refinement")

    assert refinement["status"] == "completed"
    assert refinement["source_segment_ids"] == ["u0"]
    assert refinement["raw_text"] and refinement["refined_text"]
    assert refinement["model_revision"] and refinement["template_revision"]


def test_writing_without_opening_is_an_error(tmp_path) -> None:  # noqa: ANN001
    with pytest.raises(RuntimeError):
        Journal(tmp_path / "j.jsonl", _header()).append(_segment(0))


# --- the meeting journal, which writes its own JSONL --------------------------


def test_a_meeting_journal_round_trips_refinements(tmp_path) -> None:  # noqa: ANN001
    """Meetings write their own file rather than using Journal, so the same guarantees
    have to be checked separately or they drift apart."""
    from localasr.apps.meeting import load_journal

    path = tmp_path / "meeting.jsonl"
    rows = [
        {"schema": 2, "started_at": "2026-08-11T15:00:00", "sources": ["mic", "system"]},
        {"type": "segment", "start": 0.0, "end": 1.5, "text": "第0句",
         "source": "mic", "utterance_id": "u0"},
        {"type": "segment", "start": 2.0, "end": 3.5, "text": "第1句",
         "source": "system", "utterance_id": "u1"},
        {"type": "refinement", "status": "completed", "accepted": True,
         "mode": "conservative", "source_segment_ids": ["u0"],
         "raw_text": "嗯第0句", "refined_text": "第0句。", "issues": [],
         "model_id": "r", "model_revision": "rev", "template_revision": "tpl"},
    ]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    meeting = load_journal(path)

    assert [s.text for s in meeting.segments] == ["第0句", "第1句"]
    assert meeting.sources == ("mic", "system")
    assert len(meeting.refinements) == 1
    assert meeting.refinements[0].text == "第0句。"
    # Only the unrefined utterance is redone; the meeting is not re-transcribed.
    assert meeting.pending_refinement_ids() == ("u1",)


def test_a_meeting_refinement_row_is_not_mistaken_for_a_segment(tmp_path) -> None:  # noqa: ANN001
    """It carries raw_text/refined_text, never `text`; a reader keying on the wrong
    field would splice model output into the transcript as if it were speech."""
    from localasr.apps.meeting import load_journal

    path = tmp_path / "meeting.jsonl"
    path.write_text(
        json.dumps({"schema": 2, "started_at": "2026-08-11T15:00:00", "sources": []})
        + "\n"
        + json.dumps(
            {"type": "refinement", "accepted": True, "mode": "conservative",
             "source_segment_ids": [], "raw_text": "原", "refined_text": "整理"},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    meeting = load_journal(path)
    assert meeting.segments == []
    assert len(meeting.refinements) == 1
