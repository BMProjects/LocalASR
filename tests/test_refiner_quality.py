"""A pass/fail gate on whether a refiner model is usable at all.

Skipped unless `LOCALASR_REFINER_URL` points at a running llama-server, because it needs
a real model — but it is a test rather than a script so the bar cannot quietly drift.

The gate exists because "the refinement looked fine" is not a finding. Qwen3.5-2B under
the first prompt copied the prompt's own scaffolding into every answer; under the second
it scored 19/20 on this set with none. That is the difference between a model being
unusable and usable, and one hand-picked sentence could not have told them apart.

    LOCALASR_REFINER_URL=http://127.0.0.1:48459 uv run pytest tests/test_refiner_quality.py
"""

from __future__ import annotations

import json
import os
import time

import httpx
import pytest

from localasr.refine.prompts import RESPONSE_SCHEMA, system_message, user_message
from localasr.refine.types import RefinementMode
from tests.fixtures.refinement_cases import CASES

URL = os.environ.get("LOCALASR_REFINER_URL")
pytestmark = pytest.mark.skipif(not URL, reason="set LOCALASR_REFINER_URL to a llama-server")

MIN_PASSING = 19
"""Out of 20. One failure is tolerated because the validator catches it and the raw text
is shown; a second means the model is guessing at the rules."""

MAX_LATENCY_S = 4.0
"""Co-resident, with no model to load. Beyond this the point of co-residency is gone."""

SCAFFOLD_MARKERS = (
    "三重尖括号", "TRANSCRIPT", "待整理", "语音识别结果", "refined_text", "schema",
)


def _refine(client: httpx.Client, raw: str) -> str:
    body = {
        "model": "gate",
        "messages": [
            {"role": "system", "content": system_message(RefinementMode.CONSERVATIVE)},
            {"role": "user", "content": user_message(raw)},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "n": 1,
        "stream": False,
        "max_tokens": 500,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "refinement", "schema": RESPONSE_SCHEMA},
        },
        "chat_template_kwargs": {"enable_thinking": False},
    }
    response = client.post(f"{URL}/v1/chat/completions", json=body, timeout=180)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return json.loads(content)["refined_text"]


@pytest.fixture(scope="module")
def results():  # noqa: ANN201
    from localasr.refine.fidelity import validate

    out = []
    with httpx.Client() as client:
        for raw, must_keep in CASES:
            started = time.monotonic()
            refined = _refine(client, raw)
            out.append(
                {
                    "raw": raw,
                    "refined": refined,
                    "seconds": time.monotonic() - started,
                    "issues": validate(raw, refined, RefinementMode.CONSERVATIVE),
                    "scaffold": [m for m in SCAFFOLD_MARKERS if m in refined],
                    "lost": [k for k in must_keep if k not in refined],
                }
            )
    return out


def test_the_model_never_echoes_the_prompt(results) -> None:  # noqa: ANN001
    """Zero tolerance. An answer containing the instructions is not a near miss — it
    means the model did not distinguish the task from the material."""
    echoed = [(r["raw"][:20], r["scaffold"]) for r in results if r["scaffold"]]
    assert echoed == []


def test_nothing_meaningful_is_dropped(results) -> None:  # noqa: ANN001
    """Subjects, pronouns, negation, quantities and identifiers. The subsequence check
    permits deletion, so these are the losses no automatic check can catch."""
    lost = [(r["raw"][:20], r["lost"]) for r in results if r["lost"]]
    assert lost == []


def test_enough_refinements_survive_validation(results) -> None:  # noqa: ANN001
    accepted = [r for r in results if not r["issues"]]
    rejected = [(r["raw"][:24], [i.kind for i in r["issues"]]) for r in results if r["issues"]]
    assert len(accepted) >= MIN_PASSING, f"only {len(accepted)}/{len(results)}: {rejected}"


def test_refinement_is_fast_enough_to_be_worth_co_residency(results) -> None:  # noqa: ANN001
    slowest = max(r["seconds"] for r in results)
    assert slowest < MAX_LATENCY_S, f"slowest {slowest:.1f}s"
