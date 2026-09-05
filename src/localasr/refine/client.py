"""Client for an OpenAI-compatible chat server, used only for transcript refinement.

The same bet as the transcription client: the protocol is the boundary, not the model
runtime. Anything serving ``/v1/chat/completions`` can refine — llama-server here, but
nothing in the pipeline knows that.

Sampling is deliberately joyless. Refinement is a transformation, not composition, so
every knob is set against creativity: temperature 0, one sequence, a small context, an
output budget tied to the input length, and JSON enforced by grammar rather than by
asking nicely.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import httpx

from localasr.refine.prompts import (
    RESPONSE_SCHEMA,
    TEMPLATE_REVISION,
    system_message,
    user_message,
)
from localasr.refine.types import RefinementMode, RefinementRequest

REFINEMENT_TIMEOUT = 120.0
DEFAULT_CONTEXT = 4096
"""Small on purpose. Qwen3 offers 32K natively, but KV cache on a Jetson is unified
memory taken from the same pool as the ASR model, and a refinement request never needs
more than a few hundred tokens of speech."""

OUTPUT_BUDGET_RATIO = 2.0
"""Ceiling on output length relative to input. Conservative cleaning can only shrink
text; even structured drafting adding section labels stays well inside 2x. Anything
beyond it is the model writing prose of its own."""


class RefinementError(RuntimeError):
    """Transport failure, or a server that answered with something unusable."""


@dataclass(frozen=True, slots=True)
class RefinementDraft:
    """What the model returned, before any of it has been believed."""

    refined_text: str
    warnings: tuple[str, ...] = ()
    model_id: str = ""
    template_revision: str = TEMPLATE_REVISION


def parse_draft(payload: dict, model_id: str = "") -> RefinementDraft:
    """Read one chat completion. Raises rather than guessing at a malformed answer."""
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RefinementError(f"unexpected chat response shape: {str(payload)[:200]}") from exc

    try:
        body = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RefinementError(f"model did not return JSON: {content[:200]}") from exc
    if not isinstance(body, dict) or not isinstance(body.get("refined_text"), str):
        raise RefinementError(f"response does not match the schema: {content[:200]}")

    warnings = tuple(str(item) for item in body.get("warnings", []) if str(item).strip())
    return RefinementDraft(
        refined_text=body["refined_text"],
        warnings=warnings,
        model_id=model_id or str(payload.get("model", "")),
    )


class ChatCompletionClient:
    """Talks to one chat server. Holds no model state and starts no processes."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8081",
        *,
        timeout: float = REFINEMENT_TIMEOUT,
        client: httpx.Client | None = None,
        model: str = "localasr-refiner",
        token: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        headers = {"Authorization": f"Bearer {token}"} if token else None
        self._client = client or httpx.Client(timeout=timeout, headers=headers)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> ChatCompletionClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def refine(self, request: RefinementRequest) -> RefinementDraft:
        """Ask for one refinement. The answer is unverified; validate it before use."""
        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_message(request.mode, request.instruction),
                },
                {"role": "user", "content": user_message(request.raw_text)},
            ],
            "temperature": 0.0,
            "top_p": 1.0,
            "n": 1,
            "stream": False,
            "max_tokens": self._output_budget(request),
            # llama-server accepts both spellings depending on build; sending the
            # OpenAI one and the llama.cpp one costs nothing and avoids a silently
            # unconstrained answer on whichever build is deployed.
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "refinement", "schema": RESPONSE_SCHEMA},
            },
            "json_schema": RESPONSE_SCHEMA,
            # Qwen3 emits reasoning blocks unless told otherwise; they are pure cost
            # here and would land inside the JSON payload. Two spellings because two
            # servers: llama-server honours the template kwarg, LM Studio ignores it and
            # honours `reasoning_effort`. Measured against LM Studio 0.4.23 serving
            # Qwen3.5-4B — with only the kwarg, every answer came back with an empty
            # `content`, the whole budget spent in `reasoning_content`, and refinement
            # failed as "model did not return JSON".
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_effort": "none",
        }

        try:
            response = self._client.post(f"{self.base_url}/v1/chat/completions", json=body)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:200]
            raise RefinementError(
                f"refiner returned {exc.response.status_code}: {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise RefinementError(f"refinement request failed: {exc}") from exc

        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise RefinementError(f"refiner did not return JSON: {response.text[:200]}") from exc
        return parse_draft(payload, self.model)

    @staticmethod
    def _output_budget(request: RefinementRequest) -> int:
        """Tokens allowed out, derived from what came in.

        Counted in characters rather than tokens: CJK runs near one token per character
        and this only needs to be an upper bound, not an estimate.
        """
        base = int(len(request.raw_text) * OUTPUT_BUDGET_RATIO)
        floor = 128 if request.mode is RefinementMode.CONSERVATIVE else 256
        return max(floor, min(base, DEFAULT_CONTEXT // 2))
