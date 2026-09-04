"""Asking a LocalASR node to refine text.

Refinement runs on the node, never on the desktop. That is not an arbitrary split: the
laptop's 4 GB card is already holding the ASR model, and a second llama-server beside it
does not fit. So when no node is configured, refinement is simply unavailable — reported
as such, rather than quietly starting a process that would fail to allocate.

The node has already validated whatever it returns; this only carries the verdict across
so the interface can show the original beside the tidied text and say why, when the
answer was refused.
"""

from __future__ import annotations

import httpx

from localasr.refine.serde import result_from_dict
from localasr.refine.types import RefinementRequest, RefinementResult

REFINEMENT_TIMEOUT = 180.0
"""Generous: on a Jetson in sequential mode this request may first evict the ASR model
and load the refiner, which is seconds of disk before any tokens appear."""


class NodeRefiner:
    """Refinement over a node's ``/api/v1/refinements`` endpoint."""

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        timeout: float = REFINEMENT_TIMEOUT,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {token}"} if token else None
        self._client = client or httpx.Client(timeout=timeout, headers=headers)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> NodeRefiner:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def refine(self, request: RefinementRequest) -> RefinementResult:
        """Refine one block of text. Never raises: failure degrades to the raw text.

        A refinement that cannot be produced is not an error the user has to handle —
        the dictation already worked, and the original is right there.
        """
        body = {
            "raw_text": request.raw_text,
            "mode": request.mode.value,
            "source_segment_ids": list(request.source_segment_ids),
        }
        try:
            response = self._client.post(f"{self.base_url}/api/v1/refinements", json=body)
            response.raise_for_status()
            # Parsing belongs inside the guard too. It was outside once, and a payload
            # with an unrecognised severity or a non-list `issues` raised straight
            # through a method documented as never raising — landing in a worker thread
            # that then simply ended, leaving the button re-enabled and nothing shown.
            return result_from_dict(response.json(), raw_text=request.raw_text)
        except httpx.HTTPStatusError as exc:
            return _unavailable(request, f"节点返回 {exc.response.status_code}")
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            return _unavailable(request, f"无法连接整理服务（{exc}）")



def _unavailable(request: RefinementRequest, detail: str) -> RefinementResult:
    return RefinementResult.failed(request, detail)
