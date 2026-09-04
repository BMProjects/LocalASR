"""The one path from raw transcript to shown text.

Everything the model produces goes through here, and here the answer to "should the user
see this?" is always the validator's, never the model's. A refinement that fails a check
is returned rejected rather than dropped, so the interface can say what happened instead
of appearing to have done nothing.

A transport failure is also not an error the user has to act on: dictation still works,
it just is not tidied. Both cases degrade to the raw transcript, which is why the raw
transcript is never overwritten in the first place.
"""

from __future__ import annotations

from localasr.refine.client import ChatCompletionClient, RefinementError
from localasr.refine.review import review
from localasr.refine.types import Note, RefinementRequest, RefinementResult


class RefinementService:
    """Refine text, and note anything about the result worth a second look."""

    def __init__(
        self,
        client: ChatCompletionClient,
        *,
        model_id: str = "",
        model_revision: str = "",
    ) -> None:
        """`model_id` is the catalog id of whatever the node loaded.

        Passed in rather than read back from the response: llama-server echoes whatever
        name the request used, so trusting it records an alias — or a file path — where
        the journal needs the pinned catalog id that identifies a revision and a hash.
        """
        self._client = client
        self._model_id = model_id
        self._model_revision = model_revision

    def refine(self, request: RefinementRequest) -> RefinementResult:
        try:
            draft = self._client.refine(request)
        except RefinementError as exc:
            # Not a failure of the dictation, only of the polish on top of it.
            return RefinementResult.failed(
                request,
                f"整理服务不可用：{exc}",
                model_id=self._model_id,
                model_revision=self._model_revision,
            )

        notes = review(request.raw_text, draft.refined_text)
        notes += tuple(Note("model_warning", text) for text in draft.warnings)

        return RefinementResult(
            raw_text=request.raw_text,
            refined_text=draft.refined_text,
            mode=request.mode,
            source_segment_ids=request.source_segment_ids,
            notes=notes,
            model_id=self._model_id,
            model_revision=self._model_revision,
            template_revision=draft.template_revision,
        )
