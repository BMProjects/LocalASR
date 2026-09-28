"""The LocalASR compute node.

One process in front of the model servers, rather than letting every desktop talk to
llama-server directly. That indirection buys the four things that otherwise have to be
reimplemented in each frontend: residency policy on shared memory, a single
authenticated surface, one place where model state is known, and errors that mean
something above the transport layer.

The transcription endpoint deliberately keeps its OpenAI-compatible path and shape, so
the existing `TranscriptionClient` reaches it unchanged with only a different base URL.
Refinement gets a first-party endpoint because its contract — validated, with the raw
text preserved on rejection — is ours and has no OpenAI equivalent.
"""

from __future__ import annotations

import os
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from localasr.node.coordinator import (
    DEFAULT_MIN_FREE_MB,
    Memory,
    ResourceCoordinator,
)
from localasr.refine.client import ChatCompletionClient
from localasr.refine.serde import result_to_dict
from localasr.refine.service import RefinementService
from localasr.refine.types import (
    RefinementMode,
    RefinementRequest,
    RefinementResult,
)
from localasr.registry import manager
from localasr.registry.manager import ModelKind

DEFAULT_PORT = 8090


@dataclass(frozen=True, slots=True)
class NodeConfig:
    """How this node exposes itself. Every default is the closed one."""

    bind: str = "127.0.0.1"
    port: int = DEFAULT_PORT
    token: str | None = None
    min_free_mb: int = DEFAULT_MIN_FREE_MB
    asr_model: str | None = None
    llm_model: str | None = None
    log_dir: Path | None = None

    @classmethod
    def from_env(cls) -> NodeConfig:
        return cls(
            bind=os.environ.get("LOCALASR_NODE_BIND", "127.0.0.1"),
            port=int(os.environ.get("LOCALASR_NODE_PORT", DEFAULT_PORT)),
            token=os.environ.get("LOCALASR_NODE_TOKEN") or None,
            min_free_mb=int(
                os.environ.get("LOCALASR_NODE_MIN_FREE_MB", DEFAULT_MIN_FREE_MB)
            ),
            asr_model=os.environ.get("LOCALASR_NODE_ASR_MODEL") or None,
            llm_model=os.environ.get("LOCALASR_NODE_LLM_MODEL") or None,
            log_dir=Path(os.environ["LOCALASR_NODE_LOG_DIR"])
            if os.environ.get("LOCALASR_NODE_LOG_DIR")
            else None,
        )

    def listens_beyond_loopback(self) -> bool:
        return self.bind not in {"127.0.0.1", "localhost", "::1"}


def create_app(config: NodeConfig | None = None) -> FastAPI:
    # These imports are module-level on purpose. With `from __future__ import
    # annotations` every annotation is a string that FastAPI resolves against module
    # globals; importing Header/File/Form inside this function left them unresolvable,
    # so `authorization` silently became a query parameter and every token was refused.
    config = config or NodeConfig.from_env()
    if config.listens_beyond_loopback() and not config.token:
        # A node on the LAN with no token is an open transcription service for everyone
        # on the network, and it holds whatever is said near the microphone.
        raise ValueError(
            "refusing to listen on a non-loopback address without LOCALASR_NODE_TOKEN"
        )

    coordinator = ResourceCoordinator(
        min_free_mb=config.min_free_mb,
        log_dir=config.log_dir,
    )
    @asynccontextmanager
    async def lifespan(_app: FastAPI):  # noqa: ANN202
        yield
        # Models outlive requests but must not outlive the process: a llama-server left
        # running holds the memory the next node start needs.
        coordinator.shutdown()

    app = FastAPI(title="LocalASR node", version="1", lifespan=lifespan)
    app.state.config = config
    app.state.coordinator = coordinator

    def authorise(authorization: Annotated[str | None, Header()] = None) -> None:
        if config.token is None:
            return
        expected = f"Bearer {config.token}"
        # Constant-time: a token check that returns early leaks its prefix.
        if authorization is None or not secrets.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    guard = [Depends(authorise)]

    @app.get("/healthz")
    @app.get("/health")
    def healthz() -> dict:
        """Liveness only: the process answers. Says nothing about models.

        `/health` is the spelling llama-server uses and therefore the one
        `TranscriptionClient.is_ready()` probes. Serving both is what lets an unmodified
        client point at a node: without the alias every readiness check 404s and the
        desktop silently falls back to its local engine, which looks exactly like the
        node not existing.

        Readiness deliberately does *not* answer here. The node loads ASR on demand, so
        before the first request `/readyz` is legitimately 503 — probing that instead
        would refuse a node that is working perfectly.
        """
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> JSONResponse:
        memory = Memory.read()
        loaded = coordinator.loaded()
        body = {
            "loaded": loaded,
            "loading": coordinator.loading(),
            "memory_mb": (
                {"total": memory.total_mb, "available": memory.available_mb}
                if memory
                else None
            ),
        }
        # Ready means a transcription would not have to load a model first.
        ready = ModelKind.ASR.value in loaded
        return JSONResponse(body, status_code=200 if ready else 503)

    @app.get("/api/v1/models", dependencies=guard)
    def models() -> dict:
        loaded = coordinator.loaded()
        return {
            "models": [
                {
                    "id": spec.model_id,
                    "kind": spec.kind.value,
                    "revision": spec.revision,
                    "size": sum(f.size for f in spec.files),
                    "downloaded": manager.is_downloaded(spec),
                    "loaded": loaded.get(spec.kind.value) == spec.model_id,
                    "default": spec.default,
                }
                for spec in manager.list_models()
            ],
        }

    @app.post("/api/v1/models/load", dependencies=guard)
    def load_models(body: dict | None = None) -> JSONResponse:
        """Make a role resident now, rather than on the first request that needs it.

        A cold load is 25-40 s on a Jetson. Paying that when someone presses 开始识别 —
        after they have already started talking — is the difference between a tool that
        feels instant and one that seems broken; paying it up front, deliberately, does
        not interrupt anything.
        """
        wanted = (body or {}).get("kind", ModelKind.ASR.value)
        try:
            kind = ModelKind(wanted)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"unknown kind: {wanted}") from exc

        configured = config.asr_model if kind is ModelKind.ASR else config.llm_model
        try:
            spec = manager.get_model(configured, kind)
            coordinator.ensure(spec)
        except Exception as exc:  # noqa: BLE001 - a load failure is not a server fault
            # The caller asked for a model and cannot have one. That is an answer, not a
            # crash, and it needs to say which model and why.
            return JSONResponse(
                {"loaded": coordinator.loaded(), "error": str(exc)}, status_code=409
            )
        memory = Memory.read()
        return JSONResponse(
            {
                "loaded": coordinator.loaded(),
                "memory_mb": {"available": memory.available_mb} if memory else None,
            }
        )

    @app.post("/api/v1/models/release", dependencies=guard)
    def release_models(body: dict | None = None) -> dict:
        """Evict resident models, freeing the node's memory.

        The next request loads again — this is a way to hand the board back to whatever
        else needs it, not a way to switch anything off. Releasing a role that is not
        resident is not an error: the caller wanted it gone, and it is.
        """
        wanted = (body or {}).get("kind")
        before = coordinator.loaded()
        if wanted:
            try:
                coordinator.release(ModelKind(wanted))
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=f"unknown kind: {wanted}") from exc
        else:
            coordinator.shutdown()
        memory = Memory.read()
        return {
            "released": [k for k in before if k not in coordinator.loaded()],
            "loaded": coordinator.loaded(),
            "memory_mb": {"available": memory.available_mb} if memory else None,
        }

    @app.post("/v1/audio/transcriptions", dependencies=guard)
    async def transcriptions(
        file: Annotated[UploadFile, File()],
        model: Annotated[str | None, Form()] = None,
        language: Annotated[str | None, Form()] = None,
        response_format: Annotated[str | None, Form()] = None,
    ) -> JSONResponse:
        """OpenAI-shaped, so an unmodified client reaches it by changing base_url only."""
        spec = manager.get_model(config.asr_model, ModelKind.ASR)
        base = coordinator.ensure(spec)
        payload = await file.read()
        data = {"model": model or "localasr"}
        if language:
            data["language"] = language
        if response_format:
            data["response_format"] = response_format

        async with httpx.AsyncClient(timeout=120.0) as client:
            upstream = await client.post(
                f"{base}/v1/audio/transcriptions",
                files={"file": (file.filename or "audio.wav", payload, "audio/wav")},
                data=data,
            )
        return JSONResponse(upstream.json(), status_code=upstream.status_code)

    @app.post("/api/v1/refinements", dependencies=guard)
    def refinements(body: dict) -> dict:
        raw_text = str(body.get("raw_text", ""))
        if not raw_text.strip():
            raise HTTPException(status_code=422, detail="raw_text is required")
        try:
            mode = RefinementMode(body.get("mode", RefinementMode.CONSERVATIVE.value))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"unknown mode: {exc}") from exc

        spec = manager.get_model(config.llm_model, ModelKind.LLM)
        try:
            base = coordinator.ensure(spec)
        except Exception as exc:  # noqa: BLE001 - a load failure is not a server fault
            # Refusals belong in the payload, not in a 500. The client's contract is
            # that refinement degrades to the raw text with a stated reason; an
            # unhandled exception reaches it as "节点返回 500", which says nothing about
            # a model that would not fit or a llama-server that would not start.
            return {
                **result_to_dict(
                    RefinementResult.failed(
                        RefinementRequest(raw_text=raw_text, mode=mode),
                        f"节点无法加载整理模型：{exc}",
                    )
                ),
                "text": raw_text,
            }
        request = RefinementRequest(
            raw_text=raw_text,
            source_segment_ids=tuple(str(x) for x in body.get("source_segment_ids", [])),
            mode=mode,
        )
        with ChatCompletionClient(base) as client:
            service = RefinementService(
                client, model_id=spec.model_id, model_revision=spec.revision
            )
            result = service.refine(request)

        # One serialiser, shared with the journals and the desktop client, so a field
        # added in one place cannot go missing in the other two.
        return {**result_to_dict(result), "text": result.text}

    return app


def main() -> int:
    import uvicorn

    config = NodeConfig.from_env()
    app = create_app(config)
    uvicorn.run(app, host=config.bind, port=config.port, log_level="info")
    return 0
