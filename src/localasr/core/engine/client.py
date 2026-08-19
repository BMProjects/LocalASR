"""Client for an OpenAI-compatible transcription server (llama-server by default).

The protocol boundary lives here: anything speaking ``/v1/audio/transcriptions`` can
back the whole application, so swapping the engine never reaches the pipeline.

"OpenAI-compatible" names a family, not a contract. The official protocol spans plain
JSON, ``verbose_json``, segment and word timestamps, diarized output and SSE streaming;
local servers implement different subsets, and llama.cpp's audio support is flagged
upstream as experimental. So capabilities are negotiated at startup rather than
assumed, and a missing capability is reported as such instead of surfacing as a
transport error.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import httpx

from localasr.core.audio.decode import to_wav_bytes
from localasr.core.types import Audio, Word

_ASR_TEXT_MARKER = "<asr_text>"
_LANGUAGE_PREFIX = re.compile(r"^\s*language\s+(?P<lang>[A-Za-z\- ]+?)\s*$")
TRANSCRIPTION_TIMEOUT = 90.0
"""Shorter than LiveSession's 120 s stop deadline, so a stuck HTTP request can unwind."""


class EngineError(RuntimeError):
    """Transport failure, or a server that answered but not usefully."""


class ProtocolError(EngineError):
    """The server answered with something this client cannot interpret."""


class CapabilityError(EngineError):
    """The server works but does not offer a feature that was asked for."""


@dataclass(frozen=True, slots=True)
class EngineCapabilities:
    """What the connected server actually supports, established at startup."""

    audio: bool
    verbose_json: bool
    segment_timestamps: bool
    word_timestamps: bool
    model_path: str | None = None
    build: str | None = None

    def require_audio(self) -> None:
        if not self.audio:
            raise CapabilityError(
                f"server has no audio modality (model: {self.model_path}); "
                "it is loaded with a text-only model"
            )


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """One transcription. `segments` and `words` stay empty unless the server sends
    them — they are never synthesised from text length."""

    text: str
    language: str | None = None
    segments: list[tuple[float, float, str]] = field(default_factory=list)
    words: list[Word] = field(default_factory=list)


def parse_asr_payload(raw: str) -> TranscriptionResult:
    """Split Qwen3-ASR's raw decode into language tag and transcript.

    llama-server returns the model's literal output, which for Qwen3-ASR looks like
    ``language English<asr_text>And so, my fellow Americans...``. Clients assuming a
    bare transcript therefore leak the tag into their output (ggml-org/llama.cpp#26749,
    open as of 2026-08). Output without the marker passes through unchanged, so this
    stays correct once upstream fixes it and can then be deleted.
    """
    if _ASR_TEXT_MARKER not in raw:
        return TranscriptionResult(text=raw.strip())

    head, _, tail = raw.partition(_ASR_TEXT_MARKER)
    match = _LANGUAGE_PREFIX.match(head)
    language = match.group("lang") if match else None
    return TranscriptionResult(text=tail.strip(), language=language)


def parse_response(payload: object) -> TranscriptionResult:
    """Interpret a transcription response body.

    Accepts the plain ``{"text": ...}`` shape and the richer ``verbose_json`` shape,
    carrying segment and word timings through when the server provides them.
    """
    if not isinstance(payload, dict):
        raise ProtocolError(f"expected a JSON object, got {type(payload).__name__}")

    raw_text = payload.get("text")
    if raw_text is None:
        raise ProtocolError(f"response has no 'text' field: {sorted(payload)}")
    if not isinstance(raw_text, str):
        raise ProtocolError(f"'text' is {type(raw_text).__name__}, expected string")

    result = parse_asr_payload(raw_text)

    segments: list[tuple[float, float, str]] = []
    for item in payload.get("segments") or []:
        if isinstance(item, dict) and {"start", "end", "text"} <= item.keys():
            segments.append((float(item["start"]), float(item["end"]), str(item["text"])))

    words: list[Word] = []
    for item in payload.get("words") or []:
        if isinstance(item, dict) and {"start", "end"} <= item.keys():
            text = item.get("word") or item.get("text") or ""
            words.append(Word(text=str(text), start=float(item["start"]), end=float(item["end"])))

    return TranscriptionResult(
        text=result.text,
        language=result.language or payload.get("language"),
        segments=segments,
        words=words,
    )


class TranscriptionClient:
    """Thin, synchronous client. One instance is safe to reuse across requests."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        timeout: float = TRANSCRIPTION_TIMEOUT,
        *,
        token: str | None = None,
    ) -> None:
        """`token` is what makes this client reach a remote node.

        A local llama-server needs none. A node on the LAN does, and the same client
        serves both: the deployment change is a base URL and a token, not a code path.
        """
        self.base_url = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {token}"} if token else None
        self._client = httpx.Client(timeout=timeout, headers=headers)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> TranscriptionClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def is_ready(self) -> bool:
        try:
            response = self._client.get(f"{self.base_url}/health", timeout=2.0)
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    def capabilities(self) -> EngineCapabilities:
        """Read what the server advertises.

        `/props` reports modalities reliably; the timestamp fields are probed by
        `negotiate()`, which needs a real request to tell support from silence.
        """
        try:
            response = self._client.get(f"{self.base_url}/props", timeout=5.0)
            response.raise_for_status()
            props = response.json()
        except httpx.HTTPError as exc:
            raise EngineError(f"cannot read server props at {self.base_url}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ProtocolError(f"/props did not return JSON: {exc}") from exc

        modalities = props.get("modalities") or {}
        return EngineCapabilities(
            audio=bool(modalities.get("audio")),
            verbose_json=False,
            segment_timestamps=False,
            word_timestamps=False,
            model_path=props.get("model_path"),
            build=props.get("build_info"),
        )

    def negotiate(self, probe: Audio) -> EngineCapabilities:
        """Confirm the server can actually transcribe, and detect richer formats.

        A short real request is the only reliable probe: `/props` says nothing about
        which response formats the transcription route accepts.
        """
        capabilities = self.capabilities()
        capabilities.require_audio()

        self.transcribe(probe)

        verbose = self._probe_verbose_json(probe)
        return EngineCapabilities(
            audio=True,
            verbose_json=verbose is not None,
            segment_timestamps=bool(verbose and verbose.segments),
            word_timestamps=bool(verbose and verbose.words),
            model_path=capabilities.model_path,
            build=capabilities.build,
        )

    def _probe_verbose_json(self, probe: Audio) -> TranscriptionResult | None:
        try:
            return self.transcribe(probe, response_format="verbose_json")
        except EngineError:
            return None

    def transcribe(
        self,
        audio: Audio,
        *,
        language: str | None = None,
        prompt: str | None = None,
        response_format: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe one audio span.

        `language` forces the decode language (ISO code such as ``zh``); ``None`` lets
        the model identify it, which it reports back in the result.
        """
        data: dict[str, str] = {"model": "localasr"}
        if language:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt
        if response_format:
            data["response_format"] = response_format

        try:
            response = self._client.post(
                f"{self.base_url}/v1/audio/transcriptions",
                files={"file": ("audio.wav", to_wav_bytes(audio), "audio/wav")},
                data=data,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:200]
            raise EngineError(f"engine returned {exc.response.status_code}: {detail}") from exc
        except httpx.HTTPError as exc:
            raise EngineError(f"transcription request failed: {exc}") from exc

        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise ProtocolError(f"engine did not return JSON: {response.text[:200]}") from exc

        return parse_response(payload)
