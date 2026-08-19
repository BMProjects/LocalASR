"""Contract tests for the transcription protocol.

The fixtures are real llama-server b10333 responses, recorded while serving
Qwen3-ASR. "OpenAI-compatible" is a family rather than a contract, so these lock in
what this client must tolerate: the `<asr_text>` prefix, the absent-timestamp case,
and the failure modes that must be reported as protocol or capability errors rather
than crashing.
"""

import json

import httpx
import pytest

from localasr.core.engine.client import (
    CapabilityError,
    EngineCapabilities,
    EngineError,
    ProtocolError,
    TranscriptionClient,
    parse_asr_payload,
    parse_response,
)
from localasr.core.types import SAMPLE_RATE, Audio

# Recorded from: curl -F file=@jfk.wav -F model=w localhost:8080/v1/audio/transcriptions
LLAMA_SERVER_B10333 = {
    "type": "transcript.text.done",
    "text": "language English<asr_text>And so, my fellow Americans, ask not what your "
    "country can do for you; ask what you can do for your country.",
    "usage": {
        "type": "tokens",
        "input_tokens": 171,
        "output_tokens": 30,
        "total_tokens": 201,
        "input_tokens_details": {"cached_tokens": 0},
    },
}

# Recorded from the same server with response_format=verbose_json.
LLAMA_SERVER_VERBOSE_REJECTED = {
    "error": {
        "code": 400,
        "message": "Only 'json' response_format is supported for transcription",
        "type": "invalid_request_error",
    }
}


def _audio(seconds: float = 0.5) -> Audio:
    import numpy as np

    return Audio(samples=np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32))


def _client(handler) -> TranscriptionClient:  # noqa: ANN001
    client = TranscriptionClient("http://engine.test")
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    return client


def test_parses_recorded_llama_server_response():
    result = parse_response(LLAMA_SERVER_B10333)
    assert result.text.startswith("And so, my fellow Americans")
    assert "<asr_text>" not in result.text
    assert result.language == "English"


def test_recorded_response_carries_no_timestamps():
    result = parse_response(LLAMA_SERVER_B10333)
    assert result.segments == []
    assert result.words == []


def test_carries_segments_and_words_when_a_server_provides_them():
    payload = {
        "text": "hello world",
        "language": "English",
        "segments": [{"start": 0.0, "end": 1.0, "text": "hello world"}],
        "words": [{"word": "hello", "start": 0.0, "end": 0.4}],
    }
    result = parse_response(payload)
    assert result.segments == [(0.0, 1.0, "hello world")]
    assert result.words[0].text == "hello"
    assert result.words[0].end == 0.4


def test_rejects_non_object_body():
    with pytest.raises(ProtocolError):
        parse_response(["not", "an", "object"])


def test_rejects_body_without_text():
    with pytest.raises(ProtocolError):
        parse_response({"usage": {}})


def test_rejects_non_string_text():
    with pytest.raises(ProtocolError):
        parse_response({"text": 42})


def test_ignores_malformed_entries_in_segments():
    result = parse_response({"text": "x", "segments": [{"start": 0.0}, "junk"]})
    assert result.segments == []


def test_http_error_becomes_engine_error_with_detail():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json=LLAMA_SERVER_VERBOSE_REJECTED)

    with _client(handler) as client, pytest.raises(EngineError) as excinfo:
        client.transcribe(_audio(), response_format="verbose_json")
    assert "400" in str(excinfo.value)


def test_non_json_body_becomes_protocol_error():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>gateway</html>")

    with _client(handler) as client, pytest.raises(ProtocolError):
        client.transcribe(_audio())


def test_transport_failure_becomes_engine_error():
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with _client(handler) as client, pytest.raises(EngineError):
        client.transcribe(_audio())


def test_capabilities_read_audio_modality_from_props():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model_path": "models/Qwen3-ASR-1.7B-Q8_0.gguf",
                "modalities": {"vision": False, "video": False, "audio": True},
                "build_info": "b10333-08659901c",
            },
        )

    with _client(handler) as client:
        capabilities = client.capabilities()
    assert capabilities.audio is True
    assert capabilities.build == "b10333-08659901c"


def test_text_only_model_raises_capability_error_not_engine_error():
    capabilities = EngineCapabilities(
        audio=False,
        verbose_json=False,
        segment_timestamps=False,
        word_timestamps=False,
        model_path="some-text-model.gguf",
    )
    with pytest.raises(CapabilityError):
        capabilities.require_audio()


def test_negotiate_reports_verbose_json_unsupported_against_recorded_behaviour():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/props":
            return httpx.Response(200, json={"modalities": {"audio": True}})
        body = request.content
        if b"verbose_json" in body:
            return httpx.Response(400, json=LLAMA_SERVER_VERBOSE_REJECTED)
        return httpx.Response(200, json=LLAMA_SERVER_B10333)

    with _client(handler) as client:
        capabilities = client.negotiate(_audio())

    assert capabilities.audio is True
    assert capabilities.verbose_json is False
    assert capabilities.segment_timestamps is False
    assert capabilities.word_timestamps is False


def test_negotiate_detects_a_server_that_does_support_word_timestamps():
    verbose = {
        "text": "hello",
        "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
        "words": [{"word": "hello", "start": 0.0, "end": 1.0}],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/props":
            return httpx.Response(200, json={"modalities": {"audio": True}})
        if b"verbose_json" in request.content:
            return httpx.Response(200, json=verbose)
        return httpx.Response(200, json={"text": "hello"})

    with _client(handler) as client:
        capabilities = client.negotiate(_audio())

    assert capabilities.verbose_json is True
    assert capabilities.word_timestamps is True


def test_asr_text_prefix_removal_is_a_no_op_once_upstream_fixes_it():
    """The workaround must not corrupt output after llama.cpp#26749 is resolved."""
    assert parse_asr_payload("plain transcript").text == "plain transcript"


def test_recorded_fixture_is_valid_json():
    assert json.loads(json.dumps(LLAMA_SERVER_B10333))["text"]
