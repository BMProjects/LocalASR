"""Driving LM Studio's server the way the recognition node is driven.

The wire format here is not ours — it is LM Studio's documented REST API, and the fake
below is written against that documentation rather than against the client. So these
tests are as much a record of the contract as a check of the code:

    GET  /api/v1/models          {"models": [{"key", "type", "loaded_instances": [...]}]}
    POST /api/v1/models/load     {"model": key}      → {"instance_id", "status", ...}
    POST /api/v1/models/unload   {"instance_id": id} → {"instance_id"}

https://lmstudio.ai/docs/api/rest-api
"""

from __future__ import annotations

import json

import httpx
import pytest

from localasr.refine.lmstudio import (
    DEFAULT_URL,
    LMStudioCompanion,
    LMStudioError,
    speaks_lmstudio,
)

URL = "http://127.0.0.1:1234"


class FakeLMStudio:
    """LM Studio's REST API, as documented."""

    def __init__(self, *, models: list[dict] | None = None, up: bool = True) -> None:
        self.up = up
        self.models = models if models is not None else [
            {"type": "llm", "key": "unsloth/Qwen3.5-4B-MTP-GGUF", "loaded_instances": []}
        ]
        self.loads: list[dict] = []
        self.unloads: list[dict] = []
        self.load_status = 200
        self.load_error: dict | None = None

    def _entry(self, key: str) -> dict | None:
        return next((m for m in self.models if m.get("key") == key), None)

    def handle(self, request: httpx.Request) -> httpx.Response:
        if not self.up:
            raise httpx.ConnectError("lm studio is not running", request=request)
        path = request.url.path
        if path == "/api/v1/models" and request.method == "GET":
            return httpx.Response(200, json={"models": self.models})
        if path == "/api/v1/models/load":
            body = json.loads(request.content)
            self.loads.append(body)
            if self.load_status >= 400:
                return httpx.Response(self.load_status, json=self.load_error or {})
            entry = self._entry(body["model"])
            if entry is None:
                return httpx.Response(404, json={"error": "model not found"})
            entry["loaded_instances"] = [{"id": body["model"], "config": {}}]
            return httpx.Response(
                200,
                json={
                    "type": "llm",
                    "instance_id": body["model"],
                    "load_time_seconds": 9.099,
                    "status": "loaded",
                },
            )
        if path == "/api/v1/models/unload":
            body = json.loads(request.content)
            self.unloads.append(body)
            entry = self._entry(body["instance_id"])
            if entry is not None:
                entry["loaded_instances"] = []
            return httpx.Response(200, json={"instance_id": body["instance_id"]})
        return httpx.Response(404, text="Not Found")


@pytest.fixture
def lms(monkeypatch):  # noqa: ANN001, ANN201
    fake = FakeLMStudio()
    original = httpx.Client.__init__

    def patched(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        kwargs["transport"] = httpx.MockTransport(fake.handle)
        original(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", patched)
    return fake


class TestDetection:
    """Which server this is, asked rather than configured."""

    def test_a_models_listing_identifies_lm_studio(self, lms) -> None:  # noqa: ANN001
        assert speaks_lmstudio(URL)

    def test_a_plain_openai_server_is_not_lm_studio(self, monkeypatch) -> None:  # noqa: ANN001
        """llama-server serves /v1/chat/completions and 404s everything else. Treating
        it as LM Studio would put two buttons on screen that can only fail."""
        original = httpx.Client.__init__

        def patched(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
            kwargs["transport"] = httpx.MockTransport(
                lambda request: httpx.Response(404, text="File Not Found")
            )
            original(self, *args, **kwargs)

        monkeypatch.setattr(httpx.Client, "__init__", patched)
        assert not speaks_lmstudio(URL)

    def test_an_unreachable_server_is_not_lm_studio(self, lms) -> None:  # noqa: ANN001
        lms.up = False
        assert not speaks_lmstudio(URL)

    def test_the_default_url_is_lm_studios_documented_one(self) -> None:
        assert DEFAULT_URL == "http://127.0.0.1:1234"


class TestSession:
    """Warm before the first request, and given back after the last one."""

    def test_a_session_loads_the_model_and_unloads_it(self, lms) -> None:  # noqa: ANN001
        companion = LMStudioCompanion(URL)
        companion.start()
        assert lms.loads == [{"model": "unsloth/Qwen3.5-4B-MTP-GGUF"}]

        companion.stop()
        assert lms.unloads == [{"instance_id": "unsloth/Qwen3.5-4B-MTP-GGUF"}]
        assert companion.resident() == ()

    def test_a_warm_model_is_adopted_rather_than_loaded_again(self, lms) -> None:  # noqa: ANN001
        lms.models[0]["loaded_instances"] = [{"id": "unsloth/Qwen3.5-4B-MTP-GGUF"}]
        companion = LMStudioCompanion(URL)

        assert "已加载" in companion.start()
        assert lms.loads == [], "reloading costs seconds to reach the state we are in"

    def test_a_model_this_session_only_adopted_is_still_released_by_default(self, lms) -> None:  # noqa: ANN001
        """The mistake `NodeCompanion` already made once: releasing only what we loaded
        sounds polite and means nothing is ever released, because a server that loads on
        demand is warm by the second launch. ~2.9 GB would stay held forever."""
        lms.models[0]["loaded_instances"] = [{"id": "unsloth/Qwen3.5-4B-MTP-GGUF"}]
        companion = LMStudioCompanion(URL)
        companion.start()
        companion.stop()

        assert lms.unloads == [{"instance_id": "unsloth/Qwen3.5-4B-MTP-GGUF"}]

    def test_release_on_exit_off_leaves_somebody_elses_model_alone(self, lms) -> None:  # noqa: ANN001
        """For an LM Studio whose own chat window is in use."""
        lms.models[0]["loaded_instances"] = [{"id": "unsloth/Qwen3.5-4B-MTP-GGUF"}]
        companion = LMStudioCompanion(URL, release_on_exit=False)
        companion.start()
        companion.stop()

        assert lms.unloads == []

    def test_release_on_exit_off_still_unloads_what_this_session_loaded(self, lms) -> None:  # noqa: ANN001
        companion = LMStudioCompanion(URL, release_on_exit=False)
        companion.start()
        companion.stop()

        assert lms.unloads == [{"instance_id": "unsloth/Qwen3.5-4B-MTP-GGUF"}]

    def test_stopping_without_starting_unloads_nothing(self, lms) -> None:  # noqa: ANN001
        LMStudioCompanion(URL).stop()
        assert lms.unloads == []

    def test_stop_survives_a_server_that_went_away(self, lms) -> None:  # noqa: ANN001
        """Exit must not be blocked by a machine that is already gone."""
        companion = LMStudioCompanion(URL)
        companion.start()
        lms.up = False
        companion.stop()  # must not raise


class TestWhichModel:
    def test_the_only_llm_is_not_a_guess(self, lms) -> None:  # noqa: ANN001
        assert LMStudioCompanion(URL).key() == "unsloth/Qwen3.5-4B-MTP-GGUF"

    def test_an_embedding_model_is_not_a_candidate(self, lms) -> None:  # noqa: ANN001
        lms.models.append({"type": "embedding", "key": "nomic-embed", "loaded_instances": []})
        assert LMStudioCompanion(URL).key() == "unsloth/Qwen3.5-4B-MTP-GGUF"

    def test_several_llms_are_ambiguous_rather_than_guessed(self, lms) -> None:  # noqa: ANN001
        lms.models.append({"type": "llm", "key": "openai/gpt-oss-20b", "loaded_instances": []})
        with pytest.raises(LMStudioError, match="refiner_model"):
            LMStudioCompanion(URL).key()

    def test_a_named_model_wins_without_asking(self, lms) -> None:  # noqa: ANN001
        lms.models.append({"type": "llm", "key": "openai/gpt-oss-20b", "loaded_instances": []})
        companion = LMStudioCompanion(URL, model="openai/gpt-oss-20b")
        companion.start()

        assert lms.loads == [{"model": "openai/gpt-oss-20b"}]

    def test_no_llm_at_all_says_what_to_do(self, lms) -> None:  # noqa: ANN001
        lms.models = []
        with pytest.raises(LMStudioError, match="下载"):
            LMStudioCompanion(URL).key()


class TestFailures:
    def test_a_server_that_is_not_running_says_how_to_start_it(self, lms) -> None:  # noqa: ANN001
        lms.up = False
        with pytest.raises(LMStudioError, match="lms server start"):
            LMStudioCompanion(URL).start()

    def test_a_refused_load_reports_the_servers_reason(self, lms) -> None:  # noqa: ANN001
        """Out of VRAM is the one that actually happens on a 4 GB card."""
        lms.load_status = 400
        lms.load_error = {"error": "Failed to allocate: insufficient VRAM"}
        with pytest.raises(LMStudioError, match="insufficient VRAM"):
            LMStudioCompanion(URL).start()

    def test_a_refused_load_without_a_json_reason_still_reports_something(self, lms) -> None:  # noqa: ANN001
        lms.load_status = 503
        with pytest.raises(LMStudioError, match="503"):
            LMStudioCompanion(URL).start()


def test_a_context_length_is_sent_only_when_asked_for(lms) -> None:  # noqa: ANN001
    LMStudioCompanion(URL, context_length=8192).start()
    assert lms.loads == [{"model": "unsloth/Qwen3.5-4B-MTP-GGUF", "context_length": 8192}]


class TestThroughTheApplication:
    """The buttons, and what they reach.

    `refiner_url` used to mean exactly one thing — "not ours, read-only" — because a URL
    is not a process. LM Studio makes that false: it hands residency over through its
    API, so the same two buttons work for it.
    """

    def _context(self, **settings):  # noqa: ANN001, ANN003, ANN202
        from localasr.context import AppContext, Settings

        return AppContext(settings=Settings(refiner_url=URL, **settings))

    def test_an_lm_studio_url_is_controllable_even_though_it_is_not_ours(self, lms) -> None:  # noqa: ANN001
        context = self._context()
        assert not context.refiner_managed, "we do not spawn it"
        assert context.lmstudio() is not None
        assert context.refiner_controllable, "but we can still load and unload it"

    def test_another_openai_server_stays_read_only(self, monkeypatch) -> None:  # noqa: ANN001
        original = httpx.Client.__init__

        def patched(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
            kwargs["transport"] = httpx.MockTransport(lambda r: httpx.Response(404))
            original(self, *args, **kwargs)

        monkeypatch.setattr(httpx.Client, "__init__", patched)
        context = self._context()

        assert context.lmstudio() is None
        assert not context.refiner_controllable

    def test_the_detection_is_paid_for_once(self, lms) -> None:  # noqa: ANN001
        """The status panel refreshes on a timer. A probe per refresh would be a request
        per few seconds for an answer that cannot change without a restart."""
        context = self._context()
        for _ in range(5):
            context.lmstudio()
        assert len(lms.models[0]["loaded_instances"]) == 0
        assert context.lmstudio() is context.lmstudio()

    def test_start_and_stop_reach_lm_studio(self, lms) -> None:  # noqa: ANN001
        context = self._context()
        context.lmstudio()
        context.start_refiner()
        assert lms.loads == [{"model": "unsloth/Qwen3.5-4B-MTP-GGUF"}]
        assert context.refiner_loaded

        context.stop_refiner()
        assert lms.unloads == [{"instance_id": "unsloth/Qwen3.5-4B-MTP-GGUF"}]

    def test_shutdown_hands_the_model_back(self, lms) -> None:  # noqa: ANN001
        """The whole reason this exists. LM Studio will not unload on its own, so a
        closed window otherwise leaves ~2.9 GB held on a 4 GB card."""
        context = self._context()
        context.lmstudio()
        context.start_refiner()
        context.shutdown()

        assert lms.unloads == [{"instance_id": "unsloth/Qwen3.5-4B-MTP-GGUF"}]

    def test_refiner_model_names_the_key_when_the_user_set_one(self, lms) -> None:  # noqa: ANN001
        lms.models.append({"type": "llm", "key": "openai/gpt-oss-20b", "loaded_instances": []})
        context = self._context(refiner_model="openai/gpt-oss-20b")
        context.start_refiner()

        assert lms.loads == [{"model": "openai/gpt-oss-20b"}]

    def test_the_default_model_name_is_not_mistaken_for_a_key(self, lms) -> None:  # noqa: ANN001
        """`refiner_model` defaults to "localasr-refiner", which is a label llama-server
        echoes back — not a model LM Studio has. Passing it through would ask for a model
        that cannot exist."""
        context = self._context()
        context.start_refiner()

        assert lms.loads == [{"model": "unsloth/Qwen3.5-4B-MTP-GGUF"}]

    def test_release_on_exit_is_configurable(self, lms) -> None:  # noqa: ANN001
        lms.models[0]["loaded_instances"] = [{"id": "unsloth/Qwen3.5-4B-MTP-GGUF"}]
        context = self._context(refiner_release_on_exit=False)
        context.start_refiner()
        context.shutdown()

        assert lms.unloads == []

    def test_a_spawned_refiner_is_untouched_by_any_of_this(self) -> None:
        from localasr.context import AppContext, Settings

        context = AppContext(settings=Settings(refiner_url=None))
        assert context.lmstudio() is None
        assert context.refiner_managed and context.refiner_controllable
