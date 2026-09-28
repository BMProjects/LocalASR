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


@pytest.fixture(autouse=True)
def _no_real_cli(monkeypatch):  # noqa: ANN001, ANN201
    """Nothing in this file may run the real `lms`.

    It is installed on the development machine, `lms daemon up` blocks for sixty seconds
    before failing, and a test that starts LM Studio is not a unit test.
    """
    from localasr.refine import lmstudio

    monkeypatch.setattr(lmstudio, "find_cli", lambda: None)
    monkeypatch.setattr(lmstudio, "SERVER_SETTLE", 0.0)


@pytest.fixture
def lms(monkeypatch):  # noqa: ANN001, ANN201
    fake = FakeLMStudio()
    original = httpx.Client.__init__

    def patched(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        kwargs["transport"] = httpx.MockTransport(fake.handle)
        original(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", patched)
    return fake


@pytest.fixture
def cli(monkeypatch, tmp_path):  # noqa: ANN001, ANN201
    """A stand-in for `lms` that records what it was asked to do."""
    from localasr.refine import lmstudio

    calls: list[list[str]] = []
    fake = tmp_path / "lms"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)
    monkeypatch.setattr(lmstudio, "find_cli", lambda: fake)

    class Result:
        def __init__(self, stdout: str = "") -> None:
            self.returncode = 0
            self.stdout = stdout
            self.stderr = ""

    state = {"daemon": False}

    def run(command, **kwargs):  # noqa: ANN001, ANN003, ANN202
        args = list(command[1:])
        calls.append(args)
        if args[:2] == ["daemon", "status"]:
            # `lms daemon status --json`, which exits 0 whatever the answer.
            status = "running" if state["daemon"] else "not-running"
            return Result(f'{{"status": "{status}"}}')
        if args == ["daemon", "up"]:
            state["daemon"] = True
        elif args == ["daemon", "down"]:
            state["daemon"] = False
        return Result()

    monkeypatch.setattr(lmstudio.subprocess, "run", run)
    return calls


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
        assert [load["model"] for load in lms.loads] == ["unsloth/Qwen3.5-4B-MTP-GGUF"]

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

        assert [load["model"] for load in lms.loads] == ["openai/gpt-oss-20b"]

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


def test_the_load_carries_a_config_that_fits_a_4_gb_card(lms) -> None:  # noqa: ANN001
    """LM Studio's defaults — 8192 x 4 slots x 2048-token batches — load, then die on the
    first prompt with CUDA OOM, and the request comes back 400. Measured: 3790 MiB with
    them, 3484 with these, and the prompt that crashed now completes."""
    LMStudioCompanion(URL).start()
    (load,) = lms.loads
    assert load["context_length"] == 4096, "what refine.client budgets for"
    assert load["parallel"] == 1, "one dictation, one request at a time"
    assert load["eval_batch_size"] == 512


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
        assert [load["model"] for load in lms.loads] == ["unsloth/Qwen3.5-4B-MTP-GGUF"]
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

        assert [load["model"] for load in lms.loads] == ["openai/gpt-oss-20b"]

    def test_the_default_model_name_is_not_mistaken_for_a_key(self, lms) -> None:  # noqa: ANN001
        """`refiner_model` defaults to "localasr-refiner", which is a label llama-server
        echoes back — not a model LM Studio has. Passing it through would ask for a model
        that cannot exist."""
        context = self._context()
        context.start_refiner()

        assert [load["model"] for load in lms.loads] == ["unsloth/Qwen3.5-4B-MTP-GGUF"]

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


def test_reasoning_is_turned_off_in_a_way_lm_studio_honours() -> None:
    """Measured against LM Studio 0.4.23 serving Qwen3.5-4B, not read off a page.

    `chat_template_kwargs.enable_thinking` is what llama-server honours, and LM Studio
    ignores it silently: every answer came back with `content: ""`, the entire token
    budget spent in `reasoning_content`, and refinement failed as "model did not return
    JSON". `reasoning_effort: "none"` was the spelling that produced content. Both are
    sent, because both servers are in use.
    """
    from localasr.refine.client import ChatCompletionClient
    from localasr.refine.types import RefinementRequest

    sent: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps({"refined_text": "好。"})}}],
                "model": "qwen3.5-4b-mtp",
            },
        )

    client = ChatCompletionClient(URL, client=httpx.Client(transport=httpx.MockTransport(handler)))
    client.refine(RefinementRequest(raw_text="呃那个好的"))

    assert sent["reasoning_effort"] == "none", "LM Studio honours this one"
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}, "llama-server that one"


def _down_until_lms_runs(monkeypatch, cli) -> None:  # noqa: ANN001
    """A server that answers only once `lms daemon up` and `lms server start` have run.

    The HTTP fake stays up throughout so the model listing still works; what is being
    simulated is the front end being off, which is `answers()`. Keyed on `server start`
    rather than on a call count, so it is also true for a daemon that was already up —
    otherwise that path waits out the full service timeout for a server it did start.
    """
    from localasr.refine import lmstudio

    monkeypatch.setattr(lmstudio, "startable_here", lambda _url: True)
    monkeypatch.setattr(
        lmstudio.LMStudioCompanion, "answers", lambda _self: ["server", "start"] in cli
    )


class TestStartingTheServer:
    """`lms`, doing locally what `node_ssh` does remotely.

    Without this a companion can only talk to a server somebody else started, which
    makes "starts with the app" half true — and it is the half the user meets first,
    because LM Studio's `autoStartOnLaunch` is off by default.
    """

    def test_a_silent_server_is_started_with_lm_studios_own_recipe(
        self, lms, cli, monkeypatch  # noqa: ANN001
    ) -> None:
        """Verbatim from the systemd unit in LM Studio's Linux docs: `daemon up` as
        ExecStartPre, `server start` as ExecStart."""
        from localasr.refine import lmstudio

        _down_until_lms_runs(monkeypatch, cli)
        lmstudio.LMStudioCompanion(URL).start()

        assert cli == [
            ["daemon", "status", "--json"],
            ["daemon", "up"],
            ["server", "start"],
        ]

    def test_a_server_this_session_started_is_stopped_again(self, lms, cli, monkeypatch) -> None:  # noqa: ANN001
        from localasr.refine import lmstudio

        _down_until_lms_runs(monkeypatch, cli)
        companion = lmstudio.LMStudioCompanion(URL)
        companion.start()
        cli.clear()
        companion.stop()

        assert cli == [["server", "stop"], ["daemon", "down"]], "both were ours to end"

    def test_a_server_already_running_is_not_restarted(self, lms, cli) -> None:  # noqa: ANN001
        LMStudioCompanion(URL).start()
        assert cli == [], "it was already answering"

    def test_a_server_we_did_not_start_is_left_running(self, lms, cli) -> None:  # noqa: ANN001
        companion = LMStudioCompanion(URL)
        companion.start()
        companion.stop()
        assert cli == []

    def test_autostart_off_reports_how_to_start_it_by_hand(self, lms, cli) -> None:  # noqa: ANN001
        lms.up = False
        with pytest.raises(LMStudioError, match="lms server start"):
            LMStudioCompanion(URL, autostart=False).start()
        assert cli == []


class TestWhoCanBeStarted:
    """`startable_here` — three conditions, all necessary."""

    def test_a_remote_url_is_not_ours_to_start(self, cli) -> None:  # noqa: ANN001
        from localasr.refine.lmstudio import startable_here

        assert not startable_here("http://asr-node.local:1234"), "lms starts it here, not there"

    def test_a_different_port_is_not_the_server_lms_would_start(self, cli, monkeypatch) -> None:  # noqa: ANN001
        """A down llama-server on 8091 must not get a button that brings up LM Studio on
        1234 and then reports success against a URL nothing is listening on."""
        from localasr.refine import lmstudio

        monkeypatch.setattr(lmstudio, "configured_port", lambda: 1234)
        assert not lmstudio.startable_here("http://127.0.0.1:8091")
        assert lmstudio.startable_here("http://127.0.0.1:1234")

    def test_without_the_cli_there_is_nothing_to_start_it_with(self, monkeypatch) -> None:  # noqa: ANN001
        from localasr.refine import lmstudio

        monkeypatch.setattr(lmstudio, "configured_port", lambda: 1234)
        assert not lmstudio.startable_here("http://127.0.0.1:1234"), "no lms on this machine"

    def test_the_port_comes_from_lm_studios_own_config(self, tmp_path, monkeypatch) -> None:  # noqa: ANN001
        from localasr.refine import lmstudio

        config = tmp_path / "http-server-config.json"
        config.write_text('{"port": 4321, "networkInterface": "127.0.0.1"}')
        monkeypatch.setattr(lmstudio, "SERVER_CONFIG", str(config))

        assert lmstudio.configured_port() == 4321

    def test_a_missing_config_falls_back_to_the_documented_default(
        self, tmp_path, monkeypatch  # noqa: ANN001
    ) -> None:
        from localasr.refine import lmstudio

        monkeypatch.setattr(lmstudio, "SERVER_CONFIG", str(tmp_path / "absent.json"))
        assert lmstudio.configured_port() == lmstudio.DEFAULT_PORT == 1234


class TestSilenceIsNotANo:
    """The bug that left the buttons grey with nothing able to turn them on.

    `speaks_lmstudio` returned False for a server that was merely not started, the
    application cached that as "not LM Studio", and the cache outlived the reason.
    """

    def test_an_answer_that_is_not_lm_studio_is_a_definite_no(self, monkeypatch) -> None:  # noqa: ANN001
        from localasr.refine.lmstudio import probe

        original = httpx.Client.__init__

        def patched(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
            kwargs["transport"] = httpx.MockTransport(lambda r: httpx.Response(404))
            original(self, *args, **kwargs)

        monkeypatch.setattr(httpx.Client, "__init__", patched)
        assert probe(URL) is False

    def test_no_answer_at_all_is_not_a_no(self, lms) -> None:  # noqa: ANN001
        from localasr.refine.lmstudio import probe

        lms.up = False
        assert probe(URL) is None, "not started yet is exactly what the button is for"

    def test_a_running_lm_studio_is_a_yes(self, lms) -> None:  # noqa: ANN001
        from localasr.refine.lmstudio import probe

        assert probe(URL) is True


class TestTheApplicationWhenLMStudioIsNotRunning:
    """The reported symptom: 不可达, and both buttons grey with no way back."""

    def _context(self, **settings):  # noqa: ANN001, ANN003, ANN202
        from localasr.context import AppContext, Settings

        return AppContext(settings=Settings(refiner_url=URL, **settings))

    def test_a_stopped_lm_studio_still_gets_its_buttons(self, lms, monkeypatch) -> None:  # noqa: ANN001
        from localasr.refine import lmstudio

        lms.up = False
        monkeypatch.setattr(lmstudio, "startable_here", lambda _url: True)
        context = self._context()

        assert context.lmstudio() is not None
        assert context.refiner_controllable, "it is stopped, not absent"
        assert not context.refiner_loaded

    def test_a_stopped_server_this_machine_cannot_start_stays_read_only(self, lms) -> None:  # noqa: ANN001
        """No `lms`, or a URL on another machine. Nothing here can help, and a button
        that cannot work is worse than one that is honestly grey."""
        lms.up = False
        assert self._context().lmstudio() is None

    def test_silence_is_not_cached_the_way_a_refusal_is(self, lms, monkeypatch) -> None:  # noqa: ANN001
        """The bug. LM Studio was down at launch, the answer was cached, and starting
        LM Studio afterwards changed nothing until the application was restarted."""
        from localasr.refine import lmstudio

        monkeypatch.setattr(lmstudio, "startable_here", lambda _url: False)
        lms.up = False
        context = self._context()
        assert context.lmstudio() is None

        lms.up = True
        assert context.lmstudio() is not None, "it came up; the panel must notice"

    def test_a_server_that_is_not_lm_studio_is_asked_once(self, monkeypatch) -> None:  # noqa: ANN001
        """The other half: a llama-server answers definitively, and re-probing it every
        few seconds for an answer that cannot change is waste."""
        asked = {"n": 0}
        original = httpx.Client.__init__

        def patched(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
            def handler(request: httpx.Request) -> httpx.Response:
                asked["n"] += 1
                return httpx.Response(404)

            kwargs["transport"] = httpx.MockTransport(handler)
            original(self, *args, **kwargs)

        monkeypatch.setattr(httpx.Client, "__init__", patched)
        context = self._context()
        for _ in range(5):
            context.lmstudio()

        assert asked["n"] == 1


def test_a_failing_daemon_up_names_both_fixes(lms, cli, monkeypatch) -> None:  # noqa: ANN001
    """Measured on this machine: `lms` is installed by the desktop app, but the headless
    llmster is not, so `lms daemon up` wakes a GUI and times out after ~60 s. The two
    ways out are installing llmster or opening the desktop app, and the error has to say
    so — "timed out" on its own is a dead end.
    """
    from localasr.refine import lmstudio

    monkeypatch.setattr(lmstudio, "startable_here", lambda _url: True)
    monkeypatch.setattr(lmstudio.LMStudioCompanion, "answers", lambda _self: False)

    class Failed:
        returncode = 1
        stdout = ""
        stderr = "Error: Timed out waiting for LM Studio daemon to start."

    monkeypatch.setattr(lmstudio.subprocess, "run", lambda *a, **k: Failed())

    with pytest.raises(lmstudio.LMStudioError) as caught:
        lmstudio.LMStudioCompanion(URL).start()

    message = str(caught.value)
    assert "install.sh" in message and "桌面版" in message
    assert "Timed out" in message, "the CLI's own words are the evidence"


class TestTheDaemonItself:
    """llmster, one layer below the HTTP server.

    Shutting one down ends every client's session, not just this one — so the rule that
    governs the model and the server governs it too, and more strictly: only a daemon
    this session started is this session's to stop.
    """

    def _companion(self, monkeypatch, cli):  # noqa: ANN001, ANN202
        from localasr.refine import lmstudio

        _down_until_lms_runs(monkeypatch, cli)
        return lmstudio.LMStudioCompanion(URL)

    def test_a_daemon_this_session_started_is_shut_down_again(self, lms, cli, monkeypatch) -> None:  # noqa: ANN001
        companion = self._companion(monkeypatch, cli)
        companion.start()
        cli.clear()
        companion.stop()

        assert cli == [["server", "stop"], ["daemon", "down"]]

    def test_a_daemon_already_running_is_left_alone(self, lms, cli, monkeypatch) -> None:  # noqa: ANN001
        """Somebody else's llmster, possibly with somebody else's model in it."""
        from localasr.refine import lmstudio

        monkeypatch.setattr(lmstudio, "daemon_running", lambda: True)
        companion = self._companion(monkeypatch, cli)
        companion.start()
        assert ["daemon", "up"] not in cli, "it was already up"

        cli.clear()
        companion.stop()
        assert cli == [["server", "stop"]], "the daemon was not ours to end"

    def test_the_status_is_read_from_the_body_not_the_exit_code(self, cli) -> None:  # noqa: ANN001
        """`lms daemon status --json` exits 0 whether or not llmster is running, so the
        exit code says nothing. Reading it instead would report every daemon as up."""
        from localasr.refine.lmstudio import daemon_running

        assert daemon_running() is False
        assert cli == [["daemon", "status", "--json"]]

    def test_without_lms_the_question_has_no_answer(self) -> None:
        """`None`, not False — and the difference matters, because False would mean
        "not running" and invite a start that has nothing to start it with."""
        from localasr.refine.lmstudio import daemon_running

        assert daemon_running() is None


class TestAnInstanceThatWillNotFit:
    """The 400 loop: an instance loaded with LM Studio's defaults, adopted as warm."""

    HEAVY = {"context_length": 8192, "parallel": 4, "eval_batch_size": 2048}

    def _resident(self, lms, config) -> None:  # noqa: ANN001
        lms.models[0]["loaded_instances"] = [
            {"id": "unsloth/Qwen3.5-4B-MTP-GGUF", "config": config}
        ]

    def test_an_oversized_instance_is_replaced_not_adopted(self, lms) -> None:  # noqa: ANN001
        """Adopting it only postponed the OOM to the user's first refinement."""
        self._resident(lms, self.HEAVY)
        LMStudioCompanion(URL).start()

        assert lms.unloads == [{"instance_id": "unsloth/Qwen3.5-4B-MTP-GGUF"}]
        assert lms.loads[0]["parallel"] == 1

    def test_an_instance_that_fits_is_adopted(self, lms) -> None:  # noqa: ANN001
        self._resident(lms, {"context_length": 4096, "parallel": 1, "eval_batch_size": 512})
        assert "已加载" in LMStudioCompanion(URL).start()
        assert lms.loads == [] and lms.unloads == []

    def test_somebody_elses_instance_is_left_alone_when_told_to(self, lms) -> None:  # noqa: ANN001
        self._resident(lms, self.HEAVY)
        LMStudioCompanion(URL, release_on_exit=False).start()
        assert lms.loads == [] and lms.unloads == []


def test_a_server_that_came_up_with_the_daemon_is_not_restarted(lms, cli, monkeypatch) -> None:  # noqa: ANN001
    """`lms server start` on a running server stops and restarts it, cancelling any load
    in flight — "Model load request cancelled by client disconnect" in LM Studio's log."""
    from localasr.refine import lmstudio

    monkeypatch.setattr(lmstudio, "startable_here", lambda _url: True)
    # Silent until the daemon is up; autoStartOnLaunch brings the server with it.
    monkeypatch.setattr(
        lmstudio.LMStudioCompanion, "answers", lambda _self: ["daemon", "up"] in cli
    )
    lmstudio.LMStudioCompanion(URL).start()

    assert ["server", "start"] not in cli


def test_concurrent_starts_load_once(lms) -> None:  # noqa: ANN001
    """The startup thread and the 启动 button can both be in `start`. Two loads racing is
    two instances on a card that holds one."""
    import threading

    companion = LMStudioCompanion(URL)
    threads = [threading.Thread(target=companion.start) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(lms.loads) == 1
