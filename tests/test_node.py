"""The compute node's HTTP surface, with no models and no GPU.

The coordinator is stubbed because what is being tested here is the boundary: who is
allowed in, what is refused, and whether a rejected refinement still reaches the caller
carrying the original text.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="node requires the [node] extra")

from fastapi.testclient import TestClient  # noqa: E402

from localasr.node.coordinator import Memory, ResourceCoordinator  # noqa: E402
from localasr.node.server import NodeConfig, create_app  # noqa: E402
from localasr.registry.manager import ModelKind  # noqa: E402

# --- exposure and authentication ----------------------------------------------


def test_a_lan_bind_without_a_token_is_refused() -> None:
    """An unauthenticated node on the LAN transcribes for anyone who can reach it."""
    with pytest.raises(ValueError, match="LOCALASR_NODE_TOKEN"):
        create_app(NodeConfig(bind="0.0.0.0"))


def test_a_lan_bind_with_a_token_is_allowed() -> None:
    assert create_app(NodeConfig(bind="0.0.0.0", token="s3cret")) is not None


def test_loopback_needs_no_token() -> None:
    assert create_app(NodeConfig(bind="127.0.0.1")) is not None


def test_protected_endpoints_reject_a_missing_or_wrong_token() -> None:
    client = TestClient(create_app(NodeConfig(token="right")))

    def status(token: str | None) -> int:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        return client.get("/api/v1/models", headers=headers).status_code

    assert status(None) == 401
    assert status("wrong") == 401
    assert status("right") == 200


def test_health_needs_no_token_so_a_probe_can_reach_it() -> None:
    client = TestClient(create_app(NodeConfig(token="right")))
    assert client.get("/healthz").json() == {"status": "ok"}


# --- readiness ----------------------------------------------------------------


def test_readiness_is_503_until_an_asr_model_is_resident() -> None:
    """Ready has to mean "a transcription will not first spend a minute loading"."""
    client = TestClient(create_app(NodeConfig()))
    response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["loaded"] == {}


def test_the_model_listing_reports_roles() -> None:
    client = TestClient(create_app(NodeConfig()))
    body = client.get("/api/v1/models").json()
    kinds = {entry["kind"] for entry in body["models"]}

    assert kinds == {"asr", "llm"}
    assert all("revision" in entry and entry["size"] > 0 for entry in body["models"])


def test_refinement_rejects_empty_input() -> None:
    client = TestClient(create_app(NodeConfig()))
    assert client.post("/api/v1/refinements", json={"raw_text": "  "}).status_code == 422


def test_refinement_rejects_an_unknown_mode() -> None:
    client = TestClient(create_app(NodeConfig()))
    body = {"raw_text": "有内容", "mode": "creative"}
    assert client.post("/api/v1/refinements", json=body).status_code == 422


# --- residency policy ---------------------------------------------------------


def test_memory_is_read_from_meminfo(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "meminfo"
    path.write_text("MemTotal:        7654321 kB\nMemAvailable:    5123456 kB\nMemFree: 1 kB\n")
    memory = Memory.read(path)

    assert memory is not None
    # MemAvailable, not MemFree: a warm page cache makes MemFree read near zero on a
    # Jetson at all times, which would refuse every load.
    assert memory.available_mb == 5123456 // 1024


def test_missing_meminfo_does_not_block_loading(tmp_path) -> None:  # noqa: ANN001
    assert Memory.read(tmp_path / "absent") is None


def test_a_load_that_would_exhaust_memory_is_refused(monkeypatch) -> None:  # noqa: ANN001
    from localasr.core.engine.supervisor import SupervisorError
    from localasr.registry import manager

    coordinator = ResourceCoordinator(min_free_mb=1024)
    monkeypatch.setattr(
        "localasr.node.coordinator.Memory.read",
        classmethod(lambda cls, *a: Memory(total_mb=7600, available_mb=1500)),
    )
    spec = manager.get_model(kind=ModelKind.ASR)

    with pytest.raises(SupervisorError, match="refusing to load"):
        coordinator.ensure(spec)


def test_the_memory_estimate_exceeds_the_file_size() -> None:
    """Weights are the floor, not the total: compute buffers, the CUDA context, the KV
    cache and the load-time peak are all real and none are in the file size. Measured
    on the Orin the true cost ran from 1.48x (4B) to 2.47x (0.6B)."""
    from localasr.node.coordinator import OVERHEAD_FACTOR, estimated_cost_mb
    from localasr.registry import manager

    assert OVERHEAD_FACTOR > 1.0
    for spec in manager.list_models():
        weights_mb = sum(f.size for f in spec.files) // (1024 * 1024)
        assert estimated_cost_mb(spec) > weights_mb, f"{spec.model_id} under-counted"


def test_a_model_that_fits_on_paper_but_not_in_practice_is_refused(monkeypatch) -> None:  # noqa: ANN001
    """2.4 GiB of weights with 4 GiB free passes a file-size check and still OOMs."""
    from localasr.core.engine.supervisor import SupervisorError
    from localasr.registry import manager

    monkeypatch.setattr(
        "localasr.node.coordinator.Memory.read",
        classmethod(lambda cls, *a: Memory(total_mb=7600, available_mb=4000)),
    )
    spec = manager.get_model("qwen3-asr-1_7b-q8", ModelKind.ASR)
    weights_mb = sum(f.size for f in spec.files) // (1024 * 1024)
    assert 4000 - weights_mb > 1024, "the naive file-size check would have allowed this"

    with pytest.raises(SupervisorError, match="refusing to load"):
        ResourceCoordinator(min_free_mb=1024).ensure(spec)


def test_nothing_is_loaded_before_a_request(monkeypatch) -> None:  # noqa: ANN001
    coordinator = ResourceCoordinator()
    assert coordinator.loaded() == {}


# --- reaching a remote node ---------------------------------------------------


def test_the_transcription_client_can_authenticate_to_a_node() -> None:
    """Moving the engine to another machine is a base URL and a token, not a new client."""
    from localasr.core.engine.client import TranscriptionClient

    with TranscriptionClient("http://orin.local:8090", token="s3cret") as client:
        assert client._client.headers["Authorization"] == "Bearer s3cret"

    with TranscriptionClient("http://127.0.0.1:8080") as local:
        assert "Authorization" not in local._client.headers


# --- desktop side: using a node, and surviving it being gone -------------------


def _manager(url: str | None, ready, **kw):  # noqa: ANN001, ANN202
    from localasr.core.engine.manager import EngineManager
    from localasr.registry import manager as registry

    engine = EngineManager(registry.get_model(), node_url=url, share=False, **kw)
    return engine


def test_a_configured_node_is_used_instead_of_a_local_engine(monkeypatch) -> None:  # noqa: ANN001
    from localasr.core.engine import manager as manager_module

    class Ready:
        def __init__(self, *a, **kw) -> None:
            self.kwargs = kw

        def is_ready(self) -> bool:
            return True

        def close(self) -> None:
            pass

    monkeypatch.setattr(manager_module, "TranscriptionClient", Ready)
    engine = _manager("http://asr-node.local:8090", True, node_token="tok")
    client = engine.acquire()

    assert engine.attached, "a remote node must never be stoppable from here"
    assert client.kwargs.get("token") == "tok"
    assert engine.fallback_reason is None


def test_an_unreachable_node_falls_back_to_the_local_engine(monkeypatch) -> None:  # noqa: ANN001
    """The user is mid-sentence. Slower local recognition beats none."""
    from localasr.core.engine import manager as manager_module

    class Dead:
        def __init__(self, *a, **kw) -> None:
            pass

        def is_ready(self) -> bool:
            raise OSError("network unreachable")

        def close(self) -> None:
            pass

    launched: list[str] = []
    monkeypatch.setattr(manager_module, "TranscriptionClient", Dead)
    engine = _manager("http://asr-node.local:8090", False)
    monkeypatch.setattr(engine, "_launch", lambda: launched.append("local") or "client")

    assert engine.acquire() == "client"
    assert launched == ["local"]
    assert engine.fallback_reason and "已改用本机引擎" in engine.fallback_reason


def test_a_node_that_answers_but_is_not_ready_also_falls_back(monkeypatch) -> None:  # noqa: ANN001
    from localasr.core.engine import manager as manager_module

    class NotReady:
        def __init__(self, *a, **kw) -> None:
            pass

        def is_ready(self) -> bool:
            return False

        def close(self) -> None:
            pass

    monkeypatch.setattr(manager_module, "TranscriptionClient", NotReady)
    engine = _manager("http://asr-node.local:8090", False)
    monkeypatch.setattr(engine, "_launch", lambda: "client")

    engine.acquire()
    assert "未就绪" in (engine.fallback_reason or "")


def test_no_node_configured_means_nothing_changes() -> None:
    engine = _manager(None, False)
    assert engine.node_url is None and engine.fallback_reason is None


def test_the_node_extra_pulls_no_desktop_dependencies() -> None:
    """The node is headless and has no microphone. PySide6, onnxruntime, soxr and
    sounddevice must never be part of its install — they are hundreds of megabytes of
    aarch64 wheels for code that cannot run there."""
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    node_extra = " ".join(project["optional-dependencies"]["node"]).lower()
    base = " ".join(project["dependencies"]).lower()

    for unwanted in ("pyside6", "onnxruntime", "soxr", "sounddevice", "torch"):
        assert unwanted not in node_extra, f"{unwanted} leaked into the node extra"
        assert unwanted not in base, f"{unwanted} is in base deps; the node gets it too"


def test_a_model_that_cannot_load_is_a_refusal_not_a_500(monkeypatch) -> None:  # noqa: ANN001
    """The client's contract is that refinement degrades to the raw text with a stated
    reason. An unhandled exception arrives as "节点返回 500", which tells the user
    nothing about a model that would not fit or a server that would not start."""
    from localasr.core.engine.supervisor import SupervisorError

    app = create_app(NodeConfig())

    def refuse(_spec):  # noqa: ANN001, ANN202
        raise SupervisorError("refusing to load: estimated ~4000 MiB, 1500 MiB available")

    app.state.coordinator.ensure = refuse
    monkeypatch.setattr("localasr.node.server.ResourceCoordinator.ensure", staticmethod(refuse))

    response = TestClient(app).post("/api/v1/refinements", json={"raw_text": "原始内容"})

    assert response.status_code == 200, "a load failure is not a server fault"
    body = response.json()
    assert body["accepted"] is False
    assert body["text"] == "原始内容", "the original must come back"
    assert any("无法加载整理模型" in i["detail"] for i in body["issues"])


# --- dual residency, decided by measured arithmetic ----------------------------


class _FakeSupervisor:
    started: list[str] = []
    stopped: list[str] = []

    def __init__(self, spec) -> None:  # noqa: ANN001
        self.spec = spec
        self.base_url = f"http://127.0.0.1:9/{spec.model_id}"

    def start(self, log_path=None):  # noqa: ANN001, ANN202
        _FakeSupervisor.started.append(self.spec.model_id)

        class _Probe:
            def close(self) -> None:
                pass

        return _Probe()

    def stop(self) -> None:
        _FakeSupervisor.stopped.append(self.spec.model_id)


@pytest.fixture
def fake_engines(monkeypatch):  # noqa: ANN001, ANN201
    _FakeSupervisor.started, _FakeSupervisor.stopped = [], []
    monkeypatch.setattr("localasr.node.coordinator.EngineSupervisor", _FakeSupervisor)
    return _FakeSupervisor


def _memory(monkeypatch, available_mb: int, coordinator=None) -> None:  # noqa: ANN001
    """A reading that falls as models load.

    A constant would defeat the thing under test: eviction is decided by watching
    available memory drop, so a fake that never drops always reports room for one more.
    """
    from localasr.node.coordinator import estimated_cost_mb

    def read(_cls, *_a):  # noqa: ANN001, ANN202
        resident = coordinator._specs.values() if coordinator else ()
        used = sum(estimated_cost_mb(spec) for spec in resident)
        return Memory(total_mb=7546, available_mb=available_mb - used)

    monkeypatch.setattr("localasr.node.coordinator.Memory.read", classmethod(read))


def test_loading_one_role_never_unloads_the_other(fake_engines, monkeypatch) -> None:  # noqa: ANN001
    """The two roles are independent slots.

    Loading a refiner used to evict a resident ASR model whenever a policy flag was off —
    so asking for a tidy-up cost whoever was dictating their loaded model and a ~35 s
    reload, for a request that had no claim on that memory. Nothing on screen could
    explain it either: the ASR panel simply said "not loaded" again.
    """
    from localasr.registry import manager

    coordinator = ResourceCoordinator(min_free_mb=900)
    _memory(monkeypatch, 6903, coordinator)
    llm = manager.get_model("qwen3_5-2b-refiner-q4", ModelKind.LLM)
    asr = manager.get_model("qwen3-asr-1_7b-q8", ModelKind.ASR)

    coordinator.ensure(llm)
    coordinator.ensure(asr)

    assert coordinator.loaded() == {"llm": llm.model_id, "asr": asr.model_id}
    assert fake_engines.stopped == [], "nothing may be unloaded on somebody else's behalf"


def test_both_roles_stay_resident_when_the_memory_is_there(fake_engines, monkeypatch):  # noqa: ANN001
    """Keeping both removes a ~35 s model swap from every refinement, which is the
    entire cost of taking turns."""
    from localasr.registry import manager

    coordinator = ResourceCoordinator(min_free_mb=1024)
    _memory(monkeypatch, 6917, coordinator)  # the Orin headless
    asr = manager.get_model("qwen3-asr-0_6b-q8", ModelKind.ASR)
    llm = manager.get_model("qwen3_5-2b-refiner-q4", ModelKind.LLM)

    coordinator.ensure(asr)
    coordinator.ensure(llm)

    assert coordinator.loaded() == {"asr": asr.model_id, "llm": llm.model_id}
    assert fake_engines.stopped == [], "nothing had to go"


def test_a_tight_machine_refuses_and_names_what_is_holding_the_memory(fake_engines, monkeypatch):  # noqa: ANN001
    """It used to evict instead. "Slower, not broken" sounded reasonable until you ask
    whose model gets unloaded — and the answer was "whoever was not asking".

    A refusal is only useful if the one action that would fix it is visible, so the
    message has to name what is resident.
    """
    from localasr.core.engine.supervisor import SupervisorError
    from localasr.registry import manager

    coordinator = ResourceCoordinator(min_free_mb=1024)
    # Measured on the Orin headless: 1.7B ASR is 3964 MiB and the 4B refiner 3876, so
    # 7840 against a 6917 MiB budget — they genuinely do not both fit.
    _memory(monkeypatch, 6917, coordinator)
    asr = manager.get_model("qwen3-asr-1_7b-q8", ModelKind.ASR)
    llm = manager.get_model("qwen3_5-4b-refiner-q4", ModelKind.LLM)

    coordinator.ensure(asr)
    with pytest.raises(SupervisorError, match="refusing to load") as caught:
        coordinator.ensure(llm)

    assert asr.model_id in str(caught.value), "the user cannot act on an unnamed blocker"
    assert fake_engines.stopped == []
    assert coordinator.loaded() == {"asr": asr.model_id}, "the ASR model survived"


def test_a_model_that_fits_nowhere_is_still_refused(fake_engines, monkeypatch):  # noqa: ANN001
    from localasr.core.engine.supervisor import SupervisorError
    from localasr.registry import manager

    _memory(monkeypatch, 2000)
    with pytest.raises(SupervisorError, match="refusing to load"):
        ResourceCoordinator(min_free_mb=1024).ensure(
            manager.get_model("qwen3_5-4b-refiner-q4", ModelKind.LLM)
        )


def test_reloading_the_same_model_is_a_no_op(fake_engines, monkeypatch):  # noqa: ANN001
    from localasr.registry import manager

    coordinator = ResourceCoordinator()
    _memory(monkeypatch, 6917, coordinator)
    spec = manager.get_model("qwen3-asr-0_6b-q8", ModelKind.ASR)

    first = coordinator.ensure(spec)
    assert coordinator.ensure(spec) == first
    assert fake_engines.started == [spec.model_id], "must not restart a live server"


def test_a_measured_cost_beats_the_multiplier() -> None:
    """One factor cannot describe both: 4B measured 1.29x its weights, 0.6B 1.60x,
    because mmproj and fixed per-process cost dominate a small model."""
    from localasr.node.coordinator import estimated_cost_mb
    from localasr.registry import manager

    measured = manager.get_model("qwen3_5-4b-refiner-q4", ModelKind.LLM)
    assert measured.measured_mb > 0
    assert estimated_cost_mb(measured) == measured.measured_mb

    # Measured ratios observed on the Orin span 1.48x (4B) to 2.47x (0.6B): mmproj and
    # fixed per-process cost dominate a small model. No single multiplier covers both.
    ratios = []
    for spec in manager.list_models():
        if spec.measured_mb:
            weights = sum(f.size for f in spec.files) // (1024 * 1024)
            ratios.append(spec.measured_mb / weights)
    assert max(ratios) - min(ratios) > 0.5, "if one factor sufficed, drop measured_mb"

    # Every catalog model has been measured now, so the fallback is exercised against a
    # synthetic spec — it still has to work for the next model added.
    from dataclasses import replace

    unmeasured = replace(measured, measured_mb=0)
    weights = sum(f.size for f in unmeasured.files) // (1024 * 1024)
    assert estimated_cost_mb(unmeasured) == int(weights * 1.4)


def test_managing_models_does_not_require_the_audio_stack() -> None:
    """`localasr models pull` on the compute node failed with ModuleNotFoundError:
    onnxruntime. Downloading a file has nothing to do with running a VAD, and the node
    deliberately has no audio stack — so the import must stay lazy."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    # Blocking the imports outright, rather than checking sys.modules: the desktop
    # packages *are* installed here, so a mere absence check passes while the node —
    # where they are not — still fails at import time. This reproduces the node.
    probe = (
        "import sys;"
        "blocked = ('soxr', 'sounddevice', 'onnxruntime', 'PySide6');"
        "sys.meta_path.insert(0, type('B', (), {"
        "  'find_module': staticmethod(lambda n, p=None: None),"
        "  'find_spec': staticmethod("
        "     lambda n, p=None, t=None: (_ for _ in ()).throw(ImportError(n))"
        "     if n.split('.')[0] in blocked else None)"
        "})());"
        "import localasr.frontends.cli.main;"
        "from localasr.registry import manager;"
        "assert manager.list_models()"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, cwd=root, timeout=60
    )
    assert result.returncode == 0, result.stderr.decode()[-500:]


def test_joining_an_existing_model_costs_less_than_loading_alone(fake_engines, monkeypatch):  # noqa: ANN001
    """Measured on the Orin: 1.7B ASR is 3964 MiB alone but 3791 MiB beside a running
    refiner. Summing per-model figures refuses a pair that leaves a gigabyte free."""
    from localasr.node.coordinator import SHARED_OVERHEAD_MB
    from localasr.registry import manager

    coordinator = ResourceCoordinator(min_free_mb=900)
    _memory(monkeypatch, 6903, coordinator)
    llm = manager.get_model("qwen3_5-2b-refiner-q4", ModelKind.LLM)
    asr = manager.get_model("qwen3-asr-1_7b-q8", ModelKind.ASR)

    coordinator.ensure(llm)
    coordinator.ensure(asr)

    assert SHARED_OVERHEAD_MB > 0
    assert coordinator.loaded() == {"llm": llm.model_id, "asr": asr.model_id}
    assert fake_engines.stopped == [], "the pair fits; nothing should have been evicted"


def _fake_chat_client(monkeypatch, handler):  # noqa: ANN001, ANN202
    """Point ChatCompletionClient at a MockTransport without touching its interface."""
    import httpx

    from localasr.refine.client import ChatCompletionClient

    original = ChatCompletionClient.__init__

    def patched(self, base_url="", **kwargs) -> None:  # noqa: ANN001
        kwargs["client"] = httpx.Client(transport=httpx.MockTransport(handler))
        original(self, base_url, **kwargs)

    monkeypatch.setattr(ChatCompletionClient, "__init__", patched)


# --- split backends: ASR on the Orin, refinement on the desktop card ------------


def test_a_configured_refiner_is_used_directly_not_through_the_asr_node(monkeypatch) -> None:  # noqa: ANN001
    """Two backends with different jobs and different homes. Routing refinement through
    the ASR node would put the text on a round trip to a machine that no longer holds a
    language model."""
    import httpx

    from localasr.context import AppContext
    from localasr.refine.types import RefinementRequest

    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        payload = {"refined_text": "我们下周交。"}
        import json as _json

        return httpx.Response(
            200, json={"choices": [{"message": {"content": _json.dumps(payload)}}]}
        )

    context = AppContext()
    context.settings.refiner_url = "http://127.0.0.1:8888"
    context.settings.node_url = "http://asr-node.local:8090"
    context.settings.refiner_model = "Qwen3.5-4B-Q4_K_M"

    _fake_chat_client(monkeypatch, handler)

    result = context.refine(RefinementRequest("嗯我们下周交"))

    assert "127.0.0.1:8888" in seen["url"], "must not go via the ASR node"
    assert result.accepted
    assert result.model_id == "Qwen3.5-4B-Q4_K_M", "the journal needs the real weights"


def test_refinement_output_is_validated_wherever_it_came_from(monkeypatch) -> None:  # noqa: ANN001
    """A refiner reached directly is still a model server answering over HTTP. Nothing
    it returns may skip the checks against the original."""
    import json as _json

    import httpx

    from localasr.context import AppContext
    from localasr.refine.types import RefinementRequest

    def invents(_request: httpx.Request) -> httpx.Response:
        payload = {"refined_text": "我们下周五交三个报告。"}
        return httpx.Response(
            200, json={"choices": [{"message": {"content": _json.dumps(payload)}}]}
        )

    context = AppContext()
    context.settings.refiner_url = "http://127.0.0.1:8888"
    _fake_chat_client(monkeypatch, invents)

    result = context.refine(RefinementRequest("嗯我们下周交三个报告"))

    assert not result.accepted
    assert result.text == "嗯我们下周交三个报告", "the invented date must not be shown"


def test_local_asr_is_not_started_behind_a_resident_refiner(monkeypatch) -> None:  # noqa: ANN001
    """Starting Qwen3-ASR beside a resident Qwen3.5-4B is how a 4 GB card runs out of
    memory — silently, mid-sentence. Saying the node is down is the better failure."""
    from localasr.core.engine import manager as manager_module
    from localasr.core.engine.manager import EngineManager
    from localasr.core.engine.supervisor import SupervisorError
    from localasr.registry import manager as registry

    class Dead:
        def __init__(self, *a, **kw) -> None:
            pass

        def is_ready(self) -> bool:
            raise OSError("network unreachable")

        def close(self) -> None:
            pass

    monkeypatch.setattr(manager_module, "TranscriptionClient", Dead)
    engine = EngineManager(
        registry.get_model(),
        node_url="http://asr-node.local:8090",
        share=False,
        local_fallback=False,
    )
    launched = []
    monkeypatch.setattr(engine, "_launch", lambda: launched.append("local"))

    with pytest.raises(SupervisorError, match="远程识别节点不可用"):
        engine.acquire()
    assert launched == [], "the local engine must not have been started"


def test_releasing_models_frees_the_node(fake_engines, monkeypatch) -> None:  # noqa: ANN001
    """The only way to hand an Orin's memory back without stopping the node. The next
    request loads again; this is not an off switch."""
    from localasr.registry import manager

    app = create_app(NodeConfig())
    coordinator = ResourceCoordinator(min_free_mb=900)
    _memory(monkeypatch, 6903, coordinator)
    monkeypatch.setattr("localasr.node.server.ResourceCoordinator", lambda **kw: coordinator)
    app = create_app(NodeConfig())
    coordinator.ensure(manager.get_model("qwen3-asr-1_7b-q8", ModelKind.ASR))
    assert coordinator.loaded()

    body = TestClient(app).post("/api/v1/models/release", json={}).json()

    assert body["released"] == ["asr"]
    assert body["loaded"] == {}
    assert fake_engines.stopped == ["qwen3-asr-1_7b-q8"]


def test_releasing_something_not_loaded_is_not_an_error() -> None:
    """The caller wanted it gone and it is gone. Failing here would make a retry after a
    dropped response look like a fault."""
    response = TestClient(create_app(NodeConfig())).post("/api/v1/models/release", json={})
    assert response.status_code == 200
    assert response.json()["released"] == []


def test_releasing_an_unknown_role_is_refused() -> None:
    response = TestClient(create_app(NodeConfig())).post(
        "/api/v1/models/release", json={"kind": "quantum"}
    )
    assert response.status_code == 422


def test_release_requires_the_token() -> None:
    """Anyone able to reach the node could otherwise make every request slow."""
    app = create_app(NodeConfig(token="s3cret"))
    assert TestClient(app).post("/api/v1/models/release", json={}).status_code == 401


def test_loading_up_front_avoids_paying_for_it_mid_sentence(fake_engines, monkeypatch) -> None:  # noqa: ANN001
    """A cold load is 25-40 s on a Jetson. Paying it when someone has already started
    talking is what makes the tool look broken."""
    coordinator = ResourceCoordinator(min_free_mb=900)
    _memory(monkeypatch, 6903, coordinator)
    monkeypatch.setattr("localasr.node.server.ResourceCoordinator", lambda **kw: coordinator)
    app = create_app(NodeConfig())

    body = TestClient(app).post("/api/v1/models/load", json={"kind": "asr"}).json()

    assert body["loaded"] == {"asr": "qwen3-asr-1_7b-q8"}
    assert fake_engines.started == ["qwen3-asr-1_7b-q8"]


def test_a_load_that_cannot_fit_answers_rather_than_crashing(monkeypatch) -> None:  # noqa: ANN001
    from localasr.core.engine.supervisor import SupervisorError

    def refuse(_spec):  # noqa: ANN001, ANN202
        raise SupervisorError("not enough memory")

    monkeypatch.setattr("localasr.node.server.ResourceCoordinator.ensure", staticmethod(refuse))
    response = TestClient(create_app(NodeConfig())).post("/api/v1/models/load", json={})

    assert response.status_code == 409
    assert "not enough memory" in response.json()["error"]


def test_loading_an_unknown_role_is_refused() -> None:
    response = TestClient(create_app(NodeConfig())).post(
        "/api/v1/models/load", json={"kind": "quantum"}
    )
    assert response.status_code == 422


# --- explicit lifecycle for the locally-owned refiner -------------------------


def test_the_desktop_owns_the_refiner_only_when_no_url_is_configured() -> None:
    """`refiner_url` names a server somebody else runs; `refiner_model_id` names one
    this machine starts. Which is set decides whether the buttons do anything."""
    from localasr.context import AppContext

    context = AppContext()
    context.settings.refiner_url = "http://127.0.0.1:8888"
    context.settings.refiner_model_id = "qwen3_5-4b-refiner-q4"
    assert not context.refiner_managed, "an external server is not ours to stop"
    assert context.refiner_base_url == "http://127.0.0.1:8888"

    context.settings.refiner_url = None
    assert context.refiner_managed
    assert context.refiner_base_url is None, "not started yet"


def test_starting_an_externally_managed_refiner_is_refused() -> None:
    from localasr.context import AppContext

    context = AppContext()
    context.settings.refiner_url = "http://127.0.0.1:8888"
    with pytest.raises(RuntimeError, match="外部管理"):
        context.start_refiner()


def test_stopping_a_refiner_that_never_started_is_harmless() -> None:
    from localasr.context import AppContext

    context = AppContext()
    context.settings.refiner_url = None
    context.stop_refiner()
    assert not context.refiner_loaded


def test_a_refiner_that_cannot_start_reports_the_module_s_reason() -> None:
    """The check for a usable model now lives in the module, not here.

    That is the point of the split — but it means the parent has to carry the child's
    explanation back, or "启动失败" is all anyone sees. Really spawns it: stubbing the
    registry in this process would prove nothing about a separate one, which is exactly
    the mistake that makes a passing test worthless.
    """
    from localasr.context import AppContext
    from localasr.core.engine.supervisor import SupervisorError

    context = AppContext()
    context.settings.refiner_url = None
    context.settings.refiner_model_id = "a-model-that-does-not-exist"

    with pytest.raises(SupervisorError) as caught:
        context.start_refiner()

    assert "a-model-that-does-not-exist" in str(caught.value), (
        "the child's reason has to reach the user"
    )


def test_shutdown_releases_the_refiner(monkeypatch) -> None:  # noqa: ANN001
    """Our child process holds the GPU, and the only thing that knows about it is the
    object going away.

    Keeping it alive past exit was tried and reverted: it needs a state file, a pid
    liveness check that survives pid reuse, and adoption on the next launch — a lot of
    machinery, and a race, for a saving the user can get by simply leaving the window
    open. The remote ASR node persists because it genuinely is a separate service; this
    one is a child process, and dying with its parent is what a child process does.
    """
    from localasr.context import AppContext

    stopped = []

    class FakeSupervisor:
        base_url = "http://127.0.0.1:9"

        def stop(self) -> None:
            stopped.append(True)

    context = AppContext()
    context._refiner = FakeSupervisor()
    context.shutdown()

    assert stopped == [True]
    assert not context.refiner_loaded


def test_unloading_is_what_releases_the_refiner() -> None:
    from localasr.context import AppContext

    stopped = []

    class FakeSupervisor:
        base_url = "http://127.0.0.1:9"

        def stop(self) -> None:
            stopped.append(True)

    context = AppContext()
    context._refiner = FakeSupervisor()
    context.stop_refiner()

    assert stopped == [True]
    assert not context.refiner_loaded


def test_settings_persist_only_what_differs_from_the_defaults(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    """A config file is meant to be read and edited. Writing every field buries the two
    lines the user chose, and freezes today's defaults onto disk where a later change to
    them can never take effect."""
    from localasr.context import Settings

    monkeypatch.setenv("LOCALASR_CONFIG", str(tmp_path / "config.toml"))
    settings = Settings()
    settings.node_url = "http://asr-node.local:8090"
    settings.local_asr_fallback = False
    written = settings.save().read_text()

    assert "node_url" in written
    assert "local_asr_fallback = false" in written
    assert "hotkey_hint" not in written, "an untouched default must not be persisted"
    assert "refiner_model" not in written
    assert "idle_timeout" not in written
    assert Settings.load().node_url == "http://asr-node.local:8090"


def test_a_start_failure_says_why_not_just_the_exit_code(tmp_path) -> None:  # noqa: ANN001
    """"exited with code 1" is true and useless. On a 4 GB card the likely cause is that
    something else already holds the memory, and saying so turns a dead end into a next
    step."""
    from localasr.core.engine.supervisor import _why_it_died

    log = tmp_path / "refiner.log"
    log.write_text(
        "0.01.2 I srv load_model: loading model\n"
        "0.01.3 E alloc_tensor_range: failed to allocate Vulkan1 buffer of size 1073741824\n"
    )
    assert "显存或内存不足" in _why_it_died(log)

    log.write_text("0.01.2 E common_init: unknown argument --nonsense\n")
    assert "unknown argument" in _why_it_died(log)

    assert "未捕获日志" in _why_it_died(None)
    assert "未捕获日志" in _why_it_died(tmp_path / "absent.log")
