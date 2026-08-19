"""The standalone refinement host.

What is being tested is mostly the boundary, not the model: which model it picks, what
it refuses to do, and the one property that makes it worth having as a separate process
at all — that it does not drag the application in with it.
"""

from __future__ import annotations

import pytest

from localasr.refine import host
from localasr.refine.host import DEFAULT_PORT, HostConfig


def test_it_does_not_import_the_application() -> None:
    """The reason this is a separate process rather than a flag.

    It has to start on a machine with no display, no microphone and no PySide6 — a spare
    box with a GPU. An accidental import of the desktop or the capture stack would make
    that machine need an audio library to serve text.
    """
    import subprocess
    import sys

    probe = (
        "import sys, localasr.refine.host;"
        "leaked = [m for m in sys.modules if m.startswith("
        "('localasr.frontends', 'localasr.capture', 'localasr.apps'))];"
        "leaked += [m for m in ('PySide6', 'sounddevice', 'onnxruntime') if m in sys.modules];"
        "print(leaked)"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert out.stdout.strip() == "[]", f"the host pulled in {out.stdout.strip()}"


def test_the_port_is_fixed_rather_than_ephemeral() -> None:
    """`refiner_url` is a line in a config file. A port chosen at random each start would
    mean rewriting it after every restart, which is not a configuration."""
    assert HostConfig().port == DEFAULT_PORT
    assert HostConfig().bind == "127.0.0.1"


def test_serving_on_the_network_without_a_token_is_refused() -> None:
    """An open /v1/chat/completions is an open door into this machine's memory budget,
    and it fails silently: everything keeps working, slowly, for reasons nothing on this
    host explains. Same rule the compute node applies to itself."""
    with pytest.raises(SystemExit, match="without a token"):
        host.check(HostConfig(bind="0.0.0.0"))


def test_serving_on_the_network_with_a_token_is_allowed() -> None:
    host.check(HostConfig(bind="0.0.0.0", token="s3cret"))


def test_loopback_needs_no_token() -> None:
    host.check(HostConfig())
    host.check(HostConfig(bind="localhost"))


def test_an_explicit_model_wins_over_the_configured_one(monkeypatch) -> None:  # noqa: ANN001
    seen = {}

    def fake_get_model(model_id, kind):  # noqa: ANN001, ANN202
        seen["model_id"] = model_id
        seen["kind"] = kind
        return "spec"

    monkeypatch.setattr(host.manager, "get_model", fake_get_model)
    HostConfig(model_id="local-something").resolve()

    assert seen["model_id"] == "local-something"
    assert seen["kind"] is host.ModelKind.LLM, "an ASR model must never be served here"


def test_without_an_argument_it_serves_what_the_settings_name(monkeypatch) -> None:  # noqa: ANN001
    """Useful while the application still manages a model on the same machine."""
    from localasr.context import Settings

    monkeypatch.setattr(Settings, "load", staticmethod(lambda: Settings(refiner_model_id="cfg-id")))
    seen = {}
    monkeypatch.setattr(
        host.manager, "get_model", lambda model_id, kind: seen.setdefault("id", model_id)
    )

    HostConfig().resolve()
    assert seen["id"] == "cfg-id"


def _no_setting(monkeypatch) -> None:  # noqa: ANN001
    from localasr.context import Settings

    monkeypatch.setattr(Settings, "load", staticmethod(lambda: Settings(refiner_model_id=None)))


def test_with_nothing_configured_it_serves_the_one_model_that_is_present(monkeypatch) -> None:  # noqa: ANN001
    """The bug this exists for, and it only showed up in the status panel.

    `refiner_model_id` is the application's setting for a model *it* manages. Once the
    application stops managing one — the entire point of this host — that setting is
    empty, and falling through to the catalog default picked a model whose weights had
    just been deleted, while the imported one that was actually there sat beside it.
    """
    class Spec:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id

    absent, present = Spec("catalog-default"), Spec("local-imported")
    _no_setting(monkeypatch)
    monkeypatch.setattr(host.manager, "models_of_kind", lambda _k: [absent, present])
    monkeypatch.setattr(host.manager, "is_downloaded", lambda spec: spec is present)

    assert HostConfig().resolve().model_id == "local-imported"


def test_several_present_models_are_ambiguous_rather_than_guessed(monkeypatch) -> None:  # noqa: ANN001
    class Spec:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id

    _no_setting(monkeypatch)
    monkeypatch.setattr(host.manager, "models_of_kind", lambda _k: [Spec("a"), Spec("b")])
    monkeypatch.setattr(host.manager, "is_downloaded", lambda _spec: True)

    with pytest.raises(host.manager.RegistryError, match="--model"):
        HostConfig().resolve()


def test_no_present_model_says_how_to_get_one(monkeypatch) -> None:  # noqa: ANN001
    _no_setting(monkeypatch)
    monkeypatch.setattr(host.manager, "models_of_kind", lambda _k: [])

    with pytest.raises(host.manager.RegistryError, match="models import"):
        HostConfig().resolve()


def test_a_missing_model_exits_with_instructions_rather_than_a_traceback(
    monkeypatch, capsys  # noqa: ANN001
) -> None:
    class Spec:
        model_id = "local-absent"

    monkeypatch.setattr(host.manager, "get_model", lambda *a, **k: Spec())
    monkeypatch.setattr(host.manager, "is_downloaded", lambda _spec: False)
    monkeypatch.setattr(host.manager, "model_dir", lambda _spec: "/models/local-absent")

    assert host.serve(HostConfig()) == 2
    message = capsys.readouterr().err
    assert "models pull" in message and "models import" in message


def test_the_command_line_maps_onto_the_config() -> None:
    config = host.parse_args(
        ["--model", "m", "--port", "9001", "--bind", "0.0.0.0", "--token", "t"]
    )  # noqa: E501
    assert config == HostConfig(model_id="m", port=9001, bind="0.0.0.0", token="t")


def test_environment_variables_are_honoured(monkeypatch) -> None:  # noqa: ANN001
    """A systemd unit sets environment, not arguments."""
    monkeypatch.setenv("LOCALASR_REFINER_PORT", "9100")
    monkeypatch.setenv("LOCALASR_REFINER_BIND", "0.0.0.0")
    monkeypatch.setenv("LOCALASR_REFINER_TOKEN", "from-env")

    config = host.parse_args([])
    assert (config.port, config.bind, config.token) == (9100, "0.0.0.0", "from-env")


def test_a_bound_server_reports_a_url_a_client_can_actually_use() -> None:
    """0.0.0.0 is an address to listen on, not one to connect to."""
    from localasr.core.engine.supervisor import EngineSupervisor
    from localasr.registry import manager
    from localasr.registry.manager import ModelKind

    spec = manager.get_model("qwen3_5-4b-refiner-q4", ModelKind.LLM)
    engine = EngineSupervisor(spec, port=8091, bind="0.0.0.0")
    assert engine.base_url == "http://127.0.0.1:8091"


def test_a_token_reaches_llama_server_as_an_api_key() -> None:
    """Refusing to bind without one would be theatre if it were never passed on."""
    from pathlib import Path

    from localasr.core.engine.supervisor import EngineSupervisor
    from localasr.registry import manager
    from localasr.registry.manager import ModelKind

    spec = manager.get_model("qwen3_5-4b-refiner-q4", ModelKind.LLM)
    engine = EngineSupervisor(spec, port=8091, bind="0.0.0.0", api_key="s3cret")
    command = engine._command(Path("/m.gguf"), None)

    assert "--api-key" in command
    assert command[command.index("--api-key") + 1] == "s3cret"
    assert command[command.index("--host") + 1] == "0.0.0.0"


def test_a_dead_child_is_never_reported_as_started(monkeypatch) -> None:  # noqa: ANN001
    """Found by a stray llama-server, not by reading the code.

    The loop checked the child's liveness and then the port's health. A child that had
    already exited was still reported as started, because something *else* was serving
    that port — a leftover from an earlier session. "Started" then meant "somebody is
    listening", which is not the same claim at all.
    """
    from localasr.core.engine.supervisor import SupervisorError

    class Dead:
        pid = 1

        def poll(self):  # noqa: ANN201
            return 2

    companion = host.Companion(port=9099)
    monkeypatch.setattr(host.subprocess, "Popen", lambda *a, **k: Dead())
    # The port answers throughout: this is the stale-server case exactly.
    monkeypatch.setattr(host.Companion, "_answers", lambda _self: True)

    # It answers before the spawn too, so the companion adopts rather than starting.
    assert companion.start() == "http://127.0.0.1:9099"
    assert not companion._owned, "adopting must not claim ownership"

    # Now nothing is listening beforehand, but the port answers once the child is dead.
    answers = iter([False, True, True, True])
    monkeypatch.setattr(host.Companion, "_answers", lambda _self: next(answers))
    with pytest.raises(SupervisorError, match="启动失败"):
        host.Companion(port=9099).start(timeout=5)


def test_stopping_does_not_kill_a_server_it_merely_found(monkeypatch) -> None:  # noqa: ANN001
    """A systemd-managed refiner on the same port outlives the application by design."""
    companion = host.Companion(port=9098)
    monkeypatch.setattr(host.Companion, "_answers", lambda _self: True)
    companion.start()

    killed = []

    class Alive:
        def poll(self):  # noqa: ANN201
            return None

        def terminate(self) -> None:
            killed.append(1)

    companion._process = Alive()
    companion.stop()

    assert killed == [], "it was not ours to stop"


def test_building_a_desktop_host_does_not_load_a_model(monkeypatch) -> None:  # noqa: ANN001
    """Every DesktopHost constructed in a test used to spawn a real llama-server, which
    is how a 2.8 GB model ended up resident during a unit run. Starting belongs to
    `main`, not to `__init__`.

    Checked by watching for the spawn rather than by reading the source: what matters is
    that no process appears, not which line does or does not mention it.
    """
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from localasr.frontends.desktop.app import DesktopHost

    spawned = []
    monkeypatch.setattr(host.Companion, "start", lambda _self, **_k: spawned.append(1))
    monkeypatch.setattr(
        "localasr.context.AppContext.start_node", lambda _self: spawned.append("node")
    )

    app = QApplication.instance() or QApplication([])
    desktop = DesktopHost(app)
    try:
        assert spawned == [], "constructing the host started a model"
        desktop.start_backends()
        desktop._backends.wait(5000)
        assert 1 in spawned, "and main must still be able to start it"
    finally:
        desktop.bridge.stop()
