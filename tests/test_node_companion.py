"""Binding the remote ASR model to the session.

The recognition model lives on another machine, so there is no child to spawn and no
signal to send. What follows the session is the model's residency, through the node's own
API — and the property worth most of these tests is that nothing is released which this
process did not cause.
"""

from __future__ import annotations

import httpx
import pytest

from localasr.node.companion import NodeCompanion

URL = "http://asr-node.local:8090"


class FakeNode:
    """A node that answers, with the transport swapped out under httpx."""

    def __init__(self, *, healthy: bool = True, resident: str | None = None) -> None:
        self.healthy = healthy
        self.resident = resident
        self.loads: list[dict] = []
        self.releases: list[dict] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if not self.healthy:
            raise httpx.ConnectError("node is down", request=request)
        if path == "/healthz":
            return httpx.Response(200, json={"status": "ok"})
        if path == "/readyz":
            loaded = {"asr": self.resident} if self.resident else {}
            return httpx.Response(200 if self.resident else 503, json={"loaded": loaded})
        if path == "/api/v1/models/load":
            import json

            self.loads.append(json.loads(request.content))
            self.resident = "qwen3-asr-1_7b-q8"
            return httpx.Response(200, json={"loaded": {"asr": self.resident}})
        if path == "/api/v1/models/release":
            import json

            self.releases.append(json.loads(request.content))
            self.resident = None
            return httpx.Response(200, json={"released": ["asr"]})
        return httpx.Response(404)


@pytest.fixture
def node(monkeypatch):  # noqa: ANN001, ANN201
    fake = FakeNode()
    original = httpx.Client.__init__

    def patched(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        kwargs["transport"] = httpx.MockTransport(fake.handle)
        original(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", patched)
    return fake


def test_a_session_loads_the_model_and_gives_it_back(node) -> None:  # noqa: ANN001
    companion = NodeCompanion(URL)
    companion.start()

    assert node.loads == [{"kind": "asr"}], "the model has to be warm before the first word"

    companion.stop()
    assert node.releases == [{"kind": "asr"}], "2881 MiB, measured — it must not be left held"


def test_a_warm_model_is_used_rather_than_loaded_again(node) -> None:  # noqa: ANN001
    """Adoption is still right on the way in: reloading a resident model would cost
    thirty seconds to arrive at the state already in front of us."""
    node.resident = "qwen3-asr-1_7b-q8"
    companion = NodeCompanion(URL)

    assert "已驻留" in companion.start()
    assert node.loads == [], "it was already warm"


def test_a_token_becomes_a_bearer_header(node) -> None:  # noqa: ANN001
    """The node refuses every protected endpoint without one, so an unauthenticated
    companion would fail at load time rather than at configuration time."""
    assert NodeCompanion(URL, token="s3cret")._headers == {"Authorization": "Bearer s3cret"}
    assert NodeCompanion(URL)._headers == {}


def test_an_unreachable_node_without_ssh_says_what_to_do(node) -> None:  # noqa: ANN001
    node.healthy = False
    with pytest.raises(RuntimeError, match="node_ssh"):
        NodeCompanion(URL).start()


def test_a_release_that_fails_does_not_stop_the_application_exiting(node) -> None:  # noqa: ANN001
    """Quitting must not hang on a board that has gone away; its memory is its own
    problem at that point."""
    companion = NodeCompanion(URL)
    companion.start()
    node.healthy = False

    companion.stop()  # must not raise


def test_the_service_is_only_stopped_if_this_process_started_it(node, monkeypatch) -> None:  # noqa: ANN001
    calls = []
    monkeypatch.setattr(NodeCompanion, "_run_service", lambda _self, action: calls.append(action))

    companion = NodeCompanion(URL, ssh="user@asr-node.local")
    companion.start()  # the node already answers, so nothing is started
    companion.stop()

    assert calls == [], "a node that was already up is not ours to stop"


def test_an_unreachable_node_with_ssh_is_started_and_stopped(node, monkeypatch) -> None:  # noqa: ANN001
    calls = []

    def run(_self, action):  # noqa: ANN001, ANN202
        calls.append(action)
        node.healthy = True  # systemd brought it up

    node.healthy = False
    monkeypatch.setattr(NodeCompanion, "_run_service", run)

    companion = NodeCompanion(URL, ssh="user@asr-node.local")
    companion.start()
    assert calls == ["start"]

    companion.stop()
    assert calls == ["start", "stop"]


def test_starting_the_service_uses_systemd_rather_than_a_held_ssh_pipe(monkeypatch) -> None:  # noqa: ANN001
    """An SSH pipe holding the remote process would take the node down with any network
    blip, mid-sentence, and would put the remote paths and interpreter into this
    machine's config. The unit on the far side already holds all of that."""
    seen = []

    class Result:
        returncode = 0
        stdout = stderr = ""

    monkeypatch.setattr(
        "localasr.node.companion.subprocess.run",
        lambda cmd, **kw: (seen.append(cmd), Result())[1],
    )
    NodeCompanion(URL, ssh="user@asr-node", service="localasr-node")._run_service("start")

    assert seen[0][:3] == ["ssh", "-o", "BatchMode=yes"]
    assert seen[0][-3:] == ["--user", "start", "localasr-node"]
    assert not any("nohup" in part or part == "&" for part in seen[0])


def test_a_model_left_over_from_an_earlier_session_is_still_released(node) -> None:  # noqa: ANN001
    """The bug that made the whole feature never fire.

    Refusing to release a model this session did not load is right on a shared node and
    wrong everywhere else — and it is self-perpetuating. The node loads on demand, so the
    model becomes resident the first time anyone transcribes; from then on every launch
    finds it warm, adopts it, and releases nothing. 2.8 GB stays held forever on an 8 GB
    board, which is the opposite of what binding residency to the session is for.
    """
    node.resident = "qwen3-asr-1_7b-q8"
    companion = NodeCompanion(URL)

    companion.start()
    assert node.loads == [], "it was already warm; loading again would be pointless"

    companion.stop()
    assert node.releases == [{"kind": "asr"}], "the session ended, so the memory goes back"


def test_a_shared_node_can_keep_the_old_discipline(node) -> None:  # noqa: ANN001
    """Opt-out, because "the last client to quit unloads your model" is a bad rule when
    there really is more than one client."""
    node.resident = "qwen3-asr-1_7b-q8"
    companion = NodeCompanion(URL, release_on_exit=False)

    companion.start()
    companion.stop()

    assert node.releases == [], "not ours to release, and this deployment says so"


def test_the_opt_out_still_releases_what_this_session_loaded(node) -> None:  # noqa: ANN001
    companion = NodeCompanion(URL, release_on_exit=False)
    companion.start()
    assert node.loads == [{"kind": "asr"}]

    companion.stop()
    assert node.releases == [{"kind": "asr"}], "we loaded it, so we hand it back"
