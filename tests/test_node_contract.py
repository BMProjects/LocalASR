"""The real client against the real node, with nothing mocked in between.

This file exists because of a bug the rest of the suite could not have caught. The node
served `/healthz`; `TranscriptionClient.is_ready()` probes `/health`. Every readiness
check 404'd, so a perfectly healthy node was always rejected and the desktop fell back
to its local engine — indistinguishable from the node not being there at all.

The node tests missed it by stubbing `is_ready()` to True, which asserts that the code
works if the protocol works. So here the actual client speaks to the actual FastAPI app
over an in-process transport: no stubs, no fakes, and the paths have to line up.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="node requires the [node] extra")

from fastapi.testclient import TestClient  # noqa: E402

from localasr.core.engine.client import TranscriptionClient  # noqa: E402
from localasr.node.server import NodeConfig, create_app  # noqa: E402


def _client_against(app, **kwargs) -> TranscriptionClient:  # noqa: ANN001
    """A genuine TranscriptionClient whose transport is the node app itself.

    TestClient is an httpx.Client subclass driving the ASGI app in-process, so the
    client's real request code runs against the real routes — which is the only way
    a path mismatch shows up.
    """
    client = TranscriptionClient("http://node", **kwargs)
    headers = dict(client._client.headers)
    client._client.close()
    client._client = TestClient(app, base_url="http://node", headers=headers)
    return client


def test_the_desktop_client_finds_a_running_node_ready() -> None:
    """The whole point: an unmodified client must accept a node as its engine."""
    with _client_against(create_app(NodeConfig())) as client:
        assert client.is_ready(), "the readiness path the client probes must exist"


def test_readiness_works_through_the_bearer_token() -> None:
    app = create_app(NodeConfig(token="s3cret"))

    with _client_against(app, token="s3cret") as client:
        assert client.is_ready()

    # Liveness stays open on purpose, so a probe or load balancer can reach it.
    with _client_against(app) as anonymous:
        assert anonymous.is_ready()


def test_the_protected_endpoints_still_require_the_token() -> None:
    """Health being open must not have opened anything else."""
    raw = TestClient(create_app(NodeConfig(token="s3cret")))
    assert raw.get("/api/v1/models").status_code == 401
    assert raw.post("/api/v1/refinements", json={"raw_text": "x"}).status_code == 401


def test_both_spellings_of_health_answer() -> None:
    """`/health` is llama-server's; `/healthz` is the k8s-style one already in use.
    Dropping either breaks somebody's probe."""
    raw = TestClient(create_app(NodeConfig()))
    assert raw.get("/health").status_code == 200
    assert raw.get("/healthz").status_code == 200


def test_readiness_is_not_confused_with_liveness() -> None:
    """A node that has not loaded ASR yet is alive and usable — it loads on demand.
    Probing /readyz for readiness would reject it before the first request."""
    app = create_app(NodeConfig())
    raw = TestClient(app)
    assert raw.get("/health").status_code == 200
    assert raw.get("/readyz").status_code == 503

    with _client_against(app) as client:
        assert client.is_ready(), "liveness, not model residency, is what gates use"
