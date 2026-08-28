"""The recognition node, bound to the application's lifetime.

The refinement model gets this by being a child process: the application spawns it and
terminates it. The recognition model cannot, because it lives on another machine — there
is no child to spawn and no signal to send. The analogue over a network is the node's own
API, which already knows how to load and release.

So "starts with the app, stops with the app" is implemented at the layer where the cost
actually is. Measured on the Orin:

    model resident      2277 MiB available
    model released      5158 MiB available   → the model is 2881 MiB
    node service        42-61 MiB RSS

The model is 96% of it. Binding model residency to the session needs no SSH, no remote
paths in this process, and survives a network blip mid-session; binding the service
process too recovers the remaining 4% and adds all three. So residency is the default and
the service is opt-in, through `node_ssh`.

**The session hands the model back when it ends.** That is the whole point, and getting
it wrong is subtle: an earlier version released only a model it had loaded itself, on the
reasoning that unloading somebody else's is rude. Sound in the abstract, and it made the
feature never fire. The node loads on demand, so the model becomes resident the first
time anyone transcribes; every launch after that finds it warm, adopts it, and releases
nothing. 2.8 GB stayed held forever on an 8 GB board — the opposite of the point.

So releasing on exit is the default, and `release_on_exit=False` restores the careful
behaviour for a node that genuinely has more than one client. It is a deployment fact,
not something a client can work out: the node has no notion of sessions, so "already
resident" cannot distinguish "somebody is using this" from "I left it there yesterday".
"""

from __future__ import annotations

import subprocess
import time

import httpx

PROBE_TIMEOUT = 3.0
LOAD_TIMEOUT = 300.0
"""A cold load of Qwen3-ASR-1.7B on a Jetson runs to tens of seconds."""

SERVICE_TIMEOUT = 60.0


class NodeCompanion:
    """Load the node's ASR model for the session, and put it back afterwards."""

    def __init__(
        self,
        url: str,
        *,
        token: str | None = None,
        ssh: str | None = None,
        service: str = "localasr-node",
        release_on_exit: bool = True,
    ) -> None:
        self.url = url.rstrip("/")
        self.token = token
        self.ssh = ssh
        self.service = service
        self.release_on_exit = release_on_exit
        self._loaded_model = False
        """Whether *we* made the model resident. The only thing released when
        `release_on_exit` is off."""
        self._started_service = False

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def answers(self) -> bool:
        try:
            with httpx.Client(timeout=PROBE_TIMEOUT) as client:
                return client.get(f"{self.url}/healthz").status_code == 200
        except httpx.HTTPError:
            return False

    def resident(self) -> str | None:
        """The ASR model the node is holding, if any."""
        try:
            with httpx.Client(timeout=PROBE_TIMEOUT) as client:
                response = client.get(f"{self.url}/readyz", headers=self._headers)
            if response.status_code not in (200, 503):
                return None
            return response.json().get("loaded", {}).get("asr")
        except (httpx.HTTPError, ValueError):
            return None

    def start(self) -> str:
        """Make the node ready to transcribe without a pause on the first sentence.

        Returns what it did, in words the status panel can show.
        """
        if not self.answers():
            if not self.ssh:
                raise RuntimeError(
                    f"识别节点 {self.url} 无响应。启动它，或设置 node_ssh 让本机代为启动。"
                )
            self._start_service()

        already = self.resident()
        if already:
            # Found it warm. Not ours, so not ours to unload later.
            return f"{already}（节点上已驻留）"

        with httpx.Client(timeout=LOAD_TIMEOUT) as client:
            response = client.post(
                f"{self.url}/api/v1/models/load", json={"kind": "asr"}, headers=self._headers
            )
        if response.status_code == 409:
            raise RuntimeError(f"节点拒绝加载：{response.json().get('error', '内存不足')}")
        response.raise_for_status()
        self._loaded_model = True
        loaded = ", ".join(response.json().get("loaded", {}).values()) or "未知"
        return loaded

    def stop(self) -> None:
        """End the session: give the model back, and stop the service if we started it."""
        releasing = self.release_on_exit or self._loaded_model
        self._loaded_model = False
        if releasing:
            try:
                with httpx.Client(timeout=SERVICE_TIMEOUT) as client:
                    client.post(
                        f"{self.url}/api/v1/models/release",
                        json={"kind": "asr"},
                        headers=self._headers,
                    )
            except httpx.HTTPError:
                # The node is unreachable, so its memory is its own problem now. Failing
                # here would only stop the application from exiting.
                pass

        if self._started_service:
            self._started_service = False
            self._run_service("stop")

    def _start_service(self) -> None:
        self._run_service("start")
        self._started_service = True
        deadline = SERVICE_TIMEOUT
        end = time.monotonic() + deadline
        while time.monotonic() < end:
            if self.answers():
                return
            time.sleep(1.0)
        raise RuntimeError(f"{self.ssh} 上的 {self.service} 在 {deadline:.0f}s 内没有就绪")

    def _run_service(self, action: str) -> None:
        """systemd on the far side, rather than a command held open over SSH.

        An SSH pipe holding the remote process would take the node down with any network
        blip, mid-sentence, and put the remote layout — paths, interpreter, environment —
        into this machine's config. The unit already holds all of that; it just does not
        have to be enabled at boot.
        """
        command = ["ssh", "-o", "BatchMode=yes", self.ssh, "systemctl", "--user", action,
                   self.service]
        result = subprocess.run(command, capture_output=True, text=True, timeout=SERVICE_TIMEOUT)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()[:200]
            raise RuntimeError(f"无法 {action} {self.ssh} 上的 {self.service}：{detail}")
