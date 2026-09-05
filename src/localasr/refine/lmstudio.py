"""LM Studio's own server, driven the way the recognition node is.

There are two ways to put a refinement model in front of this application, and until now
only one of them could be turned on and off from it. `refine.host` spawns llama-server as
a child, so the button that starts it is a `Popen`. An external `refiner_url` was the
other way, and it was read-only by definition: a URL is not a process, and there is no
signal to send to somebody else's server.

LM Studio breaks that assumption, because its server exposes model residency as part of
the API rather than as an implementation detail:

    GET  /api/v1/models          every downloaded model, and its `loaded_instances`
    POST /api/v1/models/load     {"model": "<key>"}       → {"instance_id", "status"}
    POST /api/v1/models/unload   {"instance_id": "<id>"}  → {"instance_id"}

That is the same shape as the compute node's own load/release endpoints, which is why
this class is the same shape as `NodeCompanion`: `answers`, what is resident, `start`
for the session, `stop` afterwards. Nothing new is invented; a second server that can
answer the same questions gets the same object.

**Loading is about latency, not correctness.** LM Studio's `justInTimeModelLoading` is on
by default, so a refinement request loads the model by itself. What it cannot do is load
it *before* the user is waiting for it, and it will not unload it when this application
exits — a 4B Q4 holds ~2.9 GB until something says otherwise. Both halves are the point.

Docs: https://lmstudio.ai/docs/api/rest-api (load, unload, list). Default port 1234.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx

DEFAULT_URL = "http://127.0.0.1:1234"
"""LM Studio's server port, and loopback unless the user changes `networkInterface`."""

PROBE_TIMEOUT = 3.0
LOAD_TIMEOUT = 300.0
"""A cold load reads several gigabytes off disk before it answers."""


class LMStudioError(RuntimeError):
    """The server answered, and the answer was no."""


def _headers(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def speaks_lmstudio(url: str, token: str | None = None) -> bool:
    """Whether `url` is an LM Studio server rather than some other OpenAI-compatible one.

    Asked rather than configured. Every server this application can talk to serves
    `/v1/chat/completions`; only LM Studio also serves `/api/v1/models` returning a
    `models` list. llama-server answers 404 there, which is an unambiguous no, and one
    fewer setting for the user to know about and get wrong.
    """
    try:
        with httpx.Client(timeout=PROBE_TIMEOUT) as client:
            response = client.get(f"{url.rstrip('/')}/api/v1/models", headers=_headers(token))
        return response.status_code == 200 and isinstance(response.json().get("models"), list)
    except (httpx.HTTPError, ValueError):
        return False


@dataclass
class LMStudioCompanion:
    """Hold LM Studio's refinement model for the session, and hand it back afterwards."""

    url: str = DEFAULT_URL
    token: str | None = None
    model: str | None = None
    """The model key, e.g. ``unsloth/Qwen3.5-4B-MTP-GGUF``. When unset, the one LLM this
    LM Studio has is not a guess; several are ambiguous and say so."""

    context_length: int | None = None
    release_on_exit: bool = True
    """Unload on exit even if this session did not load it.

    The same deployment fact `NodeCompanion` carries, for the same reason: the server has
    no notion of sessions, so "already resident" cannot distinguish "somebody else is
    using this" from "I left it there yesterday". Turn it off when LM Studio is genuinely
    shared — with its own chat window, for instance."""

    _instance: str | None = field(default=None, init=False, repr=False)
    """The instance `start` left loaded or adopted. Nothing else is ever unloaded."""

    _loaded: bool = field(default=False, init=False, repr=False)
    """Whether *we* made it resident — the only thing unloaded when `release_on_exit`
    is off."""

    def __post_init__(self) -> None:
        self.url = self.url.rstrip("/")

    def _get(self, path: str) -> dict:
        with httpx.Client(timeout=PROBE_TIMEOUT) as client:
            response = client.get(f"{self.url}{path}", headers=_headers(self.token))
        response.raise_for_status()
        return response.json()

    def answers(self) -> bool:
        try:
            self._get("/api/v1/models")
        except (httpx.HTTPError, ValueError):
            return False
        return True

    def catalog(self) -> list[dict]:
        """Every model LM Studio has downloaded, LLMs and embedding models alike."""
        return self._get("/api/v1/models").get("models", [])

    def resident(self) -> tuple[str, ...]:
        """Instance ids currently loaded. Empty is the normal state before a session."""
        return tuple(
            instance["id"]
            for entry in self.catalog()
            for instance in entry.get("loaded_instances", [])
            if "id" in instance
        )

    def key(self) -> str:
        """Which model to load.

        `model` when the user named one. Otherwise the same rule `refine.host` applies to
        this machine's own catalog: exactly one candidate is an answer, none and several
        are questions — asked here rather than resolved into whichever came first.
        """
        if self.model:
            return self.model
        llms = [entry for entry in self.catalog() if entry.get("type") == "llm"]
        if len(llms) == 1:
            return llms[0]["key"]
        if not llms:
            raise LMStudioError(f"{self.url} 上没有可用于整理的模型，请先在 LM Studio 里下载一个")
        names = "、".join(entry.get("key", "?") for entry in llms)
        raise LMStudioError(f"LM Studio 上有多个模型（{names}），请用 refiner_model 指定一个")

    def start(self) -> str:
        """Load the model now, so the first refinement is not the one that waits.

        Returns what happened, in words the status panel can show.
        """
        if not self.answers():
            raise LMStudioError(
                f"LM Studio 服务 {self.url} 无响应。请在 LM Studio 里启动本地服务器"
                "（Developer → Start Server，或 `lms server start`）。"
            )
        key = self.key()
        if key in self.resident():
            # Found it warm. Adopted, not loaded: whether it is ours to unload afterwards
            # is what `release_on_exit` answers.
            self._instance = key
            return f"{key}（LM Studio 上已加载）"

        body: dict[str, object] = {"model": key}
        if self.context_length:
            body["context_length"] = self.context_length
        try:
            with httpx.Client(timeout=LOAD_TIMEOUT) as client:
                response = client.post(
                    f"{self.url}/api/v1/models/load", json=body, headers=_headers(self.token)
                )
            if response.status_code >= 400:
                raise LMStudioError(f"LM Studio 拒绝加载 {key}：{_reason(response)}")
            payload = response.json()
        except httpx.HTTPError as exc:
            raise LMStudioError(f"加载 {key} 失败：{exc}") from exc

        self._instance = payload.get("instance_id", key)
        self._loaded = True
        seconds = payload.get("load_time_seconds")
        return f"{self._instance}（{seconds:.1f}s）" if seconds else str(self._instance)

    def stop(self) -> None:
        """Unload, so the card is not still holding a refiner after the window closes."""
        instance, loaded = self._instance, self._loaded
        self._instance, self._loaded = None, False
        if instance is None or not (self.release_on_exit or loaded):
            return
        try:
            with httpx.Client(timeout=LOAD_TIMEOUT) as client:
                client.post(
                    f"{self.url}/api/v1/models/unload",
                    json={"instance_id": instance},
                    headers=_headers(self.token),
                )
        except httpx.HTTPError:
            # Unreachable at exit means its memory is its own problem now. Raising here
            # would only stop the application from closing.
            pass


def _reason(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip()[:200] or f"HTTP {response.status_code}"
    for key in ("error", "message", "detail"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, dict) and isinstance(value.get("message"), str):
            return value["message"]
    return f"HTTP {response.status_code}"
