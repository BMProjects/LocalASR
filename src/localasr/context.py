"""The process-wide objects the three applications share.

One engine, one coordinator, one settings file — held here so the desktop host and the
CLI entry points build them the same way. Deliberately free of Qt: the desktop frontend
adds its own bridge on top of `EventBus`.
"""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING

from localasr.apps.bus import EventBus
from localasr.apps.coordinator import Activity, ActivityCoordinator
from localasr.apps.events import ModelChanged, ModelDownloaded
from localasr.core.engine.manager import EngineManager
from localasr.node.companion import NodeCompanion
from localasr.refine import host
from localasr.refine.node_client import NodeRefiner
from localasr.refine.types import (
    FidelityIssue,
    RefinementMode,
    RefinementRequest,
    RefinementResult,
)
from localasr.registry import manager
from localasr.registry.manager import ModelSpec

if TYPE_CHECKING:
    from localasr.core.audio.vad import SileroVad, VadConfig


@dataclass(slots=True)
class Settings:
    """User preferences. Absent file means defaults; unknown keys are ignored."""

    model_id: str | None = None
    language: str | None = None
    device: str | None = None
    idle_timeout: float = 0.0
    """0 keeps the model resident until the application exits, which is what a desktop
    session wants: reloading costs seconds and the user did not ask for it. Set a
    positive number of seconds to reclaim VRAM automatically instead."""
    input_device: str | None = None
    system_device: str | None = None
    node_url: str | None = None
    """Base URL of a LocalASR compute node, e.g. ``http://asr-node.local:8090``.

    Set it and transcription runs there; leave it unset and the local engine is used.
    Nothing else in the pipeline changes — the node speaks the same protocol, so this
    really is a deployment setting rather than a mode.
    """

    node_token: str | None = None
    node_ssh: str | None = None
    """SSH target that may start the recognition node, e.g. ``user@asr-node.local``.

    Optional, and off by default. Without it the node must already be running; with it
    the application starts its systemd unit on demand and stops it on exit. Measured on
    the Orin, that is worth ~100 MiB — the model, which is bound to the session either
    way, is 2881. Set it only if you want the board completely idle between sessions.
    """
    node_service: str = "localasr-node"

    refiner_url: str | None = None
    """Base URL of an OpenAI-compatible chat server for refinement, e.g. Unsloth Studio
    or llama-server on ``http://127.0.0.1:8888``.

    Separate from `node_url` on purpose rather than routed through it. The two backends
    have different jobs and different homes: ASR belongs where the 8 GB of unified memory
    is, refinement where the 4 GB card is, and neither has to know about the other. When
    this is set the desktop talks to it directly; `node_url` then serves audio only.
    """

    refiner_token: str | None = None
    refiner_model_id: str | None = None
    """Catalog id of the refiner this machine starts itself.

    Leave `refiner_url` unset and this set, and the desktop owns the refinement server:
    it can be started before a session and released after, which is the point. Set
    `refiner_url` instead to point at one somebody else runs — Unsloth Studio, a
    llama-server under systemd — and it becomes read-only from here.
    """
    refiner_model: str = "localasr-refiner"
    """Model name sent to the refiner. Unsloth Studio and llama-server both echo it, and
    it is what the journal records — so it should name the actual weights."""

    refine_instruction: str = ""
    """The user's standing refinement request, in their own words.

    Empty means conservative cleaning — punctuation and filler removal, with the
    subsequence proof behind it. Anything else switches to a custom instruction, where
    that proof does not hold and only risk screening remains. Persisted because most
    people want the same thing every time, and retyping it is friction on the one
    action they take most.
    """

    local_asr_fallback: bool = True
    """Whether an unreachable ASR node may be replaced by a local engine.

    Off is the right setting once the refiner lives on this machine: starting a local
    Qwen3-ASR beside a resident Qwen3.5-4B is how a 4 GB card runs out of memory, and it
    would do so silently, mid-sentence. Better to say the recognition node is down.
    """
    hotkey_hint: str = "配置 KDE 自定义快捷键调用 `localasr-desktop dictation` 唤起听写窗口"
    """How to reach dictation from a keyboard shortcut.

    It used to name `localasr dictate --toggle`, which has never existed — the CLI has
    `--seconds` and `--no-deliver`, and nothing that toggles. Running `localasr-desktop
    dictation` is the real path: the single-instance socket hands the request to a host
    that is already running rather than starting a second one.
    """
    keep_meeting_audio: bool = False

    @staticmethod
    def path() -> Path:
        override = os.environ.get("LOCALASR_CONFIG")
        if override:
            return Path(override).expanduser()
        xdg = os.environ.get("XDG_CONFIG_HOME", "~/.config")
        return Path(xdg).expanduser() / "localasr" / "config.toml"

    @classmethod
    def load(cls) -> Settings:
        path = cls.path()
        if not path.is_file():
            return cls()
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        known = {f for f in cls.__slots__}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def save(self) -> Path:
        """Atomically persist the user's preferences as a small TOML file.

        Only what differs from the defaults. Writing every field turns a file the user
        is meant to read and edit into a wall of settings they never chose, and makes a
        later change of default silently invisible — the stale value is already on disk.
        """
        path = self.path()
        path.parent.mkdir(parents=True, exist_ok=True)
        defaults = Settings()
        lines: list[str] = []
        for item in fields(self):
            value = getattr(self, item.name)
            if value is None or value == getattr(defaults, item.name):
                continue
            if isinstance(value, bool):
                encoded = "true" if value else "false"
            elif isinstance(value, str):
                # JSON strings are valid TOML basic strings and correctly escape
                # quotes, newlines and non-ASCII device names.
                encoded = json.dumps(value, ensure_ascii=False)
            else:
                encoded = str(value)
            lines.append(f"{item.name} = {encoded}")

        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        temporary.replace(path)
        return path


@dataclass
class AppContext:
    """Shared state for one desktop host or one CLI invocation."""

    settings: Settings = field(default_factory=Settings.load)
    coordinator: ActivityCoordinator = field(default_factory=ActivityCoordinator)
    bus: EventBus = field(default_factory=EventBus)
    _engine: EngineManager | None = field(default=None, init=False, repr=False)
    _refiner: host.Companion | None = field(default=None, init=False, repr=False)
    _node: NodeCompanion | None = field(default=None, init=False, repr=False)

    @property
    def spec(self) -> ModelSpec:
        return manager.get_model(self.settings.model_id)

    @property
    def engine(self) -> EngineManager:
        if self._engine is None:
            self._engine = EngineManager(
                self.spec,
                idle_timeout=self.settings.idle_timeout,
                device=self.settings.device or os.environ.get("LOCALASR_DEVICE"),
                node_url=self.node_url,
                node_token=self.settings.node_token
                or os.environ.get("LOCALASR_NODE_TOKEN"),
                local_fallback=self.settings.local_asr_fallback,
            )
        return self._engine

    @property
    def node_url(self) -> str | None:
        return self.settings.node_url or os.environ.get("LOCALASR_NODE_URL")

    @property
    def refiner_managed(self) -> bool:
        """Whether this machine runs the refiner itself, or merely uses one.

        Owning it means spawning `localasr.refine.host` as a child — not loading a model
        in this process. `refiner_url` is what hands the job to somebody else: a systemd
        unit, Unsloth Studio, another machine.
        """
        return not self.settings.refiner_url

    @property
    def refiner_spec(self) -> ModelSpec:
        """What the companion would serve. For display only — the module decides."""
        return host.HostConfig(model_id=self.settings.refiner_model_id).resolve()

    @property
    def refiner_base_url(self) -> str | None:
        """Where refinement requests go, if anywhere."""
        if self.settings.refiner_url:
            return self.settings.refiner_url
        return self._refiner.base_url if self._refiner is not None else None

    @property
    def refiner_loaded(self) -> bool:
        return self._refiner is not None and self._refiner.running

    def start_node(self) -> str:
        """Make the recognition node ready for this session.

        Symmetric with `start_refiner`, one layer down: there is no remote process to
        spawn, so what follows the session is the model's residency rather than the
        service. Measured on the Orin that is 2881 MiB of the 2981 the two together cost.
        """
        if not self.node_url:
            raise RuntimeError("未配置识别节点")
        if self._node is None:
            self._node = NodeCompanion(
                self.node_url,
                token=self.settings.node_token or os.environ.get("LOCALASR_NODE_TOKEN"),
                ssh=self.settings.node_ssh,
                service=self.settings.node_service,
            )
        return self._node.start()

    def stop_node(self) -> None:
        """Release what this session caused the node to hold."""
        if self._node is None:
            return
        self._node.stop()
        self._node = None

    def start_refiner(self) -> str:
        """Start the refinement module and wait until it answers.

        Everything about *how* the model loads — which file, which flags, which port —
        lives in that module. This only starts it and reads back a URL, which is what
        makes the systemd deployment and this one the same thing at different lifetimes.
        """
        if not self.refiner_managed:
            raise RuntimeError("整理服务由外部管理，无法从这里启动")
        if self._refiner is None:
            self._refiner = host.Companion(
                model_id=self.settings.refiner_model_id,
                device=self.settings.device or os.environ.get("LOCALASR_DEVICE"),
            )
        return self._refiner.start()

    def stop_refiner(self) -> None:
        """Stop the refinement module, which releases the model's memory."""
        if self._refiner is None:
            return
        self._refiner.stop()
        self._refiner = None

    @property
    def can_refine(self) -> bool:
        """Whether any refiner is reachable: one on this machine, or the ASR node."""
        return bool(self.refiner_base_url or self.node_url)

    def refinement_request(self, raw_text: str, **kwargs) -> RefinementRequest:  # noqa: ANN003
        """Build a request under the user's standing instruction, if they set one."""
        instruction = self.settings.refine_instruction.strip()
        return RefinementRequest(
            raw_text=raw_text,
            mode=RefinementMode.CUSTOM if instruction else RefinementMode.CONSERVATIVE,
            instruction=instruction,
            **kwargs,
        )

    def refine(self, request: RefinementRequest) -> RefinementResult:
        """Tidy one block of transcript. Never raises; a failure returns the original.

        Validation happens here, on the desktop, whatever produced the text. A refiner
        reached directly is still a model server answering over HTTP, and nothing it
        returns may skip the checks against the original.
        """
        base_url = self.refiner_base_url
        if base_url:
            from localasr.refine.client import ChatCompletionClient
            from localasr.refine.service import RefinementService

            with ChatCompletionClient(
                base_url,
                token=self.settings.refiner_token,
                model=self.settings.refiner_model,
            ) as client:
                model_id = (
                    self.refiner_spec.model_id
                    if self.refiner_managed
                    else self.settings.refiner_model
                )
                service = RefinementService(client, model_id=model_id)
                return service.refine(request)

        if not self.can_refine:
            return RefinementResult.rejected(
                request,
                refined_text="",
                issues=(
                    FidelityIssue(
                        "unavailable",
                        "未配置计算节点。在 config.toml 设置 node_url 后才能整理文本。",
                    ),
                ),
            )
        token = self.settings.node_token or os.environ.get("LOCALASR_NODE_TOKEN")
        with NodeRefiner(self.node_url, token=token) as refiner:
            return refiner.refine(request)

    def new_vad(self, config: VadConfig | None = None) -> SileroVad:
        """A fresh VAD per stream: the model carries state across windows, so two
        capture sources cannot share one.

        Imported here rather than at module scope. `AppContext` is what every CLI
        command builds, and a top-level import made `localasr models pull` require
        onnxruntime — so downloading a model failed on the compute node, which has no
        audio stack and no reason to need one to fetch a file.
        """
        from localasr.core.audio.vad import SileroVad

        return SileroVad(manager.vad_path(), config)

    def require_model_available(self) -> None:
        """Refuse a hidden multi-gigabyte download after live capture has started.

        Only meaningful when recognition would run *here*. With a node configured the
        weights on this disk are not what gets used: the engine attaches to the node and
        only ever launches locally as a fallback, which fails with its own message.

        Asking the disk regardless is how a working desktop came to refuse to record.
        Once the local copies were deleted on purpose — recognition having moved to the
        node — the backend panel correctly showed the node ready while 开始识别 answered
        「识别模型尚未准备好」 and named a path that machine has no reason to hold. Two
        checks, one question, opposite answers.
        """
        if self.node_url:
            return
        if manager.is_downloaded(self.spec):
            return
        raise RuntimeError(
            f"识别模型尚未准备好：{manager.model_dir(self.spec)}。"
            f"请先运行 `localasr models pull {self.spec.model_id}`。"
        )

    def prepare_engine(self) -> None:
        """Make the server ready before opening a live capture device."""
        self.require_model_available()
        self.engine.acquire()
        self.engine.release()

    @property
    def engine_loaded_model(self) -> str | None:
        """Which model is resident right now, without loading one to find out."""
        if self._engine is None:
            return None
        return self._engine.spec.model_id if self._engine.loaded else None

    def load_engine(self) -> None:
        """Start the server now, so the first recognition does not wait for it."""
        self.require_model_available()
        self.engine.acquire()
        self.engine.release()

    def unload_engine(self) -> None:
        """Release the GPU. Refused while a workload could be using it."""
        active = self.coordinator.active()
        if active:
            names = "、".join(item.value for item in active)
            raise RuntimeError(f"{names} 正在使用引擎，无法释放显存")
        if self._engine is not None:
            self._engine.shutdown()

    def hold_engine(self) -> None:
        """Load the engine and keep it loaded for the length of a capture session."""
        self.require_model_available()
        self.engine.hold()

    def release_engine(self) -> None:
        if self._engine is not None:
            self._engine.release_hold()

    def download_model(
        self,
        model_id: str,
        progress: manager.DownloadProgress | None = None,
    ) -> ModelSpec:
        """Download one catalog-pinned model as an explicit, exclusive activity."""
        spec = manager.get_model(model_id)
        token = self.coordinator.acquire(Activity.MODEL)
        try:
            manager.pull_model(spec, progress)
        finally:
            self.coordinator.release(token)
        self.bus.publish(ModelDownloaded(model_id=spec.model_id, revision=spec.revision))
        return spec

    def switch_model(self, model_id: str) -> None:
        """Load a different model. Refused while any workload is running."""
        if not self.coordinator.can_switch_model():
            active = ", ".join(a.value for a in self.coordinator.active())
            raise RuntimeError(f"cannot switch model while {active} is running")
        spec = manager.get_model(model_id)
        if not manager.is_downloaded(spec):
            raise RuntimeError(
                f"model {model_id} is not downloaded; download it explicitly before switching"
            )
        previous = self.settings.model_id
        self.settings.model_id = model_id
        try:
            self.settings.save()
        except OSError:
            self.settings.model_id = previous
            raise
        if self._engine is not None:
            self._engine.switch(spec)
        self.bus.publish(ModelChanged(model_id=spec.model_id, revision=spec.revision))

    def shutdown(self) -> None:
        if self._engine is not None:
            self._engine.shutdown()
            self._engine = None
        # A refiner left running holds the GPU for the rest of the session, and nothing
        # will reclaim it: it is our child process, started deliberately, and the only
        # thing that knows about it is going away.
        self.stop_refiner()
        # The node is another machine's, so this releases only what this session made it
        # hold — never a model somebody else loaded.
        self.stop_node()
