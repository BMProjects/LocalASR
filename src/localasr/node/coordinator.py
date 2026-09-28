"""Which models are resident, on a machine where that is a real question.

A desktop with discrete VRAM can treat "is there room?" as a question about a separate
pool. A Jetson cannot: the GPU allocates from the same memory the OS, the desktop
session and the page cache are using, so a model that fits in principle can still push
the machine into swap and take the whole system down with it.

There is one slot per role, and the two slots are **independent**: loading or releasing
one never touches the other. Keeping ASR and the refiner both resident removes a ~35 s
model swap from every refinement, which is the whole cost of the sequential arrangement.

When the memory is not there, the coordinator **refuses and says what is holding it**
rather than evicting. Evicting was the older behaviour and it was the wrong trade: it
unloaded a model out from under whoever was using it, to serve a request that had no
claim on that memory, and it made the two roles interfere in a way nothing on screen
could explain. Which model to release is the user's decision — the same principle as
the desktop's manual load/unload controls.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from localasr.core.engine.supervisor import EngineSupervisor, SupervisorError
from localasr.registry.manager import ModelKind, ModelSpec

SHARED_OVERHEAD_MB = 170
"""What a second model does *not* have to pay again.

Measured on the Orin: Qwen3-ASR-1.7B costs 3964 MiB loaded alone but 3791 MiB when it
joins an already-running refiner — the CUDA context and the shared library pages are
paid once, not per process. Summing per-model figures therefore over-states a pair, and
by enough to refuse a combination that in fact leaves a gigabyte free."""

DEFAULT_MIN_FREE_MB = 900
"""Headroom kept after any load.

Measured rather than chosen: 1.7B ASR beside the 2B refiner leaves 1002 MiB on this
board, so a 1024 MiB reserve refuses the pair by 22 MiB — precision the estimates do not
have. 900 admits it while still keeping most of a gigabyte for the OS, and the desktop
session (619 MiB, measured) can still be started on top, though that leaves ~380 MiB and
pushes swap up: usable, not comfortable."""

OVERHEAD_FACTOR = 1.4
"""Fallback multiplier on GGUF weight size, for models with no measurement yet.

A single factor cannot describe every model — Qwen3.5-4B measured 1.29x its weights
while Qwen3-ASR-0.6B measured 1.60x, since the mmproj file and the fixed per-process
cost are a much larger share of a small model. So `measured_mb` in the catalog wins
whenever it is present, and this only covers what has not been run yet. It sits above
the highest measured ratio for the large models and below the small ones deliberately:
under-estimating a big model is the expensive mistake."""


@dataclass(frozen=True, slots=True)
class Memory:
    total_mb: int
    available_mb: int

    @classmethod
    def read(cls, path: Path = Path("/proc/meminfo")) -> Memory | None:
        """Available memory as the kernel reports it.

        `MemAvailable`, not `MemFree`: reclaimable page cache is genuinely available,
        and on a Jetson with a warm cache `MemFree` reads near zero at all times.
        """
        try:
            text = path.read_text()
        except OSError:
            return None
        values: dict[str, int] = {}
        for line in text.splitlines():
            key, _, rest = line.partition(":")
            if key in {"MemTotal", "MemAvailable"}:
                values[key] = int(rest.split()[0]) // 1024
        if "MemTotal" not in values or "MemAvailable" not in values:
            return None
        return cls(total_mb=values["MemTotal"], available_mb=values["MemAvailable"])


def estimated_cost_mb(spec: ModelSpec) -> int:
    """What loading `spec` will cost, not what its files weigh.

    Weights are the floor, never the total: llama.cpp also allocates compute buffers, a
    CUDA context, and a KV cache sized by the context length, and the load itself peaks
    above the steady state.
    """
    if spec.measured_mb:
        return spec.measured_mb
    weights_mb = sum(file.size for file in spec.files) // (1024 * 1024)
    return int(weights_mb * OVERHEAD_FACTOR)


class ResourceCoordinator:
    """Owns one llama-server per role, keeping both only when the memory is there.

    All mutation happens under one lock, so two loads arriving together still see a
    consistent view of what is resident and what it costs.
    """

    def __init__(
        self,
        *,
        min_free_mb: int = DEFAULT_MIN_FREE_MB,
        log_dir: Path | None = None,
    ) -> None:
        """`min_free_mb` is the whole policy: what the machine keeps for itself.

        There used to be an `allow_coresident` switch as well, whose only effect was to
        decide whether a load may evict the *other* role. Nothing evicts another role
        now, so the switch had nothing left to select — and the headroom check is the
        honest guard in any case: it refuses on measured cost instead of quietly
        unloading a model somebody is still using.
        """
        self.min_free_mb = min_free_mb
        self.log_dir = log_dir
        self._lock = threading.RLock()
        self._servers: dict[ModelKind, EngineSupervisor] = {}
        self._specs: dict[ModelKind, ModelSpec] = {}
        self._resident: dict[str, str] = {}
        """What `loaded()` reports. Replaced whole, never mutated, so reading it needs
        no lock — see `loaded()`."""
        self._loading: frozenset[str] = frozenset()

    def loaded(self) -> dict[str, str]:
        """Resident models keyed by role. Never waits for a load in progress.

        This used to take the same lock `ensure()` holds for the whole of a load — 25-40 s
        on a Jetson — so `/readyz` hung for exactly the window in which somebody is most
        likely to ask it. The desktop's three-second probe timed out, painted the node
        red as unreachable, and turned green again when the load finished: a failure
        shown for a load that was succeeding. Readers get the last published snapshot
        instead; `loading()` says what is on its way.
        """
        return dict(self._resident)

    def loading(self) -> list[str]:
        """Roles whose load is in progress right now."""
        return sorted(self._loading)

    def _publish(self) -> None:
        # Called with the lock held, after every change to `_specs`. Rebinding the
        # attribute is atomic, so a reader sees either the old dict or the new one.
        self._resident = {kind.value: spec.model_id for kind, spec in self._specs.items()}

    def ensure(self, spec: ModelSpec) -> str:
        """Make `spec` resident and return its base URL.

        Replacing this role's own model is the only unload that happens here. The other
        role is never touched, so a refinement cannot cost somebody their loaded ASR.
        """
        with self._lock:
            current = self._specs.get(spec.kind)
            if current is not None and current.model_id == spec.model_id:
                return self._servers[spec.kind].base_url

            # Replacing this role's own model always frees its memory first: the
            # outgoing weights are exactly what the incoming ones need.
            if current is not None:
                self._release(spec.kind)

            # No eviction of the *other* role, ever. Loading a refiner used to unload
            # the ASR model out from under whoever was dictating, which is both a
            # surprise and a minute of reloading; and releasing memory is the user's
            # decision, not something a load request makes on their behalf. If it does
            # not fit, refusing and saying what is holding the memory leaves them able
            # to choose.
            self._check_headroom(spec)
            self._loading = self._loading | {spec.kind.value}
            try:
                return self._launch(spec)
            finally:
                self._loading = self._loading - {spec.kind.value}

    def release(self, kind: ModelKind) -> None:
        with self._lock:
            self._release(kind)

    def shutdown(self) -> None:
        with self._lock:
            for kind in list(self._servers):
                self._release(kind)

    def _launch(self, spec: ModelSpec) -> str:
        server = EngineSupervisor(spec)
        log = self.log_dir / f"{spec.kind.value}.log" if self.log_dir else None
        client = server.start(log_path=log)
        # The probe client has done its job; the node builds a role-appropriate one from
        # the base URL. Holding this one open would leak a connection per load.
        client.close()
        self._servers[spec.kind] = server
        self._specs[spec.kind] = spec
        self._publish()
        return server.base_url

    def _joining_cost_mb(self, spec: ModelSpec) -> int:
        """What `spec` costs *given what is already running*.

        A second llama-server does not pay for the CUDA context or the shared library
        pages again, so charging it the full solo figure is what made the two checks
        below disagree: one would admit the pair and the other then refuse it.
        """
        discount = SHARED_OVERHEAD_MB if self._servers else 0
        return max(estimated_cost_mb(spec) - discount, 0)

    def _release(self, kind: ModelKind) -> None:
        server = self._servers.pop(kind, None)
        self._specs.pop(kind, None)
        self._publish()
        if server is not None:
            server.stop()

    def estimated_cost_mb(self, spec: ModelSpec) -> int:
        """Kept as a method so callers need not import the module function."""
        return estimated_cost_mb(spec)

    def _check_headroom(self, spec: ModelSpec) -> None:
        """Refuse a load that would leave the machine with nothing to work in.

        Failing here is not the unhelpful option: the alternative is the OOM killer
        choosing what dies, and on a headless node that is usually the node.

        Since nothing is evicted to make room, the message has to name what is holding
        the memory — otherwise "does not fit" is a dead end, and the one action that
        would fix it is invisible.
        """
        memory = Memory.read()
        if memory is None:
            return
        needed = self._joining_cost_mb(spec)
        if memory.available_mb - needed < self.min_free_mb:
            resident = ", ".join(
                f"{kind.value}={loaded.model_id}" for kind, loaded in self._specs.items()
            )
            hint = f"；先卸载已驻留的模型（{resident}）" if resident else ""
            raise SupervisorError(
                f"refusing to load {spec.model_id}: needs ~{needed} MiB, "
                f"{memory.available_mb} MiB available, policy keeps "
                f"{self.min_free_mb} MiB free{hint}"
            )
