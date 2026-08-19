"""What the model panel may do, as a pure function of what is true.

Three things can be true of a model, and they are independent:

    磁盘   未下载 / 下载未完成 / 已下载
    显存   未加载 / 已加载
    选用   未选用 / 当前模型

The panel exposes one action over all three. Pressing it moves the selected model
towards "ready to recognise with", whatever it currently lacks — download it, make it
current, load it — and once it is ready the same button releases it again. Separate
buttons for each step meant most of them were greyed out most of the time, which is
what made the panel hard to read.

Download stays explicit even though it has no button of its own: the label says
「下载并使用」 and the panel confirms the size first. Deleting weights is not here at
all; it is a rare, destructive housekeeping action and lives in `localasr models
remove`.

Keeping this a dataclass-in, dataclass-out function means the rules are testable
without a display, which is how they stop drifting.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class DiskState(enum.Enum):
    ABSENT = "absent"
    PARTIAL = "partial"
    PRESENT = "present"


class Intent(enum.Enum):
    """What pressing the primary button will actually do."""

    DOWNLOAD_AND_USE = "download_and_use"
    USE = "use"
    LOAD = "load"
    RELEASE = "release"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class ModelFacts:
    """Everything the panel needs to decide. No Qt, no I/O."""

    model_id: str
    disk: DiskState
    is_current: bool
    is_loaded: bool
    is_imported: bool
    workload_running: str | None = None
    """Name of the running subtitle/dictation/meeting activity, if any."""
    operation_running: bool = False
    """True while this panel's own download/import is in flight."""


@dataclass(frozen=True, slots=True)
class Action:
    text: str
    enabled: bool
    tip: str = ""
    intent: Intent = Intent.NONE


@dataclass(frozen=True, slots=True)
class PanelState:
    chooser_enabled: bool
    primary: Action
    import_model: Action
    summary: str


def _blocked_reason(facts: ModelFacts) -> str | None:
    if facts.operation_running:
        return "正在处理模型文件，请稍候"
    if facts.workload_running:
        return f"{facts.workload_running} 正在进行，请先停止"
    return None


def plan(facts: ModelFacts) -> PanelState:
    """Decide the primary control's label, enablement, reason and effect."""
    blocked = _blocked_reason(facts)
    idle = blocked is None

    if facts.disk is not DiskState.PRESENT:
        if facts.is_imported:
            # Its source file was a one-off copy; there is nothing to re-fetch from.
            primary = Action(
                "文件缺失", False, "导入的模型文件已不完整，请重新导入", Intent.NONE
            )
        else:
            label = "继续下载并使用" if facts.disk is DiskState.PARTIAL else "下载并使用"
            primary = Action(
                label,
                idle,
                blocked or "下载 catalog 锁定的 revision，校验后设为当前模型并加载",
                Intent.DOWNLOAD_AND_USE,
            )
    elif not facts.is_current:
        primary = Action(
            "使用此模型",
            idle,
            blocked or "设为当前模型并加载到显存",
            Intent.USE,
        )
    elif facts.is_loaded:
        primary = Action(
            "释放显存",
            idle,
            blocked or "立即释放显存；下次识别会重新加载",
            Intent.RELEASE,
        )
    else:
        primary = Action(
            "加载到显存",
            idle,
            blocked or "提前加载，避免第一次识别时等待",
            Intent.LOAD,
        )

    import_model = Action(
        "导入模型…",
        not facts.operation_running,
        "正在处理模型文件，请稍候" if facts.operation_running else "从本机选择 .gguf 文件",
    )

    return PanelState(
        chooser_enabled=not facts.operation_running,
        primary=primary,
        import_model=import_model,
        summary=describe(facts),
    )


def describe(facts: ModelFacts) -> str:
    disk = {
        DiskState.ABSENT: "未下载",
        DiskState.PARTIAL: "下载未完成",
        DiskState.PRESENT: "已下载",
    }[facts.disk]
    memory = "已加载到显存" if facts.is_loaded else "未加载"
    role = "当前模型" if facts.is_current else "未选用"
    return f"{disk} · {memory} · {role}"
