"""Readiness check for the three applications.

Each application needs a different subset of the machine, and each missing piece has a
different fix. Reporting them together — with the command that repairs each one — is
cheaper than discovering them one failure at a time.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

from localasr.platform import text_output


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    ok: bool
    detail: str
    required_by: tuple[str, ...]


def _recognition_check(apps: tuple[str, ...]) -> Check:
    """Whether recognition can happen — not whether a file is on this disk.

    With a node configured the weights here are never used, so asking the disk reported
    all three applications as 未就绪 on a machine that worked perfectly: the local copies
    had been deleted precisely because recognition moved to the node. Doctor exists to
    say whether the thing will run, so it has to look where it actually runs.
    """
    from localasr.context import Settings
    from localasr.registry import manager

    settings = Settings.load()
    node_url = settings.node_url or os.environ.get("LOCALASR_NODE_URL")
    if node_url:
        import httpx

        headers = {}
        token = settings.node_token or os.environ.get("LOCALASR_NODE_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            with httpx.Client(timeout=3.0) as client:
                if client.get(f"{node_url}/healthz").status_code != 200:
                    return Check("recognition node", False, f"{node_url} 未回应健康检查", apps)
                ready = client.get(f"{node_url}/readyz", headers=headers)
            body = ready.json() if ready.status_code in (200, 503) else {}
            loaded = body.get("loaded", {}).get("asr")
        except (httpx.HTTPError, ValueError) as exc:
            detail = f"{node_url} 不可达（{type(exc).__name__}）"
            return Check("recognition node", False, detail, apps)
        # 503 is not a fault: the node loads on demand and has simply not been asked yet.
        detail = f"{node_url} · {loaded} 已驻留" if loaded else f"{node_url} · 首次识别时加载"
        return Check("recognition node", True, detail, apps)

    spec = manager.get_model()
    downloaded = manager.is_downloaded(spec)
    return Check(
        f"model {spec.model_id}",
        downloaded,
        f"{manager.model_dir(spec)}" if downloaded else "localasr models pull",
        apps,
    )


def _engine_checks() -> list[Check]:
    from localasr.core.engine.supervisor import SupervisorError, check_build, find_llama_server

    checks: list[Check] = []
    apps = ("subtitle", "dictate", "meeting")

    try:
        binary = find_llama_server()
        warning = check_build(binary)
        checks.append(
            Check("llama-server", True, warning or f"{binary}", apps)
            if warning is None
            else Check("llama-server", True, warning, apps)
        )
    except SupervisorError as exc:
        checks.append(Check("llama-server", False, str(exc), apps))

    checks.append(_recognition_check(apps))

    ffmpeg = shutil.which("ffmpeg")
    checks.append(Check("ffmpeg", bool(ffmpeg), ffmpeg or "sudo apt install ffmpeg", ("subtitle",)))
    return checks


def _capture_checks() -> list[Check]:
    from localasr.capture import pulse
    from localasr.capture.microphone import CaptureError, list_devices

    checks: list[Check] = []
    try:
        devices = list_devices()
        checks.append(
            Check(
                "microphone (PortAudio)",
                bool(devices),
                f"{len(devices)} input device(s)" if devices else "no input devices found",
                ("dictate", "meeting"),
            )
        )
    except CaptureError as exc:
        checks.append(
            Check(
                "microphone (PortAudio)",
                False,
                f"{exc}\n      sudo apt install libportaudio2",
                ("dictate", "meeting"),
            )
        )

    monitor = pulse.default_monitor() if pulse.available() else None
    if monitor:
        detail = monitor
    elif not pulse.available():
        detail = "parec not found — sudo apt install pulseaudio-utils"
    elif (unreachable := pulse.server_error()) is not None:
        # Not the same finding as "this sink has no monitor", and the remedy is
        # nothing alike. Saying the wrong one sends the user looking at their audio
        # routing for a problem that is in the environment the process was started in.
        detail = unreachable
        if not os.environ.get("XDG_RUNTIME_DIR"):
            detail += (
                "\n      XDG_RUNTIME_DIR is unset, so pactl cannot find the server's"
                "\n      socket. Start from a desktop session, or export"
                "\n      XDG_RUNTIME_DIR=/run/user/$(id -u)"
            )
    else:
        detail = "no monitor for the default sink; meeting will record microphone only"
    checks.append(Check("system audio (parec monitor)", bool(monitor), detail, ("meeting",)))
    return checks


def run_checks() -> list[Check]:
    checks = _engine_checks() + _capture_checks()
    for name, ok, detail in text_output.diagnose():
        checks.append(Check(name, ok, detail, ("dictate",)))
    return checks


def blocking_for(app: str, checks: list[Check]) -> list[Check]:
    """Failed checks that stop `app` from working at all."""
    return [c for c in checks if app in c.required_by and not c.ok]


def hotkey_hint() -> str:
    user = os.environ.get("USER", "you")
    return (
        "全局热键（Wayland 下客户端无法抢占按键，交给 KDE）：\n"
        "  系统设置 → 快捷键 → 添加 → 命令/URL\n"
        f"  命令：localasr dictate      （或 /home/{user}/.local/bin/localasr dictate）\n"
        "  绑定你习惯的组合键，例如 Meta+Z"
    )
