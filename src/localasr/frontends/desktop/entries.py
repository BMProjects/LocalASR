"""Desktop entries, so the three applications have real launchers.

All three point at the same single-instance host with a different argument: clicking
「会议记录」when the subtitle window is already open raises the meeting window in the
running process rather than starting a second one.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

APP_DIR = Path(os.environ.get("XDG_DATA_HOME", "~/.local/share")).expanduser() / "applications"
ICON_DIR = (
    Path(os.environ.get("XDG_DATA_HOME", "~/.local/share")).expanduser()
    / "icons/hicolor/scalable/apps"
)

ICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <rect width="64" height="64" rx="14" fill="#2d3436"/>
  <g fill="#00b894">
    <rect x="14" y="26" width="5" height="12" rx="2.5"/>
    <rect x="23" y="18" width="5" height="28" rx="2.5"/>
    <rect x="32" y="12" width="5" height="40" rx="2.5"/>
    <rect x="41" y="20" width="5" height="24" rx="2.5"/>
    <rect x="50" y="28" width="5" height="8" rx="2.5"/>
  </g>
</svg>
"""


@dataclass(frozen=True, slots=True)
class Entry:
    filename: str
    name: str
    comment: str
    argument: str
    terminal: bool = False

    def render(self, executable: str) -> str:
        exec_line = f"{executable} {self.argument}".strip()
        return "\n".join(
            [
                "[Desktop Entry]",
                "Type=Application",
                f"Name={self.name}",
                f"Comment={self.comment}",
                f"Exec={exec_line}",
                "Icon=localasr",
                f"Terminal={'true' if self.terminal else 'false'}",
                # Exactly one main category: listing several makes the launcher show
                # up more than once in the application menu.
                "Categories=AudioVideo;Audio;",
                "Keywords=asr;speech;subtitle;dictation;语音;字幕;听写;",
                "StartupNotify=false",
                "",
            ]
        )


ENTRIES = (
    Entry(
        "localasr-subtitle.desktop",
        "LocalASR 字幕生成",
        "为音视频文件生成字幕",
        "subtitle",
    ),
    Entry(
        "localasr-meeting.desktop",
        "LocalASR 会议记录",
        "录制并转写会议（麦克风 + 系统声音）",
        "meeting",
    ),
    Entry(
        "localasr-dictate.desktop",
        "LocalASR 语音输入",
        "常驻托盘的语音输入；建议同时绑定全局快捷键",
        "dictation",
    ),
)


def desktop_executable() -> str:
    """The command a launcher should run.

    Prefers the installed console script; falls back to `uv run` from the checkout so
    the entries work before the package is installed system-wide.
    """
    installed = shutil.which("localasr-desktop")
    if installed:
        return installed
    uv = shutil.which("uv")
    project = Path(__file__).resolve().parents[4]
    if uv and (project / "pyproject.toml").is_file():
        return f"{uv} run --project {project} localasr-desktop"
    return "localasr-desktop"


def install(target_dir: Path | None = None, icon_dir: Path | None = None) -> list[Path]:
    """Write the three launchers and the icon; returns what was written."""
    target_dir = target_dir or APP_DIR
    icon_dir = icon_dir or ICON_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    icon_dir.mkdir(parents=True, exist_ok=True)

    icon_path = icon_dir / "localasr.svg"
    icon_path.write_text(ICON_SVG, encoding="utf-8")

    executable = desktop_executable()
    written = [icon_path]
    for entry in ENTRIES:
        path = target_dir / entry.filename
        path.write_text(entry.render(executable), encoding="utf-8")
        written.append(path)

    updater = shutil.which("update-desktop-database")
    if updater:
        subprocess.run([updater, str(target_dir)], capture_output=True, timeout=30)
    return written


def uninstall(target_dir: Path | None = None, icon_dir: Path | None = None) -> list[Path]:
    target_dir = target_dir or APP_DIR
    icon_dir = icon_dir or ICON_DIR
    removed = []
    for entry in ENTRIES:
        path = target_dir / entry.filename
        if path.exists():
            path.unlink()
            removed.append(path)
    icon = icon_dir / "localasr.svg"
    if icon.exists():
        icon.unlink()
        removed.append(icon)
    return removed
