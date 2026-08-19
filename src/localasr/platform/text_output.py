"""Delivering dictated text to whatever has focus.

Wayland deliberately prevents an ordinary client from synthesising input, so this is a
chain of increasingly blunt methods. The clipboard fallback always works and never
loses the text, which matters more than the paste succeeding: text on the clipboard is
recoverable, text that vanished is not.

    ydotool  -> /dev/uinput, below the compositor; the reliable path on KDE Wayland
    wtype    -> virtual-keyboard protocol; unreliable on KDE
    clipboard-> copy only, user pastes
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass


class TextOutputError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Delivery:
    method: str
    pasted: bool
    detail: str | None = None


def _run(cmd: list[str], text: str | None = None, timeout: float = 10.0) -> bool:
    try:
        proc = subprocess.run(
            cmd,
            input=text.encode() if text is not None else None,
            capture_output=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def _session_is_wayland() -> bool:
    return os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"


def ydotool_ready() -> tuple[bool, str | None]:
    """Whether ydotool can actually type, and why not when it cannot.

    Installing the package is not enough. ydotool talks to a daemon that opens
    /dev/uinput, so it needs both the daemon running and the device writable — and
    /dev/uinput is root:input, which means group membership, which means a fresh
    login. Reporting "installed" for a binary that will fail on every call is worse
    than reporting nothing.
    """
    if not shutil.which("ydotool"):
        return False, "not installed"
    if not os.access("/dev/uinput", os.W_OK):
        return False, "/dev/uinput is not writable by this user"
    socket = os.environ.get("YDOTOOL_SOCKET") or f"/run/user/{os.getuid()}/.ydotool_socket"
    if not os.path.exists(socket):
        return False, f"ydotoold is not running (no socket at {socket})"
    return True, None


def available_methods() -> list[str]:
    """Which delivery methods this machine can currently perform."""
    methods = []
    ready, _ = ydotool_ready()
    if ready:
        methods.append("ydotool")
    if shutil.which("wtype") and _session_is_wayland():
        methods.append("wtype")
    if shutil.which("xdotool") and not _session_is_wayland():
        methods.append("xdotool")
    if shutil.which("wl-copy") or shutil.which("xclip"):
        methods.append("clipboard")
    return methods


def copy_to_clipboard(text: str) -> bool:
    if shutil.which("wl-copy") and _session_is_wayland():
        return _run(["wl-copy"], text)
    if shutil.which("xclip"):
        return _run(["xclip", "-selection", "clipboard"], text)
    if shutil.which("wl-copy"):
        return _run(["wl-copy"], text)
    return False


def _type_with_ydotool(text: str) -> bool:
    # `--` stops ydotool parsing text that begins with a dash as options.
    return _run(["ydotool", "type", "--", text])


def _type_with_wtype(text: str) -> bool:
    return _run(["wtype", "--", text])


def _type_with_xdotool(text: str) -> bool:
    return _run(["xdotool", "type", "--clearmodifiers", "--", text])


def deliver(
    text: str,
    prefer: str | None = None,
    fallback: Callable[[str], bool] | None = None,
) -> Delivery:
    """Insert `text` at the cursor, falling back to a clipboard.

    `fallback` is a last-resort "put this somewhere the user can paste from". A running
    Qt application always has a clipboard of its own, so the desktop passes one and
    never sees a delivery failure — refusing to hand over correctly recognised text
    because a command-line tool is missing loses work for no reason.

    Returns how it was delivered so the caller can tell the user to paste when typing
    was not possible. Raises only when nothing at all can hold the text.
    """
    if not text:
        return Delivery(method="none", pasted=False, detail="empty text")

    typers = {
        "ydotool": _type_with_ydotool,
        "wtype": _type_with_wtype,
        "xdotool": _type_with_xdotool,
    }
    order = [prefer] if prefer in typers else []
    order += [name for name in ("ydotool", "wtype", "xdotool") if name != prefer]

    for name in order:
        if name in available_methods() and typers[name](text):
            return Delivery(method=name, pasted=True)

    if copy_to_clipboard(text):
        return Delivery(
            method="clipboard",
            pasted=False,
            detail="typing unavailable; text copied, press Ctrl+V to paste",
        )

    if fallback is not None and fallback(text):
        return Delivery(
            method="clipboard",
            pasted=False,
            detail="无法直接输入，文字已复制到剪贴板，请在目标应用中按 Ctrl+V",
        )

    raise TextOutputError(
        "no way to deliver text: install ydotool (and grant /dev/uinput access) or wl-clipboard"
    )


def clipboard_ready() -> tuple[bool, str | None]:
    if shutil.which("wl-copy") or shutil.which("xclip"):
        return True, None
    return False, "neither wl-clipboard nor xclip is installed"


def diagnose() -> list[tuple[str, bool, str]]:
    """Per-check status for `localasr doctor`: (label, ok, remedy-or-detail).

    The remedies are the exact commands for this desktop. `sudo systemctl enable
    ydotool` is wrong on Debian — the unit it ships is a *user* service, and the
    permission it needs comes from group membership, not from the unit.
    """
    checks: list[tuple[str, bool, str]] = []

    ready, reason = ydotool_ready()
    if ready:
        checks.append(("ydotool (type at cursor)", True, "ready"))
    elif reason == "not installed":
        checks.append(("ydotool (type at cursor)", False, "sudo apt install ydotool"))
    elif "uinput" in (reason or ""):
        checks.append(
            (
                "ydotool (type at cursor)",
                False,
                f"{reason}\n"
                f"      sudo usermod -aG input {os.environ.get('USER', 'YOU')}\n"
                "      then log out and back in (a new group needs a new session)",
            )
        )
    else:
        checks.append(
            (
                "ydotool (type at cursor)",
                False,
                f"{reason}\n      systemctl --user enable --now ydotool.service",
            )
        )

    ok, reason = clipboard_ready()
    checks.append(
        (
            "clipboard fallback",
            ok,
            "ready" if ok else "sudo apt install wl-clipboard",
        )
    )
    return checks


def summary() -> str:
    """One-paragraph rendering of `diagnose()`."""
    methods = available_methods()
    if methods:
        return f"text delivery available: {', '.join(methods)}"
    return (
        "no text delivery available — dictation will transcribe but cannot hand the "
        "text anywhere. Run `localasr doctor` for the exact fix."
    )
