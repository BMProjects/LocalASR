"""One llama-server per machine, shared across processes.

A single `EngineManager` guarantees one server per process. It cannot stop the subtitle
CLI and the desktop host from each starting their own, and on a 4 GB card the second
one simply fails to allocate. This records the running server so later processes attach
to it instead of competing for the GPU.

The record is advisory: it is a file, not a lease. It is validated against the live
process before use, and a record left behind by a crash is reclaimed rather than
trusted.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EngineRecord:
    pid: int
    base_url: str
    model_id: str
    revision: str


def runtime_dir() -> Path:
    """Prefer XDG_RUNTIME_DIR: it is cleared on logout, so records cannot outlive a
    session and mislead the next one."""
    override = os.environ.get("LOCALASR_RUNTIME_DIR")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg:
        return Path(xdg) / "localasr"
    return Path(os.environ.get("TMPDIR", "/tmp")) / f"localasr-{os.getuid()}"


def record_path() -> Path:
    return runtime_dir() / "engine.json"


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read() -> EngineRecord | None:
    """The currently advertised engine, or None if there is none we can trust."""
    path = record_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        record = EngineRecord(**payload)
    except (OSError, ValueError, TypeError):
        return None
    if not _process_alive(record.pid):
        clear(owner_pid=record.pid)
        return None
    return record


def write(record: EngineRecord) -> None:
    path = record_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(record)), encoding="utf-8")
    tmp.replace(path)


def clear(owner_pid: int | None = None) -> None:
    """Remove the record, optionally only if it belongs to `owner_pid`.

    The guard stops a process from deleting a record another one has since written.
    """
    path = record_path()
    if owner_pid is not None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if payload.get("pid") != owner_pid:
            return
    path.unlink(missing_ok=True)
