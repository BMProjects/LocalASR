"""Models the user supplied themselves.

The shipped catalog pins every file by upstream revision and sha256, which is what
makes a download reproducible. A GGUF the user already has on disk has no upstream
revision to pin, so it cannot live in that catalog without weakening it.

Imported models therefore get their own file, written on this machine, where the
"revision" is a hash of the imported bytes. That keeps the same guarantee — the model
directory is content-addressed and a changed file is a different model — without
pretending the file came from a pinned repository.
"""

from __future__ import annotations

import shutil
import tomllib
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from localasr.registry.manager import (
    FileSpec,
    ModelKind,
    ModelSpec,
    RegistryError,
    data_dir,
    models_dir,
    sha256_of,
)

IMPORTED_PREFIX = "local-"
"""Imported ids are prefixed so they can never collide with a catalog id."""


class ImportError_(RegistryError):
    """Raised when a file cannot be accepted as a model."""


@dataclass(frozen=True, slots=True)
class ImportRequest:
    model_path: Path
    mmproj_path: Path | None = None
    name: str | None = None
    ctx_size: int = 4096
    kind: ModelKind = ModelKind.ASR
    """Which role the file fills. Without this every import became an ASR model, so a
    refiner could be imported and then never selected as one."""

    link: bool = False
    """Reference the file where it is instead of copying it.

    Off by default, because a copy is what makes an imported model keep working after
    the user tidies up their downloads. Worth turning on when the original is managed by
    something that is not going away — another tool's model store — and the duplicate
    would be several gigabytes on the same disk."""


def catalog_path() -> Path:
    return data_dir() / "imported.toml"


def _load() -> dict:
    path = catalog_path()
    if not path.is_file():
        return {}
    try:
        return tomllib.loads(path.read_text(encoding="utf-8")).get("models", {})
    except (OSError, ValueError):
        return {}


def _write(entries: dict) -> None:
    path = catalog_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Models imported on this machine. Managed by `localasr models import`.", ""]
    for model_id, entry in entries.items():
        lines.append(f'[models."{model_id}"]')
        for key, value in entry.items():
            if isinstance(value, dict):
                continue
            if isinstance(value, bool):
                # Before str(), which renders True — TOML wants lowercase, and the
                # reader treats an unparseable file as "nothing was ever imported".
                rendered = "true" if value else "false"
            elif isinstance(value, str):
                rendered = f'"{value}"'
            else:
                rendered = str(value)
            lines.append(f"{key} = {rendered}")
        for role in ("model", "mmproj"):
            if role in entry.get("files", {}):
                spec = entry["files"][role]
                lines.append(f'[models."{model_id}".files.{role}]')
                lines.append(f'name = "{spec["name"]}"')
                lines.append(f'sha256 = "{spec["sha256"]}"')
                lines.append(f"size = {spec['size']}")
        lines.append("")

    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


def _spec_from(model_id: str, entry: dict) -> ModelSpec:
    files = entry["files"]
    return ModelSpec(
        model_id=model_id,
        runtime=entry.get("runtime", "llama-server"),
        repo=entry.get("repo", "imported"),
        revision=entry["revision"],
        ctx_size=entry.get("ctx_size", 4096),
        default=False,
        model=FileSpec(**files["model"]),
        mmproj=FileSpec(**files["mmproj"]) if "mmproj" in files else None,
        kind=ModelKind(entry.get("kind", ModelKind.ASR.value)),
    )


def list_imported() -> list[ModelSpec]:
    return [_spec_from(model_id, entry) for model_id, entry in _load().items()]


def _validate(path: Path, role: str) -> Path:
    path = Path(path).expanduser()
    if not path.is_file():
        raise ImportError_(f"{role} 文件不存在：{path}")
    if path.suffix.lower() != ".gguf":
        raise ImportError_(f"{role} 需要 .gguf 文件，收到 {path.suffix or '无扩展名'}：{path.name}")
    if path.stat().st_size == 0:
        raise ImportError_(f"{role} 文件为空：{path.name}")
    return path


def import_model(
    request: ImportRequest,
    progress: Callable[[str, int, int], None] | None = None,
) -> ModelSpec:
    """Copy a local GGUF into the managed store and register it.

    By default the file is copied: a model that lives wherever the user happened to
    leave it disappears the moment they tidy up their downloads, and the engine would
    then fail at launch instead of at import. `request.link` trades that safety for the
    disk, which is the right trade when the original belongs to another tool's model
    store — several gigabytes duplicated on the same disk buys nothing there.
    """
    model_path = _validate(request.model_path, "模型")
    mmproj_path = _validate(request.mmproj_path, "mmproj") if request.mmproj_path else None

    if progress:
        progress("checksum", 0, 1)
    revision = sha256_of(model_path)[:16]

    stem = request.name or model_path.stem
    model_id = f"{IMPORTED_PREFIX}{stem}".strip().replace(" ", "-").lower()

    entries = _load()
    target_dir = models_dir() / model_id / revision
    target_dir.mkdir(parents=True, exist_ok=True)

    def place(source: Path, index: int, total: int) -> FileSpec:
        destination = target_dir / source.name
        if progress:
            progress(source.name, index, total)
        if request.link:
            # Replace only a link we own. Overwriting a real file here would delete
            # weights the user may have imported by copy earlier.
            if destination.is_symlink():
                destination.unlink()
            elif destination.exists():
                raise ImportError_(
                    f"{destination} 已存在且不是链接；"
                    "请先 `localasr models remove` 再以链接方式导入"
                )
            destination.symlink_to(source.resolve())
        elif not destination.exists() or destination.stat().st_size != source.stat().st_size:
            shutil.copy2(source, destination)
        return FileSpec(
            name=source.name,
            sha256=sha256_of(destination),
            size=destination.stat().st_size,
        )

    total = 2 if mmproj_path else 1
    files = {"model": asdict(place(model_path, 0, total))}
    if mmproj_path:
        files["mmproj"] = asdict(place(mmproj_path, 1, total))

    entries[model_id] = {
        "runtime": "llama-server",
        "repo": "imported",
        "revision": revision,
        "ctx_size": request.ctx_size,
        "kind": request.kind.value,
        "source": str(model_path),
        "linked": request.link,
        "files": files,
    }
    _write(entries)
    if progress:
        progress("done", total, total)
    return _spec_from(model_id, entries[model_id])


def is_imported(model_id: str) -> bool:
    return model_id.startswith(IMPORTED_PREFIX)


def forget(model_id: str) -> None:
    """Drop an imported model from the local catalog."""
    entries = _load()
    if entries.pop(model_id, None) is None:
        raise RegistryError(f"未导入的模型：{model_id}")
    _write(entries)
