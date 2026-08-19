"""Model catalog: where pinned assets live on disk and how they are fetched.

Every download is verified against the catalog's sha256 and size before it is usable,
and a failed verification leaves nothing behind. An unverified partial file that looks
complete is worse than a missing one: it fails later, somewhere else, as bad output.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import os
import shutil
import threading
import tomllib
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from importlib import resources
from pathlib import Path

_HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
_CHUNK = 1 << 20
_DOWNLOAD_LOCK = threading.Lock()

DownloadProgress = Callable[[str, int, int], None]


class RegistryError(RuntimeError):
    pass


class IntegrityError(RegistryError):
    """A downloaded or on-disk file does not match its pinned hash or size."""


class ModelKind(StrEnum):
    """What a catalog entry is for.

    Kept in the catalog rather than inferred from the model name: a refiner and an ASR
    model are launched with different flags, live in different servers, and on shared
    memory must be able to evict each other.
    """

    ASR = "asr"
    LLM = "llm"


@dataclass(frozen=True, slots=True)
class FileSpec:
    name: str
    sha256: str
    size: int


@dataclass(frozen=True, slots=True)
class ModelSpec:
    model_id: str
    runtime: str
    repo: str
    revision: str
    ctx_size: int
    default: bool
    model: FileSpec
    mmproj: FileSpec | None
    measured_mb: int = 0
    """Resident cost in MiB as actually measured on the target machine, or 0.

    Preferred over any multiplier when present. A single factor cannot describe both:
    Qwen3.5-4B measured 1.29x its weights while Qwen3-ASR-0.6B measured 1.60x, because
    the mmproj file and the fixed per-process overhead are a much larger share of a
    small model."""

    kind: ModelKind = ModelKind.ASR
    """Which role this model fills. ASR and refinement are separate servers with
    separate lifetimes, and on a Jetson they compete for the same unified memory, so
    the coordinator has to be able to tell them apart."""


    @property
    def files(self) -> list[FileSpec]:
        return [f for f in (self.model, self.mmproj) if f is not None]


def data_dir() -> Path:
    """Root for downloaded assets.

    An explicit environment override wins. In an editable checkout, reuse its existing
    ``models/`` directory: graphical launchers do not inherit shell ``.bashrc`` exports,
    and silently downloading a second 2.5 GB model into XDG data is never desirable.
    Installed packages without a checkout keep the normal XDG location.
    """
    override = os.environ.get("LOCALASR_DATA_DIR")
    if override:
        return Path(override).expanduser()
    checkout = Path(__file__).resolve().parents[3]
    if (checkout / "pyproject.toml").is_file() and (checkout / "models").is_dir():
        return checkout
    xdg = os.environ.get("XDG_DATA_HOME", "~/.local/share")
    return Path(xdg).expanduser() / "localasr"


def models_dir() -> Path:
    return data_dir() / "models"


def _catalog() -> dict:
    raw = resources.files("localasr.registry").joinpath("catalog.toml").read_bytes()
    return tomllib.loads(raw.decode("utf-8"))


def _file_spec(entry: dict) -> FileSpec:
    return FileSpec(name=entry["name"], sha256=entry["sha256"], size=entry["size"])


def remove_model(spec: ModelSpec) -> int:
    """Delete a model's downloaded weights. Returns the bytes reclaimed.

    Only the revision directory is removed, so a re-pin that introduced a new revision
    leaves the other one alone. An imported model is also dropped from the local
    catalog, since nothing could re-fetch it.
    """
    from localasr.registry import imported

    root = model_dir(spec)
    freed = 0
    if root.is_dir():
        for item in root.rglob("*"):
            # Symlinks are skipped on both counts. `rmtree` unlinks them rather than
            # following them, so the bytes on the other side are not reclaimed and
            # reporting them would claim a disk saving that did not happen — and the
            # file itself belongs to whatever tool the user linked it from.
            if item.is_file() and not item.is_symlink():
                freed += item.stat().st_size
        shutil.rmtree(root)
        parent = root.parent
        with contextlib.suppress(OSError):
            parent.rmdir()  # only succeeds once no other revision remains

    if imported.is_imported(spec.model_id):
        imported.forget(spec.model_id)
    return freed


def list_models() -> list[ModelSpec]:
    out = []
    for model_id, entry in _catalog()["models"].items():
        files = entry["files"]
        out.append(
            ModelSpec(
                model_id=model_id,
                runtime=entry["runtime"],
                repo=entry["repo"],
                revision=entry["revision"],
                ctx_size=entry.get("ctx_size", 4096),
                default=entry.get("default", False),
                model=_file_spec(files["model"]),
                mmproj=_file_spec(files["mmproj"]) if "mmproj" in files else None,
                measured_mb=entry.get("measured_mb", 0),
                kind=ModelKind(entry.get("kind", "asr")),
            )
        )
    from localasr.registry import imported

    return out + imported.list_imported()


def verified_builds(runtime: str) -> list[str]:
    """The llama.cpp builds this catalog's behaviour has been exercised against."""
    return list(_catalog().get("runtime", {}).get(runtime, {}).get("verified_builds", []))


def models_of_kind(kind: ModelKind) -> list[ModelSpec]:
    return [spec for spec in list_models() if spec.kind is kind]


def default_model_id(kind: ModelKind = ModelKind.ASR) -> str:
    """The catalog's default for one role. Roles have independent defaults: a machine
    always needs an ASR model, and may or may not have a refiner."""
    for spec in models_of_kind(kind):
        if spec.default:
            return spec.model_id
    raise RegistryError(f"catalog declares no default {kind.value} model")


def get_model(model_id: str | None = None, kind: ModelKind = ModelKind.ASR) -> ModelSpec:
    wanted = model_id or default_model_id(kind)
    for spec in list_models():
        if spec.model_id == wanted:
            return spec
    known = ", ".join(s.model_id for s in list_models())
    raise RegistryError(f"unknown model {wanted!r}; catalog has: {known}")


def model_dir(spec: ModelSpec) -> Path:
    """Revision-scoped directory, so an updated pin downloads beside the old one."""
    return models_dir() / spec.model_id / spec.revision


def local_paths(spec: ModelSpec) -> tuple[Path, Path | None]:
    root = model_dir(spec)
    return root / spec.model.name, (root / spec.mmproj.name if spec.mmproj else None)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, spec: FileSpec) -> None:
    """Raise IntegrityError unless `path` matches `spec` exactly."""
    actual_size = path.stat().st_size
    if actual_size != spec.size:
        raise IntegrityError(f"{path.name}: size {actual_size} != expected {spec.size}")
    actual_sha = sha256_of(path)
    if actual_sha != spec.sha256:
        raise IntegrityError(f"{path.name}: sha256 {actual_sha} != expected {spec.sha256}")


def is_downloaded(spec: ModelSpec) -> bool:
    """True when every file exists at the right size.

    Size only — hashing gigabytes on every CLI invocation is not worth it. The hash is
    checked when a file is downloaded and by `localasr models verify`.
    """
    root = model_dir(spec)
    return all(
        (root / f.name).is_file() and (root / f.name).stat().st_size == f.size for f in spec.files
    )


def missing_bytes(spec: ModelSpec) -> int:
    """Bytes still needed for the pinned model; wrong-sized files count in full."""
    root = model_dir(spec)
    return sum(
        file_spec.size
        for file_spec in spec.files
        if not (root / file_spec.name).is_file()
        or (root / file_spec.name).stat().st_size != file_spec.size
    )


def verify_model(spec: ModelSpec) -> None:
    """Full hash check of an already-downloaded model."""
    root = model_dir(spec)
    for file_spec in spec.files:
        path = root / file_spec.name
        if not path.is_file():
            raise IntegrityError(f"missing: {path}")
        verify(path, file_spec)


def _download(
    url: str,
    target: Path,
    spec: FileSpec,
    progress: DownloadProgress | None = None,
) -> None:
    """Fetch to a temporary sibling, verify, then rename into place.

    The partial file is removed on any failure so a retry starts clean and a corrupt
    download can never be mistaken for a complete one.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part")
    try:
        downloaded = 0
        if progress is not None:
            progress(spec.name, downloaded, spec.size)
        with urllib.request.urlopen(url) as response, tmp.open("wb") as fh:  # noqa: S310
            while chunk := response.read(_CHUNK):
                fh.write(chunk)
                downloaded += len(chunk)
                if progress is not None:
                    progress(spec.name, downloaded, spec.size)
        verify(tmp, spec)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.rename(target)


def pull_model(
    spec: ModelSpec,
    progress: DownloadProgress | None = None,
) -> tuple[Path, Path | None]:
    """Explicitly download missing pinned files, serialised within this process."""
    with _DOWNLOAD_LOCK:
        root = model_dir(spec)
        root.mkdir(parents=True, exist_ok=True)
        lock_path = models_dir() / ".download.lock"
        with lock_path.open("a+b") as lock_file:
            # A CLI pull and the desktop button can otherwise write the same .part.
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            base = f"{_HF_ENDPOINT}/{spec.repo}/resolve/{spec.revision}"

            for file_spec in spec.files:
                path = root / file_spec.name
                if path.is_file() and path.stat().st_size == file_spec.size:
                    if progress is not None:
                        progress(file_spec.name, file_spec.size, file_spec.size)
                    continue
                path.unlink(missing_ok=True)
                _download(f"{base}/{file_spec.name}", path, file_spec, progress)

    return local_paths(spec)


def vad_path() -> Path:
    """Path to the pinned Silero VAD graph, downloading it on first use (~2 MB)."""
    entry = _catalog()["vad"]
    spec = _file_spec(entry)
    target = models_dir() / "silero-vad" / entry["revision"] / spec.name
    if target.is_file() and target.stat().st_size == spec.size:
        return target
    target.unlink(missing_ok=True)
    _download(entry["url"].format(revision=entry["revision"]), target, spec)
    return target
