"""Pinning: a download that does not match the catalog must never become usable."""

import hashlib

import pytest

from localasr.registry import manager
from localasr.registry.manager import FileSpec, IntegrityError, verify


def _spec_for(data: bytes) -> FileSpec:
    return FileSpec(name="f.bin", sha256=hashlib.sha256(data).hexdigest(), size=len(data))


def test_verify_accepts_a_matching_file(tmp_path):
    data = b"weights"
    path = tmp_path / "f.bin"
    path.write_bytes(data)
    verify(path, _spec_for(data))


def test_verify_rejects_a_truncated_file(tmp_path):
    data = b"weights"
    path = tmp_path / "f.bin"
    path.write_bytes(data[:3])
    with pytest.raises(IntegrityError, match="size"):
        verify(path, _spec_for(data))


def test_verify_rejects_correct_size_but_wrong_content(tmp_path):
    """The case a size check alone would miss: a mirror serving a different build."""
    path = tmp_path / "f.bin"
    path.write_bytes(b"WEIGHTS")
    with pytest.raises(IntegrityError, match="sha256"):
        verify(path, _spec_for(b"weights"))


def test_failed_download_leaves_no_partial_file(tmp_path, monkeypatch):
    target = tmp_path / "model.gguf"

    class FakeResponse:
        """Serves content that hashes differently from what the catalog pins."""

        def __init__(self) -> None:
            self._chunks = iter([b"corrupt"])

        def read(self, _size):
            return next(self._chunks, b"")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(manager.urllib.request, "urlopen", lambda _url: FakeResponse())
    with pytest.raises(IntegrityError):
        manager._download("http://example.invalid/m", target, _spec_for(b"expected"))

    assert not target.exists()
    assert not target.with_name(target.name + ".part").exists()


def test_catalog_pins_every_file_of_every_shipped_model():
    """The pinning rule applies to the shipped catalog, which is what makes a download
    reproducible. Imported models are deliberately outside it: a file the user already
    had has no upstream commit, so its "revision" is a hash of its own bytes — the same
    guarantee, stated honestly rather than by borrowing a field that means something
    else. `list_models()` returns both, so this has to say which it is testing."""
    from localasr.registry import imported

    shipped = [s for s in manager.list_models() if not imported.is_imported(s.model_id)]
    assert shipped, "the shipped catalog cannot be empty"
    for spec in shipped:
        assert len(spec.revision) == 40, f"{spec.model_id} revision is not a full commit sha"
        for file_spec in spec.files:
            assert len(file_spec.sha256) == 64
            assert file_spec.size > 0


def test_an_imported_model_is_still_content_addressed():
    """Not pinned to a commit, but not unpinned either: the directory is named by the
    hash of the bytes, so a changed file is a different model rather than a silent
    substitution."""
    from localasr.registry import imported

    for spec in manager.list_models():
        if imported.is_imported(spec.model_id):
            assert len(spec.revision) == 16, f"{spec.model_id} is not content-addressed"
            for file_spec in spec.files:
                assert len(file_spec.sha256) == 64
                assert file_spec.size > 0


def test_each_role_declares_exactly_one_default():
    """Defaults are per role, not global: a machine always needs an ASR model and may
    additionally have a refiner, and neither may be ambiguous."""
    for kind in manager.ModelKind:
        specs = manager.models_of_kind(kind)
        assert sum(spec.default for spec in specs) == 1, f"{kind.value} has no single default"


def test_the_refiner_is_never_offered_as_a_transcription_model():
    """They are different roles on different servers. Loading a text model to transcribe
    audio fails at the capability check, which is a confusing way to learn this."""
    asr_ids = {spec.model_id for spec in manager.models_of_kind(manager.ModelKind.ASR)}
    llm_ids = {spec.model_id for spec in manager.models_of_kind(manager.ModelKind.LLM)}

    assert asr_ids and llm_ids
    assert not (asr_ids & llm_ids)
    assert manager.default_model_id(manager.ModelKind.ASR) in asr_ids


def test_model_paths_are_scoped_by_revision():
    spec = manager.get_model()
    model_path, _ = manager.local_paths(spec)
    assert spec.revision in str(model_path)
    assert spec.model_id in str(model_path)


def test_missing_bytes_counts_only_absent_or_incomplete_files(tmp_path, monkeypatch):
    spec = manager.get_model()
    monkeypatch.setenv("LOCALASR_DATA_DIR", str(tmp_path))
    root = manager.model_dir(spec)
    root.mkdir(parents=True)
    (root / spec.model.name).write_bytes(b"incomplete")
    if spec.mmproj is not None:
        (root / spec.mmproj.name).write_bytes(b"x" * spec.mmproj.size)

    assert manager.missing_bytes(spec) == spec.model.size


def test_catalog_pins_the_verified_runtime_build():
    builds = manager.verified_builds("llama-server")
    # Both machines are represented: the desktop's Vulkan release and the Jetson's
    # locally built CUDA one. A single pin would warn on every start of the other.
    assert len(builds) >= 2
    assert all(b.startswith("b") for b in builds)


def test_unknown_model_id_lists_the_alternatives():
    with pytest.raises(manager.RegistryError, match="qwen3-asr"):
        manager.get_model("does-not-exist")


def test_editable_checkout_reuses_its_models_without_shell_environment(tmp_path, monkeypatch):
    checkout = tmp_path / "LocalASR"
    module = checkout / "src" / "localasr" / "registry" / "manager.py"
    module.parent.mkdir(parents=True)
    module.touch()
    (checkout / "models").mkdir()
    (checkout / "pyproject.toml").touch()
    monkeypatch.delenv("LOCALASR_DATA_DIR", raising=False)
    monkeypatch.setattr(manager, "__file__", str(module))

    assert manager.data_dir() == checkout


# --- imported models ---------------------------------------------------------


def _gguf(path, payload: bytes = b"GGUF-weights"):
    path.write_bytes(payload)
    return path


def test_an_imported_model_is_copied_not_referenced(tmp_path, monkeypatch):
    """A model left wherever the user downloaded it disappears the moment they tidy
    up, and the engine would then fail at launch instead of at import."""
    from localasr.registry import imported

    monkeypatch.setenv("LOCALASR_DATA_DIR", str(tmp_path / "data"))
    source = _gguf(tmp_path / "mine.gguf")

    spec = imported.import_model(imported.ImportRequest(model_path=source))
    source.unlink()

    assert manager.is_downloaded(spec)
    assert (manager.model_dir(spec) / "mine.gguf").is_file()


def test_an_imported_model_is_content_addressed(tmp_path, monkeypatch):
    """No upstream revision exists to pin, so the imported bytes are the revision."""
    from localasr.registry import imported

    monkeypatch.setenv("LOCALASR_DATA_DIR", str(tmp_path / "data"))
    first = imported.import_model(
        imported.ImportRequest(model_path=_gguf(tmp_path / "a.gguf", b"one"), name="m")
    )
    second = imported.import_model(
        imported.ImportRequest(model_path=_gguf(tmp_path / "b.gguf", b"two"), name="m")
    )
    assert first.revision != second.revision


def test_imported_ids_cannot_collide_with_catalog_ids(tmp_path, monkeypatch):
    from localasr.registry import imported

    monkeypatch.setenv("LOCALASR_DATA_DIR", str(tmp_path / "data"))
    spec = imported.import_model(
        imported.ImportRequest(model_path=_gguf(tmp_path / "x.gguf"), name="qwen3-asr-1_7b-q8")
    )
    assert spec.model_id.startswith(imported.IMPORTED_PREFIX)
    assert spec.model_id not in {"qwen3-asr-1_7b-q8", "qwen3-asr-0_6b-q8"}


def test_imported_models_appear_alongside_catalog_models(tmp_path, monkeypatch):
    from localasr.registry import imported

    monkeypatch.setenv("LOCALASR_DATA_DIR", str(tmp_path / "data"))
    spec = imported.import_model(imported.ImportRequest(model_path=_gguf(tmp_path / "y.gguf")))
    assert spec.model_id in {m.model_id for m in manager.list_models()}
    assert manager.get_model(spec.model_id).revision == spec.revision


def test_a_non_gguf_file_is_refused(tmp_path, monkeypatch):
    from localasr.registry import imported

    monkeypatch.setenv("LOCALASR_DATA_DIR", str(tmp_path / "data"))
    wrong = tmp_path / "notes.txt"
    wrong.write_text("not a model")
    with pytest.raises(imported.ImportError_, match="gguf"):
        imported.import_model(imported.ImportRequest(model_path=wrong))


def test_a_missing_file_is_refused(tmp_path, monkeypatch):
    from localasr.registry import imported

    monkeypatch.setenv("LOCALASR_DATA_DIR", str(tmp_path / "data"))
    with pytest.raises(imported.ImportError_, match="不存在"):
        imported.import_model(imported.ImportRequest(model_path=tmp_path / "absent.gguf"))


def test_removing_a_model_reclaims_its_bytes(tmp_path, monkeypatch):
    from localasr.registry import imported

    monkeypatch.setenv("LOCALASR_DATA_DIR", str(tmp_path / "data"))
    spec = imported.import_model(
        imported.ImportRequest(model_path=_gguf(tmp_path / "z.gguf", b"x" * 5000))
    )
    freed = manager.remove_model(spec)

    assert freed == 5000
    assert not manager.model_dir(spec).exists()
    assert not manager.is_downloaded(spec)


def test_removing_an_imported_model_also_forgets_it(tmp_path, monkeypatch):
    """Nothing could re-fetch it, so leaving it listed would offer a model that can
    never be used again."""
    from localasr.registry import imported

    monkeypatch.setenv("LOCALASR_DATA_DIR", str(tmp_path / "data"))
    spec = imported.import_model(imported.ImportRequest(model_path=_gguf(tmp_path / "w.gguf")))
    manager.remove_model(spec)
    assert spec.model_id not in {m.model_id for m in manager.list_models()}


def test_removing_a_catalog_model_keeps_it_listed_for_re_download(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALASR_DATA_DIR", str(tmp_path / "data"))
    spec = manager.get_model("qwen3-asr-0_6b-q8")
    manager.remove_model(spec)
    assert spec.model_id in {m.model_id for m in manager.list_models()}
