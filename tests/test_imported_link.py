"""Importing a model by reference instead of by copy.

The weights being linked belong to another tool's store — several gigabytes the user
did not want duplicated. So the failure that matters here is not "the import did not
work"; it is "removing the import deleted somebody else's model". Most of these tests
exist for that one sentence.
"""

from __future__ import annotations

import pytest

from localasr.registry import imported, manager
from localasr.registry.imported import ImportRequest
from localasr.registry.manager import ModelKind


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):  # noqa: ANN001, ANN201
    monkeypatch.setenv("LOCALASR_DATA_DIR", str(tmp_path / "store"))
    return tmp_path


def _weights(tmp_path, name: str = "Qwen3.5-4B-UD-Q4_K_XL.gguf") -> object:  # noqa: ANN001
    """A stand-in for the file in the other tool's model store."""
    source = tmp_path / "elsewhere"
    source.mkdir(exist_ok=True)
    path = source / name
    path.write_bytes(b"GGUF" + b"\0" * 4096)
    return path


def test_a_linked_import_does_not_copy_the_bytes(tmp_path) -> None:  # noqa: ANN001
    original = _weights(tmp_path)
    spec = imported.import_model(ImportRequest(model_path=original, link=True))

    placed = manager.model_dir(spec) / spec.model.name
    assert placed.is_symlink()
    assert placed.resolve() == original.resolve()
    assert placed.read_bytes() == original.read_bytes(), "it must still be usable"


def test_removing_a_linked_import_leaves_the_original_alone(tmp_path) -> None:  # noqa: ANN001
    """The disaster case. `rmtree` unlinks symlinks rather than following them, and this
    test is what keeps it that way — a regression here silently deletes the weights of
    whatever tool the user linked from."""
    original = _weights(tmp_path)
    before = original.read_bytes()
    spec = imported.import_model(ImportRequest(model_path=original, link=True))

    manager.remove_model(spec)

    assert original.is_file(), "the other tool's model store was damaged"
    assert original.read_bytes() == before
    assert not manager.model_dir(spec).exists()


def test_removing_a_link_does_not_claim_disk_it_did_not_free(tmp_path) -> None:  # noqa: ANN001
    """`is_file()` follows symlinks, so the byte count used to report the full size of a
    file that is still exactly where it was."""
    original = _weights(tmp_path)
    spec = imported.import_model(ImportRequest(model_path=original, link=True))

    assert manager.remove_model(spec) == 0


def test_a_copied_import_still_copies(tmp_path) -> None:  # noqa: ANN001
    """The default is unchanged: linking is opt-in, because a copy is what survives the
    user tidying up their downloads."""
    original = _weights(tmp_path)
    spec = imported.import_model(ImportRequest(model_path=original, link=False))

    placed = manager.model_dir(spec) / spec.model.name
    assert placed.is_file() and not placed.is_symlink()
    assert manager.remove_model(spec) > 0, "a copy really does free disk"
    assert original.is_file()


def test_an_imported_refiner_is_registered_as_one(tmp_path) -> None:  # noqa: ANN001
    """Every import used to become an ASR model, so a refiner could be imported and then
    never selected as a refiner."""
    spec = imported.import_model(
        ImportRequest(model_path=_weights(tmp_path), kind=ModelKind.LLM, link=True)
    )
    assert spec.kind is ModelKind.LLM

    reloaded = {s.model_id: s for s in imported.list_imported()}[spec.model_id]
    assert reloaded.kind is ModelKind.LLM, "the role must survive a round trip through disk"


def test_an_import_without_a_kind_is_still_asr(tmp_path) -> None:  # noqa: ANN001
    spec = imported.import_model(ImportRequest(model_path=_weights(tmp_path)))
    assert spec.kind is ModelKind.ASR


def test_linking_over_a_real_file_is_refused(tmp_path) -> None:  # noqa: ANN001
    """Silently replacing it would delete weights the user imported by copy earlier."""
    original = _weights(tmp_path)
    imported.import_model(ImportRequest(model_path=original, link=False))

    with pytest.raises(imported.ImportError_, match="不是链接"):
        imported.import_model(ImportRequest(model_path=original, link=True))


def test_the_catalog_survives_a_boolean_field(tmp_path) -> None:  # noqa: ANN001
    """Regression: `str(True)` is not TOML.

    The reader treats an unparseable catalog as "nothing was ever imported", so one
    wrongly rendered boolean made every imported model on the machine disappear —
    without an error, because the parse failure is swallowed on purpose.
    """
    imported.import_model(ImportRequest(model_path=_weights(tmp_path), link=True))

    text = imported.catalog_path().read_text(encoding="utf-8")
    assert "linked = true" in text and "True" not in text

    assert imported.list_imported(), "the catalog read back empty"


def test_relinking_the_same_model_is_fine(tmp_path) -> None:  # noqa: ANN001
    original = _weights(tmp_path)
    imported.import_model(ImportRequest(model_path=original, link=True))
    spec = imported.import_model(ImportRequest(model_path=original, link=True))

    assert (manager.model_dir(spec) / spec.model.name).is_symlink()
