"""The model panel's rules, tested as the pure function they are.

One button now covers download, selection and loading, so what matters is that its
label always states what pressing it will do, and that its `intent` matches the label.
Every case names the mistake it prevents.
"""

import pytest

from localasr.frontends.desktop.model_state import DiskState, Intent, ModelFacts, plan


def facts(**overrides) -> ModelFacts:
    base = {
        "model_id": "qwen3-asr-1_7b-q8",
        "disk": DiskState.PRESENT,
        "is_current": False,
        "is_loaded": False,
        "is_imported": False,
        "workload_running": None,
        "operation_running": False,
    }
    return ModelFacts(**{**base, **overrides})


# --- the four things the one button can mean --------------------------------


def test_a_missing_model_offers_to_download_and_use_it():
    """Download has no button of its own, so the label has to say it will download."""
    action = plan(facts(disk=DiskState.ABSENT)).primary
    assert action.enabled
    assert action.text == "下载并使用"
    assert action.intent is Intent.DOWNLOAD_AND_USE


def test_an_interrupted_download_offers_to_continue():
    action = plan(facts(disk=DiskState.PARTIAL)).primary
    assert action.text == "继续下载并使用"
    assert action.intent is Intent.DOWNLOAD_AND_USE


def test_a_downloaded_spare_model_offers_to_be_used():
    action = plan(facts()).primary
    assert action.enabled
    assert action.text == "使用此模型"
    assert action.intent is Intent.USE


def test_the_current_unloaded_model_offers_to_be_loaded():
    action = plan(facts(is_current=True)).primary
    assert action.text == "加载到显存"
    assert action.intent is Intent.LOAD


def test_the_current_loaded_model_offers_release():
    action = plan(facts(is_current=True, is_loaded=True)).primary
    assert action.text == "释放显存"
    assert action.intent is Intent.RELEASE


def test_selecting_a_model_also_loads_it():
    """Merged on purpose: choosing a model you then have to load separately was two
    steps for one intention."""
    assert "使用" in plan(facts()).primary.text
    assert plan(facts()).primary.intent is Intent.USE


# --- imported models ---------------------------------------------------------


def test_an_imported_model_with_missing_files_cannot_be_repaired_by_downloading():
    """It came from a one-off copy; there is no upstream to re-fetch from."""
    action = plan(facts(disk=DiskState.ABSENT, is_imported=True)).primary
    assert not action.enabled
    assert action.intent is Intent.NONE
    assert "重新导入" in action.tip


def test_a_complete_imported_model_behaves_like_any_other():
    action = plan(facts(is_imported=True)).primary
    assert action.enabled
    assert action.intent is Intent.USE


# --- gating ------------------------------------------------------------------


@pytest.mark.parametrize(
    "state",
    [
        {"disk": DiskState.ABSENT},
        {},
        {"is_current": True},
        {"is_current": True, "is_loaded": True},
    ],
)
def test_a_running_workload_blocks_the_action_whatever_it_would_do(state):
    """Switching or releasing mid-job would pull the engine out from under it."""
    action = plan(facts(**state, workload_running="dictation")).primary
    assert not action.enabled
    assert "dictation" in action.tip


def test_an_in_flight_file_operation_blocks_the_action_and_the_import():
    state = plan(facts(operation_running=True))
    assert not state.primary.enabled
    assert not state.import_model.enabled
    assert not state.chooser_enabled


def test_the_chooser_is_locked_only_while_files_are_being_written():
    """Browsing the list during recognition is harmless; changing files is not."""
    assert plan(facts(workload_running="dictation")).chooser_enabled
    assert not plan(facts(operation_running=True)).chooser_enabled


def test_importing_stays_available_during_recognition():
    assert plan(facts(workload_running="meeting")).import_model.enabled


def test_every_disabled_action_explains_itself():
    """A greyed-out button with no reason is what made the old panel confusing."""
    for state in (
        plan(facts(disk=DiskState.ABSENT, is_imported=True)),
        plan(facts(workload_running="meeting")),
        plan(facts(operation_running=True)),
    ):
        for name in ("primary", "import_model"):
            action = getattr(state, name)
            if not action.enabled:
                assert action.tip, f"{name} is disabled with no explanation"


def test_an_enabled_action_always_has_something_to_do():
    for state in (
        facts(disk=DiskState.ABSENT),
        facts(disk=DiskState.PARTIAL),
        facts(),
        facts(is_current=True),
        facts(is_current=True, is_loaded=True),
    ):
        action = plan(state).primary
        assert action.enabled
        assert action.intent is not Intent.NONE


def test_the_summary_names_all_three_axes():
    text = plan(facts(is_current=True, is_loaded=True)).summary
    assert "已下载" in text and "已加载到显存" in text and "当前模型" in text


def test_the_panel_no_longer_exposes_a_delete_action():
    """Deleting weights is rare and destructive; it lives in `localasr models remove`."""
    state = plan(facts())
    assert not hasattr(state, "delete")
    assert not hasattr(state, "download")
