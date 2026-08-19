from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_config(tmp_path_factory, monkeypatch):
    """Point every test at a throwaway settings file.

    `switch_model` persists the user's preferences, so a test that exercises it — even
    indirectly, through a panel action — silently rewrote the real
    ~/.config/localasr/config.toml. Model weights still come from the ambient
    LOCALASR_DATA_DIR; only the preferences are isolated.
    """
    config = tmp_path_factory.mktemp("config") / "config.toml"
    monkeypatch.setenv("LOCALASR_CONFIG", str(config))
    return config

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


@pytest.fixture(scope="session")
def english_speech() -> Path:
    """11 s of speech with two clear pauses, so it segments into several utterances."""
    path = SAMPLES / "jfk.wav"
    if not path.is_file():
        pytest.skip(f"missing sample: {path}")
    return path


@pytest.fixture(scope="session")
def chinese_speech() -> Path:
    """5.6 s of continuous Mandarin, i.e. a single utterance."""
    path = SAMPLES / "zh_test.wav"
    if not path.is_file():
        pytest.skip(f"missing sample: {path}")
    return path
