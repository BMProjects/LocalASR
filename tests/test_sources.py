"""Opening capture sources: which ones, and what happens when one cannot be opened."""

import pytest

from localasr.apps.sources import SourceRequest, open_sources
from localasr.capture import overlap
from localasr.capture.microphone import CaptureError


class FakeSource:
    def __init__(self, device=None, source="mic", **_kwargs):
        self.device = device
        self.source = source
        self.warning = None
        self.opened = False

    def open(self):
        self.opened = True

    def stop(self):
        pass

    def close(self):
        pass


@pytest.fixture(autouse=True)
def no_probe(monkeypatch):
    """The probe opens real devices; every test here says what it wants explicitly."""
    monkeypatch.setattr(
        overlap, "probe", lambda *_a, **_k: overlap.OverlapResult(False, 0.0, 0.0, "stub")
    )


@pytest.fixture
def fake_mic(monkeypatch):
    monkeypatch.setattr("localasr.apps.sources.MicrophoneSource", FakeSource)


def test_dictation_style_request_opens_only_the_microphone(fake_mic, monkeypatch):
    """Pulling in whatever is playing would inject a video's narration into a document."""
    monkeypatch.setattr(
        "localasr.capture.pulse.open_system_source",
        lambda _d: pytest.fail("system audio must not be opened when not requested"),
    )
    opened = open_sources(SourceRequest(capture_system=False))
    assert opened.tags == ("mic",)


def test_meeting_style_request_opens_both(fake_mic, monkeypatch):
    monkeypatch.setattr(
        "localasr.capture.pulse.open_system_source",
        lambda _d: (FakeSource(source="system"), None),
    )
    opened = open_sources(SourceRequest(capture_system=True))
    assert opened.tags == ("mic", "system")


def test_a_missing_monitor_degrades_to_microphone_only(fake_mic, monkeypatch):
    """Half a recording beats none."""
    monkeypatch.setattr(
        "localasr.capture.pulse.open_system_source",
        lambda _d: (None, "system audio unavailable; recording microphone only"),
    )
    opened = open_sources(SourceRequest(capture_system=True))
    assert opened.tags == ("mic",)
    assert any("microphone only" in w for w in opened.warnings)


def test_a_microphone_that_will_not_open_is_fatal(monkeypatch):
    class Broken(FakeSource):
        def open(self):
            raise CaptureError("device busy")

    monkeypatch.setattr("localasr.apps.sources.MicrophoneSource", Broken)
    with pytest.raises(CaptureError, match="microphone unavailable"):
        open_sources(SourceRequest())


def test_a_detected_overlap_drops_the_system_source(fake_mic, monkeypatch):
    """Recording both would transcribe every sentence twice."""
    monkeypatch.setattr(
        overlap,
        "probe",
        lambda *_a, **_k: overlap.OverlapResult(True, 0.8, 0.2, "麦克风听得见扬声器"),
    )
    monkeypatch.setattr(
        "localasr.capture.pulse.open_system_source",
        lambda _d: pytest.fail("system audio was opened despite the overlap"),
    )
    opened = open_sources(SourceRequest(capture_system=True))
    assert opened.tags == ("mic",)
    assert any("识别两遍" in w for w in opened.warnings)


def test_an_inconclusive_probe_leaves_both_sources_on(fake_mic, monkeypatch):
    """Never disable a source on a measurement that could not be made."""
    monkeypatch.setattr(
        overlap,
        "probe",
        lambda *_a, **_k: overlap.OverlapResult(False, 0.0, 0.0, "太安静", conclusive=False),
    )
    monkeypatch.setattr(
        "localasr.capture.pulse.open_system_source",
        lambda _d: (FakeSource(source="system"), None),
    )
    assert open_sources(SourceRequest(capture_system=True)).tags == ("mic", "system")


def test_the_overlap_check_can_be_skipped(fake_mic, monkeypatch):
    probed = []
    monkeypatch.setattr(overlap, "probe", lambda *_a, **_k: probed.append(1))
    monkeypatch.setattr(
        "localasr.capture.pulse.open_system_source",
        lambda _d: (FakeSource(source="system"), None),
    )
    open_sources(SourceRequest(capture_system=True, check_overlap=False))
    assert probed == []
