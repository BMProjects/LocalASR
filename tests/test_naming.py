"""Device labels: the system's name for a device, joined to PortAudio's PCM name."""

from __future__ import annotations

import pytest

from localasr.capture import naming
from localasr.capture.microphone import DeviceInfo

# Trimmed to the fields the parser reads, in the layout `pactl list sources` emits.
PACTL_OUTPUT = """Source #52
	State: SUSPENDED
	Name: alsa_input.pci-0000_00_1f.3.HiFi__Mic1__source
	Description: Raptor Lake-P/U/H cAVS Digital Microphone
	Properties:
		alsa.card = "0"
		alsa.device = "6"

Source #53
	State: SUSPENDED
	Name: alsa_input.pci-0000_00_1f.3.HiFi__Mic2__source
	Description: Raptor Lake-P/U/H cAVS Stereo Microphone
	Properties:
		alsa.card = "0"
		alsa.device = "0"

Source #61
	State: RUNNING
	Name: alsa_output.pci-0000_00_1f.3.HiFi__Speaker__sink.monitor
	Description: Raptor Lake-P/U/H cAVS Speaker Monitor
	Properties:
		alsa.card = "0"
		alsa.device = "0"
"""


@pytest.fixture
def descriptions(monkeypatch: pytest.MonkeyPatch) -> dict[tuple[int, int], str]:
    monkeypatch.setattr(naming, "_pactl_sources", lambda: PACTL_OUTPUT)
    return naming.system_descriptions()


def test_system_descriptions_maps_card_and_device(descriptions) -> None:
    assert descriptions[(0, 6)] == "Raptor Lake-P/U/H cAVS Digital Microphone"
    assert descriptions[(0, 0)] == "Raptor Lake-P/U/H cAVS Stereo Microphone"


def test_monitor_does_not_shadow_the_microphone_on_the_same_pcm(descriptions) -> None:
    # The speaker monitor carries card 0 / device 0, the same numbers as Mic2. Taking it
    # would rename the stereo microphone to "Speaker Monitor".
    assert "Monitor" not in descriptions[(0, 0)]


def test_describe_uses_system_name_when_the_pcm_is_known(descriptions) -> None:
    label = naming.describe("sof-hda-dsp: - (hw:0,6)", descriptions)
    assert label == "Raptor Lake-P/U/H cAVS Digital Microphone (hw:0,6)"


def test_describe_keeps_portaudio_name_when_system_does_not_expose_the_pcm(
    descriptions,
) -> None:
    # hw:0,7 is a raw PCM PipeWire does not publish. Inventing a name would be worse.
    assert naming.describe("sof-hda-dsp: - (hw:0,7)", descriptions) == "sof-hda-dsp: - (hw:0,7)"


def test_describe_survives_a_name_without_hw_coordinates(descriptions) -> None:
    assert naming.describe("pulse", descriptions) == "pulse"


def test_no_pactl_leaves_every_device_under_its_portaudio_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(naming, "_pactl_sources", lambda: "")
    empty = naming.system_descriptions()
    assert empty == {}
    assert naming.describe("sof-hda-dsp: - (hw:0,6)", empty) == "sof-hda-dsp: - (hw:0,6)"


def test_device_labels_keep_channels_and_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(naming, "_pactl_sources", lambda: PACTL_OUTPUT)
    devices = [
        DeviceInfo(6, "sof-hda-dsp: - (hw:0,6)", 4, 48000.0, is_monitor=False),
        DeviceInfo(9, "JBL Flip 4 (hw:1,0)", 1, 16000.0, is_monitor=False),
    ]
    labels = naming.device_labels(devices)
    assert labels[0] == "Raptor Lake-P/U/H cAVS Digital Microphone (hw:0,6) · 4 ch · 48 kHz"
    # Unknown to the system table, but the 16 kHz that reveals an HSP-profile headset
    # still has to show.
    assert labels[1] == "JBL Flip 4 (hw:1,0) · 1 ch · 16 kHz"


def test_pactl_failure_is_not_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*_args: object, **_kwargs: object) -> None:
        raise OSError("pactl went away")

    monkeypatch.setattr(naming.subprocess, "run", explode)
    monkeypatch.setattr(naming.shutil, "which", lambda _: "/usr/bin/pactl")
    assert naming.system_descriptions() == {}
