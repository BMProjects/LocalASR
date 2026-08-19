"""Showing the device names the rest of the desktop shows.

PortAudio reports ALSA PCMs: `sof-hda-dsp: - (hw:0,6)`. Nothing in the system settings
calls it that, so picking the right microphone from that list is guesswork. PipeWire
knows the same device as "Raptor Lake-P/U/H cAVS Digital Microphone", which is what the
user sees everywhere else.

The two are joined by the ALSA card and device numbers PipeWire records on each node,
so the PortAudio name is parsed for `hw:card,device` and looked up.

Devices PipeWire does not expose — raw PCMs like `hw:0,7` on this machine — keep their
PortAudio name. Inventing a friendly label for a device the system does not list would
be worse than showing the honest one.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from localasr.capture.microphone import DeviceInfo

_HW = re.compile(r"hw:(\d+),(\d+)")
_NAME = re.compile(r"^\s*Name:\s*(\S+)", re.MULTILINE)


def _pactl_sources() -> str:
    exe = shutil.which("pactl")
    if exe is None:
        return ""
    try:
        proc = subprocess.run([exe, "list", "sources"], capture_output=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout.decode(errors="replace")


def system_descriptions() -> dict[tuple[int, int], str]:
    """Map (alsa card, alsa device) to the description the system shows.

    Only capture sources are considered; a monitor carries the same card/device numbers
    as its sink and would otherwise shadow the real microphone on that PCM.
    """
    text = _pactl_sources()
    if not text:
        return {}

    mapping: dict[tuple[int, int], str] = {}
    blocks = re.split(r"\n(?=Source #)", text)
    for block in blocks:
        name = _NAME.search(block)
        if name is None or name.group(1).endswith(".monitor"):
            continue
        description = re.search(r"^\s*Description:\s*(.+)$", block, re.MULTILINE)
        card = re.search(r'alsa\.card = "(\d+)"', block)
        device = re.search(r'alsa\.device = "(\d+)"', block)
        if description and card and device:
            mapping[(int(card.group(1)), int(device.group(1)))] = description.group(1).strip()
    return mapping


def friendly_name(portaudio_name: str, descriptions: dict[tuple[int, int], str]) -> str | None:
    """The system's name for this PortAudio device, or None when it has none."""
    match = _HW.search(portaudio_name)
    if match is None:
        return None
    return descriptions.get((int(match.group(1)), int(match.group(2))))


def describe(portaudio_name: str, descriptions: dict[tuple[int, int], str]) -> str:
    """Label to show: the system name when known, otherwise the honest PortAudio one."""
    friendly = friendly_name(portaudio_name, descriptions)
    if friendly is None:
        return portaudio_name
    hw = _HW.search(portaudio_name)
    return f"{friendly} ({hw.group(0)})" if hw else friendly


def device_labels(devices: Iterable[DeviceInfo]) -> list[str]:
    """Labels for a device list, querying the system name table once for all of them.

    Channel count and sample rate stay on the label — they are what distinguishes two
    endpoints of the same physical device, and they are what makes a Bluetooth headset
    that has dropped to 16 kHz HSP visible before a recording is wasted on it.
    """
    descriptions = system_descriptions()
    labels = []
    for device in devices:
        rate = f"{device.default_samplerate / 1000:g} kHz"
        name = describe(device.name, descriptions)
        labels.append(f"{name} · {device.channels} ch · {rate}")
    return labels
