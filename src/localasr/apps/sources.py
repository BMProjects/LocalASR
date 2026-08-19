"""Opening the capture sources a live session will read from.

Dictation and meetings differ in what they default to, not in how the devices are
opened, so the opening lives here once. Getting it wrong is expensive in the same way
for both: an ALSA capture device cannot be opened twice at once, and a microphone that
can hear the speakers turns one sentence into two transcripts.

Defaults differ because the tasks differ. A meeting wants the other party, who arrives
only through the system monitor. Dictation wants what *you* say — pulling in whatever
happens to be playing would inject a video's narration into your document — so its
system source is opt-in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from localasr.capture import overlap
from localasr.capture.microphone import CaptureError, MicrophoneSource


@dataclass(frozen=True, slots=True)
class SourceRequest:
    mic_device: str | int | None = None
    system_device: str | int | None = None
    capture_system: bool = False
    check_overlap: bool = True
    """Probe whether the microphone can already hear the speakers.

    Only meaningful when both sources are wanted; recording both then transcribes
    every sentence twice.
    """


@dataclass(slots=True)
class OpenedSources:
    sources: list = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def tags(self) -> tuple[str, ...]:
        return tuple(source.source for source in self.sources)


def open_sources(request: SourceRequest) -> OpenedSources:
    """Open the microphone, and the system monitor when it adds something.

    Raises CaptureError only if the microphone itself cannot be opened — system audio
    is best-effort, because half a recording beats none.
    """
    opened = OpenedSources()

    # Probed before anything is opened for real: an ALSA capture device cannot be
    # opened twice at once, so a probe running alongside a live microphone fails with
    # "Device unavailable".
    duplicate = None
    if request.capture_system and request.check_overlap:
        result = overlap.probe(request.mic_device, request.system_device)
        duplicate = result if result.conclusive else None

    mic = MicrophoneSource(device=request.mic_device, source="mic")
    try:
        mic.open()
    except CaptureError as exc:
        raise CaptureError(f"microphone unavailable: {exc}") from exc
    if mic.warning:
        opened.warnings.append(mic.warning)
    opened.sources.append(mic)

    if not request.capture_system:
        return opened

    if duplicate is not None and duplicate.overlapping:
        opened.warnings.append(
            f"{duplicate.reason}；已只录麦克风，避免同一句话被识别两遍。"
            "戴上耳机后可以恢复双音源。"
        )
        return opened

    system, warning = _open_system(request.system_device)
    if system is not None:
        opened.sources.append(system)
    if warning:
        opened.warnings.append(warning)
    return opened


def _open_system(device):  # noqa: ANN001, ANN202 - either capture source type
    """System audio, best effort.

    Tried in order: a monitor PortAudio can see, then `parec`. The second is not a
    fallback for exotic setups — PortAudio exposes only the host APIs it was built
    against (ALSA and OSS here), so on an ordinary PipeWire desktop it never lists a
    monitor at all and `parec` is the normal path.
    """
    from localasr.capture.pulse import open_system_source

    return open_system_source(device)
