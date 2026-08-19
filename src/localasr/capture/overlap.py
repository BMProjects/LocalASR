"""Detecting that two capture sources are recording the same sound.

A meeting records the microphone as 我 and the system monitor as 对方. That split is
free only when the microphone cannot hear the speakers. Without headphones it can, and
then every sentence is recognised twice — once per source — filling the transcript with
duplicates and spending twice the compute.

Whether that is happening is a property of the room, not of the configuration, so it is
measured rather than assumed: if the two envelopes track each other, the sources carry
the same sound.

Envelopes rather than raw waveforms, because the two paths differ in gain, in spectrum
and by a Bluetooth-sized delay — measured at +452 ms here — which destroys sample-level
correlation while leaving the loudness contour intact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from localasr.core.types import SAMPLE_RATE, Waveform

WINDOW = int(0.02 * SAMPLE_RATE)
CORRELATION_THRESHOLD = 0.6
"""Measured 0.795 for the same sound through two paths, 0.33 for unrelated audio."""

SILENCE_RMS = 0.002
"""Below this a source carries nothing to compare. Correlating two silences produces
a meaningless number, which is exactly how a first attempt at this reported
'different' while the duplication was happening."""


@dataclass(frozen=True, slots=True)
class OverlapResult:
    overlapping: bool
    correlation: float
    lag: float
    reason: str
    conclusive: bool = True
    """False when the comparison could not be made at all — too quiet, too short, or a
    source that would not open. Matching on the reason text instead let 「样本太短」 and
    「无法采集」 pass as trustworthy answers.
    """


def envelope(samples: Waveform, window: int = WINDOW) -> np.ndarray:
    """Normalised loudness contour: insensitive to gain, spectrum and delay."""
    power = np.convolve(samples.astype(np.float64) ** 2, np.ones(window) / window, mode="same")
    contour = np.sqrt(power)
    return (contour - contour.mean()) / (contour.std() + 1e-12)


def rms(samples: Waveform) -> float:
    if not len(samples):
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def compare(first: Waveform, second: Waveform) -> OverlapResult:
    """Decide whether two simultaneous recordings carry the same sound."""
    length = min(len(first), len(second))
    if length < WINDOW * 4:
        return OverlapResult(False, 0.0, 0.0, "样本太短，无法判断", conclusive=False)

    a, b = first[:length], second[:length]
    quiet = [name for name, x in (("麦克风", a), ("系统声音", b)) if rms(x) < SILENCE_RMS]
    if quiet:
        return OverlapResult(
            False, 0.0, 0.0, f"{'、'.join(quiet)}太安静，无法判断", conclusive=False
        )

    correlation = np.correlate(envelope(a), envelope(b), mode="full") / length
    peak = int(np.argmax(correlation))
    value = float(correlation[peak])
    lag = (peak - (length - 1)) / SAMPLE_RATE

    if value >= CORRELATION_THRESHOLD:
        return OverlapResult(
            True,
            value,
            lag,
            f"麦克风听得见扬声器（相关性 {value:.2f}，延迟 {lag * 1000:+.0f} ms）",
        )
    return OverlapResult(False, value, lag, f"两路内容不同（相关性 {value:.2f}）")


PROBE_SECONDS = 3.0
"""Measured: the same sound through both paths correlates at 0.52-0.61 over 1.5 s —
straddling the threshold — and settles at 0.68 from 3 s onward. The Bluetooth delay of
~0.4 s eats a large share of a short window's usable overlap."""


def probe(mic_device, system_device, seconds: float = PROBE_SECONDS) -> OverlapResult:  # noqa: ANN001
    """Record both sources briefly and compare them.

    Runs before the session opens its real streams — an ALSA capture device cannot be
    opened twice at once, and the samples compared are never audio the meeting would
    otherwise have transcribed.
    """
    import threading
    import time

    from localasr.capture.microphone import MicrophoneSource
    from localasr.capture.pulse import PulseMonitorSource

    captured: dict[str, Waveform] = {}

    def grab(name: str, source) -> None:  # noqa: ANN001
        # Best effort throughout: a probe that cannot run must leave the session to
        # decide for itself, never take the capture thread down with it.
        try:
            source.open()
        except Exception:  # noqa: BLE001
            return
        chunks = []
        deadline = time.monotonic() + seconds
        try:
            for block in source:
                chunks.append(block.samples)
                if time.monotonic() > deadline:
                    break
        except Exception:  # noqa: BLE001
            return
        finally:
            source.stop()
            source.close()
        if chunks:
            captured[name] = np.concatenate(chunks)

    threads = [
        threading.Thread(target=grab, args=("mic", MicrophoneSource(device=mic_device))),
        threading.Thread(
            target=grab, args=("system", PulseMonitorSource(device=system_device))
        ),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(seconds * 3)

    if "mic" not in captured or "system" not in captured:
        return OverlapResult(False, 0.0, 0.0, "有一路无法采集，无法判断", conclusive=False)
    return compare(captured["mic"], captured["system"])
