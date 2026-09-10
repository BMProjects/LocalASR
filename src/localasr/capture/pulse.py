"""System-audio capture through PulseAudio/PipeWire's `parec`.

PortAudio only exposes the host APIs it was built against — on this machine ALSA and
OSS — so `sounddevice.query_devices()` never lists a PipeWire monitor no matter how
many `pactl` reports. Searching it for one is guaranteed to fail, which is why system
audio needs a second capture path rather than a different device string.

`parec` writes raw PCM to stdout in whatever format we ask for, so this asks for
exactly what the pipeline wants and does no conversion of its own.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from collections.abc import Iterator

import numpy as np

from localasr.core.types import SAMPLE_RATE, AudioBlock, SourceTag

BLOCK_SECONDS = 0.1


class PulseError(RuntimeError):
    pass


def _pactl(*args: str, timeout: float = 5.0) -> str:
    exe = shutil.which("pactl")
    if exe is None:
        raise PulseError("pactl not found; install pulseaudio-utils")
    try:
        proc = subprocess.run([exe, *args], capture_output=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        raise PulseError(f"pactl {' '.join(args)} failed: {exc}") from exc
    if proc.returncode != 0:
        raise PulseError(f"pactl {' '.join(args)}: {proc.stderr.decode(errors='replace').strip()}")
    return proc.stdout.decode(errors="replace").strip()


def available() -> bool:
    return shutil.which("parec") is not None and shutil.which("pactl") is not None


def server_error() -> str | None:
    """Why the audio server cannot be reached, or None when it can.

    `default_monitor()` returns None for two unrelated reasons — the server refused the
    connection, or it answered and the default sink genuinely has no monitor — and a
    caller that only sees None will report the second when it means the first. This
    separates them, at the cost of one more `pactl` call on a path that has already
    failed.
    """
    try:
        _pactl("info")
    except PulseError as exc:
        return str(exc)
    return None


def list_monitors() -> list[str]:
    """Every monitor source PulseAudio/PipeWire knows about."""
    try:
        listing = _pactl("list", "short", "sources")
    except PulseError:
        return []
    return [
        fields[1]
        for line in listing.splitlines()
        if len(fields := line.split("\t")) > 1 and fields[1].endswith(".monitor")
    ]


def default_monitor() -> str | None:
    """The monitor of the sink currently playing audio.

    Picking the default sink's monitor matters: a laptop typically has several
    (speakers, three HDMI outputs, a Bluetooth headset), and only the active one
    carries the meeting.
    """
    try:
        sink = _pactl("get-default-sink")
    except PulseError:
        return None
    if not sink:
        return None
    monitor = f"{sink}.monitor"
    return monitor if monitor in list_monitors() else None


class PulseMonitorSource:
    """Yields AudioBlocks from a monitor source. Same interface as MicrophoneSource."""

    def __init__(
        self,
        device: str | None = None,
        source: SourceTag = "system",
        *,
        block_seconds: float = BLOCK_SECONDS,
    ) -> None:
        self.device = device
        self.source = source
        self.block_samples = max(1, int(block_seconds * SAMPLE_RATE))
        self.warning: str | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._stop = threading.Event()
        self._sequence = 0
        self._overruns = 0

    @property
    def overruns(self) -> int:
        return self._overruns

    def open(self) -> None:
        if not available():
            raise PulseError("parec not found; install pulseaudio-utils for system audio")

        monitor = self.device or default_monitor()
        if monitor is None:
            raise PulseError("no monitor source for the default sink")
        self.device = monitor

        cmd = [
            shutil.which("parec"),
            f"--device={monitor}",
            "--format=float32le",
            f"--rate={SAMPLE_RATE}",
            "--channels=1",
            "--latency-msec=100",
        ]
        self._process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._stop.clear()

    def stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        self._stop.set()
        if self._process is not None:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()
            if self._process.stdout:
                self._process.stdout.close()
            if self._process.stderr:
                self._process.stderr.close()
            self._process = None

    def __enter__(self) -> PulseMonitorSource:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __iter__(self) -> Iterator[AudioBlock]:
        if self._process is None or self._process.stdout is None:
            raise PulseError("source is not open")

        want = self.block_samples * 4  # float32
        stdout = self._process.stdout
        while not self._stop.is_set():
            data = stdout.read(want)
            if not data:
                break
            if len(data) % 4:
                # A short read at shutdown; drop the trailing partial sample rather
                # than reinterpreting the buffer.
                self._overruns += 1
                data = data[: len(data) - (len(data) % 4)]
            if not data:
                continue
            yield AudioBlock(
                samples=np.frombuffer(data, dtype=np.float32).copy(),
                captured_at=time.monotonic(),
                sequence=self._sequence,
                source=self.source,
            )
            self._sequence += 1


def open_system_source(device: str | None = None) -> tuple[object | None, str | None]:
    """Best available system-audio source, and a note when one had to be skipped.

    Order: a monitor PortAudio can see (rare, needs a Pulse-enabled build), then
    `parec`, then nothing. Returns (source, warning); the source is already open.
    """
    from localasr.capture.microphone import CaptureError, MicrophoneSource, find_monitor

    if device is None:
        try:
            portaudio_monitor = find_monitor()
        except CaptureError:
            portaudio_monitor = None
        if portaudio_monitor is not None:
            try:
                source = MicrophoneSource(device=portaudio_monitor.index, source="system")
                source.open()
                return source, None
            except CaptureError:
                pass  # fall through to parec

    try:
        source = PulseMonitorSource(device=device, source="system")
        source.open()
        return source, None
    except PulseError as exc:
        return None, f"system audio unavailable ({exc}); recording microphone only"
