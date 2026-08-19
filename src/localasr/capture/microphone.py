"""Audio capture via PortAudio (sounddevice), which sits on PipeWire here.

The callback runs on PortAudio's realtime thread. It does exactly one thing: stamp the
buffer and put it on a bounded queue. Any VAD, HTTP or file work there would eventually
overrun the device and produce dropouts, so it all happens downstream.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from queue import Empty, Full, Queue

import numpy as np

from localasr.capture.resample import Resampler, pick_source_rate
from localasr.core.types import SAMPLE_RATE, AudioBlock, SourceTag

BLOCK_SECONDS = 0.1
QUEUE_BLOCKS = 100
"""Ten seconds of slack. Beyond that the consumer is not coming back and dropping is
the honest outcome — reported, never silent."""


class CaptureError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    index: int
    name: str
    channels: int
    default_samplerate: float
    is_monitor: bool
    # Display labels live in capture.naming: they need the system's device table, which
    # is a subprocess call, so they are built once per list rather than per device.


@dataclass(frozen=True, slots=True)
class InputLevel:
    rms: float
    peak: float
    overflowed: bool = False

    @property
    def dbfs(self) -> float:
        return 20.0 * math.log10(max(self.rms, 1e-9))


def _sounddevice():  # noqa: ANN202
    try:
        import sounddevice
    except (ImportError, OSError) as exc:  # OSError: PortAudio missing
        raise CaptureError(f"audio capture unavailable: {exc}") from exc
    return sounddevice


def list_devices() -> list[DeviceInfo]:
    """Input-capable devices. Monitors are how system audio is captured on PipeWire."""
    sd = _sounddevice()
    devices = []
    for index, info in enumerate(sd.query_devices()):
        if info["max_input_channels"] < 1:
            continue
        name = info["name"]
        devices.append(
            DeviceInfo(
                index=index,
                name=name,
                channels=info["max_input_channels"],
                default_samplerate=info["default_samplerate"],
                is_monitor="monitor" in name.lower(),
            )
        )
    return devices


def find_monitor() -> DeviceInfo | None:
    """The loopback device that carries what the speakers are playing."""
    for device in list_devices():
        if device.is_monitor:
            return device
    return None


def resolve_device(hint: str | int | None) -> int | None:
    """Accept an index, a substring of a device name, or None for the default."""
    if hint is None or hint == "":
        return None
    if isinstance(hint, int):
        return hint
    if hint.isdigit():
        return int(hint)
    needle = hint.lower()
    for device in list_devices():
        if needle in device.name.lower():
            return device.index
    raise CaptureError(f"no input device matching {hint!r}")


def open_rate(index: int | None) -> int:
    """The rate to open this device at, preferring one that needs no conversion.

    Raw ALSA `hw:` devices convert nothing: they accept their native rate and reject
    every other one. Asking a 48 kHz microphone for 16 kHz is refused outright, which
    made most of the machine's inputs look broken when they were merely being asked the
    wrong question.
    """
    sd = _sounddevice()
    info = sd.query_devices(index if index is not None else sd.default.device[0])

    def accepts(rate: int) -> bool:
        try:
            sd.check_input_settings(device=index, samplerate=rate, channels=1, dtype="float32")
        except Exception:  # noqa: BLE001 - PortAudio reports unsupported rates by raising
            return False
        return True

    try:
        return pick_source_rate(accepts, info["default_samplerate"])
    except ValueError as exc:
        raise CaptureError(f"cannot open input device {index!r}: {exc}") from exc


def probe_input(device: str | int | None, duration: float = 1.0) -> InputLevel:
    """Capture a short block and report its level without starting an ASR session."""
    if duration <= 0:
        raise ValueError("duration must be positive")
    sd = _sounddevice()
    index = resolve_device(device)
    rate = open_rate(index)
    frames = max(1, int(duration * rate))
    try:
        with sd.InputStream(
            samplerate=rate,
            blocksize=min(frames, int(BLOCK_SECONDS * rate)),
            device=index,
            channels=1,
            dtype="float32",
        ) as stream:
            samples, overflowed = stream.read(frames)
    except Exception as exc:  # noqa: BLE001 - PortAudio backend errors vary by platform
        raise CaptureError(f"cannot capture from device {device!r}: {exc}") from exc

    # Level is measured after conversion, so what the test reports is what the
    # recogniser will actually receive.
    mono = Resampler(rate)(np.asarray(samples, dtype=np.float32).reshape(-1), last=True)
    rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    return InputLevel(rms=rms, peak=peak, overflowed=bool(overflowed))


class MicrophoneSource:
    """Yields AudioBlocks until stopped. Iterate it on a capture thread.

    Bluetooth headsets switch to HSP/HFP once their microphone is opened, dropping the
    sample rate to 8-16 kHz and noticeably degrading recognition. `warning` reports that
    so a controller can tell the user rather than leaving them with poor output.
    """

    def __init__(
        self,
        device: str | int | None = None,
        source: SourceTag = "mic",
        *,
        block_seconds: float = BLOCK_SECONDS,
        epoch: float | None = None,
    ) -> None:
        self.device = device
        self.source = source
        self.block_samples = max(1, int(block_seconds * SAMPLE_RATE))
        self.epoch = epoch
        self.warning: str | None = None
        self.source_rate: int | None = None
        """Rate the device was opened at; differs from SAMPLE_RATE when converting."""
        self._queue: Queue[AudioBlock] = Queue(maxsize=QUEUE_BLOCKS)
        self._stop = threading.Event()
        self._sequence = 0
        self._overruns = 0
        self._stream = None

    @property
    def overruns(self) -> int:
        """Blocks the consumer never received."""
        return self._overruns

    def stop(self) -> None:
        self._stop.set()

    def __enter__(self) -> MicrophoneSource:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def open(self) -> None:
        sd = _sounddevice()
        index = resolve_device(self.device)
        rate = open_rate(index)
        # One resampler per stream: it carries this stream's filter tail between blocks.
        resampler = Resampler(rate)
        self.source_rate = rate

        if index is not None:
            info = sd.query_devices(index)
            if info["default_samplerate"] < SAMPLE_RATE:
                self.warning = (
                    f"{info['name']} runs at {info['default_samplerate']:.0f} Hz; "
                    "Bluetooth headsets drop to HSP/HFP when their mic opens, which "
                    "noticeably degrades recognition"
                )

        def callback(indata, frames, _time_info, status) -> None:  # noqa: ANN001
            if status:
                self._overruns += 1
            # A fixed-length FIR on 100 ms of audio, which is bounded and microseconds
            # long. The rule this callback obeys is no blocking work — no locks, no
            # network, no disk — not no arithmetic.
            block = AudioBlock(
                samples=resampler(indata[:, 0]),
                captured_at=time.monotonic(),
                sequence=self._sequence,
                source=self.source,
            )
            self._sequence += 1
            try:
                self._queue.put_nowait(block)
            except Full:
                self._overruns += 1

        try:
            self._stream = sd.InputStream(
                samplerate=rate,
                # Blocks are requested in the device's own samples so each one still
                # covers block_seconds of wall time after conversion.
                blocksize=max(1, round(self.block_samples * rate / SAMPLE_RATE)),
                device=index,
                channels=1,
                dtype="float32",
                callback=callback,
            )
            self._stream.start()
        except Exception as exc:  # noqa: BLE001 - PortAudio raises backend-specific errors
            # Callers handle CaptureError; a raw PortAudioError escaping here crashed
            # the capture thread instead of degrading the session.
            self._stream = None
            raise CaptureError(f"cannot open input device {self.device!r}: {exc}") from exc

    def close(self) -> None:
        self._stop.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def __iter__(self) -> Iterator[AudioBlock]:
        while not self._stop.is_set():
            try:
                yield self._queue.get(timeout=0.2)
            except Empty:
                continue
        # Hand over whatever was captured before the stop, so releasing a
        # push-to-talk key does not truncate the last word.
        while True:
            try:
                yield self._queue.get_nowait()
            except Empty:
                return


def blocks_from_file(path, source: SourceTag = "mic", block_seconds: float = 0.1):  # noqa: ANN001
    """Replay a file as timed AudioBlocks, for testing the live path without a device."""
    from localasr.core.audio.decode import iter_pcm

    epoch = time.monotonic()
    offset = 0.0
    for sequence, chunk in enumerate(iter_pcm(path, chunk_seconds=block_seconds)):
        yield AudioBlock(
            samples=chunk,
            captured_at=epoch + offset,
            sequence=sequence,
            source=source,
        )
        offset += len(chunk) / SAMPLE_RATE
