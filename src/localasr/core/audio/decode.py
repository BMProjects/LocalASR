"""Decode any ffmpeg-readable media into the pipeline's canonical waveform.

Two entry points on purpose. `decode_file` materialises the whole waveform, which is
fine for a clip and wrong for a long recording: 16 kHz mono float32 is ~230 MB/hour, so
a two-hour meeting costs about 460 MB before anything else. `iter_pcm` streams the same
data in chunks so callers can segment as it arrives and only retain the current
utterance.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import wave
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from localasr.core.types import SAMPLE_RATE, Audio, Waveform

CHUNK_SECONDS = 5.0


class DecodeError(RuntimeError):
    pass


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise DecodeError("ffmpeg not found on PATH; install it to decode media files")
    return exe


def _command(path: Path, sample_rate: int) -> list[str]:
    return [
        _ffmpeg(),
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vn",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-",
    ]


def _check_readable(path: Path) -> Path:
    path = Path(path)
    if not path.is_file():
        raise DecodeError(f"no such file: {path}")
    return path


def decode_file(path: str | Path, sample_rate: int = SAMPLE_RATE) -> Audio:
    """Decode `path` entirely into memory as mono float32 PCM at `sample_rate`.

    Works for any container/codec ffmpeg supports, video included — video streams are
    simply not mapped. Prefer `iter_pcm` for recordings of unbounded length.
    """
    path = _check_readable(path)
    proc = subprocess.run(_command(path, sample_rate), capture_output=True)
    if proc.returncode != 0:
        detail = proc.stderr.decode(errors="replace").strip()
        raise DecodeError(f"ffmpeg failed on {path}: {detail}")

    samples: Waveform = np.frombuffer(proc.stdout, dtype=np.float32)
    if samples.size == 0:
        raise DecodeError(f"no audio stream decoded from {path}")
    return Audio(samples=samples, sample_rate=sample_rate)


@contextmanager
def _ffmpeg_pipe(path: Path, sample_rate: int) -> Iterator[subprocess.Popen[bytes]]:
    proc = subprocess.Popen(
        _command(path, sample_rate),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        yield proc
    finally:
        if proc.poll() is None:
            proc.kill()
        if proc.stdout:
            proc.stdout.close()
        proc.wait()


def iter_pcm(
    path: str | Path,
    sample_rate: int = SAMPLE_RATE,
    chunk_seconds: float = CHUNK_SECONDS,
) -> Iterator[Waveform]:
    """Yield successive mono float32 chunks without holding the whole file.

    A short final chunk is yielded as-is. Raises DecodeError if ffmpeg fails, including
    when it fails partway through.
    """
    path = _check_readable(path)
    # float32 is 4 bytes; round to a whole number of samples.
    chunk_bytes = max(4, int(chunk_seconds * sample_rate) * 4)
    produced = False

    with _ffmpeg_pipe(path, sample_rate) as proc:
        assert proc.stdout is not None
        pending = b""
        while True:
            data = proc.stdout.read(chunk_bytes)
            if not data:
                break
            pending += data
            usable = len(pending) - (len(pending) % 4)
            if usable:
                produced = True
                yield np.frombuffer(pending[:usable], dtype=np.float32)
                pending = pending[usable:]

        code = proc.wait()
        if code != 0:
            detail = (proc.stderr.read().decode(errors="replace").strip()) if proc.stderr else ""
            raise DecodeError(f"ffmpeg failed on {path}: {detail}")

    if not produced:
        raise DecodeError(f"no audio stream decoded from {path}")


def probe_duration(path: str | Path) -> float | None:
    """Container duration in seconds, or None when ffprobe cannot determine it."""
    exe = shutil.which("ffprobe")
    if exe is None:
        return None
    proc = subprocess.run(
        [
            exe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        capture_output=True,
    )
    try:
        return float(proc.stdout.decode().strip())
    except ValueError:
        return None


def to_wav_bytes(audio: Audio) -> bytes:
    """Serialise to a 16-bit PCM WAV container, the format the engine accepts."""
    pcm16 = np.clip(audio.samples, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(audio.sample_rate)
        writer.writeframes(pcm16.tobytes())
    return buf.getvalue()
