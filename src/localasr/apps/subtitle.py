"""Subtitle application controller.

Wraps the offline primitives into the thing a user actually asks for: point at one or
more files, get subtitle files out, be able to stop, and not lose completed work.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from localasr.apps.coordinator import Activity
from localasr.apps.events import Event, JobFailed
from localasr.apps.offline import JobCancelledError, OfflineJob, OfflineOptions
from localasr.context import AppContext
from localasr.core.types import Transcript
from localasr.formats.cues import CueStyle
from localasr.formats.subtitles import WRITERS, render

Listener = Callable[[Event], None]

MEDIA_SUFFIXES = {
    ".mp3",
    ".wav",
    ".flac",
    ".m4a",
    ".aac",
    ".ogg",
    ".opus",
    ".wma",
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".webm",
    ".ts",
    ".m4v",
}


@dataclass(frozen=True, slots=True)
class SubtitleOptions:
    fmt: str = "srt"
    language: str | None = None
    output_dir: Path | None = None
    overwrite: bool = False
    resume: bool = True
    style: CueStyle | None = None

    def validated(self) -> SubtitleOptions:
        if self.fmt not in WRITERS:
            raise ValueError(f"unknown format {self.fmt!r}; choose from {sorted(WRITERS)}")
        return self


@dataclass(frozen=True, slots=True)
class SubtitleResult:
    media: Path
    output: Path | None
    transcript: Transcript | None
    skipped: bool = False
    error: str | None = None


def expand_media(paths: Sequence[Path]) -> list[Path]:
    """Resolve files and directories into a sorted list of media files."""
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            found.extend(
                child
                for child in sorted(path.rglob("*"))
                if child.is_file() and child.suffix.lower() in MEDIA_SUFFIXES
            )
        elif path.is_file():
            found.append(path)
        else:
            raise FileNotFoundError(path)
    return found


class SubtitleController:
    """Runs subtitle jobs one at a time against the shared engine."""

    def __init__(self, context: AppContext, options: SubtitleOptions | None = None) -> None:
        self.context = context
        self.options = (options or SubtitleOptions()).validated()
        self._job: OfflineJob | None = None

    def cancel(self) -> None:
        """Stop the running job at its next segment boundary."""
        if self._job is not None:
            self._job.cancel()

    def output_for(self, media: Path) -> Path:
        directory = self.options.output_dir or media.parent
        return directory / f"{media.stem}.{self.options.fmt}"

    def journal_for(self, media: Path) -> Path:
        return self.output_for(media).with_suffix(f".{self.options.fmt}.jsonl")

    def run(
        self, media_paths: Sequence[Path], listener: Listener | None = None
    ) -> list[SubtitleResult]:
        """Transcribe each file. One failure does not abandon the rest of the batch."""
        emit = listener or (lambda _event: None)
        media_files = expand_media(list(media_paths))
        results: list[SubtitleResult] = []

        token = self.context.coordinator.acquire(Activity.SUBTITLE)
        try:
            for media in media_files:
                results.append(self._run_one(media, emit))
                if self._job is not None and self._job.cancelled:
                    break
        finally:
            self.context.coordinator.release(token)
            self._job = None
        return results

    def _run_one(self, media: Path, emit: Listener) -> SubtitleResult:
        output = self.output_for(media)
        if output.exists() and not self.options.overwrite:
            return SubtitleResult(media=media, output=output, transcript=None, skipped=True)

        job = OfflineJob(
            media,
            self.context.engine,
            self.context.new_vad(),
            OfflineOptions(language=self.options.language, resume=self.options.resume),
        )
        # Preserve a pending cancel across files in a batch.
        if self._job is not None and self._job.cancelled:
            job.cancel()
        self._job = job

        try:
            transcript = job.run(listener=emit, journal_path=self.journal_for(media))
        except JobCancelledError:
            return SubtitleResult(media=media, output=None, transcript=None, error="cancelled")
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the batch
            emit(JobFailed(str(exc), media=media))
            return SubtitleResult(media=media, output=None, transcript=None, error=str(exc))

        output.parent.mkdir(parents=True, exist_ok=True)
        rendered = render(transcript, self.options.fmt, self.options.style)
        output.write_text(rendered, encoding="utf-8")
        self.journal_for(media).unlink(missing_ok=True)
        return SubtitleResult(media=media, output=output, transcript=transcript)
