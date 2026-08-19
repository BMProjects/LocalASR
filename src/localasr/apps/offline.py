"""Offline transcription of a media file.

    decode -> VAD -> transcribe -> post-process -> Transcript

Two streaming passes over the file rather than one in-memory decode. The first pass
keeps only speech probabilities, so the upstream segmentation state machine still sees
the whole recording and can pick good boundaries; the second re-decodes and hands out
one span of audio at a time. Peak memory is the longest utterance, not the file.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from localasr.apps.events import (
    Event,
    JobCancelled,
    JobFinished,
    JobStarted,
    SegmentDropped,
    SegmentResumed,
    SegmentTranscribed,
)
from localasr.apps.journal import SCHEMA_VERSION, Journal, JournalHeader, fingerprint
from localasr.core.audio.decode import iter_pcm, probe_duration
from localasr.core.audio.stream import extract_spans, scan_probabilities
from localasr.core.audio.vad import SileroVad, segment_probabilities
from localasr.core.engine.manager import EngineManager
from localasr.core.postprocess.filters import rejection_reason
from localasr.core.types import Segment, Transcript

Listener = Callable[[Event], None]


class JobCancelledError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class OfflineOptions:
    language: str | None = None
    keep_rejected: bool = False
    resume: bool = True


class OfflineJob:
    """One media file, start to finish. Reusable across files via a shared manager."""

    def __init__(
        self,
        media: str | Path,
        engine: EngineManager,
        vad: SileroVad,
        options: OfflineOptions | None = None,
    ) -> None:
        self.media = Path(media)
        self.engine = engine
        self.vad = vad
        self.options = options or OfflineOptions()
        self._cancel = threading.Event()

    def cancel(self) -> None:
        """Ask the job to stop at the next segment boundary."""
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def run(self, listener: Listener | None = None, journal_path: Path | None = None) -> Transcript:
        """Transcribe the file, emitting events and optionally journalling progress.

        With `journal_path`, each finished segment is appended as it is produced and a
        later run reuses them instead of re-transcribing — provided the file, model and
        language are unchanged.
        """
        emit = listener or (lambda _event: None)

        journal = self._journal(journal_path)
        resumed = journal.resumable() if (journal and self.options.resume) else []

        spans = self._segment()
        duration = probe_duration(self.media)
        emit(JobStarted(total_segments=len(spans), duration=duration, media=self.media))

        segments: list[Segment] = list(resumed)
        for index, segment in enumerate(resumed, start=1):
            emit(SegmentResumed(index, segment))

        detected: str | None = None
        client = self.engine.acquire()
        if journal is not None:
            journal.open(resume=bool(resumed))
        try:
            stream = extract_spans(iter_pcm(self.media), spans)
            for index, (span, audio) in enumerate(stream, start=1):
                if index <= len(resumed):
                    continue
                if self._cancel.is_set():
                    emit(JobCancelled(completed=len(segments), media=self.media))
                    raise JobCancelledError(f"cancelled after {len(segments)} segment(s)")

                result = client.transcribe(audio, language=self.options.language)
                detected = detected or result.language
                segment = Segment(span=span, text=result.text, words=result.words)

                reason = rejection_reason(result.text, span.duration)
                if reason and not self.options.keep_rejected:
                    emit(SegmentDropped(index, span, result.text, reason))
                    continue

                segments.append(segment)
                emit(SegmentTranscribed(index, len(spans), segment))
                if journal is not None:
                    journal.append(segment)
        finally:
            if journal is not None:
                journal.close()
            self.engine.release()

        transcript = Transcript(
            segments=segments,
            duration=duration or (spans[-1].end if spans else 0.0),
            language=self.options.language or detected,
        )
        emit(JobFinished(transcript, media=self.media))
        return transcript

    def _journal(self, path: Path | None) -> Journal | None:
        if path is None:
            return None
        spec = self.engine.spec
        header = JournalHeader(
            schema=SCHEMA_VERSION,
            source=str(self.media.resolve()),
            source_fingerprint=fingerprint(self.media),
            model_id=spec.model_id,
            model_revision=spec.revision,
            language=self.options.language,
        )
        return Journal(path, header)

    def _segment(self) -> list:
        """First pass: probabilities only, then the offline segmentation state machine."""
        total_samples = 0

        def counted() -> Iterator:
            nonlocal total_samples
            for chunk in iter_pcm(self.media):
                total_samples += len(chunk)
                yield chunk

        probs = scan_probabilities(counted(), self.vad)
        if total_samples == 0:
            return []
        return segment_probabilities(probs, total_samples, self.vad.config)
