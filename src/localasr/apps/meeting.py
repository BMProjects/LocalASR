"""Meeting capture.

Two sources on one clock: the microphone is tagged 我, the system monitor 对方. For an
online meeting that is a free two-party split — but it is a *source* label, not speaker
diarization. Several people in the same room all arrive through the microphone and are
all tagged 我; the UI must say so.

System audio is optional by design. A missing monitor device degrades the session to
microphone-only rather than failing it: a recording of half the meeting beats no
recording.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from localasr.apps.coordinator import Activity
from localasr.apps.events import Event, SessionFailed
from localasr.apps.live import LiveOptions, LiveSession
from localasr.apps.sources import SourceRequest, open_sources
from localasr.capture.microphone import CaptureError, MicrophoneSource
from localasr.context import AppContext
from localasr.core.audio.vad import VadConfig
from localasr.core.types import Segment, Span, Transcript
from localasr.refine.prompts import TEMPLATE_REVISION
from localasr.refine.serde import refinement_from_dict, refinement_to_dict
from localasr.refine.types import RefinementResult

Listener = Callable[[Event], None]
SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class MeetingOptions:
    language: str | None = None
    mic_device: str | int | None = None
    system_device: str | int | None = None
    capture_system: bool = True
    keep_audio: bool = False
    check_overlap: bool = True
    """Probe whether the microphone can hear the speakers before recording both.

    Without headphones it can, and then every sentence is transcribed twice.
    """


@dataclass(slots=True)
class MeetingSession:
    """A running meeting. `segments` is appended to as transcription completes."""

    path: Path
    started_at: datetime
    sources: tuple[str, ...]
    segments: list[Segment] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    refinements: list[RefinementResult] = field(default_factory=list)

    def pending_refinement_ids(self) -> tuple[str, ...]:
        """Utterances transcribed but not yet refined — what a restart has to redo.

        Only refinement is redone. Re-transcribing an hour of meeting because the
        tidy-up was interrupted would be the expensive half of the work thrown away.
        """
        done = {sid for result in self.refinements for sid in result.source_segment_ids}
        return tuple(
            segment.utterance_id
            for segment in self.segments
            if segment.utterance_id and segment.utterance_id not in done
        )

    def transcript(self) -> Transcript:
        ordered = sorted(self.segments, key=lambda s: s.start)
        duration = ordered[-1].end if ordered else 0.0
        return Transcript(segments=ordered, duration=duration)


class MeetingController:
    """Start, pause, resume and stop a meeting recording."""

    def __init__(
        self,
        context: AppContext,
        options: MeetingOptions | None = None,
        listener: Listener | None = None,
    ) -> None:
        self.context = context
        self.options = options or MeetingOptions()
        self._listener = listener or context.bus.publish
        self._lock = threading.Lock()
        self._session: LiveSession | None = None
        self._sources: list[MicrophoneSource] = []
        self._token = None
        self._journal = None
        self._paused = False
        self._engine_held = False
        self.meeting: MeetingSession | None = None

    @property
    def running(self) -> bool:
        return self._session is not None

    @property
    def paused(self) -> bool:
        return self._paused

    def start(self, journal_path: Path) -> MeetingSession:
        """Begin recording, writing each finished utterance to `journal_path`."""
        with self._lock:
            if self._session is not None:
                raise RuntimeError("meeting already running")
            self._token = self.context.coordinator.acquire(Activity.MEETING)
            # Held for the whole meeting: between utterances the engine looks idle and
            # would be unloaded mid-recording, so the next speaker pays a model load.
            self.context.hold_engine()
            self._engine_held = True

            session = LiveSession(
                self.context.engine,
                self.context.new_vad,
                LiveOptions(language=self.options.language or self.context.settings.language),
                VadConfig.for_live(),
            )

            try:
                opened = open_sources(
                    SourceRequest(
                        mic_device=self.options.mic_device or self.context.settings.input_device,
                        system_device=self.options.system_device
                        or self.context.settings.system_device,
                        capture_system=self.options.capture_system,
                        check_overlap=self.options.check_overlap,
                    )
                )
            except CaptureError:
                self._release_engine()
                self.context.coordinator.release(self._token)
                self._token = None
                raise
            sources, warnings = opened.sources, opened.warnings

            journal_path.parent.mkdir(parents=True, exist_ok=True)
            self._journal = journal_path.open("a", encoding="utf-8")
            tags = tuple(source.source for source in sources)
            self._write(
                {
                    "schema": SCHEMA_VERSION,
                    "started_at": datetime.now().isoformat(timespec="seconds"),
                    "sources": list(tags),
                    "model_id": self.context.spec.model_id,
                    "model_revision": self.context.spec.revision,
                    # Recorded even when nothing is refined this run: an export has to
                    # be able to say which instructions produced the text it holds.
                    "template_revision": TEMPLATE_REVISION,
                }
            )

            meeting = MeetingSession(
                path=journal_path,
                started_at=datetime.now(),
                sources=tags,
                warnings=warnings,
            )
            session.start(listener=self._on_event, on_utterance=self._record, sources=tags)
            for source in sources:
                session.feed_in_background(source, tag=source.source)

            self._sources = sources
            self._session = session
            self._paused = False
            self.meeting = meeting
            return meeting



    def pause(self) -> None:
        """Stop consuming audio without ending the session or the journal."""
        with self._lock:
            for source in self._sources:
                source.stop()
            self._paused = True

    def resume(self) -> None:
        """Reopen each source that pause() closed.

        Rebuilt from `type(old)`, not from a fixed class: the system source may be a
        PulseMonitorSource, and recreating it as a MicrophoneSource would silently
        turn 对方 back into a second microphone stream.
        """
        with self._lock:
            if not self._paused or self._session is None:
                return
            resumed = []
            for old in self._sources:
                source = type(old)(device=old.device, source=old.source)
                source.open()
                self._session.feed_in_background(source, tag=source.source)
                resumed.append(source)
            self._sources = resumed
            self._paused = False

    def stop(self) -> MeetingSession | None:
        """End the meeting, draining anything still queued."""
        with self._lock:
            if self._session is None:
                return self.meeting
            for source in self._sources:
                source.stop()
            try:
                self._session.stop()
            finally:
                for source in self._sources:
                    source.close()
                self._sources = []
                self._session = None
                if self._journal is not None:
                    self._journal.close()
                    self._journal = None
                self._release_engine()
                if self._token is not None:
                    self.context.coordinator.release(self._token)
                    self._token = None
            return self.meeting

    def _release_engine(self) -> None:
        if self._engine_held:
            self._engine_held = False
            self.context.release_engine()

    def _record(self, segment: Segment) -> None:
        if self.meeting is not None:
            self.meeting.segments.append(segment)
        self._write(
            {
                "type": "segment",
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "source": segment.source,
                "text": segment.text,
                "utterance_id": segment.utterance_id,
            }
        )

    def record_refinement(self, result: RefinementResult) -> None:
        """Append one refinement, accepted or not.

        Rejected ones are written too: they are what stops a restart from retrying a
        refinement that has already been tried and refused, and what lets the rejection
        be explained instead of silently reappearing as unrefined text.
        """
        if self.meeting is not None:
            self.meeting.refinements.append(result)
        self._write(refinement_to_dict(result))

    def _write(self, payload: dict) -> None:
        if self._journal is None:
            return
        self._journal.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._journal.flush()

    def _on_event(self, event: Event) -> None:
        self._listener(event)
        if isinstance(event, SessionFailed) and self.meeting is not None:
            self.meeting.warnings.append(event.error)


def load_journal(path: Path) -> MeetingSession:
    """Rebuild a meeting from its journal, including one left by a crash."""
    lines = path.read_text(encoding="utf-8").splitlines()
    started = datetime.fromtimestamp(path.stat().st_mtime)
    sources: tuple[str, ...] = ()
    segments: list[Segment] = []
    refinements: list[RefinementResult] = []

    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except ValueError:
            continue  # a torn final line is expected after a crash
        if index == 0 and "schema" in row:
            sources = tuple(row.get("sources", ()))
            started = datetime.fromisoformat(row["started_at"]) if "started_at" in row else started
            continue
        if row.get("type") == "refinement":
            refinements.append(refinement_from_dict(row))
            continue
        if "text" not in row:
            continue
        segments.append(
            Segment(
                span=Span(row["start"], row["end"]),
                text=row["text"],
                source=row.get("source"),
                utterance_id=row.get("utterance_id"),
            )
        )

    return MeetingSession(
        path=path,
        started_at=started,
        sources=sources,
        segments=segments,
        refinements=refinements,
    )


def default_journal_path(root: Path | None = None) -> Path:
    base = root or (Path.home() / "Documents" / "localasr-meetings")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return base / f"meeting-{stamp}.jsonl"
