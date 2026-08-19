"""Push-to-talk dictation.

    hotkey down -> open microphone -> StreamingSegmenter
    hotkey up   -> flush -> transcribe -> deliver text

Deliberately not streaming partials. The engine transcribes whole utterances, and with
the model already resident a short clip comes back fast enough that partial results
would add machinery without changing what the user experiences.

The global hotkey lives outside this process: on Wayland a client cannot grab keys, so
the desktop's own shortcut system invokes `localasr dictate`, which talks to this.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable
from dataclasses import dataclass

from localasr.apps.coordinator import Activity, ActivityConflict
from localasr.apps.events import (
    DictationStateChanged,
    Event,
    FinalTranscript,
    SessionFailed,
    TextEmitted,
)
from localasr.apps.live import LiveOptions, LiveSession
from localasr.apps.sources import SourceRequest, open_sources
from localasr.capture.microphone import CaptureError
from localasr.context import AppContext
from localasr.core.audio.vad import VadConfig
from localasr.core.types import Segment
from localasr.platform import text_output

Listener = Callable[[Event], None]

IDLE = "idle"
PREPARING = "preparing"
RECORDING = "recording"
TRANSCRIBING = "transcribing"
FAILED = "failed"


@dataclass(frozen=True, slots=True)
class DictationOptions:
    language: str | None = None
    device: str | int | None = None
    prefer_method: str | None = None
    join_separator: str = ""
    deliver: bool = True
    capture_system: bool = False
    """Also transcribe what the computer is playing.

    Off by default: dictation is for what *you* say, and pulling in whatever happens to
    be playing would inject a video's narration into your document. A meeting defaults
    the other way, because the other party arrives only through the system monitor.
    """
    system_device: str | int | None = None
    clipboard_fallback: Callable[[str], bool] | None = None
    """Last-resort place to put the text. A GUI supplies its own clipboard so a
    missing command-line tool never costs the user a finished transcription."""


class DictationController:
    """One press-to-talk cycle at a time. `start()`/`finish()` are the whole API."""

    def __init__(
        self,
        context: AppContext,
        options: DictationOptions | None = None,
        listener: Listener | None = None,
    ) -> None:
        self.context = context
        self.options = options or DictationOptions()
        self._listener = listener or context.bus.publish
        self._lock = threading.Lock()
        self._session: LiveSession | None = None
        self._sources: list = []
        self._token = None
        self._segments: list[Segment] = []
        self._engine_held = False
        self.state = IDLE

    @property
    def recording(self) -> bool:
        """Whether a capture session is open.

        Derived from the session, not from `state`. `state` is a display label that
        moves with progress; treating it as the lifecycle would make "am I recording?"
        answerable differently depending on whether an utterance happened to finish.
        """
        return self._session is not None

    @property
    def segments(self) -> tuple[Segment, ...]:
        """Utterances settled so far, in order."""
        return tuple(self._segments)

    @property
    def current_text(self) -> str:
        """Text settled so far, including utterances completed during recording."""
        segments = tuple(self._segments)
        return self.options.join_separator.join(
            segment.text for segment in segments if segment.text
        ).strip()

    def _set_state(self, state: str, detail: str | None = None) -> None:
        self.state = state
        self._listener(DictationStateChanged(state=state, detail=detail))

    def start(self, *, force: bool = False) -> None:
        """Load the engine if needed, then open the microphone and begin segmenting.

        Preparing the engine is part of starting, not something callers are trusted to
        remember: the CLI and the global hotkey call this directly, and without it the
        model loads lazily during the first transcription — that is, while the user is
        already talking, and a missing model surfaces mid-session instead of before it.

        Raises ActivityConflict when a subtitle job or meeting is running, so the caller
        can offer to stop it rather than leaving dictation queued behind a long job.
        """
        with self._lock:
            if self._session is not None:
                return  # already capturing; a second start would open a second stream
            self._token = self.context.coordinator.acquire(Activity.DICTATION, force=force)
            self._segments = []

            try:
                self._set_state(PREPARING, "正在检查模型并启动识别引擎…")
                # Held, not merely prepared: between sentences the engine looks idle
                # and would otherwise be unloaded while the microphone is still open.
                self.context.hold_engine()
                self._engine_held = True
            except Exception as exc:
                self.context.coordinator.release(self._token)
                self._token = None
                self._set_state(FAILED, str(exc))
                raise

            try:
                opened = open_sources(
                    SourceRequest(
                        mic_device=self.options.device or self.context.settings.input_device,
                        system_device=self.options.system_device
                        or self.context.settings.system_device,
                        capture_system=self.options.capture_system,
                    )
                )
            except CaptureError as exc:
                self._release_engine()
                self.context.coordinator.release(self._token)
                self._token = None
                self._set_state(FAILED, str(exc))
                raise

            for warning in opened.warnings:
                self._listener(DictationStateChanged(state=RECORDING, detail=warning))

            session = LiveSession(
                self.context.engine,
                self.context.new_vad,
                LiveOptions(language=self.options.language or self.context.settings.language),
                VadConfig.for_live(),
            )
            session.start(
                listener=self._on_event,
                on_utterance=self._segments.append,
                sources=opened.tags,
            )
            for source in opened.sources:
                session.feed_in_background(source, tag=source.source)

            self._sources = opened.sources
            self._session = session
            self._set_state(RECORDING)

    def finish(self, *, deliver: bool | None = None) -> str:
        """Stop recording and optionally deliver the settled text.

        ``deliver=None`` preserves the configured behavior. A visible GUI can pass
        ``False`` so the result remains in its preview instead of being typed into
        the LocalASR window that currently owns keyboard focus.
        """
        with self._lock:
            if self._session is None or not self._sources:
                return ""
            self._set_state(TRANSCRIBING)
            for source in self._sources:
                source.stop()
            try:
                self._session.stop()
            finally:
                for source in self._sources:
                    source.close()
                self._sources = []
                self._session = None
                self._release_engine()
                if self._token is not None:
                    self.context.coordinator.release(self._token)
                    self._token = None

            text = self.current_text

            if not text:
                self._set_state(IDLE, "nothing recognised")
                return ""

            should_deliver = self.options.deliver if deliver is None else deliver
            if should_deliver:
                try:
                    delivery = text_output.deliver(
                        text,
                        prefer=self.options.prefer_method,
                        fallback=self.options.clipboard_fallback,
                    )
                    self._listener(TextEmitted(text=text, method=delivery.method))
                    self._set_state(IDLE, delivery.detail)
                except text_output.TextOutputError as exc:
                    self._set_state(FAILED, str(exc))
                    return text
            else:
                self._set_state(IDLE)
            return text

    def cancel(self) -> None:
        """Discard the recording without transcribing or delivering anything."""
        with self._lock:
            for source in self._sources:
                source.stop()
                source.close()
            self._sources = []
            if self._session is not None:
                # Cancelling is a discard path; a worker that will not stop must not
                # prevent the controller from returning to idle.
                with contextlib.suppress(Exception):
                    self._session.cancel(timeout=5.0)
                self._session = None
            self._release_engine()
            if self._token is not None:
                self.context.coordinator.release(self._token)
                self._token = None
            self._segments = []
            self._set_state(IDLE, "cancelled")

    def _release_engine(self) -> None:
        if self._engine_held:
            self._engine_held = False
            self.context.release_engine()

    def toggle(self, *, force: bool = False) -> str | None:
        """One entry point for a single hotkey: start if idle, finish if recording."""
        if self.recording:
            return self.finish()
        try:
            self.start(force=force)
        except ActivityConflict:
            raise
        return None

    def _on_event(self, event: Event) -> None:
        self._listener(event)
        if isinstance(event, SessionFailed):
            self._set_state(FAILED, event.error)
        elif isinstance(event, FinalTranscript) and self.recording:
            # An utterance completing mid-recording is progress, not a transition.
            # Moving to TRANSCRIBING here made `recording` false while the microphone
            # was still open, so the toggle button offered to *start* again and the
            # next press opened a second capture stream on the same device.
            #
            # The detail carries the text settled so far, which is what a live view
            # renders: each sentence appears as it is recognised rather than being
            # withheld until the user stops.
            self._listener(DictationStateChanged(RECORDING, self.current_text))
