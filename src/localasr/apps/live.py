"""Reusable live-transcription primitives.

Dictation and meeting capture are built on this: it turns one or more capture sources
into finalized utterances and transcribes them in order. The application controllers on
top decide what to do with the text.

Sources are iterators of `AudioBlock`, not devices, so the audio backend stays out of
the core and the whole path can be exercised from a file.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from queue import Empty, Full, Queue

from localasr.apps.events import (
    AudioDropped,
    Event,
    FinalTranscript,
    SegmentDropped,
    SessionCancelled,
    SessionFailed,
    SessionStarted,
    SessionStopped,
    SessionStopping,
    SpeechEnded,
    SpeechStarted,
)
from localasr.core.audio.stream import StreamingSegmenter, Utterance
from localasr.core.audio.vad import SileroVad, VadConfig
from localasr.core.engine.client import EngineError
from localasr.core.types import SAMPLE_RATE, AudioBlock, Segment, SourceTag

Listener = Callable[[Event], None]
AudioSource = Iterable[AudioBlock]

UTTERANCE_QUEUE_SIZE = 32
"""Bounded on purpose. If transcription cannot keep up, the backlog must become
visible as dropped audio rather than as unbounded memory growth."""

STOP_TIMEOUT = 120.0


class SessionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class LiveOptions:
    language: str | None = None
    drop_rejected: bool = True
    queue_size: int = UTTERANCE_QUEUE_SIZE


@dataclass(slots=True)
class _Queued:
    utterance: Utterance


class LiveSession:
    """Segments incoming audio into utterances and transcribes them in order.

    Transcription runs on a single worker so one engine instance is never asked to
    serve two requests at once — on a 4 GB card there is no second instance to spill
    onto. Capture threads only enqueue.

    Shutdown order matters and is enforced by `stop()`: capture threads are joined
    first, then each segmenter is flushed so speech still in progress is not lost, and
    only then is the worker told to finish. Putting the sentinel in first would let a
    still-running capture thread queue an utterance behind it, where it would never be
    read.
    """

    def __init__(
        self,
        engine,  # noqa: ANN001 - EngineManager, or anything with acquire/release
        vad_factory: Callable[[], SileroVad],
        options: LiveOptions | None = None,
        config: VadConfig | None = None,
    ) -> None:
        self.engine = engine
        self.options = options or LiveOptions()
        self._vad_factory = vad_factory
        self._config = config
        self._queue: Queue[_Queued | None] = Queue(maxsize=self.options.queue_size)
        self._worker: threading.Thread | None = None
        self._feeders: list[threading.Thread] = []
        self._listener: Listener = lambda _event: None
        self._on_utterance: Callable[[Segment], None] | None = None
        self._epoch: float | None = None
        self._cancelled = threading.Event()
        self._transcribed = 0
        self._lock = threading.Lock()

    @property
    def epoch(self) -> float | None:
        """The monotonic instant every span in this session is measured from."""
        return self._epoch

    def start(
        self,
        listener: Listener | None = None,
        on_utterance: Callable[[Segment], None] | None = None,
        sources: tuple[SourceTag, ...] = ("mic",),
    ) -> None:
        """Begin draining the utterance queue."""
        if self._worker is not None:
            raise SessionError("session already started")
        self._listener = listener or (lambda _event: None)
        self._on_utterance = on_utterance
        self._epoch = time.monotonic()
        self._cancelled.clear()
        self._transcribed = 0
        self._worker = threading.Thread(target=self._drain, name="localasr-asr", daemon=True)
        self._worker.start()
        self._listener(SessionStarted(sources=sources))

    def feed_in_background(self, source: AudioSource, tag: SourceTag = "mic") -> threading.Thread:
        """Run `feed` on its own thread and track it for shutdown."""
        thread = threading.Thread(
            target=self.feed, args=(source, tag), name=f"localasr-capture-{tag}", daemon=True
        )
        self._feeders.append(thread)
        thread.start()
        return thread

    def feed(self, source: AudioSource, tag: SourceTag | None = None) -> None:
        """Consume one capture source to exhaustion, enqueuing utterances.

        Runs on the caller's thread; a meeting runs one of these per source. All
        segmenters share the session epoch, so their spans are directly comparable.

        The blocks carry their own source tag, which is what a real capture device
        sets. `tag` overrides it, which is only needed when replaying a file as if it
        came from somewhere else.
        """
        segmenter = StreamingSegmenter(self._vad_factory(), self._config, epoch=self._epoch)
        speaking = False
        label: SourceTag = tag or "mic"

        overruns = 0
        for block in source:
            if self._cancelled.is_set():
                return
            # A device that overran dropped audio before we ever saw it; the block
            # stamps cannot reveal that, only the source's own counter can.
            current = getattr(source, "overruns", 0)
            if current > overruns:
                lost = current - overruns
                overruns = current
                self._listener(
                    AudioDropped(
                        at=segmenter.position,
                        seconds=lost * len(block.samples) / SAMPLE_RATE,
                        source=block.source,
                        reason=f"capture overran ({lost} block(s) never reached us)",
                    )
                )
            if self._epoch is not None and segmenter.epoch is None:
                segmenter.epoch = self._epoch
            if tag is not None:
                block = replace(block, source=tag)
            label = block.source

            utterances, gap = segmenter.push_block(block)
            if gap is not None:
                self._listener(
                    AudioDropped(
                        at=gap.at, seconds=gap.seconds, source=gap.source, reason=gap.reason
                    )
                )
            if segmenter.is_speaking and not speaking:
                speaking = True
                self._listener(SpeechStarted(at=segmenter.position, source=label))
            for utterance in utterances:
                speaking = segmenter.is_speaking
                self._publish(utterance)

        trailing = segmenter.flush()
        if trailing is not None:
            self._publish(trailing)

    def _publish(self, utterance: Utterance) -> None:
        self._listener(
            SpeechEnded(
                at=utterance.span.end,
                source=utterance.source,
                utterance_id=utterance.utterance_id,
            )
        )
        try:
            self._queue.put(_Queued(utterance), timeout=5.0)
        except Full:
            # Never silently discard: the timeline would shift with no way to tell.
            self._listener(
                AudioDropped(
                    at=utterance.span.start,
                    seconds=utterance.span.duration,
                    source=utterance.source,
                    reason="transcription backlog full",
                )
            )

    def stop(self, timeout: float = STOP_TIMEOUT) -> int:
        """Drain capture, flush pending speech, finish the queue, then shut down.

        Ordering matters. Capture threads are joined first so that every utterance —
        including the one `flush()` closes at the end of a stream — is already queued
        before the sentinel goes in; a sentinel queued first would leave anything behind
        it unread. Only then does the worker finish what it has.

        The caller stops the *device*: `stop()` waits for the sources to end, it does
        not interrupt them. Use `cancel()` to discard instead.

        Raises SessionError on timeout. A session that will not stop must be visible,
        not reported as stopped.
        """
        return self._shutdown(timeout=timeout, discard=False)

    def cancel(self, timeout: float = 15.0) -> int:
        """Stop immediately and discard whatever is still queued."""
        self._cancelled.set()
        return self._shutdown(timeout=timeout, discard=True)

    def _shutdown(self, *, timeout: float, discard: bool) -> int:
        if self._worker is None:
            return self._transcribed

        self._listener(SessionStopping())
        deadline = time.monotonic() + timeout

        for feeder in self._feeders:
            feeder.join(max(0.0, deadline - time.monotonic()))
            if feeder.is_alive():
                raise SessionError(f"capture thread {feeder.name} did not stop within {timeout}s")
        self._feeders.clear()

        if discard:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except Empty:
                    break

        self._queue.put(None)
        self._worker.join(max(1.0, deadline - time.monotonic()))
        if self._worker.is_alive():
            raise SessionError(f"transcription worker did not stop within {timeout}s")

        self._worker = None
        if discard:
            self._listener(SessionCancelled(reason="cancelled by caller"))
        else:
            self._listener(SessionStopped(utterances=self._transcribed))
        return self._transcribed

    def _drain(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.5)
            except Empty:
                continue
            if item is None:
                return
            self._handle(item)

    def _handle(self, item: _Queued) -> None:
        utterance = item.utterance
        try:
            client = self.engine.acquire()
            try:
                result = client.transcribe(utterance.audio, language=self.options.language)
            finally:
                self.engine.release()
        except EngineError as exc:
            self._listener(SessionFailed(str(exc)))
            return

        from localasr.core.postprocess.filters import rejection_reason

        reason = rejection_reason(result.text, utterance.span.duration)
        if reason and self.options.drop_rejected:
            # Say so. A sentence that vanishes with no explanation is indistinguishable
            # from audio the pipeline lost, which is how a filter becomes a bug report.
            self._listener(
                SegmentDropped(
                    index=self._transcribed + 1,
                    span=utterance.span,
                    text=result.text,
                    reason=reason,
                )
            )
            return

        segment = Segment(
            span=utterance.span,
            text=result.text,
            source=utterance.source,
            utterance_id=utterance.utterance_id,
        )
        with self._lock:
            self._transcribed += 1

        # Record before announcing. Subscribers decide whether an utterance is theirs by
        # looking it up in the controller's own record — dictation and meetings both
        # publish FinalTranscript, so a meeting window must ignore a forced dictation.
        # Announcing first makes that lookup a race the subscriber usually wins and
        # occasionally loses, silently dropping a line from the live view.
        if self._on_utterance is not None:
            self._on_utterance(segment)
        self._listener(FinalTranscript(segment))
