"""Rendering application events as terminal output.

Shared by the three CLI entry points so they report the same things the same way,
without any of them owning the event vocabulary.
"""

from __future__ import annotations

from collections.abc import Callable

import typer

from localasr.apps.events import (
    AudioDropped,
    Event,
    FinalTranscript,
    JobCancelled,
    JobFailed,
    JobStarted,
    SegmentDropped,
    SegmentResumed,
    SegmentTranscribed,
    SessionFailed,
    SessionStarted,
    SessionStopped,
    SpeechStarted,
)


def offline_reporter(quiet: bool = False) -> Callable[[Event], None]:
    def report(event: Event) -> None:
        if quiet:
            return
        if isinstance(event, JobStarted):
            length = f"{event.duration:.1f}s" if event.duration else "unknown length"
            name = event.media.name if event.media else ""
            typer.echo(f"{name}: {event.total_segments} segment(s), {length}", err=True)
        elif isinstance(event, SegmentResumed):
            typer.echo(f"[resumed {event.index}] {event.segment.text}", err=True)
        elif isinstance(event, SegmentTranscribed):
            typer.echo(
                f"[{event.index}/{event.total}] {event.segment.start:7.2f}s  {event.segment.text}",
                err=True,
            )
        elif isinstance(event, SegmentDropped):
            typer.echo(f"  dropped @{event.span.start:.2f}s ({event.reason})", err=True)
        elif isinstance(event, JobCancelled):
            typer.secho(f"cancelled after {event.completed} segment(s)", fg="yellow", err=True)
        elif isinstance(event, JobFailed):
            typer.secho(f"failed: {event.error}", fg="red", err=True)

    return report


def live_reporter(quiet: bool = False, labels: dict[str, str] | None = None):  # noqa: ANN201
    labels = labels or {"mic": "我", "system": "对方"}

    def report(event: Event) -> None:
        if quiet:
            return
        if isinstance(event, SessionStarted):
            typer.echo(f"listening: {', '.join(event.sources)}", err=True)
        elif isinstance(event, SpeechStarted):
            typer.echo(f"  ● {labels.get(event.source, event.source)}", err=True, nl=False)
        elif isinstance(event, FinalTranscript):
            who = labels.get(event.segment.source or "", "")
            prefix = f"{who} " if who else ""
            typer.echo(f"\r{event.segment.start:8.1f}s {prefix}{event.segment.text}")
        elif isinstance(event, AudioDropped):
            typer.secho(
                f"  audio dropped: {event.seconds:.2f}s from {event.source} ({event.reason})",
                fg="yellow",
                err=True,
            )
        elif isinstance(event, SessionFailed):
            typer.secho(f"  error: {event.error}", fg="red", err=True)
        elif isinstance(event, SessionStopped):
            typer.echo(f"stopped after {event.utterances} utterance(s)", err=True)

    return report
