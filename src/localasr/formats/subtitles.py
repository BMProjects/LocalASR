"""Serialisation of a Transcript into the output formats.

Subtitle formats go through the cue composer; text and JSON formats stay faithful to
the transcript, since their consumers want the segments as recognised.
"""

from __future__ import annotations

import json

from localasr.core.types import Transcript
from localasr.formats.cues import Cue, CueStyle, compose


def _clock(seconds: float, millis_sep: str) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = round(seconds * 1000)
    ms = total_ms % 1000
    total_s = total_ms // 1000
    hours, rem = divmod(total_s, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{millis_sep}{ms:03d}"


def to_srt(transcript: Transcript, style: CueStyle | None = None) -> str:
    blocks = []
    for index, cue in enumerate(compose(transcript, style), start=1):
        blocks.append(f"{index}\n{_clock(cue.start, ',')} --> {_clock(cue.end, ',')}\n{cue.text}\n")
    return "\n".join(blocks)


def to_vtt(transcript: Transcript, style: CueStyle | None = None) -> str:
    blocks = ["WEBVTT\n"]
    for cue in compose(transcript, style):
        blocks.append(f"{_clock(cue.start, '.')} --> {_clock(cue.end, '.')}\n{cue.text}\n")
    return "\n".join(blocks)


def to_txt(transcript: Transcript) -> str:
    return transcript.text + "\n"


def to_json(transcript: Transcript) -> str:
    payload = {
        "duration": round(transcript.duration, 3),
        "language": transcript.language,
        "segments": [
            {
                "start": round(s.start, 3),
                "end": round(s.end, 3),
                "text": s.text,
                **({"source": s.source} if s.source else {}),
                **(
                    {"words": [{"text": w.text, "start": w.start, "end": w.end} for w in s.words]}
                    if s.words
                    else {}
                ),
            }
            for s in transcript.segments
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def to_markdown(transcript: Transcript) -> str:
    """Meeting-notes view: a timestamped, speaker-tagged running log."""
    labels = {"mic": "我", "system": "对方"}
    lines = ["# 会议记录", ""]
    for segment in transcript.segments:
        stamp = _clock(segment.start, ".")[:8]
        who = labels.get(segment.source or "", "")
        prefix = f"**{who}**  " if who else ""
        lines.append(f"- `{stamp}` {prefix}{segment.text}")
    return "\n".join(lines) + "\n"


WRITERS = {
    "srt": to_srt,
    "vtt": to_vtt,
    "txt": to_txt,
    "json": to_json,
    "md": to_markdown,
}


def render(transcript: Transcript, fmt: str, style: CueStyle | None = None) -> str:
    try:
        writer = WRITERS[fmt]
    except KeyError:
        raise ValueError(f"unknown format {fmt!r}; choose from {sorted(WRITERS)}") from None
    if fmt in {"srt", "vtt"}:
        return writer(transcript, style)
    return writer(transcript)


__all__ = ["Cue", "CueStyle", "WRITERS", "render", "to_srt", "to_vtt", "to_txt", "to_json"]
