"""Bringing capture devices to the 16 kHz the rest of the pipeline requires.

Raw ALSA `hw:` devices convert nothing: they accept their native rate and refuse
everything else. Most of them run at 48 kHz, so demanding 16 kHz from the device makes
them unopenable — the device is fine, the request is not. The fix is to open at a rate
the device supports and convert here.

The conversion itself is libsoxr's, via `soxr.ResampleStream`. Two properties matter
and both are the reason not to hand-roll this:

* It is a real band-limited resampler. Taking every third sample of a 48 kHz stream
  folds everything between 8 and 24 kHz back into the speech band, and that aliasing is
  not recoverable downstream — it lands as noise inside exactly the range the recogniser
  reads.
* It is **stateful across chunks**. Capture arrives in 100 ms pieces; a resampler
  restarted on each piece puts a discontinuity at every boundary, which is an audible
  click ten times a second. `ResampleStream` carries the filter tail over, and it
  compensates its own delay, so a chunked stream is sample-for-sample identical to
  converting the whole recording at once.
"""

from __future__ import annotations

import numpy as np

from localasr.core.types import SAMPLE_RATE, Waveform

QUALITY = "HQ"
"""libsoxr's default preset: passband to ~0.913 of Nyquist, stopband below -100 dB.
Well past what a laptop microphone and a 16-bit path can carry."""


class Resampler:
    """Converts a capture stream to the pipeline's rate, chunk by chunk.

    One instance belongs to one stream: it holds that stream's filter state. Feeding two
    streams through the same instance interleaves their history.
    """

    def __init__(self, source_rate: int, target_rate: int = SAMPLE_RATE) -> None:
        if source_rate <= 0 or target_rate <= 0:
            raise ValueError("sample rates must be positive")
        self.source_rate = source_rate
        self.target_rate = target_rate
        self._stream = None
        self.reset()

    @property
    def passthrough(self) -> bool:
        """True when the device already runs at the target rate and no work is needed."""
        return self.source_rate == self.target_rate

    def reset(self) -> None:
        """Forget the previous stream. Call before reusing an instance on new audio.

        soxr is imported here rather than at module scope. This module sits on the
        import path from the CLI down to `capture.microphone`, so a top-level import
        made `localasr models pull` require the resampler — and it failed on the compute
        node, which has no audio stack and no reason to need one to download a file.
        """
        import soxr

        if self.passthrough:
            self._stream = None
            return
        self._stream = soxr.ResampleStream(
            self.source_rate, self.target_rate, 1, dtype="float32", quality=QUALITY
        )

    def __call__(self, block: Waveform, *, last: bool = False) -> Waveform:
        """Convert one chunk, continuing from where the previous chunk ended.

        Output length varies by a sample or two between chunks — the ratio is rarely an
        integer, so the boundary lands mid-sample. Over a stream it comes out exact.

        Pass `last=True` on the final chunk to flush the filter tail; a live capture
        that just stops does not need it, since the samples still inside the filter are
        the trailing millisecond of a stream that has already been cut off.
        """
        samples = np.asarray(block, dtype=np.float32).reshape(-1)
        if self.passthrough:
            return samples.copy()
        assert self._stream is not None
        return self._stream.resample_chunk(samples, last=last).reshape(-1)


def pick_source_rate(supported: object, preferred: float) -> int:
    """Choose the rate to open a device at, preferring one that needs no conversion.

    `supported` is a predicate taking a rate and reporting whether the device accepts
    it, so this stays testable without a sound card.
    """
    candidates = [SAMPLE_RATE, int(round(preferred)), 48_000, 44_100, 32_000, 22_050, 8_000]
    seen: set[int] = set()
    for rate in candidates:
        if rate <= 0 or rate in seen:
            continue
        seen.add(rate)
        if supported(rate):  # type: ignore[operator]
            return rate
    raise ValueError("device accepts none of the usual capture rates")
