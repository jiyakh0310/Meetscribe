"""SELaD Phase 4 -- FPS-aware frame sampling (Part 5).

SELaD sampled approximately every 60 frames (~2 seconds at 30 FPS).
MeetScribe reproduces the METHODOLOGICAL RATE (a target sampling
interval in seconds, derived from SELaD's own 60-frames-at-30fps
convention: 60 / 30 = 2.0 seconds), not a literal "skip 60 frames"
rule -- skipping a fixed frame COUNT independent of FPS would sample a
60fps recording twice as densely (every 1s) and a 24fps recording more
sparsely (every 2.5s) than SELaD's own methodology intended. Deriving
the frame-skip count from the video's actual FPS keeps the same target
~2-second cadence regardless of source frame rate (Part 5's explicit
requirement).
"""

from __future__ import annotations

from dataclasses import dataclass

TARGET_SAMPLE_INTERVAL_SECONDS = 2.0
"""SELaD's own methodological interval: 60 frames / 30 fps = 2.0 seconds."""

_FALLBACK_FPS_FOR_INTERVAL_ONLY = 30.0
"""Used ONLY to size the sampling interval when a video's own FPS is
unknown (see sample_frame_indices) -- never used to fabricate a
timestamp for an actual frame; timestamps always come from the real,
inspected FPS when available (Part 5: "timestamps must be derived from
actual video FPS")."""


@dataclass(frozen=True, slots=True)
class SampledFrame:
    frame_index: int
    timestamp_seconds: float


def sample_frame_indices(
    *,
    fps: float | None,
    frame_count: int | None,
    duration_seconds: float,
) -> list[SampledFrame]:
    """Return the deterministic list of (frame_index, timestamp) pairs to
    sample, at SELaD's target ~2-second cadence.

    If ``fps`` is unknown, falls back to sampling by TIME directly
    (every ``TARGET_SAMPLE_INTERVAL_SECONDS``) rather than skipping a
    fixed frame count blind to the real frame rate -- this still respects
    "timestamps must be derived from actual video FPS" by simply not
    depending on frame-index arithmetic when FPS is unavailable.
    """

    if duration_seconds <= 0:
        return []

    if fps is None or fps <= 0:
        # Time-based fallback: still deterministic, still ~2s cadence,
        # frame_index is left as a best-effort estimate under the
        # assumed fallback FPS purely for traceability -- callers must
        # not treat it as authoritative when fps is None.
        samples: list[SampledFrame] = []
        timestamp = 0.0
        while timestamp < duration_seconds:
            estimated_index = int(round(timestamp * _FALLBACK_FPS_FOR_INTERVAL_ONLY))
            samples.append(SampledFrame(frame_index=estimated_index, timestamp_seconds=timestamp))
            timestamp += TARGET_SAMPLE_INTERVAL_SECONDS
        return samples

    interval_frames = max(1, round(fps * TARGET_SAMPLE_INTERVAL_SECONDS))
    total_frames = frame_count if frame_count is not None else int(round(duration_seconds * fps))
    if total_frames <= 0:
        return []

    samples = []
    for frame_index in range(0, total_frames, interval_frames):
        timestamp_seconds = frame_index / fps
        samples.append(SampledFrame(frame_index=frame_index, timestamp_seconds=timestamp_seconds))
    return samples
