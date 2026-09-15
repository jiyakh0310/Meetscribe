"""Shared timeline foundation for SELaD-style meeting analytics.

Phase 0 only: a read-only, additive timestamped view built from the existing
reviewed transcript. See ``timeline.model`` for the full explanation of the
data model and its integration point. Nothing in this package is imported by
the factual MoM pipeline (STT, MiniLM, ANN, the deterministic formatter, or
Gemma), and it does not read or influence Voxels.
"""

from __future__ import annotations

from timeline.model import (
    Timeline,
    TimelineEvent,
    TimingStatus,
    build_timeline,
    timed_events,
)

__all__ = [
    "Timeline",
    "TimelineEvent",
    "TimingStatus",
    "build_timeline",
    "timed_events",
]
