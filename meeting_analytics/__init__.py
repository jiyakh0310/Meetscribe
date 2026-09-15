"""SELaD Phase 1: deterministic conversation-behaviour analytics.

Consumes only ``timeline.Timeline`` (Phase 0). See ``meeting_analytics.model``
for the full explanation of the metrics and the one-way dependency
direction (Timeline -> meeting_analytics -> UI). This package is not
imported by, and does not import from, any part of the factual MoM
pipeline (STT, MiniLM, the ANN, the deterministic formatter, Gemma) or
Voxels.
"""

from __future__ import annotations

from meeting_analytics.model import (
    ConversationAnalytics,
    SpeakerAnalytics,
    analyze_conversation,
)

__all__ = [
    "ConversationAnalytics",
    "SpeakerAnalytics",
    "analyze_conversation",
]
