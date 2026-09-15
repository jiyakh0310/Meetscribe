"""SELaD Phases 1-3: deterministic conversation-behaviour, content, and
audio communication analytics.

Consumes only ``timeline.Timeline`` (Phase 0) and, for Phase 3, the
already-computed Voxels result. See ``meeting_analytics.model`` (Phase 1:
speaking speed, turn-taking, talk-time proportion, questions),
``meeting_analytics.content`` (Phase 2: topic grouping, keyword extraction,
positive-language proportion), and ``meeting_analytics.communication``
(Phase 3: topic-aligned Voxels communication insights) for the metrics
themselves. The dependency direction is strictly one-way (Timeline/Voxels
result -> meeting_analytics -> UI). This package is not imported by, and
does not import from, any part of the factual MoM pipeline (STT, MiniLM's
factual usage, the ANN, the deterministic formatter, Gemma). Phase 2 reuses
only the shared MiniLM model-loading/caching infrastructure (see
``meeting_analytics.content._embed_texts``), never formatter logic. Phase 3
never runs or modifies Voxels inference -- it only reads the result Voxels
already produced.
"""

from __future__ import annotations

from meeting_analytics.model import (
    ConversationAnalytics,
    SpeakerAnalytics,
    analyze_conversation,
)
from meeting_analytics.content import (
    ContentAnalytics,
    KeywordAnalytics,
    PositiveLanguageAnalytics,
    PositiveLanguageSpeaker,
    SpeakerKeywords,
    TopicAnalytics,
    analyze_content,
)
from meeting_analytics.communication import (
    CommunicationInsights,
    TopicCommunicationInsight,
    align_topics_with_voxels,
)

__all__ = [
    "ConversationAnalytics",
    "SpeakerAnalytics",
    "analyze_conversation",
    "ContentAnalytics",
    "KeywordAnalytics",
    "PositiveLanguageAnalytics",
    "PositiveLanguageSpeaker",
    "SpeakerKeywords",
    "TopicAnalytics",
    "analyze_content",
    "CommunicationInsights",
    "TopicCommunicationInsight",
    "align_topics_with_voxels",
]
