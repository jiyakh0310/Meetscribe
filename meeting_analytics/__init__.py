"""SELaD Phases 1-2: deterministic conversation-behaviour and content
analytics.

Consumes only ``timeline.Timeline`` (Phase 0). See ``meeting_analytics.model``
(Phase 1: speaking speed, turn-taking, talk-time proportion, questions) and
``meeting_analytics.content`` (Phase 2: topic grouping, keyword extraction,
positive-language proportion) for the metrics themselves. The dependency
direction is strictly one-way (Timeline -> meeting_analytics -> UI). This
package is not imported by, and does not import from, any part of the
factual MoM pipeline (STT, MiniLM's factual usage, the ANN, the
deterministic formatter, Gemma) or Voxels -- Phase 2 reuses only the shared
MiniLM model-loading/caching infrastructure (see
``meeting_analytics.content._embed_texts``), never formatter logic.
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
]
