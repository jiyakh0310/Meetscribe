"""State contracts for meeting analysis orchestration."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict

from summarization.base_summarizer import (
    ActionItem,
    BaseSummarizer,
    Decision,
    KeyDiscussionPoint,
    MeetingAnalysisResult,
    MeetingSummary,
)

SpeakerMapping = dict[str, str]


class AgenticMeetingState(TypedDict):
    """Shared context passed between meeting analysis agents."""

    transcript: str
    speaker_mapping: SpeakerMapping
    meeting_metadata: dict[str, Any]
    extracted_metadata: NotRequired[dict[str, Any]]
    evidence: NotRequired[dict[str, Any]]
    confidence: NotRequired[dict[str, Any]]
    cleaned_transcript: NotRequired[str]
    summary: NotRequired[MeetingSummary | None]
    discussion_points: NotRequired[list[KeyDiscussionPoint]]
    decisions: NotRequired[list[Decision]]
    action_items: NotRequired[list[ActionItem]]
    validation_results: NotRequired[dict[str, Any]]
    analysis_result: NotRequired[MeetingAnalysisResult]
    summarizer: NotRequired[BaseSummarizer]
