"""Python workflow orchestration for MeetScribe meeting analysis."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from functools import lru_cache
from typing import Any, Protocol

from agentic.nodes import (
    ActionItemAgent,
    DecisionAgent,
    DiscussionAgent,
    MetadataAgent,
    SummaryAgent,
    ValidationAgent,
)
from agentic.state import AgenticMeetingState, SpeakerMapping
from summarization.base_summarizer import (
    BaseSummarizer,
    MeetingAnalysisResult,
    MeetingSummary,
)

logger = logging.getLogger(__name__)


class MeetingAgent(Protocol):
    """Minimal interface implemented by all meeting analysis agents."""

    name: str

    def run(self, context: AgenticMeetingState) -> dict[str, object]:
        """Return this agent's output for the shared context."""


class MeetingAnalysisWorkflow:
    """Lightweight sequential orchestrator for meeting analysis agents."""

    def __init__(self, agents: Iterable[MeetingAgent]) -> None:
        self._agents = tuple(agents)

    def invoke(self, context: AgenticMeetingState) -> AgenticMeetingState:
        """Run each agent in order and return the merged context."""

        state: AgenticMeetingState = {
            **context,
            "validation_results": dict(context.get("validation_results", {})),
        }
        for agent in self._agents:
            try:
                update = agent.run(state)
            except Exception as exc:
                logger.exception(
                    "%s error captured by orchestrator; continuing workflow",
                    agent.name,
                )
                state["validation_results"] = {
                    **state.get("validation_results", {}),
                    agent.name: {
                        "status": "failed",
                        "error": str(exc),
                    },
                }
                continue

            state.update(update)

        state["analysis_result"] = _merge_analysis_result(state)
        state["meeting_metadata"] = {
            **state.get("meeting_metadata", {}),
            "evidence": state.get("evidence", {}),
            "confidence": state.get("confidence", {}),
        }
        return state


@lru_cache(maxsize=1)
def build_meeting_analysis_workflow() -> MeetingAnalysisWorkflow:
    """Build the sequential Python workflow used by the app."""

    return MeetingAnalysisWorkflow(
        agents=(
            MetadataAgent(),
            SummaryAgent(),
            DiscussionAgent(),
            DecisionAgent(),
            ActionItemAgent(),
            ValidationAgent(),
        )
    )


def run_meeting_analysis_workflow(
    transcript: str,
    *,
    summarizer: BaseSummarizer,
    speaker_mapping: SpeakerMapping | None = None,
    meeting_metadata: dict[str, Any] | None = None,
) -> MeetingAnalysisResult:
    """Run meeting analysis and return the existing result type."""

    initial_state: AgenticMeetingState = {
        "transcript": transcript,
        "speaker_mapping": speaker_mapping or {},
        "meeting_metadata": meeting_metadata or {},
        "cleaned_transcript": "",
        "summary": None,
        "discussion_points": [],
        "decisions": [],
        "action_items": [],
        "evidence": {},
        "confidence": {},
        "validation_results": {},
        "summarizer": summarizer,
    }

    final_state = build_meeting_analysis_workflow().invoke(initial_state)
    if meeting_metadata is not None:
        meeting_metadata.clear()
        meeting_metadata.update(final_state.get("meeting_metadata", {}))
    return _merge_analysis_result(final_state)


def _merge_analysis_result(state: AgenticMeetingState) -> MeetingAnalysisResult:
    """Merge available agent outputs into the public result contract."""

    previous_analysis = state.get("analysis_result")
    if not isinstance(previous_analysis, MeetingAnalysisResult):
        previous_analysis = None

    transcript = str(state.get("transcript", "")).strip()
    cleaned_transcript = (
        str(state.get("cleaned_transcript", "")).strip()
        or (previous_analysis.cleaned_transcript if previous_analysis else "")
        or transcript
    )
    summary = state.get("summary")
    if not isinstance(summary, MeetingSummary):
        summary = (
            previous_analysis.summary
            if previous_analysis is not None
            else _fallback_summary(transcript)
        )

    return MeetingAnalysisResult(
        cleaned_transcript=cleaned_transcript,
        summary=summary,
        key_discussion_points=_merged_list(
            state.get("discussion_points"),
            previous_analysis.key_discussion_points if previous_analysis else [],
        ),
        decisions=_merged_list(
            state.get("decisions"),
            previous_analysis.decisions if previous_analysis else [],
        ),
        action_items=_merged_list(
            state.get("action_items"),
            previous_analysis.action_items if previous_analysis else [],
        ),
    )


def _merged_list(value: object, fallback: list[Any]) -> list[Any]:
    if isinstance(value, list) and value:
        return list(value)
    return list(fallback)


def _fallback_summary(transcript: str) -> MeetingSummary:
    """Create a minimal summary when every summary path fails."""

    first_line = next(
        (line.strip() for line in transcript.splitlines() if line.strip()),
        "Meeting transcript",
    )
    short_summary = (
        "Meeting analysis could not be completed automatically. "
        "The original transcript is preserved for review."
    )
    return MeetingSummary(
        title=first_line[:80],
        short_summary=short_summary,
        detailed_summary=short_summary,
        topics_discussed=["Transcript review"],
    )
