"""Base contracts for meeting transcript cleanup and future analysis."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class TranscriptCleanupInput:
    """Speaker-wise transcript text produced by the Phase 1 pipeline."""

    transcript_text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TranscriptCleanupResult:
    """Cleaned transcript while preserving speaker labels and timestamps."""

    cleaned_transcript: str
    original_transcript: str


@dataclass(frozen=True, slots=True)
class MeetingSummaryInput:
    """Cleaned speaker-wise transcript for meeting summary generation."""

    transcript_text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MeetingSummary:
    """High-level summary output for a meeting transcript."""

    title: str
    short_summary: str
    detailed_summary: str
    topics_discussed: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ActionItemInput:
    """Cleaned speaker-wise transcript for action item extraction."""

    transcript_text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ActionItem:
    """One task or follow-up extracted from a meeting transcript."""

    task: str
    owner: str | None
    due_date: str | None
    timestamp: str | None
    status: str


@dataclass(frozen=True, slots=True)
class ActionItemExtractionResult:
    """Structured action item extraction output."""

    action_items: list[ActionItem] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class DecisionInput:
    """Cleaned speaker-wise transcript for decision extraction."""

    transcript_text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Decision:
    """One decision extracted from a meeting transcript."""

    decision: str
    owner: str | None
    timestamp: str | None
    confidence: str


@dataclass(frozen=True, slots=True)
class DecisionExtractionResult:
    """Structured decision extraction output."""

    decisions: list[Decision] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class KeyDiscussionPointInput:
    """Cleaned speaker-wise transcript for key discussion point extraction."""

    transcript_text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class KeyDiscussionPoint:
    """One important discussion point extracted from a meeting transcript."""

    point: str
    speakers: list[str] = field(default_factory=list)
    timestamp: str | None = None


@dataclass(frozen=True, slots=True)
class KeyDiscussionPointExtractionResult:
    """Structured key discussion point extraction output."""

    key_discussion_points: list[KeyDiscussionPoint] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class MeetingAnalysisResult:
    """Complete end-to-end meeting analysis output."""

    cleaned_transcript: str
    summary: MeetingSummary
    key_discussion_points: list[KeyDiscussionPoint] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    action_items: list[ActionItem] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "cleaned_transcript": self.cleaned_transcript,
            "summary": {
                "title": self.summary.title,
                "short_summary": self.summary.short_summary,
                "detailed_summary": self.summary.detailed_summary,
                "topics_discussed": self.summary.topics_discussed,
            },
            "key_discussion_points": [
                {
                    "point": item.point,
                    "speakers": item.speakers,
                    "timestamp": item.timestamp,
                }
                for item in self.key_discussion_points
            ],
            "decisions": [
                {
                    "decision": item.decision,
                    "owner": item.owner,
                    "timestamp": item.timestamp,
                    "confidence": item.confidence,
                }
                for item in self.decisions
            ],
            "action_items": [
                {
                    "task": item.task,
                    "owner": item.owner,
                    "due_date": item.due_date,
                    "timestamp": item.timestamp,
                    "status": item.status,
                }
                for item in self.action_items
            ],
        }


class BaseSummarizer(ABC):
    """Provider-neutral interface for transcript cleanup and later analysis."""

    @abstractmethod
    def cleanup_transcript(
        self,
        cleanup_input: TranscriptCleanupInput | str,
    ) -> TranscriptCleanupResult:
        """Clean a Phase 1 speaker-wise transcript."""

    @abstractmethod
    def generate_meeting_summary(
        self,
        summary_input: MeetingSummaryInput | str,
    ) -> MeetingSummary:
        """Generate meeting title, summaries, and topics discussed."""

    @abstractmethod
    def extract_action_items(
        self,
        action_item_input: ActionItemInput | str,
    ) -> ActionItemExtractionResult:
        """Extract tasks, owners, due dates, timestamps, and statuses."""

    @abstractmethod
    def extract_decisions(
        self,
        decision_input: DecisionInput | str,
    ) -> DecisionExtractionResult:
        """Extract decisions, owners, timestamps, and confidence levels."""

    @abstractmethod
    def extract_key_discussion_points(
        self,
        key_point_input: KeyDiscussionPointInput | str,
    ) -> KeyDiscussionPointExtractionResult:
        """Extract important discussion points, speakers, and timestamps."""

    @abstractmethod
    def analyze_meeting(
        self,
        transcript_text: str,
    ) -> MeetingAnalysisResult:
        """Run the full cleanup, summary, key point, decision, and action pipeline."""
