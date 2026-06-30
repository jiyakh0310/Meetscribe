"""Shared schemas for the future ML-based meeting analysis service.

Purpose:
    Define lightweight data contracts for passing transcript and meeting data
    into the ML Minutes of Meeting pipeline.

Responsibilities:
    - Represent service inputs in a predictable structure.
    - Represent placeholder analysis outputs without invoking real ML.
    - Represent validation messages that can be safely displayed by the UI.

Inputs:
    Reviewed transcript text, meeting metadata, and optional participant names.

Outputs:
    Structured result objects that future code can adapt to existing export
    expectations.

Future Implementation Notes:
    These schemas should remain stable while the underlying ML implementation
    evolves. Export-specific mapping should live in an adapter rather than in
    these schema classes.
"""

from dataclasses import dataclass, field


@dataclass(slots=True)
class ValidationMessage:
    """User-safe validation message produced before ML processing.

    Attributes:
        code: Stable machine-readable validation code.
        message: Friendly message that can be displayed to the user.
        is_blocking: Whether processing should stop until the issue is fixed.
    """

    code: str
    message: str
    is_blocking: bool = True


@dataclass(slots=True)
class MeetingAnalysisRequest:
    """Input contract for future ML-based Minutes of Meeting generation.

    Attributes:
        transcript_text: Reviewed transcript text to analyze.
        meeting_title: Optional meeting title from the existing UI.
        meeting_date: Optional meeting date from the existing UI.
        participants: Optional participant names collected or reviewed by users.
        metadata: Optional extra fields that should pass through the pipeline.
    """

    transcript_text: str
    meeting_title: str | None = None
    meeting_date: str | None = None
    participants: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class MeetingAnalysisResult:
    """Output contract for future ML-based Minutes of Meeting generation.

    Attributes:
        summary: Meeting summary bullets or paragraphs.
        discussion_points: Key discussion points grouped for export.
        decisions: Decisions detected from the transcript.
        action_items: Action items detected from the transcript.
        information_items: Informational statements worth preserving.
        validation_messages: Non-crashing validation messages.
        metadata: Optional pipeline metadata for diagnostics and traceability.
    """

    summary: list[str] = field(default_factory=list)
    discussion_points: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)
    information_items: list[str] = field(default_factory=list)
    validation_messages: list[ValidationMessage] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
