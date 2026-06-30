"""Validation helpers for future ML-based meeting analysis.

Purpose:
    Keep user-safe validation separate from the ML pipeline so transcript
    structure problems never surface as uncaught exceptions.

Responsibilities:
    - Validate required transcript input.
    - Validate whether speaker information appears to be present.
    - Return friendly blocking messages instead of raising runtime errors.

Inputs:
    A ``MeetingAnalysisRequest`` instance.

Outputs:
    A list of ``ValidationMessage`` objects.

Future Implementation Notes:
    Speaker detection will later delegate to ``ml_mom.transcript_parser`` after
    that parser is fully implemented and tested against real transcript formats.
"""

from meeting_analysis.schemas import MeetingAnalysisRequest, ValidationMessage


def validate_analysis_request(
    request: MeetingAnalysisRequest,
) -> list[ValidationMessage]:
    """Validate a meeting analysis request without invoking ML.

    Args:
        request: Request containing transcript text and optional meeting data.

    Returns:
        A list of validation messages. An empty list means no blocking issue was
        found by the placeholder validation layer.
    """

    messages: list[ValidationMessage] = []

    # The service should fail gently when transcript text is absent because the
    # existing UI must remain responsible for presenting recoverable feedback.
    if not request.transcript_text.strip():
        messages.append(
            ValidationMessage(
                code="EMPTY_TRANSCRIPT",
                message="Transcript text is required before generating Minutes of Meeting.",
            )
        )

    # TODO:
    # Replace this lightweight placeholder with parser-backed speaker validation.
    return messages
