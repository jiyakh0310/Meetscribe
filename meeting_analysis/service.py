"""Service boundary for future ML-based Minutes of Meeting generation.

Purpose:
    Provide a single integration point for the future Streamlit-to-ML migration
    while keeping the current application untouched in this phase.

Responsibilities:
    - Accept a structured meeting analysis request.
    - Run non-crashing validation.
    - Delegate to the ML MoM pipeline placeholder.
    - Return a structured result object.

Inputs:
    ``MeetingAnalysisRequest`` containing reviewed transcript text and optional
    meeting metadata.

Outputs:
    ``MeetingAnalysisResult`` containing placeholder MoM sections or validation
    messages.

Future Implementation Notes:
    This module is not imported by the current Streamlit UI yet. A later
    migration phase can connect it after exporter compatibility is confirmed.
"""

from meeting_analysis.schemas import MeetingAnalysisRequest, MeetingAnalysisResult
from meeting_analysis.validation import validate_analysis_request
from ml_mom.pipeline import generate_minutes_placeholder


class MeetingAnalysisService:
    """Facade for the future ML-based meeting analysis workflow."""

    def generate_minutes(
        self,
        request: MeetingAnalysisRequest,
    ) -> MeetingAnalysisResult:
        """Generate a placeholder Minutes of Meeting result.

        Args:
            request: Structured request containing transcript text and meeting
                metadata.

        Returns:
            A placeholder ``MeetingAnalysisResult``. Validation messages are
            returned in the result instead of being raised as exceptions.
        """

        validation_messages = validate_analysis_request(request)
        if any(message.is_blocking for message in validation_messages):
            return MeetingAnalysisResult(validation_messages=validation_messages)

        # The service delegates to the ML namespace so future implementation can
        # evolve behind a stable application-facing contract.
        return generate_minutes_placeholder(request)
