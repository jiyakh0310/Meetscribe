"""Public interface for the future ML-based meeting analysis layer.

Purpose:
    Provide a stable package boundary between the Streamlit application and the
    new Machine Learning Minutes of Meeting pipeline.

Responsibilities:
    - Expose schema objects used by future meeting-analysis services.
    - Keep the new ML architecture importable without touching current UI,
      export, transcription, or SMTP behavior.

Inputs:
    Future callers will pass reviewed transcript text and meeting metadata.

Outputs:
    Future callers will receive a structured analysis result suitable for the
    existing PDF and DOCX exporters.

Future Implementation Notes:
    The current package is intentionally not connected to the app. A later
    migration phase can route report generation through
    ``meeting_analysis.service`` after compatibility testing.
"""

from meeting_analysis.schemas import (
    MeetingAnalysisRequest,
    MeetingAnalysisResult,
    ValidationMessage,
)

__all__ = [
    "MeetingAnalysisRequest",
    "MeetingAnalysisResult",
    "ValidationMessage",
]
