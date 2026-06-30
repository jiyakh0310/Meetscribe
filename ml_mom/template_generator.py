"""Template generation boundary for future ML-based MoM output.

Purpose:
    Reserve the deterministic assembly layer that will convert classified
    transcript evidence into professional Minutes of Meeting sections.

Responsibilities:
    - Group classified sentences into MoM sections.
    - Preserve traceability to speakers and timestamps.
    - Avoid inventing owners, deadlines, decisions, or conclusions.

Inputs:
    Classified sentences, topic clusters, and meeting metadata.

Outputs:
    Structured MoM section content.

Future Implementation Notes:
    Template generation should stay rule-based and auditable even after ML
    classification is added.
"""

from meeting_analysis.schemas import MeetingAnalysisResult


def build_minutes_placeholder() -> MeetingAnalysisResult:
    """Build an empty placeholder Minutes of Meeting result.

    Returns:
        Empty ``MeetingAnalysisResult`` with metadata identifying the skeleton
        pipeline.
    """

    # Empty sections make this skeleton safe to import without affecting current
    # report-generation behavior.
    # TODO:
    # Assemble MoM sections from classified transcript evidence.
    return MeetingAnalysisResult(metadata={"pipeline": "ml_mom_placeholder"})
