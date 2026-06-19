"""Temporary local smoke test for Minutes of Meeting PDF export.

Usage:
    python temp_test_pdf_export.py
"""

from __future__ import annotations

from exports.pdf_exporter import export_to_pdf, export_transcript_to_pdf
from summarization.base_summarizer import (
    ActionItem,
    Decision,
    KeyDiscussionPoint,
    MeetingAnalysisResult,
    MeetingSummary,
)


def build_sample_analysis() -> MeetingAnalysisResult:
    return MeetingAnalysisResult(
        cleaned_transcript=(
            "Speaker 1 [00:00 - 00:08]\n"
            "Hello team, let's review Phase 2 of MeetScribe today.\n\n"
            "Speaker 2 [00:08 - 00:18]\n"
            "The transcription pipeline is working now, so we should keep "
            "Phase 1 unchanged.\n\n"
            "Speaker 1 [00:18 - 00:32]\n"
            "Agreed. We will use the speaker-wise transcript as input for "
            "cleanup, summary, decisions, action items, and key points."
        ),
        summary=MeetingSummary(
            title="MeetScribe Phase 2 Review",
            short_summary=(
                "The team reviewed the stable Phase 1 transcription pipeline "
                "and confirmed the Phase 2 meeting analysis workflow."
            ),
            detailed_summary=(
                "The meeting focused on preserving the working transcription "
                "pipeline while layering Gemini-powered meeting analysis on top. "
                "The team agreed that the speaker-wise transcript should feed "
                "cleanup, summary generation, key discussion points, decisions, "
                "and action item extraction."
            ),
            topics_discussed=[
                "Phase 1 transcription stability",
                "Phase 2 meeting analysis",
                "Gemini-powered extraction",
            ],
        ),
        key_discussion_points=[
            KeyDiscussionPoint(
                point="Phase 1 is stable and should remain unchanged.",
                speakers=["Speaker 2"],
                timestamp="00:08 - 00:18",
            ),
            KeyDiscussionPoint(
                point=(
                    "Phase 2 uses the speaker-wise transcript for all analysis "
                    "modules."
                ),
                speakers=["Speaker 1"],
                timestamp="00:18 - 00:32",
            ),
        ],
        decisions=[
            Decision(
                decision="Keep Phase 1 unchanged.",
                owner="Speaker 2",
                timestamp="00:08 - 00:18",
                confidence="high",
            )
        ],
        action_items=[
            ActionItem(
                task="Connect the Streamlit UI to the analysis pipeline.",
                owner="Rahul",
                due_date="Friday",
                timestamp="00:32 - 00:45",
                status="open",
            )
        ],
    )


def main() -> int:
    mom_path = export_to_pdf(
        build_sample_analysis(),
        meeting_info={"Source File": "sample.aac"},
    )
    transcript_path = export_transcript_to_pdf(
        build_sample_analysis(),
        meeting_info={"Source File": "sample.aac"},
    )
    print(f"Generated MoM PDF: {mom_path}")
    print(f"Generated Transcript PDF: {transcript_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
