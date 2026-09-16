"""PDF export tests for Phase 3.3 (uncertainty-aware Communication Signals).

Generates real PDFs via ``exports.pdf_exporter.export_to_pdf`` and verifies
their extracted text programmatically (via ``pypdf``), per the task's
explicit "TESTS -- PDF" requirement: clear distribution, mixed
distribution, and a quality-warning case. Verifies factual MoM sections
are present and unchanged, and that no unsupported categorical statement
(e.g. "Derived meeting tone: Negative") appears for an uncertain
distribution.
"""

import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pypdf import PdfReader

from exports.pdf_exporter import export_to_pdf
from meeting_analytics.communication import interpret_emotion_distribution
from summarization.base_summarizer import ActionItem, Decision, KeyDiscussionPoint, MeetingAnalysisResult, MeetingSummary

ANALYSIS = MeetingAnalysisResult(
    cleaned_transcript="dummy",
    summary=MeetingSummary(
        title="Synthetic Meeting",
        short_summary="We reviewed the API integration plan.",
        detailed_summary="Detailed summary here.",
        topics_discussed=["API Integration"],
    ),
    key_discussion_points=[KeyDiscussionPoint(point="API Integration: the team reviewed the plan.", timestamp="00:10", speakers=["Nora"])],
    decisions=[Decision(decision="We agreed to proceed with the API integration plan.", owner="Nora", timestamp="00:20", confidence="High")],
    action_items=[ActionItem(task="Finalize the API spec.", owner="Devraj", due_date="Friday", timestamp="00:25", status="Open")],
)


def _extract_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _normalized(text: str) -> str:
    """Collapse all whitespace (including PDF table-cell line wraps) to a
    single space, so a phrase that visually wraps across lines in a narrow
    table column can still be matched as one continuous phrase."""

    return re.sub(r"\s+", " ", text)


def _voxels(probabilities: dict, *, quality_warnings=None, topic_probabilities: dict | None = None) -> dict:
    dominant = max(probabilities, key=probabilities.get)
    topic_probs = topic_probabilities or probabilities
    topic_interpretation = interpret_emotion_distribution(topic_probs)
    return {
        "available": True,
        "dominant_emotion": dominant.title(),
        "confidence": probabilities[dominant],
        "probabilities": probabilities,
        "tone": "Negative" if dominant in {"angry", "disgust", "fear", "sad"} else "Positive" if dominant == "happy" else "Neutral",
        "observation": "Synthetic observation for testing.",
        "windows_analyzed": 5,
        "quality_warnings": quality_warnings or [],
        "note": "Detected speech-emotion patterns only; this is not a measure of a person's true psychological state.",
        "topic_insights": [
            {
                "topic": "API Integration",
                "timestamp": "00:10",
                "dominant_emotion": (topic_interpretation.highest_class or "uncertain").title(),
                "average_confidence": 0.5,
                "tone": "Negative",
                "pattern": topic_interpretation.display_pattern,
                "windows_analyzed": 3,
                "probabilities": topic_probs,
                "status": topic_interpretation.status,
                "highest_probability": topic_interpretation.highest_probability,
            },
        ],
    }


_CLEAR_PROBABILITIES = {"happy": 0.70, "neutral": 0.10, "sad": 0.08, "angry": 0.05, "fear": 0.03, "disgust": 0.02, "surprise": 0.02}
# A different close-top-two shape than the real validation audio's actual
# numbers -- proves the general rule, not a hardcoded real-audio case.
_MIXED_PROBABILITIES = {"disgust": 0.19, "neutral": 0.18, "sad": 0.17, "angry": 0.16, "happy": 0.12, "fear": 0.09, "surprise": 0.09}


def _factual_sections_present(text: str) -> bool:
    return all(marker in text for marker in ("EXECUTIVE SUMMARY", "KEY DISCUSSION POINTS", "DECISIONS TAKEN", "ACTION ITEMS"))


# --- A: clear distribution -----------------------------------------------------------


def test_clear_distribution_pdf_names_the_acoustic_class_not_a_person() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = export_to_pdf(
            ANALYSIS,
            meeting_info={"Meeting Title": "Synthetic Meeting"},
            voxels_emotion=_voxels(_CLEAR_PROBABILITIES),
            output_path=Path(tmp) / "clear.pdf",
        )
        text = _extract_text(path)
    assert "COMMUNICATION SIGNALS" in text
    assert "Happy-associated acoustic pattern" in text
    assert "Highest model probability" in text
    assert "Derived meeting tone" not in text
    assert "Dominant detected speech emotion:" not in text
    assert _factual_sections_present(text)


# --- B: mixed distribution -------------------------------------------------------------


def test_mixed_distribution_pdf_contains_uncertainty_wording_not_categorical_negative() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = export_to_pdf(
            ANALYSIS,
            meeting_info={"Meeting Title": "Synthetic Meeting"},
            voxels_emotion=_voxels(_MIXED_PROBABILITIES),
            output_path=Path(tmp) / "mixed.pdf",
        )
        text = _extract_text(path)
    assert "No clearly dominant acoustic pattern was detected." in text
    assert "Derived meeting tone: Negative" not in text
    assert "Highest model probability" in text
    assert _factual_sections_present(text)


# --- C: quality-warning case -----------------------------------------------------------


def test_quality_warning_case_pdf_surfaces_the_warning() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = export_to_pdf(
            ANALYSIS,
            meeting_info={"Meeting Title": "Synthetic Meeting"},
            voxels_emotion=_voxels(
                _MIXED_PROBABILITIES,
                quality_warnings=["Low audio volume detected in some windows.", "Limited usable speech windows."],
            ),
            output_path=Path(tmp) / "warning.pdf",
        )
        text = _extract_text(path)
    assert "Low audio volume detected in some windows." in text
    assert "Limited usable speech windows." in text
    assert _factual_sections_present(text)


# --- Phase 3.4: topic-level acoustic table was removed from the PDF ------------------


def test_topic_level_acoustic_table_is_no_longer_rendered() -> None:
    # Phase 3.4: this test previously asserted the OPPOSITE (that a
    # "TOPIC-LEVEL ACOUSTIC PATTERNS" table renders with uncertainty-aware
    # wording). That table was keyed to Phase 2/3.2's internal micro-topic
    # clustering, which Phase 3.4 stopped presenting to users as "the
    # meeting's topics" -- reprinting the same cluster labels in the PDF,
    # merely relabeled as acoustic evidence, would have re-exposed the
    # same fabricated-looking topic titles Phase 3.4 removed from the
    # Minutes UI (see PART P of the Phase 3.4 task: "prefer omitting that
    # subsection rather than presenting fake topics"). The meeting-level
    # acoustic distribution (asserted elsewhere in this file) is
    # unaffected and remains the primary Communication Signals evidence.
    with tempfile.TemporaryDirectory() as tmp:
        path = export_to_pdf(
            ANALYSIS,
            meeting_info={"Meeting Title": "Synthetic Meeting"},
            voxels_emotion=_voxels(_CLEAR_PROBABILITIES, topic_probabilities=_MIXED_PROBABILITIES),
            output_path=Path(tmp) / "topic_mixed.pdf",
        )
        text = _normalized(_extract_text(path))
    assert "TOPIC-LEVEL ACOUSTIC PATTERNS" not in text
    # Meeting-level evidence must still be present.
    assert "COMMUNICATION SIGNALS" in text


# --- No Voxels payload / transcript-only: no emotion section at all -------------------


def test_no_voxels_payload_omits_communication_signals_section() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = export_to_pdf(
            ANALYSIS,
            meeting_info={"Meeting Title": "Synthetic Meeting"},
            voxels_emotion=None,
            output_path=Path(tmp) / "no_voxels.pdf",
        )
        text = _extract_text(path)
    assert "COMMUNICATION SIGNALS" not in text
    assert _factual_sections_present(text)


def test_voxels_unavailable_flag_omits_communication_signals_section() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = export_to_pdf(
            ANALYSIS,
            meeting_info={"Meeting Title": "Synthetic Meeting"},
            voxels_emotion={"available": False},
            output_path=Path(tmp) / "unavailable.pdf",
        )
        text = _extract_text(path)
    assert "COMMUNICATION SIGNALS" not in text
