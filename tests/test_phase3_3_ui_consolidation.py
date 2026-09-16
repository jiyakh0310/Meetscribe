"""AppTest coverage for Phase 3.3 (communication signals consolidation).

Verifies the Minutes page renders exactly ONE user-facing communication/
emotion section (Meeting Analytics -> Communication Signals), with no
duplicate topic-emotion cards, uncertainty-aware wording for a mixed
distribution, no fabricated acoustic data for transcript-only meetings,
and that Positive-language cues remain a visibly separate signal.

Uses the real Streamlit app (``streamlit.testing.v1.AppTest``) with
synthetic session state -- no live browser needed. Synthetic Voxels
payloads use distributions distinct from the real validation audio's
actual percentages (a different close-top-two shape), per the Phase 3.3
anti-overfitting requirement.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest
from summarization.base_summarizer import MeetingAnalysisResult, MeetingSummary, KeyDiscussionPoint
from transcription.sarvam_client import TranscriptionResult, TranscriptionSegment

APP_PATH = str(Path(__file__).resolve().parents[1] / "app" / "main.py")

BASE_ANALYSIS = MeetingAnalysisResult(
    cleaned_transcript="dummy transcript",
    summary=MeetingSummary(title="Meeting", short_summary="Summary text.", detailed_summary="Detail text.", topics_discussed=["Topic"]),
    key_discussion_points=[KeyDiscussionPoint(point="API integration was discussed.", timestamp="00:10", speakers=["Speaker 1"])],
    decisions=[],
    action_items=[],
)

TIMED_SEGMENTS = TranscriptionResult(
    transcript="ignored",
    segments=[
        TranscriptionSegment("Let's talk about the API integration approach for the new service.", "0", 0.0, 6.0),
        TranscriptionSegment("The API integration looks solid, great work on the authentication flow.", "1", 6.0, 14.0),
        TranscriptionSegment("Now let's discuss the database migration timeline for next quarter.", "0", 14.0, 22.0),
        TranscriptionSegment("Database migration planning sounds good, thank you for the update.", "1", 22.0, 30.0),
    ],
)

SINGLE_SEGMENT = TranscriptionResult(
    transcript="ignored",
    segments=[TranscriptionSegment("Release testing update, everything looks solid today.", "0", 0.0, 8.0)],
)


def _mixed_probabilities() -> dict:
    # A different close-top-two shape than the real validation audio's
    # actual numbers -- proves the general rule, not a hardcoded case.
    return {"disgust": 0.19, "neutral": 0.18, "sad": 0.17, "angry": 0.16, "happy": 0.12, "fear": 0.09, "surprise": 0.09}


def _clear_probabilities() -> dict:
    return {"happy": 0.70, "neutral": 0.10, "sad": 0.08, "angry": 0.05, "fear": 0.03, "disgust": 0.02, "surprise": 0.02}


def _voxels_payload(probabilities: dict, *, windows_analyzed: int = 4, quality_warnings=None) -> dict:
    dominant = max(probabilities, key=probabilities.get)
    windows = [
        {"start_time": float(i * 6), "end_time": float(i * 6 + 4), "probabilities": probabilities, "confidence": 0.5}
        for i in range(windows_analyzed)
    ]
    return {
        "available": True,
        "dominant_emotion": dominant.title(),
        "confidence": probabilities[dominant],
        "probabilities": probabilities,
        "tone": "Negative" if dominant in {"angry", "disgust", "fear", "sad"} else "Positive" if dominant == "happy" else "Neutral",
        "observation": "Synthetic observation for testing.",
        "windows": windows,
        "windows_analyzed": windows_analyzed,
        "average_window_confidence": 0.5,
        "quality_warnings": quality_warnings or [],
        "note": "Detected speech-emotion patterns only; this is not a measure of a person's true psychological state.",
        "alignment_available": True,
    }


def _run(*, workflow_source, transcript_result, speaker_mapping=None, voxels=None, stage="minutes"):
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.session_state["workflow_open"] = True
    at.session_state["workflow_stage"] = stage
    at.session_state["workflow_source"] = workflow_source
    at.session_state["analysis_result"] = BASE_ANALYSIS
    at.session_state["transcript_result"] = transcript_result
    at.session_state["speaker_mapping"] = speaker_mapping or {}
    at.session_state["voxels_emotion_result"] = voxels
    at.session_state["meeting_info"] = {"meeting_title": "Test Meeting"}
    at.session_state["uploaded_filename"] = "test.wav"
    at.run()
    combined = "\n".join(m.value for m in at.markdown if m.value)
    return at, combined


# --- 1: audio + clear Voxels distribution ------------------------------------------


def test_audio_with_clear_distribution_renders_without_exception() -> None:
    at, combined = _run(
        workflow_source="audio",
        transcript_result=TIMED_SEGMENTS,
        speaker_mapping={"Speaker 1": "Nora", "Speaker 2": "Devraj"},
        voxels=_voxels_payload(_clear_probabilities()),
    )
    assert not at.exception
    assert not at.error
    assert "Happy-associated acoustic pattern" in combined
    assert "Derived meeting tone" not in combined


# --- 2: audio + mixed distribution --------------------------------------------------


def test_audio_with_mixed_distribution_shows_uncertainty_aware_wording() -> None:
    at, combined = _run(
        workflow_source="audio",
        transcript_result=TIMED_SEGMENTS,
        speaker_mapping={"Speaker 1": "Nora", "Speaker 2": "Devraj"},
        voxels=_voxels_payload(_mixed_probabilities()),
    )
    assert not at.exception
    assert "No clearly dominant acoustic pattern was detected." in combined
    assert "Highest model probability" in combined
    # No categorical Negative headline for an uncertain distribution.
    assert "Derived meeting tone" not in combined


# --- 3: audio + quality warning ------------------------------------------------------


def test_quality_warning_is_surfaced_in_communication_signals() -> None:
    at, combined = _run(
        workflow_source="audio",
        transcript_result=TIMED_SEGMENTS,
        voxels=_voxels_payload(_mixed_probabilities(), quality_warnings=["Low audio volume detected in some windows."]),
    )
    assert not at.exception
    assert "Low audio volume detected in some windows." in combined


# --- 4: audio + topic insights -------------------------------------------------------


def test_topic_level_acoustic_cards_use_shared_interpretation() -> None:
    at, combined = _run(
        workflow_source="audio",
        transcript_result=TIMED_SEGMENTS,
        speaker_mapping={"Speaker 1": "Nora", "Speaker 2": "Devraj"},
        voxels=_voxels_payload(_mixed_probabilities()),
    )
    assert "ms-analytics-comm-card" in combined
    assert "ms-analytics-comm-pattern" in combined


# --- 5: transcript-only ---------------------------------------------------------------


def test_transcript_only_meeting_does_not_fabricate_acoustic_data() -> None:
    at, combined = _run(workflow_source="transcript", transcript_result=TIMED_SEGMENTS, voxels=None)
    assert not at.exception
    assert "Speech-emotion insights are available for meetings with audio." in combined
    assert "ms-analytics-fill" not in combined[combined.find("Communication Signals"):combined.find("Communication Signals") + 2000]


# --- 6: audio + Voxels unavailable ----------------------------------------------------


def test_audio_with_voxels_unavailable_is_distinguished_from_no_audio() -> None:
    at, combined = _run(workflow_source="audio", transcript_result=TIMED_SEGMENTS, voxels={"available": False})
    assert not at.exception
    idx = combined.find("Communication Signals")
    section = combined[idx : idx + 300]
    assert "unavailable" in section.lower()
    assert "Speech-emotion insights are available for meetings with audio." not in section


# --- 7: no topics (single-event meeting still produces a topic; verify no crash) -----


def test_single_topic_meeting_renders_without_exception() -> None:
    at, combined = _run(
        workflow_source="audio",
        transcript_result=SINGLE_SEGMENT,
        voxels=_voxels_payload(_clear_probabilities(), windows_analyzed=1),
    )
    assert not at.exception
    assert not at.error


# --- 8: single topic (explicit, same as above kept for matrix completeness) ---------


def test_single_topic_shows_exactly_one_communication_signals_section() -> None:
    at, combined = _run(
        workflow_source="audio",
        transcript_result=SINGLE_SEGMENT,
        voxels=_voxels_payload(_clear_probabilities(), windows_analyzed=1),
    )
    assert combined.count("Communication Signals") == 1


# --- 9: many topics ---------------------------------------------------------------------


def test_many_topics_still_render_one_consolidated_section() -> None:
    at, combined = _run(
        workflow_source="audio",
        transcript_result=TIMED_SEGMENTS,
        speaker_mapping={"Speaker 1": "Nora", "Speaker 2": "Devraj"},
        voxels=_voxels_payload(_clear_probabilities()),
    )
    assert combined.count("Communication Signals") == 1
    assert "Discussion-Level Emotion Insights" not in combined


# --- 10: Export stage ---------------------------------------------------------------------


def test_export_stage_still_renders_without_exception() -> None:
    at, _ = _run(
        workflow_source="audio",
        transcript_result=TIMED_SEGMENTS,
        voxels=_voxels_payload(_mixed_probabilities()),
        stage="export",
    )
    assert not at.exception
    assert not at.error


# --- Positive-language cues remain a visibly separate signal --------------------------


def test_positive_language_cues_remain_separate_from_speech_emotion() -> None:
    at, combined = _run(
        workflow_source="audio",
        transcript_result=TIMED_SEGMENTS,
        speaker_mapping={"Speaker 1": "Nora", "Speaker 2": "Devraj"},
        voxels=_voxels_payload(_clear_probabilities()),
    )
    assert "Positive-language cues" in combined
    assert "derived from transcript TEXT (lexical)" in combined


# --- Technical timeline remains collapsed ----------------------------------------------


def test_technical_timeline_details_remains_a_collapsed_expander() -> None:
    at, _ = _run(workflow_source="audio", transcript_result=TIMED_SEGMENTS, voxels=_voxels_payload(_clear_probabilities()))
    labels = [expander.label for expander in at.expander]
    assert "Technical timeline details" in labels
