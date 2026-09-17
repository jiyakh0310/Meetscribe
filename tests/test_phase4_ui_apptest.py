"""SELaD Phase 4 -- real Streamlit AppTest smoke coverage (Part 20/24).

Complements tests/test_phase4_video_analytics.py's direct
video_analytics_html() unit tests by driving the ACTUAL Minutes page
(streamlit.testing.v1.AppTest) end to end, confirming: transcript-only
and audio-only meetings never show "Visual Interaction Insights", a
video meeting with successful analytics does, and setting
video_analytics_result never breaks the existing Minutes page.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import streamlit as st

# Some pre-existing tests (e.g. test_phase3_6_upload_error_classification.py)
# replace the module-level `st.session_state` with a plain dict to drive
# app.main.process_upload() directly outside of a real script run. That
# permanently overwrites Streamlit's session-state proxy for the rest of
# the process (module-level assignment, not a context-scoped patch), which
# breaks any AppTest run afterward. Capture the pristine proxy now, at
# collection time, before any test function body has run, and restore it
# before every test here so this file's AppTest-based tests are unaffected
# by test execution order elsewhere in the suite.
_ORIGINAL_SESSION_STATE = st.session_state


@pytest.fixture(autouse=True)
def _restore_real_streamlit_session_state():
    st.session_state = _ORIGINAL_SESSION_STATE
    yield


from streamlit.testing.v1 import AppTest
from summarization.base_summarizer import MeetingAnalysisResult, MeetingSummary, KeyDiscussionPoint
from transcription.sarvam_client import TranscriptionResult, TranscriptionSegment
from video_analytics.model import (
    TrackExpressionSummary,
    VideoAnalyticsResult,
    VideoMetadata,
    VisualWindow,
    WindowInteractionSummary,
)
from video_analytics.pipeline import STANDARD_LIMITATIONS, interpret_expression_distribution

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
    ],
)


def _video_result() -> VideoAnalyticsResult:
    probs = (0.05, 0.02, 0.02, 0.8, 0.05, 0.03, 0.03)
    summary = TrackExpressionSummary(
        track_id="Face A",
        observation_count=5,
        aggregated_probabilities=probs,
        interpretation=interpret_expression_distribution(probs, 5),
    )
    window = VisualWindow(
        start_time_seconds=0.0,
        end_time_seconds=20.0,
        track_summaries=(summary,),
        pairwise_synchrony=(),
        aggregate_synchrony=None,
        synchrony_participant_count=1,
        interaction=WindowInteractionSummary(
            available=False,
            unavailable_reason="Fewer than two participants had usable facial geometry in this window.",
            tracks_with_geometry=1,
            mutual_orientation_score=None,
            display_label="Visual interaction cue unavailable for this window.",
        ),
        coverage_note=None,
    )
    return VideoAnalyticsResult(
        available=True,
        unavailable_reason=None,
        metadata=VideoMetadata(duration_seconds=20.0, fps=25.0, width=640, height=480, frame_count=500),
        sampled_frame_count=10,
        detected_tracks=("Face A",),
        windows=(window,),
        quality_warnings=(),
        limitations=STANDARD_LIMITATIONS,
    )


def _run(*, workflow_source: str, video_analytics_result=None):
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.session_state["workflow_open"] = True
    at.session_state["workflow_stage"] = "minutes"
    at.session_state["workflow_source"] = workflow_source
    at.session_state["analysis_result"] = BASE_ANALYSIS
    at.session_state["transcript_result"] = TIMED_SEGMENTS
    at.session_state["speaker_mapping"] = {}
    at.session_state["voxels_emotion_result"] = None
    at.session_state["video_analytics_result"] = video_analytics_result
    at.session_state["meeting_info"] = {"meeting_title": "Test Meeting"}
    at.session_state["uploaded_filename"] = "test.wav"
    at.run()
    combined = "\n".join(m.value for m in at.markdown if m.value)
    return at, combined


def test_transcript_only_meeting_never_shows_visual_section() -> None:
    at, combined = _run(workflow_source="transcript")
    assert not at.exception
    assert "Visual Interaction Insights" not in combined


def test_audio_only_meeting_without_video_never_shows_visual_section() -> None:
    at, combined = _run(workflow_source="audio", video_analytics_result=None)
    assert not at.exception
    assert "Visual Interaction Insights" not in combined


def test_video_meeting_with_successful_analytics_shows_visual_section() -> None:
    at, combined = _run(workflow_source="audio", video_analytics_result=_video_result())
    assert not at.exception
    assert "Visual Interaction Insights" in combined
    assert "Facial Expression Patterns" in combined


def test_video_analytics_unavailable_result_does_not_break_minutes_page() -> None:
    from video_analytics.pipeline import STANDARD_LIMITATIONS

    unavailable = VideoAnalyticsResult(
        available=False,
        unavailable_reason="Visual analytics could not be completed for this video.",
        metadata=None,
        sampled_frame_count=0,
        detected_tracks=(),
        windows=(),
        quality_warnings=(),
        limitations=STANDARD_LIMITATIONS,
        processing_error="simulated failure",
    )
    at, combined = _run(workflow_source="audio", video_analytics_result=unavailable)

    assert not at.exception
    assert not at.error
    # MoM content must still be present even though video analytics failed.
    assert "Meeting" in combined
    assert "Visual Interaction Insights" not in combined
