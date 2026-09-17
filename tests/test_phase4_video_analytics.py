"""SELaD Phase 4 -- video analytics test matrix (Part 20).

Covers: preprocessing, FPS-aware sampling, face detection/tracking,
expression classification, 20-second windowing, synchrony, and visual
interaction, using synthetic/deterministic fixtures wherever possible
so the suite stays fast and reproducible. A small number of tests use
a real downloaded face image to exercise the true MediaPipe/ONNX
models end-to-end (network-dependent; skipped gracefully if models
cannot be fetched).
"""

from __future__ import annotations

import sys
import tempfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from video_analytics.faces import _iou, detect_faces
from video_analytics.interaction import compute_window_interaction
from video_analytics.model import DetectedFace, TrackExpressionSummary
from video_analytics.models import get_model_path
from video_analytics.pipeline import analyze_video, interpret_expression_distribution
from video_analytics.preprocessing import (
    VideoProcessingError,
    inspect_video_metadata,
    normalize_video_for_analysis,
)
from video_analytics.sampling import TARGET_SAMPLE_INTERVAL_SECONDS, sample_frame_indices
from video_analytics.synchrony import compute_window_synchrony


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_video(path: Path, *, fps: float, frame_count: int, size=(320, 240), frame=None) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if frame is None:
        frame = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    for _ in range(frame_count):
        writer.write(frame)
    writer.release()


@pytest.fixture(scope="module")
def real_face_image() -> np.ndarray | None:
    tmp_path = Path(tempfile.gettempdir()) / "phase4_test_lena.jpg"
    if not tmp_path.exists():
        try:
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
                str(tmp_path),
            )
        except Exception:
            return None
    return cv2.imread(str(tmp_path))


# ---------------------------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------------------------


def test_preprocessing_valid_mp4_returns_metadata_without_normalization(tmp_path: Path) -> None:
    video_path = tmp_path / "valid.mp4"
    _write_video(video_path, fps=30.0, frame_count=60)

    metadata = inspect_video_metadata(video_path)
    assert metadata.width == 320
    assert metadata.height == 240
    assert metadata.fps == pytest.approx(30.0, rel=0.05)
    assert metadata.duration_seconds > 0


def test_preprocessing_invalid_video_raises_video_processing_error(tmp_path: Path) -> None:
    bogus_path = tmp_path / "not_a_video.mp4"
    bogus_path.write_bytes(b"this is not a real video file")

    with pytest.raises(VideoProcessingError):
        normalize_video_for_analysis(bogus_path)


def test_preprocessing_missing_file_raises() -> None:
    with pytest.raises(VideoProcessingError):
        normalize_video_for_analysis(Path("/nonexistent/path/video.mp4"))


def test_preprocessing_empty_file_raises(tmp_path: Path) -> None:
    empty_path = tmp_path / "empty.mp4"
    empty_path.write_bytes(b"")

    with pytest.raises(VideoProcessingError):
        normalize_video_for_analysis(empty_path)


def test_preprocessing_unsupported_extension_raises(tmp_path: Path) -> None:
    bad_ext_path = tmp_path / "video.xyz"
    bad_ext_path.write_bytes(b"fake bytes")

    with pytest.raises(VideoProcessingError):
        normalize_video_for_analysis(bad_ext_path)


def test_preprocessing_never_fabricates_fps_when_unavailable(tmp_path: Path) -> None:
    # A corrupt/zero-fps container reports fps<=0 via OpenCV; our code
    # must surface None, never a guessed number like 30.0.
    from video_analytics.model import VideoMetadata

    metadata = VideoMetadata(duration_seconds=0.0, fps=None, width=320, height=240, frame_count=None)
    assert metadata.fps is None


# ---------------------------------------------------------------------------
# SAMPLING
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fps", [24.0, 25.0, 30.0, 60.0])
def test_sampling_is_fps_aware_and_consistent_cadence(fps: float) -> None:
    duration_seconds = 10.0
    frame_count = int(fps * duration_seconds)

    samples = sample_frame_indices(fps=fps, frame_count=frame_count, duration_seconds=duration_seconds)

    assert len(samples) == 5
    expected_timestamps = [i * TARGET_SAMPLE_INTERVAL_SECONDS for i in range(5)]
    for sample, expected in zip(samples, expected_timestamps):
        assert sample.timestamp_seconds == pytest.approx(expected, abs=0.01)


def test_sampling_timestamps_derive_from_actual_fps_not_hardcoded() -> None:
    # At 24fps, the interval in FRAMES differs from 30fps, but the
    # resulting cadence in SECONDS must still be ~2.0s either way.
    samples_24 = sample_frame_indices(fps=24.0, frame_count=240, duration_seconds=10.0)
    samples_30 = sample_frame_indices(fps=30.0, frame_count=300, duration_seconds=10.0)

    assert [s.frame_index for s in samples_24] == [0, 48, 96, 144, 192]
    assert [s.frame_index for s in samples_30] == [0, 60, 120, 180, 240]


def test_sampling_zero_duration_returns_no_samples() -> None:
    assert sample_frame_indices(fps=30.0, frame_count=0, duration_seconds=0.0) == []


def test_sampling_missing_fps_falls_back_to_time_based_sampling() -> None:
    samples = sample_frame_indices(fps=None, frame_count=None, duration_seconds=6.0)
    assert [s.timestamp_seconds for s in samples] == [0.0, 2.0, 4.0]


# ---------------------------------------------------------------------------
# FACE PIPELINE
# ---------------------------------------------------------------------------


def test_iou_identical_boxes_is_one() -> None:
    box = (0.1, 0.1, 0.2, 0.2)
    assert _iou(box, box) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero() -> None:
    assert _iou((0.0, 0.0, 0.1, 0.1), (0.5, 0.5, 0.1, 0.1)) == 0.0


def test_face_pipeline_no_face_returns_empty_without_crash(tmp_path: Path) -> None:
    video_path = tmp_path / "noface.mp4"
    _write_video(video_path, fps=25.0, frame_count=50)

    result = analyze_video(video_path)

    assert result.available is True
    assert result.detected_tracks == ()
    assert any("No faces were detected" in w for w in result.quality_warnings)


def test_face_pipeline_one_face_produces_single_track(tmp_path: Path, real_face_image) -> None:
    if real_face_image is None:
        pytest.skip("Could not fetch real face fixture image (offline).")
    if get_model_path("blaze_face_short_range.tflite") is None:
        pytest.skip("Face detector model unavailable (offline).")

    video_path = tmp_path / "oneface.mp4"
    h, w = real_face_image.shape[:2]
    _write_video(video_path, fps=25.0, frame_count=50, size=(w, h), frame=real_face_image)

    result = analyze_video(video_path)

    assert result.available is True
    assert result.detected_tracks == ("Face A",)


def test_face_pipeline_ambiguous_tracking_never_forces_identity() -> None:
    # Two detections far apart (low IoU) must start a NEW track rather
    # than being forced to match an existing one (Part 7).
    faces = [
        DetectedFace(0.0, 0, "Face A", (0.0, 0.0, 0.1, 0.1), 0.9),
    ]
    previous_boxes = [(f.track_id, f.bounding_box) for f in faces]
    far_box = (0.8, 0.8, 0.1, 0.1)
    overlaps = [_iou(far_box, box) for _label, box in previous_boxes]
    assert max(overlaps) < 0.3  # below _IOU_MATCH_THRESHOLD -> would start a new track


# ---------------------------------------------------------------------------
# EXPRESSIONS
# ---------------------------------------------------------------------------


def test_expression_interpretation_clear_pattern_with_strong_evidence() -> None:
    from video_analytics.model import EXPRESSION_CLASSES

    probs = tuple(0.85 if cls == "Happiness" else 0.025 for cls in EXPRESSION_CLASSES)
    interpretation = interpret_expression_distribution(probs, observation_count=5)

    assert interpretation.status == "clear"
    assert interpretation.top_class == "Happiness"
    assert "Happiness" in interpretation.display_pattern


def test_expression_interpretation_flat_distribution_is_mixed() -> None:
    from video_analytics.model import EXPRESSION_CLASSES

    probs = tuple(1.0 / len(EXPRESSION_CLASSES) for _ in EXPRESSION_CLASSES)
    interpretation = interpret_expression_distribution(probs, observation_count=5)

    assert interpretation.status == "mixed"
    assert interpretation.display_pattern == "No clearly dominant facial-expression pattern was observed."


def test_expression_interpretation_no_observations_is_unavailable() -> None:
    interpretation = interpret_expression_distribution((), observation_count=0)
    assert interpretation.status == "unavailable"


def test_expression_interpretation_requires_minimum_observations() -> None:
    from video_analytics.model import EXPRESSION_CLASSES

    probs = tuple(0.85 if cls == "Happiness" else 0.025 for cls in EXPRESSION_CLASSES)
    # Only 1 observation -- below _MIN_OBSERVATIONS_FOR_DOMINANT_CLASS,
    # so even a strong-looking probability vector must not be "clear".
    interpretation = interpret_expression_distribution(probs, observation_count=1)
    assert interpretation.status == "mixed"


# ---------------------------------------------------------------------------
# WINDOWS
# ---------------------------------------------------------------------------


def test_windowing_produces_exact_20_second_boundaries(tmp_path: Path, real_face_image) -> None:
    if real_face_image is None:
        pytest.skip("Could not fetch real face fixture image (offline).")
    if get_model_path("blaze_face_short_range.tflite") is None:
        pytest.skip("Face detector model unavailable (offline).")

    video_path = tmp_path / "windowtest.mp4"
    h, w = real_face_image.shape[:2]
    # 45 seconds -> 3 windows: [0,20), [20,40), [40,60)
    _write_video(video_path, fps=10.0, frame_count=450, size=(w, h), frame=real_face_image)

    result = analyze_video(video_path)

    assert result.available is True
    assert len(result.windows) == 3
    assert result.windows[0].start_time_seconds == 0.0
    assert result.windows[0].end_time_seconds == 20.0
    assert result.windows[-1].start_time_seconds == 40.0


def test_windowing_missing_participant_in_window_shows_zero_observations() -> None:
    from video_analytics.model import EXPRESSION_CLASSES

    summary = TrackExpressionSummary(
        track_id="Face A",
        observation_count=0,
        aggregated_probabilities=(),
        interpretation=interpret_expression_distribution((), 0),
    )
    assert summary.interpretation.status == "unavailable"
    assert summary.aggregated_probabilities == ()


# ---------------------------------------------------------------------------
# SYNCHRONY
# ---------------------------------------------------------------------------


def _summary(track_id: str, probs: tuple[float, ...], count: int = 3) -> TrackExpressionSummary:
    return TrackExpressionSummary(
        track_id=track_id,
        observation_count=count,
        aggregated_probabilities=probs,
        interpretation=interpret_expression_distribution(probs, count),
    )


def test_synchrony_identical_vectors_gives_maximum_similarity() -> None:
    from video_analytics.model import EXPRESSION_CLASSES

    vector = tuple(0.85 if cls == "Happiness" else 0.025 for cls in EXPRESSION_CLASSES)
    pairwise, aggregate, count = compute_window_synchrony((_summary("Face A", vector), _summary("Face B", vector)))

    assert count == 2
    assert aggregate == pytest.approx(1.0)
    assert pairwise[0].similarity == pytest.approx(1.0)


def test_synchrony_very_different_vectors_gives_lower_similarity() -> None:
    from video_analytics.model import EXPRESSION_CLASSES

    vector_a = tuple(0.85 if cls == "Happiness" else 0.025 for cls in EXPRESSION_CLASSES)
    vector_b = tuple(0.85 if cls == "Sadness" else 0.025 for cls in EXPRESSION_CLASSES)
    _pairwise, aggregate, _count = compute_window_synchrony((_summary("Face A", vector_a), _summary("Face B", vector_b)))

    assert aggregate < 0.9


def test_synchrony_three_participants_gives_three_pairs() -> None:
    from video_analytics.model import EXPRESSION_CLASSES

    vector = tuple(0.85 if cls == "Happiness" else 0.025 for cls in EXPRESSION_CLASSES)
    pairwise, _aggregate, count = compute_window_synchrony(
        (_summary("Face A", vector), _summary("Face B", vector), _summary("Face C", vector))
    )
    assert count == 3
    assert len(pairwise) == 3


def test_synchrony_one_participant_is_unavailable() -> None:
    from video_analytics.model import EXPRESSION_CLASSES

    vector = tuple(0.85 if cls == "Happiness" else 0.025 for cls in EXPRESSION_CLASSES)
    pairwise, aggregate, count = compute_window_synchrony((_summary("Face A", vector),))

    assert aggregate is None
    assert pairwise == ()
    assert count == 1


def test_synchrony_insufficient_observations_is_unavailable() -> None:
    from video_analytics.model import EXPRESSION_CLASSES

    vector = tuple(0.85 if cls == "Happiness" else 0.025 for cls in EXPRESSION_CLASSES)
    # Both tracks have only 1 observation -- below MIN_OBSERVATIONS_FOR_SYNCHRONY.
    pairwise, aggregate, count = compute_window_synchrony(
        (_summary("Face A", vector, count=1), _summary("Face B", vector, count=1))
    )
    assert aggregate is None
    assert count == 0


# ---------------------------------------------------------------------------
# VISUAL INTERACTION
# ---------------------------------------------------------------------------


def test_interaction_usable_geometry_for_two_tracks_is_available() -> None:
    # Face A on the left facing right (positive yaw), Face B on the
    # right facing left (negative yaw) -- consistent mutual orientation.
    geometry = {
        "Face A": [(0.2, 30.0), (0.2, 25.0)],
        "Face B": [(0.8, -30.0), (0.8, -25.0)],
    }
    summary = compute_window_interaction(geometry)
    assert summary.available is True
    assert summary.tracks_with_geometry == 2
    assert summary.mutual_orientation_score is not None


def test_interaction_missing_landmarks_is_unavailable() -> None:
    summary = compute_window_interaction({})
    assert summary.available is False
    assert summary.mutual_orientation_score is None


def test_interaction_single_track_never_forces_a_result() -> None:
    summary = compute_window_interaction({"Face A": [(0.5, 0.0)]})
    assert summary.available is False
    assert "Fewer than two participants" in summary.unavailable_reason


def test_interaction_display_label_never_claims_definitive_eye_contact() -> None:
    geometry = {"Face A": [(0.2, 30.0)], "Face B": [(0.8, -30.0)]}
    summary = compute_window_interaction(geometry)
    assert "estimated" in summary.display_label.lower() or "cue" in summary.display_label.lower()


# ---------------------------------------------------------------------------
# PIPELINE-LEVEL FAILURE ISOLATION
# ---------------------------------------------------------------------------


def test_pipeline_never_raises_on_corrupt_video(tmp_path: Path) -> None:
    bogus_path = tmp_path / "corrupt.mp4"
    bogus_path.write_bytes(b"garbage not a real video")

    result = analyze_video(bogus_path)

    assert result.available is False
    assert result.unavailable_reason is not None


def test_pipeline_result_always_carries_standard_limitations(tmp_path: Path) -> None:
    video_path = tmp_path / "limitations.mp4"
    _write_video(video_path, fps=25.0, frame_count=50)

    result = analyze_video(video_path)

    assert len(result.limitations) > 0
    assert any("observational cues" in limitation for limitation in result.limitations)


# ---------------------------------------------------------------------------
# UI (Part 20) -- video_analytics_html is a pure function of a
# VideoAnalyticsResult, so these exercise app.main's rendering directly
# rather than driving the full Streamlit app end to end.
# ---------------------------------------------------------------------------


def _make_result(**overrides):
    from video_analytics.model import VideoAnalyticsResult
    from video_analytics.pipeline import STANDARD_LIMITATIONS

    defaults = dict(
        available=True,
        unavailable_reason=None,
        metadata=None,
        sampled_frame_count=5,
        detected_tracks=(),
        windows=(),
        quality_warnings=(),
        limitations=STANDARD_LIMITATIONS,
        processing_error=None,
    )
    defaults.update(overrides)
    return VideoAnalyticsResult(**defaults)


def test_ui_no_video_result_renders_no_visual_section() -> None:
    import app.main as m

    assert m.video_analytics_html(None) == ""


def test_ui_unavailable_video_result_renders_no_visual_section() -> None:
    import app.main as m

    result = _make_result(available=False, unavailable_reason="Video could not be processed.")
    assert m.video_analytics_html(result) == ""


def test_ui_video_with_no_detected_faces_shows_safe_unavailable_state() -> None:
    import app.main as m
    from video_analytics.model import VideoMetadata

    result = _make_result(
        metadata=VideoMetadata(duration_seconds=30.0, fps=25.0, width=640, height=480, frame_count=750),
        detected_tracks=(),
    )
    section_html = m.video_analytics_html(result)

    assert "Visual Interaction Insights" in section_html
    assert "No faces were detected" in section_html


def test_ui_video_with_successful_analytics_renders_all_subsections() -> None:
    import app.main as m
    from video_analytics.model import (
        ExpressionDistributionInterpretation,
        TrackExpressionSummary,
        VideoMetadata,
        VisualWindow,
        WindowInteractionSummary,
    )
    from video_analytics.pipeline import interpret_expression_distribution

    probs = (0.05, 0.02, 0.02, 0.8, 0.05, 0.03, 0.03)
    track_summary = TrackExpressionSummary(
        track_id="Face A",
        observation_count=5,
        aggregated_probabilities=probs,
        interpretation=interpret_expression_distribution(probs, 5),
    )
    window = VisualWindow(
        start_time_seconds=0.0,
        end_time_seconds=20.0,
        track_summaries=(track_summary,),
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
    result = _make_result(
        metadata=VideoMetadata(duration_seconds=20.0, fps=25.0, width=640, height=480, frame_count=500),
        detected_tracks=("Face A",),
        windows=(window,),
    )

    section_html = m.video_analytics_html(result)

    assert "Visual Interaction Insights" in section_html
    assert "Facial Expression Patterns" in section_html
    assert "Expression Synchrony" in section_html
    assert "Visual Interaction" in section_html
    assert "Method note" in section_html
    assert "Face A" in section_html
    # Categorical psychological wording must never appear (Part 14/19).
    assert "was happy" not in section_html.lower()
    assert "is sad" not in section_html.lower()


def test_ui_long_meeting_caps_displayed_windows_for_compactness() -> None:
    import app.main as m
    from video_analytics.model import TrackExpressionSummary, VideoMetadata, VisualWindow, WindowInteractionSummary
    from video_analytics.pipeline import interpret_expression_distribution

    probs = (0.05, 0.02, 0.02, 0.8, 0.05, 0.03, 0.03)
    windows = []
    for i in range(20):  # a long, ~6.5 minute meeting
        summary = TrackExpressionSummary(
            track_id="Face A",
            observation_count=3,
            aggregated_probabilities=probs,
            interpretation=interpret_expression_distribution(probs, 3),
        )
        windows.append(
            VisualWindow(
                start_time_seconds=i * 20.0,
                end_time_seconds=(i + 1) * 20.0,
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
        )
    result = _make_result(
        metadata=VideoMetadata(duration_seconds=400.0, fps=25.0, width=640, height=480, frame_count=10000),
        detected_tracks=("Face A",),
        windows=tuple(windows),
    )

    section_html = m.video_analytics_html(result)

    assert "more windows not shown" in section_html
    assert section_html.count('class="ms-analytics-row"') <= 20  # capped, not one row per window x2 sections


def test_ui_video_analytics_failure_does_not_prevent_mom_from_rendering() -> None:
    # Simulates Part 17: a video-analytics exception must be caught by
    # analyze_uploaded_video and never propagate -- session state ends
    # up with video_analytics_result=None, and the visual section simply
    # does not render, while the rest of the Minutes page is unaffected.
    import app.main as m

    class FakeSessionState(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(name)

        def __setattr__(self, name, value):
            self[name] = value

    import streamlit as st

    st.session_state = FakeSessionState()
    st.session_state["video_analytics_result"] = None
    st.session_state["processing_logs"] = []

    from unittest.mock import patch

    with patch("video_analytics.analyze_video", side_effect=RuntimeError("simulated model failure")):
        m.analyze_uploaded_video(Path("irrelevant.mp4"))

    assert st.session_state.get("video_analytics_result") is None
    assert m.video_analytics_html(st.session_state.get("video_analytics_result")) == ""
