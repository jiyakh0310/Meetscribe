"""SELaD Phase 4 -- top-level video analytics pipeline (Parts 4-19).

Public entry point: ``analyze_video(video_path) -> VideoAnalyticsResult``.

This is the ONE place that orchestrates preprocessing -> sampling ->
face detection/tracking -> expression classification -> landmark
extraction -> 20-second windowing -> synchrony -> visual interaction.
Every internal failure is caught HERE so that a video-analytics error
can never propagate into or affect the factual MoM pipeline (Part 17):
callers always receive a ``VideoAnalyticsResult``, never an exception.

Video analytics is purely additive evidence, not a diagnostic or
scoring tool -- see STANDARD_LIMITATIONS below for the required
standing disclaimer (Part 19).
"""

from __future__ import annotations

import logging
from pathlib import Path

from video_analytics.expressions import ExpressionClassifier
from video_analytics.faces import detect_faces
from video_analytics.interaction import compute_window_interaction
from video_analytics.landmarks import LandmarkExtractor
from video_analytics.model import (
    EXPRESSION_CLASSES,
    ExpressionDistributionInterpretation,
    TrackExpressionSummary,
    VideoAnalyticsResult,
    VisualWindow,
)
from video_analytics.preprocessing import (
    VideoProcessingError,
    inspect_video_metadata,
    normalize_video_for_analysis,
)
from video_analytics.sampling import sample_frame_indices
from video_analytics.synchrony import compute_window_synchrony

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 20.0
"""SELaD's temporal aggregation window length (Part 10)."""

STANDARD_LIMITATIONS: tuple[str, ...] = (
    "Visual-expression and interaction signals are observational cues derived from "
    "sampled video frames. They may be affected by camera angle, lighting, occlusion "
    "and participant position, and should not be interpreted as definitive emotional "
    "or psychological states.",
    "Face tracks are locally-assigned labels, not identity recognition -- they are not "
    "necessarily linked to transcript speaker labels.",
    "Facial-expression synchrony reflects similarity of observed expression patterns "
    "only; it does not indicate agreement, happiness, team health, or psychological "
    "compatibility.",
    "The visual-interaction cue is a conservative approximation of mutual visual "
    "orientation, not laboratory-grade eye tracking.",
)

_CLEAR_TOP_PROBABILITY_THRESHOLD = 0.40
_CLEAR_MARGIN_THRESHOLD = 0.15
_MIN_OBSERVATIONS_FOR_DOMINANT_CLASS = 2


def _unavailable(reason: str, processing_error: str | None = None) -> VideoAnalyticsResult:
    return VideoAnalyticsResult(
        available=False,
        unavailable_reason=reason,
        metadata=None,
        sampled_frame_count=0,
        detected_tracks=(),
        windows=(),
        quality_warnings=(),
        limitations=STANDARD_LIMITATIONS,
        processing_error=processing_error,
    )


def interpret_expression_distribution(
    aggregated_probabilities: tuple[float, ...],
    observation_count: int,
) -> ExpressionDistributionInterpretation:
    """Independent reimplementation of the same uncertainty-aware
    judgment as meeting_analytics.communication.interpret_emotion_distribution
    (Part 14) -- NOT imported from there, since video analytics must
    stay isolated from the audio Communication Signals module.

    "clear" requires: enough observations, a leading class probability
    >= _CLEAR_TOP_PROBABILITY_THRESHOLD, AND a margin over the runner-up
    >= _CLEAR_MARGIN_THRESHOLD. Otherwise "mixed". No observations at
    all is "unavailable".
    """

    if observation_count == 0 or not aggregated_probabilities:
        return ExpressionDistributionInterpretation(
            status="unavailable",
            top_class=None,
            top_probability=None,
            second_class=None,
            second_probability=None,
            margin=None,
            display_pattern="No facial-expression evidence is available for this window.",
        )

    ranked = sorted(
        zip(EXPRESSION_CLASSES, aggregated_probabilities),
        key=lambda item: (-item[1], item[0]),
    )
    top_class, top_probability = ranked[0]
    second_class, second_probability = ranked[1] if len(ranked) > 1 else (None, None)
    margin = (top_probability - second_probability) if second_probability is not None else None

    is_clear = (
        observation_count >= _MIN_OBSERVATIONS_FOR_DOMINANT_CLASS
        and top_probability >= _CLEAR_TOP_PROBABILITY_THRESHOLD
        and (margin is None or margin >= _CLEAR_MARGIN_THRESHOLD)
    )

    if is_clear:
        return ExpressionDistributionInterpretation(
            status="clear",
            top_class=top_class,
            top_probability=top_probability,
            second_class=second_class,
            second_probability=second_probability,
            margin=margin,
            display_pattern=f"Most frequently observed facial-expression class: {top_class}",
        )

    return ExpressionDistributionInterpretation(
        status="mixed",
        top_class=top_class,
        top_probability=top_probability,
        second_class=second_class,
        second_probability=second_probability,
        margin=margin,
        display_pattern="No clearly dominant facial-expression pattern was observed.",
    )


def analyze_video(video_path: Path) -> VideoAnalyticsResult:
    """Run the full Phase 4 visual-analytics pipeline on ``video_path``.

    Never raises. On any failure, returns
    ``VideoAnalyticsResult(available=False, ...)`` with a safe,
    stack-trace-free ``processing_error`` string.
    """

    video_path = Path(video_path)
    normalized_path: Path | None = None
    is_temp_normalized = False

    try:
        try:
            normalized_path, is_temp_normalized = normalize_video_for_analysis(video_path)
        except VideoProcessingError as exc:
            return _unavailable(str(exc))

        metadata = inspect_video_metadata(normalized_path)
        sampled_frames = sample_frame_indices(
            fps=metadata.fps,
            frame_count=metadata.frame_count,
            duration_seconds=metadata.duration_seconds,
        )
        if not sampled_frames:
            return _unavailable("No frames could be sampled from this video.")

        detected_faces, frame_crops, face_quality_warnings = detect_faces(
            normalized_path, sampled_frames
        )
        quality_warnings = list(face_quality_warnings)

        if not detected_faces:
            return VideoAnalyticsResult(
                available=True,
                unavailable_reason=None,
                metadata=metadata,
                sampled_frame_count=len(sampled_frames),
                detected_tracks=(),
                windows=(),
                quality_warnings=tuple(quality_warnings + ["No faces were detected in this video."]),
                limitations=STANDARD_LIMITATIONS,
            )

        classifier = ExpressionClassifier()
        landmark_extractor = LandmarkExtractor()

        expression_observations = []
        landmark_observations = []
        for face in detected_faces:
            frame = frame_crops.get(face.frame_index)
            if frame is None:
                continue
            expression = classifier.classify_face_crop(
                frame, face.bounding_box, face.timestamp_seconds, face.track_id
            )
            if expression is not None:
                expression_observations.append(expression)
            landmark = landmark_extractor.extract(
                frame, face.bounding_box, face.timestamp_seconds, face.track_id
            )
            if landmark is not None:
                landmark_observations.append(landmark)

        del frame_crops  # release full-resolution frames as soon as possible (Part 18)

        detected_tracks = tuple(sorted({face.track_id for face in detected_faces}))
        max_timestamp = max(f.timestamp_seconds for f in sampled_frames)
        window_count = int(max_timestamp // WINDOW_SECONDS) + 1

        windows: list[VisualWindow] = []
        for window_index in range(window_count):
            start = window_index * WINDOW_SECONDS
            end = start + WINDOW_SECONDS

            window_expression_by_track: dict[str, list] = {track: [] for track in detected_tracks}
            for obs in expression_observations:
                if start <= obs.timestamp_seconds < end:
                    window_expression_by_track.setdefault(obs.track_id, []).append(obs)

            track_summaries = []
            for track_id, observations in window_expression_by_track.items():
                if observations:
                    count = len(EXPRESSION_CLASSES)
                    aggregated = tuple(
                        sum(o.probabilities[i] for o in observations) / len(observations)
                        for i in range(count)
                    )
                else:
                    aggregated = ()
                interpretation = interpret_expression_distribution(aggregated, len(observations))
                track_summaries.append(
                    TrackExpressionSummary(
                        track_id=track_id,
                        observation_count=len(observations),
                        aggregated_probabilities=aggregated,
                        interpretation=interpretation,
                    )
                )
            track_summaries.sort(key=lambda s: s.track_id)

            pairwise_synchrony, aggregate_synchrony, synchrony_count = compute_window_synchrony(
                tuple(track_summaries)
            )

            window_landmarks_by_track: dict[str, list] = {}
            for landmark in landmark_observations:
                if start <= landmark.timestamp_seconds < end:
                    window_landmarks_by_track.setdefault(landmark.track_id, []).append(landmark)

            window_faces_by_track: dict[str, list] = {}
            for face in detected_faces:
                if start <= face.timestamp_seconds < end:
                    window_faces_by_track.setdefault(face.track_id, []).append(face)

            track_positions_and_yaws: dict[str, list[tuple[float, float]]] = {}
            for track_id, landmarks_for_track in window_landmarks_by_track.items():
                faces_for_track = {f.timestamp_seconds: f for f in window_faces_by_track.get(track_id, [])}
                samples = []
                for landmark in landmarks_for_track:
                    if landmark.face_yaw_estimate is None:
                        continue
                    face_at_time = faces_for_track.get(landmark.timestamp_seconds)
                    if face_at_time is None:
                        continue
                    x, _y, w, _h = face_at_time.bounding_box
                    horizontal_position = x + w / 2
                    samples.append((horizontal_position, landmark.face_yaw_estimate))
                if samples:
                    track_positions_and_yaws[track_id] = samples

            interaction_summary = compute_window_interaction(track_positions_and_yaws)

            sampled_in_window = [f for f in sampled_frames if start <= f.timestamp_seconds < end]
            frames_with_any_face = len(
                {f.frame_index for f in detected_faces if start <= f.timestamp_seconds < end}
            )
            coverage_note = None
            if sampled_in_window and frames_with_any_face < len(sampled_in_window):
                coverage_note = (
                    f"A face was detected in {frames_with_any_face} of "
                    f"{len(sampled_in_window)} sampled frames in this window."
                )

            windows.append(
                VisualWindow(
                    start_time_seconds=start,
                    end_time_seconds=end,
                    track_summaries=tuple(track_summaries),
                    pairwise_synchrony=pairwise_synchrony,
                    aggregate_synchrony=aggregate_synchrony,
                    synchrony_participant_count=synchrony_count,
                    interaction=interaction_summary,
                    coverage_note=coverage_note,
                )
            )

        return VideoAnalyticsResult(
            available=True,
            unavailable_reason=None,
            metadata=metadata,
            sampled_frame_count=len(sampled_frames),
            detected_tracks=detected_tracks,
            windows=tuple(windows),
            quality_warnings=tuple(quality_warnings),
            limitations=STANDARD_LIMITATIONS,
        )
    except Exception as exc:  # noqa: BLE001 -- Part 17: never propagate to MoM pipeline
        logger.exception("Video analytics pipeline failed unexpectedly.")
        return _unavailable(
            "Visual analytics could not be completed for this video.",
            processing_error=str(exc)[:300],
        )
    finally:
        if is_temp_normalized and normalized_path is not None:
            normalized_path.unlink(missing_ok=True)
