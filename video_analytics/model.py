"""SELaD Phase 4 -- typed data model for visual (video) analytics.

This module defines the data shapes for MeetScribe's video analytics
branch: a functional adaptation of the visual-sensing methodology in
"Why stressed, Mom?: Exploring Family Reflection on Social and Emotional
Sensor Data through Family Informatics" (SELaD), applied to
professional/team meetings instead of family conversations.

Architectural isolation (see video_analytics/__init__.py for the full
policy): this module is never imported by transcription/, ml_mom/,
meeting_analytics/, or integrations/voxels_ser.py, and never writes back
into any of their data structures. It has no dependency on the factual
MoM pipeline at all -- video analytics is purely additive, sensing-layer
evidence.

Canonical 7 expression classes (matching SELaD's facial-expression
methodology):
    Anger, Disgust, Fear, Happiness, Sadness, Surprise, Neutral
"""

from __future__ import annotations

from dataclasses import dataclass, field

EXPRESSION_CLASSES: tuple[str, ...] = (
    "Anger",
    "Disgust",
    "Fear",
    "Happiness",
    "Sadness",
    "Surprise",
    "Neutral",
)
"""The canonical 7-class SELaD-compatible expression set. Any underlying
model's own label set is mapped onto this tuple explicitly (see
video_analytics/expressions.py's MODEL_CLASS_MAP) -- never assumed to
already match."""


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Inspected properties of the uploaded video file.

    ``fps``/``frame_count`` are None (never fabricated) when the
    container does not reliably report them -- see
    video_analytics/preprocessing.py.
    """

    duration_seconds: float
    fps: float | None
    width: int
    height: int
    frame_count: int | None


@dataclass(frozen=True, slots=True)
class DetectedFace:
    """One face detection at one sampled frame."""

    timestamp_seconds: float
    frame_index: int
    track_id: str
    """Stable LOCAL track identifier ("Face A", "Face B", ...) -- never a
    real name and never biometric identity. See video_analytics/faces.py."""

    bounding_box: tuple[float, float, float, float]
    """(x, y, width, height) in normalized [0, 1] image coordinates."""

    detection_confidence: float | None


@dataclass(frozen=True, slots=True)
class FacialExpressionObservation:
    """One facial-expression classification for one detected face."""

    timestamp_seconds: float
    track_id: str
    probabilities: tuple[float, ...]
    """Probability for each of EXPRESSION_CLASSES, in that order. Always
    the FULL distribution -- never only the argmax (see Part 8)."""

    predicted_class: str
    confidence: float
    """The predicted class's own probability (== max(probabilities))."""


@dataclass(frozen=True, slots=True)
class FaceLandmarkObservation:
    """Normalized facial-landmark evidence for one detected face, used
    only as input to interaction/gaze approximation -- never rendered
    directly in the UI (Part 11)."""

    timestamp_seconds: float
    track_id: str
    left_eye: tuple[float, float]
    right_eye: tuple[float, float]
    nose_tip: tuple[float, float]
    face_yaw_estimate: float | None
    """Coarse left/right head-turn estimate in degrees, derived from eye/
    nose landmark geometry -- an approximation, not a calibrated pose
    estimate. None when geometry is insufficient to estimate."""

    landmark_count: int
    """How many raw landmark points were available for this observation
    (informational/quality only)."""


@dataclass(frozen=True, slots=True)
class ExpressionDistributionInterpretation:
    """Shared uncertainty-aware judgment for an aggregated expression
    distribution -- same philosophy as
    meeting_analytics.communication.interpret_emotion_distribution, kept
    as an independent copy here since video analytics must not import
    from the audio Communication Signals module (isolation, Part 2)."""

    status: str
    """"clear", "mixed", or "unavailable"."""

    top_class: str | None
    top_probability: float | None
    second_class: str | None
    second_probability: float | None
    margin: float | None
    display_pattern: str
    """Human-facing phrase, e.g. "Most frequently observed facial-
    expression class: Happiness" or "No clearly dominant facial-
    expression pattern was observed."."""


@dataclass(frozen=True, slots=True)
class TrackExpressionSummary:
    """One face track's aggregated expression evidence within one
    20-second window."""

    track_id: str
    observation_count: int
    aggregated_probabilities: tuple[float, ...]
    """Mean of all observations' probability vectors for this track in
    this window. Empty tuple when observation_count == 0."""

    interpretation: ExpressionDistributionInterpretation


@dataclass(frozen=True, slots=True)
class PairwiseSynchrony:
    """Facial-expression-pattern similarity between two tracks within one
    window. Similarity only -- see the module docstring in synchrony.py
    for the explicit semantic disclaimer (never agreement/happiness/team
    health)."""

    track_id_a: str
    track_id_b: str
    similarity: float
    """1 / (1 + euclidean_distance) -- in (0, 1], higher means more
    similar aggregated expression distributions."""


@dataclass(frozen=True, slots=True)
class WindowInteractionSummary:
    """Approximate visual-interaction/orientation evidence within one
    20-second window. Never called definitive eye contact (Part 12)."""

    available: bool
    unavailable_reason: str | None
    tracks_with_geometry: int
    mutual_orientation_score: float | None
    """A conservative [0, 1] approximation of how often sampled tracks'
    estimated head/eye orientation pointed toward each other during this
    window. None when unavailable."""

    display_label: str
    """E.g. "Estimated eye-contact / visual-interaction cue" or
    "Visual interaction cue unavailable for this window."."""


@dataclass(frozen=True, slots=True)
class VisualWindow:
    """One 20-second temporal aggregation window (Part 10)."""

    start_time_seconds: float
    end_time_seconds: float
    track_summaries: tuple[TrackExpressionSummary, ...]
    pairwise_synchrony: tuple[PairwiseSynchrony, ...]
    aggregate_synchrony: float | None
    """Mean of pairwise_synchrony values; None when fewer than 2 tracks
    have sufficient evidence in this window."""

    synchrony_participant_count: int
    interaction: WindowInteractionSummary
    coverage_note: str | None
    """E.g. "Only 2 of 8 sampled frames in this window contained a
    detectable face." -- None when coverage was unremarkable."""


@dataclass(frozen=True, slots=True)
class VideoAnalyticsResult:
    """Top-level Phase 4 result. ``available=False`` covers every failure
    path (unsupported video, model unavailable, processing error) so
    callers never have to guess from partially-populated fields (Part 17:
    a video-analytics failure must never affect the factual MoM result)."""

    available: bool
    unavailable_reason: str | None
    metadata: VideoMetadata | None
    sampled_frame_count: int
    detected_tracks: tuple[str, ...]
    windows: tuple[VisualWindow, ...]
    quality_warnings: tuple[str, ...]
    limitations: tuple[str, ...]
    """Standing, always-shown disclaimers (Part 19), not per-run
    warnings -- see video_analytics.pipeline.STANDARD_LIMITATIONS."""

    processing_error: str | None = None
    """A short, safe (no stack trace) description of what failed, for
    optional display/logging -- None on success or graceful unavailability
    without an exception."""
