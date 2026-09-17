"""SELaD Phase 4 -- FaceMesh landmark extraction (Part 11).

Uses MediaPipe Tasks' FaceLandmarker (the current equivalent of the
legacy FaceMesh solution; mediapipe's installed version in this
environment does not expose ``mp.solutions``) to extract up to 468
normalized face landmarks per detected face.

Landmarks are used ONLY as evidence for the visual-interaction/gaze
approximation in interaction.py -- they are never rendered as raw
coordinate dumps in the UI (Part 11).
"""

from __future__ import annotations

import logging

import numpy as np

from video_analytics.model import FaceLandmarkObservation
from video_analytics.models import get_model_path

logger = logging.getLogger(__name__)

# Standard MediaPipe FaceMesh landmark indices for the features we need.
_LEFT_EYE_INDEX = 33
_RIGHT_EYE_INDEX = 263
_NOSE_TIP_INDEX = 1


def _load_landmarker():
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
    except ImportError:
        logger.warning("mediapipe is unavailable; landmark extraction disabled.")
        return None, None

    model_path = get_model_path("face_landmarker.task")
    if model_path is None:
        return None, None

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        num_faces=1,
        min_face_detection_confidence=0.5,
    )
    try:
        landmarker = vision.FaceLandmarker.create_from_options(options)
    except Exception:
        logger.exception("Failed to initialize MediaPipe FaceLandmarker.")
        return None, None
    return mp, landmarker


def _estimate_yaw(left_eye: tuple[float, float], right_eye: tuple[float, float], nose_tip: tuple[float, float]) -> float | None:
    """Coarse left/right head-turn approximation from eye/nose geometry:
    how far the nose tip sits off the eye-to-eye midpoint, relative to
    inter-eye distance. Not a calibrated 3D pose estimate -- purely an
    ordinal signal for interaction.py's orientation heuristic."""

    eye_dx = right_eye[0] - left_eye[0]
    eye_dy = right_eye[1] - left_eye[1]
    inter_eye_distance = (eye_dx**2 + eye_dy**2) ** 0.5
    if inter_eye_distance <= 1e-6:
        return None

    midpoint_x = (left_eye[0] + right_eye[0]) / 2
    offset = (nose_tip[0] - midpoint_x) / inter_eye_distance
    return float(np.clip(offset * 90.0, -90.0, 90.0))


class LandmarkExtractor:
    """Cacheable wrapper around the FaceLandmarker Tasks API session."""

    def __init__(self) -> None:
        self._mp = None
        self._landmarker = None
        self._load_attempted = False

    @property
    def available(self) -> bool:
        return self._landmarker is not None

    def load(self) -> bool:
        if self._load_attempted:
            return self._landmarker is not None
        self._load_attempted = True
        self._mp, self._landmarker = _load_landmarker()
        return self._landmarker is not None

    def extract(
        self,
        frame_bgr: np.ndarray,
        bounding_box: tuple[float, float, float, float],
        timestamp_seconds: float,
        track_id: str,
    ) -> FaceLandmarkObservation | None:
        """Extract eye/nose landmarks and a coarse yaw estimate for one
        detected face's crop. Returns None (never raises) if landmarks
        are unavailable or insufficiently confident."""

        if not self.load() or self._landmarker is None:
            return None

        height, width = frame_bgr.shape[:2]
        x, y, w, h = bounding_box
        pad = 0.2
        x1 = max(0, int((x - pad * w) * width))
        y1 = max(0, int((y - pad * h) * height))
        x2 = min(width, int((x + w + pad * w) * width))
        y2 = min(height, int((y + h + pad * h) * height))
        if x2 <= x1 or y2 <= y1:
            return None

        import cv2

        crop_bgr = frame_bgr[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=crop_rgb)

        try:
            result = self._landmarker.detect(mp_image)
        except Exception:
            logger.exception("Landmark extraction failed for a face crop.")
            return None

        if not result.face_landmarks:
            return None

        points = result.face_landmarks[0]
        landmark_count = len(points)
        if landmark_count <= max(_LEFT_EYE_INDEX, _RIGHT_EYE_INDEX, _NOSE_TIP_INDEX):
            return None

        left_eye = (points[_LEFT_EYE_INDEX].x, points[_LEFT_EYE_INDEX].y)
        right_eye = (points[_RIGHT_EYE_INDEX].x, points[_RIGHT_EYE_INDEX].y)
        nose_tip = (points[_NOSE_TIP_INDEX].x, points[_NOSE_TIP_INDEX].y)
        yaw_estimate = _estimate_yaw(left_eye, right_eye, nose_tip)

        return FaceLandmarkObservation(
            timestamp_seconds=timestamp_seconds,
            track_id=track_id,
            left_eye=left_eye,
            right_eye=right_eye,
            nose_tip=nose_tip,
            face_yaw_estimate=yaw_estimate,
            landmark_count=landmark_count,
        )
