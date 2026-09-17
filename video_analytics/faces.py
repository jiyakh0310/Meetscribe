"""SELaD Phase 4 -- face detection and conservative local tracking
(Parts 6-7).

Face detection here is NEVER speaker identification. Detected faces
receive neutral local track identifiers ("Face A", "Face B", ...) --
never a real name, and never derived from the transcript's speaker
count. This module implements no biometric face recognition: tracking
is a lightweight bounding-box-overlap heuristic across nearby sampled
frames, sufficient only to say "this appears to be the same visible
face as the previous sampled frame" -- not an identity claim.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from video_analytics.model import DetectedFace
from video_analytics.models import get_model_path
from video_analytics.sampling import SampledFrame

logger = logging.getLogger(__name__)

_MIN_DETECTION_CONFIDENCE = 0.5
_IOU_MATCH_THRESHOLD = 0.3
"""Minimum bounding-box IoU between a face in the current sampled frame
and a face in the previous sampled frame to treat them as the same
local track. Conservative on purpose -- an ambiguous/low-overlap case
starts a NEW track rather than forcing a match (Part 7)."""

_TRACK_LABELS = tuple(f"Face {chr(ord('A') + i)}" for i in range(26))


def _iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1, inter_y1 = max(ax, bx), max(ay, by)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    if intersection <= 0:
        return 0.0

    union = (aw * ah) + (bw * bh) - intersection
    return intersection / union if union > 0 else 0.0


def _load_detector():
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
    except ImportError:
        logger.warning("mediapipe is unavailable; face detection disabled.")
        return None, None

    model_path = get_model_path("blaze_face_short_range.tflite")
    if model_path is None:
        return None, None

    options = vision.FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        min_detection_confidence=_MIN_DETECTION_CONFIDENCE,
    )
    try:
        detector = vision.FaceDetector.create_from_options(options)
    except Exception:
        logger.exception("Failed to initialize MediaPipe FaceDetector.")
        return None, None
    return mp, detector


def detect_faces(
    video_path: Path,
    sampled_frames: list[SampledFrame],
) -> tuple[list[DetectedFace], dict[int, np.ndarray], list[str]]:
    """Detect faces at each sampled frame and associate them into local
    tracks across nearby frames via bounding-box overlap.

    Returns (detected_faces, frame_crops_by_index, quality_warnings).
    ``frame_crops_by_index`` maps sampled frame_index -> the full BGR
    frame image (kept only transiently in memory for the caller to pass
    to expression/landmark extraction; never persisted to session
    state or disk beyond this pipeline run -- Part 18).
    """

    quality_warnings: list[str] = []
    mp, detector = _load_detector()
    if detector is None:
        quality_warnings.append("Face detection model was unavailable; visual analytics were skipped.")
        return [], {}, quality_warnings

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        quality_warnings.append("Video could not be reopened for face detection.")
        return [], {}, quality_warnings

    detected_faces: list[DetectedFace] = []
    frame_crops: dict[int, np.ndarray] = {}
    previous_boxes: list[tuple[str, tuple[float, float, float, float]]] = []
    next_label_index = 0
    frames_with_face = 0

    try:
        for sample in sampled_frames:
            capture.set(cv2.CAP_PROP_POS_FRAMES, sample.frame_index)
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                continue

            height, width = frame_bgr.shape[:2]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            try:
                result = detector.detect(mp_image)
            except Exception:
                logger.exception("Face detection failed for a sampled frame; skipping frame.")
                continue

            current_boxes: list[tuple[str, tuple[float, float, float, float]]] = []
            if result.detections:
                frames_with_face += 1
            for detection in result.detections:
                bbox = detection.bounding_box
                normalized_box = (
                    bbox.origin_x / width,
                    bbox.origin_y / height,
                    bbox.width / width,
                    bbox.height / height,
                )
                confidence = (
                    float(detection.categories[0].score) if detection.categories else None
                )

                best_label: str | None = None
                best_iou = 0.0
                for label, prev_box in previous_boxes:
                    overlap = _iou(normalized_box, prev_box)
                    if overlap > best_iou:
                        best_iou = overlap
                        best_label = label
                if best_label is not None and best_iou >= _IOU_MATCH_THRESHOLD:
                    track_id = best_label
                else:
                    if next_label_index >= len(_TRACK_LABELS):
                        track_id = f"Face {next_label_index + 1}"
                    else:
                        track_id = _TRACK_LABELS[next_label_index]
                    next_label_index += 1

                current_boxes.append((track_id, normalized_box))
                detected_faces.append(
                    DetectedFace(
                        timestamp_seconds=sample.timestamp_seconds,
                        frame_index=sample.frame_index,
                        track_id=track_id,
                        bounding_box=normalized_box,
                        detection_confidence=confidence,
                    )
                )

            if current_boxes:
                frame_crops[sample.frame_index] = frame_bgr
            previous_boxes = current_boxes
    finally:
        capture.release()

    if sampled_frames and frames_with_face / len(sampled_frames) < 0.3:
        quality_warnings.append(
            "A face was detected in fewer than 30% of sampled frames; visual analytics coverage is limited."
        )
    distinct_tracks = {face.track_id for face in detected_faces}
    if len(distinct_tracks) > 4:
        quality_warnings.append(
            "An unusually high number of distinct face tracks were inferred; tracking stability may be poor "
            "(e.g. due to camera movement, occlusion, or lighting changes)."
        )

    return detected_faces, frame_crops, quality_warnings
