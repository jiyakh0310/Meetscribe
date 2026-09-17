"""SELaD Phase 4 -- facial-expression classification (Parts 8-9).

Uses the ONNX Model Zoo's pretrained ``emotion-ferplus`` model (~35MB)
via ONNX Runtime, chosen specifically to avoid pulling in a large,
unrelated deep-learning framework (e.g. TensorFlow) when a small,
reliable, pretrained classifier is sufficient (no training is
performed here -- Part 8).

Model: emotion-ferplus-8.onnx (ONNX Model Zoo, FER+ dataset).
Input:  "Input3", shape [1, 1, 64, 64] -- single-channel (grayscale),
        64x64 pixel face crop.
Output: "Plus692_Output_0", shape [1, 8] -- raw (pre-softmax) logits
        over 8 FER+ classes, in this fixed order:
        neutral, happiness, surprise, sadness, anger, disgust, fear,
        contempt.

SELaD's methodology uses 7 canonical classes (video_analytics.model.
EXPRESSION_CLASSES): Anger, Disgust, Fear, Happiness, Sadness,
Surprise, Neutral. FER+'s "contempt" has no SELaD equivalent and is
dropped; the remaining 7 raw probabilities are re-normalized (after
softmax) so they still sum to 1.0 over exactly the canonical set.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from video_analytics.model import EXPRESSION_CLASSES, FacialExpressionObservation
from video_analytics.models import get_model_path

logger = logging.getLogger(__name__)

_FERPLUS_CLASS_ORDER = (
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
)

_FERPLUS_TO_CANONICAL: dict[str, str] = {
    "anger": "Anger",
    "disgust": "Disgust",
    "fear": "Fear",
    "happiness": "Happiness",
    "sadness": "Sadness",
    "surprise": "Surprise",
    "neutral": "Neutral",
}
"""Explicit FER+ label -> SELaD canonical label mapping (Part 8).
"contempt" is intentionally absent -- it is dropped, never mapped."""

_INPUT_SIZE = 64


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


class ExpressionClassifier:
    """Thin, cacheable wrapper around the ONNX Runtime session. Callers
    should create ONE instance per pipeline run (and may cache it across
    runs via Streamlit's resource cache) rather than reloading the model
    per frame (Part 18)."""

    def __init__(self) -> None:
        self._session = None
        self._input_name: str | None = None
        self._output_name: str | None = None

    @property
    def available(self) -> bool:
        return self._session is not None

    def load(self) -> bool:
        if self._session is not None:
            return True
        model_path = get_model_path("emotion-ferplus.onnx")
        if model_path is None:
            return False
        try:
            import onnxruntime as ort

            self._session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
            return True
        except Exception:
            logger.exception("Failed to load emotion-ferplus ONNX model.")
            self._session = None
            return False

    def classify_face_crop(
        self,
        frame_bgr: np.ndarray,
        bounding_box: tuple[float, float, float, float],
        timestamp_seconds: float,
        track_id: str,
    ) -> FacialExpressionObservation | None:
        """Classify one face crop. Returns None (never raises) if the
        model is unavailable or the crop is degenerate."""

        if not self.load() or self._session is None:
            return None

        height, width = frame_bgr.shape[:2]
        x, y, w, h = bounding_box
        x1 = max(0, int(x * width))
        y1 = max(0, int(y * height))
        x2 = min(width, int((x + w) * width))
        y2 = min(height, int((y + h) * height))
        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame_bgr[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (_INPUT_SIZE, _INPUT_SIZE), interpolation=cv2.INTER_AREA)
        input_tensor = resized.astype(np.float32).reshape(1, 1, _INPUT_SIZE, _INPUT_SIZE)

        try:
            outputs = self._session.run([self._output_name], {self._input_name: input_tensor})
        except Exception:
            logger.exception("ONNX inference failed for a face crop.")
            return None

        raw_logits = outputs[0][0]
        ferplus_probabilities = _softmax(raw_logits)

        canonical_probabilities: dict[str, float] = {cls: 0.0 for cls in EXPRESSION_CLASSES}
        for label, probability in zip(_FERPLUS_CLASS_ORDER, ferplus_probabilities):
            canonical_label = _FERPLUS_TO_CANONICAL.get(label)
            if canonical_label is not None:
                canonical_probabilities[canonical_label] = float(probability)

        total = sum(canonical_probabilities.values())
        if total <= 0:
            return None
        normalized = tuple(canonical_probabilities[cls] / total for cls in EXPRESSION_CLASSES)

        top_index = max(range(len(normalized)), key=lambda i: normalized[i])
        return FacialExpressionObservation(
            timestamp_seconds=timestamp_seconds,
            track_id=track_id,
            probabilities=normalized,
            predicted_class=EXPRESSION_CLASSES[top_index],
            confidence=normalized[top_index],
        )
