"""SELaD Phase 4 -- lazy model download/cache helper.

Shared by faces.py, landmarks.py, and expressions.py. Model weights
(~39MB total: two MediaPipe Tasks models + one ONNX classifier) are
NOT committed to git -- they are downloaded once into a gitignored
local cache directory on first use, then reused.

Every download is wrapped so that a network failure or corrupted
partial file never raises past this module's callers into the
factual MoM pipeline (Part 17) -- callers receive ``None`` and are
expected to degrade to ``available=False``.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from urllib.error import URLError

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).resolve().parent / ".model_cache"

_MODEL_SOURCES: dict[str, tuple[str, int]] = {
    "blaze_face_short_range.tflite": (
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
        200_000,
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task",
        3_000_000,
    ),
    "emotion-ferplus.onnx": (
        "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/"
        "emotion_ferplus/model/emotion-ferplus-8.onnx",
        30_000_000,
    ),
}
"""filename -> (download URL, minimum expected byte size for a sanity
check against a truncated/HTML-error-page download)."""


def get_model_path(model_name: str) -> Path | None:
    """Return a local path to ``model_name``, downloading it into the
    cache directory on first use. Returns ``None`` (never raises) if
    the model is unknown, the download fails, or the downloaded file
    fails a basic size sanity check."""

    if model_name not in _MODEL_SOURCES:
        logger.error("Unknown video-analytics model requested: %s", model_name)
        return None

    url, min_size = _MODEL_SOURCES[model_name]
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target_path = _CACHE_DIR / model_name

    if target_path.exists() and target_path.stat().st_size >= min_size:
        return target_path

    tmp_path = target_path.with_suffix(target_path.suffix + ".partial")
    try:
        logger.info("Downloading video-analytics model: %s", model_name)
        urllib.request.urlretrieve(url, str(tmp_path))
        if tmp_path.stat().st_size < min_size:
            raise OSError(
                f"Downloaded file for {model_name} is smaller than expected "
                f"({tmp_path.stat().st_size} bytes) -- likely a failed/partial download."
            )
        tmp_path.replace(target_path)
        return target_path
    except (URLError, OSError, TimeoutError) as exc:
        logger.warning("Could not download video-analytics model %s: %s", model_name, exc)
        tmp_path.unlink(missing_ok=True)
        return None
