"""MeetScribe adapter for the existing Voxels speech-emotion predictor.

This module deliberately keeps emotion analysis separate from the transcript and
meeting-minutes pipeline.  It loads the Voxels predictor lazily, analyzes short
windows from the already-normalized audio file, and returns a serializable result
that can be stored independently in Streamlit session state.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VOXELS_ROOT = PROJECT_ROOT / "integrations" / "voxels"
VOXELS_SOURCE = VOXELS_ROOT / "src"
DEFAULT_CHECKPOINT = VOXELS_ROOT / "models" / "wav2vec2_real_upper_layers"
VOXELS_CHECKPOINT_ENV = "VOXELS_CHECKPOINT_PATH"

# The Voxels checkpoint was trained on short speech clips.  Bounded windows keep
# inference compatible with that model even when a meeting recording is long.
WINDOW_SECONDS = 4.0
MAX_WINDOWS = 60
EMOTIONS = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")

_predictor: Any | None = None


def _checkpoint_path() -> Path:
    configured = os.environ.get(VOXELS_CHECKPOINT_ENV, "").strip()
    if not configured:
        return DEFAULT_CHECKPOINT
    path = Path(configured)
    return path if path.is_absolute() else VOXELS_ROOT / path


def _get_predictor() -> Any:
    """Load the bundled Voxels predictor once, only when audio analysis is used."""
    global _predictor
    if _predictor is not None:
        return _predictor
    source = str(VOXELS_SOURCE)
    if source not in sys.path:
        sys.path.insert(0, source)

    from voxels.audio import AudioPreprocessorConfig
    from voxels.inference import ConfidenceConfig, EmotionPredictor

    _predictor = EmotionPredictor.from_checkpoint(
        _checkpoint_path(),
        audio_config=AudioPreprocessorConfig(
            target_sample_rate=16000,
            min_duration_seconds=0.5,
            max_duration_seconds=10.0,
            near_silence_rms=1e-4,
        ),
        confidence_config=ConfidenceConfig(),
    )
    return _predictor


def _tone_for(emotion: str) -> str:
    normalized = emotion.strip().lower()
    if normalized == "happy":
        return "Positive"
    if normalized == "neutral":
        return "Neutral"
    if normalized in {"angry", "disgust", "fear", "sad"}:
        return "Negative"
    if normalized == "surprise":
        return "Mixed / Uncertain"
    return "Mixed / Uncertain"


def _window_starts(frame_count: int, sample_rate: int) -> list[int]:
    window_frames = max(1, int(round(WINDOW_SECONDS * sample_rate)))
    total_windows = max(1, math.ceil(frame_count / window_frames))
    if total_windows <= MAX_WINDOWS:
        return [index * window_frames for index in range(total_windows)]
    # Sample evenly across a long meeting so the result is not biased toward
    # the opening seconds while keeping model work bounded.
    return [
        round(index * (total_windows - 1) / (MAX_WINDOWS - 1)) * window_frames
        for index in range(MAX_WINDOWS)
    ]


def _write_windows(audio_path: Path) -> list[tuple[Path, float, float]]:
    import numpy as np
    import soundfile as sf

    waveform, sample_rate = sf.read(str(audio_path), always_2d=False, dtype="float32")
    samples = np.asarray(waveform, dtype=np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1, dtype=np.float32)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("The normalized audio contains no usable samples.")

    window_frames = max(1, int(round(WINDOW_SECONDS * int(sample_rate))))
    windows: list[tuple[Path, float, float]] = []
    temp_dir = Path(tempfile.mkdtemp(prefix="meetscribe_voxels_"))
    try:
        for index, start in enumerate(_window_starts(int(samples.size), int(sample_rate))):
            chunk = samples[start : start + window_frames]
            if chunk.size == 0:
                continue
            if chunk.size < window_frames:
                chunk = np.pad(chunk, (0, window_frames - chunk.size))
            path = temp_dir / f"window_{index:03d}.wav"
            sf.write(str(path), chunk, int(sample_rate), subtype="PCM_16")
            start_time = start / float(sample_rate)
            end_time = min((start + window_frames) / float(sample_rate), samples.size / float(sample_rate))
            windows.append((path, start_time, end_time))
    except Exception:
        for path, _, _ in windows:
            path.unlink(missing_ok=True)
        temp_dir.rmdir()
        raise
    return windows


def run_voxels_emotion(audio_path: Path | str) -> dict[str, Any]:
    """Run Voxels on the supplied normalized audio and aggregate probabilities.

    The function intentionally raises on model/dependency/input failures.  The
    MeetScribe audio boundary catches that failure and continues its normal STT
    and MoM path, while retaining the diagnostic separately from the report.
    """
    source_path = Path(audio_path)
    predictor = _get_predictor()
    window_specs = _write_windows(source_path)
    if not window_specs:
        raise ValueError("No Voxels inference windows could be created.")

    totals = {label: 0.0 for label in EMOTIONS}
    confidences: list[float] = []
    warnings: list[str] = []
    window_predictions: list[dict[str, Any]] = []
    try:
        for window_path, start_time, end_time in window_specs:
            prediction = predictor.predict_file(window_path)
            probabilities = prediction.probabilities
            for label in EMOTIONS:
                totals[label] += float(probabilities.get(label, 0.0))
            confidences.append(float(prediction.confidence))
            warnings.extend(str(item) for item in prediction.quality_warnings)
            window_predictions.append(
                {
                    "start_time": float(start_time),
                    "end_time": float(end_time),
                    "predicted_emotion": str(prediction.predicted_emotion),
                    "confidence": float(prediction.confidence),
                    "probabilities": {
                        label: float(probabilities.get(label, 0.0)) for label in EMOTIONS
                    },
                }
            )
    finally:
        window_dir = window_specs[0][0].parent
        for path, _, _ in window_specs:
            path.unlink(missing_ok=True)
        window_dir.rmdir()

    count = len(window_specs)
    probabilities = {label: value / count for label, value in totals.items()}
    dominant = max(probabilities, key=probabilities.get)
    confidence = probabilities[dominant]
    tone = _tone_for(dominant)
    if confidence >= 0.75:
        observation = f"The sampled speech segments show a consistent {dominant.title().lower()} pattern."
    else:
        observation = "The sampled speech segments show a mixed or uncertain emotion pattern."

    return {
        "available": True,
        "dominant_emotion": dominant.title(),
        "confidence": float(confidence),
        "probabilities": probabilities,
        "tone": tone,
        "observation": observation,
        "windows": window_predictions,
        "windows_analyzed": count,
        "average_window_confidence": sum(confidences) / len(confidences),
        "quality_warnings": sorted(set(warnings)),
        "note": (
            "Detected speech-emotion patterns only; this is not a measure of a person's "
            "true psychological state."
        ),
    }
