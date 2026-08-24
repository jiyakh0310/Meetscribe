"""Canonical labels and RAVDESS emotion mapping for Voxels."""

from __future__ import annotations

from types import MappingProxyType

CANONICAL_LABELS: tuple[str, ...] = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
)

LABEL2ID = MappingProxyType({label: index for index, label in enumerate(CANONICAL_LABELS)})
ID2LABEL = MappingProxyType({index: label for label, index in LABEL2ID.items()})

RAVDESS_EMOTIONS = MappingProxyType(
    {
        "01": ("Neutral", "neutral"),
        "02": ("Calm", "neutral"),
        "03": ("Happy", "happy"),
        "04": ("Sad", "sad"),
        "05": ("Angry", "angry"),
        "06": ("Fearful", "fear"),
        "07": ("Disgust", "disgust"),
        "08": ("Surprised", "surprise"),
    }
)


def map_ravdess_emotion(emotion_code: str) -> tuple[str, str]:
    """Return the original RAVDESS emotion and canonical Voxels label."""
    try:
        return RAVDESS_EMOTIONS[emotion_code]
    except KeyError as exc:
        raise ValueError(f"Unsupported RAVDESS emotion code: {emotion_code}") from exc
