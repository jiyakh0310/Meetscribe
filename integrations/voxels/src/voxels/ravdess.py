"""RAVDESS metadata parsing for Voxels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from voxels.labels import CANONICAL_LABELS, map_ravdess_emotion

RAVDESS_FILENAME_PARTS = 7
AUDIO_ONLY_MODALITY = "03"
SPEECH_VOCAL_CHANNEL = "01"
WAV_SUFFIX = ".wav"

METADATA_COLUMNS: tuple[str, ...] = (
    "file_path",
    "filename",
    "modality",
    "vocal_channel",
    "emotion_code",
    "original_emotion",
    "emotion",
    "intensity",
    "statement",
    "repetition",
    "actor_id",
    "actor_gender",
)

SKIPPED_COLUMNS: tuple[str, ...] = ("file_path", "filename", "reason")


@dataclass(frozen=True)
class MetadataResult:
    """Parsed metadata plus skipped-file diagnostics."""

    metadata: pd.DataFrame
    skipped: pd.DataFrame


def actor_gender(actor_id: int) -> str:
    """Return RAVDESS documented actor gender from odd/even actor ID."""
    return "male" if actor_id % 2 == 1 else "female"


def _skip_record(path: Path, reason: str) -> dict[str, str]:
    return {
        "file_path": str(path),
        "filename": path.name,
        "reason": reason,
    }


def parse_ravdess_filename(path: Path) -> dict[str, object]:
    """Parse one valid RAVDESS speech WAV filename into metadata fields."""
    if path.suffix.lower() != WAV_SUFFIX:
        raise ValueError("unsupported file extension")

    parts = path.stem.split("-")
    if len(parts) != RAVDESS_FILENAME_PARTS:
        raise ValueError("malformed RAVDESS filename")

    modality, vocal_channel, emotion_code, intensity, statement, repetition, actor = parts
    if modality != AUDIO_ONLY_MODALITY:
        raise ValueError("not an audio-only modality file")
    if vocal_channel != SPEECH_VOCAL_CHANNEL:
        raise ValueError("not a speech vocal-channel file")

    if not actor.isdigit():
        raise ValueError("actor ID is not numeric")

    original_emotion, emotion = map_ravdess_emotion(emotion_code)
    if emotion not in CANONICAL_LABELS:
        raise ValueError(f"mapped emotion is not canonical: {emotion}")

    actor_id = int(actor)
    return {
        "file_path": str(path.resolve()),
        "filename": path.name,
        "modality": modality,
        "vocal_channel": vocal_channel,
        "emotion_code": emotion_code,
        "original_emotion": original_emotion,
        "emotion": emotion,
        "intensity": intensity,
        "statement": statement,
        "repetition": repetition,
        "actor_id": actor_id,
        "actor_gender": actor_gender(actor_id),
    }


def build_metadata_from_paths(paths: Iterable[Path]) -> MetadataResult:
    """Build metadata from candidate paths and record skipped-file reasons."""
    records: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []
    seen_paths: set[Path] = set()

    for path in paths:
        resolved_path = path.resolve()
        if resolved_path in seen_paths:
            skipped.append(_skip_record(path, "duplicate resolved path"))
            continue
        seen_paths.add(resolved_path)

        try:
            records.append(parse_ravdess_filename(path))
        except ValueError as exc:
            skipped.append(_skip_record(path, str(exc)))

    metadata = pd.DataFrame.from_records(records, columns=METADATA_COLUMNS)
    skipped_df = pd.DataFrame.from_records(skipped, columns=SKIPPED_COLUMNS)
    if not metadata.empty:
        metadata = metadata.sort_values(["actor_id", "filename"]).reset_index(drop=True)
    return MetadataResult(metadata=metadata, skipped=skipped_df)


def scan_ravdess_metadata(data_dir: Path | str) -> MetadataResult:
    """Recursively scan a RAVDESS folder and parse valid speech WAV metadata."""
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"RAVDESS data directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"RAVDESS data path is not a directory: {root}")

    candidate_paths = sorted(
        (path for path in root.rglob("*") if path.is_file() and not path.name.startswith(".")),
        key=lambda p: str(p),
    )
    return build_metadata_from_paths(candidate_paths)


def save_metadata(result: MetadataResult, output_path: Path | str) -> Path:
    """Save parsed metadata CSV and return the written path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.metadata.to_csv(path, index=False)
    return path


def format_distribution_report(result: MetadataResult) -> str:
    """Format observed dataset distributions for command-line output."""
    lines = [
        f"Observed valid speech WAV files: {len(result.metadata)}",
        f"Skipped files: {len(result.skipped)}",
    ]

    if result.metadata.empty:
        lines.append("No valid metadata rows were produced.")
        return "\n".join(lines)

    distributions = {
        "emotion": result.metadata["emotion"].value_counts().sort_index(),
        "original_emotion": result.metadata["original_emotion"].value_counts().sort_index(),
        "actor_gender": result.metadata["actor_gender"].value_counts().sort_index(),
        "actor_id": result.metadata["actor_id"].value_counts().sort_index(),
    }

    for name, series in distributions.items():
        lines.append("")
        lines.append(f"{name} distribution:")
        for key, value in series.items():
            lines.append(f"  {key}: {value}")

    if not result.skipped.empty:
        lines.append("")
        lines.append("Skipped-file reasons:")
        for reason, count in result.skipped["reason"].value_counts().sort_index().items():
            lines.append(f"  {reason}: {count}")

    return "\n".join(lines)
