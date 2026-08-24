"""Dataset and audio validation for Voxels."""

from __future__ import annotations

import hashlib
import json
import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from voxels.ravdess import METADATA_COLUMNS

AUDIO_COLUMNS: tuple[str, ...] = (
    "readable",
    "validation_status",
    "validation_error",
    "sample_rate",
    "channels",
    "num_samples",
    "duration_seconds",
    "peak_abs_amplitude",
    "rms_amplitude",
    "file_size_bytes",
    "content_hash",
    "near_silent",
    "below_min_duration",
    "above_max_duration",
)


@dataclass(frozen=True)
class ValidationThresholds:
    """Configurable audio validation thresholds."""

    min_duration_seconds: float
    max_duration_seconds: float
    near_silence_rms: float = 1e-4


@dataclass(frozen=True)
class ValidationResult:
    """Audio metadata and summary report."""

    audio_metadata: pd.DataFrame
    report: dict[str, Any]


def _empty_audio_fields(status: str, error: str) -> dict[str, object]:
    return {
        "readable": False,
        "validation_status": status,
        "validation_error": error,
        "sample_rate": None,
        "channels": None,
        "num_samples": None,
        "duration_seconds": None,
        "peak_abs_amplitude": None,
        "rms_amplitude": None,
        "file_size_bytes": None,
        "content_hash": None,
        "near_silent": False,
        "below_min_duration": False,
        "above_max_duration": False,
    }


def _pcm_to_float(samples: bytes, sample_width: int) -> np.ndarray:
    """Convert PCM WAV bytes to float samples in approximately [-1, 1]."""
    if sample_width == 1:
        values = np.frombuffer(samples, dtype=np.uint8).astype(np.float32)
        return (values - 128.0) / 128.0
    if sample_width == 2:
        values = np.frombuffer(samples, dtype="<i2").astype(np.float32)
        return values / 32768.0
    if sample_width == 4:
        values = np.frombuffer(samples, dtype="<i4").astype(np.float64)
        return (values / 2147483648.0).astype(np.float32)
    raise ValueError(f"Unsupported PCM sample width: {sample_width} bytes")


def content_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a SHA-256 hash for a local file without loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_audio_file(path: Path | str, thresholds: ValidationThresholds) -> dict[str, object]:
    """Inspect one WAV file read-only and return validation fields."""
    audio_path = Path(path)
    if not audio_path.exists():
        return _empty_audio_fields("missing", "file does not exist")
    if not audio_path.is_file():
        return _empty_audio_fields("unreadable", "path is not a file")

    file_size = audio_path.stat().st_size
    try:
        with wave.open(str(audio_path), "rb") as handle:
            sample_rate = int(handle.getframerate())
            channels = int(handle.getnchannels())
            frames = int(handle.getnframes())
            sample_width = int(handle.getsampwidth())
            raw_samples = handle.readframes(frames)
        values = _pcm_to_float(raw_samples, sample_width)
    except (EOFError, OSError, ValueError, wave.Error) as exc:
        fields = _empty_audio_fields("unreadable", str(exc))
        fields["file_size_bytes"] = file_size
        return fields

    duration = frames / sample_rate if sample_rate else 0.0
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    rms = float(math.sqrt(float(np.mean(np.square(values))))) if values.size else 0.0

    return {
        "readable": True,
        "validation_status": "readable",
        "validation_error": "",
        "sample_rate": sample_rate,
        "channels": channels,
        "num_samples": frames,
        "duration_seconds": duration,
        "peak_abs_amplitude": peak,
        "rms_amplitude": rms,
        "file_size_bytes": file_size,
        "content_hash": content_hash(audio_path),
        "near_silent": rms <= thresholds.near_silence_rms,
        "below_min_duration": duration < thresholds.min_duration_seconds,
        "above_max_duration": duration > thresholds.max_duration_seconds,
    }


def read_metadata_csv(path: Path | str) -> pd.DataFrame:
    """Read generated metadata CSV and verify its required schema."""
    metadata_path = Path(path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")
    metadata = pd.read_csv(metadata_path)
    missing_columns = [column for column in METADATA_COLUMNS if column not in metadata.columns]
    if missing_columns:
        raise ValueError(f"Metadata CSV is missing required columns: {missing_columns}")
    return metadata


def _is_malformed_record(row: pd.Series) -> bool:
    file_path = row.get("file_path")
    return not isinstance(file_path, str) or not file_path.strip()


def validate_metadata(metadata: pd.DataFrame, thresholds: ValidationThresholds) -> ValidationResult:
    """Inspect all metadata-referenced audio files and build a validation report."""
    records: list[dict[str, object]] = []
    malformed_indices: list[int] = []

    for index, row in metadata.iterrows():
        base_record = row.to_dict()
        if _is_malformed_record(row):
            malformed_indices.append(int(index))
            records.append({**base_record, **_empty_audio_fields("malformed_record", "missing file_path")})
            continue

        audio_fields = inspect_audio_file(Path(str(row["file_path"])), thresholds)
        records.append({**base_record, **audio_fields})

    audio_metadata = pd.DataFrame.from_records(records, columns=[*metadata.columns, *AUDIO_COLUMNS])
    report = build_validation_report(metadata, audio_metadata, malformed_indices)
    return ValidationResult(audio_metadata=audio_metadata, report=report)


def _value_counts(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {}
    counts = series.dropna().value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def _duration_stats(series: pd.Series) -> dict[str, float | None]:
    durations = pd.to_numeric(series, errors="coerce").dropna()
    if durations.empty:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
        }
    return {
        "min": float(durations.min()),
        "max": float(durations.max()),
        "mean": float(durations.mean()),
        "median": float(durations.median()),
        "std": float(durations.std(ddof=0)),
    }


def _duplicate_values(series: pd.Series) -> dict[str, int]:
    values = series.dropna()
    duplicates = values[values.duplicated(keep=False)].value_counts().sort_index()
    return {str(key): int(value) for key, value in duplicates.items()}


def build_validation_report(
    metadata: pd.DataFrame,
    audio_metadata: pd.DataFrame,
    malformed_indices: list[int],
) -> dict[str, Any]:
    """Build a JSON-serializable validation summary."""
    readable = audio_metadata["readable"] == True  # noqa: E712
    status_counts = _value_counts(audio_metadata["validation_status"])

    return {
        "total_metadata_rows": int(len(metadata)),
        "readable_files": int(readable.sum()),
        "unreadable_files": int((audio_metadata["validation_status"] == "unreadable").sum()),
        "missing_files": int((audio_metadata["validation_status"] == "missing").sum()),
        "malformed_records": int(len(malformed_indices)),
        "malformed_record_indices": malformed_indices,
        "duplicate_paths": _duplicate_values(audio_metadata["file_path"]),
        "duplicate_content_hashes": _duplicate_values(audio_metadata.loc[readable, "content_hash"]),
        "emotion_distribution": _value_counts(metadata.get("emotion", pd.Series(dtype=object))),
        "actor_distribution": _value_counts(metadata.get("actor_id", pd.Series(dtype=object))),
        "gender_distribution": _value_counts(metadata.get("actor_gender", pd.Series(dtype=object))),
        "sample_rate_distribution": _value_counts(audio_metadata.loc[readable, "sample_rate"]),
        "channel_count_distribution": _value_counts(audio_metadata.loc[readable, "channels"]),
        "duration_statistics": _duration_stats(audio_metadata.loc[readable, "duration_seconds"]),
        "near_silent_files": int(audio_metadata["near_silent"].fillna(False).sum()),
        "clips_below_min_duration": int(audio_metadata["below_min_duration"].fillna(False).sum()),
        "clips_above_max_duration": int(audio_metadata["above_max_duration"].fillna(False).sum()),
        "validation_status_distribution": status_counts,
    }


def save_validation_outputs(
    result: ValidationResult,
    output_dir: Path | str,
) -> dict[str, Path]:
    """Save JSON report, enriched metadata CSV, and distribution charts."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "report": out_dir / "data_validation_report.json",
        "audio_metadata": out_dir / "audio_metadata.csv",
        "class_distribution": out_dir / "class_distribution.png",
        "duration_distribution": out_dir / "duration_distribution.png",
        "sample_rate_distribution": out_dir / "sample_rate_distribution.png",
    }

    with paths["report"].open("w", encoding="utf-8") as handle:
        json.dump(result.report, handle, indent=2)
        handle.write("\n")

    result.audio_metadata.to_csv(paths["audio_metadata"], index=False)
    save_distribution_charts(result.audio_metadata, paths)
    return paths


def save_distribution_charts(audio_metadata: pd.DataFrame, paths: dict[str, Path]) -> None:
    """Save validation charts with titles, axis labels, and tight layout."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    readable = audio_metadata[audio_metadata["readable"] == True]  # noqa: E712

    def bar_chart(series: pd.Series, title: str, xlabel: str, ylabel: str, output_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        counts = series.dropna().value_counts().sort_index()
        if counts.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_xticks([])
        else:
            counts.plot(kind="bar", ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)

    bar_chart(
        audio_metadata.get("emotion", pd.Series(dtype=object)),
        "Voxels Class Distribution",
        "Emotion",
        "Count",
        paths["class_distribution"],
    )
    bar_chart(
        readable.get("sample_rate", pd.Series(dtype=object)),
        "Audio Sample Rate Distribution",
        "Sample rate (Hz)",
        "Count",
        paths["sample_rate_distribution"],
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    durations = pd.to_numeric(readable.get("duration_seconds", pd.Series(dtype=float)), errors="coerce").dropna()
    if durations.empty:
        ax.text(0.5, 0.5, "No readable audio", ha="center", va="center")
        ax.set_xticks([])
    else:
        ax.hist(durations, bins=min(30, max(1, len(durations))))
    ax.set_title("Audio Duration Distribution")
    ax.set_xlabel("Duration (seconds)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(paths["duration_distribution"])
    plt.close(fig)

