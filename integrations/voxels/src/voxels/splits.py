"""Actor-independent dataset splitting for Voxels."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

SPLIT_NAMES: tuple[str, str, str] = ("train", "validation", "test")


class SplitError(ValueError):
    """Raised when actor-independent splits cannot be created safely."""


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for actor-independent splitting."""

    seed: int = 42
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    train_count: int | None = None
    validation_count: int | None = None
    test_count: int | None = None


@dataclass(frozen=True)
class SplitResult:
    """Split dataframes plus persistent manifest."""

    splits: dict[str, pd.DataFrame]
    manifest: dict[str, Any]


def validate_metadata_for_splitting(metadata: pd.DataFrame) -> None:
    """Validate source metadata before actor-level splitting."""
    if metadata.empty:
        raise SplitError("Metadata is empty; cannot create train/validation/test splits.")
    if "actor_id" not in metadata.columns:
        raise SplitError("Metadata is missing required actor_id column.")
    missing_actor_ids = metadata["actor_id"].isna() | (metadata["actor_id"].astype(str).str.strip() == "")
    if missing_actor_ids.any():
        indices = [int(index) for index in metadata.index[missing_actor_ids].tolist()]
        raise SplitError(f"Metadata contains missing actor_id values at rows: {indices}")

    actor_count = metadata["actor_id"].nunique()
    if actor_count < len(SPLIT_NAMES):
        raise SplitError(
            f"At least {len(SPLIT_NAMES)} unique actors are required for non-empty splits; found {actor_count}."
        )


def actor_counts_from_config(actor_count: int, config: SplitConfig) -> dict[str, int]:
    """Resolve actor counts from configured ratios or explicit counts."""
    explicit_counts = {
        "train": config.train_count,
        "validation": config.validation_count,
        "test": config.test_count,
    }
    provided_counts = [count for count in explicit_counts.values() if count is not None]
    if provided_counts:
        if len(provided_counts) != len(SPLIT_NAMES):
            raise SplitError("Actor counts must be provided for train, validation, and test together.")
        counts = {name: int(explicit_counts[name]) for name in SPLIT_NAMES}
        if any(count <= 0 for count in counts.values()):
            raise SplitError("Actor counts must be positive for every split.")
        if sum(counts.values()) != actor_count:
            raise SplitError(f"Actor counts sum to {sum(counts.values())}, but metadata has {actor_count} actors.")
        return counts

    ratios = {
        "train": config.train_ratio,
        "validation": config.validation_ratio,
        "test": config.test_ratio,
    }
    if any(value <= 0 for value in ratios.values()):
        raise SplitError("Train, validation, and test ratios must all be positive.")
    ratio_total = sum(ratios.values())
    if abs(ratio_total - 1.0) > 1e-6:
        raise SplitError(f"Train, validation, and test ratios must sum to 1.0; got {ratio_total:.6f}.")

    raw_counts = {name: ratios[name] * actor_count for name in SPLIT_NAMES}
    counts = {name: max(1, int(raw_counts[name])) for name in SPLIT_NAMES}
    while sum(counts.values()) > actor_count:
        candidates = [name for name in SPLIT_NAMES if counts[name] > 1]
        if not candidates:
            raise SplitError("Could not allocate at least one actor to every split.")
        name = min(candidates, key=lambda split: raw_counts[split] - int(raw_counts[split]))
        counts[name] -= 1
    while sum(counts.values()) < actor_count:
        name = max(SPLIT_NAMES, key=lambda split: raw_counts[split] - counts[split])
        counts[name] += 1
    return counts


def create_actor_assignments(metadata: pd.DataFrame, config: SplitConfig) -> dict[str, list[int]]:
    """Create deterministic actor assignments without row-wise splitting."""
    validate_metadata_for_splitting(metadata)
    actors = sorted(int(actor_id) for actor_id in metadata["actor_id"].unique())
    counts = actor_counts_from_config(len(actors), config)

    shuffled_actors = list(actors)
    random.Random(config.seed).shuffle(shuffled_actors)

    assignments: dict[str, list[int]] = {}
    cursor = 0
    for split_name in SPLIT_NAMES:
        split_count = counts[split_name]
        assignments[split_name] = sorted(shuffled_actors[cursor : cursor + split_count])
        cursor += split_count
    return assignments


def split_metadata_by_actor(metadata: pd.DataFrame, config: SplitConfig) -> SplitResult:
    """Split metadata into train, validation, and test actor-disjoint dataframes."""
    source = metadata.copy()
    if "source_row_id" not in source.columns:
        source.insert(0, "source_row_id", range(len(source)))

    assignments = create_actor_assignments(source, config)
    splits: dict[str, pd.DataFrame] = {}
    for split_name, actors in assignments.items():
        split_df = source[source["actor_id"].astype(int).isin(actors)].copy()
        splits[split_name] = split_df.sort_values(["actor_id", "source_row_id"]).reset_index(drop=True)

    leakage_detected = actor_leakage_detected(assignments)
    if leakage_detected:
        raise SplitError("Actor leakage detected between splits.")
    ensure_complete_assignment(source, splits)

    manifest = build_split_manifest(source, splits, assignments, config, leakage_detected)
    return SplitResult(splits=splits, manifest=manifest)


def actor_leakage_detected(assignments: dict[str, list[int]]) -> bool:
    """Return True when any actor appears in more than one split."""
    seen: set[int] = set()
    for actors in assignments.values():
        actor_set = set(actors)
        if seen.intersection(actor_set):
            return True
        seen.update(actor_set)
    return False


def ensure_complete_assignment(source: pd.DataFrame, splits: dict[str, pd.DataFrame]) -> None:
    """Fail if any source row is missing or duplicated across splits."""
    source_ids = set(source["source_row_id"].tolist())
    assigned_ids: list[int] = []
    for split_df in splits.values():
        assigned_ids.extend(int(row_id) for row_id in split_df["source_row_id"].tolist())

    if len(assigned_ids) != len(set(assigned_ids)):
        raise SplitError("Duplicate source rows detected across splits.")
    if set(assigned_ids) != source_ids:
        missing = sorted(source_ids.difference(assigned_ids))
        extra = sorted(set(assigned_ids).difference(source_ids))
        raise SplitError(f"Split assignment mismatch. Missing rows: {missing}; extra rows: {extra}")


def _distribution(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df.columns or df.empty:
        return {}
    counts = df[column].value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def split_summary(split_df: pd.DataFrame, actor_ids: list[int]) -> dict[str, Any]:
    """Build one split summary for console output and manifests."""
    return {
        "actor_count": int(len(actor_ids)),
        "actor_ids": actor_ids,
        "file_count": int(len(split_df)),
        "emotion_distribution": _distribution(split_df, "emotion"),
        "gender_distribution": _distribution(split_df, "actor_gender"),
    }


def build_split_manifest(
    source: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    assignments: dict[str, list[int]],
    config: SplitConfig,
    leakage_detected: bool,
) -> dict[str, Any]:
    """Build a JSON-serializable split manifest."""
    return {
        "seed": config.seed,
        "ratios": {
            "train": config.train_ratio,
            "validation": config.validation_ratio,
            "test": config.test_ratio,
        },
        "actor_counts_requested": {
            "train": config.train_count,
            "validation": config.validation_count,
            "test": config.test_count,
        },
        "total_rows": int(len(source)),
        "total_actors": int(source["actor_id"].nunique()),
        "actor_leakage_detected": bool(leakage_detected),
        "splits": {
            split_name: split_summary(splits[split_name], assignments[split_name]) for split_name in SPLIT_NAMES
        },
    }


def save_split_outputs(result: SplitResult, output_dir: Path | str) -> dict[str, Path]:
    """Save split metadata CSVs and the actor assignment manifest."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": out_dir / "train_metadata.csv",
        "validation": out_dir / "validation_metadata.csv",
        "test": out_dir / "test_metadata.csv",
        "manifest": out_dir / "split_manifest.json",
    }
    for split_name in SPLIT_NAMES:
        result.splits[split_name].to_csv(paths[split_name], index=False)
    with paths["manifest"].open("w", encoding="utf-8") as handle:
        json.dump(result.manifest, handle, indent=2)
        handle.write("\n")
    return paths


def format_split_summary(result: SplitResult) -> str:
    """Format detailed split summaries for command-line output."""
    lines: list[str] = []
    for split_name in SPLIT_NAMES:
        summary = result.manifest["splits"][split_name]
        lines.append(f"{split_name}:")
        lines.append(f"  actor count: {summary['actor_count']}")
        lines.append(f"  actor IDs: {summary['actor_ids']}")
        lines.append(f"  file count: {summary['file_count']}")
        lines.append(f"  emotion distribution: {summary['emotion_distribution']}")
        lines.append(f"  gender distribution: {summary['gender_distribution']}")
    lines.append(f"Actor leakage detected: {result.manifest['actor_leakage_detected']}")
    return "\n".join(lines)

