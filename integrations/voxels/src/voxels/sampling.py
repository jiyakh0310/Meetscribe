"""Deterministic metadata sampling helpers for smoke and diagnostic runs."""

from __future__ import annotations

import pandas as pd

from voxels.labels import CANONICAL_LABELS


class SamplingError(ValueError):
    """Raised when a requested metadata sample cannot be built safely."""


def stratified_limit_metadata(
    metadata: pd.DataFrame,
    limit: int | None,
    seed: int,
    label_column: str = "emotion",
) -> pd.DataFrame:
    """Return a deterministic stratified subset without changing full-data behavior.

    When ``limit`` is at least the number of canonical labels, the subset includes
    every class that is present in the source metadata. Rows are sampled only from
    the provided split, so actor-independent split boundaries are preserved.
    """
    if limit is None or limit >= len(metadata):
        return metadata.reset_index(drop=True)
    if limit <= 0:
        raise SamplingError("Row limits must be positive.")
    if label_column not in metadata.columns:
        raise SamplingError(f"Metadata is missing required label column: {label_column}")

    frame = metadata.reset_index(drop=True).copy()
    labels = frame[label_column].astype(str)
    unknown = sorted(set(labels).difference(CANONICAL_LABELS))
    if unknown:
        raise SamplingError(f"Metadata contains unknown labels: {unknown}")

    present_labels = [label for label in CANONICAL_LABELS if (labels == label).any()]
    if limit >= len(CANONICAL_LABELS) and len(present_labels) < len(CANONICAL_LABELS):
        missing = [label for label in CANONICAL_LABELS if label not in present_labels]
        raise SamplingError(f"Cannot include all canonical classes because this split is missing: {missing}")

    if limit < len(present_labels):
        sampled_labels = present_labels[:limit]
    else:
        sampled_labels = present_labels

    selected_indices: list[int] = []
    for label in sampled_labels:
        group = frame[labels == label]
        selected_indices.extend(group.sample(n=1, random_state=seed).index.tolist())

    remaining = limit - len(selected_indices)
    if remaining > 0:
        base = remaining // len(present_labels)
        extra = remaining % len(present_labels)
        for position, label in enumerate(present_labels):
            group = frame[labels == label].drop(index=selected_indices, errors="ignore")
            take = min(len(group), base + (1 if position < extra else 0))
            if take > 0:
                selected_indices.extend(group.sample(n=take, random_state=seed + position + 1).index.tolist())

    if len(selected_indices) < limit:
        remaining_pool = frame.drop(index=selected_indices, errors="ignore")
        needed = min(limit - len(selected_indices), len(remaining_pool))
        if needed > 0:
            selected_indices.extend(remaining_pool.sample(n=needed, random_state=seed + 997).index.tolist())

    if len(selected_indices) != min(limit, len(frame)):
        raise SamplingError(
            f"Could not build requested stratified subset: requested {limit}, selected {len(selected_indices)}."
        )

    subset = frame.loc[selected_indices].sort_values(["actor_id", "file_path"], kind="stable").reset_index(drop=True)
    return subset


def label_distribution(metadata: pd.DataFrame, label_column: str = "emotion") -> dict[str, int]:
    """Return canonical-label counts for a metadata frame."""
    counts = metadata[label_column].astype(str).value_counts().to_dict() if label_column in metadata.columns else {}
    return {label: int(counts.get(label, 0)) for label in CANONICAL_LABELS}
