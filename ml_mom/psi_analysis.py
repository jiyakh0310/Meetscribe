"""Population Stability Index analysis for MeetScribe label distributions.

Purpose:
    Compare the current labeled dataset against an original dataset snapshot to
    quantify class-distribution drift for research review only.

Inputs:
    - Current dataset: ``datasets/processed/master_dataset.csv``
    - Original dataset: ``datasets/processed/master_dataset_old.csv``

Outputs:
    - ``datasets/models/psi_report.json``
    - ``datasets/models/psi_report.txt``
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

try:
    from ml_mom.training_dataset import LABEL_TO_ID
except ModuleNotFoundError:  # pragma: no cover - supports direct execution.
    from training_dataset import LABEL_TO_ID


CURRENT_DATASET_PATH = Path("datasets") / "processed" / "master_dataset.csv"
ORIGINAL_DATASET_PATH = Path("datasets") / "processed" / "master_dataset_old.csv"
MODEL_DIR = Path("datasets") / "models"
JSON_OUTPUT_PATH = MODEL_DIR / "psi_report.json"
TEXT_OUTPUT_PATH = MODEL_DIR / "psi_report.txt"
EPSILON = 1e-6


def load_label_counts(dataset_path: Path) -> dict[str, int]:
    """Load class counts from a dataset CSV."""

    counts = {label: 0 for label in LABEL_TO_ID}
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return counts
    seen: set[tuple[str, str, str]] = set()
    with dataset_path.open("r", newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            sentence = (row.get("sentence") or "").strip()
            label = (row.get("selected_label") or "").strip()
            if not sentence or label not in counts:
                continue
            key = (
                (row.get("turn_id") or "").strip(),
                (row.get("speaker") or "").strip().casefold(),
                sentence.casefold(),
            )
            if key in seen:
                continue
            seen.add(key)
            counts[label] += 1
    return counts


def proportions(counts: dict[str, int]) -> dict[str, float]:
    """Convert class counts to smoothed proportions."""

    total = sum(counts.values())
    if total == 0:
        return {label: EPSILON for label in counts}
    return {label: max(count / total, EPSILON) for label, count in counts.items()}


def psi_value(expected: float, actual: float) -> float:
    """Compute one PSI component from expected and actual proportions."""

    expected = max(expected, EPSILON)
    actual = max(actual, EPSILON)
    return (actual - expected) * math.log(actual / expected)


def interpret_psi(value: float) -> str:
    """Return the standard PSI interpretation bucket."""

    if value < 0.1:
        return "Stable"
    if value <= 0.25:
        return "Moderate Drift"
    return "Significant Drift"


def build_psi_report() -> dict[str, Any]:
    """Build overall and per-class PSI report data."""

    original_counts = load_label_counts(ORIGINAL_DATASET_PATH)
    current_counts = load_label_counts(CURRENT_DATASET_PATH)
    original_proportions = proportions(original_counts)
    current_proportions = proportions(current_counts)
    per_class = {}
    overall_psi = 0.0
    for label in LABEL_TO_ID:
        value = psi_value(original_proportions[label], current_proportions[label])
        overall_psi += value
        per_class[label] = {
            "original_count": original_counts[label],
            "current_count": current_counts[label],
            "original_proportion": original_proportions[label],
            "current_proportion": current_proportions[label],
            "psi": value,
            "interpretation": interpret_psi(value),
        }
    return {
        "current_dataset": str(CURRENT_DATASET_PATH),
        "original_dataset": str(ORIGINAL_DATASET_PATH),
        "overall_psi": overall_psi,
        "overall_interpretation": interpret_psi(overall_psi),
        "per_class_psi": per_class,
    }


def render_text_report(report: dict[str, Any]) -> str:
    """Render PSI report as human-readable text."""

    lines = [
        "Population Stability Index Report",
        "",
        f"Current dataset: {report['current_dataset']}",
        f"Original dataset: {report['original_dataset']}",
        f"Overall PSI: {report['overall_psi']:.6f}",
        f"Interpretation: {report['overall_interpretation']}",
        "",
        "Per-class PSI",
    ]
    for label, metrics in report["per_class_psi"].items():
        lines.extend(
            [
                "",
                label,
                f" Original Count     : {metrics['original_count']}",
                f" Current Count      : {metrics['current_count']}",
                f" Original Proportion: {metrics['original_proportion']:.6f}",
                f" Current Proportion : {metrics['current_proportion']:.6f}",
                f" PSI                : {metrics['psi']:.6f}",
                f" Interpretation     : {metrics['interpretation']}",
            ]
        )
    return "\n".join(lines)


def run_psi_analysis() -> None:
    """Compute and save PSI analysis artifacts."""

    report = build_psi_report()
    text_report = render_text_report(report)
    print(text_report)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    TEXT_OUTPUT_PATH.write_text(text_report, encoding="utf-8")
    print(f"PSI JSON saved to: {JSON_OUTPUT_PATH}")
    print(f"PSI text report saved to: {TEXT_OUTPUT_PATH}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    run_psi_analysis()
