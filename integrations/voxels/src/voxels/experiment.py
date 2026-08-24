"""Lightweight local experiment tracking for Voxels training runs."""

from __future__ import annotations

import csv
import json
import platform
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch


def utc_timestamp() -> str:
    """Return a filesystem-friendly UTC timestamp."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def create_experiment_dir(base_dir: Path | str, mode: str, seed: int) -> Path:
    """Create a unique local experiment directory."""
    root = Path(base_dir)
    root.mkdir(parents=True, exist_ok=True)
    safe_mode = mode.replace("/", "_")
    path = root / f"{utc_timestamp()}_{safe_mode}_seed{seed}_{uuid4().hex[:8]}"
    path.mkdir(parents=False, exist_ok=False)
    return path


def device_info() -> dict[str, Any]:
    """Return execution-device information."""
    cuda_available = torch.cuda.is_available()
    return {
        "device": "cuda" if cuda_available else "cpu",
        "cuda_available": bool(cuda_available),
        "cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def to_jsonable(value: Any) -> Any:
    """Convert dataclasses and Paths for JSON serialization."""
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def save_resolved_config(experiment_dir: Path, config: Any, metadata: dict[str, Any]) -> Path:
    """Save resolved configuration and run metadata."""
    path = experiment_dir / "resolved_config.json"
    payload = {"configuration": to_jsonable(config), "metadata": to_jsonable(metadata)}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return path


def save_history(experiment_dir: Path, history: list[dict[str, Any]]) -> dict[str, Path]:
    """Save machine-readable epoch history as JSON and CSV."""
    json_path = experiment_dir / "history.json"
    csv_path = experiment_dir / "history.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
        handle.write("\n")
    fieldnames = sorted({key for row in history for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    return {"json": json_path, "csv": csv_path}


def save_run_summary(experiment_dir: Path, summary: dict[str, Any]) -> Path:
    """Save final run summary."""
    path = experiment_dir / "run_summary.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(summary), handle, indent=2)
        handle.write("\n")
    return path


def plot_history_curves(experiment_dir: Path, history: list[dict[str, Any]]) -> dict[str, Path]:
    """Generate local PNG curves, tolerating missing optional values."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = {
        "loss": experiment_dir / "loss_curve.png",
        "accuracy": experiment_dir / "validation_accuracy_curve.png",
        "macro_f1": experiment_dir / "validation_macro_f1_curve.png",
        "learning_rate": experiment_dir / "learning_rate_curve.png",
    }

    epochs = [row.get("epoch") for row in history]

    def plot_series(filename: Path, title: str, ylabel: str, keys: list[str]) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = False
        for key in keys:
            values = [row.get(key) for row in history]
            valid = [(epoch, value) for epoch, value in zip(epochs, values, strict=False) if value is not None]
            if valid:
                ax.plot([item[0] for item in valid], [item[1] for item in valid], marker="o", label=key)
                plotted = True
        if not plotted:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
        else:
            ax.legend()
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(filename)
        plt.close(fig)

    plot_series(paths["loss"], "Training and Validation Loss", "Loss", ["training_loss", "validation_loss"])
    plot_series(paths["accuracy"], "Validation Accuracy", "Accuracy", ["validation_accuracy"])
    plot_series(paths["macro_f1"], "Validation Macro F1", "Macro F1", ["validation_macro_f1"])
    plot_series(paths["learning_rate"], "Learning Rate", "Learning rate", ["learning_rate"])
    return paths

