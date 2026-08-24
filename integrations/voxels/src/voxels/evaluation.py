"""Final held-out test evaluation for Voxels."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from voxels.audio import AudioPreprocessor, AudioPreprocessorConfig
from voxels.experiment import device_info
from voxels.labels import CANONICAL_LABELS, LABEL2ID
from voxels.wav2vec2 import DataCollatorWav2Vec2, Wav2Vec2SpeechDataset


class EvaluationError(RuntimeError):
    """Raised when final evaluation cannot run safely."""


@dataclass(frozen=True)
class EvaluationArtifacts:
    """Paths produced by final held-out evaluation."""

    directory: Path
    metrics: Path
    classification_report: Path
    predictions: Path
    error_analysis: Path
    confusion_matrix: Path
    normalized_confusion_matrix: Path
    summary: Path


def timestamped_evaluation_dir(base_dir: Path | str) -> Path:
    """Create a timestamped final-evaluation output directory."""
    root = Path(base_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ_final_evaluation")
    suffix = 1
    candidate = path
    while candidate.exists():
        suffix += 1
        candidate = root / f"{path.name}_{suffix}"
    candidate.mkdir(parents=False)
    return candidate


def require_checkpoint(checkpoint_dir: Path | str) -> Path:
    """Validate that a checkpoint directory contains required model files."""
    path = Path(checkpoint_dir)
    if not path.exists():
        raise EvaluationError(f"Checkpoint directory not found: {path}")
    if not path.is_dir():
        raise EvaluationError(f"Checkpoint path is not a directory: {path}")
    has_model = (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()
    if not (path / "config.json").exists() or not has_model:
        raise EvaluationError(f"Checkpoint is missing config/model files: {path}")
    if not (path / "label_mapping.json").exists():
        raise EvaluationError(f"Checkpoint is missing label_mapping.json: {path}")
    return path


def require_test_metadata(metadata_path: Path | str) -> pd.DataFrame:
    """Load and validate held-out test metadata."""
    path = Path(metadata_path)
    if not path.exists():
        raise EvaluationError(f"Test metadata not found: {path}")
    metadata = pd.read_csv(path)
    if metadata.empty:
        raise EvaluationError(f"Test metadata is empty: {path}")
    required = {"file_path", "filename", "emotion", "actor_id"}
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise EvaluationError(f"Test metadata is missing required columns: {missing}")
    return metadata


def load_label_mapping(checkpoint_dir: Path | str) -> dict[str, dict[str, int] | dict[int, str]]:
    """Load saved label mapping and verify canonical order."""
    path = Path(checkpoint_dir) / "label_mapping.json"
    with path.open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    label2id = {str(label): int(index) for label, index in mapping.get("label2id", {}).items()}
    if label2id != dict(LABEL2ID):
        raise EvaluationError("Checkpoint label mapping does not match the canonical Voxels label order.")
    return {
        "label2id": label2id,
        "id2label": {index: label for label, index in label2id.items()},
    }


def load_saved_processor(checkpoint_dir: Path | str) -> Any | None:
    """Load a saved processor when present.

    Smoke checkpoints may not include a processor; production checkpoints should.
    The evaluator remains explicit by returning None only when no processor files
    are present in the checkpoint directory.
    """
    path = Path(checkpoint_dir)
    processor_markers = ["preprocessor_config.json", "processor_config.json", "tokenizer_config.json"]
    if not any((path / marker).exists() for marker in processor_markers):
        return None
    try:
        from transformers import AutoProcessor

        return AutoProcessor.from_pretrained(path)
    except Exception as exc:
        raise EvaluationError(f"Could not load saved processor from {path}: {exc}") from exc


def load_saved_model(checkpoint_dir: Path | str) -> torch.nn.Module:
    """Load a saved Wav2Vec2 sequence classifier without modifying it."""
    try:
        from transformers import Wav2Vec2ForSequenceClassification

        model = Wav2Vec2ForSequenceClassification.from_pretrained(Path(checkpoint_dir))
    except Exception as exc:
        raise EvaluationError(f"Could not load model checkpoint from {checkpoint_dir}: {exc}") from exc
    model.eval()
    return model


def probability_columns() -> list[str]:
    """Return prediction probability columns in canonical label order."""
    return [f"prob_{label}" for label in CANONICAL_LABELS]


def build_predictions_dataframe(
    metadata: pd.DataFrame,
    true_labels: list[str],
    predicted_labels: list[str],
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """Build per-file prediction output with confidence and probabilities."""
    if probabilities.shape[1] != len(CANONICAL_LABELS):
        raise EvaluationError(f"Expected probabilities for {len(CANONICAL_LABELS)} labels, got {probabilities.shape[1]}.")
    rows: list[dict[str, Any]] = []
    for index, row in metadata.reset_index(drop=True).iterrows():
        probs = probabilities[index]
        predicted_label = predicted_labels[index]
        rows.append(
            {
                "filename": row["filename"],
                "actor_id": row["actor_id"],
                "true_label": true_labels[index],
                "predicted_label": predicted_label,
                "confidence": float(probs[CANONICAL_LABELS.index(predicted_label)]),
                **{column: float(probs[label_index]) for label_index, column in enumerate(probability_columns())},
                "correct": bool(true_labels[index] == predicted_label),
            }
        )
    return pd.DataFrame(rows)


def compute_evaluation_metrics(true_labels: list[str], predicted_labels: list[str]) -> dict[str, Any]:
    """Compute final held-out metrics in canonical label order."""
    matrix = confusion_matrix(true_labels, predicted_labels, labels=list(CANONICAL_LABELS))
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=list(CANONICAL_LABELS),
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "macro_f1": float(f1_score(true_labels, predicted_labels, labels=list(CANONICAL_LABELS), average="macro", zero_division=0)),
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in CANONICAL_LABELS
        },
        "confusion_matrix": matrix.tolist(),
        "normalized_confusion_matrix": normalized.tolist(),
        "labels": list(CANONICAL_LABELS),
    }


def classification_report_dataframe(metrics: dict[str, Any]) -> pd.DataFrame:
    """Return per-class report rows."""
    return pd.DataFrame(
        [
            {
                "label": label,
                "precision": metrics["per_class"][label]["precision"],
                "recall": metrics["per_class"][label]["recall"],
                "f1": metrics["per_class"][label]["f1"],
                "support": metrics["per_class"][label]["support"],
            }
            for label in CANONICAL_LABELS
        ]
    )


def most_confused_pairs(metrics: dict[str, Any], top_n: int = 10) -> list[dict[str, Any]]:
    """Identify most-confused true/predicted emotion pairs, excluding correct cells."""
    matrix = np.array(metrics["confusion_matrix"])
    pairs: list[dict[str, Any]] = []
    for true_index, true_label in enumerate(CANONICAL_LABELS):
        for predicted_index, predicted_label in enumerate(CANONICAL_LABELS):
            if true_index == predicted_index:
                continue
            count = int(matrix[true_index, predicted_index])
            if count:
                pairs.append({"true_label": true_label, "predicted_label": predicted_label, "count": count})
    return sorted(pairs, key=lambda item: item["count"], reverse=True)[:top_n]


def per_actor_performance(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-actor accuracy when actor IDs are available."""
    if predictions.empty or "actor_id" not in predictions.columns:
        return pd.DataFrame(columns=["actor_id", "file_count", "accuracy"])
    grouped = predictions.groupby("actor_id", dropna=False)
    return grouped["correct"].agg(file_count="count", accuracy="mean").reset_index()


def build_error_analysis(predictions: pd.DataFrame, metrics: dict[str, Any]) -> pd.DataFrame:
    """Build error-analysis rows for high-confidence errors, low confidence, and confused pairs."""
    rows: list[dict[str, Any]] = []
    incorrect = predictions[predictions["correct"] == False].copy()  # noqa: E712
    high_confidence_incorrect = incorrect.sort_values("confidence", ascending=False).head(20)
    for _, row in high_confidence_incorrect.iterrows():
        rows.append({**row.to_dict(), "analysis_type": "high_confidence_incorrect"})
    for _, row in predictions.sort_values("confidence", ascending=True).head(20).iterrows():
        rows.append({**row.to_dict(), "analysis_type": "lowest_confidence"})
    for pair in most_confused_pairs(metrics):
        rows.append(
            {
                "filename": "",
                "actor_id": "",
                "true_label": pair["true_label"],
                "predicted_label": pair["predicted_label"],
                "confidence": "",
                "correct": False,
                "analysis_type": "most_confused_pair",
                "count": pair["count"],
            }
        )
    return pd.DataFrame(rows)


def save_confusion_matrix_plot(
    matrix: list[list[float]],
    output_path: Path,
    title: str,
    normalized: bool = False,
) -> None:
    """Save confusion-matrix image."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.array(matrix)
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(values, cmap="Blues", vmin=0.0)
    ax.set_title(title)
    ax.set_xlabel("Predicted emotion")
    ax.set_ylabel("True emotion")
    ax.set_xticks(range(len(CANONICAL_LABELS)), labels=CANONICAL_LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(CANONICAL_LABELS)), labels=CANONICAL_LABELS)
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            text = f"{values[row_index, column_index]:.2f}" if normalized else str(int(values[row_index, column_index]))
            ax.text(column_index, row_index, text, ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_evaluation_summary(
    path: Path,
    checkpoint_dir: Path,
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
    error_analysis: pd.DataFrame,
) -> None:
    """Write human-readable final evaluation summary."""
    incorrect_count = int((predictions["correct"] == False).sum())  # noqa: E712
    lines = [
        "# Voxels Final Held-Out Evaluation",
        "",
        f"Checkpoint: `{checkpoint_dir}`",
        "",
        f"Accuracy: {metrics['accuracy']:.6f}",
        f"Macro F1: {metrics['macro_f1']:.6f}",
        f"Files evaluated: {len(predictions)}",
        f"Incorrect predictions: {incorrect_count}",
        "",
        "The held-out test set is used here only after validation-based checkpoint selection.",
        "",
        "Most confused pairs:",
    ]
    pairs = most_confused_pairs(metrics)
    if not pairs:
        lines.append("- None")
    else:
        for pair in pairs:
            lines.append(f"- {pair['true_label']} -> {pair['predicted_label']}: {pair['count']}")
    lines.append("")
    lines.append(f"Error-analysis rows: {len(error_analysis)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_evaluation_outputs(
    output_dir: Path,
    checkpoint_dir: Path,
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
) -> EvaluationArtifacts:
    """Save final evaluation files."""
    classification_df = classification_report_dataframe(metrics)
    error_df = build_error_analysis(predictions, metrics)
    artifacts = EvaluationArtifacts(
        directory=output_dir,
        metrics=output_dir / "metrics.json",
        classification_report=output_dir / "classification_report.csv",
        predictions=output_dir / "predictions.csv",
        error_analysis=output_dir / "error_analysis.csv",
        confusion_matrix=output_dir / "confusion_matrix.png",
        normalized_confusion_matrix=output_dir / "normalized_confusion_matrix.png",
        summary=output_dir / "evaluation_summary.md",
    )
    with artifacts.metrics.open("w", encoding="utf-8") as handle:
        json.dump({**metrics, "device": device_info()}, handle, indent=2)
        handle.write("\n")
    classification_df.to_csv(artifacts.classification_report, index=False)
    predictions.to_csv(artifacts.predictions, index=False)
    error_df.to_csv(artifacts.error_analysis, index=False)
    save_confusion_matrix_plot(metrics["confusion_matrix"], artifacts.confusion_matrix, "Held-Out Test Confusion Matrix")
    save_confusion_matrix_plot(
        metrics["normalized_confusion_matrix"],
        artifacts.normalized_confusion_matrix,
        "Normalized Held-Out Test Confusion Matrix",
        normalized=True,
    )
    write_evaluation_summary(artifacts.summary, checkpoint_dir, metrics, predictions, error_df)
    return artifacts


@torch.no_grad()
def run_model_predictions(
    model: torch.nn.Module,
    processor: Any | None,
    metadata: pd.DataFrame,
    batch_size: int = 4,
) -> tuple[list[str], np.ndarray]:
    """Run model inference on held-out test metadata and return labels/probabilities."""
    preprocessor = AudioPreprocessor(AudioPreprocessorConfig())
    dataset = Wav2Vec2SpeechDataset(metadata, preprocessor, mode="inference")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=DataCollatorWav2Vec2(processor))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    all_probabilities: list[np.ndarray] = []
    predicted_labels: list[str] = []
    for batch in loader:
        outputs = model(
            input_values=batch["input_values"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        probabilities = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        all_probabilities.append(probabilities)
        predicted_labels.extend(CANONICAL_LABELS[int(index)] for index in probabilities.argmax(axis=1))
    return predicted_labels, np.vstack(all_probabilities)

