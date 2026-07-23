"""TF-IDF Logistic Regression baseline for MeetScribe sentence labels.

Purpose:
    Provide a simple non-ANN research baseline for comparing against the
    MiniLM + ANN classifier. This script does not affect production inference,
    saved ANN checkpoints, or the Streamlit application.

Inputs:
    ``datasets/processed/master_dataset.csv``.

Outputs:
    - ``datasets/models/logistic_baseline_results.json``
    - ``datasets/models/logistic_classification_report.txt``
    - ``datasets/models/logistic_confusion_matrix.png``
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

try:
    from ml_mom.training_dataset import LABEL_TO_ID
except ModuleNotFoundError:  # pragma: no cover - supports direct execution.
    from training_dataset import LABEL_TO_ID


MASTER_DATASET_PATH = Path("datasets") / "processed" / "master_dataset.csv"
MODEL_DIR = Path("datasets") / "models"
RESULTS_PATH = MODEL_DIR / "logistic_baseline_results.json"
REPORT_PATH = MODEL_DIR / "logistic_classification_report.txt"
CONFUSION_MATRIX_PATH = MODEL_DIR / "logistic_confusion_matrix.png"
RANDOM_SEED = 42


def load_text_dataset(dataset_path: Path = MASTER_DATASET_PATH) -> tuple[list[str], list[int]]:
    """Load sentences and numeric labels from the master dataset."""

    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return [], []
    sentences: list[str] = []
    labels: list[int] = []
    seen: set[tuple[str, str, str]] = set()
    with dataset_path.open("r", newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            sentence = (row.get("sentence") or "").strip()
            label = (row.get("selected_label") or "").strip()
            if not sentence or label not in LABEL_TO_ID:
                continue
            key = (
                (row.get("turn_id") or "").strip(),
                (row.get("speaker") or "").strip().casefold(),
                sentence.casefold(),
            )
            if key in seen:
                continue
            seen.add(key)
            sentences.append(sentence)
            labels.append(LABEL_TO_ID[label])
    return sentences, labels


def ordered_labels() -> list[str]:
    """Return supported labels in numeric class order."""

    return [label for label, _id in sorted(LABEL_TO_ID.items(), key=lambda item: item[1])]


def save_confusion_matrix(matrix: list[list[int]], labels: list[str]) -> None:
    """Save a labeled confusion matrix plot as PNG."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_title("Logistic Baseline Confusion Matrix")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_xticks(range(len(labels)))
    axis.set_yticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_yticklabels(labels)
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            axis.text(column_index, row_index, str(value), ha="center", va="center")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH)
    plt.close(fig)


def run_logistic_baseline() -> None:
    """Train and evaluate the TF-IDF Logistic Regression baseline."""

    sentences, labels = load_text_dataset()
    if not sentences:
        print("No valid samples were available for logistic baseline evaluation.")
        return
    if min(labels.count(class_id) for class_id in set(labels)) < 2:
        print("Each class needs at least 2 samples for stratified baseline evaluation.")
        return

    train_texts, validation_texts, train_labels, validation_labels = train_test_split(
        sentences,
        labels,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=labels,
    )
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
    train_features = vectorizer.fit_transform(train_texts)
    validation_features = vectorizer.transform(validation_texts)
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    model.fit(train_features, train_labels)
    predictions = model.predict(validation_features)

    label_ids = list(range(len(LABEL_TO_ID)))
    label_names = ordered_labels()
    precision, recall, macro_f1, _support = precision_recall_fscore_support(
        validation_labels,
        predictions,
        average="macro",
        zero_division=0,
    )
    _weighted_precision, _weighted_recall, weighted_f1, _support = (
        precision_recall_fscore_support(
            validation_labels,
            predictions,
            average="weighted",
            zero_division=0,
        )
    )
    matrix = confusion_matrix(validation_labels, predictions, labels=label_ids).tolist()
    report = classification_report(
        validation_labels,
        predictions,
        labels=label_ids,
        target_names=label_names,
        zero_division=0,
    )
    results: dict[str, Any] = {
        "accuracy": accuracy_score(validation_labels, predictions),
        "precision": precision,
        "recall": recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": matrix,
        "train_samples": len(train_texts),
        "validation_samples": len(validation_texts),
    }
    print("Logistic Regression Baseline")
    print(json.dumps(results, indent=2))
    print("Classification Report")
    print(report)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(report, encoding="utf-8")
    save_confusion_matrix(matrix, label_names)
    print(f"Baseline results saved to: {RESULTS_PATH}")
    print(f"Classification report saved to: {REPORT_PATH}")
    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    run_logistic_baseline()
