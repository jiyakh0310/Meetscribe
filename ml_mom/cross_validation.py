"""Stratified cross-validation for the MeetScribe ANN classifier.

Purpose:
    Evaluate the existing MiniLM + ANN approach across stratified folds without
    touching the production training script, saved checkpoints, inference code,
    Streamlit UI, or MoM generation pipeline.

Responsibilities:
    - Load ``datasets/processed/master_dataset.csv``.
    - Generate sentence embeddings through the existing ``EmbeddingService``.
    - Train fresh in-memory ANN instances for Stratified 5-Fold evaluation.
    - Report fold metrics and aggregate averages/standard deviations.
    - Save research-only results to ``datasets/models/cross_validation_results.json``.

Inputs:
    Current processed master dataset.

Outputs:
    Cross-validation metrics JSON. No models are saved.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
import random
import sys
from typing import Any

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from ml_mom.ann_model import SentenceClassifierANN
    from ml_mom.embeddings import EmbeddingService
    from ml_mom.feature_extraction import SentenceFeature
    from ml_mom.training_dataset import LABEL_TO_ID
except ModuleNotFoundError:  # pragma: no cover - supports direct execution.
    from ann_model import SentenceClassifierANN
    from embeddings import EmbeddingService
    from feature_extraction import SentenceFeature
    from training_dataset import LABEL_TO_ID


MASTER_DATASET_PATH = Path("datasets") / "processed" / "master_dataset.csv"
OUTPUT_PATH = Path("datasets") / "models" / "cross_validation_results.json"
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001
RANDOM_SEED = 42
FOLD_COUNT = 5


@dataclass(slots=True)
class EvaluationSample:
    """One labeled sentence used for cross-validation."""

    sentence_id: int
    turn_id: int
    speaker: str
    timestamp: int | None
    sentence: str
    selected_label: str
    label_id: int


def load_samples(dataset_path: Path = MASTER_DATASET_PATH) -> list[EvaluationSample]:
    """Load valid labeled samples from the master dataset.

    Args:
        dataset_path: CSV dataset path.

    Returns:
        Cleaned samples with numeric label IDs. Invalid, duplicate, and empty
        rows are skipped so research evaluation does not crash.
    """

    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return []

    samples: list[EvaluationSample] = []
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
            samples.append(
                EvaluationSample(
                    sentence_id=parse_optional_int(row.get("sentence_id")) or len(samples) + 1,
                    turn_id=parse_optional_int(row.get("turn_id")) or 0,
                    speaker=(row.get("speaker") or "").strip(),
                    timestamp=parse_optional_int(row.get("timestamp")),
                    sentence=sentence,
                    selected_label=label,
                    label_id=LABEL_TO_ID[label],
                )
            )
    return samples


def parse_optional_int(value: str | None) -> int | None:
    """Parse integer-like CSV values safely."""

    if value is None or not str(value).strip():
        return None
    try:
        return int(float(str(value).strip()))
    except ValueError:
        return None


def sample_to_sentence_feature(sample: EvaluationSample) -> SentenceFeature:
    """Convert a labeled sample into the embedding service input shape."""

    words = sample.sentence.split()
    average_word_length = sum(len(word) for word in words) / len(words) if words else 0.0
    return SentenceFeature(
        sentence_id=sample.sentence_id,
        turn_id=sample.turn_id,
        speaker=sample.speaker,
        timestamp=sample.timestamp,
        original_sentence=sample.sentence,
        word_count=len(words),
        character_count=len(sample.sentence),
        average_word_length=round(average_word_length, 3),
        sentence_position=sample.sentence_id,
        is_question=sample.sentence.rstrip().endswith("?"),
        is_exclamation="!" in sample.sentence,
        contains_number=any(char.isdigit() for char in sample.sentence),
        contains_currency=False,
        contains_percentage="%" in sample.sentence,
        contains_date=False,
        contains_time=False,
        contains_action_keyword=False,
        contains_decision_keyword=False,
        contains_discussion_keyword=False,
        contains_deadline_keyword=False,
        speaker_name=sample.speaker,
        speaker_turn_index=1,
        previous_sentence=None,
        next_sentence=None,
    )


def generate_embedding_tensor(samples: list[EvaluationSample]) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Generate MiniLM embeddings for all cross-validation samples."""

    service = EmbeddingService()
    features = [sample_to_sentence_feature(sample) for sample in samples]
    result = service.generate_embeddings(features)
    if result.error_message:
        print(result.error_message)
        return None
    vectors = [embedding.embedding_vector for embedding in result.embeddings if embedding.embedding_vector]
    if len(vectors) != len(samples):
        print("Embedding count did not match sample count.")
        return None
    labels = [sample.label_id for sample in samples]
    return torch.tensor(vectors, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def compute_class_weights(labels: torch.Tensor, class_count: int) -> torch.Tensor:
    """Compute balanced inverse-frequency weights for one fold."""

    counts = torch.bincount(labels, minlength=class_count).float()
    total = counts.sum()
    weights = torch.zeros(class_count, dtype=torch.float32)
    for index, count in enumerate(counts):
        if count.item() > 0:
            weights[index] = total / (class_count * count)
    return weights


def train_fold(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
) -> dict[str, float]:
    """Train an in-memory ANN for one fold and return validation metrics."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceClassifierANN(input_dim=train_features.size(1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss(
        weight=compute_class_weights(train_labels, len(LABEL_TO_ID)).to(device)
    )
    loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    for _epoch in range(EPOCHS):
        model.train()
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_features), batch_labels)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(validation_features.to(device))
        predictions = torch.argmax(logits, dim=1).cpu().tolist()
    targets = validation_labels.cpu().tolist()
    precision, recall, macro_f1, _support = precision_recall_fscore_support(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )
    _weighted_precision, _weighted_recall, weighted_f1, _support = (
        precision_recall_fscore_support(
            targets,
            predictions,
            average="weighted",
            zero_division=0,
        )
    )
    return {
        "accuracy": accuracy_score(targets, predictions),
        "precision": precision,
        "recall": recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def summarize_folds(fold_metrics: list[dict[str, float]]) -> dict[str, Any]:
    """Aggregate fold metrics into average and standard deviation values."""

    metric_names = ["accuracy", "precision", "recall", "macro_f1", "weighted_f1"]
    summary: dict[str, Any] = {"folds": fold_metrics}
    for metric_name in metric_names:
        values = [fold[metric_name] for fold in fold_metrics]
        average = sum(values) / len(values) if values else 0.0
        variance = (
            sum((value - average) ** 2 for value in values) / len(values)
            if values
            else 0.0
        )
        summary[f"average_{metric_name}"] = average
        summary[f"std_{metric_name}"] = variance ** 0.5
    return summary


def run_cross_validation() -> None:
    """Run Stratified 5-Fold ANN cross-validation and save research metrics."""

    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    samples = load_samples()
    if not samples:
        print("No valid samples were available for cross-validation.")
        return
    labels = [sample.label_id for sample in samples]
    min_class_count = min(labels.count(class_id) for class_id in set(labels))
    if min_class_count < FOLD_COUNT:
        print("Stratified 5-Fold Cross Validation requires at least 5 samples per class.")
        return

    tensors = generate_embedding_tensor(samples)
    if tensors is None:
        return
    features, label_tensor = tensors
    splitter = StratifiedKFold(n_splits=FOLD_COUNT, shuffle=True, random_state=RANDOM_SEED)
    fold_metrics: list[dict[str, float]] = []
    for fold_index, (train_index, validation_index) in enumerate(
        splitter.split(features.numpy(), labels),
        start=1,
    ):
        metrics = train_fold(
            train_features=features[train_index],
            train_labels=label_tensor[train_index],
            validation_features=features[validation_index],
            validation_labels=label_tensor[validation_index],
        )
        fold_metrics.append({"fold": fold_index, **metrics})
        print(f"Fold {fold_index}")
        print(json.dumps(metrics, indent=2))

    summary = summarize_folds(fold_metrics)
    print(f"Average Accuracy: {summary['average_accuracy']:.4f}")
    print(f"Average Precision: {summary['average_precision']:.4f}")
    print(f"Average Recall: {summary['average_recall']:.4f}")
    print(f"Average Macro F1: {summary['average_macro_f1']:.4f}")
    print(f"Average Weighted F1: {summary['average_weighted_f1']:.4f}")
    print("Standard Deviation")
    print(json.dumps({key: value for key, value in summary.items() if key.startswith("std_")}, indent=2))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Cross-validation results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    run_cross_validation()
