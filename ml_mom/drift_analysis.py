"""Embedding drift visualizations for MeetScribe datasets.

Purpose:
    Compare current and original dataset sentence embeddings for research-only
    drift analysis. This module never trains or saves ANN models.

Inputs:
    - Current dataset: ``datasets/processed/master_dataset.csv``
    - Original dataset: ``datasets/processed/master_dataset_old.csv``

Outputs:
    - ``datasets/models/pca.png``
    - ``datasets/models/tsne.png``
    - ``datasets/models/cosine_similarity.png``
    - ``datasets/models/drift_summary.json``
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
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

try:
    from ml_mom.embeddings import EmbeddingService
    from ml_mom.feature_extraction import SentenceFeature
    from ml_mom.training_dataset import LABEL_TO_ID
except ModuleNotFoundError:  # pragma: no cover - supports direct execution.
    from embeddings import EmbeddingService
    from feature_extraction import SentenceFeature
    from training_dataset import LABEL_TO_ID


CURRENT_DATASET_PATH = Path("datasets") / "processed" / "master_dataset.csv"
ORIGINAL_DATASET_PATH = Path("datasets") / "processed" / "master_dataset_old.csv"
MODEL_DIR = Path("datasets") / "models"
PCA_PATH = MODEL_DIR / "pca.png"
TSNE_PATH = MODEL_DIR / "tsne.png"
COSINE_PATH = MODEL_DIR / "cosine_similarity.png"
SUMMARY_PATH = MODEL_DIR / "drift_summary.json"
RANDOM_SEED = 42


def load_sentences(dataset_path: Path) -> list[str]:
    """Load valid labeled sentences from a dataset CSV."""

    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return []
    sentences: list[str] = []
    seen: set[str] = set()
    with dataset_path.open("r", newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            sentence = (row.get("sentence") or "").strip()
            label = (row.get("selected_label") or "").strip()
            if not sentence or label not in LABEL_TO_ID:
                continue
            key = sentence.casefold()
            if key in seen:
                continue
            seen.add(key)
            sentences.append(sentence)
    return sentences


def sentence_feature(sentence_id: int, sentence: str) -> SentenceFeature:
    """Build the minimal feature object required by ``EmbeddingService``."""

    words = sentence.split()
    average_word_length = sum(len(word) for word in words) / len(words) if words else 0.0
    return SentenceFeature(
        sentence_id=sentence_id,
        turn_id=0,
        speaker="",
        timestamp=None,
        original_sentence=sentence,
        word_count=len(words),
        character_count=len(sentence),
        average_word_length=round(average_word_length, 3),
        sentence_position=sentence_id,
        is_question=sentence.rstrip().endswith("?"),
        is_exclamation="!" in sentence,
        contains_number=any(char.isdigit() for char in sentence),
        contains_currency=False,
        contains_percentage="%" in sentence,
        contains_date=False,
        contains_time=False,
        contains_action_keyword=False,
        contains_decision_keyword=False,
        contains_discussion_keyword=False,
        contains_deadline_keyword=False,
        speaker_name="",
        speaker_turn_index=0,
        previous_sentence=None,
        next_sentence=None,
    )


def generate_embeddings(sentences: list[str]) -> np.ndarray | None:
    """Generate MiniLM embeddings for drift analysis sentences."""

    service = EmbeddingService()
    features = [
        sentence_feature(index, sentence)
        for index, sentence in enumerate(sentences, start=1)
    ]
    result = service.generate_embeddings(features)
    if result.error_message:
        print(result.error_message)
        return None
    vectors = [embedding.embedding_vector for embedding in result.embeddings if embedding.embedding_vector]
    if len(vectors) != len(sentences):
        print("Embedding count did not match sentence count.")
        return None
    return np.array(vectors, dtype=float)


def save_scatter_plot(
    coordinates: np.ndarray,
    labels: list[str],
    title: str,
    output_path: Path,
) -> None:
    """Save a two-dimensional scatter plot for dataset groups."""

    colors = {"Original": "#7c3aed", "Current": "#fb7185"}
    fig, axis = plt.subplots(figsize=(8, 6))
    for group in sorted(set(labels)):
        indices = [index for index, label in enumerate(labels) if label == group]
        axis.scatter(
            coordinates[indices, 0],
            coordinates[indices, 1],
            s=18,
            alpha=0.72,
            label=group,
            color=colors.get(group),
        )
    axis.set_title(title)
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_cosine_distribution(
    original_embeddings: np.ndarray,
    current_embeddings: np.ndarray,
) -> dict[str, float]:
    """Save histogram of current-to-original maximum cosine similarities."""

    similarities = cosine_similarity(current_embeddings, original_embeddings)
    max_similarities = similarities.max(axis=1)
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.hist(max_similarities, bins=30, color="#fb7185", edgecolor="#9f1239", alpha=0.8)
    axis.set_title("Current-to-Original Cosine Similarity Distribution")
    axis.set_xlabel("Maximum cosine similarity")
    axis.set_ylabel("Sentence count")
    fig.tight_layout()
    fig.savefig(COSINE_PATH)
    plt.close(fig)
    return {
        "mean_max_cosine_similarity": float(max_similarities.mean()),
        "min_max_cosine_similarity": float(max_similarities.min()),
        "max_max_cosine_similarity": float(max_similarities.max()),
    }


def centroid_summary(
    original_embeddings: np.ndarray,
    current_embeddings: np.ndarray,
) -> dict[str, float]:
    """Compute embedding centroid distance and cosine similarity."""

    original_centroid = original_embeddings.mean(axis=0)
    current_centroid = current_embeddings.mean(axis=0)
    centroid_distance = float(np.linalg.norm(current_centroid - original_centroid))
    centroid_cosine = float(
        cosine_similarity(
            current_centroid.reshape(1, -1),
            original_centroid.reshape(1, -1),
        )[0][0]
    )
    return {
        "centroid_euclidean_distance": centroid_distance,
        "centroid_cosine_similarity": centroid_cosine,
    }


def run_drift_analysis() -> None:
    """Generate PCA, t-SNE, cosine, and centroid drift artifacts."""

    original_sentences = load_sentences(ORIGINAL_DATASET_PATH)
    current_sentences = load_sentences(CURRENT_DATASET_PATH)
    if not original_sentences or not current_sentences:
        print("Both original and current datasets need valid sentences for drift analysis.")
        return
    original_embeddings = generate_embeddings(original_sentences)
    current_embeddings = generate_embeddings(current_sentences)
    if original_embeddings is None or current_embeddings is None:
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    combined = np.vstack([original_embeddings, current_embeddings])
    labels = ["Original"] * len(original_embeddings) + ["Current"] * len(current_embeddings)
    pca_coordinates = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(combined)
    save_scatter_plot(pca_coordinates, labels, "PCA Embedding Drift", PCA_PATH)

    perplexity = max(5, min(30, (len(combined) - 1) // 3))
    tsne_coordinates = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=RANDOM_SEED,
        init="pca",
        learning_rate="auto",
    ).fit_transform(combined)
    save_scatter_plot(tsne_coordinates, labels, "t-SNE Embedding Drift", TSNE_PATH)

    cosine_stats = save_cosine_distribution(original_embeddings, current_embeddings)
    summary: dict[str, Any] = {
        "current_dataset": str(CURRENT_DATASET_PATH),
        "original_dataset": str(ORIGINAL_DATASET_PATH),
        "current_sentence_count": len(current_sentences),
        "original_sentence_count": len(original_sentences),
        **centroid_summary(original_embeddings, current_embeddings),
        **cosine_stats,
        "pca_plot": str(PCA_PATH),
        "tsne_plot": str(TSNE_PATH),
        "cosine_similarity_plot": str(COSINE_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"PCA plot saved to: {PCA_PATH}")
    print(f"t-SNE plot saved to: {TSNE_PATH}")
    print(f"Cosine similarity plot saved to: {COSINE_PATH}")
    print(f"Drift summary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    run_drift_analysis()
