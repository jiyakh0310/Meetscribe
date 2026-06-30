"""Run ANN predictions for transcript sentences.

Purpose:
    Provide an offline inference utility for the trained ML-based Minutes of
    Meeting sentence classifier.

Responsibilities:
    - Accept a transcript text file.
    - Parse the transcript with the existing transcript parser.
    - Preprocess parsed turns with the existing preprocessing module.
    - Extract sentence features with the existing feature extraction module.
    - Generate sentence embeddings with the existing embedding service.
    - Load the trained ANN model and label mapping.
    - Save structured prediction output to JSON and CSV.

Inputs:
    A speaker-labeled transcript text file.

Outputs:
    - ``datasets/predictions/<transcript_name>_prediction.json``
    - ``datasets/predictions/<transcript_name>_prediction.csv``

Safety:
    This script is not part of the production MeetScribe workflow and does not
    modify production UI, email, workflow, parser logic, training logic, or
    export modules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import re
import sys
from typing import Any

import torch

try:
    from ml_mom.ann_model import SentenceClassifierANN
    from ml_mom.embeddings import EmbeddingService
    from ml_mom.feature_extraction import SentenceFeature, extract_sentence_features
    from ml_mom.preprocessing import preprocess_transcript
    from ml_mom.training_dataset import LABEL_TO_ID
    from ml_mom.transcript_parser import parse_transcript
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution.
    from ann_model import SentenceClassifierANN
    from embeddings import EmbeddingService
    from feature_extraction import SentenceFeature, extract_sentence_features
    from preprocessing import preprocess_transcript
    from training_dataset import LABEL_TO_ID
    from transcript_parser import parse_transcript


MODEL_PATH = Path("datasets") / "models" / "best_model.pt"
LABEL_MAPPING_PATH = Path("datasets") / "models" / "label_mapping.json"
PREDICTION_DIR = Path("datasets") / "predictions"


@dataclass(slots=True)
class PredictionResult:
    """Prediction output for one transcript sentence.

    Attributes:
        sentence: Original preprocessed sentence text.
        speaker: Speaker associated with the sentence.
        timestamp: Timestamp in seconds, when available.
        predicted_label: Predicted MoM sentence label.
        confidence_score: Probability assigned to the predicted class.
    """

    sentence: str
    speaker: str
    timestamp: int | None
    predicted_label: str
    confidence_score: float


def load_transcript_text(transcript_path: Path) -> str | None:
    """Load transcript text safely.

    Args:
        transcript_path: Transcript text file path.

    Returns:
        Transcript text, or ``None`` when the file cannot be read.
    """

    try:
        return transcript_path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        print(f"Transcript file not found: {transcript_path}")
    except UnicodeDecodeError as exc:
        print(f"Transcript file is not valid UTF-8: {exc}")
    except OSError as exc:
        print(f"Transcript file could not be read: {exc}")
    return None


def prediction_output_paths(transcript_path: Path) -> tuple[Path, Path]:
    """Build transcript-specific output paths for prediction artifacts.

    Args:
        transcript_path: Transcript file path supplied by the user.

    Returns:
        JSON and CSV output paths inside ``datasets/predictions``.
    """

    # Use the transcript filename so multiple prediction runs do not overwrite
    # each other when annotators or evaluators work with several meetings.
    transcript_stem = transcript_path.stem or "transcript"

    # Keep filenames filesystem-friendly while preserving the recognizable
    # meeting name for auditability of generated prediction outputs.
    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", transcript_stem).strip("_")
    safe_stem = safe_stem or "transcript"

    json_path = PREDICTION_DIR / f"{safe_stem}_prediction.json"
    csv_path = PREDICTION_DIR / f"{safe_stem}_prediction.csv"
    return json_path, csv_path


def prepare_sentence_features(transcript_text: str) -> list[SentenceFeature]:
    """Parse, preprocess, and feature-extract transcript sentences.

    Args:
        transcript_text: Raw transcript text.

    Returns:
        Sentence features ready for embedding generation.
    """

    parsed = parse_transcript(transcript_text)
    if not parsed.is_valid:
        print(parsed.validation_message or "Transcript could not be parsed.")
        return []

    preprocessed_turns = preprocess_transcript(parsed.turns)
    sentence_features = extract_sentence_features(preprocessed_turns)
    if not sentence_features:
        print("No sentences were found after parsing and preprocessing.")
    return sentence_features


def load_label_mapping(path: Path = LABEL_MAPPING_PATH) -> dict[int, str] | None:
    """Load label mapping and invert it for class-ID lookup.

    Args:
        path: Label mapping JSON path.

    Returns:
        Mapping from numeric class IDs to labels, or ``None`` on failure.
    """

    try:
        label_to_id = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"Label mapping not found: {path}")
        return None
    except json.JSONDecodeError as exc:
        print(f"Label mapping JSON is invalid: {exc}")
        return None

    return {int(class_id): label for label, class_id in label_to_id.items()}


def load_trained_model(
    embedding_dimension: int,
    class_count: int,
    model_path: Path = MODEL_PATH,
) -> SentenceClassifierANN | None:
    """Load the trained ANN model safely.

    Args:
        embedding_dimension: Input dimension expected by the saved model.
        class_count: Number of output classes.
        model_path: Saved model path.

    Returns:
        Loaded model in evaluation mode, or ``None`` on failure.
    """

    if not model_path.exists():
        print(f"Trained model not found: {model_path}")
        return None

    try:
        model = SentenceClassifierANN(
            input_dim=embedding_dimension,
            output_dim=class_count,
        )
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as exc:
        print(f"Trained model could not be loaded: {exc}")
        return None


def generate_embedding_tensor(
    sentence_features: list[SentenceFeature],
    embedding_service: EmbeddingService,
) -> torch.Tensor | None:
    """Generate an embedding tensor from sentence features.

    Args:
        sentence_features: Sentence features to embed.
        embedding_service: Existing embedding service.

    Returns:
        Embedding tensor, or ``None`` when embeddings are unavailable.
    """

    embedding_result = embedding_service.generate_embeddings(sentence_features)
    if embedding_result.error_message:
        print(embedding_result.error_message)
        return None

    vectors = [embedding.embedding_vector for embedding in embedding_result.embeddings]
    if not vectors:
        print("No embeddings were generated.")
        return None

    try:
        tensor = torch.tensor(vectors, dtype=torch.float32)
    except Exception as exc:
        print(f"Embeddings could not be converted to a tensor: {exc}")
        return None

    print(f"Generated embeddings: {tensor.size(0)}")
    print(f"Embedding tensor shape: {tuple(tensor.shape)}")
    return tensor


def predict_labels(
    model: SentenceClassifierANN,
    embedding_tensor: torch.Tensor,
    sentence_features: list[SentenceFeature],
    id_to_label: dict[int, str],
) -> list[PredictionResult]:
    """Predict labels and confidence scores for each sentence.

    Args:
        model: Loaded ANN model.
        embedding_tensor: Embedding tensor.
        sentence_features: Sentence metadata aligned to the embedding tensor.
        id_to_label: Numeric class ID to label mapping.

    Returns:
        Structured prediction results.
    """

    with torch.no_grad():
        probabilities = model.predict_probabilities(embedding_tensor)

    if probabilities.numel() == 0:
        print(model.last_error or "Model returned no probabilities.")
        return []

    results: list[PredictionResult] = []
    confidence_values, class_indices = torch.max(probabilities, dim=1)
    for feature, class_index, confidence in zip(
        sentence_features,
        class_indices.tolist(),
        confidence_values.tolist(),
    ):
        results.append(
            PredictionResult(
                sentence=feature.original_sentence,
                speaker=feature.speaker,
                timestamp=feature.timestamp,
                predicted_label=id_to_label.get(int(class_index), "Information"),
                confidence_score=round(float(confidence), 6),
            )
        )
    return results


def prediction_statistics(results: list[PredictionResult]) -> dict[str, Any]:
    """Calculate and print prediction statistics.

    Args:
        results: Prediction results.

    Returns:
        Statistics dictionary.
    """

    counts = {label: 0 for label in LABEL_TO_ID}
    for result in results:
        counts[result.predicted_label] = counts.get(result.predicted_label, 0) + 1

    average_confidence = (
        sum(result.confidence_score for result in results) / len(results)
        if results
        else 0.0
    )
    stats = {
        "Total Sentences": len(results),
        "Discussion Count": counts.get("Discussion", 0),
        "Decision Count": counts.get("Decision", 0),
        "Action Count": counts.get("Action_Item", 0),
        "Summary Count": counts.get("Summary", 0),
        "Information Count": counts.get("Information", 0),
        "Average Confidence": round(average_confidence, 6),
    }

    print("Prediction statistics")
    for key, value in stats.items():
        print(f"{key}: {value}")
    return stats


def save_predictions(results: list[PredictionResult], transcript_path: Path) -> None:
    """Save prediction results to JSON and CSV.

    Args:
        results: Prediction results to save.
        transcript_path: Source transcript path used to name the output files.
    """

    # Ensure the prediction directory exists before writing so the script can
    # run on a fresh checkout where datasets/predictions has not been created.
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    result_rows = [asdict(result) for result in results]
    json_path, csv_path = prediction_output_paths(transcript_path)

    # JSON preserves the structured result shape for downstream MoM generation.
    json_path.write_text(
        json.dumps(result_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # CSV gives dataset builders and reviewers a spreadsheet-friendly view.
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "sentence",
                "speaker",
                "timestamp",
                "predicted_label",
                "confidence_score",
            ],
        )
        writer.writeheader()
        writer.writerows(result_rows)

    print(f"Prediction JSON saved to: {json_path}")
    print(f"Prediction CSV saved to: {csv_path}")


def run_prediction(transcript_path: Path) -> list[PredictionResult]:
    """Run the full ANN prediction workflow.

    Args:
        transcript_path: Transcript text file path.

    Returns:
        Prediction results. Returns an empty list if any required stage fails.
    """

    transcript_text = load_transcript_text(transcript_path)
    if transcript_text is None or not transcript_text.strip():
        print("Transcript is empty or unavailable.")
        return []

    sentence_features = prepare_sentence_features(transcript_text)
    if not sentence_features:
        return []

    embedding_service = EmbeddingService()
    embedding_dimension = embedding_service.get_embedding_dimension()
    if embedding_dimension is None:
        print(embedding_service.error_message or "Embedding model is unavailable.")
        return []

    embedding_tensor = generate_embedding_tensor(sentence_features, embedding_service)
    if embedding_tensor is None:
        return []

    id_to_label = load_label_mapping()
    if id_to_label is None:
        return []

    model = load_trained_model(
        embedding_dimension=embedding_tensor.size(1),
        class_count=len(id_to_label),
    )
    if model is None:
        return []

    results = predict_labels(model, embedding_tensor, sentence_features, id_to_label)
    prediction_statistics(results)
    save_predictions(results, transcript_path)
    return results


def print_usage() -> None:
    """Print a friendly command-line usage message."""

    # The prediction script should only run against an explicit transcript so
    # reviewers know exactly which meeting produced each prediction artifact.
    print("Please provide a transcript path to run ANN prediction.")
    print("")
    print("Usage:")
    print("  python ml_mom/predict_ann.py datasets/raw_transcripts/meeting10.txt")


def main() -> None:
    """Command-line entry point for ANN prediction."""

    if len(sys.argv) < 2:
        print_usage()
        return

    # Resolve the user-supplied transcript argument without altering parser or
    # production workflow modules; this file remains a standalone inference CLI.
    transcript_path = Path(sys.argv[1])
    run_prediction(transcript_path)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    main()
