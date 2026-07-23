"""Run the isolated experimental MoM formatter.

Purpose:
    Provide a reversible research-only runner that uses the existing trained
    ANN inference path and then formats predictions with the experimental
    deterministic formatter.

Responsibilities:
    - Load an existing transcript from disk.
    - Reuse the existing parser, preprocessing, feature extraction, MiniLM
      embedding service, ANN model loader, and label mapping helpers.
    - Avoid the production prediction writer so this experiment never writes
      into the normal prediction or MoM output locations.
    - Save only experimental Markdown and text artifacts.

Inputs:
    Optional transcript path. If omitted, the runner uses
    ``datasets/raw_transcripts/meeting10.txt`` when available.

Outputs:
    - ``datasets/models/experimental/experimental_mom_output.md``
    - ``datasets/models/experimental/experimental_mom_output.txt``

Safety:
    This file is standalone and does not modify production imports, models,
    datasets, training scripts, Streamlit UI, exports, or inference modules.
"""

from __future__ import annotations

from pathlib import Path
import sys


# When this file is executed directly, Python starts import resolution from the
# experimental directory. Adding the project root keeps imports pointed at the
# existing project modules without changing any production import path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_mom.experimental.env_compat import patch_experimental_embedding_environment

patch_experimental_embedding_environment()

import ml_mom.embeddings as embeddings_module

patch_experimental_embedding_environment(embeddings_module)

from ml_mom.embeddings import EmbeddingService
from ml_mom.experimental.mom_formatter import ExperimentalMomFormatter
from ml_mom.predict_ann import (
    generate_embedding_tensor,
    load_label_mapping,
    load_trained_model,
    load_transcript_text,
    predict_labels,
    prediction_statistics,
    prepare_sentence_features,
)


DEFAULT_TRANSCRIPT_PATH = Path("datasets") / "raw_transcripts" / "meeting10.txt"
EXPERIMENT_OUTPUT_DIR = Path("datasets") / "models" / "experimental"
MARKDOWN_OUTPUT_PATH = EXPERIMENT_OUTPUT_DIR / "experimental_mom_output.md"
TEXT_OUTPUT_PATH = EXPERIMENT_OUTPUT_DIR / "experimental_mom_output.txt"


def resolve_transcript_path(argv: list[str]) -> Path:
    """Resolve the transcript path for the formatter demo.

    Args:
        argv: Command-line arguments excluding the Python executable name.

    Returns:
        A transcript path supplied by the caller, or the default real transcript
        used for local research demos.
    """

    # The runner accepts a transcript path for controlled experiments while
    # keeping a local default so reviewers can run it quickly on the existing
    # project data.
    if argv:
        return Path(argv[0])
    return DEFAULT_TRANSCRIPT_PATH


def run_experimental_formatter(transcript_path: Path) -> bool:
    """Run existing ANN inference and format the result experimentally.

    Args:
        transcript_path: Path to a speaker-labeled transcript text file.

    Returns:
        ``True`` when experimental artifacts are written, otherwise ``False``.
    """

    transcript_text = load_transcript_text(transcript_path)
    if transcript_text is None or not transcript_text.strip():
        print("Transcript is empty or unavailable.")
        return False

    sentence_features = prepare_sentence_features(transcript_text)
    if not sentence_features:
        print("No sentence features were produced from the transcript.")
        return False

    embedding_service = EmbeddingService()
    embedding_dimension = embedding_service.get_embedding_dimension()
    if embedding_dimension is None:
        print(embedding_service.error_message or "Embedding model is unavailable.")
        return False

    embedding_tensor = generate_embedding_tensor(sentence_features, embedding_service)
    if embedding_tensor is None:
        print("Embedding tensor could not be generated.")
        return False

    id_to_label = load_label_mapping()
    if id_to_label is None:
        print("Label mapping could not be loaded.")
        return False

    model = load_trained_model(
        embedding_dimension=embedding_tensor.size(1),
        class_count=len(id_to_label),
    )
    if model is None:
        print("Trained ANN model could not be loaded.")
        return False

    predictions = predict_labels(model, embedding_tensor, sentence_features, id_to_label)
    if not predictions:
        print("No ANN predictions were produced.")
        return False

    prediction_statistics(predictions)

    formatter = ExperimentalMomFormatter()
    experimental_mom = formatter.format(
        predictions,
        meeting_title=transcript_path.stem.replace("_", " ").replace("-", " ").title(),
    )

    # The experiment writes only under datasets/models/experimental so deleting
    # this folder fully removes all artifacts produced by this prototype.
    EXPERIMENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MARKDOWN_OUTPUT_PATH.write_text(experimental_mom.to_markdown(), encoding="utf-8")
    TEXT_OUTPUT_PATH.write_text(experimental_mom.to_text(), encoding="utf-8")

    print(f"Experimental Markdown saved to: {MARKDOWN_OUTPUT_PATH}")
    print(f"Experimental text saved to: {TEXT_OUTPUT_PATH}")
    return True


def main() -> None:
    """Command-line entry point for the experimental formatter demo."""

    transcript_path = resolve_transcript_path(sys.argv[1:])
    print(f"Using transcript: {transcript_path}")
    run_experimental_formatter(transcript_path)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    main()
