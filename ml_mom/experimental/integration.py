"""Reusable integration layer for the Experimental MoM Formatter V4.

Purpose:
    Provide one backend function that turns final reviewed transcript text into
    an experimental Minutes of Meeting document using the existing ANN inference
    path and the V4 deterministic formatter.

Responsibilities:
    - Accept transcript text from any caller, including Streamlit audio and
      transcript-upload workflows.
    - Reuse the existing parser, preprocessing, feature extraction, MiniLM
      embeddings, ANN model loader, and label mapping helpers.
    - Run ``ExperimentalMomFormatter`` without duplicating formatter logic.
    - Return Markdown plus structured data so callers can adapt the result to
      their existing UI/export contracts.

Inputs:
    Final reviewed transcript text after speaker and transcript edits.

Outputs:
    ``GeneratedMomResult`` containing the experimental MoM object, Markdown,
    plain text, structured JSON-compatible data, ANN predictions, and a friendly
    error message when generation cannot continue.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ml_mom.experimental.env_compat import patch_experimental_embedding_environment

patch_experimental_embedding_environment()

import ml_mom.embeddings as embeddings_module

patch_experimental_embedding_environment(embeddings_module)

from ml_mom.embeddings import EmbeddingService
from ml_mom.experimental.mom_formatter import ExperimentalMom
from ml_mom.experimental.mom_formatter import ExperimentalMomFormatter
from ml_mom.local_gemma_rewriter import LocalGemmaRewriter
from ml_mom.predict_ann import (
    PredictionResult,
    generate_embedding_tensor,
    load_label_mapping,
    load_trained_model,
    predict_labels,
    prepare_sentence_features,
)


@dataclass(slots=True)
class GeneratedMomResult:
    """Container returned by the reusable experimental MoM integration.

    Attributes:
        experimental_mom: Structured V4 formatter output, when generation
            succeeds.
        predictions: Sentence-level ANN predictions used by the formatter.
        markdown: Markdown render of the generated MoM.
        deterministic_markdown: Unmodified formatter Markdown retained as the
            internal source of truth.
        text: Plain-text render of the generated MoM.
        structured_json: JSON-compatible representation of the generated MoM.
        sentence_count: Number of sentence features passed to MiniLM/ANN.
        error_message: Friendly message when a required ML stage is unavailable.
    """

    experimental_mom: ExperimentalMom | None = None
    deterministic_mom: ExperimentalMom | None = None
    predictions: list[PredictionResult] | None = None
    markdown: str = ""
    deterministic_markdown: str = ""
    text: str = ""
    structured_json: dict[str, Any] | None = None
    sentence_count: int = 0
    error_message: str | None = None
    gemma_refinement_applied: bool = False
    gemma_model: str = ""
    gemma_error: str | None = None

    @property
    def is_valid(self) -> bool:
        """Return whether generation completed successfully."""

        return self.experimental_mom is not None and not self.error_message


def generate_mom(
    transcript_text: str,
    *,
    meeting_title: str = "Minutes of Meeting",
    meeting_date: str = "",
    participants: list[str] | None = None,
    audio_quality_mode: bool = False,
) -> GeneratedMomResult:
    """Generate MoM output with MiniLM, ANN inference, and Formatter V4.

    Args:
        transcript_text: Final edited transcript text.
        meeting_title: Optional title to pass into the formatter.
        meeting_date: Optional meeting date to pass into the formatter.
        participants: Optional manually confirmed participants. When omitted,
            the formatter infers participants from sentence speakers.

    Returns:
        ``GeneratedMomResult`` with Markdown and structured output. The function
        never raises for expected ML availability or validation failures; those
        are returned as friendly ``error_message`` values.
    """

    if not transcript_text or not transcript_text.strip():
        return GeneratedMomResult(error_message="Transcript cannot be empty.")

    # The existing helper keeps runtime inference aligned with the standalone
    # prediction CLI: parser -> preprocessing -> feature extraction.
    sentence_features = prepare_sentence_features(transcript_text)
    if not sentence_features:
        return GeneratedMomResult(
            error_message="No usable transcript sentences were found for meeting analysis."
        )

    # MiniLM embeddings are produced by the existing embedding service so the
    # production path and experimental path use the same vector representation.
    embedding_service = EmbeddingService()
    embedding_tensor = generate_embedding_tensor(sentence_features, embedding_service)
    if embedding_tensor is None:
        return GeneratedMomResult(
            sentence_count=len(sentence_features),
            error_message=(
                embedding_service.error_message
                or "Sentence embeddings could not be generated for this transcript."
            ),
        )

    id_to_label = load_label_mapping()
    if not id_to_label:
        return GeneratedMomResult(
            sentence_count=len(sentence_features),
            error_message=(
                "The ML label mapping is missing or invalid. Please train the ANN model first."
            ),
        )

    model = load_trained_model(
        embedding_dimension=embedding_tensor.size(1),
        class_count=len(id_to_label),
    )
    if model is None:
        return GeneratedMomResult(
            sentence_count=len(sentence_features),
            error_message=(
                "The trained ANN model is missing or invalid. Please train the ANN model first."
            ),
        )

    predictions = predict_labels(
        model=model,
        embedding_tensor=embedding_tensor,
        sentence_features=sentence_features,
        id_to_label=id_to_label,
    )
    if not predictions:
        return GeneratedMomResult(
            sentence_count=len(sentence_features),
            error_message="The ANN model did not return any sentence predictions.",
        )

    formatter = ExperimentalMomFormatter()
    experimental_mom = formatter.format(
        predictions,
        meeting_title=meeting_title,
        meeting_date=meeting_date,
        participants=participants,
    )
    rewrite = LocalGemmaRewriter().rewrite(
        experimental_mom,
        audio_quality_mode=audio_quality_mode,
    )
    output_mom = rewrite.mom

    return GeneratedMomResult(
        experimental_mom=output_mom,
        deterministic_mom=experimental_mom,
        predictions=predictions,
        markdown=output_mom.to_markdown(),
        deterministic_markdown=experimental_mom.to_markdown(),
        text=output_mom.to_text(),
        structured_json=experimental_mom_to_dict(output_mom),
        sentence_count=len(sentence_features),
        gemma_refinement_applied=rewrite.applied,
        gemma_model=rewrite.model,
        gemma_error=rewrite.error,
    )


def experimental_mom_to_dict(experimental_mom: ExperimentalMom) -> dict[str, Any]:
    """Convert formatter output into a JSON-compatible dictionary.

    Args:
        experimental_mom: Structured V4 formatter object.

    Returns:
        Dictionary suitable for logging, storage, or future API responses.
    """

    # ``asdict`` preserves nested dataclass fields such as action items without
    # forcing downstream callers to know the formatter's internal class names.
    return asdict(experimental_mom)
