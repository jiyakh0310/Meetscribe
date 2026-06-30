"""Sentence classification boundary for the future ML MoM pipeline.

Purpose:
    Reserve the layer that will classify transcript sentences into MoM roles.

Responsibilities:
    - Classify sentences as Summary, Decision, Action Item, Discussion, or
      Information.
    - Combine embeddings and handcrafted features later.
    - Keep ANN inference isolated from parsing and template generation.

Inputs:
    Sentence features, embeddings, and optional cluster context.

Outputs:
    Sentence labels and confidence scores.

Future Implementation Notes:
    A trained ANN classifier will be introduced later. This skeleton contains no
    model training, loading, or inference.
"""

from dataclasses import dataclass


@dataclass(slots=True)
class SentenceClassification:
    """Classification result for one transcript sentence.

    Attributes:
        label: Predicted MoM class label.
        confidence: Confidence score from the future classifier.
    """

    label: str
    confidence: float


def classify_sentence_placeholder(sentence: str) -> SentenceClassification:
    """Return a neutral placeholder classification.

    Args:
        sentence: Sentence text to classify in a future implementation.

    Returns:
        Placeholder classification marked as ``Information`` with zero
        confidence.
    """

    # A neutral label prevents the skeleton from pretending to detect decisions
    # or action items before a trained ANN exists.
    _ = sentence

    # TODO:
    # Replace placeholder classifier with trained ANN model.
    return SentenceClassification(label="Information", confidence=0.0)
