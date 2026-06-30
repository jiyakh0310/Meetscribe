"""Artificial Neural Network architecture for sentence classification.

Purpose:
    Define the PyTorch neural network that will later classify transcript
    sentences into Minutes of Meeting categories.

Responsibilities:
    - Accept sentence embedding vectors as model input.
    - Produce logits for the five supported MoM sentence classes.
    - Provide helper methods for logits, probabilities, and predicted classes.
    - Validate input dimensions and empty tensors without crashing.

Inputs:
    Embedding tensors with default dimension ``384``.

Outputs:
    Logits, probability distributions, and predicted class labels for:
    ``Discussion``, ``Decision``, ``Action_Item``, ``Summary``, and
    ``Information``.

Future Implementation Notes:
    This module defines architecture only. It does not train the model, load
    datasets, save checkpoints, or integrate with the production application.
"""

import sys
from typing import Any

import torch
from torch import nn


CLASS_LABELS = (
    "Discussion",
    "Decision",
    "Action_Item",
    "Summary",
    "Information",
)
DEFAULT_INPUT_DIM = 384
DEFAULT_HIDDEN_DIM_1 = 256
DEFAULT_HIDDEN_DIM_2 = 128
DEFAULT_OUTPUT_DIM = len(CLASS_LABELS)
DEFAULT_DROPOUT = 0.3


class SentenceClassifierANN(nn.Module):
    """Feed-forward ANN for MoM sentence classification.

    The model maps one sentence embedding to class logits. It intentionally
    stays simple and explainable for the first supervised classifier phase.
    """

    def __init__(
        self,
        input_dim: int = DEFAULT_INPUT_DIM,
        hidden_dim_1: int = DEFAULT_HIDDEN_DIM_1,
        hidden_dim_2: int = DEFAULT_HIDDEN_DIM_2,
        output_dim: int = DEFAULT_OUTPUT_DIM,
        dropout_rate: float = DEFAULT_DROPOUT,
    ) -> None:
        """Initialize the sentence classifier architecture.

        Args:
            input_dim: Expected embedding vector dimension.
            hidden_dim_1: Width of the first hidden representation.
            hidden_dim_2: Width of the second hidden representation.
            output_dim: Number of output classes.
            dropout_rate: Dropout probability used after hidden activations.
        """

        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.class_labels = CLASS_LABELS[:output_dim]
        self.last_error: str | None = None

        # Input layer projects dense sentence embeddings into a task-specific
        # feature space for MoM classification.
        self.input_layer = nn.Linear(input_dim, hidden_dim_1)

        # ReLU introduces non-linearity so the model can learn interactions
        # between semantic embedding dimensions.
        self.activation_1 = nn.ReLU()

        # Dropout reduces overfitting risk once real training begins.
        self.dropout_1 = nn.Dropout(dropout_rate)

        # Second hidden layer compresses the learned representation before the
        # final class decision layer.
        self.hidden_layer_2 = nn.Linear(hidden_dim_1, hidden_dim_2)

        # A second activation gives the compact representation non-linear
        # capacity without making the initial architecture too deep.
        self.activation_2 = nn.ReLU()

        # Second dropout layer regularizes the deeper representation.
        self.dropout_2 = nn.Dropout(dropout_rate)

        # Output layer emits raw logits for the supported sentence classes.
        self.output_layer = nn.Linear(hidden_dim_2, output_dim)

        # TODO:
        # Add optional BiLSTM support for sentence-sequence context.
        # TODO:
        # Add optional Transformer encoder support for richer context modeling.
        # TODO:
        # Add hyperparameter tuning hooks for hidden dimensions and dropout.

    def forward(self, embeddings: Any) -> torch.Tensor:
        """Run a forward pass through the ANN.

        Args:
            embeddings: Tensor-like embedding input with shape
                ``(batch_size, input_dim)`` or ``(input_dim,)``.

        Returns:
            Logit tensor with shape ``(batch_size, output_dim)``. Invalid input
            returns an empty tensor and stores a friendly message in
            ``last_error``.
        """

        valid_embeddings = self._validate_embeddings(embeddings)
        if valid_embeddings is None:
            return self._empty_output()

        hidden_1 = self.input_layer(valid_embeddings)
        hidden_1 = self.activation_1(hidden_1)
        hidden_1 = self.dropout_1(hidden_1)

        hidden_2 = self.hidden_layer_2(hidden_1)
        hidden_2 = self.activation_2(hidden_2)
        hidden_2 = self.dropout_2(hidden_2)

        return self.output_layer(hidden_2)

    def predict_logits(self, embeddings: Any) -> torch.Tensor:
        """Predict class logits without training side effects.

        Args:
            embeddings: Tensor-like embedding input.

        Returns:
            Logit tensor. Invalid input returns an empty tensor.
        """

        self.eval()
        with torch.no_grad():
            return self.forward(embeddings)

    def predict_probabilities(self, embeddings: Any) -> torch.Tensor:
        """Predict class probability distributions.

        Args:
            embeddings: Tensor-like embedding input.

        Returns:
            Probability tensor with shape ``(batch_size, output_dim)``.
            Invalid input returns an empty tensor.
        """

        logits = self.predict_logits(embeddings)
        if logits.numel() == 0:
            return logits

        # Softmax converts raw logits into a distribution that later evaluation
        # code can threshold or inspect per class.
        return torch.softmax(logits, dim=1)

    def predict_class(self, embeddings: Any) -> list[str]:
        """Predict class labels for embedding inputs.

        Args:
            embeddings: Tensor-like embedding input.

        Returns:
            List of predicted class labels. Invalid input returns an empty list.
        """

        probabilities = self.predict_probabilities(embeddings)
        if probabilities.numel() == 0:
            return []

        class_indices = torch.argmax(probabilities, dim=1).tolist()
        return [self.class_labels[int(index)] for index in class_indices]

    def _validate_embeddings(self, embeddings: Any) -> torch.Tensor | None:
        """Validate and normalize embedding input.

        Args:
            embeddings: Tensor-like embedding input.

        Returns:
            Float tensor with shape ``(batch_size, input_dim)``, or ``None`` if
            validation fails.
        """

        try:
            tensor = torch.as_tensor(embeddings, dtype=torch.float32)
        except Exception as exc:
            self.last_error = f"Embedding input could not be converted to a tensor: {exc}"
            return None

        if tensor.numel() == 0:
            self.last_error = "Embedding input is empty."
            return None

        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)

        if tensor.dim() != 2:
            self.last_error = "Embedding input must be a 1D or 2D tensor."
            return None

        if tensor.shape[1] != self.input_dim:
            self.last_error = (
                f"Expected embedding dimension {self.input_dim}, "
                f"received {tensor.shape[1]}."
            )
            return None

        if not torch.isfinite(tensor).all():
            self.last_error = "Embedding input contains NaN or infinite values."
            return None

        self.last_error = None
        return tensor

    def _empty_output(self) -> torch.Tensor:
        """Create an empty output tensor for invalid inputs.

        Returns:
            Empty tensor with shape ``(0, output_dim)``.
        """

        return torch.empty((0, self.output_dim), dtype=torch.float32)


def model_demo() -> None:
    """Run a no-training demo forward pass with dummy embeddings.

    Returns:
        None. Prints output shape, probability distribution, and predicted
        class labels.
    """

    torch.manual_seed(42)
    model = SentenceClassifierANN()
    model.eval()

    dummy_embeddings = torch.randn(3, DEFAULT_INPUT_DIM)
    logits = model.predict_logits(dummy_embeddings)
    probabilities = model.predict_probabilities(dummy_embeddings)
    predicted_classes = model.predict_class(dummy_embeddings)

    print("Output tensor shape")
    print(tuple(logits.shape))
    print("Probability distribution")
    print(probabilities)
    print("Predicted class")
    print(predicted_classes)

    invalid_logits = model.predict_logits(torch.randn(1, 10))
    print("Invalid input output shape")
    print(tuple(invalid_logits.shape))
    print("Invalid input message")
    print(model.last_error)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    model_demo()
