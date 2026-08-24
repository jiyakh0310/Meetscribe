"""Single-audio inference service for Voxels."""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from voxels.audio import AudioPreprocessingError, AudioPreprocessor, AudioPreprocessorConfig
from voxels.labels import CANONICAL_LABELS
from voxels.wav2vec2 import DataCollatorWav2Vec2


class PredictionError(RuntimeError):
    """Raised when inference cannot produce a real model prediction."""


@dataclass(frozen=True)
class ConfidenceConfig:
    """Configurable confidence thresholds."""

    high_confidence: float = 0.75
    moderate_confidence: float = 0.45


@dataclass(frozen=True)
class EmotionPrediction:
    """Structured prediction result for one audio input."""

    predicted_emotion: str
    confidence: float
    probabilities: dict[str, float]
    top_three: list[dict[str, float | str]]
    original_sample_rate: int
    processed_sample_rate: int
    original_channel_count: int
    processed_duration: float
    quality_warnings: list[str]
    uncertainty_status: str
    inference_duration_seconds: float
    device: str
    note: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return asdict(self)


def classify_confidence(confidence: float, config: ConfidenceConfig) -> str:
    """Convert confidence score into a user-facing uncertainty status."""
    if config.high_confidence <= config.moderate_confidence:
        raise PredictionError("high_confidence must be greater than moderate_confidence.")
    if confidence >= config.high_confidence:
        return "high_confidence"
    if confidence >= config.moderate_confidence:
        return "moderate_confidence"
    return "low_confidence_uncertain"


class EmotionPredictor:
    """Reusable Voxels single-audio inference service."""

    def __init__(
        self,
        checkpoint_path: Path | str,
        audio_config: AudioPreprocessorConfig | None = None,
        confidence_config: ConfidenceConfig | None = None,
        model: torch.nn.Module | None = None,
        processor: Any | None = None,
        device: torch.device | None = None,
        preprocessor: AudioPreprocessor | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.audio_config = audio_config or AudioPreprocessorConfig()
        self.confidence_config = confidence_config or ConfidenceConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.processor = processor
        self.preprocessor = preprocessor or AudioPreprocessor(self.audio_config)
        self._loaded = model is not None
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path | str,
        audio_config: AudioPreprocessorConfig | None = None,
        confidence_config: ConfidenceConfig | None = None,
    ) -> "EmotionPredictor":
        """Create a predictor and load model resources once."""
        predictor = cls(checkpoint_path, audio_config, confidence_config)
        predictor.load()
        return predictor

    def load(self) -> None:
        """Load the trained model and processor once."""
        if self._loaded:
            return
        if not self.checkpoint_path.exists():
            raise PredictionError(f"Checkpoint directory does not exist: {self.checkpoint_path}")
        if not self.checkpoint_path.is_dir():
            raise PredictionError(f"Checkpoint path is not a directory: {self.checkpoint_path}")
        if not (self.checkpoint_path / "config.json").exists():
            raise PredictionError(f"Checkpoint is missing config.json: {self.checkpoint_path}")
        has_weights = (self.checkpoint_path / "model.safetensors").exists() or (
            self.checkpoint_path / "pytorch_model.bin"
        ).exists()
        if not has_weights:
            raise PredictionError(f"Checkpoint is missing model weights: {self.checkpoint_path}")

        try:
            from transformers import AutoProcessor, Wav2Vec2ForSequenceClassification

            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(self.checkpoint_path)
            processor_markers = ("preprocessor_config.json", "processor_config.json", "tokenizer_config.json")
            if any((self.checkpoint_path / marker).exists() for marker in processor_markers):
                self.processor = AutoProcessor.from_pretrained(self.checkpoint_path)
            else:
                self.processor = None
        except Exception as exc:
            raise PredictionError(f"Failed to load model or processor from checkpoint: {exc}") from exc

        self.model.to(self.device)
        self.model.eval()
        self._loaded = True

    def predict_file(self, audio_path: Path | str) -> EmotionPrediction:
        """Run inference for a local audio file."""
        self.load()
        try:
            processed_audio = self.preprocessor.preprocess_file(audio_path, mode="inference")
        except AudioPreprocessingError as exc:
            raise PredictionError(f"Invalid audio input: {exc}") from exc
        return self._predict_processed(processed_audio)

    def predict_bytes(self, audio_bytes: bytes, suffix: str = ".wav") -> EmotionPrediction:
        """Run inference for uploaded audio bytes."""
        if not audio_bytes:
            raise PredictionError("Uploaded audio bytes are empty.")
        safe_suffix = suffix if suffix.startswith(".") else f".{suffix}"
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / f"upload{safe_suffix}"
            path.write_bytes(audio_bytes)
            return self.predict_file(path)

    def _predict_processed(self, processed_audio: Any) -> EmotionPrediction:
        if self.model is None:
            raise PredictionError("Model is not loaded.")
        start = time.perf_counter()
        batch = DataCollatorWav2Vec2(self.processor)(
            [
                {
                    "input_values": processed_audio.waveform,
                    "valid_sample_length": processed_audio.valid_sample_length,
                    "labels": 0,
                }
            ]
        )
        with torch.inference_mode():
            outputs = self.model(
                input_values=batch["input_values"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
            )
            probabilities_tensor = torch.softmax(outputs.logits, dim=-1).detach().cpu()[0]
        if probabilities_tensor.numel() != len(CANONICAL_LABELS):
            raise PredictionError(
                f"Model returned {probabilities_tensor.numel()} probabilities; expected {len(CANONICAL_LABELS)}."
            )

        probabilities = {
            label: float(probabilities_tensor[index].item()) for index, label in enumerate(CANONICAL_LABELS)
        }
        top_three = [
            {"emotion": label, "probability": probability}
            for label, probability in sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:3]
        ]
        predicted_emotion = str(top_three[0]["emotion"])
        confidence = float(top_three[0]["probability"])
        return EmotionPrediction(
            predicted_emotion=predicted_emotion,
            confidence=confidence,
            probabilities=probabilities,
            top_three=top_three,
            original_sample_rate=int(processed_audio.original_sample_rate),
            processed_sample_rate=int(processed_audio.processed_sample_rate),
            original_channel_count=int(processed_audio.original_channels),
            processed_duration=float(processed_audio.duration),
            quality_warnings=list(processed_audio.quality_warnings),
            uncertainty_status=classify_confidence(confidence, self.confidence_config),
            inference_duration_seconds=float(time.perf_counter() - start),
            device=str(self.device),
            note="This is a speech-emotion model prediction, not a measure of a person's true internal psychological state.",
        )


def prediction_to_json(prediction: EmotionPrediction) -> str:
    """Serialize a prediction for command-line output."""
    return json.dumps(prediction.to_dict(), indent=2)
