"""Wav2Vec2 dataset, collation, and model helpers for Voxels."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from voxels.audio import AudioPreprocessor
from voxels.labels import CANONICAL_LABELS, ID2LABEL, LABEL2ID

TransformerMode = Literal["head_only", "upper_layers", "full_transformer"]


class Wav2Vec2IntegrationError(RuntimeError):
    """Raised when Wav2Vec2 integration cannot proceed safely."""


def label2id() -> dict[str, int]:
    """Return the shared canonical label-to-id mapping."""
    return dict(LABEL2ID)


def id2label() -> dict[int, str]:
    """Return the shared canonical id-to-label mapping."""
    return dict(ID2LABEL)


@dataclass(frozen=True)
class Wav2Vec2Item:
    """One preprocessed dataset item before dynamic padding."""

    input_values: np.ndarray
    label: int
    attention_mask: np.ndarray
    file_path: str
    emotion: str
    valid_sample_length: int
    quality_warnings: list[str]


class Wav2Vec2SpeechDataset(Dataset):
    """PyTorch dataset that reads split metadata rows and raw audio files."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        preprocessor: AudioPreprocessor,
        mode: Literal["training", "validation", "inference"] = "training",
        random_crop: bool = False,
        augmentation_seed: int | None = None,
    ) -> None:
        required = {"file_path", "emotion"}
        missing = sorted(required.difference(metadata.columns))
        if missing:
            raise Wav2Vec2IntegrationError(f"Metadata is missing required columns: {missing}")
        self.metadata = metadata.reset_index(drop=True)
        self.preprocessor = preprocessor
        self.mode = mode
        self.random_crop = random_crop
        self.augmentation_seed = augmentation_seed

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.metadata.iloc[index]
        emotion = str(row["emotion"])
        if emotion not in LABEL2ID:
            raise Wav2Vec2IntegrationError(f"Unknown emotion label: {emotion}")
        rng = random.Random(self.augmentation_seed + index) if self.augmentation_seed is not None else None
        result = self.preprocessor.preprocess_file(
            Path(str(row["file_path"])),
            mode=self.mode,
            random_crop=self.random_crop,
            rng=rng,
        )
        waveform = result.waveform.astype(np.float32, copy=False)
        attention_mask = np.zeros(waveform.shape[0], dtype=np.int64)
        attention_mask[: result.valid_sample_length] = 1
        return {
            "input_values": waveform,
            "attention_mask": attention_mask,
            "labels": int(LABEL2ID[emotion]),
            "file_path": str(row["file_path"]),
            "filename": Path(str(row["file_path"])).name,
            "emotion": emotion,
            "valid_sample_length": result.valid_sample_length,
            "quality_warnings": result.quality_warnings,
        }


class DataCollatorWav2Vec2:
    """Dynamically pad raw waveform batches and create attention masks."""

    def __init__(self, processor: Any | None = None, padding: bool | str = True) -> None:
        self.processor = processor
        self.padding = padding

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        labels = torch.tensor([feature["labels"] for feature in features], dtype=torch.long)
        input_features = [feature["input_values"] for feature in features]

        if self.processor is not None:
            batch = self.processor(
                input_features,
                sampling_rate=16000,
                padding=self.padding,
                return_attention_mask=True,
                return_tensors="pt",
            )
            batch["attention_mask"] = self._manual_attention_mask(features, batch["input_values"].shape[1])
        else:
            batch = self._manual_pad(features)

        batch["labels"] = labels
        batch["file_path"] = [str(feature.get("file_path", "")) for feature in features]
        batch["filename"] = [str(feature.get("filename", Path(str(feature.get("file_path", ""))).name)) for feature in features]
        batch["emotion"] = [str(feature.get("emotion", "")) for feature in features]
        batch["valid_sample_length"] = torch.tensor(
            [int(feature["valid_sample_length"]) for feature in features],
            dtype=torch.long,
        )
        return batch

    @staticmethod
    def _manual_attention_mask(features: list[dict[str, Any]], padded_length: int) -> torch.Tensor:
        mask = torch.zeros((len(features), padded_length), dtype=torch.long)
        for row_index, feature in enumerate(features):
            valid_length = min(int(feature["valid_sample_length"]), padded_length)
            mask[row_index, :valid_length] = 1
        return mask

    def _manual_pad(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_length = max(int(feature["input_values"].shape[0]) for feature in features)
        input_values = torch.zeros((len(features), max_length), dtype=torch.float32)
        attention_mask = torch.zeros((len(features), max_length), dtype=torch.long)
        for row_index, feature in enumerate(features):
            waveform = torch.tensor(feature["input_values"], dtype=torch.float32)
            length = waveform.shape[0]
            input_values[row_index, :length] = waveform
            valid_length = min(int(feature["valid_sample_length"]), length)
            attention_mask[row_index, :valid_length] = 1
        return {"input_values": input_values, "attention_mask": attention_mask}


def build_wav2vec2_config_kwargs() -> dict[str, Any]:
    """Return label configuration for Wav2Vec2ForSequenceClassification."""
    return {
        "num_labels": len(CANONICAL_LABELS),
        "label2id": label2id(),
        "id2label": id2label(),
        "apply_spec_augment": False,
        "mask_time_prob": 0.0,
        "mask_feature_prob": 0.0,
    }


def load_processor(pretrained_name: str, local_files_only: bool = False) -> Any:
    """Load the appropriate Wav2Vec2 processor for the installed Transformers version."""
    try:
        from transformers import AutoProcessor

        return AutoProcessor.from_pretrained(pretrained_name, local_files_only=local_files_only)
    except Exception as auto_exc:
        try:
            from transformers import Wav2Vec2Processor

            return Wav2Vec2Processor.from_pretrained(pretrained_name, local_files_only=local_files_only)
        except Exception as wav_exc:
            raise Wav2Vec2IntegrationError(
                f"Could not load Wav2Vec2 processor '{pretrained_name}'. "
                f"AutoProcessor error: {auto_exc}; Wav2Vec2Processor error: {wav_exc}"
            ) from wav_exc


def load_sequence_classifier(pretrained_name: str, local_files_only: bool = False) -> Any:
    """Load Wav2Vec2ForSequenceClassification with the shared label mapping."""
    try:
        from transformers import Wav2Vec2ForSequenceClassification

        return Wav2Vec2ForSequenceClassification.from_pretrained(
            pretrained_name,
            local_files_only=local_files_only,
            **build_wav2vec2_config_kwargs(),
        )
    except Exception as exc:
        raise Wav2Vec2IntegrationError(
            f"Could not load pretrained checkpoint '{pretrained_name}'. "
            "The checkpoint may be unavailable locally, internet may be blocked, or dependencies may be incomplete. "
            f"Original error: {exc}"
        ) from exc


def freeze_cnn_feature_encoder(model: Any) -> None:
    """Freeze the Wav2Vec2 CNN feature encoder."""
    if hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()
    feature_extractor = getattr(getattr(model, "wav2vec2", None), "feature_extractor", None)
    if feature_extractor is None:
        raise Wav2Vec2IntegrationError("Model does not expose wav2vec2.feature_extractor.")
    for parameter in feature_extractor.parameters():
        parameter.requires_grad = False


def apply_transformer_mode(
    model: Any,
    mode: TransformerMode,
    trainable_upper_layers: int = 4,
) -> None:
    """Configure trainable Wav2Vec2 parameters for the requested fine-tuning mode."""
    if mode not in {"head_only", "upper_layers", "full_transformer"}:
        raise Wav2Vec2IntegrationError(f"Invalid transformer mode: {mode}")

    freeze_cnn_feature_encoder(model)
    wav2vec2 = getattr(model, "wav2vec2", None)
    if wav2vec2 is None:
        raise Wav2Vec2IntegrationError("Model does not expose wav2vec2 base model.")

    if mode == "full_transformer":
        for name, parameter in model.named_parameters():
            if "feature_extractor" not in name:
                parameter.requires_grad = True
        freeze_cnn_feature_encoder(model)
        return

    for parameter in wav2vec2.parameters():
        parameter.requires_grad = False

    if mode == "upper_layers":
        layers = list(getattr(getattr(wav2vec2, "encoder", None), "layers", []))
        if not layers:
            raise Wav2Vec2IntegrationError("Model does not expose transformer encoder layers.")
        count = max(0, min(trainable_upper_layers, len(layers)))
        for layer in layers[-count:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    for name, parameter in model.named_parameters():
        if not name.startswith("wav2vec2."):
            parameter.requires_grad = True


def parameter_counts(model: Any) -> dict[str, int]:
    """Report total, frozen, and trainable parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen = total - trainable
    return {"total": int(total), "frozen": int(frozen), "trainable": int(trainable)}


def validate_feature_attention_mask(
    model: Any,
    raw_attention_mask: torch.Tensor,
    feature_vector_length: int,
) -> torch.Tensor | None:
    """Validate Wav2Vec2 feature-level attention masks before masked pooling.

    Hugging Face Wav2Vec2 sequence classification performs masked mean pooling
    after converting the raw waveform mask to feature-vector length. If that
    feature mask is all zero for a sample, the pooling denominator becomes zero
    and logits/loss can become non-finite. This check fails loudly before that
    point instead of hiding the invalid batch.
    """
    if raw_attention_mask.ndim != 2:
        raise Wav2Vec2IntegrationError(f"Expected 2D attention mask, got shape {tuple(raw_attention_mask.shape)}")
    raw_sums = raw_attention_mask.long().sum(dim=1)
    if torch.any(raw_sums <= 0):
        raise Wav2Vec2IntegrationError("Raw attention mask contains an all-zero sample.")

    if not hasattr(model, "_get_feature_vector_attention_mask"):
        return None

    feature_mask = model._get_feature_vector_attention_mask(feature_vector_length, raw_attention_mask)
    feature_sums = feature_mask.long().sum(dim=1)
    if torch.any(feature_sums <= 0):
        raise Wav2Vec2IntegrationError(
            "Feature-level attention mask contains an all-zero sample; masked pooling denominator would be zero."
        )
    return feature_mask
