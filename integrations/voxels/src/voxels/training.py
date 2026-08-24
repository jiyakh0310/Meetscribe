"""Safe Wav2Vec2 training utilities for Voxels."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

from voxels.experiment import (
    create_experiment_dir,
    device_info,
    plot_history_curves,
    save_history,
    save_resolved_config,
    save_run_summary,
)
from voxels.labels import CANONICAL_LABELS, ID2LABEL, LABEL2ID
from voxels.reproducibility import set_seed
from voxels.wav2vec2 import apply_transformer_mode, id2label, label2id, parameter_counts, validate_feature_attention_mask


class TrainingConfigError(ValueError):
    """Raised when a training configuration is unsafe or invalid."""


@dataclass(frozen=True)
class Wav2Vec2TrainingConfig:
    """Configurable Wav2Vec2 training settings."""

    seed: int = 42
    epochs: int = 1
    batch_size: int = 2
    gradient_accumulation_steps: int = 1
    pretrained_learning_rate: float = 1e-5
    head_learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.0
    max_grad_norm: float = 1.0
    use_class_weights: bool = False
    early_stopping_patience: int = 2
    mixed_precision: str = "auto"
    output_dir: str = "models/wav2vec2"
    transformer_mode: str = "head_only"
    trainable_upper_layers: int = 4
    source_checkpoint: str | None = None
    config_path: str | None = None


@dataclass(frozen=True)
class EpochMetrics:
    """Metrics captured after one training epoch."""

    epoch: int
    training_loss: float
    validation_loss: float
    validation_accuracy: float
    validation_macro_f1: float
    learning_rate: float | None = None


class EarlyStopping:
    """Track validation Macro F1 and stop after patience is exhausted."""

    def __init__(self, patience: int) -> None:
        if patience < 0:
            raise TrainingConfigError("early_stopping_patience must be non-negative.")
        self.patience = patience
        self.best_score = -math.inf
        self.best_epoch: int | None = None
        self.bad_epochs = 0

    def update(self, score: float, epoch: int) -> bool:
        """Return True when training should stop."""
        if score > self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs > self.patience


def validate_training_config(config: Wav2Vec2TrainingConfig) -> None:
    """Fail early on unsafe or invalid training settings."""
    if config.epochs <= 0:
        raise TrainingConfigError("epochs must be positive.")
    if config.batch_size <= 0:
        raise TrainingConfigError("batch_size must be positive.")
    if config.gradient_accumulation_steps <= 0:
        raise TrainingConfigError("gradient_accumulation_steps must be positive.")
    if config.pretrained_learning_rate <= 0 or config.head_learning_rate <= 0:
        raise TrainingConfigError("learning rates must be positive.")
    if config.weight_decay < 0:
        raise TrainingConfigError("weight_decay must be non-negative.")
    if not 0 <= config.warmup_ratio <= 1:
        raise TrainingConfigError("warmup_ratio must be between 0 and 1.")
    if config.max_grad_norm <= 0:
        raise TrainingConfigError("max_grad_norm must be positive.")
    if config.transformer_mode not in {"head_only", "upper_layers", "full_transformer", "full"}:
        raise TrainingConfigError(f"Invalid transformer_mode: {config.transformer_mode}")
    if config.mixed_precision not in {"auto", "on", "off"}:
        raise TrainingConfigError("mixed_precision must be one of: auto, on, off.")


def calculate_class_weights(labels: Iterable[str]) -> torch.Tensor:
    """Calculate inverse-frequency class weights from training labels only."""
    label_list = list(labels)
    if not label_list:
        raise TrainingConfigError("Cannot calculate class weights from an empty training set.")
    unknown = sorted(set(label_list).difference(CANONICAL_LABELS))
    if unknown:
        raise TrainingConfigError(f"Cannot calculate class weights for unknown labels: {unknown}")
    counts = {label: label_list.count(label) for label in CANONICAL_LABELS}
    present_count = sum(1 for count in counts.values() if count > 0)
    if present_count == 0:
        raise TrainingConfigError("Cannot calculate class weights without at least one known label.")
    weights = []
    for label in CANONICAL_LABELS:
        count = counts[label]
        if count == 0:
            weights.append(0.0)
        else:
            weights.append(len(label_list) / (present_count * count))
    tensor = torch.tensor(weights, dtype=torch.float32)
    if not torch.isfinite(tensor).all():
        raise TrainingConfigError("Class-weight calculation produced non-finite weights.")
    return tensor


def build_loss_criterion(
    use_class_weights: bool,
    train_labels: Iterable[str],
    device: torch.device,
) -> torch.nn.CrossEntropyLoss:
    """Build cross-entropy loss with explicit safe class-weight handling."""
    class_weights = calculate_class_weights(train_labels).to(device) if use_class_weights else None
    return torch.nn.CrossEntropyLoss(weight=class_weights)


def class_weight_report(labels: Iterable[str]) -> dict[str, Any]:
    """Return per-class training counts and finite class weights."""
    label_list = list(labels)
    weights = calculate_class_weights(label_list)
    counts = {label: int(label_list.count(label)) for label in CANONICAL_LABELS}
    values = {label: float(weights[LABEL2ID[label]].item()) for label in CANONICAL_LABELS}
    return {
        "training_counts": counts,
        "class_weights": values,
        "all_finite": bool(torch.isfinite(weights).all().item()),
    }


def build_optimizer_parameter_groups(
    model: torch.nn.Module,
    pretrained_learning_rate: float,
    head_learning_rate: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """Create optimizer groups with separate pretrained and head learning rates."""
    pretrained_params: list[torch.nn.Parameter] = []
    head_params: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("wav2vec2."):
            pretrained_params.append(parameter)
        else:
            head_params.append(parameter)
    groups: list[dict[str, Any]] = []
    if pretrained_params:
        groups.append({"params": pretrained_params, "lr": pretrained_learning_rate, "weight_decay": weight_decay, "name": "pretrained"})
    if head_params:
        groups.append({"params": head_params, "lr": head_learning_rate, "weight_decay": weight_decay, "name": "classification_head"})
    if not groups:
        raise TrainingConfigError("No trainable parameters found for optimizer.")
    return groups


def compute_metrics(true_ids: np.ndarray, predicted_ids: np.ndarray) -> dict[str, float]:
    """Calculate validation Accuracy and Macro F1."""
    return {
        "accuracy": float(accuracy_score(true_ids, predicted_ids)),
        "macro_f1": float(f1_score(true_ids, predicted_ids, labels=list(range(len(CANONICAL_LABELS))), average="macro", zero_division=0)),
    }


def should_use_mixed_precision(config_value: str, device: torch.device) -> bool:
    """Resolve mixed precision setting for the current device."""
    if config_value == "off":
        return False
    if config_value == "on":
        return device.type == "cuda"
    return device.type == "cuda"


def normalize_transformer_mode(mode: str) -> str:
    """Map CLI-friendly mode names to internal Wav2Vec2 freeze modes."""
    if mode == "full":
        return "full_transformer"
    if mode in {"head_only", "upper_layers", "full_transformer"}:
        return mode
    raise TrainingConfigError(f"Invalid transformer_mode: {mode}")


def validate_model_attention_mask(model: torch.nn.Module, input_values: torch.Tensor, attention_mask: torch.Tensor) -> None:
    """Validate raw and feature-level attention masks before Wav2Vec2 masked pooling."""
    if hasattr(model, "_get_feat_extract_output_lengths"):
        input_length = torch.tensor(input_values.shape[1], device=input_values.device)
        feature_vector_length = int(model._get_feat_extract_output_lengths(input_length).item())
        validate_feature_attention_mask(model, attention_mask, feature_vector_length)


def _tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach()
    finite = torch.isfinite(detached)
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "minimum": float(detached.min().cpu().item()) if detached.numel() else math.nan,
        "maximum": float(detached.max().cpu().item()) if detached.numel() else math.nan,
        "mean": float(detached.float().mean().cpu().item()) if detached.numel() else math.nan,
        "standard_deviation": float(detached.float().std(unbiased=False).cpu().item()) if detached.numel() else math.nan,
        "nan_count": int(torch.isnan(detached).sum().cpu().item()),
        "infinity_count": int(torch.isinf(detached).sum().cpu().item()),
        "finite": bool(finite.all().cpu().item()),
    }


def _first_invalid_parameter(model: torch.nn.Module) -> str | None:
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter.detach()).all():
            return name
    return None


def _iter_tensors(value: Any) -> Iterable[torch.Tensor]:
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)
    elif hasattr(value, "to_tuple"):
        yield from _iter_tensors(value.to_tuple())


def _first_nonfinite_module(model: torch.nn.Module, forward_call: Any) -> str | None:
    first_invalid: dict[str, str] = {}
    handles = []

    def hook(name: str):
        def inspect_output(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if first_invalid:
                return
            for tensor in _iter_tensors(output):
                if tensor.numel() and not torch.isfinite(tensor.detach()).all():
                    first_invalid["name"] = name
                    return

        return inspect_output

    for name, module in model.named_modules():
        if name:
            handles.append(module.register_forward_hook(hook(name)))
    try:
        forward_call()
    finally:
        for handle in handles:
            handle.remove()
    return first_invalid.get("name")


@torch.no_grad()
def debug_first_batch_forward(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    """Inspect the first training batch and no-grad FP32 forward path."""
    model.train()
    batch = next(iter(loader))
    input_values = batch["input_values"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    if labels.dtype != torch.long:
        raise TrainingConfigError(f"Labels must be torch.long, got {labels.dtype}")
    if int(labels.min().cpu().item()) < 0 or int(labels.max().cpu().item()) >= len(CANONICAL_LABELS):
        raise TrainingConfigError(f"Labels are outside canonical range 0-{len(CANONICAL_LABELS) - 1}: {labels.tolist()}")

    invalid_parameter = _first_invalid_parameter(model)
    all_parameters_finite = invalid_parameter is None

    input_diagnostics = {
        "source_filename": batch.get("filename", [""])[0],
        "source_emotion": batch.get("emotion", [""])[0],
        "encoded_label": int(labels[0].detach().cpu().item()),
        "input_values": _tensor_stats(input_values),
        "valid_sample_length": int(batch["valid_sample_length"][0].detach().cpu().item())
        if torch.is_tensor(batch.get("valid_sample_length"))
        else int(batch.get("valid_sample_length", [0])[0]),
        "attention_mask_shape": list(attention_mask.shape),
        "attention_mask_sum": [int(value) for value in attention_mask.long().sum(dim=1).detach().cpu().tolist()],
        "raw_mask_all_zero": [bool(value <= 0) for value in attention_mask.long().sum(dim=1).detach().cpu().tolist()],
    }

    feature_extractor_output = None
    wav2vec2 = getattr(model, "wav2vec2", None)
    if wav2vec2 is not None and hasattr(wav2vec2, "feature_extractor"):
        feature_extractor_output = wav2vec2.feature_extractor(input_values)

    validate_model_attention_mask(model, input_values, attention_mask)

    def forward_without_labels() -> None:
        model(input_values=input_values, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)

    first_invalid_module = _first_nonfinite_module(model, forward_without_labels)
    outputs = model(input_values=input_values, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
    logits = outputs.logits
    manual_loss = criterion(logits.float(), labels)
    returned = model(
        input_values=input_values,
        attention_mask=attention_mask,
        labels=labels,
        output_hidden_states=True,
        return_dict=True,
    )
    model_loss = returned.loss

    hidden_state = None
    if getattr(outputs, "hidden_states", None):
        hidden_state = outputs.hidden_states[-1]
    elif wav2vec2 is not None:
        wav_outputs = wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_state = wav_outputs.last_hidden_state

    feature_mask = None
    pooled = None
    projected = None
    pooling_denominator = None
    if hidden_state is not None:
        feature_mask = validate_feature_attention_mask(model, attention_mask, hidden_state.shape[1])
        if feature_mask is not None:
            expanded_mask = feature_mask.unsqueeze(-1).to(dtype=hidden_state.dtype, device=hidden_state.device)
            pooling_denominator = expanded_mask.sum(dim=1)
            if torch.any(pooling_denominator <= 0):
                raise TrainingConfigError("Masked pooling denominator is zero.")
            pooled = (hidden_state * expanded_mask).sum(dim=1) / pooling_denominator
        else:
            pooled = hidden_state.mean(dim=1)
            pooling_denominator = torch.tensor([hidden_state.shape[1]], device=hidden_state.device, dtype=hidden_state.dtype)
        projector = getattr(model, "projector", None)
        if projector is not None:
            projected = projector(pooled)

    config = getattr(model, "config", None)
    model_diagnostics = {
        "all_model_parameters_finite": all_parameters_finite,
        "first_invalid_parameter_name": invalid_parameter,
        "first_nonfinite_forward_module": first_invalid_module,
        "apply_spec_augment": getattr(config, "apply_spec_augment", None),
        "mask_time_prob": getattr(config, "mask_time_prob", None),
        "mask_time_length": getattr(config, "mask_time_length", None),
        "mask_feature_prob": getattr(config, "mask_feature_prob", None),
        "feature_extractor_output": _tensor_stats(feature_extractor_output) if feature_extractor_output is not None else None,
        "transformer_hidden_state": _tensor_stats(hidden_state) if hidden_state is not None else None,
        "feature_attention_mask_shape": list(feature_mask.shape) if feature_mask is not None else None,
        "feature_attention_mask_sum": [int(value) for value in feature_mask.long().sum(dim=1).detach().cpu().tolist()]
        if feature_mask is not None
        else None,
        "masked_pooling_denominator": pooling_denominator.detach().cpu().reshape(-1).tolist()
        if pooling_denominator is not None
        else None,
        "pooled_representation": _tensor_stats(pooled) if pooled is not None else None,
        "projector_output": _tensor_stats(projected) if projected is not None else None,
        "classifier_logits": _tensor_stats(logits),
        "classifier_logits_values": logits.detach().cpu().tolist(),
        "manual_cross_entropy_loss": float(manual_loss.detach().cpu().item()),
        "manual_cross_entropy_loss_finite": bool(torch.isfinite(manual_loss).detach().cpu().item()),
        "model_returned_loss": float(model_loss.detach().cpu().item()) if model_loss is not None else None,
        "model_returned_loss_finite": bool(torch.isfinite(model_loss).detach().cpu().item()) if model_loss is not None else None,
    }

    return {
        "input_diagnostics": input_diagnostics,
        "model_diagnostics": model_diagnostics,
    }


def print_first_batch_diagnostics(diagnostics: dict[str, Any]) -> None:
    """Print first-batch diagnostics as readable JSON."""
    print(json.dumps(diagnostics, indent=2, sort_keys=True))


def save_training_checkpoint(
    output_dir: Path,
    model: torch.nn.Module,
    processor: Any | None,
    config: Wav2Vec2TrainingConfig,
    epoch_metrics: EpochMetrics,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
) -> Path:
    """Save model, processor, label mappings, config, and training state."""
    checkpoint_dir = output_dir / f"checkpoint-epoch-{epoch_metrics.epoch}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(checkpoint_dir)
    else:
        torch.save(model.state_dict(), checkpoint_dir / "pytorch_model.bin")
    if processor is not None and hasattr(processor, "save_pretrained"):
        processor.save_pretrained(checkpoint_dir)
    with (checkpoint_dir / "label_mapping.json").open("w", encoding="utf-8") as handle:
        json.dump({"label2id": label2id(), "id2label": id2label()}, handle, indent=2)
        handle.write("\n")
    with (checkpoint_dir / "training_config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)
        handle.write("\n")
    with (checkpoint_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(epoch_metrics), handle, indent=2)
        handle.write("\n")
    state: dict[str, Any] = {"epoch": epoch_metrics.epoch}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    torch.save(state, checkpoint_dir / "trainer_state.pt")
    return checkpoint_dir


def select_best_checkpoint(history: list[EpochMetrics]) -> EpochMetrics:
    """Select the best checkpoint using validation Macro F1."""
    if not history:
        raise TrainingConfigError("Cannot select a checkpoint from empty history.")
    return max(history, key=lambda item: item.validation_macro_f1)


def resume_training_state(checkpoint_dir: Path, optimizer: torch.optim.Optimizer | None = None, scheduler: Any | None = None) -> int:
    """Resume optimizer/scheduler state and return the next epoch number."""
    state_path = checkpoint_dir / "trainer_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"Checkpoint trainer state not found: {state_path}")
    state = torch.load(state_path, map_location="cpu")
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    return int(state.get("epoch", 0)) + 1


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    criterion: torch.nn.Module,
    device: torch.device,
    config: Wav2Vec2TrainingConfig,
    use_amp: bool,
) -> float:
    """Run one training epoch and return average loss."""
    model.train()
    total_loss = 0.0
    steps = 0
    optimizer.zero_grad(set_to_none=True)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    for step, batch in enumerate(loader, start=1):
        input_values = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        validate_model_attention_mask(model, input_values, attention_mask)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(input_values=input_values, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels) / config.gradient_accumulation_steps
        if not torch.isfinite(loss):
            raise FloatingPointError("NaN or infinite training loss detected.")
        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if step % config.gradient_accumulation_steps == 0 or step == len(loader):
            if scaler is not None and use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            if scaler is not None and use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.detach().cpu().item()) * config.gradient_accumulation_steps
        steps += 1
    return total_loss / max(1, steps)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    """Evaluate on validation data only."""
    model.eval()
    losses: list[float] = []
    predictions: list[int] = []
    true_labels: list[int] = []
    for batch in loader:
        input_values = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        validate_model_attention_mask(model, input_values, attention_mask)
        outputs = model(input_values=input_values, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("NaN or infinite validation loss detected.")
        losses.append(float(loss.cpu().item()))
        predictions.extend(outputs.logits.argmax(dim=-1).cpu().numpy().tolist())
        true_labels.extend(labels.cpu().numpy().tolist())
    metrics = compute_metrics(np.array(true_labels), np.array(predictions))
    return float(np.mean(losses)) if losses else math.inf, metrics


def train_wav2vec2_model(
    model: torch.nn.Module,
    processor: Any | None,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: Wav2Vec2TrainingConfig,
    output_dir: Path,
    train_labels: Iterable[str],
    resume_from_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Train Wav2Vec2 safely using train and validation splits only."""
    started_at = time.time()
    validate_training_config(config)
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    apply_transformer_mode(model, normalize_transformer_mode(config.transformer_mode), config.trainable_upper_layers)

    weight_report = class_weight_report(train_labels) if config.use_class_weights else None
    if weight_report is not None:
        print(f"Class-weight training counts: {weight_report['training_counts']}")
        print(f"Class weights: {weight_report['class_weights']}")
        print(f"Class weights finite: {weight_report['all_finite']}")
    criterion = build_loss_criterion(config.use_class_weights, train_labels, device)
    groups = build_optimizer_parameter_groups(
        model,
        config.pretrained_learning_rate,
        config.head_learning_rate,
        config.weight_decay,
    )
    optimizer = torch.optim.AdamW(groups)
    update_steps_per_epoch = max(1, math.ceil(len(train_loader) / config.gradient_accumulation_steps))
    total_steps = max(1, update_steps_per_epoch * config.epochs)
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    start_epoch = 1
    if resume_from_checkpoint is not None:
        start_epoch = 1

    use_amp = should_use_mixed_precision(config.mixed_precision, device)
    early_stopping = EarlyStopping(config.early_stopping_patience)
    history: list[EpochMetrics] = []
    best_checkpoint: Path | None = None
    experiment_dir = create_experiment_dir(output_dir, normalize_transformer_mode(config.transformer_mode), config.seed)
    save_resolved_config(
        experiment_dir,
        config,
        {
            "timestamp": experiment_dir.name.split("_")[0],
            "seed": config.seed,
            "mode": normalize_transformer_mode(config.transformer_mode),
            "parameter_counts": parameter_counts(model),
            "device": device_info(),
            "resume_from_checkpoint": str(resume_from_checkpoint) if resume_from_checkpoint else None,
            "source_checkpoint": config.source_checkpoint,
            "config_path": config.config_path,
            "class_weight_report": weight_report,
        },
    )
    early_stopping_reason = "max_epochs_reached"

    for epoch in range(start_epoch, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, config, use_amp)
        validation_loss, validation_metrics = evaluate_model(model, validation_loader, criterion, device)
        learning_rate = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else None
        epoch_metrics = EpochMetrics(
            epoch=epoch,
            training_loss=train_loss,
            validation_loss=validation_loss,
            validation_accuracy=validation_metrics["accuracy"],
            validation_macro_f1=validation_metrics["macro_f1"],
            learning_rate=learning_rate,
        )
        history.append(epoch_metrics)
        if epoch_metrics == select_best_checkpoint(history):
            best_checkpoint = save_training_checkpoint(experiment_dir, model, processor, config, epoch_metrics, optimizer, scheduler)
        if early_stopping.update(epoch_metrics.validation_macro_f1, epoch):
            early_stopping_reason = f"patience_exhausted_after_epoch_{epoch}"
            break

    best_metrics = select_best_checkpoint(history)
    history_rows = [asdict(item) for item in history]
    save_history(experiment_dir, history_rows)
    plot_history_curves(experiment_dir, history_rows)
    runtime_seconds = time.time() - started_at
    summary = {
        "experiment_dir": str(experiment_dir),
        "history": history_rows,
        "best_epoch": best_metrics.epoch,
        "best_validation_macro_f1": best_metrics.validation_macro_f1,
        "best_checkpoint": str(best_checkpoint) if best_checkpoint else None,
        "early_stopping_reason": early_stopping_reason,
        "parameter_counts": parameter_counts(model),
        "config_path": config.config_path,
        "source_checkpoint": config.source_checkpoint or (str(resume_from_checkpoint) if resume_from_checkpoint else None),
        "class_weight_report": weight_report,
        "device": device_info(),
        "runtime": {
            "seconds": runtime_seconds,
            "epochs_completed": len(history),
            "started_from_epoch": start_epoch,
        },
        "test_set_evaluated": False,
    }
    save_run_summary(experiment_dir, summary)
    return summary


class TensorSpeechDataset(Dataset):
    """Tiny tensor dataset used for deterministic smoke tests and smoke CLI mode."""

    def __init__(self, waveforms: list[torch.Tensor], labels: list[int]) -> None:
        self.waveforms = waveforms
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        waveform = self.waveforms[index].float()
        return {
            "input_values": waveform.numpy(),
            "attention_mask": np.ones(waveform.numel(), dtype=np.int64),
            "labels": self.labels[index],
            "valid_sample_length": int(waveform.numel()),
        }
