"""Classical MFCC + SVM baseline for Voxels."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.fft import dct
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from voxels.audio import AudioPreprocessor, AudioPreprocessorConfig
from voxels.labels import CANONICAL_LABELS


@dataclass(frozen=True)
class BaselineFeatureConfig:
    """Compact MFCC feature settings for the classical baseline."""

    sample_rate: int = 16000
    n_mfcc: int = 13
    n_mels: int = 26
    frame_length_ms: float = 25.0
    frame_step_ms: float = 10.0
    pre_emphasis: float = 0.97
    include_acoustic_stats: bool = True


@dataclass(frozen=True)
class BaselineTrainingConfig:
    """Reproducible SVM baseline training settings."""

    seed: int = 42
    c_values: tuple[float, ...] = (0.1, 1.0, 10.0)
    gamma_values: tuple[str, ...] = ("scale",)
    kernel: str = "rbf"


@dataclass(frozen=True)
class BaselineArtifacts:
    """Paths written by a baseline training/evaluation run."""

    model: Path
    scaler: Path
    feature_config: Path
    label_mapping: Path
    predictions: Path
    metrics: Path
    confusion_matrix: Path


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    """Convert frequency in Hz to mel scale."""
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    """Convert mel scale to frequency in Hz."""
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def frame_waveform(waveform: np.ndarray, sample_rate: int, config: BaselineFeatureConfig) -> np.ndarray:
    """Frame a mono waveform for short-time spectral analysis."""
    frame_length = max(1, int(round(config.frame_length_ms * sample_rate / 1000.0)))
    frame_step = max(1, int(round(config.frame_step_ms * sample_rate / 1000.0)))
    if waveform.size < frame_length:
        padded = np.zeros(frame_length, dtype=np.float32)
        padded[: waveform.size] = waveform
        waveform = padded

    frame_count = 1 + int(np.ceil((waveform.size - frame_length) / frame_step))
    padded_length = (frame_count - 1) * frame_step + frame_length
    padded = np.zeros(padded_length, dtype=np.float32)
    padded[: waveform.size] = waveform
    indices = np.arange(frame_length)[None, :] + frame_step * np.arange(frame_count)[:, None]
    return padded[indices] * np.hamming(frame_length).astype(np.float32)


def mel_filterbank(sample_rate: int, fft_size: int, n_mels: int) -> np.ndarray:
    """Create triangular mel filterbank weights."""
    low_mel = hz_to_mel(0.0)
    high_mel = hz_to_mel(sample_rate / 2.0)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((fft_size + 1) * hz_points / sample_rate).astype(int)
    bins = np.clip(bins, 0, fft_size // 2)

    filters = np.zeros((n_mels, fft_size // 2 + 1), dtype=np.float32)
    for mel_index in range(1, n_mels + 1):
        left, center, right = bins[mel_index - 1], bins[mel_index], bins[mel_index + 1]
        if center == left:
            center += 1
        if right == center:
            right += 1
        for bin_index in range(left, min(center, filters.shape[1])):
            filters[mel_index - 1, bin_index] = (bin_index - left) / (center - left)
        for bin_index in range(center, min(right, filters.shape[1])):
            filters[mel_index - 1, bin_index] = (right - bin_index) / (right - center)
    return filters


def extract_mfcc_matrix(
    waveform: np.ndarray,
    sample_rate: int,
    config: BaselineFeatureConfig,
) -> np.ndarray:
    """Extract MFCC coefficients over time from a preprocessed waveform."""
    if waveform.ndim != 1:
        raise ValueError("MFCC extraction expects a one-dimensional waveform.")
    if waveform.size == 0:
        raise ValueError("Cannot extract MFCC features from an empty waveform.")

    emphasized = np.append(waveform[0], waveform[1:] - config.pre_emphasis * waveform[:-1]).astype(np.float32)
    frames = frame_waveform(emphasized, sample_rate, config)
    fft_size = 1
    while fft_size < frames.shape[1]:
        fft_size *= 2
    power = (np.abs(np.fft.rfft(frames, n=fft_size)) ** 2) / fft_size
    filters = mel_filterbank(sample_rate, fft_size, config.n_mels)
    mel_energy = np.maximum(np.dot(power, filters.T), np.finfo(np.float32).eps)
    log_mel_energy = np.log(mel_energy)
    mfcc = dct(log_mel_energy, type=2, axis=1, norm="ortho")[:, : config.n_mfcc]
    return mfcc.astype(np.float32)


def summarize_mfcc(mfcc: np.ndarray) -> np.ndarray:
    """Summarize MFCC coefficients across time with compact statistics."""
    return np.concatenate(
        [
            mfcc.mean(axis=0),
            mfcc.std(axis=0),
            mfcc.min(axis=0),
            mfcc.max(axis=0),
        ]
    ).astype(np.float32)


def acoustic_statistics(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """Return a small transparent set of acoustic statistics."""
    duration = waveform.size / sample_rate
    rms = float(np.sqrt(np.mean(np.square(waveform)))) if waveform.size else 0.0
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    zero_crossings = float(np.mean(waveform[:-1] * waveform[1:] < 0)) if waveform.size > 1 else 0.0
    mean = float(np.mean(waveform)) if waveform.size else 0.0
    std = float(np.std(waveform)) if waveform.size else 0.0
    return np.array([duration, rms, peak, zero_crossings, mean, std], dtype=np.float32)


def extract_feature_vector(
    waveform: np.ndarray,
    sample_rate: int,
    config: BaselineFeatureConfig,
) -> np.ndarray:
    """Extract the compact baseline feature vector."""
    mfcc_summary = summarize_mfcc(extract_mfcc_matrix(waveform, sample_rate, config))
    if not config.include_acoustic_stats:
        features = mfcc_summary
    else:
        features = np.concatenate([mfcc_summary, acoustic_statistics(waveform, sample_rate)])
    if not np.isfinite(features).all():
        raise ValueError("Extracted baseline features contain NaN or infinite values.")
    return features.astype(np.float32)


def features_from_metadata(
    metadata: pd.DataFrame,
    preprocessor: AudioPreprocessor,
    feature_config: BaselineFeatureConfig,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Preprocess files and extract feature rows from metadata."""
    features: list[np.ndarray] = []
    labels: list[str] = []
    records: list[dict[str, Any]] = []
    for _, row in metadata.iterrows():
        result = preprocessor.preprocess_file(Path(str(row["file_path"])), mode=mode)  # type: ignore[arg-type]
        feature_vector = extract_feature_vector(result.waveform[: result.valid_sample_length], result.processed_sample_rate, feature_config)
        features.append(feature_vector)
        label = str(row["emotion"])
        labels.append(label)
        records.append(
            {
                "file_path": row["file_path"],
                "filename": row.get("filename", Path(str(row["file_path"])).name),
                "emotion": label,
                "quality_warnings": "; ".join(result.quality_warnings),
            }
        )
    if not features:
        raise ValueError("No feature rows were extracted.")
    return np.vstack(features).astype(np.float32), np.array(labels), records


def fit_scaler_on_training_features(train_features: np.ndarray) -> StandardScaler:
    """Fit feature scaler on training data only."""
    scaler = StandardScaler()
    scaler.fit(train_features)
    return scaler


def choose_svm_hyperparameters(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    config: BaselineTrainingConfig,
) -> dict[str, Any]:
    """Tune a small SVM grid using validation data, never the test set."""
    best_params: dict[str, Any] | None = None
    best_score = -1.0
    for c_value in config.c_values:
        for gamma in config.gamma_values:
            model = SVC(C=c_value, gamma=gamma, kernel=config.kernel, class_weight="balanced", random_state=config.seed)
            model.fit(train_features, train_labels)
            predictions = model.predict(validation_features)
            score = f1_score(validation_labels, predictions, labels=list(CANONICAL_LABELS), average="macro", zero_division=0)
            params = {"C": c_value, "gamma": gamma, "kernel": config.kernel}
            if score > best_score:
                best_score = float(score)
                best_params = params
    if best_params is None:
        raise ValueError("Could not tune SVM hyperparameters.")
    return best_params


def train_svm(train_features: np.ndarray, train_labels: np.ndarray, params: dict[str, Any], seed: int) -> SVC:
    """Train the final SVM on actor-independent training data only."""
    model = SVC(
        C=float(params["C"]),
        gamma=params["gamma"],
        kernel=str(params["kernel"]),
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(train_features, train_labels)
    return model


def evaluate_predictions(true_labels: np.ndarray, predicted_labels: np.ndarray) -> dict[str, Any]:
    """Calculate baseline metrics using the shared canonical label order."""
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=list(CANONICAL_LABELS),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(true_labels, predicted_labels, labels=list(CANONICAL_LABELS))
    return {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "macro_f1": float(f1_score(true_labels, predicted_labels, labels=list(CANONICAL_LABELS), average="macro", zero_division=0)),
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in CANONICAL_LABELS
        },
        "confusion_matrix": matrix.tolist(),
        "labels": list(CANONICAL_LABELS),
    }


def save_confusion_matrix_image(matrix: list[list[int]], output_path: Path) -> None:
    """Save a confusion matrix image for the baseline report."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(np.array(matrix), cmap="Blues")
    ax.set_title("Classical Baseline Confusion Matrix")
    ax.set_xlabel("Predicted emotion")
    ax.set_ylabel("True emotion")
    ax.set_xticks(range(len(CANONICAL_LABELS)), labels=CANONICAL_LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(CANONICAL_LABELS)), labels=CANONICAL_LABELS)
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            ax.text(column_index, row_index, str(value), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_baseline_artifacts(
    output_dir: Path | str,
    model: SVC,
    scaler: StandardScaler,
    feature_config: BaselineFeatureConfig,
    label_mapping: dict[str, int],
    predictions: pd.DataFrame,
    metrics: dict[str, Any],
) -> BaselineArtifacts:
    """Persist all classical baseline artifacts."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = BaselineArtifacts(
        model=out_dir / "baseline_svm_model.joblib",
        scaler=out_dir / "baseline_scaler.joblib",
        feature_config=out_dir / "baseline_feature_config.json",
        label_mapping=out_dir / "baseline_label_mapping.json",
        predictions=out_dir / "baseline_predictions.csv",
        metrics=out_dir / "baseline_metrics.json",
        confusion_matrix=out_dir / "baseline_confusion_matrix.png",
    )
    joblib.dump(model, artifacts.model)
    joblib.dump(scaler, artifacts.scaler)
    with artifacts.feature_config.open("w", encoding="utf-8") as handle:
        json.dump(asdict(feature_config), handle, indent=2)
        handle.write("\n")
    with artifacts.label_mapping.open("w", encoding="utf-8") as handle:
        json.dump(label_mapping, handle, indent=2)
        handle.write("\n")
    predictions.to_csv(artifacts.predictions, index=False)
    with artifacts.metrics.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    save_confusion_matrix_image(metrics["confusion_matrix"], artifacts.confusion_matrix)
    return artifacts


def label_mapping() -> dict[str, int]:
    """Return canonical label-to-index mapping."""
    return {label: index for index, label in enumerate(CANONICAL_LABELS)}
