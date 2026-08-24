"""Raw audio loading and preprocessing for Wav2Vec2 input."""

from __future__ import annotations

import math
import random
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac"}


class AudioPreprocessingError(ValueError):
    """Fatal error that prevents audio from being used for inference/training."""


@dataclass(frozen=True)
class WaveformAugmentationConfig:
    """Mild training-only waveform augmentation settings."""

    enabled: bool = False
    gain_db_min: float = -1.5
    gain_db_max: float = 1.5
    time_shift_ms: float = 30.0
    noise_rms: float = 0.001
    speed_min: float = 0.98
    speed_max: float = 1.02


@dataclass(frozen=True)
class AudioPreprocessorConfig:
    """Configurable preprocessing settings."""

    target_sample_rate: int = 16000
    min_duration_seconds: float = 0.5
    max_duration_seconds: float = 10.0
    near_silence_rms: float = 1e-4
    crop_seconds: float | None = None
    pad_seconds: float | None = None
    augmentation: WaveformAugmentationConfig = field(default_factory=WaveformAugmentationConfig)


@dataclass(frozen=True)
class OriginalAudioMetadata:
    """Useful source-audio metadata preserved during preprocessing."""

    path: str
    extension: str
    sample_rate: int
    channels: int
    num_frames: int
    duration_seconds: float
    dtype: str


@dataclass(frozen=True)
class AudioPreprocessingResult:
    """Structured output for model-ready raw waveform input."""

    waveform: np.ndarray
    processed_sample_rate: int
    original_sample_rate: int
    original_channels: int
    duration: float
    original_duration: float
    quality_warnings: list[str]
    valid_sample_length: int
    original_metadata: OriginalAudioMetadata
    fatal_errors: list[str] = field(default_factory=list)


def _pcm_to_float(samples: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        values = np.frombuffer(samples, dtype=np.uint8).astype(np.float32)
        return (values - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(samples, dtype="<i2").astype(np.float32) / 32768.0
    if sample_width == 4:
        return (np.frombuffer(samples, dtype="<i4").astype(np.float64) / 2147483648.0).astype(np.float32)
    raise AudioPreprocessingError(f"Unsupported WAV PCM sample width: {sample_width} bytes")


def _load_with_wave(path: Path) -> tuple[np.ndarray, int, int, str]:
    try:
        with wave.open(str(path), "rb") as handle:
            sample_rate = int(handle.getframerate())
            channels = int(handle.getnchannels())
            frames = int(handle.getnframes())
            sample_width = int(handle.getsampwidth())
            raw_samples = handle.readframes(frames)
    except (EOFError, OSError, wave.Error) as exc:
        raise AudioPreprocessingError(f"Could not read audio file: {exc}") from exc

    samples = _pcm_to_float(raw_samples, sample_width)
    if channels > 1:
        samples = samples.reshape(-1, channels)
    return samples.astype(np.float32, copy=False), sample_rate, channels, f"pcm{sample_width * 8}"


def load_audio(path: Path) -> tuple[np.ndarray, int, int, str]:
    """Load WAV/MP3/FLAC audio using installed backends where supported."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise AudioPreprocessingError(f"Unsupported audio extension '{suffix}'. Supported: {supported}")

    try:
        import soundfile as sf
    except ImportError:
        if suffix != ".wav":
            raise AudioPreprocessingError(
                f"Backend for {suffix} is not installed. Install soundfile or another supported audio backend."
            )
        return _load_with_wave(path)

    try:
        data, sample_rate = sf.read(str(path), always_2d=False, dtype="float32")
        info = sf.info(str(path))
    except Exception as exc:
        if suffix == ".wav":
            return _load_with_wave(path)
        raise AudioPreprocessingError(f"Could not read audio file with installed backend: {exc}") from exc

    array = np.asarray(data, dtype=np.float32)
    channels = int(info.channels)
    return array, int(sample_rate), channels, str(info.subtype)


def to_mono(waveform: np.ndarray) -> np.ndarray:
    """Convert loaded audio to one-dimensional mono by averaging channels."""
    if waveform.ndim == 1:
        return waveform.astype(np.float32, copy=False)
    if waveform.ndim == 2:
        return waveform.mean(axis=1, dtype=np.float32)
    raise AudioPreprocessingError(f"Expected 1D or 2D audio, got shape {waveform.shape}")


def resample_waveform(waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample a mono waveform with deterministic linear interpolation."""
    if source_rate <= 0 or target_rate <= 0:
        raise AudioPreprocessingError("Sample rates must be positive.")
    if source_rate == target_rate:
        return waveform.astype(np.float32, copy=False)
    if waveform.size == 0:
        return waveform.astype(np.float32, copy=False)

    target_length = max(1, int(round(waveform.size * target_rate / source_rate)))
    old_positions = np.linspace(0.0, waveform.size - 1, num=waveform.size, dtype=np.float64)
    new_positions = np.linspace(0.0, waveform.size - 1, num=target_length, dtype=np.float64)
    return np.interp(new_positions, old_positions, waveform).astype(np.float32)


def rms_amplitude(waveform: np.ndarray, valid_sample_length: int | None = None) -> float:
    """Calculate RMS over genuine waveform samples, excluding padding."""
    length = waveform.size if valid_sample_length is None else min(valid_sample_length, waveform.size)
    if length <= 0:
        return 0.0
    valid = waveform[:length].astype(np.float64, copy=False)
    return float(math.sqrt(float(np.mean(np.square(valid)))))


class AudioPreprocessor:
    """Prepare raw one-dimensional waveforms for Wav2Vec2 processors."""

    def __init__(self, config: AudioPreprocessorConfig) -> None:
        self.config = config

    def preprocess_file(
        self,
        path: Path | str,
        mode: Literal["training", "validation", "inference"] = "inference",
        random_crop: bool = False,
        rng: random.Random | None = None,
    ) -> AudioPreprocessingResult:
        """Load and preprocess an audio file without creating spectrograms."""
        audio_path = Path(path)
        if not audio_path.exists():
            raise AudioPreprocessingError(f"Audio file does not exist: {audio_path}")
        if not audio_path.is_file():
            raise AudioPreprocessingError(f"Audio path is not a file: {audio_path}")

        loaded, original_sample_rate, original_channels, dtype = load_audio(audio_path)
        original_frames = int(loaded.shape[0]) if loaded.ndim > 0 else 0
        original_duration = original_frames / original_sample_rate if original_sample_rate else 0.0
        metadata = OriginalAudioMetadata(
            path=str(audio_path),
            extension=audio_path.suffix.lower(),
            sample_rate=original_sample_rate,
            channels=original_channels,
            num_frames=original_frames,
            duration_seconds=original_duration,
            dtype=dtype,
        )

        mono = to_mono(loaded)
        if mono.size == 0:
            raise AudioPreprocessingError("Audio file contains no samples.")

        processed = resample_waveform(mono, original_sample_rate, self.config.target_sample_rate)
        if processed.size == 0:
            raise AudioPreprocessingError("Audio file contains no samples after preprocessing.")

        quality_warnings = self._quality_warnings(processed)
        if mode == "training" and self.config.augmentation.enabled:
            processed = self._augment_training_waveform(processed, rng or random)
        processed, valid_sample_length = self._crop_and_pad(processed, mode, random_crop, rng)

        return AudioPreprocessingResult(
            waveform=processed.astype(np.float32, copy=False),
            processed_sample_rate=self.config.target_sample_rate,
            original_sample_rate=original_sample_rate,
            original_channels=original_channels,
            duration=valid_sample_length / self.config.target_sample_rate,
            original_duration=original_duration,
            quality_warnings=quality_warnings,
            valid_sample_length=valid_sample_length,
            original_metadata=metadata,
        )

    def _quality_warnings(self, waveform: np.ndarray) -> list[str]:
        duration = waveform.size / self.config.target_sample_rate
        warnings: list[str] = []
        if rms_amplitude(waveform) <= self.config.near_silence_rms:
            warnings.append("Audio is near silent.")
        if duration < self.config.min_duration_seconds:
            warnings.append(
                f"Audio is shorter than the configured minimum duration ({self.config.min_duration_seconds:.2f}s)."
            )
        if duration > self.config.max_duration_seconds:
            warnings.append(
                f"Audio is longer than the configured maximum duration ({self.config.max_duration_seconds:.2f}s)."
            )
        return warnings

    def _crop_and_pad(
        self,
        waveform: np.ndarray,
        mode: Literal["training", "validation", "inference"],
        random_crop: bool,
        rng: random.Random | None,
    ) -> tuple[np.ndarray, int]:
        valid_length = int(waveform.size)
        target_crop = self._seconds_to_samples(self.config.crop_seconds)
        if target_crop is not None and waveform.size > target_crop:
            if random_crop:
                if mode != "training":
                    raise AudioPreprocessingError("Random cropping is only allowed in training mode.")
                generator = rng or random
                start = generator.randint(0, waveform.size - target_crop)
            else:
                start = (waveform.size - target_crop) // 2
            end = start + target_crop
            waveform = waveform[start:end]
            valid_length = int(waveform.size)

        target_pad = self._seconds_to_samples(self.config.pad_seconds)
        if target_pad is not None and waveform.size < target_pad:
            padded = np.zeros(target_pad, dtype=np.float32)
            padded[: waveform.size] = waveform
            return padded, valid_length

        return waveform.astype(np.float32, copy=False), valid_length

    def _augment_training_waveform(self, waveform: np.ndarray, rng: random.Random) -> np.ndarray:
        """Apply mild deterministic training-only waveform augmentation."""
        config = self.config.augmentation
        augmented = waveform.astype(np.float32, copy=True)

        if config.speed_min <= 0 or config.speed_max <= 0 or config.speed_min > config.speed_max:
            raise AudioPreprocessingError("Invalid speed perturbation range.")
        speed = rng.uniform(config.speed_min, config.speed_max)
        if abs(speed - 1.0) > 1e-6 and augmented.size > 1:
            new_length = max(1, int(round(augmented.size / speed)))
            old_positions = np.linspace(0.0, augmented.size - 1, num=augmented.size, dtype=np.float64)
            new_positions = np.linspace(0.0, augmented.size - 1, num=new_length, dtype=np.float64)
            augmented = np.interp(new_positions, old_positions, augmented).astype(np.float32)

        gain_db = rng.uniform(config.gain_db_min, config.gain_db_max)
        augmented *= float(10 ** (gain_db / 20.0))

        max_shift = int(round(config.time_shift_ms * self.config.target_sample_rate / 1000.0))
        if max_shift > 0 and augmented.size > 1:
            shift = rng.randint(-max_shift, max_shift)
            if shift > 0:
                augmented = np.concatenate([np.zeros(shift, dtype=np.float32), augmented[:-shift]])
            elif shift < 0:
                augmented = np.concatenate([augmented[-shift:], np.zeros(-shift, dtype=np.float32)])

        if config.noise_rms > 0:
            noise = np.array([rng.gauss(0.0, config.noise_rms) for _ in range(augmented.size)], dtype=np.float32)
            augmented = augmented + noise

        return np.clip(augmented, -1.0, 1.0).astype(np.float32, copy=False)

    def _seconds_to_samples(self, seconds: float | None) -> int | None:
        if seconds is None:
            return None
        if seconds <= 0:
            raise AudioPreprocessingError("Crop and padding durations must be positive.")
        return int(round(seconds * self.config.target_sample_rate))
