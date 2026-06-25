"""Audio preprocessing utilities for uploaded meeting recordings."""

from __future__ import annotations

import shutil
import tempfile
import logging
from pathlib import Path
from typing import BinaryIO

import ffmpeg

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".aac", ".m4a", ".mp3", ".mp4", ".wav"}
)
CONVERT_EXTENSIONS: frozenset[str] = SUPPORTED_EXTENSIONS


class AudioProcessingError(Exception):
    """Raised when audio preprocessing fails."""


def _ffmpeg_command() -> str:
    try:
        import imageio_ffmpeg

        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        logger.info("Using imageio-ffmpeg binary: %s", ffmpeg_path)
        return ffmpeg_path
    except Exception:
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg:
            logger.info("Using system ffmpeg binary: %s", system_ffmpeg)
            return system_ffmpeg

    raise AudioProcessingError(
        "FFmpeg is unavailable. Streamlit Cloud should install it from "
        "requirements.txt and packages.txt. Reboot the app after deployment."
    )


def _validate_extension(filename: str) -> str:
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise AudioProcessingError(
            f"Unsupported audio format '{extension or 'unknown'}'. "
            f"Supported formats: {supported}."
        )
    return extension


def _materialize_source(
    source: str | Path | BinaryIO | bytes,
    *,
    filename: str | None = None,
) -> tuple[Path, bool]:
    """Write uploads to disk when needed. Returns (path, is_temporary)."""
    logger.info(
        "preprocess_uploaded_audio source materialization started: source_type=%s filename=%r",
        type(source).__name__,
        filename,
    )
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise AudioProcessingError(f"Audio file not found: {path}")
        if path.stat().st_size == 0:
            raise AudioProcessingError(f"Audio file is empty: {path}")
        _validate_extension(path.name)
        logger.info(
            "Using existing audio path: path=%s size=%s",
            path,
            path.stat().st_size,
        )
        return path, False

    if isinstance(source, bytes):
        data = source
        name = filename or "upload.bin"
    else:
        name = filename or getattr(source, "name", None) or "upload.bin"
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            data = source.read()
        except OSError as exc:
            raise AudioProcessingError("Failed to read uploaded audio.") from exc

    if not data:
        raise AudioProcessingError("Uploaded audio is empty.")

    extension = _validate_extension(name)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=extension)
    try:
        temp_file.write(data)
        temp_file.close()
        temp_path = Path(temp_file.name)
        logger.info(
            "Uploaded audio stored in temporary file: filename=%r path=%s size=%s",
            name,
            temp_path,
            temp_path.stat().st_size,
        )
        return temp_path, True
    except OSError as exc:
        temp_file.close()
        Path(temp_file.name).unlink(missing_ok=True)
        raise AudioProcessingError("Failed to store uploaded audio.") from exc


def _convert_to_wav(input_path: Path) -> Path:
    ffmpeg_cmd = _ffmpeg_command()

    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    output_file.close()
    output_path = Path(output_file.name)

    try:
        logger.info(
            "Starting ffmpeg WAV conversion: input=%s input_size=%s output=%s",
            input_path,
            input_path.stat().st_size if input_path.exists() else None,
            output_path,
        )
        input_stream = ffmpeg.input(str(input_path))
        (
            ffmpeg.output(
                input_stream.audio,
                str(output_path),
                format="wav",
                acodec="pcm_s16le",
            )
            .overwrite_output()
            .run(
                cmd=ffmpeg_cmd,
                capture_stdout=True,
                capture_stderr=True,
            )
        )
    except ffmpeg.Error as exc:
        output_path.unlink(missing_ok=True)
        stderr_bytes = exc.stderr or b""
        stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
        detail = f" ffmpeg stderr: {stderr}" if stderr else ""
        raise AudioProcessingError(
            f"Could not decode audio file '{input_path.name}'. "
            "The file may be corrupt or use an unsupported codec."
            f"{detail}"
        ) from exc
    except OSError as exc:
        output_path.unlink(missing_ok=True)
        raise AudioProcessingError(
            "Audio conversion failed because the bundled FFmpeg executable could "
            "not be started."
        ) from exc
    except Exception as exc:
        output_path.unlink(missing_ok=True)
        raise AudioProcessingError(
            f"Unexpected error while converting '{input_path.name}' to WAV."
        ) from exc

    if output_path.stat().st_size == 0:
        output_path.unlink(missing_ok=True)
        raise AudioProcessingError(
            f"Conversion produced an empty WAV file for '{input_path.name}'."
        )

    logger.info(
        "Completed ffmpeg WAV conversion: output=%s output_size=%s",
        output_path,
        output_path.stat().st_size,
    )
    return output_path


def preprocess_uploaded_audio(
    source: str | Path | BinaryIO | bytes,
    *,
    filename: str | None = None,
) -> Path:
    """
    Preprocess uploaded or local audio and return the path to a WAV file.

    Accepts a file path, bytes, or file-like upload (e.g. Streamlit UploadedFile).
    MP3, MP4, M4A, AAC, and WAV inputs are converted to WAV via ffmpeg-python
    and the cloud-compatible FFmpeg binary provided by imageio-ffmpeg.

    The caller is responsible for deleting returned temporary files when done.
    """
    logger.info(
        "preprocess_uploaded_audio entered: source_type=%s filename=%r",
        type(source).__name__,
        filename,
    )
    input_path, input_is_temp = _materialize_source(source, filename=filename)

    try:
        extension = input_path.suffix.lower()
        if extension not in CONVERT_EXTENSIONS:
            raise AudioProcessingError(
                f"Unsupported audio format '{extension}'. "
                f"Convertible formats: {', '.join(sorted(CONVERT_EXTENSIONS))}."
            )

        wav_path = _convert_to_wav(input_path)
        logger.info(
            "preprocess_uploaded_audio exiting: wav_path=%s wav_exists=%s wav_size=%s",
            wav_path,
            wav_path.exists(),
            wav_path.stat().st_size if wav_path.exists() else None,
        )
        return wav_path
    finally:
        if input_is_temp:
            logger.info("Deleting temporary source audio: path=%s", input_path)
            input_path.unlink(missing_ok=True)
