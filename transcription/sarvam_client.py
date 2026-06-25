"""Sarvam AI speech-to-text client for MeetScribe."""

from __future__ import annotations

import json
import logging
import tempfile
import traceback
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sarvamai import SarvamAI
from sarvamai.core.api_error import ApiError

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

REALTIME_MAX_DURATION_SECONDS = 30.0
BATCH_POLL_TIMEOUT_SECONDS = 3600
BATCH_UPLOAD_TIMEOUT_SECONDS = 300.0
DEFAULT_MODEL = "saaras:v3"
DEFAULT_MODE = "translit"
DEFAULT_LANGUAGE = "unknown"


class TranscriptionError(Exception):
    """Raised when Sarvam speech-to-text processing fails."""


def _mask_key(value: str) -> str:
    return f"{value[:6]}..." if value else "missing"


def _json_safe(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except TypeError:
        return str(value)


def _api_error_details(exc: ApiError) -> dict[str, Any]:
    response = getattr(exc, "response", None)
    response_body = getattr(exc, "body", None)

    if response_body is None and response is not None:
        try:
            response_body = response.json()
        except Exception:
            response_body = getattr(response, "text", None)

    return {
        "status_code": getattr(exc, "status_code", None),
        "message": str(exc),
        "body": response_body,
        "headers": getattr(exc, "headers", None),
        "response_text": getattr(response, "text", None)
        if response is not None
        else None,
    }


def _format_api_error(prefix: str, exc: ApiError) -> str:
    details = _api_error_details(exc)
    return (
        f"{prefix}\n"
        f"HTTP status: {details.get('status_code')}\n"
        f"Error message: {details.get('message')}\n"
        f"Response body: {_json_safe(details.get('body'))}\n"
        f"Sarvam error payload: {_json_safe(details)}"
    )


def _log_api_error(context: str, audio_name: str, exc: ApiError) -> None:
    details = _api_error_details(exc)
    logger.error(
        "%s for '%s': status_code=%s error_message=%s response_body=%s "
        "sarvam_error_payload=%s",
        context,
        audio_name,
        details.get("status_code"),
        details.get("message"),
        _json_safe(details.get("body")),
        _json_safe(details),
    )


@dataclass(frozen=True, slots=True)
class TranscriptionSegment:
    """One speaker-attributed transcript segment."""

    transcript: str
    speaker_id: str | None = None
    start_time_seconds: float | None = None
    end_time_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """Detailed transcription output returned by Sarvam."""

    transcript: str
    language_code: str | None = None
    segments: list[TranscriptionSegment] = field(default_factory=list)
    timestamps: dict[str, Any] = field(default_factory=dict)
    raw_payload: dict[str, Any] = field(default_factory=dict)


def _validate_audio_path(audio_path: Path) -> None:
    if not audio_path.is_file():
        raise TranscriptionError(f"Audio file not found: {audio_path}")
    if audio_path.stat().st_size == 0:
        raise TranscriptionError(f"Audio file is empty: {audio_path}")


def _get_audio_duration_seconds(audio_path: Path) -> float:
    try:
        with wave.open(str(audio_path), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            if frame_rate <= 0:
                raise TranscriptionError(
                    f"Invalid WAV frame rate for '{audio_path.name}'."
                )
            return wav_file.getnframes() / float(frame_rate)
    except wave.Error as exc:
        raise TranscriptionError(
            f"Could not read audio duration for '{audio_path.name}'."
        ) from exc


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_response_value(response: Any, key: str, default: Any = None) -> Any:
    if isinstance(response, dict):
        return response.get(key, default)
    return getattr(response, key, default)


def _parse_diarized_segments(payload: dict[str, Any]) -> list[TranscriptionSegment]:
    diarized_transcript = payload.get("diarized_transcript")
    if not isinstance(diarized_transcript, dict):
        return []

    entries = diarized_transcript.get("entries")
    if not isinstance(entries, list):
        return []

    segments: list[TranscriptionSegment] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue

        transcript = str(entry.get("transcript", "")).strip()
        if not transcript:
            continue

        speaker_id = entry.get("speaker_id")
        segments.append(
            TranscriptionSegment(
                transcript=transcript,
                speaker_id=str(speaker_id) if speaker_id is not None else None,
                start_time_seconds=_coerce_float(entry.get("start_time_seconds")),
                end_time_seconds=_coerce_float(entry.get("end_time_seconds")),
            )
        )

    return segments


def _result_from_payload(payload: dict[str, Any]) -> TranscriptionResult:
    transcript = str(payload.get("transcript", "")).strip()
    segments = _parse_diarized_segments(payload)

    if not transcript and segments:
        transcript = " ".join(segment.transcript for segment in segments).strip()

    if not transcript:
        raise TranscriptionError("Sarvam returned an empty transcript.")

    timestamps = payload.get("timestamps")
    return TranscriptionResult(
        transcript=transcript,
        language_code=(
            str(payload["language_code"]) if payload.get("language_code") else None
        ),
        segments=segments,
        timestamps=timestamps if isinstance(timestamps, dict) else {},
        raw_payload=payload,
    )


def _extract_result_from_batch_output(output_dir: Path) -> TranscriptionResult:
    json_files = sorted(output_dir.glob("*.json"))
    if not json_files:
        raise TranscriptionError(
            "Batch transcription completed but no transcript file was returned."
        )

    for json_file in json_files:
        try:
            payload = json.loads(json_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TranscriptionError(
                f"Failed to parse batch transcript output '{json_file.name}'."
            ) from exc

        try:
            result = _result_from_payload(payload)
        except TranscriptionError:
            continue

        if result.transcript:
            return result

    raise TranscriptionError("Sarvam returned an empty transcript.")


def _extract_transcript_from_batch_output(output_dir: Path) -> str:
    return _extract_result_from_batch_output(output_dir).transcript


class SarvamTranscriptionClient:
    """Upload audio to Sarvam and return transcript text."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        logger.info(
            "Initializing SarvamAI client with SARVAM_API_KEY=%s",
            _mask_key(self._settings.sarvam_api_key),
        )
        self._client = SarvamAI(
            api_subscription_key=self._settings.sarvam_api_key,
        )
        logger.debug("Initialized Sarvam transcription client")

    def transcribe(self, audio_path: str | Path) -> str:
        return self.transcribe_detailed(audio_path).transcript

    def transcribe_detailed(
        self,
        audio_path: str | Path,
        *,
        with_diarization: bool = False,
        num_speakers: int | None = None,
    ) -> TranscriptionResult:
        path = Path(audio_path)
        _validate_audio_path(path)

        duration_seconds = _get_audio_duration_seconds(path)
        logger.info(
            "Starting transcription for '%s' (%.1f seconds)",
            path.name,
            duration_seconds,
        )

        if duration_seconds < REALTIME_MAX_DURATION_SECONDS and not with_diarization:
            return self._transcribe_realtime(path)

        return self._transcribe_batch(
            path,
            with_diarization=with_diarization,
            num_speakers=num_speakers,
        )

    def _transcribe_realtime(self, audio_path: Path) -> TranscriptionResult:
        logger.info("Using Sarvam real-time API for '%s'", audio_path.name)
        try:
            with audio_path.open("rb") as audio_file:
                response = self._client.speech_to_text.transcribe(
                    file=audio_file,
                    model=DEFAULT_MODEL,
                    mode=DEFAULT_MODE,
                    language_code=DEFAULT_LANGUAGE,
                )
        except ApiError as exc:
            _log_api_error("Sarvam real-time API error", audio_path.name, exc)
            raise TranscriptionError(
                _format_api_error("Real-time transcription failed.", exc)
            ) from exc
        except OSError as exc:
            logger.error("Failed to read audio file '%s'", audio_path.name)
            raise TranscriptionError(
                f"Could not read audio file '{audio_path.name}'."
            ) from exc

        transcript = response.transcript.strip()
        if not transcript:
            raise TranscriptionError("Sarvam returned an empty transcript.")

        language_code = _get_response_value(response, "language_code")
        raw_payload = (
            response.model_dump()
            if hasattr(response, "model_dump")
            else vars(response)
            if hasattr(response, "__dict__")
            else {}
        )

        logger.info(
            "Real-time transcription completed for '%s' (%d characters)",
            audio_path.name,
            len(transcript),
        )
        return TranscriptionResult(
            transcript=transcript,
            language_code=str(language_code) if language_code else None,
            raw_payload=raw_payload,
        )

    def _transcribe_batch(
        self,
        audio_path: Path,
        *,
        with_diarization: bool = False,
        num_speakers: int | None = None,
    ) -> TranscriptionResult:
        logger.info("Using Sarvam batch API for '%s'", audio_path.name)
        try:
            create_job_kwargs: dict[str, Any] = {
                "model": DEFAULT_MODEL,
                "mode": DEFAULT_MODE,
                "language_code": DEFAULT_LANGUAGE,
                "with_diarization": with_diarization,
            }
            if num_speakers is not None:
                create_job_kwargs["num_speakers"] = num_speakers

            logger.info(
                "Sarvam batch request for '%s': model=%s mode=%s "
                "language_code=%s with_diarization=%s num_speakers=%s",
                audio_path.name,
                create_job_kwargs.get("model"),
                create_job_kwargs.get("mode"),
                create_job_kwargs.get("language_code"),
                create_job_kwargs.get("with_diarization"),
                create_job_kwargs.get("num_speakers"),
            )

            logger.info("Sarvam batch create_job starting for '%s'", audio_path.name)
            job = self._client.speech_to_text_job.create_job(
                **create_job_kwargs,
            )
            logger.info("Sarvam batch upload starting for '%s'", audio_path.name)
            job.upload_files(
                file_paths=[str(audio_path)],
                timeout=BATCH_UPLOAD_TIMEOUT_SECONDS,
            )
            logger.info("Sarvam batch job start requested for '%s'", audio_path.name)
            job.start()
            logger.info("Sarvam batch wait started for '%s'", audio_path.name)
            status = job.wait_until_complete(timeout=BATCH_POLL_TIMEOUT_SECONDS)
            logger.info(
                "Sarvam batch response received for '%s': job_state=%s",
                audio_path.name,
                getattr(status, "job_state", None),
            )
        except TimeoutError as exc:
            logger.error(
                "Batch transcription timed out for '%s': traceback=%s",
                audio_path.name,
                traceback.format_exc(),
            )
            raise TranscriptionError(
                "Batch transcription timed out. Try a shorter recording."
            ) from exc
        except ApiError as exc:
            _log_api_error("Sarvam batch API error", audio_path.name, exc)
            raise TranscriptionError(
                _format_api_error("Batch transcription failed.", exc)
            ) from exc
        except RuntimeError as exc:
            logger.error(
                "Batch transcription runtime error for '%s': error=%s traceback=%s",
                audio_path.name,
                exc,
                traceback.format_exc(),
            )
            raise TranscriptionError(
                "Batch transcription failed during upload or download."
            ) from exc

        if status.job_state.lower() == "failed" or job.is_failed():
            results = job.get_file_results()
            failed_files = results.get("failed", [])
            error_message = (
                failed_files[0].get("error_message")
                if failed_files
                else "Unknown batch job failure."
            )
            logger.error(
                "Batch job failed for '%s': %s",
                audio_path.name,
                error_message,
            )
            raise TranscriptionError(
                f"Batch transcription failed: {error_message}"
            )

        with tempfile.TemporaryDirectory(prefix="meetscribe_sarvam_") as temp_dir:
            output_dir = Path(temp_dir)
            logger.info("Sarvam batch output download starting for '%s'", audio_path.name)
            job.download_outputs(output_dir=str(output_dir))
            logger.info("Sarvam batch output downloaded for '%s'", audio_path.name)
            result = _extract_result_from_batch_output(output_dir)

        logger.info(
            "Batch transcription completed for '%s' (%d characters)",
            audio_path.name,
            len(result.transcript),
        )
        return result


def transcribe_audio(
    audio_path: str | Path,
    *,
    settings: Settings | None = None,
) -> str:
    """Transcribe a local audio file and return transcript text."""
    client = SarvamTranscriptionClient(settings=settings)
    return client.transcribe(audio_path)


def transcribe_audio_detailed(
    audio_path: str | Path,
    *,
    settings: Settings | None = None,
    with_diarization: bool = False,
    num_speakers: int | None = None,
) -> TranscriptionResult:
    """Transcribe a local audio file and return detailed transcript metadata."""
    client = SarvamTranscriptionClient(settings=settings)
    return client.transcribe_detailed(
        audio_path,
        with_diarization=with_diarization,
        num_speakers=num_speakers,
    )
