"""Focused, fully mocked checks for Sarvam credit exhaustion."""

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sarvamai.core.api_error import ApiError

from config.settings import Settings, _debug_loaded
from transcription.sarvam_client import (
    SarvamTranscriptionClient,
    TranscriptionError,
    TranscriptionQuotaError,
)


def _client(error=None, response=None):
    client = SarvamTranscriptionClient.__new__(SarvamTranscriptionClient)
    client._settings = Settings(sarvam_api_key="test-secret-key")
    client._client = MagicMock()
    client._client.speech_to_text_job.create_job.side_effect = error
    client._client.speech_to_text.transcribe.side_effect = error
    if response is not None:
        client._client.speech_to_text.transcribe.return_value = response
    return client


def _raises(error_type, callback):
    try:
        callback()
    except error_type:
        return
    raise AssertionError(f"Expected {error_type.__name__}")


def test_http_402_quota_code_is_classified() -> None:
    error = ApiError(status_code=402, body={"error": {"code": "insufficient_quota_error"}})
    _raises(TranscriptionQuotaError, lambda: _client(error)._transcribe_batch(Path("audio.wav")))


def test_structured_no_credits_message_is_classified() -> None:
    error = ApiError(status_code=400, body={"error": {"message": "No credits available."}})
    _raises(TranscriptionQuotaError, lambda: _client(error)._transcribe_realtime(Path(__file__)))


def test_normal_api_failure_is_not_quota() -> None:
    error = ApiError(status_code=500, body={"error": {"message": "Server error"}})
    try:
        _client(error)._transcribe_batch(Path("audio.wav"))
    except TranscriptionError as exc:
        assert not isinstance(exc, TranscriptionQuotaError)
    else:
        raise AssertionError("Expected TranscriptionError")


def test_timeout_is_not_quota() -> None:
    try:
        _client(TimeoutError("timed out"))._transcribe_batch(Path("audio.wav"))
    except TranscriptionError as exc:
        assert not isinstance(exc, TranscriptionQuotaError)
    else:
        raise AssertionError("Expected TranscriptionError")


def test_successful_realtime_response_is_unchanged() -> None:
    response = MagicMock(transcript="Hello world.", language_code="en-IN")
    result = _client(response=response)._transcribe_realtime(Path(__file__))
    assert result.transcript == "Hello world."
    assert result.language_code == "en-IN"


def test_api_keys_and_prefixes_are_not_logged() -> None:
    output = io.StringIO()
    with redirect_stdout(output):
        _debug_loaded("SARVAM_API_KEY", ".env", "sarvam-secret-value")
        _debug_loaded("GEMINI_API_KEY", ".env", "gemini-secret-value")
    with patch("transcription.sarvam_client.SarvamAI"), patch("transcription.sarvam_client.logger") as logger:
        SarvamTranscriptionClient(Settings(sarvam_api_key="sarvam-secret-value"))
    logged = output.getvalue() + repr(logger.info.call_args_list)
    assert "sarvam-secret-value" not in logged
    assert "gemini-secret-value" not in logged
    assert "sarvam-secret" not in logged
    assert "gemini-secret" not in logged
