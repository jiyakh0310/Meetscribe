"""Tests for Phase 3.6's Sarvam SDK exception-classification safety net
(transcription/sarvam_client.py's ``_transcribe_realtime``/``_transcribe_batch``).

Background: a reported runtime blocker ("Something went wrong while
preparing your report") turned out, on investigation, to have no
reproducible code-level cause in this repository -- a real, live Sarvam
batch transcription call (see the Phase 3.6 report) completed
successfully end-to-end. What WAS found and fixed is a real robustness
gap: several stages of the Sarvam SDK interaction (job status checks,
result download, output extraction) ran OUTSIDE any exception handler
that classifies failures as ``TranscriptionError`` -- any unexpected
exception type there escaped as a raw, unclassified exception straight
to app.main.process_upload's generic catch-all, which shows the user an
unhelpful message with no indication the problem was the transcription
service specifically.

These tests mock the Sarvam SDK client to simulate exception types this
project's code does not explicitly enumerate (matching Part 9.F/G/H of
the Phase 3.6 task: mocked API failure, malformed/missing segment
field), proving every such failure is now classified as
``TranscriptionError`` rather than escaping raw.

Style matches the rest of this repo: plain pytest-style functions with
bare asserts, run via a manual script (no pytest installed in this venv).
"""

import sys
import tempfile
import wave
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import Settings
from transcription.sarvam_client import SarvamTranscriptionClient, TranscriptionError


def _settings() -> Settings:
    return Settings(sarvam_api_key="test-key-not-real")


def _client_with_mock_sdk(mock_sdk_client) -> SarvamTranscriptionClient:
    client = SarvamTranscriptionClient.__new__(SarvamTranscriptionClient)
    client._settings = _settings()
    client._client = mock_sdk_client
    return client


def _make_real_wav() -> Path:
    tmp_dir = Path(tempfile.mkdtemp(prefix="meetscribe_test_audio_"))
    path = tmp_dir / "test.wav"
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 16000)  # 1 second of silence
    return path


def _assert_raises_transcription_error(callable_fn, *, match: str | None = None) -> None:
    try:
        callable_fn()
    except TranscriptionError as exc:
        if match is not None:
            assert match in str(exc), f"expected {match!r} in {exc!r}"
        return
    raise AssertionError("expected TranscriptionError to be raised, but nothing was raised")


# --- Realtime path: unexpected SDK exception types -----------------------------------


def test_realtime_unexpected_exception_is_classified_as_transcription_error() -> None:
    mock_sdk = MagicMock()
    mock_sdk.speech_to_text.transcribe.side_effect = KeyError("unexpected_field")
    client = _client_with_mock_sdk(mock_sdk)
    audio_path = _make_real_wav()

    _assert_raises_transcription_error(lambda: client._transcribe_realtime(audio_path))


def test_realtime_attribute_error_from_malformed_response_is_classified() -> None:
    # Simulates a response object missing an expected attribute (a
    # "changed/malformed Sarvam response schema" scenario).
    mock_sdk = MagicMock()
    mock_response = MagicMock(spec=[])  # no attributes at all, including .transcript
    mock_sdk.speech_to_text.transcribe.return_value = mock_response
    client = _client_with_mock_sdk(mock_sdk)
    audio_path = _make_real_wav()

    _assert_raises_transcription_error(lambda: client._transcribe_realtime(audio_path))


# --- Batch path: unexpected exceptions during submission -----------------------------


def test_batch_unexpected_exception_during_job_creation_is_classified() -> None:
    mock_sdk = MagicMock()
    mock_sdk.speech_to_text_job.create_job.side_effect = ValueError("unexpected SDK internal error")
    client = _client_with_mock_sdk(mock_sdk)
    audio_path = _make_real_wav()

    _assert_raises_transcription_error(
        lambda: client._transcribe_batch(audio_path, with_diarization=True)
    )


def test_batch_unexpected_exception_during_upload_is_classified() -> None:
    mock_sdk = MagicMock()
    mock_job = MagicMock()
    mock_sdk.speech_to_text_job.create_job.return_value = mock_job
    mock_job.upload_files.side_effect = AttributeError("unexpected attribute access in SDK")
    client = _client_with_mock_sdk(mock_sdk)
    audio_path = _make_real_wav()

    _assert_raises_transcription_error(
        lambda: client._transcribe_batch(audio_path, with_diarization=True)
    )


# --- Batch path: unexpected exceptions AFTER polling (the real gap found) ------------


def test_batch_unexpected_exception_during_status_check_is_classified() -> None:
    # Simulates status.job_state having an unexpected type (e.g. None
    # instead of a string) -- a "changed/malformed Sarvam response schema"
    # failure in the post-polling phase that previously ran OUTSIDE any
    # exception handler entirely.
    mock_sdk = MagicMock()
    mock_job = MagicMock()
    mock_sdk.speech_to_text_job.create_job.return_value = mock_job
    mock_status = MagicMock()
    mock_status.job_state = None  # .lower() on None -> AttributeError
    mock_job.wait_until_complete.return_value = mock_status
    client = _client_with_mock_sdk(mock_sdk)
    audio_path = _make_real_wav()

    _assert_raises_transcription_error(
        lambda: client._transcribe_batch(audio_path, with_diarization=True)
    )


def test_batch_unexpected_exception_during_download_is_classified() -> None:
    # download_outputs() performs real network I/O -- a connection error
    # or any other unexpected exception there previously escaped raw.
    mock_sdk = MagicMock()
    mock_job = MagicMock()
    mock_sdk.speech_to_text_job.create_job.return_value = mock_job
    mock_status = MagicMock()
    mock_status.job_state = "completed"
    mock_job.wait_until_complete.return_value = mock_status
    mock_job.is_failed.return_value = False
    mock_job.download_outputs.side_effect = ConnectionError("connection reset")
    client = _client_with_mock_sdk(mock_sdk)
    audio_path = _make_real_wav()

    _assert_raises_transcription_error(
        lambda: client._transcribe_batch(audio_path, with_diarization=True)
    )


def test_batch_malformed_status_object_missing_job_state_is_classified() -> None:
    # A response object missing job_state entirely (spec=[] means any
    # attribute access raises AttributeError, not a graceful default).
    mock_sdk = MagicMock()
    mock_job = MagicMock()
    mock_sdk.speech_to_text_job.create_job.return_value = mock_job
    mock_status = MagicMock(spec=[])
    mock_job.wait_until_complete.return_value = mock_status
    client = _client_with_mock_sdk(mock_sdk)
    audio_path = _make_real_wav()

    _assert_raises_transcription_error(
        lambda: client._transcribe_batch(audio_path, with_diarization=True)
    )


# --- Existing (already-classified) failure types remain classified correctly ---------


def test_batch_explicit_job_failure_still_reports_as_transcription_error() -> None:
    mock_sdk = MagicMock()
    mock_job = MagicMock()
    mock_sdk.speech_to_text_job.create_job.return_value = mock_job
    mock_status = MagicMock()
    mock_status.job_state = "failed"
    mock_job.wait_until_complete.return_value = mock_status
    mock_job.is_failed.return_value = True
    mock_job.get_file_results.return_value = {"failed": [{"error_message": "audio too short"}]}
    client = _client_with_mock_sdk(mock_sdk)
    audio_path = _make_real_wav()

    _assert_raises_transcription_error(
        lambda: client._transcribe_batch(audio_path, with_diarization=True),
        match="audio too short",
    )
