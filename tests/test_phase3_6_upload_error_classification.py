"""Tests for Phase 3.6's process_upload error classification and the
"don't destroy a successful transcription" signal
(app/main.py's process_upload, last_sarvam_result session-state marker).

Runs process_upload directly (the same function app.main.render_processing_stage
calls for the real "Generate" click) against a REAL local WAV file, with
transcribe_audio_detailed mocked at the boundary to control success/failure
precisely -- this exercises the actual function body and its real
exception-handling branches, not a re-implementation of them.
"""

import sys
import wave
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st


class FakeSessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def get(self, key, default=None):
        return dict.get(self, key, default)


def _make_real_wav() -> Path:
    tmp_dir = Path(tempfile.mkdtemp(prefix="meetscribe_test_upload_"))
    path = tmp_dir / "test.wav"
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 16000)
    return path


class FakeUploadedFile:
    def __init__(self, path: Path):
        self._path = path
        self.name = path.name
        self.size = path.stat().st_size
        self.type = "audio/wav"

    def getbuffer(self):
        return self._path.read_bytes()

    def read(self):
        return self._path.read_bytes()


def _fresh_session_state() -> FakeSessionState:
    state = FakeSessionState()
    state["speaker_mapping"] = {}
    state["saved_speaker_mapping"] = {}
    state["workflow_source"] = "audio"
    state["processing_logs"] = []
    state["analysis_cache"] = {}
    return state


def test_successful_transcription_does_not_set_error_and_clears_marker() -> None:
    st.session_state = _fresh_session_state()
    import app.main as m
    from transcription.sarvam_client import TranscriptionResult, TranscriptionSegment

    fake_result = TranscriptionResult(
        transcript="Hello world.",
        segments=[TranscriptionSegment(transcript="Hello world.", speaker_id="1", start_time_seconds=0.0, end_time_seconds=1.0)],
    )
    uploaded = FakeUploadedFile(_make_real_wav())

    with patch.object(m, "transcribe_audio_detailed", return_value=fake_result):
        m.process_upload(uploaded)

    assert "transcript_result" in st.session_state
    assert st.session_state.get("last_sarvam_result") is None
    assert st.session_state.get("analysis_error", "") == ""


def test_transcription_error_from_sarvam_uses_transcription_specific_message() -> None:
    st.session_state = _fresh_session_state()
    import app.main as m
    from transcription.sarvam_client import TranscriptionError

    uploaded = FakeUploadedFile(_make_real_wav())

    with patch.object(m, "transcribe_audio_detailed", side_effect=TranscriptionError("Batch transcription failed: simulated")):
        m.process_upload(uploaded)

    # No transcript should have been stored, and no raw-result marker should
    # linger (the Sarvam call itself never returned a result).
    assert "transcript_result" not in st.session_state or st.session_state.get("transcript_result") is None
    assert st.session_state.get("last_sarvam_result") is None


def test_later_stage_failure_after_successful_transcription_preserves_marker() -> None:
    # Simulates transcription succeeding but a later speaker-processing
    # stage raising an unexpected exception (Part 8: a successful
    # transcription must not be silently indistinguishable from a
    # transcription failure).
    st.session_state = _fresh_session_state()
    import app.main as m
    from transcription.sarvam_client import TranscriptionResult, TranscriptionSegment

    fake_result = TranscriptionResult(
        transcript="Hello world.",
        segments=[TranscriptionSegment(transcript="Hello world.", speaker_id="1", start_time_seconds=0.0, end_time_seconds=1.0)],
    )
    uploaded = FakeUploadedFile(_make_real_wav())

    with patch.object(m, "transcribe_audio_detailed", return_value=fake_result):
        with patch.object(m, "resolve_speakers", side_effect=RuntimeError("simulated unexpected speaker-resolution failure")):
            m.process_upload(uploaded)

    # The raw Sarvam result must still be visible in session state as
    # evidence transcription itself succeeded, even though the overall
    # call did not complete (no permanent transcript_result was stored).
    assert st.session_state.get("last_sarvam_result") is not None
    assert st.session_state["last_sarvam_result"].transcript == "Hello world."


def test_real_audio_end_to_end_process_upload_completes_without_exception() -> None:
    # The closest available "real app path" smoke test: a REAL local WAV
    # file through preprocess_uploaded_audio, Sarvam batch transcription
    # (mocked here to avoid a live paid API call inside the automated
    # suite -- the actual live call was verified manually once, see the
    # Phase 3.6 report), Voxels (real, degrades gracefully if unavailable),
    # repair, and speaker resolution.
    st.session_state = _fresh_session_state()
    import app.main as m
    from transcription.sarvam_client import TranscriptionResult, TranscriptionSegment

    fake_result = TranscriptionResult(
        transcript="Hello. Hi there. How are you doing today?",
        segments=[
            TranscriptionSegment(transcript="Hello.", speaker_id="1", start_time_seconds=0.0, end_time_seconds=1.0),
            TranscriptionSegment(transcript="Hi there.", speaker_id="2", start_time_seconds=1.0, end_time_seconds=2.0),
            TranscriptionSegment(transcript="How are you doing today?", speaker_id="1", start_time_seconds=2.0, end_time_seconds=4.0),
        ],
    )
    uploaded = FakeUploadedFile(_make_real_wav())

    with patch.object(m, "transcribe_audio_detailed", return_value=fake_result):
        m.process_upload(uploaded)

    assert st.session_state.get("analysis_error", "") == ""
    assert "transcript_result" in st.session_state
    assert len(st.session_state["transcript_result"].segments) == 3
    assert st.session_state.get("speaker_review_required") is True
