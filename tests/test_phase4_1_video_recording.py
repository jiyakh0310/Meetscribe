"""SELaD Phase 4.1 -- direct video meeting recording / acquisition tests.

Covers the Python/integration boundary of the new camera+microphone
recorder (Part 24), and confirms a recorded meeting (video+audio) feeds
BOTH the existing, unmodified audio pipeline AND the existing,
unmodified Phase 4 analyze_video() (Part 25). Browser MediaRecorder
behavior itself cannot be exercised in this Python test environment --
see tests/test_phase4_1_ui_apptest.py for UI-level coverage and the
final report's manual browser-validation checklist for what a human
must verify in a real browser.
"""

from __future__ import annotations

import base64
import io
import subprocess
import sys
import tempfile
import urllib.request
import wave
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import streamlit as st

from video_analytics.preprocessing import SUPPORTED_VIDEO_EXTENSIONS


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


def _fresh_session_state() -> FakeSessionState:
    state = FakeSessionState()
    state["speaker_mapping"] = {}
    state["saved_speaker_mapping"] = {}
    state["workflow_source"] = "audio"
    state["processing_logs"] = []
    state["analysis_cache"] = {}
    state["recording_meeting_state"] = None
    state["recording_meeting_version"] = 0
    state["recording_meeting_open"] = False
    state["recording_video_meeting_state"] = None
    state["recording_video_meeting_version"] = 0
    state["recording_video_meeting_open"] = False
    state["video_analytics_result"] = None
    return state


# ---------------------------------------------------------------------------
# Fixtures: a synthetic "browser recording" -- a webm container with both a
# real-face video track and an audio track, built with ffmpeg (the same
# binary discovery Phase 4's preprocessing.py already uses), standing in
# for what MediaRecorder would have produced client-side.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def recorded_meeting_webm() -> Path | None:
    try:
        import imageio_ffmpeg

        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None

    lena_path = Path(tempfile.gettempdir()) / "phase4_1_test_lena.jpg"
    if not lena_path.exists():
        try:
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
                str(lena_path),
            )
        except Exception:
            return None

    output_path = Path(tempfile.gettempdir()) / "phase4_1_recorded_face_test.webm"
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    command = [
        ffmpeg_bin, "-y",
        "-loop", "1", "-i", str(lena_path),
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=12",
        "-t", "12",
        "-c:v", "libvpx", "-c:a", "libopus", "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    try:
        subprocess.run(command, capture_output=True, check=True, timeout=120)
    except Exception:
        return None
    if not output_path.exists() or output_path.stat().st_size == 0:
        return None
    return output_path


def _recording_payload(video_path: Path) -> dict:
    data = video_path.read_bytes()
    return {
        "name": "meeting-video-test.webm",
        "mime_type": "video/webm",
        "duration_seconds": 12,
        "size_bytes": len(data),
        "data_base64": base64.b64encode(data).decode("ascii"),
    }


# ---------------------------------------------------------------------------
# 1-2: recorder payload conversion produces valid bytes + retains MIME/name
# ---------------------------------------------------------------------------


def test_video_recording_payload_converts_to_valid_upload_bytes(recorded_meeting_webm) -> None:
    if recorded_meeting_webm is None:
        pytest.skip("ffmpeg or network fixture unavailable in this environment.")
    import app.main as m

    payload = _recording_payload(recorded_meeting_webm)
    upload = m._video_recording_payload_to_upload(payload)

    assert isinstance(upload, io.BytesIO)
    assert upload.getbuffer().nbytes == recorded_meeting_webm.stat().st_size
    assert upload.getbuffer().nbytes == payload["size_bytes"]


def test_video_recording_payload_retains_mime_and_name(recorded_meeting_webm) -> None:
    if recorded_meeting_webm is None:
        pytest.skip("ffmpeg or network fixture unavailable in this environment.")
    import app.main as m

    payload = _recording_payload(recorded_meeting_webm)
    upload = m._video_recording_payload_to_upload(payload)

    assert upload.name == "meeting-video-test.webm"
    assert upload.type == "video/webm"
    assert Path(upload.name).suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS


# ---------------------------------------------------------------------------
# 3-4: empty / malformed recordings are handled safely
# ---------------------------------------------------------------------------


def test_empty_recording_payload_produces_zero_length_upload() -> None:
    import app.main as m

    payload = {"name": "empty.webm", "mime_type": "video/webm", "duration_seconds": 0, "size_bytes": 0, "data_base64": ""}
    upload = m._video_recording_payload_to_upload(payload)

    assert upload.getbuffer().nbytes == 0


def test_render_video_recorder_card_rejects_empty_recorded_event() -> None:
    # Mirrors render_record_video_meeting_card's guard: an "event":
    # "recorded" payload with no data/size must not be accepted as a
    # usable pending recording.
    import app.main as m

    st.session_state = _fresh_session_state()

    class FakeComponent:
        def __call__(self, *args, **kwargs):
            return {"event": "recorded", "recording": {"name": "x.webm", "mime_type": "video/webm", "duration_seconds": 0, "size_bytes": 0, "data_base64": ""}}

    with patch.object(m, "VIDEO_MEETING_RECORDER_COMPONENT", FakeComponent()), patch.object(m.st, "error") as show_error, patch.object(m.st, "button", return_value=False), patch.object(m.st, "rerun", side_effect=RuntimeError("stop")):
        with pytest.raises(RuntimeError):
            m.render_record_video_meeting_card()

    assert st.session_state.get("recording_video_meeting_state") is None
    assert show_error.call_count == 1


def test_malformed_base64_recording_does_not_crash_conversion() -> None:
    import app.main as m

    payload = {"name": "bad.webm", "mime_type": "video/webm", "duration_seconds": 5, "size_bytes": 5, "data_base64": "not-valid-base64!!!"}
    # base64.b64decode with default validate=False silently drops invalid
    # characters rather than raising -- conversion must not crash either way.
    upload = m._video_recording_payload_to_upload(payload)
    assert isinstance(upload, io.BytesIO)


# ---------------------------------------------------------------------------
# 5-6: record-again replaces pending recording; no double processing
# ---------------------------------------------------------------------------


def test_record_again_clears_previous_pending_recording() -> None:
    import app.main as m

    st.session_state = _fresh_session_state()
    st.session_state["recording_video_meeting_state"] = {"name": "first.webm", "size_bytes": 10}
    st.session_state["recording_video_meeting_version"] = 0

    with patch.object(m.st, "rerun", side_effect=RuntimeError("stop")):
        with pytest.raises(RuntimeError):
            m.clear_video_recording_state()

    assert st.session_state.get("recording_video_meeting_state") is None
    assert st.session_state.get("recording_video_meeting_version") == 1


def test_start_recorded_video_meeting_workflow_consumes_pending_state_once() -> None:
    import app.main as m

    st.session_state = _fresh_session_state()
    recording = {"name": "meeting.webm", "mime_type": "video/webm", "duration_seconds": 3, "size_bytes": 4, "data_base64": base64.b64encode(b"data").decode()}
    st.session_state["recording_video_meeting_state"] = recording

    with patch.object(m.st, "rerun", side_effect=RuntimeError("stop")):
        with pytest.raises(RuntimeError):
            m.start_recorded_video_meeting_workflow(recording)

    # The pending recording is cleared as part of starting the workflow, so
    # a second "process" click (e.g. a double-submit) has nothing left to act on.
    assert st.session_state.get("recording_video_meeting_state") is None
    assert st.session_state.get("workflow_pending_action") == "prepare"
    assert st.session_state.get("workflow_source") == "audio"
    assert getattr(st.session_state.get("workflow_file"), "name", "") == "meeting.webm"


# ---------------------------------------------------------------------------
# 7-10: isolation from existing workflows
# ---------------------------------------------------------------------------


def _make_wav_upload() -> "FakeUploadedFile":
    tmp_dir = Path(tempfile.mkdtemp(prefix="meetscribe_test_video_recording_"))
    path = tmp_dir / "test.wav"
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 16000)
    return FakeUploadedFile(path)


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


def test_video_recorder_failure_does_not_affect_plain_audio_upload() -> None:
    # A plain .wav upload must never trigger the video-preservation branch
    # or analyze_uploaded_video at all -- recorder/video-analytics code is
    # simply not on this path (Part 7 / freeze on existing audio pipeline).
    import app.main as m
    from transcription.sarvam_client import TranscriptionResult, TranscriptionSegment

    st.session_state = _fresh_session_state()
    fake_result = TranscriptionResult(
        transcript="Hello world.",
        segments=[TranscriptionSegment(transcript="Hello world.", speaker_id="1", start_time_seconds=0.0, end_time_seconds=1.0)],
    )
    uploaded = _make_wav_upload()

    with patch.object(m, "transcribe_audio_detailed", return_value=fake_result), patch.object(m, "analyze_uploaded_video") as mock_video_analytics:
        m.process_upload(uploaded)

    mock_video_analytics.assert_not_called()
    assert st.session_state.get("video_analytics_result") is None
    assert "transcript_result" in st.session_state


def test_video_upload_still_triggers_both_existing_branches(recorded_meeting_webm) -> None:
    if recorded_meeting_webm is None:
        pytest.skip("ffmpeg or network fixture unavailable in this environment.")
    import app.main as m
    from transcription.sarvam_client import TranscriptionResult, TranscriptionSegment

    st.session_state = _fresh_session_state()
    fake_result = TranscriptionResult(
        transcript="Hello world.",
        segments=[TranscriptionSegment(transcript="Hello world.", speaker_id="1", start_time_seconds=0.0, end_time_seconds=1.0)],
    )

    class FakeVideoUpload:
        def __init__(self, path: Path):
            self._path = path
            self.name = "uploaded-meeting.webm"
            self.size = path.stat().st_size
            self.type = "video/webm"

        def getbuffer(self):
            return self._path.read_bytes()

        def read(self):
            return self._path.read_bytes()

    uploaded = FakeVideoUpload(recorded_meeting_webm)

    with patch.object(m, "transcribe_audio_detailed", return_value=fake_result):
        m.process_upload(uploaded)

    assert "transcript_result" in st.session_state
    video_result = st.session_state.get("video_analytics_result")
    assert video_result is not None
    assert video_result.available is True


def test_audio_only_workflow_unaffected_by_video_recording_additions() -> None:
    import app.main as m
    from transcription.sarvam_client import TranscriptionResult, TranscriptionSegment

    st.session_state = _fresh_session_state()
    fake_result = TranscriptionResult(
        transcript="Hello world.",
        segments=[TranscriptionSegment(transcript="Hello world.", speaker_id="1", start_time_seconds=0.0, end_time_seconds=1.0)],
    )
    uploaded = _make_wav_upload()

    with patch.object(m, "transcribe_audio_detailed", return_value=fake_result):
        m.process_upload(uploaded)

    assert st.session_state.get("analysis_error", "") == ""
    assert "transcript_result" in st.session_state


# ---------------------------------------------------------------------------
# Part 25: recorded video -> BOTH existing independent branches
# ---------------------------------------------------------------------------


def test_recorded_video_feeds_existing_audio_extraction_path(recorded_meeting_webm) -> None:
    if recorded_meeting_webm is None:
        pytest.skip("ffmpeg or network fixture unavailable in this environment.")
    from transcription.audio_utils import preprocess_uploaded_audio

    wav_path = preprocess_uploaded_audio(recorded_meeting_webm, filename=recorded_meeting_webm.name)
    try:
        assert wav_path.exists()
        assert wav_path.suffix == ".wav"
        assert wav_path.stat().st_size > 0
    finally:
        wav_path.unlink(missing_ok=True)


def test_recorded_video_feeds_existing_phase4_analyze_video(recorded_meeting_webm) -> None:
    if recorded_meeting_webm is None:
        pytest.skip("ffmpeg or network fixture unavailable in this environment.")
    from video_analytics import analyze_video

    result = analyze_video(recorded_meeting_webm)

    assert result.available is True
    assert result.metadata is not None
    assert result.metadata.duration_seconds > 0
    assert result.sampled_frame_count > 0
    assert result.detected_tracks == ("Face A",)
    assert len(result.windows) >= 1

    window = result.windows[0]
    assert window.track_summaries
    track_summary = window.track_summaries[0]
    assert len(track_summary.aggregated_probabilities) == 7  # canonical seven classes

    # Only one participant is visible -- synchrony/interaction must
    # correctly report unavailable, never a fabricated value.
    assert window.aggregate_synchrony is None
    assert window.interaction.available is False
    assert isinstance(result.quality_warnings, tuple)
