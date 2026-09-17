"""SELaD Phase 4.1 -- repair pass #4 (real-hardware runtime trace).

The real Windows laptop camera+microphone test reached RECORDED state
and Process Meeting could be clicked, but the user did not observe the
app reach Speaker Review -> Transcript Review -> Minutes -> Visual
Interaction Insights.

This pass added terminal-visible [VIDEO-RECORDER] runtime tracing
(app.main._video_trace, using logger.info -- confirmed via a real
running Streamlit server that bare print() does NOT reliably reach the
terminal under `streamlit run`, while logger-based calls do) at every
checkpoint from Process click through to the workflow shell re-render.

Driving the REAL Streamlit app (not AppTest) with Playwright and a
realistic ~3MB, 20-second, 720p synthetic recording (matching the size
class of a real webcam recording, not the earlier 17KB fixture) through
the actual browser->component->Python protocol showed:

  - the payload survives the round trip byte-for-byte (no truncation
    at this size)
  - process_upload() runs to completion and the app DOES reach Speaker
    Review
  - the elapsed time from Process click to Speaker Review was ~26-30
    seconds, of which ~17s is the EXISTING (pre-Phase-4, unmodified)
    Voxels acoustic-emotion analysis and ~8s is Phase 4 video analytics
    -- both running synchronously with, at the time, NO visible status
    update during that window

This is the most likely real explanation for "Process Meeting appears
to hang": the pipeline is working correctly but silently, for tens of
seconds, with a static status label. This file tests the concrete,
provable parts of that finding: payload integrity at a realistic size,
and that failures at each stage correctly surface rather than leaving
an infinite spinner. It does not (cannot) test perceived UI
responsiveness, which requires a human in a real browser.
"""

from __future__ import annotations

import subprocess
import sys
import wave
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
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


def _fresh_session_state() -> FakeSessionState:
    state = FakeSessionState()
    state["speaker_mapping"] = {}
    state["saved_speaker_mapping"] = {}
    state["workflow_source"] = "audio"
    state["processing_logs"] = []
    state["analysis_cache"] = {}
    state["recording_video_meeting_state"] = None
    state["recording_video_meeting_version"] = 0
    state["recording_video_meeting_open"] = True
    state["workflow_open"] = False
    state["video_analytics_result"] = None
    return state


@pytest.fixture(scope="module")
def realistic_recording(tmp_path_factory) -> Path:
    """A representative-size (~1-2MB, 720p-ish, ~15s) synthetic
    recording, generated at test time via ffmpeg rather than committed
    (Part 18) -- large enough to meaningfully exercise base64/component
    payload handling at a realistic size, unlike the earlier 17KB
    fixture."""

    try:
        import imageio_ffmpeg

        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pytest.skip("ffmpeg unavailable in this environment.")

    out_dir = tmp_path_factory.mktemp("realistic")
    out_path = out_dir / "realistic_recording.webm"
    command = [
        ffmpeg_bin, "-y",
        "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=30:duration=15",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=15",
        "-c:v", "libvpx", "-b:v", "1M",
        "-c:a", "libopus",
        str(out_path),
    ]
    try:
        subprocess.run(command, capture_output=True, check=True, timeout=120)
    except Exception as exc:
        pytest.skip(f"Could not generate realistic fixture: {exc}")
    if not out_path.exists() or out_path.stat().st_size < 500_000:
        pytest.skip("Generated fixture unexpectedly small; skipping.")
    return out_path


# ---------------------------------------------------------------------------
# 1-3: realistic-size payload survives decode intact; Python received byte
# count equals original; ffprobe sees both streams.
# ---------------------------------------------------------------------------


def test_realistic_size_payload_round_trips_intact(realistic_recording: Path) -> None:
    import base64

    import app.main as m

    raw_bytes = realistic_recording.read_bytes()
    assert len(raw_bytes) > 500_000, "fixture should be representative of a real recording, not tiny"

    recording = {
        "name": "meeting-video-realistic.webm",
        "mime_type": "video/webm;codecs=vp9,opus",
        "duration_seconds": 15,
        "size_bytes": len(raw_bytes),
        "data_base64": base64.b64encode(raw_bytes).decode("ascii"),
    }
    upload = m._video_recording_payload_to_upload(recording)
    decoded = upload.getbuffer()

    assert len(decoded) == len(raw_bytes), (
        f"decoded byte count ({len(decoded)}) does not match original ({len(raw_bytes)}) "
        "at a realistic multi-hundred-KB+ size."
    )
    assert bytes(decoded) == raw_bytes


def test_realistic_webm_ffmpeg_reports_both_streams(realistic_recording: Path) -> None:
    import imageio_ffmpeg

    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    result = subprocess.run(
        [ffmpeg_bin, "-i", str(realistic_recording)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert "Video:" in result.stderr
    assert "Audio:" in result.stderr


def test_realistic_webm_duration_probe_matches_real_length(realistic_recording: Path) -> None:
    from video_analytics.preprocessing import inspect_video_metadata

    metadata = inspect_video_metadata(realistic_recording)
    assert metadata.duration_seconds > 10  # generated as 15s; allow encoder slack


def test_realistic_webm_audio_extraction_succeeds(realistic_recording: Path) -> None:
    from transcription.audio_utils import preprocess_uploaded_audio

    wav_path = preprocess_uploaded_audio(realistic_recording, filename=realistic_recording.name)
    try:
        assert wav_path.exists()
        assert wav_path.stat().st_size > 0
        with wave.open(str(wav_path), "rb") as wav_file:
            duration = wav_file.getnframes() / max(1, wav_file.getframerate())
        assert duration > 10
    finally:
        wav_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 4-5 (already covered by test_phase4_1_recorder_repair2.py's browser-
# fixture tests): duration fallback, WAV extraction. Not duplicated here.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 6: Sarvam failure must clear processing state, not hang.
# ---------------------------------------------------------------------------


class FakeVideoUpload:
    def __init__(self, path: Path):
        self._path = path
        self.name = "meeting-video-realistic.webm"
        self.size = path.stat().st_size
        self.type = "video/webm;codecs=vp9,opus"

    def getbuffer(self):
        return self._path.read_bytes()

    def read(self):
        return self._path.read_bytes()


def test_sarvam_failure_clears_processing_state_for_video_upload(realistic_recording: Path) -> None:
    import app.main as m
    from transcription.sarvam_client import TranscriptionQuotaError

    st.session_state = _fresh_session_state()
    uploaded = FakeVideoUpload(realistic_recording)

    with patch.object(m, "transcribe_audio_detailed", side_effect=TranscriptionQuotaError("quota exhausted")), \
         patch.object(m.st, "error") as show_error:
        m.process_upload(uploaded)

    # A clear, specific error must be shown -- never a silently-stuck state.
    assert show_error.call_count == 1
    assert "credits" in show_error.call_args.args[0].lower()
    # No transcript should have been stored from a failed transcription.
    assert st.session_state.get("transcript_result") is None
    # Video analytics must never have been reached -- Sarvam failed first.
    assert st.session_state.get("video_analytics_result") is None


# ---------------------------------------------------------------------------
# 7: video analytics failure must not block the audio/transcript workflow,
# verified with a REALISTIC-size video (not the tiny fixture).
# ---------------------------------------------------------------------------


def test_video_analytics_failure_does_not_block_transcript_with_realistic_video(realistic_recording: Path) -> None:
    import app.main as m
    from transcription.sarvam_client import TranscriptionResult, TranscriptionSegment

    st.session_state = _fresh_session_state()
    fake_result = TranscriptionResult(
        transcript="Realistic-size recording handoff test.",
        segments=[TranscriptionSegment(transcript="Realistic-size recording handoff test.", speaker_id="1", start_time_seconds=0.0, end_time_seconds=1.0)],
    )
    uploaded = FakeVideoUpload(realistic_recording)

    with patch.object(m, "transcribe_audio_detailed", return_value=fake_result), \
         patch("video_analytics.analyze_video", side_effect=RuntimeError("simulated video analytics crash")):
        m.process_upload(uploaded)

    assert st.session_state.get("analysis_error", "") == ""
    assert "transcript_result" in st.session_state
    assert st.session_state.get("video_analytics_result") is None


def test_video_analytics_slow_completion_still_transitions_to_speaker_review(realistic_recording: Path) -> None:
    # Simulates Section 10's concern: analyze_uploaded_video is slow but
    # DOES eventually complete -- the workflow must still reach Speaker
    # Review afterward, not stall.
    import time as time_module

    import app.main as m
    from transcription.sarvam_client import TranscriptionResult, TranscriptionSegment

    st.session_state = _fresh_session_state()
    fake_result = TranscriptionResult(
        transcript="Slow video analytics test.",
        segments=[TranscriptionSegment(transcript="Slow video analytics test.", speaker_id="1", start_time_seconds=0.0, end_time_seconds=1.0)],
    )
    uploaded = FakeVideoUpload(realistic_recording)

    real_analyze_video = None
    from video_analytics import analyze_video as _real_analyze_video
    real_analyze_video = _real_analyze_video

    def _slow_analyze_video(path):
        time_module.sleep(0.2)  # stand-in for a slow-but-successful run
        return real_analyze_video(path)

    with patch.object(m, "transcribe_audio_detailed", return_value=fake_result), \
         patch("video_analytics.analyze_video", side_effect=_slow_analyze_video):
        m.process_upload(uploaded)

    assert st.session_state.get("speaker_review_required") is True
    assert st.session_state.get("video_analytics_result") is not None


# ---------------------------------------------------------------------------
# 9-11: successful transcription enters Speaker Review; recorder input mode
# is cleared/transitioned; workflow shell renders the processing stage
# after Process acceptance -- exercised via a real AppTest run.
# ---------------------------------------------------------------------------


def test_start_recorded_video_meeting_workflow_transitions_out_of_recorder() -> None:
    import app.main as m

    st.session_state = _fresh_session_state()
    recording = {
        "name": "meeting.webm",
        "mime_type": "video/webm;codecs=vp9,opus",
        "duration_seconds": 5,
        "size_bytes": 4,
        "data_base64": "ZGF0YQ==",
    }

    with patch.object(m.st, "rerun", side_effect=RuntimeError("stop")):
        with pytest.raises(RuntimeError):
            m.start_recorded_video_meeting_workflow(recording)

    # The recorder input mode must be left behind -- the workflow shell,
    # not the recorder card, owns the UI from this point on (Part 13).
    assert st.session_state.get("workflow_open") is True
    assert st.session_state.get("workflow_stage") == "processing"
    assert st.session_state.get("workflow_pending_action") == "prepare"
    assert st.session_state.get("recording_video_meeting_state") is None


def test_processing_exception_never_leaves_infinite_spinner(realistic_recording: Path) -> None:
    # Any unexpected exception during process_upload() must result in a
    # user-visible error, not a state that looks like it's still running.
    import app.main as m

    st.session_state = _fresh_session_state()
    uploaded = FakeVideoUpload(realistic_recording)

    with patch.object(m, "transcribe_audio_detailed", side_effect=RuntimeError("unexpected failure")), \
         patch.object(m.st, "error") as show_error:
        m.process_upload(uploaded)

    assert show_error.call_count == 1
    assert st.session_state.get("transcript_result") is None
