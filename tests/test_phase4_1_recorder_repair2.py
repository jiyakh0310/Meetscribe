"""SELaD Phase 4.1 -- final end-to-end repair (real browser test #3).

Regression tests for the two real functional failures found by the
authoritative real-browser test after the previous UI-visibility repair:

1. AUDIO: recorded playback silent in some cases / server-side audio
   extraction receiving a corrupted payload -- root cause was
   `dataUrl.split(",")[1]` truncating the base64 payload whenever
   `mediaRecorder.mimeType` itself contains a comma (e.g.
   "video/webm;codecs=vp9,opus" -> naive split lands on "opus;base64",
   not the actual recording bytes).

2. PROCESS: Process Meeting appeared to hang with no downstream
   MeetScribe processing -- root cause was NOT a broken Streamlit
   handoff (proven working via a real Streamlit server + Playwright
   test), but video_analytics/preprocessing.py's inspect_video_metadata()
   reporting duration_seconds=0.0 for real browser-MediaRecorder-authored
   WebM files (which lack a finalized duration header, unlike
   ffmpeg-authored test videos), which
   normalize_video_for_analysis() then correctly-but-unhelpfully
   rejected as "zero duration" -- except the CORRUPTED payload from bug
   #1 was what actually reached process_upload() in the real user's
   test, producing "We couldn't process this audio format" instead.

Both bugs were confirmed against tests/fixtures/browser_recorded_sample.webm,
a REAL (small, ~17KB) file produced by an actual Chromium
getUserMedia/MediaRecorder session (via Playwright, camera/microphone
replaced with a synthetic canvas+WebAudio MediaStream) -- not an
ffmpeg-authored stand-in, since ffmpeg-authored WebM files do not
reproduce the streamed/unfinalized-header characteristic that caused
bug #2.
"""

from __future__ import annotations

import io
import subprocess
import sys
import wave
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import streamlit as st

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "browser_recorded_sample.webm"


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
    state["recording_video_meeting_state"] = None
    state["recording_video_meeting_version"] = 0
    state["video_analytics_result"] = None
    return state


@pytest.fixture(scope="module")
def real_browser_webm() -> Path:
    assert FIXTURE_PATH.exists(), "tests/fixtures/browser_recorded_sample.webm must exist"
    return FIXTURE_PATH


# ---------------------------------------------------------------------------
# 1: JS fix -- static contract check that the marker-based split replaced
# the naive comma split (browser-level behavior confirmed separately by
# tests/test_phase4_1_recorder_playwright.py).
# ---------------------------------------------------------------------------


def test_data_url_extraction_uses_base64_marker_not_naive_comma_split() -> None:
    import re

    html = Path(__file__).resolve().parents[1].joinpath(
        "components", "video_meeting_recorder", "frontend", "index.html"
    ).read_text(encoding="utf-8")
    script = re.search(r"<script>(.*)</script>", html, re.DOTALL).group(1)

    assert 'dataUrl.split(",")[1]' not in script, (
        "naive comma-split truncates the payload whenever mediaRecorder.mimeType "
        "itself contains a comma (e.g. codecs=vp9,opus)"
    )
    assert 'const marker = ";base64,";' in script
    assert "dataUrl.indexOf(marker)" in script


# ---------------------------------------------------------------------------
# 2: Python fix -- real browser-recorded WebM no longer reports zero
# duration, and the full Phase 4 pipeline actually runs on it.
# ---------------------------------------------------------------------------


def test_real_browser_webm_duration_is_not_zero(real_browser_webm: Path) -> None:
    from video_analytics.preprocessing import inspect_video_metadata

    metadata = inspect_video_metadata(real_browser_webm)
    assert metadata.duration_seconds > 0, (
        "A real browser-recorded WebM must not be reported as zero-duration -- "
        "this was the root cause blocking the entire recorded-video pipeline."
    )


def test_real_browser_webm_normalizes_without_raising(real_browser_webm: Path) -> None:
    from video_analytics.preprocessing import normalize_video_for_analysis

    output_path, is_temp = normalize_video_for_analysis(real_browser_webm)
    try:
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    finally:
        if is_temp:
            output_path.unlink(missing_ok=True)


def test_real_browser_webm_reaches_analyze_video_available(real_browser_webm: Path) -> None:
    from video_analytics import analyze_video

    result = analyze_video(real_browser_webm)
    assert result.available is True, f"analyze_video() unavailable: {result.unavailable_reason}"
    assert result.sampled_frame_count > 0
    assert result.metadata is not None
    assert result.metadata.duration_seconds > 0


def test_duration_probe_helper_returns_none_for_unreadable_file(tmp_path: Path) -> None:
    from video_analytics.preprocessing import _probe_duration_seconds_via_ffmpeg

    bogus = tmp_path / "not_a_video.webm"
    bogus.write_bytes(b"not a real video")
    # Must never raise -- a garbage file simply yields no usable duration.
    assert _probe_duration_seconds_via_ffmpeg(bogus) is None


def test_fps_implausibility_guard_rejects_sentinel_values() -> None:
    # Reproduces the exact real-world sentinel observed (fps=1000.0 on a
    # browser-recorded WebM whose header does not declare a real frame
    # rate) without needing a file that happens to trigger it -- OpenCV's
    # reported value for real cameras never legitimately exceeds a couple
    # hundred fps.
    import cv2

    from video_analytics.preprocessing import inspect_video_metadata

    class FakeCapture:
        def isOpened(self):
            return True

        def get(self, prop):
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 640.0
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 360.0
            if prop == cv2.CAP_PROP_FPS:
                return 1000.0  # the exact bogus sentinel observed
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return -9.223372036854776e18  # int64-min sentinel as float
            return 0.0

        def release(self):
            pass

    with patch("video_analytics.preprocessing.cv2.VideoCapture", return_value=FakeCapture()), \
         patch("video_analytics.preprocessing._probe_duration_seconds_via_ffmpeg", return_value=2.5):
        metadata = inspect_video_metadata(Path("irrelevant.webm"))

    assert metadata.fps is None, "an implausible fps sentinel must never be reported as a real value"
    assert metadata.frame_count is None
    assert metadata.duration_seconds == 2.5  # recovered via the ffmpeg fallback


# ---------------------------------------------------------------------------
# 3: recorded webm -> BOTH existing branches, using the REAL fixture.
# ---------------------------------------------------------------------------


def test_real_browser_webm_audio_extraction_is_not_silent(real_browser_webm: Path) -> None:
    from transcription.audio_utils import preprocess_uploaded_audio

    wav_path = preprocess_uploaded_audio(real_browser_webm, filename=real_browser_webm.name)
    try:
        assert wav_path.exists()
        assert wav_path.stat().st_size > 0

        with wave.open(str(wav_path), "rb") as wav_file:
            frame_count = wav_file.getnframes()
            assert frame_count > 0
            raw = wav_file.readframes(frame_count)

        # RMS energy check -- catches "container says audio but the
        # stream is silent" (Part 8), not just "the file is non-empty".
        import struct

        sample_count = len(raw) // 2
        samples = struct.unpack(f"<{sample_count}h", raw[: sample_count * 2])
        rms = (sum(s * s for s in samples) / max(1, len(samples))) ** 0.5
        assert rms > 50, f"extracted WAV audio is near-silent (RMS={rms:.2f})"
    finally:
        wav_path.unlink(missing_ok=True)


def test_real_browser_webm_ffmpeg_reports_both_streams(real_browser_webm: Path) -> None:
    import imageio_ffmpeg

    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    result = subprocess.run(
        [ffmpeg_bin, "-i", str(real_browser_webm)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    stderr = result.stderr
    assert "Video:" in stderr
    assert "Audio:" in stderr


# ---------------------------------------------------------------------------
# 4: full process_upload() flow with the REAL fixture bytes, confirming
# BOTH branches actually populate real results (Part 20, items 14-16).
# ---------------------------------------------------------------------------


class FakeVideoUpload:
    def __init__(self, path: Path):
        self._path = path
        self.name = "meeting-video-real.webm"
        self.size = path.stat().st_size
        self.type = "video/webm;codecs=vp9,opus"

    def getbuffer(self):
        return self._path.read_bytes()

    def read(self):
        return self._path.read_bytes()


def test_real_browser_webm_process_upload_populates_both_results(real_browser_webm: Path) -> None:
    import app.main as m
    from transcription.sarvam_client import TranscriptionResult, TranscriptionSegment

    st.session_state = _fresh_session_state()
    fake_result = TranscriptionResult(
        transcript="Real browser recording handoff test.",
        segments=[TranscriptionSegment(transcript="Real browser recording handoff test.", speaker_id="1", start_time_seconds=0.0, end_time_seconds=1.0)],
    )
    uploaded = FakeVideoUpload(real_browser_webm)

    with patch.object(m, "transcribe_audio_detailed", return_value=fake_result):
        m.process_upload(uploaded)

    assert st.session_state.get("analysis_error", "") == ""
    assert "transcript_result" in st.session_state

    video_result = st.session_state.get("video_analytics_result")
    assert video_result is not None, "analyze_uploaded_video must actually run on the preserved recording"
    assert video_result.available is True, f"video analytics unavailable: {video_result.unavailable_reason}"


# ---------------------------------------------------------------------------
# 5: Process handoff -- exact payload shape the JS component now sends,
# fed through render_record_video_meeting_card() with the REAL fixture's
# base64 (not a synthetic stand-in), verifying start_recorded_video_meeting_workflow
# is reached exactly once with a fully-intact payload.
# ---------------------------------------------------------------------------


def test_process_handoff_with_real_fixture_payload_reaches_workflow_start(real_browser_webm: Path) -> None:
    import base64

    import app.main as m

    st.session_state = _fresh_session_state()
    raw_bytes = real_browser_webm.read_bytes()
    recording = {
        "name": "meeting-video-real.webm",
        "mime_type": "video/webm;codecs=vp9,opus",
        "duration_seconds": 1,
        "size_bytes": len(raw_bytes),
        "data_base64": base64.b64encode(raw_bytes).decode("ascii"),
    }

    class FakeComponent:
        def __call__(self, *args, **kwargs):
            return {"event": "process", "recording": recording}

    with patch.object(m, "VIDEO_MEETING_RECORDER_COMPONENT", FakeComponent()), patch.object(m, "start_recorded_video_meeting_workflow") as mock_start, patch.object(m.st, "button", return_value=False):
        m.render_record_video_meeting_card()

    mock_start.assert_called_once_with(recording)
    called_recording = mock_start.call_args.args[0]
    # The exact bytes must survive the round trip -- this is precisely
    # what bug #1 broke (a truncated data_base64 with real content lost).
    decoded = base64.b64decode(called_recording["data_base64"])
    assert decoded == raw_bytes
