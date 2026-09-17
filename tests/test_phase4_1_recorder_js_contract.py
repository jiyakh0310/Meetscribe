"""SELaD Phase 4.1 repair -- static contract checks on the recorder's
client-side JavaScript (Part 20, items 1-13).

These are STATIC SOURCE AUDITS, not behavioral browser tests: real
MediaRecorder/getUserMedia execution cannot happen in this Python test
environment (no browser, no camera/microphone hardware). Each check
below asserts that a specific, textually-verifiable invariant holds in
the shipped source -- e.g. "the constant is 3600", "no setComponentValue
call exists on the Stop path", "duration is captured before the timer
reset that would zero it". This gives real signal about the exact bugs
the repair targeted without pretending to be a browser test.

Actual browser behavior (timer accuracy, audio audibility, mirroring,
finalization race conditions) can only be confirmed by a human in a
real browser -- see the final report's manual retest checklist.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RECORDER_HTML_PATH = Path(__file__).resolve().parents[1] / "components" / "video_meeting_recorder" / "frontend" / "index.html"


def _source() -> str:
    return RECORDER_HTML_PATH.read_text(encoding="utf-8")


def _script() -> str:
    match = re.search(r"<script>(.*)</script>", _source(), re.DOTALL)
    assert match, "recorder index.html must contain a <script> block"
    return match.group(1)


# ---------------------------------------------------------------------------
# 1: max duration = 3600 seconds
# ---------------------------------------------------------------------------


def test_max_duration_constant_is_3600_seconds() -> None:
    script = _script()
    assert re.search(r"MAX_SECONDS\s*=\s*60\s*\*\s*60\s*;", script), (
        "MAX_SECONDS must be defined as 60*60 (3600s / 60 minutes) per the repair's Part 8 requirement."
    )
    assert "15 * 60" not in script
    assert "Maximum recording duration: 60 minutes." in _source()


# ---------------------------------------------------------------------------
# 2-3: manual early stop preserves actual elapsed duration; never == max
# ---------------------------------------------------------------------------


def test_duration_is_captured_before_any_state_reset() -> None:
    script = _script()
    stop_fn = re.search(r"function stopRecording\(.*?\n    \}", script, re.DOTALL).group(0)
    # The authoritative duration must be computed as the FIRST statement of
    # stopRecording -- before clearInterval/clearTimeout/state transition --
    # so nothing can zero the underlying timer state first (root cause of
    # the observed 15:00-after-early-stop bug: a stray full UI reset was
    # able to run between Stop and duration capture).
    capture_index = stop_fn.find("actualRecordedDurationSeconds = Math.max(1, Math.round(currentLiveDuration()))")
    clear_interval_index = stop_fn.find("clearInterval(timerId)")
    assert capture_index != -1, "stopRecording must capture actualRecordedDurationSeconds"
    assert clear_interval_index != -1
    assert capture_index < clear_interval_index, (
        "Duration must be captured BEFORE timers are cleared/state is reset, not after."
    )


def test_completion_duration_never_hardcodes_max_seconds() -> None:
    script = _script()
    # The completion screen must read the captured per-recording duration
    # (recordedPayload.duration_seconds), never MAX_SECONDS.
    render_fn = re.search(r"function render\(\).*?\n    \}", script, re.DOTALL).group(0)
    assert "MAX_SECONDS" not in render_fn
    assert "recordedPayload.duration_seconds" in render_fn


def test_recorder_state_no_longer_synced_from_python_mid_recording() -> None:
    # Root cause of the broken-duration / missing-completion-controls bugs:
    # applyState(args.state) used to run on every incoming streamlit:render
    # (which fires on ANY unrelated Streamlit rerun), unconditionally
    # resetting startedAt/mediaRecorder/timers even mid-recording. The
    # repaired component must never call any state-applying function from
    # the message listener -- only postReady/postHeight.
    script = _script()
    listener = re.search(r'window\.addEventListener\("message".*?\}\);', script, re.DOTALL).group(0)
    assert "applyState" not in listener
    assert "recorderState =" not in listener
    assert "postReady()" in listener
    assert "postHeight()" in listener


# ---------------------------------------------------------------------------
# 4, 12: finalization waits for recorder stop; last chunks not lost
# ---------------------------------------------------------------------------


def test_finalize_runs_only_via_onstop_callback() -> None:
    script = _script()
    assert 'mediaRecorder.onstop = finalizeRecording;' in script
    # finalizeRecording must not be invoked eagerly elsewhere on the Stop path.
    stop_fn = re.search(r"function stopRecording\(.*?\n    \}", script, re.DOTALL).group(0)
    assert "finalizeRecording()" not in stop_fn.replace("catch (error) {\n        finalizeRecording();\n      }", "")


def test_tracks_stopped_after_recorder_stop_not_before() -> None:
    script = _script()
    stop_fn = re.search(r"function stopRecording\(.*?\n    \}", script, re.DOTALL).group(0)
    finalize_fn = re.search(r"function finalizeRecording\(\).*?\n    \}", script, re.DOTALL).group(0)
    # stopTracks() must not be called inside stopRecording() itself -- only
    # after finalization has produced the Blob (Part 12: "MediaStream
    # tracks can be stopped after MediaRecorder has finalized").
    assert "stopTracks()" not in stop_fn
    assert "stopTracks()" in finalize_fn


# ---------------------------------------------------------------------------
# 5, 8: actual recorder MIME propagated into Blob/payload
# ---------------------------------------------------------------------------


def test_actual_mime_type_used_for_blob_not_requested_candidate() -> None:
    script = _script()
    finalize_fn = re.search(r"function finalizeRecording\(\).*?\n    \}", script, re.DOTALL).group(0)
    assert 'mediaRecorder && mediaRecorder.mimeType' in finalize_fn
    assert "new Blob(chunks, { type: actualMimeType }" in finalize_fn
    assert "mime_type: actualMimeType" in finalize_fn


# ---------------------------------------------------------------------------
# 6: empty Blob rejected
# ---------------------------------------------------------------------------


def test_empty_blob_is_rejected_with_error() -> None:
    script = _script()
    finalize_fn = re.search(r"function finalizeRecording\(\).*?\n    \}", script, re.DOTALL).group(0)
    assert "if (!blob.size)" in finalize_fn
    assert 'recorderState = "ERROR"' in finalize_fn


# ---------------------------------------------------------------------------
# 7: audio-track requirement represented in the recorder contract
# ---------------------------------------------------------------------------


def test_audio_track_presence_is_validated_before_recording_starts() -> None:
    script = _script()
    start_fn = re.search(r"async function startRecording\(\).*?\n    \}", script, re.DOTALL).group(0)
    assert "mediaStream.getAudioTracks().length === 0" in start_fn
    assert "mediaStream.getVideoTracks().length === 0" in start_fn
    assert "Microphone audio was not captured" in start_fn


def test_recorded_playback_is_never_forcibly_muted() -> None:
    script = _script()
    finalize_fn = re.search(r"function finalizeRecording\(\).*?\n    \}", script, re.DOTALL).group(0)
    assert "playbackVideo.muted = false;" in finalize_fn
    assert "playbackVideo.defaultMuted = false;" in finalize_fn
    # The static markup must not hardcode a muted attribute on playback.
    playback_tag = re.search(r'<video id="playbackVideo"[^>]*>', _source()).group(0)
    assert "muted" not in playback_tag
    live_tag = re.search(r'<video id="liveVideo"[^>]*>', _source()).group(0)
    assert "muted" in live_tag  # live preview SHOULD stay muted to avoid feedback


# ---------------------------------------------------------------------------
# 9-10: malformed payload safety; no payload submission merely on Stop
# ---------------------------------------------------------------------------


def test_no_component_value_sent_on_stop_or_finalize() -> None:
    script = _script()
    stop_fn = re.search(r"function stopRecording\(.*?\n    \}", script, re.DOTALL).group(0)
    finalize_fn = re.search(r"function finalizeRecording\(\).*?\n    \}", script, re.DOTALL).group(0)
    assert "setComponentValue" not in stop_fn
    assert "setComponentValue" not in finalize_fn


def test_only_process_event_sends_a_component_value() -> None:
    script = _script()
    set_component_value_calls = re.findall(r'event:\s*"(\w+)"', script)
    assert set_component_value_calls == ["process"], (
        f"Only a 'process' event should ever be posted to Python; found {set_component_value_calls}"
    )


# ---------------------------------------------------------------------------
# 11-12: Process action sends payload; double Process prevented
# ---------------------------------------------------------------------------


def test_process_meeting_sends_payload_and_guards_double_submit() -> None:
    script = _script()
    process_fn = re.search(r"function processMeeting\(\).*?\n    \}", script, re.DOTALL).group(0)
    assert "submitted" in process_fn
    assert "setComponentValue" in process_fn
    assert 'recorderState !== "RECORDED"' in process_fn or "!recordedPayload" in process_fn


def test_process_button_disabled_until_base64_payload_ready() -> None:
    script = _script()
    render_fn = re.search(r"function render\(\).*?\n    \}", script, re.DOTALL).group(0)
    assert "processBtn.disabled = !isRecorded || !recordedPayload || !recordedPayload.data_base64" in render_fn


# ---------------------------------------------------------------------------
# 13: Record Again clears old recording
# ---------------------------------------------------------------------------


def test_record_again_revokes_object_url_and_resets_to_ready() -> None:
    script = _script()
    go_ready_fn = re.search(r"function goReady\(\).*?\n    \}", script, re.DOTALL).group(0)
    assert "revokeRecordedPayload()" in go_ready_fn
    assert 'recorderState = "READY"' in go_ready_fn
    revoke_fn = re.search(r"function revokeRecordedPayload\(\).*?\n    \}", script, re.DOTALL).group(0)
    assert "URL.revokeObjectURL" in revoke_fn


# ---------------------------------------------------------------------------
# Layout / bitrate sanity (Part 6, 17 -- not strictly in the item list, but
# directly targeted by this repair and cheaply verifiable statically).
# ---------------------------------------------------------------------------


def test_camera_preview_uses_large_responsive_frame() -> None:
    source = _source()
    assert "aspect-ratio:16/9" in source or "aspect-ratio: 16/9" in source
    assert "width:100%" in source


def test_recorder_uses_meeting_quality_bitrate_not_unbounded() -> None:
    script = _script()
    assert "videoBitsPerSecond" in script
    assert "audioBitsPerSecond" in script


def test_large_recording_warning_threshold_exists() -> None:
    script = _script()
    assert "LARGE_RECORDING_WARN_BYTES" in script


# ---------------------------------------------------------------------------
# Fix #2 (invisible Stop button): robust iframe height reporting.
#
# Root cause: the iframe's reported height lagged behind the larger
# RECORDING-state layout (video-frame growing from ~340px max to ~620px
# max). Streamlit's component iframe has no scrollbar of its own, so
# anything past the last-reported height was silently clipped -- not
# merely scrolled out of view. These checks confirm a ResizeObserver
# independently re-reports height on any layout change, not only from
# the manual postHeight() calls scattered through render()/setError().
# ---------------------------------------------------------------------------


def test_resize_observer_reports_height_independently_of_render_calls() -> None:
    script = _script()
    assert "new ResizeObserver(() => postHeight())" in script
    assert "resizeObserver.observe(document.body)" in script


def test_window_resize_also_triggers_height_report() -> None:
    script = _script()
    assert 'window.addEventListener("resize", () => postHeight());' in script


def test_video_frame_max_height_leaves_room_for_controls() -> None:
    # Reduced from the earlier min(70vh,640px) to min(60vh,620px) (Part 5)
    # specifically to leave more guaranteed headroom for the action row
    # below the camera, on top of the ResizeObserver fix.
    source = _source()
    assert "max-height:min(60vh,620px)" in source
    assert "max-height:min(70vh,640px)" not in source


def test_stop_button_gets_prominent_styling_while_recording() -> None:
    source = _source()
    assert ".recording #stopBtn{" in source


# ---------------------------------------------------------------------------
# Fix #2 (selfie mirror): live preview mirrored, playback/encoding untouched.
# ---------------------------------------------------------------------------


def test_live_video_can_receive_mirrored_class_playback_never_does() -> None:
    script = _script()
    assert 'liveVideo.classList.toggle("mirrored"' in script
    assert "playbackVideo.classList" not in script  # playback is never mirrored


def test_mirror_is_a_pure_css_transform_scoped_to_live_video_only() -> None:
    source = _source()
    assert "#liveVideo.mirrored{transform:scaleX(-1)" in source
    # The mirror rule must be scoped to #liveVideo specifically, never to
    # the whole .video-frame container (which would also flip badges/timer/
    # buttons) and never to #playbackVideo.
    assert ".video-frame.mirrored" not in source
    assert "#playbackVideo.mirrored" not in source


def test_facing_mode_user_requested_for_default_selfie_camera() -> None:
    script = _script()
    start_fn = re.search(r"async function startRecording\(\).*?\n    \}", script, re.DOTALL).group(0)
    assert 'facingMode: "user"' in start_fn


def test_environment_facing_camera_is_not_mirrored() -> None:
    script = _script()
    start_fn = re.search(r"async function startRecording\(\).*?\n    \}", script, re.DOTALL).group(0)
    assert 'trackSettings.facingMode !== "environment"' in start_fn


def test_record_again_clears_stale_mirror_state() -> None:
    script = _script()
    go_ready_fn = re.search(r"function goReady\(\).*?\n    \}", script, re.DOTALL).group(0)
    assert 'liveVideo.classList.remove("mirrored")' in go_ready_fn


# ---------------------------------------------------------------------------
# Part 13: explicit button-visibility contract per state, read directly out
# of render()'s assignment expressions (each keyed only off recorderState).
# ---------------------------------------------------------------------------


def test_button_visibility_contract_matches_state_machine() -> None:
    script = _script()
    render_fn = re.search(r"function render\(\).*?\n    \}", script, re.DOTALL).group(0)

    # READY: only Start is tied to isReady.
    assert 'startBtn.style.display = isReady ? "inline-flex" : "none";' in render_fn
    # RECORDING: only Stop is tied to isRecording.
    assert 'stopBtn.style.display = isRecording ? "inline-flex" : "none";' in render_fn
    # RECORDED/SUBMITTED (showPlayback): Record Again + Process Meeting.
    assert 'clearBtn.style.display = showPlayback ? "inline-flex" : "none";' in render_fn
    assert 'processBtn.style.display = showPlayback ? "inline-flex" : "none";' in render_fn
    assert "const showPlayback = isRecorded || isSubmitted;" in render_fn
    # SUBMITTED: Record Again disabled, Process Meeting shows a busy label.
    assert "clearBtn.disabled = isSubmitted;" in render_fn
    assert 'processBtn.textContent = isSubmitted ? "Processing…" : "Process Meeting →";' in render_fn


def test_ready_banner_and_playback_precede_action_row_in_dom_order() -> None:
    # DOM order matters for the requested completion hierarchy ("Recording
    # ready" -> playback -> action buttons): readyBanner must appear before
    # videoFrame, which must appear before the actions row.
    source = _source()
    ready_banner_index = source.index('id="readyBanner"')
    video_frame_index = source.index('id="videoFrame"')
    actions_index = source.index('<div class="actions">')
    assert ready_banner_index < video_frame_index < actions_index
