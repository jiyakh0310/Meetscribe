"""Streamlit UI for MeetScribe audio transcription."""

from __future__ import annotations

import html
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import SettingsError
from llm_clients.gemini_client import GeminiClient, GeminiClientError
from summarization.base_summarizer import MeetingAnalysisResult
from summarization.llm_summarizer import (
    ActionItemExtractionError,
    DecisionExtractionError,
    KeyDiscussionPointExtractionError,
    LLMSummarizer,
    MeetingSummaryError,
    TranscriptCleanupError,
)
from transcription.audio_utils import AudioProcessingError, preprocess_uploaded_audio
from transcription.sarvam_client import (
    TranscriptionError,
    TranscriptionResult,
    TranscriptionSegment,
    transcribe_audio_detailed,
)

SUPPORTED_FILE_TYPES = ("wav", "mp3", "m4a", "aac", "mp4")
logger = logging.getLogger(__name__)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)


def initialize_session_state() -> None:
    st.session_state.setdefault("transcript_text", "")
    st.session_state.setdefault("transcript_result", None)
    st.session_state.setdefault("uploaded_filename", "")
    st.session_state.setdefault("analysis_result", None)
    st.session_state.setdefault("analysis_error", "")
    st.session_state.setdefault("processing_logs", [])
    st.session_state.setdefault("last_logged_upload", "")


def log_stage(stage: str, message: str, **details: Any) -> None:
    detail_text = " ".join(f"{key}={value!r}" for key, value in details.items())
    entry = f"{stage}: {message}"
    if detail_text:
        entry = f"{entry} ({detail_text})"

    logger.info(entry)
    st.session_state.processing_logs.append(entry)


def render_processing_logs() -> None:
    logs = st.session_state.get("processing_logs", [])
    if not logs:
        return

    with st.expander("Processing log", expanded=False):
        for entry in logs:
            st.code(entry, language=None)


def format_timestamp(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"

    total_seconds = max(0, round(seconds))
    minutes, remaining_seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours:
        return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"
    return f"{minutes:02d}:{remaining_seconds:02d}"


def speaker_label(segment: TranscriptionSegment) -> str:
    if segment.speaker_id is None:
        return "Speaker"

    try:
        return f"Speaker {int(segment.speaker_id) + 1}"
    except ValueError:
        return f"Speaker {segment.speaker_id}"


def format_segment(segment: TranscriptionSegment) -> str:
    start_time = format_timestamp(segment.start_time_seconds)
    end_time = format_timestamp(segment.end_time_seconds)
    return f"{speaker_label(segment)} [{start_time} - {end_time}]\n{segment.transcript}"


def format_transcript(result: TranscriptionResult) -> str:
    log_stage(
        "Diarization parsing",
        "Formatting transcription result.",
        has_segments=bool(result.segments),
        segment_count=len(result.segments),
        transcript_chars=len(result.transcript or ""),
    )

    if not result.segments:
        return result.transcript.strip()

    return "\n\n".join(format_segment(segment) for segment in result.segments)


def render_transcript(result: TranscriptionResult) -> None:
    log_stage(
        "Transcript rendering",
        "Rendering transcript.",
        has_segments=bool(result.segments),
        segment_count=len(result.segments),
    )

    if result.segments:
        for segment in result.segments:
            start_time = format_timestamp(segment.start_time_seconds)
            end_time = format_timestamp(segment.end_time_seconds)

            with st.container(border=True):
                st.markdown(f"**{speaker_label(segment)}** `[{start_time} - {end_time}]`")
                st.write(segment.transcript)
        return

    st.text_area(
        "Transcript",
        value=result.transcript,
        height=260,
        label_visibility="collapsed",
    )


def render_summary_tab(analysis: MeetingAnalysisResult) -> None:
    summary = analysis.summary
    st.markdown(f"### {summary.title}")
    st.markdown("**Short Summary**")
    st.write(summary.short_summary)
    st.markdown("**Detailed Summary**")
    st.write(summary.detailed_summary)

    st.markdown("**Topics Discussed**")
    if summary.topics_discussed:
        for topic in summary.topics_discussed:
            st.markdown(f"- {topic}")
    else:
        st.info("No topics were extracted.")


def render_key_points_tab(analysis: MeetingAnalysisResult) -> None:
    if not analysis.key_discussion_points:
        st.info("No key discussion points were extracted.")
        return

    for item in analysis.key_discussion_points:
        speakers = ", ".join(item.speakers)
        timestamp = item.timestamp or "--:--"
        with st.container(border=True):
            st.markdown(f"**[{timestamp}]**")
            st.write(item.point)
            st.caption(f"Speakers: {speakers}")


def render_decisions_tab(analysis: MeetingAnalysisResult) -> None:
    if not analysis.decisions:
        st.info("No decisions were extracted.")
        return

    for item in analysis.decisions:
        timestamp = item.timestamp or "--:--"
        owner = item.owner or "Unassigned"
        with st.container(border=True):
            st.markdown(f"**[{timestamp}] {item.decision}**")
            st.caption(f"Owner: {owner} | Confidence: {item.confidence}")


def render_action_items_tab(analysis: MeetingAnalysisResult) -> None:
    if not analysis.action_items:
        st.info("No action items were extracted.")
        return

    for item in analysis.action_items:
        timestamp = item.timestamp or "--:--"
        owner = item.owner or "Unassigned"
        due_date = item.due_date or "No due date"
        with st.container(border=True):
            st.markdown(f"**[{timestamp}] {item.task}**")
            st.caption(f"Owner: {owner} | Due: {due_date} | Status: {item.status}")


def render_analysis_error() -> None:
    analysis_error = st.session_state.get("analysis_error", "")
    if analysis_error:
        st.warning(analysis_error)


def run_meeting_analysis(transcript_text: str) -> MeetingAnalysisResult | None:
    analysis_progress = st.progress(0, text="Starting meeting analysis...")

    try:
        log_stage(
            "Meeting analysis",
            "Initializing Gemini analysis pipeline.",
            transcript_chars=len(transcript_text),
        )
        analysis_progress.progress(10, text="Connecting to Gemini...")

        gemini_client = GeminiClient()
        summarizer = LLMSummarizer(llm_client=gemini_client)

        analysis_progress.progress(
            25,
            text="Running cleanup, summary, key points, decisions, and action items...",
        )
        log_stage("Meeting analysis", "Calling analyze_meeting().")
        analysis = summarizer.analyze_meeting(transcript_text)

        analysis_progress.progress(85, text="Storing meeting analysis...")
        st.session_state.analysis_result = analysis
        st.session_state.analysis_error = ""
        log_stage(
            "Meeting analysis",
            "Stored analysis in session state.",
            key_points=len(analysis.key_discussion_points),
            decisions=len(analysis.decisions),
            action_items=len(analysis.action_items),
        )

        analysis_progress.progress(100, text="Meeting analysis complete.")
        st.success("Meeting analysis complete.")
        return analysis
    except (
        GeminiClientError,
        TranscriptCleanupError,
        MeetingSummaryError,
        KeyDiscussionPointExtractionError,
        DecisionExtractionError,
        ActionItemExtractionError,
    ) as exc:
        analysis_progress.empty()
        st.session_state.analysis_result = None
        st.session_state.analysis_error = f"Meeting analysis failed: {exc}"
        log_stage("Meeting analysis", "Analysis failed.", error=str(exc))
        st.error(st.session_state.analysis_error)
        st.exception(exc)
        return None
    except Exception as exc:
        analysis_progress.empty()
        st.session_state.analysis_result = None
        st.session_state.analysis_error = (
            f"Unexpected error while analyzing meeting: {exc}"
        )
        log_stage(
            "Meeting analysis",
            "Unexpected analysis error.",
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        st.error(st.session_state.analysis_error)
        st.exception(exc)
        return None


def validate_transcription_result(result: object) -> TranscriptionResult:
    if not isinstance(result, TranscriptionResult):
        raise TypeError(
            "transcribe_audio_detailed() returned "
            f"{type(result).__name__}, expected TranscriptionResult."
        )

    if not isinstance(result.transcript, str):
        raise TypeError(
            "TranscriptionResult.transcript must be a string, got "
            f"{type(result.transcript).__name__}."
        )

    if not isinstance(result.segments, list):
        raise TypeError(
            "TranscriptionResult.segments must be a list, got "
            f"{type(result.segments).__name__}."
        )

    for index, segment in enumerate(result.segments, start=1):
        if not isinstance(segment, TranscriptionSegment):
            raise TypeError(
                "TranscriptionResult.segments contains "
                f"{type(segment).__name__} at position {index}, "
                "expected TranscriptionSegment."
            )

    if not result.transcript.strip() and not result.segments:
        raise ValueError("TranscriptionResult contains no transcript text or segments.")

    return result


def render_copy_button(transcript: str) -> None:
    escaped_transcript = json.dumps(transcript)
    components.html(
        f"""
        <button
            id="copy-transcript"
            style="
                border: 1px solid #d0d5dd;
                border-radius: 6px;
                background: #ffffff;
                color: #101828;
                cursor: pointer;
                font: 14px system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                padding: 0.45rem 0.75rem;
            "
            type="button"
        >
            Copy transcript
        </button>
        <span
            id="copy-status"
            style="
                color: #667085;
                font: 13px system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                margin-left: 0.65rem;
            "
        ></span>
        <script>
            const button = document.getElementById("copy-transcript");
            const status = document.getElementById("copy-status");
            const transcript = {escaped_transcript};

            button.addEventListener("click", async () => {{
                try {{
                    await navigator.clipboard.writeText(transcript);
                    status.textContent = "Copied";
                }} catch (error) {{
                    status.textContent = "Copy failed";
                }}

                setTimeout(() => {{
                    status.textContent = "";
                }}, 2200);
            }});
        </script>
        """,
        height=44,
    )


def process_upload(uploaded_file: object) -> None:
    prepared_path: Path | None = None
    progress = st.progress(0, text="Preparing audio...")
    st.session_state.processing_logs = []
    st.session_state.analysis_result = None
    st.session_state.analysis_error = ""

    try:
        log_stage(
            "File upload",
            "Received uploaded file.",
            filename=getattr(uploaded_file, "name", ""),
            size=getattr(uploaded_file, "size", None),
            type=getattr(uploaded_file, "type", ""),
        )

        progress.progress(15, text="Reading uploaded file...")
        prepared_path = preprocess_uploaded_audio(
            uploaded_file,
            filename=getattr(uploaded_file, "name", None),
        )
        log_stage(
            "WAV conversion",
            "Audio preprocessing completed.",
            path=str(prepared_path),
            size=prepared_path.stat().st_size if prepared_path.exists() else None,
        )

        progress.progress(40, text="Audio converted to WAV. Starting Sarvam...")
        log_stage(
            "Sarvam API call",
            "Calling transcribe_audio_detailed().",
            with_diarization=True,
            path=str(prepared_path),
        )
        result = transcribe_audio_detailed(
            prepared_path,
            with_diarization=True,
        )
        log_stage(
            "Sarvam API call",
            "transcribe_audio_detailed() returned.",
            result_type=type(result).__name__,
        )

        result = validate_transcription_result(result)
        log_stage(
            "Diarization parsing",
            "Validated transcription result.",
            segment_count=len(result.segments),
            transcript_chars=len(result.transcript),
        )

        progress.progress(90, text="Formatting transcript...")
        transcript_text = format_transcript(result)
        if not transcript_text.strip():
            raise ValueError("Formatted transcript is empty.")

        st.session_state.transcript_result = result
        st.session_state.transcript_text = transcript_text
        st.session_state.uploaded_filename = getattr(uploaded_file, "name", "")
        log_stage(
            "Session state update",
            "Stored transcript in session state.",
            transcript_chars=len(transcript_text),
            segment_count=len(result.segments),
        )

        progress.progress(100, text="Transcription complete.")
        st.success("Transcription complete.")
        run_meeting_analysis(transcript_text)
    except (AudioProcessingError, SettingsError, TranscriptionError) as exc:
        progress.empty()
        log_stage("Error", "Pipeline error.", error=str(exc))
        st.error(str(exc))
        st.exception(exc)
    except Exception as exc:
        progress.empty()
        log_stage(
            "Error",
            "Unexpected error while processing audio.",
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        st.error(f"Unexpected error while processing audio: {exc}")
        st.exception(exc)
    finally:
        if prepared_path is not None:
            prepared_path.unlink(missing_ok=True)
            log_stage("Cleanup", "Deleted temporary WAV file.", path=str(prepared_path))


def main() -> None:
    st.set_page_config(
        page_title="MeetScribe",
        layout="centered",
    )
    initialize_session_state()

    st.title("MeetScribe")
    st.caption("Upload meeting audio and generate a speaker-wise transcript.")

    uploaded_file = st.file_uploader(
        "Audio file",
        type=SUPPORTED_FILE_TYPES,
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.info("Upload a WAV, MP3, M4A, AAC, or MP4 file to begin.")
    else:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        upload_signature = f"{uploaded_file.name}:{uploaded_file.size}"
        if st.session_state.last_logged_upload != upload_signature:
            log_stage(
                "File upload",
                "File selected in UI.",
                filename=uploaded_file.name,
                size=uploaded_file.size,
            )
            st.session_state.last_logged_upload = upload_signature

        st.success(
            f"Uploaded `{html.escape(uploaded_file.name)}` "
            f"({file_size_mb:.2f} MB)."
        )

    process_clicked = st.button(
        "Process",
        type="primary",
        disabled=uploaded_file is None,
        use_container_width=True,
    )

    if process_clicked and uploaded_file is not None:
        process_upload(uploaded_file)

    transcript_text = st.session_state.transcript_text
    result = st.session_state.transcript_result
    analysis = st.session_state.analysis_result
    log_stage(
        "Transcript rendering",
        "Checking transcript display conditions.",
        has_transcript=bool(transcript_text),
        result_type=type(result).__name__ if result is not None else None,
        analysis_type=type(analysis).__name__ if analysis is not None else None,
    )

    if transcript_text and result is not None:
        st.divider()
        st.subheader("Meeting Output")

        if st.session_state.uploaded_filename:
            st.caption(f"Source: {st.session_state.uploaded_filename}")

        render_analysis_error()

        (
            transcript_tab,
            summary_tab,
            key_points_tab,
            decisions_tab,
            action_items_tab,
        ) = st.tabs(
            [
                "Transcript",
                "Summary",
                "Key Discussion Points",
                "Decisions",
                "Action Items",
            ]
        )

        with transcript_tab:
            render_copy_button(transcript_text)
            render_transcript(result)

        with summary_tab:
            if analysis is None:
                st.info("Meeting analysis is not available yet.")
            else:
                render_summary_tab(analysis)

        with key_points_tab:
            if analysis is None:
                st.info("Meeting analysis is not available yet.")
            else:
                render_key_points_tab(analysis)

        with decisions_tab:
            if analysis is None:
                st.info("Meeting analysis is not available yet.")
            else:
                render_decisions_tab(analysis)

        with action_items_tab:
            if analysis is None:
                st.info("Meeting analysis is not available yet.")
            else:
                render_action_items_tab(analysis)

    render_processing_logs()


if __name__ == "__main__":
    main()
