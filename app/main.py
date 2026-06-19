"""Streamlit UI for MeetScribe audio transcription."""

from __future__ import annotations

import html
import hashlib
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import SettingsError
from exports.docx_exporter import export_to_docx, export_transcript_to_docx
from exports.pdf_exporter import export_to_pdf, export_transcript_to_pdf
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

PROCESSING_STAGES = [
    "Processing Audio...",
    "Transcribing with Sarvam...",
    "Analyzing with Gemini...",
    "Generating Minutes of Meeting...",
    "Finalizing Report...",
]


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
    st.session_state.setdefault("analysis_cache", {})
    st.session_state.setdefault("success_metrics", None)
    st.session_state.setdefault("docx_export_path", "")
    st.session_state.setdefault("docx_export_error", "")
    st.session_state.setdefault("pdf_export_path", "")
    st.session_state.setdefault("pdf_export_error", "")
    st.session_state.setdefault("transcript_docx_export_path", "")
    st.session_state.setdefault("transcript_docx_export_error", "")
    st.session_state.setdefault("transcript_pdf_export_path", "")
    st.session_state.setdefault("transcript_pdf_export_error", "")
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


def inject_processing_styles() -> None:
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

          :root {
            --ms-bg: #0B1020;
            --ms-card: rgba(17, 24, 39, 0.86);
            --ms-card-solid: #111827;
            --ms-border: rgba(148, 163, 184, 0.18);
            --ms-accent: #4F46E5;
            --ms-success: #22C55E;
            --ms-text: #F8FAFC;
            --ms-muted: #94A3B8;
            --ms-soft: rgba(79, 70, 229, 0.12);
          }

          html, body, [data-testid="stAppViewContainer"] {
            background:
              radial-gradient(circle at 12% 8%, rgba(79, 70, 229, 0.22), transparent 34%),
              radial-gradient(circle at 88% 0%, rgba(34, 197, 94, 0.10), transparent 26%),
              linear-gradient(180deg, #0B1020 0%, #080C18 100%);
            color: var(--ms-text);
            font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          }

          [data-testid="stHeader"] {
            background: rgba(11, 16, 32, 0);
          }

          .block-container {
            max-width: 1440px;
            padding-top: 2.4rem;
            padding-bottom: 4rem;
            padding-left: clamp(1rem, 3vw, 2.75rem);
            padding-right: clamp(1rem, 3vw, 2.75rem);
          }

          h1, h2, h3, h4, h5, h6, p, label, span, div {
            font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          }

          h1, h2, h3, h4 {
            color: var(--ms-text);
            letter-spacing: 0;
          }

          p, li, label, [data-testid="stMarkdownContainer"] {
            color: #CBD5E1;
          }

          .ms-hero {
            border: 1px solid var(--ms-border);
            border-radius: 24px;
            background:
              linear-gradient(135deg, rgba(17, 24, 39, 0.92), rgba(17, 24, 39, 0.68)),
              linear-gradient(135deg, rgba(79, 70, 229, 0.20), rgba(34, 197, 94, 0.06));
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.28);
            padding: clamp(1.75rem, 4vw, 3.4rem);
            margin-bottom: 1.6rem;
            backdrop-filter: blur(18px);
          }

          .ms-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            border: 1px solid rgba(79, 70, 229, 0.38);
            border-radius: 999px;
            background: rgba(79, 70, 229, 0.12);
            color: #C7D2FE;
            font-size: 0.78rem;
            font-weight: 700;
            padding: 0.35rem 0.72rem;
            margin-bottom: 1rem;
          }

          .ms-hero h1 {
            color: var(--ms-text);
            font-size: clamp(2.45rem, 7vw, 5.2rem);
            font-weight: 800;
            line-height: 0.95;
            margin: 0;
          }

          .ms-hero h2 {
            color: #C7D2FE;
            font-size: clamp(1.15rem, 2.6vw, 1.7rem);
            font-weight: 700;
            margin: 0.9rem 0 0.65rem;
          }

          .ms-hero p {
            color: #CBD5E1;
            font-size: clamp(1rem, 2vw, 1.15rem);
            line-height: 1.65;
            max-width: 760px;
            margin: 0;
          }

          .ms-panel,
          .ms-output-card,
          .ms-export-card,
          div[data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid var(--ms-border) !important;
            border-radius: 20px !important;
            background: var(--ms-card) !important;
            box-shadow: 0 18px 60px rgba(0, 0, 0, 0.22);
            backdrop-filter: blur(16px);
          }

          .ms-panel,
          .ms-output-card,
          .ms-export-card {
            padding: clamp(1.15rem, 2vw, 1.55rem);
            margin: 1.15rem 0;
          }

          .ms-section-title {
            color: var(--ms-text);
            font-size: 1.12rem;
            font-weight: 800;
            margin: 0 0 0.25rem;
          }

          .ms-section-copy {
            color: var(--ms-muted);
            font-size: 0.92rem;
            margin: 0 0 1.15rem;
          }

          .ms-metrics-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.9rem;
            margin: 1.15rem 0;
          }

          .ms-stat-card {
            border: 1px solid var(--ms-border);
            border-radius: 20px;
            background: linear-gradient(180deg, rgba(17, 24, 39, 0.92), rgba(15, 23, 42, 0.72));
            padding: 1.15rem;
            transition: border-color 160ms ease, transform 160ms ease, background 160ms ease;
          }

          .ms-stat-card:hover,
          .ms-item-card:hover,
          .ms-output-card:hover {
            border-color: rgba(129, 140, 248, 0.42);
            transform: translateY(-1px);
          }

          .ms-stat-card span {
            color: var(--ms-muted);
            display: block;
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.45rem;
          }

          .ms-stat-card strong {
            color: var(--ms-text);
            display: block;
            font-size: 1.45rem;
            font-weight: 800;
            line-height: 1.1;
          }

          .ms-stat-card small {
            color: var(--ms-success);
            display: block;
            font-size: 0.78rem;
            margin-top: 0.45rem;
          }

          .ms-card-title {
            color: var(--ms-text);
            font-size: 1.2rem;
            font-weight: 800;
            margin: 0 0 0.55rem;
          }

          .ms-card-label {
            color: #A5B4FC;
            font-size: 0.78rem;
            font-weight: 800;
            margin: 0 0 0.4rem;
            text-transform: uppercase;
          }

          .ms-card-body {
            color: #DDE6F3;
            line-height: 1.65;
            margin: 0;
          }

          .ms-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.75rem;
          }

          .ms-chip {
            border: 1px solid rgba(79, 70, 229, 0.28);
            border-radius: 999px;
            background: rgba(79, 70, 229, 0.14);
            color: #C7D2FE;
            font-size: 0.82rem;
            font-weight: 600;
            padding: 0.35rem 0.65rem;
          }

          .ms-item-card {
            border: 1px solid var(--ms-border);
            border-radius: 20px;
            background: rgba(15, 23, 42, 0.72);
            padding: 1.15rem;
            margin: 0 0 0.95rem;
            transition: border-color 160ms ease, transform 160ms ease, background 160ms ease;
          }

          .ms-item-card h4 {
            color: var(--ms-text);
            font-size: 1rem;
            font-weight: 800;
            margin: 0 0 0.55rem;
          }

          .ms-item-card p {
            color: #DDE6F3;
            line-height: 1.55;
            margin: 0 0 0.7rem;
          }

          .ms-meta-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
          }

          .ms-meta {
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 999px;
            color: #CBD5E1;
            background: rgba(148, 163, 184, 0.08);
            font-size: 0.78rem;
            font-weight: 600;
            padding: 0.28rem 0.55rem;
          }

          .ms-empty {
            border: 1px dashed rgba(148, 163, 184, 0.3);
            border-radius: 16px;
            color: var(--ms-muted);
            padding: 1rem;
            background: rgba(15, 23, 42, 0.45);
          }

          .ms-upload-ready {
            border: 1px solid rgba(34, 197, 94, 0.24);
            border-radius: 14px;
            background: rgba(34, 197, 94, 0.08);
            color: #DCFCE7;
            font-size: 0.9rem;
            font-weight: 700;
            padding: 0.7rem 0.85rem;
            margin: 0.8rem 0 1rem;
          }

          .ms-upload-ready span {
            color: #86EFAC;
            font-weight: 600;
          }

          div[data-testid="stFileUploader"] {
            border: 1px dashed rgba(148, 163, 184, 0.38);
            border-radius: 18px;
            background: rgba(15, 23, 42, 0.62);
            padding: 0.85rem;
          }

          div[data-testid="stFileUploader"] label,
          div[data-testid="stFileUploader"] small {
            color: #CBD5E1 !important;
          }

          .stButton > button,
          .stDownloadButton > button {
            border: 1px solid rgba(79, 70, 229, 0.45);
            border-radius: 14px;
            background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%);
            color: #FFFFFF;
            font-weight: 800;
            min-height: 2.85rem;
            box-shadow: 0 14px 34px rgba(79, 70, 229, 0.25);
            transition: transform 160ms ease, border-color 160ms ease, box-shadow 160ms ease;
          }

          .stButton > button:hover,
          .stDownloadButton > button:hover {
            border-color: rgba(129, 140, 248, 0.8);
            color: #FFFFFF;
            transform: translateY(-1px);
            box-shadow: 0 18px 42px rgba(79, 70, 229, 0.32);
          }

          button[kind="secondary"] {
            background: rgba(15, 23, 42, 0.72) !important;
            color: #F8FAFC !important;
            box-shadow: none !important;
          }

          .stTabs [data-baseweb="tab-list"] {
            gap: 0.45rem;
            border-bottom: 0;
            overflow-x: visible;
            flex-wrap: wrap;
            padding-bottom: 1rem;
            margin-bottom: 1rem;
          }

          .stTabs [data-baseweb="tab"] {
            border: 1px solid var(--ms-border);
            border-radius: 999px;
            background: rgba(15, 23, 42, 0.7);
            color: #CBD5E1;
            font-weight: 800;
            padding: 0.45rem 0.9rem;
            white-space: nowrap;
          }

          .stTabs [aria-selected="true"] {
            background: rgba(79, 70, 229, 0.24);
            border-color: rgba(129, 140, 248, 0.62);
            color: #FFFFFF;
          }

          .stTabs [data-baseweb="tab-panel"] {
            border-top: 1px solid rgba(148, 163, 184, 0.14);
            padding-top: 1.25rem;
          }

          .stTabs [data-baseweb="tab-highlight"] {
            display: none;
          }

          .stTextArea textarea {
            border: 1px solid var(--ms-border);
            border-radius: 16px;
            background: rgba(15, 23, 42, 0.78);
            color: var(--ms-text);
          }

          div[data-testid="stAlert"] {
            border-radius: 16px;
            background: rgba(15, 23, 42, 0.72);
            color: var(--ms-text);
          }

          .ms-processing-card {
            border: 1px solid #d0d5dd;
            border-radius: 20px;
            padding: 1rem;
            background: var(--ms-card);
            margin: 0.75rem 0;
            box-shadow: 0 18px 60px rgba(0, 0, 0, 0.22);
            backdrop-filter: blur(16px);
          }
          .ms-processing-header {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: flex-start;
            border-bottom: 1px solid var(--ms-border);
            padding-bottom: 0.75rem;
            margin-bottom: 0.75rem;
          }
          .ms-processing-header h3 {
            margin: 0.1rem 0 0;
            font-size: 1.05rem;
            line-height: 1.35;
          }
          .ms-eyebrow {
            color: #A5B4FC;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0;
            margin: 0;
            text-transform: uppercase;
          }
          .ms-elapsed {
            color: var(--ms-muted);
            font-size: 0.78rem;
            min-width: 5.5rem;
            text-align: right;
          }
          .ms-elapsed strong {
            color: var(--ms-text);
            font-size: 0.95rem;
          }
          .ms-note {
            color: #CBD5E1;
            margin: 0 0 0.75rem;
            font-size: 0.9rem;
          }
          .ms-stage-list {
            margin: 0;
            padding-left: 1.15rem;
          }
          .ms-stage-list li {
            margin: 0.35rem 0;
            color: var(--ms-text);
          }
          .ms-stage-list span {
            color: var(--ms-muted);
            float: right;
            font-size: 0.85rem;
          }
          @media (max-width: 900px) {
            .ms-metrics-grid {
              grid-template-columns: repeat(2, minmax(0, 1fr));
            }
          }

          @media (max-width: 640px) {
            .block-container {
              padding-left: 1rem;
              padding-right: 1rem;
              padding-top: 1.2rem;
            }
            .ms-hero,
            .ms-panel,
            .ms-output-card,
            .ms-export-card {
              border-radius: 18px;
              padding: 1rem;
            }
            .ms-metrics-grid {
              grid-template-columns: 1fr;
            }
            .ms-processing-header {
              display: block;
            }
            .ms-elapsed {
              margin-top: 0.75rem;
              text-align: left;
            }
            .ms-stage-list span {
              float: none;
              display: block;
              margin-top: 0.15rem;
            }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_gemini_client() -> GeminiClient:
    return GeminiClient()


def analysis_cache_key(transcript_text: str) -> str:
    return hashlib.sha256(transcript_text.strip().encode("utf-8")).hexdigest()


def format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes, remaining_seconds = divmod(round(seconds), 60)
    return f"{minutes}m {remaining_seconds}s"


def update_elapsed(
    placeholder: st.delta_generator.DeltaGenerator | None,
    started_at: float | None,
) -> None:
    if placeholder is None or started_at is None:
        return

    placeholder.caption(f"Elapsed time: {format_elapsed(time.perf_counter() - started_at)}")


def estimated_duration_message(uploaded_file: object) -> str:
    size = getattr(uploaded_file, "size", 0) or 0
    size_mb = size / (1024 * 1024)
    if size_mb <= 10:
        estimate = "about 1-3 minutes"
    elif size_mb <= 50:
        estimate = "about 3-7 minutes"
    else:
        estimate = "7+ minutes"

    return (
        f"Estimated duration: {estimate}. Larger files and long meetings may take "
        "longer while Sarvam transcribes and Gemini prepares the minutes."
    )


def render_stage_status(
    placeholder: st.delta_generator.DeltaGenerator,
    *,
    active_index: int,
    started_at: float,
    note: str = "",
) -> None:
    elapsed = format_elapsed(time.perf_counter() - started_at)
    heading_index = min(active_index, len(PROCESSING_STAGES) - 1)
    rows = []
    for index, stage in enumerate(PROCESSING_STAGES):
        if active_index >= len(PROCESSING_STAGES) or index < active_index:
            status = "Done"
        elif index == active_index:
            status = "In progress"
        else:
            status = "Waiting"
        rows.append(f"<li><strong>{stage}</strong> <span>{status}</span></li>")

    note_html = f"<p class='ms-note'>{html.escape(note)}</p>" if note else ""
    placeholder.markdown(
        f"""
        <div class="ms-processing-card">
          <div class="ms-processing-header">
            <div>
              <p class="ms-eyebrow">MeetScribe is preparing your meeting</p>
              <h3>{PROCESSING_STAGES[heading_index]}</h3>
            </div>
            <div class="ms-elapsed">Elapsed<br><strong>{elapsed}</strong></div>
          </div>
          {note_html}
          <ol class="ms-stage-list">
            {''.join(rows)}
          </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )


def audio_duration_from_result(result: TranscriptionResult) -> str:
    if not result.segments:
        return "N/A"

    starts = [
        segment.start_time_seconds
        for segment in result.segments
        if segment.start_time_seconds is not None
    ]
    ends = [
        segment.end_time_seconds
        for segment in result.segments
        if segment.end_time_seconds is not None
    ]
    if not starts or not ends:
        return "N/A"

    return format_elapsed(max(0, max(ends) - min(starts)))


def speakers_detected(result: TranscriptionResult) -> int:
    if result.segments:
        labels = {speaker_label(segment) for segment in result.segments}
        return len(labels)
    return 1 if result.transcript.strip() else 0


def store_success_metrics(
    *,
    result: TranscriptionResult,
    analysis: MeetingAnalysisResult | None,
    started_at: float,
) -> None:
    st.session_state.success_metrics = {
        "Duration": audio_duration_from_result(result),
        "Speakers": str(speakers_detected(result)),
        "Decisions": str(len(analysis.decisions) if analysis else 0),
        "Action Items": str(len(analysis.action_items) if analysis else 0),
        "Processing Time": format_elapsed(time.perf_counter() - started_at),
    }


def render_success_metrics() -> None:
    metrics = st.session_state.get("success_metrics")
    if not metrics:
        return

    st.markdown(
        f"""
        <div class="ms-metrics-grid">
          <div class="ms-stat-card">
            <span>Speakers Detected</span>
            <strong>{html.escape(metrics.get("Speakers", "N/A"))}</strong>
            <small>Speaker diarization</small>
          </div>
          <div class="ms-stat-card">
            <span>Meeting Duration</span>
            <strong>{html.escape(metrics.get("Duration", "N/A"))}</strong>
            <small>Audio analyzed</small>
          </div>
          <div class="ms-stat-card">
            <span>Decisions Extracted</span>
            <strong>{html.escape(metrics.get("Decisions", "0"))}</strong>
            <small>Ready for review</small>
          </div>
          <div class="ms-stat-card">
            <span>Action Items Generated</span>
            <strong>{html.escape(metrics.get("Action Items", "0"))}</strong>
            <small>Next steps captured</small>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Processed in {metrics.get('Processing Time', 'N/A')}.")


def render_hero() -> None:
    st.markdown(
        """
        <section class="ms-hero">
          <div class="ms-badge">AI meeting intelligence</div>
          <h1>MeetScribe</h1>
          <h2>AI-Powered Minutes of Meeting Generator</h2>
          <p>
            Transform meeting recordings into structured summaries, decisions,
            and action items in minutes.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_panel_header(title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div>
          <p class="ms-section-title">{html.escape(title)}</p>
          <p class="ms-section-copy">{html.escape(copy)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_card(message: str) -> None:
    st.markdown(
        f"<div class='ms-empty'>{html.escape(message)}</div>",
        unsafe_allow_html=True,
    )


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

            st.markdown(
                f"""
                <div class="ms-item-card">
                  <h4>📝 {html.escape(speaker_label(segment))}</h4>
                  <p>{html.escape(segment.transcript)}</p>
                  <div class="ms-meta-row">
                    <span class="ms-meta">{html.escape(start_time)} - {html.escape(end_time)}</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        return

    st.text_area(
        "Transcript",
        value=result.transcript,
        height=260,
        label_visibility="collapsed",
    )


def render_summary_tab(analysis: MeetingAnalysisResult) -> None:
    summary = analysis.summary
    topics_html = ""
    if summary.topics_discussed:
        topics_html = "".join(
            f"<span class='ms-chip'>{html.escape(topic)}</span>"
            for topic in summary.topics_discussed
        )
    else:
        topics_html = "<span class='ms-chip'>No topics extracted</span>"

    st.markdown(
        f"""
        <div class="ms-output-card">
          <p class="ms-card-label">Meeting Title</p>
          <h3 class="ms-card-title">{html.escape(summary.title)}</h3>
          <p class="ms-card-label">Short Summary</p>
          <p class="ms-card-body">{html.escape(summary.short_summary)}</p>
          <p class="ms-card-label" style="margin-top: 1rem;">Topics Discussed</p>
          <div class="ms-chip-row">{topics_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_key_points_tab(analysis: MeetingAnalysisResult) -> None:
    if not analysis.key_discussion_points:
        empty_card("No key discussion points were extracted.")
        return

    for item in analysis.key_discussion_points:
        speakers = ", ".join(item.speakers)
        timestamp = item.timestamp or "--:--"
        st.markdown(
            f"""
            <div class="ms-item-card">
              <h4>💡 Discussion Point</h4>
              <p>{html.escape(item.point)}</p>
              <div class="ms-meta-row">
                <span class="ms-meta">Timestamp: {html.escape(timestamp)}</span>
                <span class="ms-meta">Speaker(s): {html.escape(speakers or "N/A")}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_decisions_tab(analysis: MeetingAnalysisResult) -> None:
    if not analysis.decisions:
        empty_card("No decisions were extracted.")
        return

    for item in analysis.decisions:
        owner = item.owner or "Unassigned"
        timestamp = item.timestamp or "--:--"
        st.markdown(
            f"""
            <div class="ms-item-card">
              <h4>✅ Decision</h4>
              <p>{html.escape(item.decision)}</p>
              <div class="ms-meta-row">
                <span class="ms-meta">Confidence Level: {html.escape(item.confidence)}</span>
                <span class="ms-meta">Owner: {html.escape(owner)}</span>
                <span class="ms-meta">Timestamp: {html.escape(timestamp)}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_action_items_tab(analysis: MeetingAnalysisResult) -> None:
    if not analysis.action_items:
        empty_card("No action items were extracted.")
        return

    for item in analysis.action_items:
        owner = item.owner or "Unassigned"
        due_date = item.due_date or "No due date"
        timestamp = item.timestamp or "--:--"
        st.markdown(
            f"""
            <div class="ms-item-card">
              <h4>📌 Task</h4>
              <p>{html.escape(item.task)}</p>
              <div class="ms-meta-row">
                <span class="ms-meta">Owner: {html.escape(owner)}</span>
                <span class="ms-meta">Due Date: {html.escape(due_date)}</span>
                <span class="ms-meta">Status: {html.escape(item.status)}</span>
                <span class="ms-meta">Timestamp: {html.escape(timestamp)}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_analysis_error() -> None:
    analysis_error = st.session_state.get("analysis_error", "")
    if analysis_error:
        st.warning(analysis_error)


def toast_download_success(message: str) -> None:
    st.toast(message)


def prepared_export_path(
    state_key: str,
    export_factory: Any,
    analysis: MeetingAnalysisResult,
) -> Path:
    export_path_text = st.session_state.get(state_key, "")
    if export_path_text:
        export_path = Path(export_path_text)
        if export_path.is_file():
            return export_path

    export_path = export_factory(
        analysis,
        meeting_info={
            "Source File": st.session_state.get("uploaded_filename", ""),
        },
    )
    st.session_state[state_key] = str(export_path)
    log_stage("Export", "Prepared export document.", path=str(export_path))
    return export_path


def render_download_button(
    *,
    label: str,
    export_path: Path,
    mime: str,
    key: str,
    success_message: str,
) -> None:
    st.download_button(
        label,
        data=export_path.read_bytes(),
        file_name=export_path.name,
        mime=mime,
        key=key,
        use_container_width=True,
        on_click=toast_download_success,
        args=(success_message,),
    )


def render_export_card(analysis: MeetingAnalysisResult) -> None:
    with st.container(border=True):
        render_panel_header(
            "Export Center",
            "Download polished MoM documents or the full transcript report.",
        )
        try:
            mom_pdf_path = prepared_export_path("pdf_export_path", export_to_pdf, analysis)
            mom_docx_path = prepared_export_path("docx_export_path", export_to_docx, analysis)
            transcript_pdf_path = prepared_export_path(
                "transcript_pdf_export_path",
                export_transcript_to_pdf,
                analysis,
            )
            transcript_docx_path = prepared_export_path(
                "transcript_docx_export_path",
                export_transcript_to_docx,
                analysis,
            )
        except Exception as exc:
            log_stage(
                "Export",
                "Could not prepare export documents.",
                error=str(exc),
                traceback=traceback.format_exc(),
            )
            st.error("Downloads could not be prepared. Please try again.")
            return

        top_left, top_right = st.columns(2)
        with top_left:
            render_download_button(
                label="📄 Download MoM (.pdf)",
                export_path=mom_pdf_path,
                mime="application/pdf",
                key="download_mom_pdf",
                success_message="PDF downloaded successfully",
            )
        with top_right:
            render_download_button(
                label="📝 Download MoM (.docx)",
                export_path=mom_docx_path,
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "wordprocessingml.document"
                ),
                key="download_mom_docx",
                success_message="DOCX downloaded successfully",
            )

        bottom_left, bottom_right = st.columns(2)
        with bottom_left:
            render_download_button(
                label="📑 Download Transcript (.pdf)",
                export_path=transcript_pdf_path,
                mime="application/pdf",
                key="download_transcript_pdf",
                success_message="PDF downloaded successfully",
            )
        with bottom_right:
            render_download_button(
                label="📋 Download Transcript (.docx)",
                export_path=transcript_docx_path,
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "wordprocessingml.document"
                ),
                key="download_transcript_docx",
                success_message="DOCX downloaded successfully",
            )


def run_meeting_analysis(
    transcript_text: str,
    *,
    progress: st.delta_generator.DeltaGenerator | None = None,
    elapsed_placeholder: st.delta_generator.DeltaGenerator | None = None,
    status_placeholder: st.delta_generator.DeltaGenerator | None = None,
    started_at: float | None = None,
    estimate_note: str = "",
) -> MeetingAnalysisResult | None:
    analysis_progress = progress or st.progress(0, text="Analyzing with Gemini...")
    cache_key = analysis_cache_key(transcript_text)

    try:
        cached_analysis = st.session_state.analysis_cache.get(cache_key)
        if cached_analysis is not None:
            analysis_progress.progress(75, text="Analyzing with Gemini...")
            if status_placeholder is not None and started_at is not None:
                render_stage_status(
                    status_placeholder,
                    active_index=2,
                    started_at=started_at,
                    note="Preparing the meeting intelligence for this transcript.",
                )
            analysis_progress.progress(90, text="Generating Minutes of Meeting...")
            if status_placeholder is not None and started_at is not None:
                render_stage_status(
                    status_placeholder,
                    active_index=3,
                    started_at=started_at,
                    note="Structuring the transcript analysis into meeting minutes.",
                )
            analysis_progress.progress(100, text="Finalizing Report... complete")
            if status_placeholder is not None and started_at is not None:
                render_stage_status(
                    status_placeholder,
                    active_index=len(PROCESSING_STAGES),
                    started_at=started_at,
                    note="Report metrics and downloads are ready.",
                )
            st.session_state.analysis_result = cached_analysis
            st.session_state.analysis_error = ""
            log_stage("Meeting analysis", "Loaded analysis from session cache.")
            update_elapsed(elapsed_placeholder, started_at)
            return cached_analysis

        log_stage(
            "Meeting analysis",
            "Initializing Gemini analysis pipeline.",
            transcript_chars=len(transcript_text),
        )
        analysis_progress.progress(70, text="Analyzing with Gemini...")
        if status_placeholder is not None and started_at is not None:
            render_stage_status(
                status_placeholder,
                active_index=2,
                started_at=started_at,
                note=estimate_note,
            )
        update_elapsed(elapsed_placeholder, started_at)

        gemini_client = get_gemini_client()
        summarizer = LLMSummarizer(llm_client=gemini_client)

        analysis_progress.progress(
            75,
            text="Analyzing with Gemini...",
        )
        if status_placeholder is not None and started_at is not None:
            render_stage_status(
                status_placeholder,
                active_index=2,
                started_at=started_at,
                note="Gemini is cleaning the transcript and extracting summaries, decisions, and action items.",
            )
        update_elapsed(elapsed_placeholder, started_at)
        log_stage("Meeting analysis", "Calling analyze_meeting() with single Gemini request.")
        analysis = summarizer.analyze_meeting(transcript_text)

        analysis_progress.progress(90, text="Generating Minutes of Meeting...")
        if status_placeholder is not None and started_at is not None:
            render_stage_status(
                status_placeholder,
                active_index=3,
                started_at=started_at,
                note="Structuring the transcript analysis into meeting minutes.",
            )
        update_elapsed(elapsed_placeholder, started_at)
        st.session_state.analysis_result = analysis
        st.session_state.analysis_error = ""
        st.session_state.analysis_cache[cache_key] = analysis
        log_stage(
            "Meeting analysis",
            "Stored analysis in session state.",
            key_points=len(analysis.key_discussion_points),
            decisions=len(analysis.decisions),
            action_items=len(analysis.action_items),
        )

        analysis_progress.progress(96, text="Finalizing Report...")
        if status_placeholder is not None and started_at is not None:
            render_stage_status(
                status_placeholder,
                active_index=4,
                started_at=started_at,
                note="Preparing the transcript, meeting summary, exports, and report metrics.",
            )
        update_elapsed(elapsed_placeholder, started_at)
        time.sleep(0.2)
        analysis_progress.progress(100, text="Finalizing Report... complete")
        if status_placeholder is not None and started_at is not None:
            render_stage_status(
                status_placeholder,
                active_index=len(PROCESSING_STAGES),
                started_at=started_at,
                note="Report metrics and downloads are ready.",
            )
        st.toast("Meeting report is ready")
        return analysis
    except (
        GeminiClientError,
        TranscriptCleanupError,
        MeetingSummaryError,
        KeyDiscussionPointExtractionError,
        DecisionExtractionError,
        ActionItemExtractionError,
    ) as exc:
        if progress is None:
            analysis_progress.empty()
        st.session_state.analysis_result = None
        st.session_state.analysis_error = f"Meeting analysis failed: {exc}"
        log_stage("Meeting analysis", "Analysis failed.", error=str(exc))
        st.error(st.session_state.analysis_error)
        st.exception(exc)
        return None
    except Exception as exc:
        if progress is None:
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
    started_at = time.perf_counter()
    estimate_note = estimated_duration_message(uploaded_file)
    status_placeholder = st.empty()
    render_stage_status(
        status_placeholder,
        active_index=0,
        started_at=started_at,
        note=estimate_note,
    )
    progress = st.progress(0, text="Processing Audio...")
    st.session_state.processing_logs = []
    st.session_state.analysis_result = None
    st.session_state.analysis_error = ""
    st.session_state.success_metrics = None
    st.session_state.docx_export_path = ""
    st.session_state.docx_export_error = ""
    st.session_state.pdf_export_path = ""
    st.session_state.pdf_export_error = ""
    st.session_state.transcript_docx_export_path = ""
    st.session_state.transcript_docx_export_error = ""
    st.session_state.transcript_pdf_export_path = ""
    st.session_state.transcript_pdf_export_error = ""

    try:
        log_stage(
            "File upload",
            "Received uploaded file.",
            filename=getattr(uploaded_file, "name", ""),
            size=getattr(uploaded_file, "size", None),
            type=getattr(uploaded_file, "type", ""),
        )

        progress.progress(10, text="Processing Audio...")
        render_stage_status(
            status_placeholder,
            active_index=0,
            started_at=started_at,
            note="Preparing and converting the uploaded audio into a clean WAV file.",
        )
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

        progress.progress(35, text="Transcribing with Sarvam...")
        render_stage_status(
            status_placeholder,
            active_index=1,
            started_at=started_at,
            note="Sarvam is transcribing the audio and detecting speaker turns.",
        )
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

        progress.progress(60, text="Transcribing with Sarvam...")
        render_stage_status(
            status_placeholder,
            active_index=1,
            started_at=started_at,
            note="Formatting the diarized transcript for meeting analysis.",
        )
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

        progress.progress(65, text="Analyzing with Gemini...")
        render_stage_status(
            status_placeholder,
            active_index=2,
            started_at=started_at,
            note=estimate_note,
        )
        analysis = run_meeting_analysis(
            transcript_text,
            progress=progress,
            status_placeholder=status_placeholder,
            started_at=started_at,
            estimate_note=estimate_note,
        )
        if analysis is not None:
            store_success_metrics(
                result=result,
                analysis=analysis,
                started_at=started_at,
            )
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
        layout="wide",
    )
    initialize_session_state()
    inject_processing_styles()

    render_hero()

    with st.container(border=True):
        render_panel_header(
            "Upload Meeting Audio",
            "Add a WAV, MP3, M4A, AAC, or MP4 recording. MeetScribe will transcribe, analyze, and prepare a structured MoM.",
        )
        uploaded_file = st.file_uploader(
            "Audio file",
            type=SUPPORTED_FILE_TYPES,
            accept_multiple_files=False,
        )

        if uploaded_file is None:
            st.info("Upload a meeting recording to begin.")
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

            st.markdown(
                f"""
                <div class="ms-upload-ready">
                  Selected {html.escape(uploaded_file.name)}
                  <span>({file_size_mb:.2f} MB)</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        process_clicked = st.button(
            "Generate Minutes of Meeting",
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
        st.markdown("<br>", unsafe_allow_html=True)
        render_panel_header(
            "Analysis Workspace",
            "Review the generated summary, discussion points, decisions, action items, and transcript.",
        )
        if st.session_state.uploaded_filename:
            st.caption(f"Source: {st.session_state.uploaded_filename}")

        render_analysis_error()
        render_success_metrics()
        if analysis is not None:
            render_export_card(analysis)

        (
            summary_tab,
            key_points_tab,
            decisions_tab,
            action_items_tab,
            transcript_tab,
        ) = st.tabs(
            [
                "📄 Summary",
                "💡 Discussion Points",
                "✅ Decisions",
                "📌 Action Items",
                "📝 Transcript",
            ]
        )

        with summary_tab:
            if analysis is None:
                empty_card("Meeting analysis is not available yet.")
            else:
                render_summary_tab(analysis)

        with key_points_tab:
            if analysis is None:
                empty_card("Meeting analysis is not available yet.")
            else:
                render_key_points_tab(analysis)

        with decisions_tab:
            if analysis is None:
                empty_card("Meeting analysis is not available yet.")
            else:
                render_decisions_tab(analysis)

        with action_items_tab:
            if analysis is None:
                empty_card("Meeting analysis is not available yet.")
            else:
                render_action_items_tab(analysis)

        with transcript_tab:
            render_copy_button(transcript_text)
            render_transcript(result)

    # Keep logs in session state for debugging, but do not show internal pipeline
    # details in the premium user-facing interface.


if __name__ == "__main__":
    main()
