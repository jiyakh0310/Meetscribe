"""Streamlit UI for MeetScribe audio transcription — refined UI."""

from __future__ import annotations

import html
import hashlib
import json
import logging
import re
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
from agentic.workflow import run_meeting_analysis_workflow
from exports.docx_exporter import export_to_docx
from exports.email_sender import (
    EmailDeliveryError,
    EmailValidationError,
    SMTPConfigurationError,
    send_report_email,
)
from exports.pdf_exporter import export_to_pdf
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
from transcription.speaker_mapping import (
    SpeakerMapping,
    display_speaker_label,
)
from transcription.speaker_resolution import (
    apply_speaker_resolution,
    detect_speaker_labels,
    resolve_speakers,
    update_mapping,
)
from transcription.transcript_editing import (
    apply_transcript_edits,
    speaker_mapping_from_segments,
)
from transcription.transcript_file_utils import (
    TranscriptFileError,
    extract_uploaded_transcript,
)
from transcription.sarvam_client import (
    TranscriptionError,
    TranscriptionResult,
    TranscriptionSegment,
    transcribe_audio_detailed,
)

SUPPORTED_FILE_TYPES = ("wav", "mp3", "m4a", "aac", "mp4")
SUPPORTED_TRANSCRIPT_TYPES = ("pdf", "docx", "txt")
logger = logging.getLogger(__name__)
MEETING_INFO_FIELDS = (
    ("meeting_title", "Meeting Title"),
    ("meeting_date", "Meeting Date"),
    ("meeting_time", "Meeting Time"),
    ("organization", "Organization / Company"),
    ("project_name", "Project Name"),
    ("prepared_by", "Prepared By"),
    ("participants", "Participants"),
)

PROCESSING_STAGES = [
    "Step 1: Uploading Recording",
    "Step 2: Preparing Audio",
    "Step 3: Identifying Speakers",
    "Step 4: Generating Meeting Notes",
    "Step 5: Preparing Exports",
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
    st.session_state.setdefault("meeting_metadata", {})
    st.session_state.setdefault("meeting_info", {})
    st.session_state.setdefault("meeting_info_initialized", False)
    st.session_state.setdefault("meeting_info_last_saved", {})
    st.session_state.setdefault("show_email_form", False)
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
    st.session_state.setdefault("audio_upload_version", 0)
    st.session_state.setdefault("transcript_upload_version", 0)
    st.session_state.setdefault("speaker_mapping", {})
    st.session_state.setdefault("saved_speaker_mapping", {})
    st.session_state.setdefault("speaker_names_available", False)
    st.session_state.setdefault("speaker_review_required", False)
    st.session_state.setdefault("transcript_review_required", False)
    st.session_state.setdefault("edited_transcript_text", "")


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
          @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

          /* ══ DESIGN TOKENS ══════════════════════════ */
          :root {
            --pink:        #FB7185;
            --pink-soft:   #FDE7EF;
            --pink-mid:    #FBBECB;
            --pink-deep:   #E11D48;
            --lav:         #A78BFA;
            --lav-soft:    #EDE9FE;
            --lav-mid:     #C4B5FD;
            --warm:        #1F2937;
            --warm-2:      #374151;
            --warm-3:      #6B7280;
            --warm-4:      #9CA3AF;
            --warm-5:      #F5F5F4;
            --warm-6:      #FAFAF9;
            --surface:     #FFFFFF;
            --border:      #ECECEC;
            --border-soft: #ECECEC;
            --green:       #10B981;
            --green-soft:  #D1FAE5;
            --amber:       #F59E0B;
            --amber-soft:  #FEF3C7;
            --blue:        #3B82F6;
            --blue-soft:   #DBEAFE;
            --red:         #EF4444;
            --r:           10px;
            --r-lg:        14px;
            --r-xl:        18px;
            --shadow-sm:   0 1px 3px rgba(0,0,0,0.06),0 1px 2px rgba(0,0,0,0.04);
            --shadow-md:   0 4px 12px rgba(0,0,0,0.08),0 2px 4px rgba(0,0,0,0.04);
            --shadow-lg:   0 8px 24px rgba(0,0,0,0.10),0 4px 8px rgba(0,0,0,0.06);
          }

          /* ══ RESET & BASE ═══════════════════════════ */
          *,*::before,*::after { box-sizing: border-box; margin: 0; padding: 0; }

          html, body,
          [data-testid="stAppViewContainer"],
          [data-testid="stApp"] {
            background: var(--warm-6) !important;
            color: var(--warm) !important;
            font-family: Inter, system-ui, -apple-system, sans-serif !important;
            -webkit-font-smoothing: antialiased;
            overflow-x: hidden !important;
          }

          [data-testid="stHeader"]     { background: transparent !important; }
          [data-testid="stDecoration"] { display: none !important; }
          [data-testid="stToolbar"]    { display: none !important; }

          [data-testid="stSidebar"] {
            background: var(--surface) !important;
            border-right: 1px solid var(--border-soft) !important;
            box-shadow: 8px 0 24px rgba(28,25,23,0.03) !important;
          }
          [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0.35rem !important;
          }
          [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            margin: 0 !important;
          }
          [data-testid="stSidebar"] > div:first-child {
            padding-top: 0.35rem !important;
          }
          .ms-side-shell {
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
            padding: 0 0.15rem 0.75rem;
            transform: translateY(-0.15rem);
          }
          .ms-side-brand {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            padding: 0.45rem 0.35rem 0.9rem;
            border-bottom: 1px solid var(--border-soft);
            margin-bottom: 0.25rem;
          }
          .ms-side-logo-mark {
            width: 30px;
            height: 30px;
            border-radius: 9px;
            background: var(--pink-soft);
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--pink);
            font-weight: 800;
            font-size: 0.68rem;
          }
          .ms-side-title {
            color: var(--warm) !important;
            font-weight: 800;
            font-size: 0.95rem;
            letter-spacing: -0.02em;
          }
          .ms-side-nav {
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
          }
          .ms-side-item {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            min-height: 38px;
            padding: 0.55rem 0.75rem;
            border-radius: 10px;
            color: var(--warm-2) !important;
            font-size: 0.84rem;
            font-weight: 600;
            transition: background 160ms ease, color 160ms ease, transform 160ms ease;
          }
          .ms-side-item:hover {
            background: var(--warm-5);
            transform: translateX(2px);
          }
          .ms-side-item.active {
            background: var(--pink-soft);
            color: var(--pink-deep) !important;
          }
          .ms-side-icon {
            width: 18px;
            color: inherit;
            text-align: center;
            font-size: 0.64rem;
            font-weight: 800;
          }
          .block-container {
            max-width: 1180px !important;
            margin: 0 auto !important;
            padding-top: 1.1rem !important;
            padding-bottom: 6rem !important;
            padding-left:  clamp(1rem, 3vw, 2rem) !important;
            padding-right: clamp(1rem, 3vw, 2rem) !important;
          }

          h1,h2,h3,h4,h5,h6 { color: var(--warm) !important; letter-spacing: -0.02em; }
          p, li, label,
          [data-testid="stMarkdownContainer"] { color: var(--warm-3) !important; }

          /* ══ HERO ═════════════════════════════════════ */
          @keyframes fadein { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:none} }

          .ms-hero {
            text-align: left;
            padding: 0.25rem 0 1.85rem;
            animation: fadein 0.35s ease both;
            display: flex; flex-direction: column; align-items: flex-start;
          }

          .ms-hero-brand {
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            margin: 0 0 0.85rem;
            color: var(--warm) !important;
            font-size: 0.82rem;
            font-weight: 800;
            letter-spacing: -0.01em;
          }
          .ms-brand-recording {
            width: 32px;
            height: 32px;
            border-radius: 11px;
            background: var(--pink-soft);
            border: 1px solid #FAD4DC;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 2px;
            box-shadow: 0 4px 16px rgba(251,113,133,0.08);
          }
          .ms-brand-recording span {
            display: block;
            width: 3px;
            border-radius: 999px;
            background: var(--pink);
          }
          .ms-brand-recording span:nth-child(1),
          .ms-brand-recording span:nth-child(5) { height: 9px; opacity: 0.65; }
          .ms-brand-recording span:nth-child(2),
          .ms-brand-recording span:nth-child(4) { height: 15px; opacity: 0.82; }
          .ms-brand-recording span:nth-child(3) { height: 20px; }

          .ms-hero h1 {
            font-size: clamp(1.72rem, 3.7vw, 2.55rem) !important;
            font-weight: 800 !important;
            letter-spacing: -0.035em !important;
            line-height: 1.15 !important;
            color: var(--warm) !important;
            margin-bottom: 0.65rem;
            max-width: 720px;
          }

          .ms-hero-sub {
            color: var(--warm-3) !important;
            font-size: 0.96rem; line-height: 1.68;
            max-width: 680px; margin: 0 0 1.35rem;
          }

          .ms-workflow-timeline {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.55rem;
            padding: 0;
            border: 0;
            border-radius: 0;
            background: transparent;
            box-shadow: none;
            max-width: 100%;
          }
          .ms-workflow-step {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: var(--warm-3) !important;
            font-size: 0.78rem;
            font-weight: 700;
            padding: 0.48rem 0.78rem;
            border-radius: 999px;
            background: #FFF1F5;
            border: 1px solid #FAD4DC;
            box-shadow: 0 4px 16px rgba(251,113,133,0.04);
          }
          .ms-workflow-step.active {
            background: var(--pink);
            border-color: var(--pink);
            color: #FFFFFF !important;
            box-shadow: 0 8px 22px rgba(251,113,133,0.18);
          }
          .ms-workflow-dot {
            width: 6px;
            height: 6px;
            border-radius: 999px;
            background: var(--pink);
            opacity: 0.85;
            flex: 0 0 auto;
          }

          /* ══ PANELS ══════════════════════════════════ */
          div[data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid #FAD4DC !important;
            border-radius: var(--r-xl) !important;
            background: var(--surface) !important;
            box-shadow: 0 4px 16px rgba(251,113,133,0.06) !important;
            backdrop-filter: none !important;
            transition: border-color 200ms ease, box-shadow 200ms ease, transform 200ms ease;
          }
          div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            border-color: var(--pink) !important;
            box-shadow: 0 10px 26px rgba(251,113,133,0.10) !important;
          }

          /* ══ UPLOAD PANEL ═════════════════════════════ */
          .ms-upload-card {
            border: 1px solid var(--border-soft);
            border-radius: var(--r-xl);
            background: var(--surface);
            padding: 1.5rem 1.5rem 1.25rem;
            margin-bottom: 1rem;
            box-shadow: var(--shadow-sm);
            animation: fadein 0.35s ease both;
            transition: border-color 200ms, box-shadow 200ms;
          }
          .ms-upload-card:hover {
            border-color: var(--pink-mid);
            box-shadow: var(--shadow-md);
          }

          .ms-dashboard-grid {
            display: grid;
            grid-template-columns: 1.15fr 0.9fr 1fr;
            gap: 1rem;
            margin: 1rem 0 1.2rem;
          }
          .ms-dashboard-card {
            border: 1px solid var(--border-soft);
            border-radius: var(--r-xl);
            background: var(--surface);
            box-shadow: var(--shadow-sm);
            padding: 1rem;
            min-width: 0;
          }
          .ms-dashboard-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            margin-bottom: 0.8rem;
          }
          .ms-dashboard-title {
            color: var(--warm) !important;
            font-size: 0.82rem;
            font-weight: 800;
          }
          .ms-dashboard-muted {
            color: var(--warm-4) !important;
            font-size: 0.7rem;
            font-weight: 600;
          }
          .ms-activity-row {
            display: grid;
            grid-template-columns: 28px minmax(0,1fr) auto;
            gap: 0.65rem;
            align-items: center;
            padding: 0.65rem 0;
            border-top: 1px solid var(--border-soft);
          }
          .ms-activity-row:first-of-type { border-top: none; }
          .ms-activity-icon {
            width: 28px;
            height: 28px;
            border-radius: 8px;
            background: var(--pink-soft);
            color: var(--pink-deep) !important;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.76rem;
          }
          .ms-activity-name {
            color: var(--warm) !important;
            font-size: 0.78rem;
            font-weight: 700;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
          }
          .ms-activity-meta {
            color: var(--warm-4) !important;
            font-size: 0.68rem;
          }
          .ms-badge {
            border-radius: 999px;
            padding: 0.2rem 0.5rem;
            font-size: 0.65rem;
            font-weight: 800;
            white-space: nowrap;
          }
          .ms-badge.ready { background: var(--green-soft); color: #047857 !important; }
          .ms-badge.queue { background: var(--amber-soft); color: #92400E !important; }
          .ms-badge.idle { background: var(--warm-5); color: var(--warm-3) !important; }
          .ms-quick-stats {
            display: grid;
            grid-template-columns: repeat(4,minmax(0,1fr));
            gap: 0.6rem;
          }
          .ms-quick-stat {
            border: 1px solid var(--border-soft);
            border-radius: var(--r-lg);
            background: var(--warm-6);
            padding: 0.65rem 0.45rem;
            text-align: center;
            min-height: 72px;
          }
          .ms-quick-value {
            display: block;
            color: var(--warm) !important;
            font-size: 0.95rem;
            font-weight: 850;
            letter-spacing: -0.03em;
            line-height: 1.15;
          }
          .ms-quick-label {
            display: block;
            color: var(--warm-4) !important;
            font-size: 0.62rem;
            font-weight: 700;
            margin-top: 0.28rem;
            line-height: 1.25;
          }
          .ms-queue-line {
            display: grid;
            grid-template-columns: minmax(0,1fr) auto;
            gap: 0.75rem;
            align-items: center;
            padding: 0.5rem 0;
            border-top: 1px solid var(--border-soft);
          }
          .ms-queue-line:first-of-type { border-top: none; }
          .ms-queue-title {
            color: var(--warm) !important;
            font-size: 0.76rem;
            font-weight: 700;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
          }
          .ms-queue-sub {
            color: var(--warm-4) !important;
            font-size: 0.66rem;
            margin-top: 0.15rem;
          }

          .ms-premium-section {
            border: 1px solid var(--border-soft);
            border-radius: var(--r-xl);
            background: var(--surface);
            box-shadow: var(--shadow-sm);
            padding: 1rem;
            margin: 0.9rem 0 1rem;
          }
          .ms-section-kicker {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            color: var(--pink-deep) !important;
            font-size: 0.72rem;
            font-weight: 850;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            margin-bottom: 0.3rem;
          }
          .ms-section-heading {
            color: var(--warm) !important;
            font-size: 1.05rem;
            font-weight: 850;
            letter-spacing: -0.025em;
            margin: 0 0 0.2rem;
          }
          .ms-section-subcopy {
            color: var(--warm-3) !important;
            font-size: 0.82rem;
            line-height: 1.55;
            margin: 0 0 0.95rem;
          }
          .ms-speaker-row {
            display: grid;
            grid-template-columns: 38px minmax(0,1fr);
            gap: 0.75rem;
            align-items: center;
            padding: 0.65rem 0.7rem;
            border: 1px solid var(--border-soft);
            border-radius: var(--r-lg);
            background: var(--warm-6);
            margin-bottom: 0.55rem;
          }
          .ms-speaker-avatar {
            width: 38px;
            height: 38px;
            border-radius: 12px;
            background: var(--lav-soft);
            color: #6D28D9 !important;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 850;
            font-size: 0.78rem;
          }
          .ms-speaker-label {
            color: var(--warm) !important;
            font-weight: 800;
            font-size: 0.84rem;
            line-height: 1.2;
          }
          .ms-speaker-badge-text {
            color: var(--warm-4) !important;
            font-size: 0.68rem;
            margin-top: 0.12rem;
          }
          .ms-transcript-toolbar {
            position: sticky;
            top: 0;
            z-index: 4;
            border: 1px solid var(--border-soft);
            border-radius: var(--r-lg);
            background: rgba(255,255,255,0.96);
            padding: 0.75rem;
            margin-bottom: 0.8rem;
            box-shadow: var(--shadow-sm);
          }
          .ms-conversation-preview {
            display: flex;
            flex-direction: column;
            gap: 0.65rem;
            max-height: 300px;
            overflow-y: auto;
            padding: 0.15rem 0.1rem 0.8rem;
            margin-bottom: 0.8rem;
          }
          .ms-convo-row {
            border: 1px solid var(--border-soft);
            border-radius: var(--r-lg);
            background: var(--surface);
            padding: 0.85rem 0.95rem;
            box-shadow: var(--shadow-sm);
          }
          .ms-convo-head {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            align-items: center;
            margin-bottom: 0.45rem;
          }
          .ms-convo-speaker {
            background: var(--pink-soft);
            color: var(--pink-deep) !important;
            border-radius: 999px;
            padding: 0.2rem 0.55rem;
            font-weight: 850;
            font-size: 0.68rem;
          }
          .ms-time-chip {
            background: var(--warm-5);
            color: var(--warm-3) !important;
            border-radius: 999px;
            padding: 0.2rem 0.55rem;
            font-weight: 700;
            font-size: 0.66rem;
          }
          .ms-convo-text {
            color: var(--warm-2) !important;
            font-size: 0.88rem;
            line-height: 1.68;
          }
          .ms-output-card,
          .ms-item-card {
            box-shadow: var(--shadow-sm);
          }
          .ms-output-card {
            border-color: var(--pink-soft);
            background: linear-gradient(180deg,#FFFFFF 0%,#FFF7F8 100%);
          }
          .ms-card-label::before,
          .ms-item-card h4::before {
            content: "";
            display: inline-block;
            width: 7px;
            height: 7px;
            border-radius: 999px;
            background: currentColor;
            margin-right: 0.4rem;
            vertical-align: 0.08rem;
          }
          .ms-export-wrap {
            border: 1px solid var(--border-soft);
            border-radius: var(--r-xl);
            background: var(--surface);
            padding: 1rem;
            box-shadow: var(--shadow-sm);
          }
          .ms-email-attachments {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin: 0.4rem 0 0.85rem;
          }
          .ms-attachment-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            border: 1px solid var(--pink-mid);
            border-radius: 999px;
            background: var(--pink-soft);
            color: var(--pink-deep) !important;
            font-size: 0.72rem;
            font-weight: 800;
            padding: 0.35rem 0.7rem;
          }
          .ms-empty {
            position: relative;
          }
          .ms-empty::before {
            content: "";
            display: block;
            width: 44px;
            height: 44px;
            border-radius: 14px;
            background: var(--pink-soft);
            margin: 0 auto 0.8rem;
            box-shadow: inset 0 0 0 1px var(--pink-mid);
          }
          .ms-skeleton-line {
            height: 10px;
            border-radius: 999px;
            background: linear-gradient(90deg,var(--warm-5),#fff,var(--warm-5));
            background-size: 220% 100%;
            animation: shimmer 1.3s ease-in-out infinite;
          }

          .ms-upload-title-row {
            display: flex; align-items: center; gap: 0.55rem; margin-bottom: 0.25rem;
          }

          .ms-upload-icon-badge {
            width: 28px; height: 28px; border-radius: 8px;
            background: var(--pink-soft); border: 1px solid var(--pink-mid);
            display: flex; align-items: center; justify-content: center;
            font-size: 0.80rem; color: var(--pink); flex-shrink: 0;
          }

          .ms-upload-title {
            color: var(--warm) !important;
            font-size: 1rem; font-weight: 800; letter-spacing: -0.015em;
          }

          .ms-upload-desc {
            color: var(--warm-3) !important;
            font-size: 0.86rem; line-height: 1.55; margin: 0 0 1.1rem;
          }

          .ms-sub-label {
            color: var(--warm) !important;
            font-size: 0.9rem; font-weight: 800; margin-bottom: 0.2rem;
          }
          .ms-sub-fmt {
            color: var(--warm-3) !important;
            font-size: 0.78rem; margin-bottom: 1rem; line-height: 1.45;
          }

          /* ══ FILE UPLOADER ZONE ═══════════════════════ */
          div[data-testid="stFileUploader"] {
            width: 100% !important;
          }
          div[data-testid="stFileUploader"] > div { width: 100% !important; }

          div[data-testid="stFileUploader"] section {
            position: relative;
            min-height: 150px !important;
            width: 100% !important;
            border: 1.5px dashed #FAD4DC !important;
            border-radius: 16px !important;
            background: #FFFFFF !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            padding: 1.45rem 1rem !important;
            cursor: pointer !important;
            transition: border-color 200ms ease, background 200ms ease, box-shadow 200ms ease !important;
          }
          div[data-testid="stFileUploader"] section:hover {
            border-color: var(--pink) !important;
            background: #FFF1F5 !important;
            box-shadow: 0 8px 24px rgba(251,113,133,0.08) !important;
          }

          div[data-testid="stFileUploader"] section button {
            position: static !important;
            opacity: 1 !important;
            border: 1px solid var(--pink-mid) !important;
            border-radius: 999px !important;
            background: var(--pink-soft) !important;
            color: var(--pink-deep) !important;
            font-family: Inter, sans-serif !important;
            font-size: 0.75rem !important;
            font-weight: 700 !important;
            cursor: pointer !important;
            transition: background 150ms !important;
          }
          div[data-testid="stFileUploader"] section button:hover {
            background: var(--pink-mid) !important;
          }

          div[data-testid="stFileUploader"] section svg {
            color: var(--pink) !important;
            opacity: 0.85 !important;
          }

          div[data-testid="stFileUploader"] label,
          div[data-testid="stFileUploader"] small,
          div[data-testid="stFileUploader"] p { color: var(--warm-3) !important; opacity: 1 !important; }
          div[data-testid="stFileUploader"] section [data-testid="stMarkdownContainer"],
          div[data-testid="stFileUploader"] section span,
          div[data-testid="stFileUploader"] section div {
            color: var(--warm-3) !important;
          }
          div[data-testid="stFileUploader"] small { display: none !important; }

          /* hide native file card */
          [data-testid="stFileUploaderFile"],
          [data-testid="stFileUploaderFileName"],
          [data-testid="stFileUploaderDeleteBtn"],
          [data-testid="stFileUploaderFileData"],
          [data-testid="stFileUploaderFileStatus"],
          [data-testid="stFileUploaderUploadedFile"],
          div[data-testid="stFileUploader"] section + div,
          div[data-testid="stFileUploader"] section ~ div,
          div[data-testid="stFileUploader"] ul,
          div[data-testid="stFileUploader"] li { display: none !important; }

          /* ══ FILE CARD ════════════════════════════════ */
          /* Row containing file card + delete button; keep the Streamlit button but merge it visually */
          div[data-testid="stHorizontalBlock"]:has(.ms-file-card) {
            display: flex !important;
            align-items: center !important;
            flex-wrap: nowrap !important;
            gap: 0.75rem !important;
            margin-top: 0.75rem !important;
            width: 100% !important;
            position: relative !important;
          }
          div[data-testid="stHorizontalBlock"]:has(.ms-file-card)
            > div[data-testid="column"]:has(button[data-testid="stBaseButton-secondary"]) {
            flex: 0 0 48px !important;
            width: 48px !important;
            min-width: 48px !important;
            position: absolute !important;
            right: 12px !important;
            top: 50% !important;
            transform: translateY(-50%) !important;
            z-index: 3 !important;
          }
          div[data-testid="stHorizontalBlock"]:has(.ms-file-card)
            > div[data-testid="column"]:not(:has(button[data-testid="stBaseButton-secondary"])) {
            flex: 1 1 auto !important;
          }
          div[data-testid="stHorizontalBlock"]:has(.ms-file-card)
            > div[data-testid="column"]:has(button[data-testid="stBaseButton-secondary"])
            .stButton { height: 100% !important; display: flex !important; flex-direction: column !important; }
          div[data-testid="stHorizontalBlock"]:has(.ms-file-card)
            > div[data-testid="column"]:has(button[data-testid="stBaseButton-secondary"])
            .stButton > button {
            height: 34px !important; min-height: 34px !important;
            flex: 0 0 34px !important; transform: none !important;
          }

          .ms-file-card {
            display: flex; align-items: center; gap: 0.65rem;
            border: 1px solid #FAD4DC;
            border-radius: 12px;
            background: #FFF1F5;
            padding: 0 4.25rem 0 1rem;
            width: 100%; height: 52px;
            min-height: 52px; max-height: 52px; overflow: hidden;
          }
          .ms-file-icon {
            width: 26px; height: 26px; border-radius: 7px; flex-shrink: 0;
            background: var(--surface); border: 1px solid var(--pink-mid);
            display: flex; align-items: center; justify-content: center;
            color: var(--pink);
          }
          .ms-file-info { flex: 1; min-width: 0; }
          .ms-file-name {
            color: var(--warm) !important;
            font-size: 0.83rem; font-weight: 600;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
          }
          .ms-file-size { color: var(--warm-4) !important; font-size: 0.71rem; }

          /* ══ OR DIVIDER ═══════════════════════════════ */
          .ms-or-wrap {
            min-height: 238px;
            display: flex; align-items: center; justify-content: center;
            position: relative;
          }
          .ms-or-wrap::before {
            content: "";
            position: absolute; top: 0.5rem; bottom: 0.5rem; width: 1px;
            background: linear-gradient(180deg, transparent, var(--border), transparent);
          }
          .ms-or-divider {
            display: flex; align-items: center; justify-content: center;
            color: var(--warm-4) !important;
            font-size: 0.68rem; font-weight: 700;
            width: 32px; height: 32px; border-radius: 50%;
            border: 1px solid var(--border);
            background: var(--surface);
            position: relative; z-index: 1;
            box-shadow: var(--shadow-sm);
          }

          /* ══ BUTTONS ══════════════════════════════════ */
          /* Primary – Generate Meeting Report */
          .stButton > button[data-testid="stBaseButton-primary"] {
            border: none !important;
            border-radius: var(--r-lg) !important;
            background: var(--pink) !important;
            color: #FFFFFF !important;
            font-family: Inter, sans-serif !important;
            font-weight: 700 !important;
            font-size: 0.97rem !important;
            min-height: 3.1rem !important;
            box-shadow: 0 2px 10px rgba(251,113,133,0.30) !important;
            transition: opacity 160ms, transform 160ms, box-shadow 160ms !important;
            letter-spacing: 0.01em !important;
          }
          .stButton > button[data-testid="stBaseButton-primary"]:hover {
            opacity: 0.91 !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 18px rgba(251,113,133,0.36) !important;
          }
          .stButton > button[data-testid="stBaseButton-primary"]:active { transform: none !important; }
          .stButton > button[data-testid="stBaseButton-primary"]:disabled {
            opacity: 0.42 !important; transform: none !important;
          }
          .stButton > button[data-testid="stBaseButton-primary"] p,
          .stButton > button[data-testid="stBaseButton-primary"] span {
            color: #FFFFFF !important; font-weight: 700 !important;
          }

          /* Uploaded-file delete button */
          div[data-testid="stHorizontalBlock"]:has(.ms-file-card)
            .stButton > button[data-testid="stBaseButton-secondary"] {
            width: 34px !important; min-width: 34px !important;
            height: 34px !important; min-height: 34px !important;
            max-height: 34px !important; padding: 0 !important; margin: 0 !important;
            border: 1px solid rgba(239,68,68,0.22) !important;
            border-radius: 10px !important;
            background: #FDE7EF !important;
            color: var(--red) !important;
            font-family: Inter, sans-serif !important;
            font-size: 0.9rem !important; font-weight: 700 !important;
            display: flex !important; align-items: center !important;
            justify-content: center !important;
            transition: background 150ms, border-color 150ms !important;
          }
          div[data-testid="stHorizontalBlock"]:has(.ms-file-card)
            .stButton > button[data-testid="stBaseButton-secondary"]:hover {
            background: #FEE2E2 !important;
            border-color: var(--red) !important;
            box-shadow: 0 3px 10px rgba(239,68,68,0.18) !important;
          }
          div[data-testid="stHorizontalBlock"]:has(.ms-file-card)
            .stButton > button[data-testid="stBaseButton-secondary"] p {
            font-size: 0 !important; width: 15px !important; height: 15px !important;
            background-color: var(--red) !important;
            -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='3 6 5 6 21 6'/%3E%3Cpath d='M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6'/%3E%3Cpath d='M10 11v6'/%3E%3Cpath d='M14 11v6'/%3E%3Cpath d='M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2'/%3E%3C/svg%3E") !important;
            mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='3 6 5 6 21 6'/%3E%3Cpath d='M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6'/%3E%3Cpath d='M10 11v6'/%3E%3Cpath d='M14 11v6'/%3E%3Cpath d='M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2'/%3E%3C/svg%3E") !important;
            -webkit-mask-size: contain !important; mask-size: contain !important;
            -webkit-mask-repeat: no-repeat !important; mask-repeat: no-repeat !important;
            -webkit-mask-position: center !important; mask-position: center !important;
            display: block !important;
          }

          /* ══ PROCESSING ═══════════════════════════════ */
          @keyframes shimmer {
            0%  { background-position: -400% 0 }
            100%{ background-position:  400% 0 }
          }

          .ms-proc-wrap {
            background: var(--surface);
            border: 1px solid var(--border-soft);
            border-radius: var(--r-xl);
            padding: 1.4rem 1.5rem;
            margin: 1.5rem 0;
            box-shadow: var(--shadow-sm);
            animation: fadein 0.3s ease both;
          }

          .ms-proc-top {
            display: flex; align-items: center;
            justify-content: space-between; margin-bottom: 1rem;
          }
          .ms-proc-title {
            font-size: 0.95rem; font-weight: 700; color: var(--warm) !important;
          }
          .ms-elapsed-col { text-align: right; }
          .ms-elapsed-label {
            display: block; color: var(--warm-4) !important;
            font-size: 0.60rem; font-weight: 700;
            letter-spacing: 0.10em; text-transform: uppercase;
          }
          .ms-elapsed-value {
            display: block; color: var(--warm) !important;
            font-size: 1.25rem; font-weight: 800; letter-spacing: -0.03em;
          }

          .ms-bar-track {
            height: 3px; border-radius: 999px;
            background: var(--warm-5); margin-bottom: 1.5rem; overflow: hidden;
          }
          .ms-bar-fill {
            height: 100%; border-radius: 999px;
            background: linear-gradient(90deg, var(--pink), var(--lav), var(--pink));
            background-size: 300% auto;
            animation: shimmer 2s linear infinite;
          }

          .ms-steps {
            display: grid; grid-template-columns: repeat(5, 1fr);
            position: relative;
          }
          .ms-steps::before {
            content: ""; position: absolute;
            top: 13px; left: 10%; right: 10%;
            height: 1px; background: var(--border);
          }

          .ms-step {
            display: flex; flex-direction: column;
            align-items: center; gap: 0.40rem; position: relative; z-index: 1;
          }
          .ms-step-circle {
            width: 26px; height: 26px; border-radius: 50%;
            border: 1.5px solid var(--border);
            background: var(--surface);
            display: flex; align-items: center; justify-content: center;
            font-size: 0.70rem; font-weight: 700;
            color: var(--warm-4) !important;
            transition: all 200ms;
          }
          .ms-step.done .ms-step-circle {
            border-color: var(--green); background: var(--green-soft);
            color: var(--green) !important;
          }
          .ms-step.active .ms-step-circle {
            border-color: var(--pink); background: var(--pink); color: #fff !important;
            box-shadow: 0 0 0 4px var(--pink-soft);
          }
          .ms-step-label {
            color: var(--warm-4) !important;
            font-size: 0.67rem; font-weight: 500; text-align: center; line-height: 1.3;
          }
          .ms-step.done  .ms-step-label { color: var(--green) !important; }
          .ms-step.active .ms-step-label { color: var(--pink-deep) !important; font-weight: 600; }

          /* ══ METRICS GRID ═════════════════════════════ */
          .ms-metrics-grid {
            display: grid; grid-template-columns: repeat(4, minmax(0,1fr));
            gap: 0.85rem; margin: 1.1rem 0 0.65rem;
          }

          .ms-stat-card {
            border: 1px solid var(--border-soft); border-radius: var(--r-lg);
            background: var(--surface); padding: 1.1rem 1.1rem 0.95rem;
            box-shadow: var(--shadow-sm);
            transition: border-color 200ms, transform 200ms, box-shadow 200ms;
          }
          .ms-stat-card:hover {
            transform: translateY(-2px); box-shadow: var(--shadow-md);
          }
          .ms-stat-icon-wrap { display: none !important; }
          .ms-stat-label {
            display: block; color: var(--warm-4) !important;
            font-size: 0.68rem; font-weight: 600;
            letter-spacing: 0.04em; margin-bottom: 0.22rem;
          }
          .ms-stat-value {
            display: block; color: var(--warm) !important;
            font-size: 1.45rem; font-weight: 800;
            letter-spacing: -0.04em; line-height: 1.0;
          }
          .ms-stat-sub {
            display: block; font-size: 0.64rem; font-weight: 700;
            letter-spacing: 0.06em; text-transform: uppercase; margin-top: 0.38rem;
          }
          .ms-stat-card.c-purple  .ms-stat-sub { color: var(--lav); }
          .ms-stat-card.c-emerald .ms-stat-sub { color: var(--green); }
          .ms-stat-card.c-amber   .ms-stat-sub { color: var(--amber); }
          .ms-stat-card.c-violet  .ms-stat-sub { color: var(--pink); }
          .ms-stat-card.c-purple  { border-color: var(--lav-soft); }
          .ms-stat-card.c-emerald { border-color: var(--green-soft); }
          .ms-stat-card.c-amber   { border-color: var(--amber-soft); }
          .ms-stat-card.c-violet  { border-color: var(--pink-soft); }

          /* ══ SECURITY ════════════════════════════════ */
          .ms-security {
            display: flex; align-items: center; justify-content: center;
            gap: 0.4rem; color: var(--warm-4) !important; font-size: 0.74rem;
            margin: 0.4rem 0 1.2rem;
          }

          /* ══ EXPORT ══════════════════════════════════ */
          .ms-export-section { margin: 1.5rem 0 0.5rem; }
          .ms-export-wrap {
            border: 1px solid #FAD4DC;
            border-radius: 18px;
            background: linear-gradient(180deg,#FFFFFF 0%,#FFF7FA 100%);
            padding: 1.25rem 1.35rem 1rem;
            box-shadow: 0 4px 16px rgba(251,113,133,0.06);
            margin-bottom: 1rem;
          }
          .ms-export-hdr {
            display: flex; align-items: flex-start; gap: 0.65rem;
          }
          .ms-export-hdr-icon {
            width: 36px; height: 36px; border-radius: 10px;
            background: var(--pink-soft); border: 1px solid var(--pink-mid);
            display: flex; align-items: center; justify-content: center;
            font-size: 0.95rem; color: var(--pink); flex-shrink: 0;
          }
          .ms-export-hdr-title {
            color: var(--warm) !important;
            font-size: 1rem; font-weight: 800; letter-spacing: -0.02em;
          }
          .ms-export-sub {
            color: var(--warm-3) !important;
            font-size: 0.8rem; margin: 0.2rem 0 0; line-height: 1.5;
          }
          .ms-export-actions {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.85rem;
            margin-top: 0.25rem;
          }
          .ms-export-section ~ div[data-testid="stHorizontalBlock"] {
            gap: 0.85rem !important;
            margin-top: 0.15rem !important;
          }
          .ms-export-section ~ div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            border: 1px solid var(--border-soft);
            border-radius: var(--r-xl);
            background: var(--surface);
            padding: 0.35rem 0.35rem 0.5rem !important;
            box-shadow: var(--shadow-sm);
            transition: border-color 180ms, box-shadow 180ms, transform 180ms;
          }
          .ms-export-section ~ div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:hover {
            border-color: var(--pink-mid);
            box-shadow: var(--shadow-md);
            transform: translateY(-2px);
          }
          .ms-export-option {
            border: 1px solid #FAD4DC;
            border-radius: 16px;
            background: #FFFFFF;
            padding: 1.2rem 1.15rem 0.9rem;
            box-shadow: 0 4px 16px rgba(251,113,133,0.06);
            transition: border-color 180ms, box-shadow 180ms, transform 180ms;
            min-height: 128px;
          }
          .ms-export-option:hover {
            border-color: var(--pink-mid);
            box-shadow: var(--shadow-md);
            transform: translateY(-2px);
          }
          .ms-export-option-icon {
            width: 40px; height: 40px; border-radius: 11px;
            display: flex; align-items: center; justify-content: center;
            font-size: 0.68rem; font-weight: 800; letter-spacing: 0.04em;
            margin-bottom: 0.75rem;
          }
          .ms-export-option-icon.pdf {
            background: var(--pink-soft); color: var(--pink-deep) !important;
            border: 1px solid var(--pink-mid);
          }
          .ms-export-option-icon.docx {
            background: var(--lav-soft); color: #5B21B6 !important;
            border: 1px solid var(--lav-mid);
          }
          .ms-export-option-icon.email {
            background: #ECFDF5; color: #047857 !important;
            border: 1px solid #A7F3D0;
          }
          .ms-export-option-title {
            color: var(--warm) !important;
            font-size: 0.9rem; font-weight: 800; letter-spacing: -0.02em;
            margin-bottom: 0.25rem;
          }
          .ms-export-option-desc {
            color: var(--warm-3) !important;
            font-size: 0.74rem; line-height: 1.45;
          }
          .ms-export-actions .stDownloadButton,
          .ms-export-actions .stButton { margin-top: 0.65rem; }
          .ms-export-actions .stDownloadButton > button,
          .ms-export-actions .stButton > button[data-testid="stBaseButton-secondary"] {
            min-height: 2.65rem !important;
            border-radius: var(--r-lg) !important;
            font-size: 0.82rem !important;
          }
          .ms-export-email-wrap .stButton > button {
            min-height: 2.65rem !important;
            width: 100% !important;
          }

          .stDownloadButton > button {
            min-height: 3.35rem !important;
            border: 1px solid #FAD4DC !important;
            border-radius: 14px !important;
            background: var(--surface) !important;
            color: var(--warm) !important;
            font-family: Inter, sans-serif !important;
            text-align: left !important;
            box-shadow: var(--shadow-sm) !important;
            transition: transform 160ms ease, border-color 160ms ease, box-shadow 160ms ease !important;
          }
          .stDownloadButton > button:hover {
            transform: translateY(-2px) !important;
            border-color: var(--pink-mid) !important;
            box-shadow: 0 6px 18px rgba(251,113,133,0.14) !important;
          }
          .stDownloadButton > button p {
            color: var(--warm) !important; font-weight: 700 !important;
            font-size: 0.86rem !important;
          }

          /* ══ TABS ════════════════════════════════════ */
          .stTabs [data-baseweb="tab-list"] {
            gap: 0;
            border-bottom: 1px solid var(--border-soft);
            padding: 0; margin: 0 0 0.55rem;
            background: transparent;
          }
          .stTabs [data-baseweb="tab"] {
            border: none; border-bottom: 2px solid transparent;
            border-radius: 0; background: transparent;
            color: var(--warm-3) !important;
            font-family: Inter, sans-serif;
            font-weight: 500; font-size: 0.82rem;
            padding: 0.70rem 1.1rem; white-space: nowrap;
            transition: color 150ms;
          }
          .stTabs [data-baseweb="tab"]:hover { color: var(--warm-2) !important; }
          .stTabs [aria-selected="true"] {
            color: var(--pink-deep) !important; font-weight: 700 !important;
            background: transparent !important;
          }
          .stTabs [data-baseweb="tab-highlight"] {
            background: var(--pink) !important; height: 2px !important;
          }
          .stTabs [data-baseweb="tab-panel"] { padding: 0 !important; margin: 0 !important; }
          .stTabs [data-baseweb="tab-panel"] [data-testid="stVerticalBlock"] {
            gap: 0 !important; padding: 0 !important; margin: 0 !important;
          }
          .stTabs [data-baseweb="tab-panel"] > div,
          .stTabs [data-baseweb="tab-panel"] > div > div,
          .stTabs [data-baseweb="tab-panel"] [data-testid="stVerticalBlock"] > div,
          .stTabs [data-baseweb="tab-panel"] [data-testid="stMarkdownContainer"] {
            margin-top: 0 !important; padding-top: 0 !important;
          }

          /* per-tab accent — Summary, Discussion, Decisions, Action Items, Transcript */
          .stTabs [data-baseweb="tab-list"] button:nth-of-type(1)[aria-selected="true"] {
            border-bottom-color: var(--lav) !important; color: #5B21B6 !important;
          }
          .stTabs [data-baseweb="tab-list"] button:nth-of-type(2)[aria-selected="true"] {
            border-bottom-color: #34D399 !important; color: #047857 !important;
          }
          .stTabs [data-baseweb="tab-list"] button:nth-of-type(3)[aria-selected="true"] {
            border-bottom-color: var(--amber) !important; color: #92400E !important;
          }
          .stTabs [data-baseweb="tab-list"] button:nth-of-type(4)[aria-selected="true"] {
            border-bottom-color: var(--pink) !important; color: var(--pink-deep) !important;
          }
          .stTabs [data-baseweb="tab-list"] button:nth-of-type(5)[aria-selected="true"] {
            border-bottom-color: var(--warm-4) !important; color: var(--warm-2) !important;
          }

          /* ══ SCROLLABLE CONTENT ══════════════════════ */
          .ms-tab-scroll {
            height: 440px; overflow-y: auto; padding: 0 !important; margin-top: 0 !important;
            scrollbar-width: thin; scrollbar-color: var(--warm-4) transparent;
          }
          .ms-tab-scroll::-webkit-scrollbar { width: 4px; }
          .ms-tab-scroll::-webkit-scrollbar-track { background: transparent; }
          .ms-tab-scroll::-webkit-scrollbar-thumb {
            background: var(--warm-4); border-radius: 999px;
          }

          /* ══ TRANSCRIPT VIEWER ════════════════════════ */
          .ms-transcript-scroll {
            height: 480px; overflow-y: auto; margin-top: 0;
            scrollbar-width: thin; scrollbar-color: var(--warm-4) transparent;
          }
          .ms-transcript-scroll::-webkit-scrollbar { width: 4px; }
          .ms-transcript-scroll::-webkit-scrollbar-track { background: transparent; }
          .ms-transcript-scroll::-webkit-scrollbar-thumb {
            background: var(--warm-4); border-radius: 999px;
          }

          .ms-tr-row {
            display: grid; grid-template-columns: 116px 1fr;
            border-bottom: 1px solid var(--border-soft);
            transition: background 120ms;
          }
          .ms-tr-row:last-child { border-bottom: none; }
          .ms-tr-row:hover { background: var(--pink-soft); }

          .ms-tr-left {
            padding: 0.85rem 0.75rem 0.85rem 0.95rem;
            border-right: 1px solid var(--border-soft);
            display: flex; flex-direction: column; gap: 0.25rem;
          }

          .ms-speaker-badge {
            display: inline-block; border-radius: 5px;
            font-size: 0.62rem; font-weight: 700;
            letter-spacing: 0.06em; text-transform: uppercase;
            padding: 0.15rem 0.40rem;
          }
          .ms-speaker-badge.s1 { background: var(--pink-soft);  color: var(--pink-deep); }
          .ms-speaker-badge.s2 { background: var(--green-soft); color: #065F46; }
          .ms-speaker-badge.s3 { background: var(--blue-soft);  color: #1E40AF; }
          .ms-speaker-badge.s4 { background: var(--lav-soft);   color: #5B21B6; }

          .ms-tr-timestamp {
            color: var(--warm-4) !important;
            font-size: 0.65rem; font-variant-numeric: tabular-nums; font-weight: 500;
          }

          .ms-tr-right { padding: 0.85rem 0.95rem; display: flex; align-items: center; }
          .ms-tr-text { color: var(--warm-2) !important; font-size: 0.88rem; line-height: 1.6; }

          /* ══ CONTENT CARDS ════════════════════════════ */
          /* Summary */
          .ms-output-card {
            border: 1px solid #DDD6FE;
            border-radius: var(--r-lg);
            background: #FAF7FF;
            border-left: 4px solid var(--lav);
            padding: 1.2rem 1.25rem;
            margin: 0 0 1.1rem;
            box-shadow: 0 4px 16px rgba(167,139,250,0.07);
            transition: border-color 180ms, box-shadow 180ms;
          }
          .ms-output-card:hover {
            border-color: var(--lav-mid);
            box-shadow: var(--shadow-md);
          }
          .ms-card-label {
            color: var(--lav) !important;
            font-size: 0.63rem; font-weight: 700;
            letter-spacing: 0.09em; text-transform: uppercase; margin-bottom: 0.22rem;
          }
          .ms-card-title {
            color: var(--warm) !important;
            font-size: 1.05rem; font-weight: 700; margin: 0 0 0.65rem;
            letter-spacing: -0.015em;
          }
          .ms-card-body { color: var(--warm-3) !important; font-size: 0.89rem; line-height: 1.7; }
          .ms-chip-row { display: flex; flex-wrap: wrap; gap: 0.38rem; margin-top: 0.75rem; }
          .ms-chip {
            border: 1px solid var(--pink-mid); border-radius: 999px;
            background: var(--pink-soft); color: var(--pink-deep) !important;
            font-size: 0.73rem; font-weight: 600; padding: 0.22rem 0.60rem;
          }

          /* Discussion */
          .ms-item-card {
            border: 1px solid var(--border); border-radius: var(--r-lg);
            background: var(--surface); padding: 1rem 1.05rem;
            margin: 0 0 1.0rem;
            transition: border-color 180ms, transform 180ms, box-shadow 180ms;
          }
          .ms-item-card:hover { transform: translateY(-2px); box-shadow: var(--shadow-md); }

          .ms-item-card.discussion {
            border-color: #BBF7D0;
            background: #F0FDF4;
            border-left: 4px solid #34D399;
          }
          .ms-item-card.discussion h4 { color: #047857 !important; }
          .ms-item-card.discussion:hover { border-color: #A7F3D0; }

          .ms-item-card.decision {
            border-color: #FDE68A;
            background: #FFFBEB;
            border-left: 4px solid var(--amber);
          }
          .ms-item-card.decision h4 { color: #92400E !important; }
          .ms-item-card.decision:hover { border-color: #FCD34D; }

          .ms-item-card.action {
            border-color: #FAD4DC;
            background: #FFF1F5;
            border-left: 4px solid var(--pink);
          }
          .ms-item-card.action h4 { color: var(--pink-deep) !important; }
          .ms-item-card.action:hover { border-color: var(--pink-mid); }

          .ms-item-card h4 {
            font-size: 0.62rem; font-weight: 700;
            letter-spacing: 0.09em; text-transform: uppercase; margin-bottom: 0.30rem;
          }
          .ms-item-card p {
            color: var(--warm-2) !important;
            font-size: 0.88rem; line-height: 1.62; margin-bottom: 0.50rem;
          }
          .ms-meta-row { display: flex; flex-wrap: wrap; gap: 0.35rem; }
          .ms-meta {
            border: 1px solid var(--border); border-radius: 999px;
            color: var(--warm-4) !important; background: var(--warm-5);
            font-size: 0.66rem; font-weight: 500; padding: 0.17rem 0.48rem;
          }
          .ms-item-card.discussion .ms-meta {
            border-color: #BBF7D0; color: #065F46 !important; background: #F0FDF4;
          }
          .ms-item-card.decision .ms-meta {
            border-color: #FDE68A; color: #92400E !important; background: #FFFBEB;
          }
          .ms-item-card.action .ms-meta {
            border-color: var(--pink-mid); color: var(--pink-deep) !important;
            background: var(--pink-soft);
          }

          .ms-report-stack {
            display: grid;
            gap: 1rem;
            margin: 0.35rem 0 1.25rem;
          }
          .ms-report-block {
            border: 1px solid var(--border-soft);
            border-radius: var(--r-xl);
            background: var(--surface);
            padding: 1rem;
            box-shadow: var(--shadow-sm);
          }
          .ms-report-block-title {
            color: var(--warm) !important;
            font-size: 0.95rem;
            font-weight: 850;
            letter-spacing: -0.02em;
            margin-bottom: 0.7rem;
          }
          .ms-report-block.summary { border-top: 3px solid var(--lav); }
          .ms-report-block.discussion { border-top: 3px solid #34D399; }
          .ms-report-block.decisions { border-top: 3px solid var(--amber); }
          .ms-report-block.actions { border-top: 3px solid var(--pink); }
          .ms-report-block.transcript { border-top: 3px solid var(--warm-4); }

          /* ══ MISC ════════════════════════════════════ */
          .ms-empty {
            border: 1.5px dashed var(--border);
            border-radius: var(--r-xl);
            background: var(--surface);
            color: var(--warm-3) !important;
            font-size: 0.84rem;
            padding: 2rem 1.5rem;
            text-align: center;
            box-shadow: var(--shadow-sm);
          }
          .ms-empty-state {
            border: 1px solid #FAD4DC;
            border-radius: 16px;
            background: #FFFBFE;
            padding: 3rem 1.75rem;
            text-align: center;
            margin: 0.5rem 0 1.25rem;
            box-shadow: 0 4px 16px rgba(251,113,133,0.06);
            display: flex;
            min-height: 260px;
            flex-direction: column;
            align-items: center;
            justify-content: center;
          }
          .ms-empty-icon {
            width: 52px; height: 52px; border-radius: 16px;
            margin: 0 auto 1rem;
            background: linear-gradient(135deg, var(--pink-soft), var(--lav-soft));
            border: 1px solid var(--border-soft);
            display: flex; align-items: center; justify-content: center;
            color: var(--pink-deep) !important;
            font-size: 1.25rem; font-weight: 800;
          }
          .ms-empty-title {
            color: var(--warm) !important;
            font-size: 1.05rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin-bottom: 0.45rem;
          }
          .ms-empty-copy {
            color: var(--warm-3) !important;
            font-size: 0.84rem;
            line-height: 1.6;
            max-width: 420px;
            margin: 0 auto;
          }
          .ms-email-compose {
            border: 1px solid #FAD4DC;
            border-radius: 18px;
            background: linear-gradient(180deg,#FFFFFF 0%,#FFFBFE 100%);
            padding: 1.35rem 1.45rem 1.15rem;
            box-shadow: 0 4px 16px rgba(251,113,133,0.06);
            margin-top: 1rem;
          }
          .ms-email-compose-hdr {
            color: var(--warm) !important;
            font-size: 1rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin-bottom: 0.2rem;
          }
          .ms-email-compose-sub {
            color: var(--warm-3) !important;
            font-size: 0.8rem;
            line-height: 1.5;
            margin-bottom: 1rem;
          }
          .ms-email-field-label {
            color: var(--warm-2) !important;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
          }
          .ms-attachment-preview {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            padding: 0.75rem 0.9rem;
            border: 1px solid #FAD4DC;
            border-radius: 14px;
            background: #FFF1F5;
            margin-bottom: 1rem;
          }
          .ms-attachment-preview-icon {
            width: 34px; height: 34px; border-radius: 9px;
            background: var(--pink-soft);
            border: 1px solid var(--pink-mid);
            color: var(--pink-deep) !important;
            display: flex; align-items: center; justify-content: center;
            font-size: 0.62rem; font-weight: 800;
          }
          .ms-attachment-preview-name {
            color: var(--warm) !important;
            font-size: 0.82rem;
            font-weight: 700;
          }
          .ms-attachment-preview-meta {
            color: var(--warm-3) !important;
            font-size: 0.72rem;
          }

          .ms-section-header { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.10rem; }
          .ms-section-icon { color: var(--pink); font-size: 0.95rem; }
          .ms-section-title {
            color: var(--warm) !important;
            font-size: 0.97rem; font-weight: 700; margin: 0; letter-spacing: -0.015em;
          }
          .ms-section-copy { color: var(--warm-4) !important; font-size: 0.80rem; margin: 0.10rem 0 1.1rem; }

          /* Inputs */
          .stTextArea textarea {
            border: 1px solid var(--border) !important;
            border-radius: var(--r) !important;
            background: var(--warm-6) !important;
            color: var(--warm-2) !important;
            font-size: 0.87rem !important;
            transition: border-color 150ms !important;
          }
          .stTextArea textarea:focus {
            border-color: var(--pink) !important;
            box-shadow: 0 0 0 3px var(--pink-soft) !important;
          }

          div[data-testid="stTextInput"] input {
            border: 1px solid var(--border) !important;
            border-radius: var(--r) !important;
            background: var(--warm-6) !important;
            color: var(--warm-2) !important;
          }
          div[data-testid="stTextInput"] input:focus {
            border-color: var(--pink) !important;
            box-shadow: 0 0 0 3px var(--pink-soft) !important;
          }

          div[data-testid="stAlert"] {
            border-radius: var(--r-lg) !important;
            border: 1px solid var(--border) !important;
            background: var(--surface) !important;
          }
          div[data-testid="stInfoAlert"] {
            background: var(--pink-soft) !important;
            border-color: var(--pink-mid) !important;
          }

          div[data-testid="stProgressBar"] > div {
            background: var(--warm-5) !important; border-radius: 999px !important;
          }
          div[data-testid="stProgressBar"] > div > div {
            background: var(--pink) !important;
            border-radius: 999px !important;
          }

          [data-testid="stExpander"] {
            border: 1px solid var(--border-soft) !important;
            border-radius: var(--r-lg) !important;
            background: var(--surface) !important;
          }
          [data-testid="stCaptionContainer"] {
            color: var(--warm-4) !important; font-size: 0.74rem !important;
          }

          hr { border-color: var(--border-soft) !important; }

          /* ══ FOOTER ══════════════════════════════════ */
          .ms-footer {
            text-align: center; color: var(--warm-4) !important;
            font-size: 0.70rem;
            padding: 1.75rem 0 0.5rem;
            border-top: 1px solid var(--border-soft);
            margin-top: 2rem;
          }

          /* ══ COPY BUTTON ════════════════════════════ */
          .ms-copy-row {
            display: flex; align-items: center; gap: 0.5rem;
            padding: 0.55rem 0 0.65rem;
            border-bottom: 1px solid var(--border-soft);
            margin-bottom: 0;
          }

          /* ══ SCROLLBAR ══════════════════════════════ */
          .stButton > button[data-testid="stBaseButton-secondary"] {
            width: 100% !important;
            min-height: 38px !important;
            height: auto !important;
            max-height: none !important;
            padding: 0.55rem 0.8rem !important;
            border: 1px solid #FAD4DC !important;
            border-radius: 12px !important;
            background: var(--surface) !important;
            color: var(--warm-2) !important;
            box-shadow: var(--shadow-sm) !important;
            font-size: 0.78rem !important;
            font-weight: 700 !important;
          }
          .stButton > button[data-testid="stBaseButton-secondary"]:hover {
            border-color: var(--pink-mid) !important;
            background: var(--pink-soft) !important;
            color: var(--pink-deep) !important;
            box-shadow: 0 6px 18px rgba(251,113,133,0.12) !important;
          }
          .stButton > button[data-testid="stBaseButton-secondary"] p {
            color: inherit !important;
            font-size: inherit !important;
            width: auto !important;
            height: auto !important;
            background: transparent !important;
            -webkit-mask-image: none !important;
            mask-image: none !important;
          }
          div[data-testid="stHorizontalBlock"]:has(.ms-file-card)
            .stButton > button[data-testid="stBaseButton-secondary"] {
            width: 34px !important;
            min-width: 34px !important;
            height: 34px !important;
            min-height: 34px !important;
            max-height: 34px !important;
            padding: 0 !important;
            margin: 0 !important;
            border: 1px solid rgba(239,68,68,0.22) !important;
            border-radius: 10px !important;
            background: #FDE7EF !important;
            box-shadow: none !important;
          }
          div[data-testid="stHorizontalBlock"]:has(.ms-file-card)
            .stButton > button[data-testid="stBaseButton-secondary"]:hover {
            background: #FEE2E2 !important;
            border-color: var(--red) !important;
            box-shadow: 0 3px 10px rgba(239,68,68,0.18) !important;
          }
          div[data-testid="stHorizontalBlock"]:has(.ms-file-card)
            .stButton > button[data-testid="stBaseButton-secondary"] p {
            font-size: 0 !important;
            width: 15px !important;
            height: 15px !important;
            background-color: var(--red) !important;
            -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='3 6 5 6 21 6'/%3E%3Cpath d='M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6'/%3E%3Cpath d='M10 11v6'/%3E%3Cpath d='M14 11v6'/%3E%3Cpath d='M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2'/%3E%3C/svg%3E") !important;
            mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='3 6 5 6 21 6'/%3E%3Cpath d='M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6'/%3E%3Cpath d='M10 11v6'/%3E%3Cpath d='M14 11v6'/%3E%3Cpath d='M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2'/%3E%3C/svg%3E") !important;
            -webkit-mask-size: contain !important;
            mask-size: contain !important;
            -webkit-mask-repeat: no-repeat !important;
            mask-repeat: no-repeat !important;
            -webkit-mask-position: center !important;
            mask-position: center !important;
            display: block !important;
          }

          ::-webkit-scrollbar { width: 4px; height: 4px; }
          ::-webkit-scrollbar-track { background: transparent; }
          ::-webkit-scrollbar-thumb { background: var(--warm-4); border-radius: 999px; }
          ::-webkit-scrollbar-thumb:hover { background: var(--warm-3); }

          /* ══ RESPONSIVE ═════════════════════════════ */
          @media (max-width: 860px) {
            .block-container { max-width: 100% !important; }
            .ms-metrics-grid { grid-template-columns: repeat(2, minmax(0,1fr)); }
            .ms-dashboard-grid { grid-template-columns: 1fr; }
            .ms-quick-stats { grid-template-columns: repeat(4,minmax(0,1fr)); }
            .ms-steps { grid-template-columns: repeat(3,1fr); }
            .ms-steps > .ms-step:nth-child(n+4) { display: none; }
          }
          @media (max-width: 768px) {
            .block-container {
              padding-left: 0.9rem !important;
              padding-right: 0.9rem !important;
            }
            div[data-testid="stHorizontalBlock"]:not(:has(button[data-testid="stBaseButton-secondary"])) {
              flex-direction: column !important; gap: 0.8rem !important;
            }
            div[data-testid="stHorizontalBlock"]:not(:has(button[data-testid="stBaseButton-secondary"]))
              > div[data-testid="column"] {
              width: 100% !important; min-width: 0 !important; flex: 1 1 auto !important;
            }
            .ms-or-wrap {
              min-height: 44px;
              margin: 0;
            }
            .ms-or-wrap::before {
              left: 0.75rem; right: 0.75rem; top: 50%; bottom: auto;
              width: auto; height: 1px;
              background: linear-gradient(90deg, transparent, var(--border), transparent);
            }
            div[data-testid="stFileUploader"] section { min-height: 120px !important; }
            .ms-navbar { align-items: flex-start; flex-direction: column; }
            .ms-export-actions { grid-template-columns: 1fr; }
            .ms-workflow-timeline { flex-direction: column; align-items: flex-start; }
          }
          @media (max-width: 640px) {
            .ms-hero h1 { font-size: 1.9rem !important; }
            .ms-metrics-grid { grid-template-columns: repeat(2,minmax(0,1fr)); gap: 0.55rem; }
            .ms-tr-row { grid-template-columns: 94px 1fr; }
            .ms-transcript-scroll { height: 340px; }
            .stTabs [data-baseweb="tab-list"] { overflow-x: auto; scrollbar-width: none; }
            .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
          }
          @media (max-width: 420px) {
            .ms-metrics-grid { grid-template-columns: 1fr; }
            .ms-quick-stats { grid-template-columns: repeat(2,minmax(0,1fr)); }
            .ms-pill-row { flex-direction: column; align-items: center; }
            .ms-workflow-timeline { padding: 0.75rem; }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
        "longer while your meeting report is prepared."
    )


def render_stage_status(
    placeholder: st.delta_generator.DeltaGenerator,
    *,
    active_index: int,
    started_at: float,
    note: str = "",
) -> None:
    elapsed = format_elapsed(time.perf_counter() - started_at)

    step_labels = [
        "Upload",
        "Preparing",
        "Identify Speakers",
        "Generate Report",
        "Export",
    ]
    steps_html = ""
    for i, label in enumerate(step_labels):
        if active_index >= len(PROCESSING_STAGES) or i < active_index:
            cls = "done"; circle = "&#10003;"
        elif i == active_index:
            cls = "active"; circle = str(i + 1)
        else:
            cls = "waiting"; circle = str(i + 1)
        steps_html += f"""<div class="ms-step {cls}">
            <div class="ms-step-circle">{circle}</div>
            <div class="ms-step-label">{label}</div>
          </div>"""

    pct  = int((active_index / len(PROCESSING_STAGES)) * 100)
    note_html = f"<p style='color:var(--warm-3);font-size:.82rem;margin-top:.75rem;text-align:center'>{html.escape(note)}</p>" if note else ""

    # FIX #6: Removed empty ms-proc-icon div — title is now clean text only
    placeholder.markdown(
        f"""
        <div class="ms-proc-wrap">
          <div class="ms-proc-top">
            <div class="ms-proc-title">Preparing Your Report <span class="ms-badge queue">Processing</span></div>
            <div class="ms-elapsed-col">
              <span class="ms-elapsed-label">Elapsed</span>
              <span class="ms-elapsed-value">{elapsed}</span>
            </div>
          </div>
          <div class="ms-bar-track">
            <div class="ms-bar-fill" style="width:{pct}%"></div>
          </div>
          <div style="display:grid;gap:.45rem;margin:.85rem 0 .25rem;">
            <div class="ms-skeleton-line" style="width:92%"></div>
            <div class="ms-skeleton-line" style="width:74%"></div>
          </div>
          <div class="ms-steps">{steps_html}</div>
          {note_html}
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
        "Discussion Points": str(len(analysis.key_discussion_points) if analysis else 0),
        "Decisions": str(len(analysis.decisions) if analysis else 0),
        "Action Items": str(len(analysis.action_items) if analysis else 0),
        "Processing Time": format_elapsed(time.perf_counter() - started_at),
    }


def render_success_metrics() -> None:
    metrics = st.session_state.get("success_metrics")
    if not metrics:
        return

    dp  = html.escape(metrics.get("Discussion Points", "0"))
    dec = html.escape(metrics.get("Decisions", "0"))
    act = html.escape(metrics.get("Action Items", "0"))
    # FIX #5: Removed ms-stat-icon-wrap divs — cards now show only label, value, subtitle
    st.markdown(
        f"""
        <div class="ms-metrics-grid">
          <div class="ms-stat-card c-purple">
            <span class="ms-stat-label">Summary</span>
            <strong class="ms-stat-value">Ready</strong>
            <small class="ms-stat-sub">Meeting summary generated</small>
          </div>
          <div class="ms-stat-card c-emerald">
            <span class="ms-stat-label">Discussion Points</span>
            <strong class="ms-stat-value">{dp}</strong>
            <small class="ms-stat-sub">Topics captured</small>
          </div>
          <div class="ms-stat-card c-amber">
            <span class="ms-stat-label">Decisions</span>
            <strong class="ms-stat-value">{dec}</strong>
            <small class="ms-stat-sub">Decisions identified</small>
          </div>
          <div class="ms-stat-card c-violet">
            <span class="ms-stat-label">Action Items</span>
            <strong class="ms-stat-value">{act}</strong>
            <small class="ms-stat-sub">Follow-up tasks</small>
          </div>
        </div>
        <div class="ms-security">Your data is secure and never stored.</div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_shell() -> None:
    st.sidebar.markdown(
        """
        <div class="ms-side-shell">
          <div class="ms-side-brand">
            <div class="ms-side-logo-mark">MS</div>
            <div class="ms-side-title">MeetScribe</div>
          </div>
          <div class="ms-side-nav">
            <div class="ms-side-item active"><span class="ms-side-icon">01</span><span>Dashboard</span></div>
            <div class="ms-side-item"><span class="ms-side-icon">02</span><span>Meetings</span></div>
            <div class="ms-side-item"><span class="ms-side-icon">03</span><span>Reports</span></div>
            <div class="ms-side-item"><span class="ms-side-icon">04</span><span>Templates</span></div>
            <div class="ms-side-item"><span class="ms-side-icon">05</span><span>Settings</span></div>
            <div class="ms-side-item"><span class="ms-side-icon">?</span><span>Support</span></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="ms-empty-state">
          <div class="ms-empty-icon">◎</div>
          <h3 class="ms-empty-title">No meeting uploaded yet</h3>
          <p class="ms-empty-copy">
            Upload a recording or transcript above to review speakers, verify the transcript,
            and generate professional Minutes of Meeting.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def dashboard_file_rows(uploaded_file: object | None, transcript_file: object | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for file_obj, label in (
        (uploaded_file, "Recording"),
        (transcript_file, "Transcript"),
    ):
        if file_obj is None:
            continue
        size = getattr(file_obj, "size", 0) or 0
        rows.append(
            {
                "name": getattr(file_obj, "name", label),
                "meta": f"{label} - {size / (1024 * 1024):.2f} MB",
                "status": "Queued",
                "class": "queue",
            }
        )

    stored_filename = st.session_state.get("uploaded_filename", "")
    if stored_filename and not rows:
        rows.append(
            {
                "name": stored_filename,
                "meta": "Processed meeting input",
                "status": "Ready",
                "class": "ready",
            }
        )
    return rows


def render_dashboard_context(
    *,
    uploaded_file: object | None,
    transcript_file: object | None,
    analysis: MeetingAnalysisResult | None,
) -> None:
    rows = dashboard_file_rows(uploaded_file, transcript_file)
    if not rows:
        rows = [
            {
                "name": "No meeting input selected",
                "meta": "Upload a recording or transcript to begin",
                "status": "Idle",
                "class": "idle",
            }
        ]

    meetings = "1" if st.session_state.get("transcript_text") else "0"
    reports = "1" if analysis is not None else "0"
    action_items = str(len(analysis.action_items)) if analysis is not None else "0"
    success_metrics = st.session_state.get("success_metrics") or {}
    duration = success_metrics.get("Duration", "0m")
    active_files = sum(1 for item in (uploaded_file, transcript_file) if item is not None)
    processing_status = "Ready" if analysis is not None else ("Queued" if active_files else "Idle")
    status_class = "ready" if analysis is not None else ("queue" if active_files else "idle")

    activity_html = "".join(
        f"""
        <div class="ms-activity-row">
          <div class="ms-activity-icon">File</div>
          <div>
            <div class="ms-activity-name">{html.escape(row["name"])}</div>
            <div class="ms-activity-meta">{html.escape(row["meta"])}</div>
          </div>
          <span class="ms-badge {html.escape(row["class"])}">{html.escape(row["status"])}</span>
        </div>
        """
        for row in rows[:3]
    )
    queue_html = "".join(
        f"""
        <div class="ms-queue-line">
          <div>
            <div class="ms-queue-title">{html.escape(row["name"])}</div>
            <div class="ms-queue-sub">{html.escape(row["meta"])}</div>
          </div>
          <span class="ms-badge {html.escape(row["class"])}">{html.escape(row["status"])}</span>
        </div>
        """
        for row in rows[:3]
    )

    st.markdown(
        f"""
        <div class="ms-dashboard-grid">
          <div class="ms-dashboard-card">
            <div class="ms-dashboard-head">
              <div class="ms-dashboard-title">Recent Activity</div>
              <div class="ms-dashboard-muted">{active_files} file{"s" if active_files != 1 else ""} active</div>
            </div>
            {activity_html}
          </div>
          <div class="ms-dashboard-card">
            <div class="ms-dashboard-head">
              <div class="ms-dashboard-title">Quick Stats</div>
              <div class="ms-dashboard-muted">Current session</div>
            </div>
            <div class="ms-quick-stats">
              <div class="ms-quick-stat"><span class="ms-quick-value">{meetings}</span><span class="ms-quick-label">Meetings</span></div>
              <div class="ms-quick-stat"><span class="ms-quick-value">{reports}</span><span class="ms-quick-label">Reports</span></div>
              <div class="ms-quick-stat"><span class="ms-quick-value">{action_items}</span><span class="ms-quick-label">Actions</span></div>
              <div class="ms-quick-stat"><span class="ms-quick-value">{html.escape(duration)}</span><span class="ms-quick-label">Time Saved</span></div>
            </div>
          </div>
          <div class="ms-dashboard-card">
            <div class="ms-dashboard-head">
              <div class="ms-dashboard-title">Current Processing</div>
              <span class="ms-badge {status_class}">{processing_status}</span>
            </div>
            {queue_html}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="ms-hero">
          <div class="ms-hero-brand">
            <span class="ms-brand-recording"><span></span><span></span><span></span><span></span><span></span></span>
            <span>MeetScribe</span>
          </div>
          <h1>Every meeting deserves clear decisions.</h1>
          <p class="ms-hero-sub">
            Upload a recording or transcript, review speakers, verify the transcript,
            and generate professional Minutes of Meeting in minutes.
          </p>
          <div class="ms-workflow-timeline">
            <span class="ms-workflow-step active">Upload</span>
            <span class="ms-workflow-dot"></span>
            <span class="ms-workflow-step">Identify Speakers</span>
            <span class="ms-workflow-dot"></span>
            <span class="ms-workflow-step">Review Transcript</span>
            <span class="ms-workflow-dot"></span>
            <span class="ms-workflow-step">Generate Report</span>
            <span class="ms-workflow-dot"></span>
            <span class="ms-workflow-step">Export</span>
          </div>
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


def current_speaker_mapping() -> SpeakerMapping:
    mapping = st.session_state.get("speaker_mapping", {})
    return mapping if isinstance(mapping, dict) else {}


def saved_speaker_mapping() -> SpeakerMapping:
    mapping = st.session_state.get("saved_speaker_mapping", {})
    return mapping if isinstance(mapping, dict) else {}


def store_speaker_mapping(mapping: SpeakerMapping) -> None:
    st.session_state.speaker_mapping = dict(mapping)
    st.session_state.saved_speaker_mapping = {
        **saved_speaker_mapping(),
        **dict(mapping),
    }


def source_meeting_metadata(
    result: TranscriptionResult | None,
    *,
    source_file: str = "",
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"source_file": source_file}
    if result is None:
        return metadata

    if result.language_code:
        metadata["language_code"] = result.language_code
        metadata["transcript_language"] = result.language_code

    end_times = [
        segment.end_time_seconds
        for segment in result.segments
        if segment.end_time_seconds is not None
    ]
    if end_times:
        metadata["duration_seconds"] = max(end_times)
    return metadata


def format_duration_for_report(seconds: Any) -> str:
    try:
        seconds_value = float(seconds)
    except (TypeError, ValueError):
        return ""
    total_seconds = max(0, round(seconds_value))
    minutes, remaining_seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"
    return f"{minutes:02d}:{remaining_seconds:02d}"


def default_meeting_title() -> str:
    filename = st.session_state.get("uploaded_filename", "")
    if filename:
        return Path(filename).stem.replace("_", " ").replace("-", " ").strip().title()
    analysis = st.session_state.get("analysis_result")
    if isinstance(analysis, MeetingAnalysisResult) and analysis.summary.title:
        return analysis.summary.title
    return "Meeting Report"


def participants_from_current_context(result: TranscriptionResult | None = None) -> list[str]:
    metadata = st.session_state.get("meeting_metadata", {})
    participant_list = metadata.get("participant_list")
    if isinstance(participant_list, list):
        participants = [str(item).strip() for item in participant_list if str(item).strip()]
        if participants:
            return participants

    mapping = current_speaker_mapping()
    participants = []
    seen = set()
    for label, mapped_name in mapping.items():
        participant = (mapped_name or label).strip()
        if participant and participant.lower() not in seen:
            participants.append(participant)
            seen.add(participant.lower())
    if participants:
        return participants

    transcript_text = st.session_state.get("transcript_text", "")
    if transcript_text:
        for line in transcript_text.splitlines():
            label = line.split("[", 1)[0].split(":", 1)[0].strip()
            if label and label.lower().startswith("speaker") and label.lower() not in seen:
                participants.append(label)
                seen.add(label.lower())
    elif result is not None:
        labels = detect_speaker_labels(result)
        participants = list(labels)
    return participants


def initialize_meeting_info(result: TranscriptionResult | None = None) -> None:
    current = dict(st.session_state.get("meeting_info", {}))
    if st.session_state.get("meeting_info_initialized") and current:
        return

    metadata = st.session_state.get("meeting_metadata", {})
    participants = participants_from_current_context(result)
    info = {
        "meeting_title": current.get("meeting_title") or default_meeting_title(),
        "meeting_date": current.get("meeting_date") or time.strftime("%Y-%m-%d"),
        "meeting_time": current.get("meeting_time", ""),
        "organization": current.get("organization", ""),
        "project_name": current.get("project_name", ""),
        "prepared_by": current.get("prepared_by") or "MeetScribe",
        "participants": current.get("participants") or ", ".join(participants),
        "duration": current.get("duration")
        or metadata.get("meeting_duration")
        or format_duration_for_report(metadata.get("duration_seconds")),
        "source_file": current.get("source_file") or st.session_state.get("uploaded_filename", ""),
    }
    st.session_state.meeting_info = info
    st.session_state.meeting_info_initialized = True
    st.session_state.meeting_info_last_saved = dict(info)
    log_stage(
        "Meeting information",
        "Collected meeting information defaults.",
        meeting_title=info.get("meeting_title"),
        meeting_date=info.get("meeting_date"),
        participants=info.get("participants"),
        duration=info.get("duration"),
    )


def meeting_info_for_export() -> dict[str, str]:
    info = dict(st.session_state.get("meeting_info", {}))
    metadata = st.session_state.get("meeting_metadata", {})
    participants = info.get("participants") or ", ".join(participants_from_current_context())
    duration = (
        info.get("duration")
        or metadata.get("meeting_duration")
        or format_duration_for_report(metadata.get("duration_seconds"))
    )
    source_file = info.get("source_file") or st.session_state.get("uploaded_filename", "")
    return {
        "Meeting Title": info.get("meeting_title", ""),
        "Date": info.get("meeting_date", ""),
        "Time": info.get("meeting_time", ""),
        "Organization / Company": info.get("organization", ""),
        "Project Name": info.get("project_name", ""),
        "Prepared By": info.get("prepared_by", ""),
        "Participants": participants,
        "Attendees": participants,
        "Duration": duration or "",
        "Source File": source_file,
    }


def render_meeting_information_panel(result: TranscriptionResult | None = None) -> None:
    initialize_meeting_info(result)
    info = dict(st.session_state.get("meeting_info", {}))
    st.markdown(
        """
        <div class="ms-premium-section">
          <div class="ms-section-kicker">Meeting Details</div>
          <h3 class="ms-section-heading">Meeting Information</h3>
          <p class="ms-section-subcopy">Review the report header details before generating the final meeting documentation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns(2)
    with col_left:
        meeting_title = st.text_input(
            "Meeting Title",
            value=info.get("meeting_title", ""),
            key="meeting_info_title_input",
        )
        meeting_date = st.text_input(
            "Meeting Date",
            value=info.get("meeting_date", ""),
            key="meeting_info_date_input",
        )
        meeting_time = st.text_input(
            "Meeting Time (optional)",
            value=info.get("meeting_time", ""),
            key="meeting_info_time_input",
        )
        prepared_by = st.text_input(
            "Prepared By",
            value=info.get("prepared_by", ""),
            key="meeting_info_prepared_by_input",
        )
    with col_right:
        organization = st.text_input(
            "Organization / Company (optional)",
            value=info.get("organization", ""),
            key="meeting_info_organization_input",
        )
        project_name = st.text_input(
            "Project Name (optional)",
            value=info.get("project_name", ""),
            key="meeting_info_project_name_input",
        )
        participants = st.text_area(
            "Participants",
            value=info.get("participants", ""),
            height=98,
            key="meeting_info_participants_input",
        )

    updated = {
        **info,
        "meeting_title": meeting_title.strip(),
        "meeting_date": meeting_date.strip(),
        "meeting_time": meeting_time.strip(),
        "organization": organization.strip(),
        "project_name": project_name.strip(),
        "prepared_by": prepared_by.strip(),
        "participants": participants.strip(),
        "source_file": st.session_state.get("uploaded_filename", ""),
    }
    previous = dict(st.session_state.get("meeting_info_last_saved", {}))
    st.session_state.meeting_info = updated
    changed_fields = [
        label
        for key, label in MEETING_INFO_FIELDS
        if str(previous.get(key, "")) != str(updated.get(key, ""))
    ]
    if changed_fields:
        st.session_state.meeting_info_last_saved = dict(updated)
        reset_export_state()
        log_stage(
            "Meeting information",
            "User edited meeting information.",
            fields=", ".join(changed_fields),
        )


def speaker_label(
    segment: TranscriptionSegment,
    mapping: SpeakerMapping | None = None,
) -> str:
    return display_speaker_label(segment, mapping or current_speaker_mapping())


def format_segment(
    segment: TranscriptionSegment,
    mapping: SpeakerMapping | None = None,
) -> str:
    start_time = format_timestamp(segment.start_time_seconds)
    end_time = format_timestamp(segment.end_time_seconds)
    return f"{speaker_label(segment, mapping)} [{start_time} - {end_time}]\n{segment.transcript}"


def format_transcript(
    result: TranscriptionResult,
    mapping: SpeakerMapping | None = None,
) -> str:
    log_stage(
        "Diarization parsing",
        "Formatting transcription result.",
        has_segments=bool(result.segments),
        segment_count=len(result.segments),
        transcript_chars=len(result.transcript or ""),
    )

    if not result.segments:
        return result.transcript.strip()

    return "\n\n".join(format_segment(segment, mapping) for segment in result.segments)


def render_transcript(
    result: TranscriptionResult,
    mapping: SpeakerMapping | None = None,
) -> None:
    log_stage(
        "Transcript rendering",
        "Rendering transcript.",
        has_segments=bool(result.segments),
        segment_count=len(result.segments),
    )

    if result.segments:
        seen: dict[str, str] = {}
        colour_cycle = ["s1", "s2", "s3", "s4"]
        rows_html = ""
        for segment in result.segments:
            label = speaker_label(segment, mapping)
            if label not in seen:
                seen[label] = colour_cycle[len(seen) % len(colour_cycle)]
            cls = seen[label]
            start_time = format_timestamp(segment.start_time_seconds)
            end_time   = format_timestamp(segment.end_time_seconds)
            rows_html += f"""
            <div class="ms-tr-row">
              <div class="ms-tr-left">
                <span class="ms-speaker-badge {cls}">{html.escape(label)}</span>
                <span class="ms-tr-timestamp">{html.escape(start_time)} &ndash; {html.escape(end_time)}</span>
              </div>
              <div class="ms-tr-right">
                <span class="ms-tr-text">{html.escape(segment.transcript)}</span>
              </div>
            </div>"""

        st.markdown(
            f"""<div class="ms-transcript-scroll">{rows_html}</div>""",
            unsafe_allow_html=True,
        )
        return

    st.text_area(
        "Transcript",
        value=result.transcript,
        height=480,
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
            <div class="ms-item-card discussion">
              <h4>Discussion Point</h4>
              <p>{html.escape(item.point)}</p>
              <div class="ms-meta-row">
                <span class="ms-meta">Time: {html.escape(timestamp)}</span>
                <span class="ms-meta">Speakers: {html.escape(speakers or "N/A")}</span>
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
            <div class="ms-item-card decision">
              <h4>Decision</h4>
              <p>{html.escape(item.decision)}</p>
              <div class="ms-meta-row">
                <span class="ms-meta">Confidence: {html.escape(item.confidence)}</span>
                <span class="ms-meta">Owner: {html.escape(owner)}</span>
                <span class="ms-meta">Time: {html.escape(timestamp)}</span>
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
            <div class="ms-item-card action">
              <h4>Task</h4>
              <p>{html.escape(item.task)}</p>
              <div class="ms-meta-row">
                <span class="ms-meta">Owner: {html.escape(owner)}</span>
                <span class="ms-meta">Due: {html.escape(due_date)}</span>
                <span class="ms-meta">Status: {html.escape(item.status)}</span>
                <span class="ms-meta">Time: {html.escape(timestamp)}</span>
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
        meeting_info=meeting_info_for_export(),
    )
    st.session_state[state_key] = str(export_path)
    log_stage(
        "Export",
        "Prepared export document.",
        path=str(export_path),
        meeting_title=st.session_state.get("meeting_info", {}).get("meeting_title", ""),
    )
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


def default_email_subject() -> str:
    meeting_title = (
        st.session_state.get("meeting_info", {}).get("meeting_title")
        or meeting_info_for_export().get("Meeting Title")
        or "Meeting Report"
    )
    return f"Minutes of Meeting - {meeting_title}"


def default_email_message() -> str:
    return (
        "Hello,\n\n"
        "Please find the attached Minutes of Meeting generated using MeetScribe.\n\n"
        "Regards,\n"
        "MeetScribe"
    )


def render_email_form(
    *,
    analysis: MeetingAnalysisResult,
    mom_pdf_path: Path,
) -> None:
    attachment_name = mom_pdf_path.name if mom_pdf_path.is_file() else "minutes_of_meeting.pdf"
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="ms-email-compose">
              <div class="ms-email-compose-hdr">Compose Email</div>
              <p class="ms-email-compose-sub">
                Send the generated Minutes of Meeting PDF using your configured SMTP account.
              </p>
              <div class="ms-attachment-preview">
                <div class="ms-attachment-preview-icon">PDF</div>
                <div>
                  <div class="ms-attachment-preview-name">{html.escape(attachment_name)}</div>
                  <div class="ms-attachment-preview-meta">PDF attached automatically</div>
                </div>
                <span class="ms-attachment-pill" style="margin-left:auto;">PDF Attached</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.form("send_mom_email_form"):
            st.markdown("<div class='ms-email-field-label'>Recipient</div>", unsafe_allow_html=True)
            recipient = st.text_input("Recipient Email", key="email_recipient_input", label_visibility="collapsed")
            st.markdown("<div class='ms-email-field-label'>CC</div>", unsafe_allow_html=True)
            cc = st.text_input("CC (optional)", key="email_cc_input", label_visibility="collapsed")
            st.markdown("<div class='ms-email-field-label'>Subject</div>", unsafe_allow_html=True)
            subject = st.text_input(
                "Subject",
                value=default_email_subject(),
                key="email_subject_input",
                label_visibility="collapsed",
            )
            st.markdown("<div class='ms-email-field-label'>Message</div>", unsafe_allow_html=True)
            message = st.text_area(
                "Message",
                value=default_email_message(),
                height=180,
                key="email_message_input",
                label_visibility="collapsed",
            )
            send_clicked = st.form_submit_button(
                "Send Email",
                type="primary",
                use_container_width=True,
            )

    if not send_clicked:
        return

    log_stage(
        "Email",
        "Email requested.",
        recipient=recipient.strip(),
        cc=cc.strip(),
    )

    try:
        attachment_path = mom_pdf_path
        attachment_existed = attachment_path.is_file()
        if not attachment_existed:
            attachment_path = prepared_export_path(
                "pdf_export_path",
                export_to_pdf,
                analysis,
            )
        log_stage(
            "Email",
            "Attachment reused." if attachment_existed else "Attachment generated.",
            path=str(attachment_path),
        )
        send_report_email(
            recipient=recipient,
            cc=cc,
            subject=subject,
            message=message,
            attachment_path=attachment_path,
        )
    except EmailValidationError as exc:
        log_stage("Email", "Validation failed.", error=str(exc))
        st.warning(str(exc))
        return
    except SMTPConfigurationError as exc:
        log_stage("Email", "SMTP configuration missing or invalid.", error=str(exc))
        st.error(str(exc))
        return
    except EmailDeliveryError as exc:
        log_stage("Email", "Email sending failed.", error=str(exc))
        st.error(str(exc))
        return
    except Exception as exc:
        log_stage("Email", "Unexpected email error.", error=str(exc), traceback=traceback.format_exc())
        st.error("Email could not be sent. Please try again.")
        return

    log_stage("Email", "Email sent successfully.", recipient=recipient.strip())
    st.success("Email sent successfully.")
    st.session_state.show_email_form = False


def render_export_card(analysis: MeetingAnalysisResult) -> None:
    st.markdown(
        """
        <div class="ms-export-section">
          <div class="ms-export-wrap">
            <div class="ms-export-hdr">
              <div class="ms-export-hdr-icon">&#8659;</div>
              <div>
                <div class="ms-export-hdr-title">Export Center</div>
                <p class="ms-export-sub">Download polished MoM documents or send them by email.</p>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    try:
        mom_pdf_path = prepared_export_path("pdf_export_path", export_to_pdf, analysis)
        mom_docx_path = prepared_export_path("docx_export_path", export_to_docx, analysis)
    except Exception as exc:
        log_stage(
            "Export",
            "Could not prepare export documents.",
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        st.error("Downloads could not be prepared. Please try again.")
        return

    pdf_col, docx_col, email_col = st.columns(3)

    with pdf_col:
        st.markdown(
            """
            <div class="ms-export-option">
              <div class="ms-export-option-icon pdf">PDF</div>
              <div class="ms-export-option-title">Download PDF</div>
              <div class="ms-export-option-desc">Print-ready Minutes of Meeting document.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_download_button(
            label="Download PDF",
            export_path=mom_pdf_path,
            mime="application/pdf",
            key="download_mom_pdf",
            success_message="Download started",
        )

    with docx_col:
        st.markdown(
            """
            <div class="ms-export-option">
              <div class="ms-export-option-icon docx">DOC</div>
              <div class="ms-export-option-title">Download DOCX</div>
              <div class="ms-export-option-desc">Editable Word document for your team.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_download_button(
            label="Download DOCX",
            export_path=mom_docx_path,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            key="download_mom_docx",
            success_message="Download started",
        )

    with email_col:
        st.markdown(
            """
            <div class="ms-export-option ms-export-email-wrap">
              <div class="ms-export-option-icon email">✉</div>
              <div class="ms-export-option-title">Send Email</div>
              <div class="ms-export-option-desc">Email the MoM PDF to stakeholders.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(
            "Send Email",
            key="show_send_email_form",
            use_container_width=True,
        ):
            st.session_state.show_email_form = True

    if st.session_state.get("show_email_form", False):
        render_email_form(analysis=analysis, mom_pdf_path=mom_pdf_path)


def run_meeting_analysis(
    transcript_text: str,
    *,
    progress: st.delta_generator.DeltaGenerator | None = None,
    elapsed_placeholder: st.delta_generator.DeltaGenerator | None = None,
    status_placeholder: st.delta_generator.DeltaGenerator | None = None,
    started_at: float | None = None,
    estimate_note: str = "",
) -> MeetingAnalysisResult | None:
    analysis_progress = progress or st.progress(0, text="Generating Meeting Notes")
    cache_key = analysis_cache_key(transcript_text)

    try:
        cached_analysis = st.session_state.analysis_cache.get(cache_key)
        if cached_analysis is not None:
            analysis_progress.progress(75, text="Generating Meeting Notes")
            if status_placeholder is not None and started_at is not None:
                render_stage_status(
                    status_placeholder,
                    active_index=3,
                    started_at=started_at,
                    note="Preparing organized notes for this recording.",
                )
            analysis_progress.progress(90, text="Preparing Exports")
            if status_placeholder is not None and started_at is not None:
                render_stage_status(
                    status_placeholder,
                    active_index=4,
                    started_at=started_at,
                    note="Arranging the report into a clean meeting format.",
                )
            analysis_progress.progress(100, text="Preparing Exports")
            if status_placeholder is not None and started_at is not None:
                render_stage_status(
                    status_placeholder,
                    active_index=len(PROCESSING_STAGES),
                    started_at=started_at,
                    note="Report metrics and downloads are ready.",
                )
            initialize_meeting_info(st.session_state.get("transcript_result"))
            st.session_state.analysis_result = cached_analysis
            st.session_state.analysis_error = ""
            log_stage("Meeting analysis", "Loaded analysis from session cache.")
            log_stage(
                "Report generation",
                "Final meeting report generation completed from cache.",
                meeting_title=st.session_state.get("meeting_info", {}).get("meeting_title", ""),
                key_points=min(len(cached_analysis.key_discussion_points), 5),
                decisions=min(len(cached_analysis.decisions), 5),
                action_items=min(len(cached_analysis.action_items), 7),
            )
            update_elapsed(elapsed_placeholder, started_at)
            return cached_analysis

        log_stage(
            "Meeting analysis",
            "Initializing Gemini analysis pipeline.",
            transcript_chars=len(transcript_text),
        )
        analysis_progress.progress(70, text="Generating Meeting Notes")
        if status_placeholder is not None and started_at is not None:
            render_stage_status(
                status_placeholder,
                active_index=3,
                started_at=started_at,
                note=estimate_note,
            )
        update_elapsed(elapsed_placeholder, started_at)

        gemini_client = get_gemini_client()
        summarizer = LLMSummarizer(llm_client=gemini_client)

        analysis_progress.progress(75, text="Generating Meeting Notes")
        if status_placeholder is not None and started_at is not None:
            render_stage_status(
                status_placeholder,
                active_index=3,
                started_at=started_at,
                note="Organizing the transcript into notes, decisions, and follow-up tasks.",
        )
        update_elapsed(elapsed_placeholder, started_at)
        log_stage("Meeting analysis", "Calling Python meeting workflow.")
        meeting_metadata = {
            **st.session_state.get("meeting_metadata", {}),
            "source_file": st.session_state.get("uploaded_filename", ""),
        }
        analysis = run_meeting_analysis_workflow(
            transcript_text,
            summarizer=summarizer,
            speaker_mapping=current_speaker_mapping(),
            meeting_metadata=meeting_metadata,
        )
        st.session_state.meeting_metadata = meeting_metadata
        initialize_meeting_info(st.session_state.get("transcript_result"))
        current_info = dict(st.session_state.get("meeting_info", {}))
        if not current_info.get("duration") and meeting_metadata.get("meeting_duration"):
            current_info["duration"] = meeting_metadata["meeting_duration"]
        if not current_info.get("participants") and meeting_metadata.get("participant_list"):
            current_info["participants"] = ", ".join(
                str(item).strip()
                for item in meeting_metadata.get("participant_list", [])
                if str(item).strip()
            )
        if current_info != st.session_state.get("meeting_info", {}):
            st.session_state.meeting_info = current_info
        log_stage(
            "Meeting information",
            "Merged metadata into meeting information.",
            duration=st.session_state.get("meeting_info", {}).get("duration", ""),
            participants=st.session_state.get("meeting_info", {}).get("participants", ""),
        )

        analysis_progress.progress(90, text="Preparing Exports")
        if status_placeholder is not None and started_at is not None:
            render_stage_status(
                status_placeholder,
                active_index=4,
                started_at=started_at,
                note="Arranging the report into a clean meeting format.",
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
        log_stage(
            "Report generation",
            "Final meeting report generation completed.",
            meeting_title=st.session_state.get("meeting_info", {}).get("meeting_title", ""),
            key_points=min(len(analysis.key_discussion_points), 5),
            decisions=min(len(analysis.decisions), 5),
            action_items=min(len(analysis.action_items), 7),
        )

        analysis_progress.progress(96, text="Preparing Exports")
        if status_placeholder is not None and started_at is not None:
            render_stage_status(
                status_placeholder,
                active_index=4,
                started_at=started_at,
                note="Preparing your report and download files.",
            )
        update_elapsed(elapsed_placeholder, started_at)
        time.sleep(0.2)
        analysis_progress.progress(100, text="Preparing Exports")
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
        st.session_state.analysis_error = (
            "We could not generate the meeting notes. Please try again."
        )
        log_stage("Meeting analysis", "Analysis failed.", error=str(exc))
        st.error(st.session_state.analysis_error)
        return None
    except Exception as exc:
        if progress is None:
            analysis_progress.empty()
        st.session_state.analysis_result = None
        st.session_state.analysis_error = (
            "Something went wrong while preparing your meeting notes. Please try again."
        )
        log_stage(
            "Meeting analysis",
            "Unexpected analysis error.",
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        st.error(st.session_state.analysis_error)
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
                border: 1px solid rgba(109,93,246,0.30);
                border-radius: 8px;
                background: rgba(109,93,246,0.10);
                color: #A78BFA;
                cursor: pointer;
                font: 600 12px Inter, system-ui, sans-serif;
                padding: 0.38rem 0.75rem;
                letter-spacing: 0.01em;
                transition: border-color 160ms ease, background 160ms ease;
                display: inline-flex; align-items: center; gap: 6px;
            "
            onmouseover="this.style.borderColor='rgba(109,93,246,0.55)';this.style.background='rgba(109,93,246,0.18)';"
            onmouseout="this.style.borderColor='rgba(109,93,246,0.30)';this.style.background='rgba(109,93,246,0.10)';"
            type="button"
        >
            Copy transcript
        </button>
        <span
            id="copy-status"
            style="
                color: #A78BFA;
                font: 12px Inter, system-ui, sans-serif;
                margin-left: 0.6rem;
            "
        ></span>
        <script>
            const button = document.getElementById("copy-transcript");
            const status = document.getElementById("copy-status");
            const transcript = {escaped_transcript};

            button.addEventListener("click", async () => {{
                try {{
                    await navigator.clipboard.writeText(transcript);
                    status.textContent = "Copied!";
                }} catch (error) {{
                    status.textContent = "Copy failed";
                }}

                setTimeout(() => {{
                    status.textContent = "";
                }}, 2200);
            }});
        </script>
        """,
        height=42,
    )


def reset_report_state() -> None:
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
    st.session_state.show_email_form = False


def reset_export_state() -> None:
    st.session_state.docx_export_path = ""
    st.session_state.docx_export_error = ""
    st.session_state.pdf_export_path = ""
    st.session_state.pdf_export_error = ""
    st.session_state.transcript_docx_export_path = ""
    st.session_state.transcript_docx_export_error = ""
    st.session_state.transcript_pdf_export_path = ""
    st.session_state.transcript_pdf_export_error = ""
    st.session_state.show_email_form = False


def reset_speaker_mapping_state() -> None:
    st.session_state.speaker_mapping = {}
    st.session_state.speaker_names_available = False
    st.session_state.speaker_review_required = False
    st.session_state.transcript_review_required = False
    st.session_state.edited_transcript_text = ""


def clear_current_report() -> None:
    st.session_state.transcript_text = ""
    st.session_state.transcript_result = None
    st.session_state.uploaded_filename = ""
    st.session_state.meeting_metadata = {}
    st.session_state.meeting_info = {}
    st.session_state.meeting_info_initialized = False
    st.session_state.meeting_info_last_saved = {}
    reset_speaker_mapping_state()
    reset_report_state()


def render_speaker_review(result: TranscriptionResult) -> None:
    if not st.session_state.get("speaker_review_required", False):
        return

    labels = list(current_speaker_mapping()) or detect_speaker_labels(result)
    if not labels:
        return

    mapping = current_speaker_mapping() or {label: label for label in labels}
    with st.container(border=True):
        st.markdown(
            """
            <div class="ms-section-kicker">Speaker Resolution</div>
            <h3 class="ms-section-heading">Review Speaker Names</h3>
            <p class="ms-section-subcopy">Confirm each speaker label so the transcript and report use clear participant names.</p>
            """,
            unsafe_allow_html=True,
        )
        with st.form("speaker_name_review_form"):
            submitted_values: SpeakerMapping = {}
            for index, label in enumerate(labels):
                initials = "".join(part[0] for part in str(label).split()[:2]).upper()[:2] or "S"
                st.markdown(
                    f"""
                    <div class="ms-speaker-row">
                      <div class="ms-speaker-avatar">{html.escape(initials)}</div>
                      <div>
                        <div class="ms-speaker-label">{html.escape(str(label))}</div>
                        <div class="ms-speaker-badge-text">Editable participant name</div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                value = st.text_input(
                    label,
                    value=mapping.get(label, label),
                    key=f"speaker_name_{index}",
                )
                submitted_values[label] = value

            submitted = st.form_submit_button(
                "Save & Continue",
                type="primary",
                use_container_width=True,
            )

    if not submitted:
        return

    updated_mapping = update_mapping(labels, submitted_values)
    store_speaker_mapping(updated_mapping)
    mapped_result = apply_speaker_resolution(result, updated_mapping)
    transcript_text = format_transcript(mapped_result, {})
    st.session_state.transcript_result = mapped_result
    st.session_state.transcript_text = transcript_text
    st.session_state.meeting_metadata = source_meeting_metadata(
        mapped_result,
        source_file=st.session_state.get("uploaded_filename", ""),
    )
    st.session_state.meeting_info_initialized = False
    initialize_meeting_info(mapped_result)
    st.session_state.speaker_review_required = False
    st.session_state.transcript_review_required = True
    st.session_state.edited_transcript_text = transcript_text
    st.session_state.analysis_result = None
    st.session_state.analysis_error = ""
    st.session_state.success_metrics = None
    reset_export_state()
    log_stage(
        "Speaker mapping",
        "Stored manual speaker mappings.",
        speaker_count=len(updated_mapping),
        mapping=updated_mapping,
    )
    st.rerun()


def transcript_review_preview_html(transcript_text: str, query: str = "") -> str:
    query_normalized = query.strip().lower()
    rows: list[str] = []
    for block in re.split(r"\n\s*\n", transcript_text.strip()):
        block = block.strip()
        if not block:
            continue
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        header = lines[0]
        body = " ".join(lines[1:]).strip()
        haystack = f"{header} {body}".lower()
        if query_normalized and query_normalized not in haystack:
            continue
        timestamp = ""
        speaker = header
        match = re.match(r"^(?P<speaker>.*?)\s*\[(?P<time>[^\]]+)\]", header)
        if match:
            speaker = match.group("speaker").strip() or header
            timestamp = match.group("time").strip()
        rows.append(
            f"""
            <div class="ms-convo-row">
              <div class="ms-convo-head">
                <span class="ms-convo-speaker">{html.escape(speaker)}</span>
                <span class="ms-time-chip">{html.escape(timestamp or "--:--")}</span>
              </div>
              <div class="ms-convo-text">{html.escape(body or header)}</div>
            </div>
            """
        )
        if len(rows) >= 8:
            break

    if not rows:
        return "<div class='ms-empty'>No transcript lines match the current search.</div>"
    return "<div class='ms-conversation-preview'>" + "".join(rows) + "</div>"


def render_editable_transcript_review(result: TranscriptionResult) -> None:
    if st.session_state.get("speaker_review_required", False):
        return
    if not st.session_state.get("transcript_review_required", False):
        return

    transcript_text = (
        st.session_state.get("edited_transcript_text")
        or st.session_state.get("transcript_text", "")
    )
    if not transcript_text:
        return

    with st.container(border=True):
        st.markdown(
            """
            <div class="ms-section-kicker">Transcript Review</div>
            <h3 class="ms-section-heading">Review Transcript</h3>
            <p class="ms-section-subcopy">Search the conversation preview, then make any final edits in the transcript editor below.</p>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div class='ms-transcript-toolbar'>", unsafe_allow_html=True)
        transcript_search = st.text_input(
            "Search transcript",
            placeholder="Search speaker, timestamp, or phrase...",
            key="transcript_review_search",
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            transcript_review_preview_html(transcript_text, transcript_search),
            unsafe_allow_html=True,
        )
        edited_text = st.text_area(
            "Resolved transcript",
            value=transcript_text,
            height=360,
            key="editable_transcript_text_area",
            label_visibility="collapsed",
        )
        render_meeting_information_panel(result)
        continue_clicked = st.button(
            "Generate Meeting Report",
            type="primary",
            use_container_width=True,
            key="continue_to_analysis",
        )

    if not continue_clicked:
        return

    started_at = time.perf_counter()
    edited_text = edited_text.strip()
    if not edited_text:
        st.warning("Transcript cannot be empty.")
        return

    previous_text = st.session_state.get("transcript_text", "")
    edited_result = apply_transcript_edits(result, edited_text)
    edited_mapping = speaker_mapping_from_segments(edited_result)
    if edited_mapping:
        st.session_state.speaker_mapping = edited_mapping
    st.session_state.transcript_result = edited_result
    st.session_state.transcript_text = edited_text
    st.session_state.edited_transcript_text = edited_text
    st.session_state.transcript_review_required = False
    st.session_state.analysis_result = None
    st.session_state.analysis_error = ""
    st.session_state.success_metrics = None
    st.session_state.meeting_metadata = source_meeting_metadata(
        edited_result,
        source_file=st.session_state.get("uploaded_filename", ""),
    )
    current_info = dict(st.session_state.get("meeting_info", {}))
    if not current_info.get("duration"):
        current_info["duration"] = format_duration_for_report(
            st.session_state.meeting_metadata.get("duration_seconds")
        )
    current_info["source_file"] = st.session_state.get("uploaded_filename", "")
    st.session_state.meeting_info = current_info
    reset_export_state()

    log_stage(
        "Transcript editing",
        "Transcript confirmed for analysis.",
        edited=edited_text != previous_text,
        previous_chars=len(previous_text),
        edited_chars=len(edited_text),
        segment_count=len(edited_result.segments),
    )

    analysis = run_meeting_analysis(
        edited_text,
        started_at=started_at,
        estimate_note="Generating summary, discussion points, decisions, and action items.",
    )
    if analysis is not None:
        store_success_metrics(
            result=edited_result,
            analysis=analysis,
            started_at=started_at,
        )
    st.rerun()


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
    progress = st.progress(0, text="Uploading Recording")
    clear_current_report()

    try:
        log_stage(
            "File upload",
            "Received uploaded file.",
            filename=getattr(uploaded_file, "name", ""),
            size=getattr(uploaded_file, "size", None),
            type=getattr(uploaded_file, "type", ""),
        )

        progress.progress(10, text="Uploading Recording")
        render_stage_status(
            status_placeholder,
            active_index=0,
            started_at=started_at,
            note="Receiving your recording and getting it ready.",
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

        progress.progress(35, text="Preparing Audio")
        render_stage_status(
            status_placeholder,
            active_index=1,
            started_at=started_at,
            note="Preparing the recording for speaker identification.",
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

        progress.progress(60, text="Identifying Speakers")
        render_stage_status(
            status_placeholder,
            active_index=2,
            started_at=started_at,
            note="Organizing the transcript by speaker.",
        )
        resolution = resolve_speakers(
            result,
            previous_mapping=saved_speaker_mapping(),
        )
        speaker_mapping = resolution.mapping
        session_result = (
            result
            if resolution.review_required
            else apply_speaker_resolution(result, speaker_mapping)
        )
        transcript_text = format_transcript(session_result, speaker_mapping)
        if not transcript_text.strip():
            raise ValueError("Formatted transcript is empty.")

        st.session_state.transcript_result = session_result
        st.session_state.transcript_text = transcript_text
        st.session_state.uploaded_filename = getattr(uploaded_file, "name", "")
        st.session_state.meeting_metadata = source_meeting_metadata(
            session_result,
            source_file=st.session_state.uploaded_filename,
        )
        if resolution.review_required:
            st.session_state.speaker_mapping = speaker_mapping
        else:
            store_speaker_mapping(speaker_mapping)
        st.session_state.speaker_names_available = resolution.names_available
        st.session_state.speaker_review_required = resolution.review_required
        st.session_state.transcript_review_required = not resolution.review_required
        st.session_state.edited_transcript_text = transcript_text
        log_stage(
            "Speaker mapping",
            "Resolved speakers for audio transcript.",
            detected=resolution.detected_speakers,
            automatic=resolution.names_available,
            reused=resolution.reused_labels,
            review_required=resolution.review_required,
        )
        log_stage(
            "Session state update",
            "Stored transcript in session state.",
            transcript_chars=len(transcript_text),
            segment_count=len(session_result.segments),
            speaker_review_required=st.session_state.speaker_review_required,
        )

        if st.session_state.speaker_review_required:
            progress.progress(100, text="Review Speakers")
            render_stage_status(
                status_placeholder,
                active_index=2,
                started_at=started_at,
                note="Review speaker names before generating meeting notes.",
            )
            st.toast("Transcript is ready for speaker review")
            return

        progress.progress(100, text="Review Transcript")
        render_stage_status(
            status_placeholder,
            active_index=2,
            started_at=started_at,
            note="Review the resolved transcript before generating meeting notes.",
        )
        st.toast("Transcript is ready for review")
        return
    except (AudioProcessingError, SettingsError, TranscriptionError) as exc:
        progress.empty()
        log_stage("Error", "Pipeline error.", error=str(exc))
        st.error(
            "We could not prepare this recording. Please check the file and try again."
        )
    except Exception as exc:
        progress.empty()
        log_stage(
            "Error",
            "Unexpected error while processing audio.",
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        st.error("Something went wrong while preparing your report. Please try again.")
    finally:
        if prepared_path is not None:
            prepared_path.unlink(missing_ok=True)
            log_stage("Cleanup", "Deleted temporary WAV file.", path=str(prepared_path))


def process_transcript_upload(transcript_file: object) -> None:
    started_at = time.perf_counter()
    status_placeholder = st.empty()
    estimate_note = "Reading your transcript and preparing the meeting report."
    render_stage_status(
        status_placeholder,
        active_index=0,
        started_at=started_at,
        note=estimate_note,
    )
    progress = st.progress(0, text="Uploading Transcript")
    clear_current_report()

    try:
        log_stage(
            "Transcript upload",
            "Received uploaded transcript.",
            filename=getattr(transcript_file, "name", ""),
            size=getattr(transcript_file, "size", None),
            type=getattr(transcript_file, "type", ""),
        )

        progress.progress(20, text="Preparing File")
        render_stage_status(
            status_placeholder,
            active_index=1,
            started_at=started_at,
            note="Extracting text from the uploaded transcript.",
        )
        extracted = extract_uploaded_transcript(
            transcript_file,
            filename=getattr(transcript_file, "name", None),
        )
        result = validate_transcription_result(extracted.result)
        resolution = resolve_speakers(
            result,
            previous_mapping=saved_speaker_mapping(),
        )
        speaker_mapping = resolution.mapping
        session_result = (
            result
            if resolution.review_required
            else apply_speaker_resolution(result, speaker_mapping)
        )
        transcript_text = format_transcript(session_result, speaker_mapping)
        if not transcript_text.strip():
            raise TranscriptFileError("No transcript text was found in the uploaded file.")

        st.session_state.transcript_result = session_result
        st.session_state.transcript_text = transcript_text
        st.session_state.uploaded_filename = getattr(transcript_file, "name", "")
        st.session_state.meeting_metadata = source_meeting_metadata(
            session_result,
            source_file=st.session_state.uploaded_filename,
        )
        if resolution.review_required:
            st.session_state.speaker_mapping = speaker_mapping
        else:
            store_speaker_mapping(speaker_mapping)
        st.session_state.speaker_names_available = resolution.names_available
        st.session_state.speaker_review_required = resolution.review_required
        st.session_state.transcript_review_required = not resolution.review_required
        st.session_state.edited_transcript_text = transcript_text
        log_stage(
            "Speaker mapping",
            "Resolved speakers for uploaded transcript.",
            detected=resolution.detected_speakers,
            automatic=resolution.names_available,
            reused=resolution.reused_labels,
            review_required=resolution.review_required,
        )
        log_stage(
            "Session state update",
            "Stored uploaded transcript in session state.",
            transcript_chars=len(transcript_text),
            segment_count=len(session_result.segments),
            speaker_names_available=resolution.names_available,
        )

        if st.session_state.speaker_review_required:
            progress.progress(100, text="Review Speakers")
            render_stage_status(
                status_placeholder,
                active_index=2,
                started_at=started_at,
                note="Review speaker names before generating meeting notes.",
            )
            st.toast("Transcript is ready for speaker review")
            return

        progress.progress(100, text="Review Transcript")
        render_stage_status(
            status_placeholder,
            active_index=2,
            started_at=started_at,
            note="Review the resolved transcript before generating meeting notes.",
        )
        st.toast("Transcript is ready for review")
        return
    except TranscriptFileError as exc:
        progress.empty()
        log_stage("Transcript upload", "Transcript validation failed.", error=str(exc))
        st.error(str(exc))
    except SettingsError as exc:
        progress.empty()
        log_stage("Transcript upload", "Settings error.", error=str(exc))
        st.error("Meeting notes could not be generated because the app is not configured correctly.")
    except Exception as exc:
        progress.empty()
        log_stage(
            "Transcript upload",
            "Unexpected error while processing transcript.",
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        st.error("Something went wrong while reading this transcript. Please try another file.")


def main() -> None:
    st.set_page_config(page_title="MeetScribe", layout="wide")
    initialize_session_state()
    inject_processing_styles()

    render_sidebar_shell()
    render_hero()

    # ── UPLOAD PANEL ──────────────────────────────────────────────
    with st.container(border=True):
        st.markdown(
            """
            <div class="ms-upload-title-row">
              <div class="ms-upload-icon-badge">&#9729;</div>
              <span class="ms-upload-title">Upload Input</span>
            </div>
            <p class="ms-upload-desc">
              Upload a meeting recording or a transcript to generate a structured meeting report.
            </p>
            """,
            unsafe_allow_html=True,
        )

        left_col, mid_col, right_col = st.columns([1, 0.08, 1])

        with left_col:
            with st.container(border=True):
                st.markdown(
                    "<p class='ms-sub-label'>Upload Meeting Recording</p>"
                    "<p class='ms-sub-fmt'>Supported formats: MP3, WAV, M4A, AAC, MP4</p>",
                    unsafe_allow_html=True,
                )
                uploaded_file = st.file_uploader(
                    "Drag and drop your audio file here, or click to browse",
                    type=SUPPORTED_FILE_TYPES,
                    accept_multiple_files=False,
                    label_visibility="collapsed",
                    key=f"audio_upload_{st.session_state.audio_upload_version}",
                )

                if uploaded_file is not None:
                    file_size_mb = uploaded_file.size / (1024 * 1024)
                    upload_signature = f"{uploaded_file.name}:{uploaded_file.size}"
                    if st.session_state.last_logged_upload != upload_signature:
                        log_stage("File upload", "File selected in UI.",
                                  filename=uploaded_file.name, size=uploaded_file.size)
                        st.session_state.last_logged_upload = upload_signature

                    file_col, remove_col = st.columns([1, 0.16], gap="small", vertical_alignment="center")
                    with file_col:
                        st.markdown(
                            f"""<div class="ms-file-card">
                                    <div class="ms-file-icon">
                                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg>
                                    </div>
                                  <div class="ms-file-info">
                                    <div class="ms-file-name">{html.escape(uploaded_file.name)}</div>
                                    <div class="ms-file-size">{file_size_mb:.2f} MB</div>
                                  </div>
                                </div>""",
                            unsafe_allow_html=True,
                        )
                    with remove_col:
                        if st.button("🗑", key="remove_audio_file", type="secondary", help="Remove file"):
                            st.session_state.audio_upload_version += 1
                            st.session_state.last_logged_upload = ""
                            clear_current_report()
                            st.rerun()

        with mid_col:
            st.markdown(
                "<div class='ms-or-wrap'><span class='ms-or-divider'>OR</span></div>",
                unsafe_allow_html=True,
            )

        with right_col:
            with st.container(border=True):
                st.markdown(
                    "<p class='ms-sub-label'>Upload Transcript</p>"
                    "<p class='ms-sub-fmt'>Supported formats: PDF, DOCX, TXT</p>",
                    unsafe_allow_html=True,
                )
                transcript_file = st.file_uploader(
                    "Drag and drop your transcript here, or click to browse",
                    type=list(SUPPORTED_TRANSCRIPT_TYPES),
                    accept_multiple_files=False,
                    label_visibility="collapsed",
                    key=f"transcript_upload_{st.session_state.transcript_upload_version}",
                )

                if transcript_file is not None:
                    tf_size_mb = transcript_file.size / (1024 * 1024)
                    file_col, remove_col = st.columns([1, 0.16], gap="small", vertical_alignment="center")
                    with file_col:
                        st.markdown(
                            f"""<div class="ms-file-card">
                                    <div class="ms-file-icon">
                                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
                                    </div>
                                  <div class="ms-file-info">
                                    <div class="ms-file-name">{html.escape(transcript_file.name)}</div>
                                    <div class="ms-file-size">{tf_size_mb:.2f} MB</div>
                                  </div>
                                </div>""",
                            unsafe_allow_html=True,
                        )
                    with remove_col:
                        if st.button("🗑", key="remove_transcript_file", type="secondary", help="Remove file"):
                            st.session_state.transcript_upload_version += 1
                            clear_current_report()
                            st.rerun()

        process_clicked = st.button(
            "Upload and Prepare Review",
            type="primary",
            disabled=uploaded_file is None and transcript_file is None,
            use_container_width=True,
        )

        if process_clicked:
            if transcript_file is not None:
                process_transcript_upload(transcript_file)
            elif uploaded_file is not None:
                process_upload(uploaded_file)

    has_session_input = (
        uploaded_file is not None
        or transcript_file is not None
        or bool(st.session_state.get("transcript_text"))
    )
    if not has_session_input:
        render_empty_state()

    # ── POST-PROCESSING RESULTS ───────────────────────────────────
    transcript_text = st.session_state.transcript_text
    result          = st.session_state.transcript_result
    analysis        = st.session_state.analysis_result
    log_stage(
        "Transcript rendering", "Checking transcript display conditions.",
        has_transcript=bool(transcript_text),
        result_type=type(result).__name__ if result is not None else None,
        analysis_type=type(analysis).__name__ if analysis is not None else None,
    )

    if transcript_text and result is not None:
        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

        render_speaker_review(result)
        render_editable_transcript_review(result)

        if analysis is not None:
            st.markdown(
                """
                <div class="ms-premium-section">
                  <div class="ms-section-kicker">Professional Report</div>
                  <h3 class="ms-section-heading">Meeting Report</h3>
                  <p class="ms-section-subcopy">
                    Summary, discussion points, decisions, action items, and transcript —
                    generated from your reviewed meeting.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_analysis_error()
            render_success_metrics()

            with st.container(border=True):
                st.markdown("<div class='ms-report-block-title'>Summary</div>", unsafe_allow_html=True)
                render_summary_tab(analysis)

            with st.container(border=True):
                st.markdown("<div class='ms-report-block-title'>Discussion</div>", unsafe_allow_html=True)
                render_key_points_tab(analysis)

            with st.container(border=True):
                st.markdown("<div class='ms-report-block-title'>Decisions</div>", unsafe_allow_html=True)
                render_decisions_tab(analysis)

            with st.container(border=True):
                st.markdown("<div class='ms-report-block-title'>Action Items</div>", unsafe_allow_html=True)
                render_action_items_tab(analysis)

            with st.container(border=True):
                st.markdown("<div class='ms-report-block-title'>Transcript</div>", unsafe_allow_html=True)
                render_copy_button(transcript_text)
                render_transcript(result, current_speaker_mapping())

            render_export_card(analysis)

    st.markdown(
        "<div class='ms-footer'>&#169; 2026 MeetScribe. All rights reserved.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
