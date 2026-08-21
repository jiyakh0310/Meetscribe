"""Streamlit UI for MeetScribe audio transcription — refined UI."""

from __future__ import annotations

import base64
import io
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

BRAND_LOGO_PATH = Path(__file__).resolve().parent / "assets" / "meetscribe-logo.png"
MEETING_RECORDER_COMPONENT = components.declare_component(
    "meeting_recorder",
    path=str(PROJECT_ROOT / "components" / "meeting_recorder" / "frontend"),
)


def brand_logo_html(class_name: str, *, alt: str = "MeetScribe") -> str:
    """Return the official transparent logo as presentation-only inline HTML."""

    encoded = base64.b64encode(BRAND_LOGO_PATH.read_bytes()).decode("ascii")
    return (
        f'<img class="{html.escape(class_name)}" '
        f'src="data:image/png;base64,{encoded}" alt="{html.escape(alt)}">'
    )

from config.settings import OLLAMA_GEMMA_MODEL, USE_LOCAL_GEMMA, SettingsError
from exports.docx_exporter import export_to_docx
from exports.email_sender import (
    EmailDeliveryError,
    EmailValidationError,
    SMTPConfigurationError,
    send_report_email,
)
from exports.pdf_exporter import export_to_pdf
from ml_mom.experimental.integration import GeneratedMomResult, generate_mom
from ml_mom.mom_generator import MeetingMinutes, PredictionRecord
from ml_mom.predict_ann import PredictionResult
from ml_mom.transcript_parser import TranscriptTurn, parse_transcript
from ml_ner.entity_extractor import EntityExtractionResult, extract_meeting_metadata
from summarization.base_summarizer import (
    ActionItem,
    Decision,
    KeyDiscussionPoint,
    MeetingAnalysisResult,
    MeetingSummary,
)
from transcription.audio_utils import AudioProcessingError, preprocess_uploaded_audio
from transcription.audio_transcript_normalization import (
    coalesce_contiguous_audio_segments,
)
from transcription.audio_transcript_repair import repair_audio_transcription
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


def log_runtime_environment() -> None:
    """Log the Python runtime used by Streamlit at application startup.

    The report-generation stack depends on native ML packages. When Streamlit is
    launched outside the project virtual environment, Windows may resolve
    packages from a different Python installation. Logging the runtime here
    makes that visible without changing any application behavior.
    """

    logger.info("Runtime sys.executable: %s", sys.executable)
    logger.info("Runtime sys.prefix: %s", sys.prefix)
    logger.info("Runtime sys.path: %s", sys.path)


log_runtime_environment()
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
    st.session_state.setdefault("workflow_open", False)
    st.session_state.setdefault("workflow_source", "audio")
    st.session_state.setdefault("workflow_stage", "upload")
    st.session_state.setdefault("workflow_file", None)
    st.session_state.setdefault("workflow_pending_action", "")
    st.session_state.setdefault("recording_meeting_state", None)
    st.session_state.setdefault("recording_meeting_version", 0)
    st.session_state.setdefault("recording_meeting_open", False)


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
            padding-top: 1.1rem !important;
          }
          [data-testid="stSidebarHeader"] {
            padding-top: 0 !important;
            padding-bottom: 0 !important;
            min-height: 0 !important;
            height: 0 !important;
            overflow: hidden !important;
          }
          [data-testid="stSidebarUserContent"],
          [data-testid="stSidebarContent"] {
            padding-top: 0 !important;
          }
          .ms-side-shell {
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
            padding: 0 0.15rem 0.75rem;
            margin-top: -5px;
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
          .ms-side-brand-logo { width: 40px; height: 40px; object-fit: contain; background: transparent; border: 0; }
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
            justify-content: flex-start;
            gap: 0.65rem;
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
            display: none;
          }

          /* ══ PANELS ══════════════════════════════════ */
          div[data-testid="stVerticalBlockBorderWrapper"] {
            border: 1.35px solid #F8C7D2 !important;
            border-radius: 20px !important;
            background: linear-gradient(180deg,#FFFFFF 0%,#FFFBFE 100%) !important;
            box-shadow: 0 8px 24px rgba(251,113,133,0.075) !important;
            backdrop-filter: none !important;
            transition: border-color 220ms ease, box-shadow 220ms ease, transform 220ms ease, background 220ms ease;
          }
          div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            border-color: var(--pink-mid) !important;
            box-shadow: 0 14px 34px rgba(251,113,133,0.12) !important;
            transform: translateY(-1px);
          }

          /* ══ UPLOAD PANEL ═════════════════════════════ */
          .ms-upload-card {
            border: 1.35px solid #F8C7D2;
            border-radius: 20px;
            background: var(--surface);
            padding: 1.5rem 1.5rem 1.25rem;
            margin-bottom: 1rem;
            box-shadow: 0 8px 24px rgba(251,113,133,0.075);
            animation: fadein 0.35s ease both;
            transition: border-color 220ms, box-shadow 220ms, transform 220ms;
          }
          .ms-upload-card:hover {
            border-color: var(--pink-mid);
            box-shadow: 0 14px 34px rgba(251,113,133,0.12);
            transform: translateY(-1px);
          }

          .ms-dashboard-grid {
            display: grid;
            grid-template-columns: 1.15fr 0.9fr 1fr;
            gap: 1rem;
            margin: 1rem 0 1.2rem;
          }
          .ms-dashboard-card {
            border: 1.35px solid #F8C7D2;
            border-radius: 20px;
            background: linear-gradient(180deg,#FFFFFF 0%,#FFFBFE 100%);
            box-shadow: 0 8px 24px rgba(251,113,133,0.075);
            padding: 1.05rem;
            min-width: 0;
            transition: border-color 220ms ease, box-shadow 220ms ease, transform 220ms ease;
          }
          .ms-dashboard-card:hover {
            border-color: var(--pink-mid);
            box-shadow: 0 14px 34px rgba(251,113,133,0.12);
            transform: translateY(-1px);
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
            border: 1.25px solid #F8C7D2;
            border-radius: 16px;
            background: var(--warm-6);
            padding: 0.65rem 0.45rem;
            text-align: center;
            min-height: 72px;
            box-shadow: 0 5px 14px rgba(251,113,133,0.055);
            transition: border-color 200ms ease, box-shadow 200ms ease, transform 200ms ease;
          }
          .ms-quick-stat:hover {
            border-color: var(--pink-mid);
            box-shadow: 0 9px 20px rgba(251,113,133,0.095);
            transform: translateY(-1px);
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
            border: 1.35px solid #F8C7D2;
            border-radius: 20px;
            background: linear-gradient(180deg,#FFFFFF 0%,#FFFBFE 100%);
            box-shadow: 0 8px 24px rgba(251,113,133,0.075);
            padding: 1.05rem;
            margin: 0.9rem 0 1rem;
            transition: border-color 220ms ease, box-shadow 220ms ease, transform 220ms ease;
          }
          .ms-premium-section:hover {
            border-color: var(--pink-mid);
            box-shadow: 0 14px 34px rgba(251,113,133,0.12);
            transform: translateY(-1px);
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
            line-height: 1.5;
            margin: 0 0 0.72rem;
          }
          .ms-speaker-row {
            display: grid;
            grid-template-columns: 34px minmax(0,1fr);
            gap: 0.65rem;
            align-items: center;
            padding: 0.48rem 0.58rem;
            border: 1.25px solid #F8C7D2;
            border-radius: 14px;
            background: #FFFBFE;
            margin-bottom: 0.38rem;
            box-shadow: 0 5px 14px rgba(251,113,133,0.055);
            transition: border-color 200ms ease, box-shadow 200ms ease, transform 200ms ease;
          }
          .ms-speaker-row:hover {
            border-color: var(--pink-mid);
            box-shadow: 0 9px 20px rgba(251,113,133,0.095);
            transform: translateY(-1px);
          }
          .ms-speaker-avatar {
            width: 34px;
            height: 34px;
            border-radius: 11px;
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
            font-size: 0.8rem;
            line-height: 1.2;
          }
          .ms-speaker-badge-text {
            color: var(--warm-4) !important;
            font-size: 0.64rem;
            margin-top: 0.08rem;
          }
          .ms-transcript-toolbar {
            position: sticky;
            top: 0;
            z-index: 4;
            border: 1.25px solid #F8C7D2;
            border-radius: 16px;
            background: rgba(255,255,255,0.96);
            padding: 0.75rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 8px 24px rgba(251,113,133,0.075);
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
            border: 1.25px solid #F8C7D2;
            border-radius: 16px;
            background: #FFFFFF;
            padding: 0.85rem 0.95rem;
            box-shadow: 0 8px 24px rgba(251,113,133,0.075);
            transition: border-color 200ms ease, box-shadow 200ms ease, transform 200ms ease;
          }
          .ms-convo-row:hover {
            border-color: var(--pink-mid);
            box-shadow: 0 14px 34px rgba(251,113,133,0.12);
            transform: translateY(-1px);
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
            box-shadow: 0 8px 24px rgba(251,113,133,0.075);
          }
          .ms-output-card {
            border-color: #F8C7D2;
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
            border: 1.35px solid #F8C7D2;
            border-radius: 20px;
            background: linear-gradient(180deg,#FFFFFF 0%,#FFFBFE 100%);
            padding: 1.05rem;
            box-shadow: 0 8px 24px rgba(251,113,133,0.075);
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
          /* Row: file card takes full width; delete button column is absolutely overlaid inside */
          div[data-testid="stHorizontalBlock"]:has(.ms-file-card) {
            display: flex !important;
            align-items: center !important;
            flex-wrap: nowrap !important;
            gap: 0 !important;
            margin-top: 0.75rem !important;
            width: 100% !important;
            position: relative !important;
          }
          div[data-testid="stHorizontalBlock"]:has(.ms-file-card)
            > div[data-testid="column"] [data-testid="stElementContainer"],
          div[data-testid="stHorizontalBlock"]:has(.ms-file-card)
            > div[data-testid="column"] [data-testid="stVerticalBlock"] {
            margin: 0 !important;
            padding: 0 !important;
            gap: 0 !important;
          }
          /* Delete button column: absolutely positioned, right-aligned inside the card */
          div[data-testid="stHorizontalBlock"]:has(.ms-file-card)
            > div[data-testid="column"]:has(button[data-testid="stBaseButton-secondary"]) {
            position: absolute !important;
            right: 14px !important;
            top: 50% !important;
            transform: translateY(-50%) !important;
            width: 36px !important;
            min-width: 36px !important;
            flex: 0 0 36px !important;
            z-index: 3 !important;
          }
          /* File card column: takes full width */
          div[data-testid="stHorizontalBlock"]:has(.ms-file-card)
            > div[data-testid="column"]:not(:has(button[data-testid="stBaseButton-secondary"])) {
            flex: 1 1 100% !important;
            width: 100% !important;
          }
          div[data-testid="stHorizontalBlock"]:has(.ms-file-card)
            > div[data-testid="column"]:has(button[data-testid="stBaseButton-secondary"])
            .stButton {
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
          }
          div[data-testid="stHorizontalBlock"]:has(.ms-file-card)
            > div[data-testid="column"]:has(button[data-testid="stBaseButton-secondary"])
            .stButton > button {
            width: 36px !important; min-width: 36px !important;
            height: 36px !important; min-height: 36px !important;
            flex: 0 0 36px !important; transform: none !important;
          }

          .ms-file-card {
            display: flex; align-items: center; gap: 0.65rem;
            border: 1px solid #FAD4DC;
            border-radius: 12px;
            background: #FFF1F5;
            padding: 0 3.75rem 0 1rem;
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

          /* Uploaded-file delete button — circular X inside the card */
          div[data-testid="stHorizontalBlock"]:has(.ms-file-card)
            .stButton > button[data-testid="stBaseButton-secondary"],
          div[data-testid="stHorizontalBlock"]:has(.ms-file-card)
            .stButton > button[kind="secondary"] {
            width: 36px !important; min-width: 36px !important;
            height: 36px !important; min-height: 36px !important;
            max-height: 36px !important; padding: 0 !important; margin: 0 !important;
            border: 1.5px solid #FAD4DC !important;
            border-radius: 50% !important;
            background: #FFFFFF !important;
            background-color: #FFFFFF !important;
            color: var(--pink) !important;
            font-family: Inter, sans-serif !important;
            font-size: 0 !important; font-weight: 700 !important;
            display: flex !important; align-items: center !important;
            justify-content: center !important;
            box-shadow: 0 1px 4px rgba(251,113,133,0.12) !important;
            transition: background 150ms, border-color 150ms, box-shadow 150ms !important;
          }
          div[data-testid="stHorizontalBlock"]:has(.ms-file-card)
            .stButton > button[data-testid="stBaseButton-secondary"]:hover,
          div[data-testid="stHorizontalBlock"]:has(.ms-file-card)
            .stButton > button[kind="secondary"]:hover {
            background: #FFF0F3 !important;
            background-color: #FFF0F3 !important;
            border-color: var(--pink) !important;
            box-shadow: 0 3px 10px rgba(251,113,133,0.22) !important;
          }
          div[data-testid="stHorizontalBlock"]:has(.ms-file-card)
            .stButton > button[data-testid="stBaseButton-secondary"] p,
          div[data-testid="stHorizontalBlock"]:has(.ms-file-card)
            .stButton > button[kind="secondary"] p {
            font-size: 0 !important; width: 16px !important; height: 16px !important;
            background-color: var(--pink) !important;
            -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='18' y1='6' x2='6' y2='18'/%3E%3Cline x1='6' y1='6' x2='18' y2='18'/%3E%3C/svg%3E") !important;
            mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='18' y1='6' x2='6' y2='18'/%3E%3Cline x1='6' y1='6' x2='18' y2='18'/%3E%3C/svg%3E") !important;
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

          div[data-testid="stProgress"] > div {
            background: #FFF1F5 !important;
            border-radius: 999px !important;
            overflow: hidden !important;
            box-shadow: inset 0 0 0 1px #FAD4DC !important;
          }
          div[data-testid="stProgress"] div[role="progressbar"] {
            background: linear-gradient(90deg, #FAD4DC 0%, #FB7185 50%, #FBCFE8 100%) !important;
            border-radius: 999px !important;
            transition: width 220ms ease, background 220ms ease !important;
          }
          div[data-testid="stProgress"] > div > div > div > div {
            background: linear-gradient(90deg, #FAD4DC 0%, #FB7185 50%, #FBCFE8 100%) !important;
            border-radius: 999px !important;
            transition: width 220ms ease, background 220ms ease !important;
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
            border: 1.35px solid #F8C7D2; border-radius: 18px;
            background: linear-gradient(180deg,#FFFFFF 0%,#FFFBFE 100%); padding: 1.1rem 1.1rem 0.95rem;
            box-shadow: 0 8px 24px rgba(251,113,133,0.075);
            transition: border-color 220ms, transform 220ms, box-shadow 220ms;
          }
          .ms-stat-card:hover {
            transform: translateY(-2px); box-shadow: 0 14px 34px rgba(251,113,133,0.12);
            border-color: var(--pink-mid);
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
            border: 1px solid #FAD4DC;
            border-radius: 16px;
            background: #FFFFFF;
            padding: 0 !important;
            box-shadow: 0 4px 16px rgba(251,113,133,0.06);
            transition: border-color 180ms, box-shadow 180ms, transform 180ms, background 180ms;
            min-height: 178px;
            overflow: hidden;
            position: relative;
          }
          .ms-export-section ~ div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(1) {
            border-color: #FAD4DC;
            background: linear-gradient(180deg, #FFFFFF 0%, #FFF7FA 100%);
          }
          .ms-export-section ~ div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) {
            border-color: #DDD6FE;
            background: linear-gradient(180deg, #FFFFFF 0%, #FAF7FF 100%);
          }
          .ms-export-section ~ div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3) {
            border-color: #A7F3D0;
            background: linear-gradient(180deg, #FFFFFF 0%, #F0FDF4 100%);
          }
          .ms-export-section ~ div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:hover {
            border-color: var(--pink-mid);
            box-shadow: var(--shadow-md);
            transform: translateY(-2px);
          }
          .ms-export-option {
            border: 0;
            border-radius: 16px 16px 0 0;
            background: transparent;
            padding: 1.15rem 1.15rem 0.35rem;
            box-shadow: none;
            min-height: 118px;
            pointer-events: none;
          }
          .ms-export-option:hover {
            box-shadow: none;
            transform: none;
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
          .ms-export-section ~ div[data-testid="stHorizontalBlock"] > div[data-testid="column"] .stDownloadButton,
          .ms-export-section ~ div[data-testid="stHorizontalBlock"] > div[data-testid="column"] .stButton {
            position: absolute !important;
            inset: 0 !important;
            z-index: 4 !important;
            margin: 0 !important;
            padding: 0 !important;
          }
          .ms-export-section ~ div[data-testid="stHorizontalBlock"] > div[data-testid="column"] .stDownloadButton > button,
          .ms-export-section ~ div[data-testid="stHorizontalBlock"] > div[data-testid="column"] .stButton > button {
            min-height: 100% !important;
            height: 100% !important;
            border-radius: 16px !important;
            font-size: 0.82rem !important;
            width: 100% !important;
            justify-content: center !important;
            opacity: 0 !important;
            cursor: pointer !important;
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
            height: 560px; overflow-y: auto; margin-top: 0;
            display: flex; flex-direction: column; gap: 0.7rem;
            padding: 0 0.35rem 0.35rem 0;
            scroll-behavior: smooth;
            scrollbar-width: thin; scrollbar-color: var(--warm-4) transparent;
          }
          .ms-transcript-scroll::-webkit-scrollbar { width: 4px; }
          .ms-transcript-scroll::-webkit-scrollbar-track { background: transparent; }
          .ms-transcript-scroll::-webkit-scrollbar-thumb {
            background: var(--warm-4); border-radius: 999px;
          }

          .ms-transcript-sticky-head {
            position: sticky;
            top: 0;
            z-index: 3;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.75rem;
            padding: 0.75rem 0.9rem;
            border: 1.35px solid #F8C7D2;
            border-radius: 18px;
            background: rgba(255, 251, 254, 0.96);
            backdrop-filter: blur(10px);
            box-shadow: 0 10px 24px rgba(251,113,133,0.09);
          }
          .ms-transcript-sticky-head span:first-child {
            color: var(--warm) !important;
            font-size: 0.82rem;
            font-weight: 850;
          }
          .ms-transcript-sticky-head span:last-child {
            color: var(--warm-4) !important;
            font-size: 0.68rem;
            font-weight: 750;
          }

          .ms-tr-row {
            display: grid; grid-template-columns: 150px minmax(0, 1fr);
            border: 1.35px solid #F8C7D2;
            border-radius: 18px;
            background: #FFFFFF;
            box-shadow: 0 8px 22px rgba(251,113,133,0.07);
            overflow: visible;
            transition: background 180ms, border-color 180ms, box-shadow 180ms, transform 180ms;
          }
          .ms-tr-row:hover {
            background: #FFF7FA;
            border-color: var(--pink-mid);
            box-shadow: 0 14px 30px rgba(251,113,133,0.12);
            transform: translateY(-1px);
          }

          .ms-tr-left {
            padding: 0.95rem 0.8rem 0.95rem 0.95rem;
            border-right: 1px solid var(--border-soft);
            display: grid;
            grid-template-columns: 34px minmax(0, 1fr);
            gap: 0.55rem;
            align-items: start;
            min-width: 0;
          }

          .ms-speaker-avatar {
            width: 34px;
            height: 34px;
            border-radius: 12px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.76rem;
            font-weight: 900;
            flex: 0 0 auto;
          }
          .ms-speaker-avatar.s1 { background: var(--pink-soft);  color: var(--pink-deep); }
          .ms-speaker-avatar.s2 { background: var(--green-soft); color: #065F46; }
          .ms-speaker-avatar.s3 { background: var(--blue-soft);  color: #1E40AF; }
          .ms-speaker-avatar.s4 { background: var(--lav-soft);   color: #5B21B6; }

          .ms-speaker-meta {
            display: flex;
            flex-direction: column;
            gap: 0.2rem;
            min-width: 0;
          }
          .ms-speaker-name {
            color: var(--warm) !important;
            font-size: 0.78rem;
            font-weight: 850;
            line-height: 1.25;
            overflow-wrap: anywhere;
          }

          .ms-tr-timestamp {
            color: var(--warm-4) !important;
            font-size: 0.65rem; font-variant-numeric: tabular-nums; font-weight: 500;
          }

          .ms-tr-right { padding: 0.9rem 1rem; display: block; min-width: 0; }
          .ms-tr-text {
            color: var(--warm-2) !important; font-size: 0.88rem; line-height: 1.6;
            display: block; white-space: pre-wrap; overflow-wrap: anywhere; word-break: normal;
          }

          @media (max-width: 720px) {
            .ms-transcript-scroll { height: 500px; gap: 0.62rem; }
            .ms-tr-row { grid-template-columns: 1fr; }
            .ms-tr-left {
              border-right: 0;
              border-bottom: 1px solid var(--border-soft);
              grid-template-columns: 34px minmax(0, 1fr);
            }
            .ms-tr-right { padding: 0.85rem 0.95rem 0.95rem; }
          }

          /* ══ CONTENT CARDS ════════════════════════════ */
          /* Summary */
          .ms-output-card {
            border: 1.35px solid #DDD6FE;
            border-radius: 18px;
            background: #FAF7FF;
            border-left: 4px solid var(--lav);
            padding: 1.25rem 1.3rem;
            margin: 0 0 1.1rem;
            box-shadow: 0 8px 24px rgba(167,139,250,0.08);
            transition: border-color 220ms, box-shadow 220ms, transform 220ms;
          }
          .ms-output-card:hover {
            border-color: var(--lav-mid);
            box-shadow: 0 14px 34px rgba(167,139,250,0.13);
            transform: translateY(-1px);
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
            border: 1.35px solid var(--border); border-radius: 18px;
            background: var(--surface); padding: 1.05rem 1.1rem;
            margin: 0 0 1.0rem;
            box-shadow: 0 8px 24px rgba(251,113,133,0.075);
            transition: border-color 220ms, transform 220ms, box-shadow 220ms;
          }
          .ms-item-card:hover { transform: translateY(-2px); box-shadow: 0 14px 34px rgba(251,113,133,0.12); }

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
            border: 1.35px solid #F8C7D2;
            border-radius: 20px;
            background: linear-gradient(180deg,#FFFFFF 0%,#FFFBFE 100%);
            padding: 1.05rem;
            box-shadow: 0 8px 24px rgba(251,113,133,0.075);
            transition: border-color 220ms ease, box-shadow 220ms ease, transform 220ms ease;
          }
          .ms-report-block:hover {
            border-color: var(--pink-mid);
            box-shadow: 0 14px 34px rgba(251,113,133,0.12);
            transform: translateY(-1px);
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
            border: 1.5px dashed #F8C7D2;
            border-radius: 20px;
            background: var(--surface);
            color: var(--warm-3) !important;
            font-size: 0.84rem;
            padding: 2rem 1.5rem;
            text-align: center;
            box-shadow: 0 8px 24px rgba(251,113,133,0.075);
          }
          .ms-empty-state {
            border: 1.35px solid #F8C7D2;
            border-radius: 20px;
            background: #FFFBFE;
            padding: 3rem 1.75rem;
            text-align: center;
            margin: 0.5rem 0 1.25rem;
            box-shadow: 0 8px 24px rgba(251,113,133,0.075);
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
          .ms-empty-brand-logo { display:block; width:46px; height:46px; object-fit:contain; background:transparent; border:0; }
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
            min-height: 44px !important;
            border: 1.35px solid #F8C7D2 !important;
            border-radius: 14px !important;
            background: #FFFBFE !important;
            color: var(--warm-2) !important;
            font-size: 0.88rem !important;
            line-height: 1.55 !important;
            padding: 0.72rem 0.85rem !important;
            box-shadow: 0 4px 14px rgba(251,113,133,0.05) !important;
            transition: border-color 180ms ease, box-shadow 180ms ease, background 180ms ease !important;
          }
          .stTextArea textarea:focus {
            border-color: var(--pink) !important;
            background: #FFFFFF !important;
            box-shadow: 0 0 0 3px var(--pink-soft), 0 8px 22px rgba(251,113,133,0.10) !important;
            outline: none !important;
          }

          div[data-testid="stTextInput"] input {
            min-height: 42px !important;
            border: 1.35px solid #F8C7D2 !important;
            border-radius: 14px !important;
            background: #FFFBFE !important;
            color: var(--warm-2) !important;
            font-size: 0.88rem !important;
            padding: 0.55rem 0.8rem !important;
            box-shadow: 0 4px 14px rgba(251,113,133,0.05) !important;
            transition: border-color 180ms ease, box-shadow 180ms ease, background 180ms ease !important;
          }
          div[data-testid="stTextInput"] input:focus {
            border-color: var(--pink) !important;
            background: #FFFFFF !important;
            box-shadow: 0 0 0 3px var(--pink-soft), 0 8px 22px rgba(251,113,133,0.10) !important;
            outline: none !important;
          }
          div[data-testid="stTextInput"] input,
          .stTextArea textarea {
            font-family: Inter, system-ui, sans-serif !important;
          }
          div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
            min-height: 42px !important;
            border: 1.35px solid #F8C7D2 !important;
            border-radius: 14px !important;
            background: #FFFBFE !important;
            box-shadow: 0 4px 14px rgba(251,113,133,0.05) !important;
            transition: border-color 180ms ease, box-shadow 180ms ease, background 180ms ease !important;
          }
          div[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within {
            border-color: var(--pink) !important;
            background: #FFFFFF !important;
            box-shadow: 0 0 0 3px var(--pink-soft), 0 8px 22px rgba(251,113,133,0.10) !important;
          }
          div[data-testid="stSelectbox"] [data-baseweb="select"] span {
            color: var(--warm-2) !important;
            font-family: Inter, system-ui, sans-serif !important;
            font-size: 0.88rem !important;
          }
          div[data-testid="stTextInput"] label,
          div[data-testid="stTextArea"] label,
          div[data-testid="stSelectbox"] label {
            color: var(--warm-2) !important;
            font-size: 0.76rem !important;
            font-weight: 750 !important;
            letter-spacing: 0.01em !important;
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
            background: #FFF1F5 !important; border-radius: 999px !important;
            box-shadow: inset 0 0 0 1px #FAD4DC !important;
          }
          div[data-testid="stProgressBar"] > div > div {
            background: linear-gradient(90deg, #FAD4DC 0%, #FB7185 50%, #FBCFE8 100%) !important;
            border-radius: 999px !important;
            transition: width 220ms ease !important;
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
            .ms-tr-row { grid-template-columns: 104px minmax(0,1fr); }
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


def participants_from_transcript_turns(turns: list[TranscriptTurn]) -> list[str]:
    """Return ordered, non-generic participant names from parsed transcript turns.

    Args:
        turns: Parsed speaker turns produced by ``ml_mom.transcript_parser``.

    Returns:
        Speaker names in transcript order, with duplicates and generic labels
        such as ``Speaker 1`` removed.
    """

    # Manual meeting-info participants remain the source of truth; this helper
    # only supplies ordered, non-generic names when the UI field is empty.
    participants: list[str] = []
    seen: set[str] = set()
    for turn in turns:
        speaker = (turn.speaker_normalized or turn.speaker_raw or "").strip()
        if not speaker or re.fullmatch(r"(?i)speaker\s+[A-Za-z0-9]+", speaker):
            continue
        key = speaker.casefold()
        if key in seen:
            continue
        participants.append(speaker)
        seen.add(key)
    return participants


def prediction_records_from_results(
    prediction_results: list[PredictionResult],
) -> list[PredictionRecord]:
    """Convert ANN prediction DTOs into rule-based MoM generator records.

    Args:
        prediction_results: Sentence-level predictions returned by
            ``ml_mom.predict_ann.predict_labels``.

    Returns:
        Prediction records accepted by ``ml_mom.mom_generator.generate_minutes``.
    """

    # The predictor and rule-based MoM generator use separate DTOs. Keeping this
    # conversion narrow avoids coupling production UI code to prediction internals.
    records: list[PredictionRecord] = []
    for result in prediction_results:
        records.append(
            PredictionRecord(
                sentence=result.sentence,
                speaker=result.speaker,
                timestamp=result.timestamp,
                predicted_label=result.predicted_label,
                confidence_score=result.confidence_score,
            )
        )
    return records


def meeting_minutes_to_analysis_result(
    transcript_text: str,
    topic_labels: list[str],
    minutes: MeetingMinutes,
) -> MeetingAnalysisResult:
    """Adapt rule-generated ML minutes to the stable frontend/export contract.

    Args:
        transcript_text: Reviewed transcript text used for generation.
        topic_labels: Temporary cluster labels derived from sentence embeddings.
        minutes: Rule-based MoM object produced from ANN predictions.

    Returns:
        ``MeetingAnalysisResult`` consumed by the existing UI, PDF exporter,
        DOCX exporter, and email workflow.
    """

    # The frontend, PDF, DOCX, and email exporters already depend on
    # MeetingAnalysisResult, so the ML pipeline is adapted back into that stable
    # contract instead of changing downstream surfaces.
    info = st.session_state.get("meeting_info", {})
    title = (
        str(info.get("meeting_title") or "").strip()
        or str(minutes.meeting_title or "").strip()
        or default_meeting_title()
    )
    summary_sentences = [item.strip() for item in minutes.executive_summary if item.strip()]
    discussion_sentences = [item.strip() for item in minutes.discussion_points if item.strip()]
    additional_sentences = [
        item.strip() for item in minutes.additional_information if item.strip()
    ]
    summary_pool = summary_sentences or discussion_sentences[:3] or additional_sentences[:3]
    short_summary = summary_pool[0] if summary_pool else "No summary sentences were identified."
    detailed_summary = (
        "\n".join(summary_pool)
        if summary_pool
        else "The transcript was processed, but the ML pipeline did not identify summary content."
    )

    topics = []
    seen_topics: set[str] = set()
    for topic in topic_labels + discussion_sentences[:5]:
        topic = topic.strip()
        if not topic or topic.casefold() in seen_topics:
            continue
        topics.append(topic)
        seen_topics.add(topic.casefold())

    return MeetingAnalysisResult(
        cleaned_transcript=transcript_text,
        summary=MeetingSummary(
            title=title,
            short_summary=short_summary,
            detailed_summary=detailed_summary,
            topics_discussed=topics,
        ),
        key_discussion_points=[
            KeyDiscussionPoint(point=point, speakers=[], timestamp=None)
            for point in discussion_sentences
        ],
        decisions=[
            Decision(decision=decision, owner=None, timestamp=None, confidence="ML")
            for decision in minutes.key_decisions
            if decision.strip()
        ],
        action_items=[
            ActionItem(
                task=item.task,
                owner=None if item.owner == "-" else item.owner,
                due_date=None if item.deadline == "-" else item.deadline,
                timestamp=None,
                status=item.status,
            )
            for item in minutes.action_items
            if item.task.strip()
        ],
    )


def inject_premium_redesign_styles() -> None:
    """Apply the presentation-only MeetScribe editorial design system."""

    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,340;9..144,440;9..144,520&family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500;600&display=swap');

          :root {
            --ms-sand: #EFE8D9;
            --ms-warm-white: #FBFAF7;
            --ms-slate-900: #33363B;
            --ms-slate-600: #767B82;
            --ms-muted-blue: #6D8CAC;
            --ms-lavender: #A79BC7;
            --ms-green: #7C9B79;
            --ms-bg: #FBFAF7;
            --ms-surface: #FFFFFF;
            --ms-primary: #33363B;
            --ms-primary-hover: #2A2D30;
            --ms-accent: rgba(167,152,199,.14);
            --ms-success: rgba(124,152,121,.14);
            --ms-warning: rgba(241,235,216,.85);
            --ms-text: #33363B;
            --ms-muted: #767B82;
            --ms-border: #E1DCCE;
            --ms-border-light: #E1DCCE;
            --ms-radius-sm: 8px;
            --ms-radius: 14px;
            --ms-radius-lg: 22px;
            --ms-radius-pill: 999px;
            --ms-shadow-sm: 0 1px 2px rgba(51,54,59,.06);
            --ms-shadow: 0 6px 20px rgba(51,54,59,.08);
            --ms-shadow-lg: 0 20px 48px rgba(51,54,59,.14);
            --ms-font-serif: "Fraunces", Georgia, "Times New Roman", serif;
            --ms-font-sans: Inter, "Segoe UI", system-ui, -apple-system, sans-serif;
            --ms-font-mono: "IBM Plex Mono", ui-monospace, monospace;
          }

          html { scroll-behavior: smooth; }
          body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
            background: var(--ms-bg) !important;
            color: var(--ms-text) !important;
            font-family: var(--ms-font-sans) !important;
            font-size: 15px !important;
            line-height: 1.4667 !important;
          }
          [data-testid="stHeader"] { background: transparent !important; }
          [data-testid="stSidebar"] { display: none !important; }
          [data-testid="stAppViewContainer"] > .main {
            padding-top: 0 !important;
          }
          .main .block-container {
            max-width: 1180px !important;
            padding: 4.75rem 2.5rem 4rem !important;
          }
          :focus-visible {
            outline: 3px solid rgba(51,58,63,.25) !important;
            outline-offset: 3px !important;
          }

          .ms-topnav {
            position: fixed;
            inset: 0 0 auto 0;
            z-index: 999;
            height: 76px;
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            align-items: center;
            gap: 1.5rem;
            padding: 0 max(2.5rem, calc((100vw - 1180px)/2));
            border-bottom: 1px solid var(--ms-border-light);
            background: rgba(251,250,247,.94);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            transition: box-shadow .25s ease, background .25s ease;
          }
          .ms-topnav.scrolled { box-shadow: var(--ms-shadow-sm); }
          .ms-nav-brand {
            display: flex; align-items: center; gap: .65rem;
            color: var(--ms-text) !important; text-decoration: none !important;
            font-family: var(--ms-font-serif); font-weight: 520; font-size: 1.1875rem;
            justify-self: start;
          }
          .ms-logo {
            width: 38px; height: 38px; border-radius: 0;
            display: grid; place-items: center;
            color: inherit; background: transparent;
          }
          .ms-brand-image{display:block;max-width:100%;height:auto;object-fit:contain;background:transparent;border:0}
          .ms-brand-image-nav{width:38px;height:38px}
          .ms-brand-image-footer{width:52px;height:52px}
          .ms-nav-links {
            display: flex; align-items: center; gap: 2.125rem;
            justify-self: center;
          }
          .ms-nav-links a {
            color: var(--ms-slate-600) !important; text-decoration: none !important;
            font-size: .875rem; font-weight: 500; letter-spacing: 0;
            transition: color .2s ease, opacity .2s ease;
            position: relative; padding: .35rem 0;
          }
          .ms-nav-links a:hover { color: var(--ms-text) !important; opacity: .85; }
          .ms-nav-links a.active { color: var(--ms-text) !important; font-weight: 600; }
          .ms-nav-links a.active::after {
            content: ""; position: absolute; left: 0; right: 0; bottom: -4px;
            height: 1px; background: var(--ms-slate-900); border-radius: 1px;
          }
          .ms-nav-spacer { justify-self: end; display: flex; align-items: center; }
          .ms-nav-toggle {
            display: none; align-items: center; justify-content: center;
            width: 40px; height: 40px; padding: 0; border: 1px solid var(--ms-border);
            border-radius: var(--ms-radius-sm); background: var(--ms-warm-white);
            color: var(--ms-text); cursor: pointer;
            transition: background .2s ease, border-color .2s ease;
          }
          .ms-nav-toggle:hover { background: var(--ms-sand); border-color: var(--ms-border); }
          .ms-nav-toggle svg { width: 18px; height: 18px; }
          .ms-nav-mobile-panel {
            display: none; position: fixed; inset: 72px 0 auto 0; z-index: 998;
            padding: 1rem max(1.5rem, calc((100vw - 1160px)/2)) 1.25rem;
            background: rgba(251,249,247,.98); border-bottom: 1px solid var(--ms-border-light);
            box-shadow: var(--ms-shadow-sm);
          }
          .ms-nav-mobile-panel.open { display: block; }
          .ms-nav-mobile-panel a {
            display: block; padding: .85rem 0; color: var(--ms-text) !important;
            text-decoration: none !important; font-size: 1rem; font-weight: 500;
            border-bottom: 1px solid var(--ms-border-light);
          }
          .ms-nav-mobile-panel a:last-child { border-bottom: 0; }
          .ms-nav-mobile-panel a.active { font-weight: 650; color: var(--ms-slate-900) !important; }
          body .stApp .ms-topnav a,
          body .stApp a.ms-btn-primary,
          body .stApp a.ms-btn-secondary {
            text-decoration: none !important;
          }

          /* Hero */
          .ms-hero {
            position: relative !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            text-align: center !important;
            padding: 6.5rem 0 3.75rem !important;
            background: transparent !important;
            border: 0 !important; box-shadow: none !important;
            min-height: auto !important; overflow: visible !important;
          }
          .ms-hero::before { display: none !important; }
          .ms-hero-badge {
            display: inline-flex; align-items: center; gap: .45rem;
            color: var(--ms-muted-blue); background: rgba(109,140,172,.1);
            border: 1px solid rgba(109,140,172,.18);
            border-radius: var(--ms-radius-pill); padding: .375rem .75rem;
            margin-bottom: 1.625rem;
            font-family: var(--ms-font-mono); font-size: .75rem; font-weight: 500;
            letter-spacing: .06em; text-transform: uppercase;
          }
          .ms-hero h1 {
            max-width: 820px !important; margin: 0 auto !important;
            color: var(--ms-text) !important;
            font-family: var(--ms-font-serif) !important;
            font-size: clamp(3.5rem, 5.4vw, 4.25rem) !important;
            line-height: 1.06 !important; letter-spacing: -.02em !important;
            font-weight: 440 !important;
          }
          .ms-hero-sub {
            max-width: 560px !important; margin: 1.375rem auto 0 !important;
            color: var(--ms-muted) !important; font-size: 1.0625rem !important;
            line-height: 1.5 !important;
          }
          .ms-hero-actions {
            display: flex; align-items: center; justify-content: center;
            gap: .75rem; margin-top: 2rem; flex-wrap: wrap;
          }
          .ms-btn-primary, .ms-btn-secondary {
            display: inline-flex; align-items: center; justify-content: center;
            gap: .45rem; padding: .8125rem 1.375rem; border-radius: var(--ms-radius-pill);
            font-size: .90625rem; font-weight: 500; text-decoration: none !important;
            transition: transform .2s ease, box-shadow .2s ease, background .2s ease;
          }
          .ms-btn-primary {
            background: var(--ms-slate-900); color: #FBF9F7 !important;
            box-shadow: var(--ms-shadow-sm);
          }
          .ms-btn-primary:hover {
            transform: translateY(-1px); background: var(--ms-primary-hover);
            box-shadow: var(--ms-shadow);
          }
          .ms-btn-secondary {
            color: var(--ms-text) !important; background: var(--ms-warm-white);
            border: 1px solid var(--ms-border);
          }
          .ms-btn-secondary:hover {
            transform: translateY(-1px); box-shadow: var(--ms-shadow-sm);
          }

          /* Hero preview card */
          .ms-hero-preview {
            width: 100%; max-width: 920px; margin: 4.5rem auto 0;
            padding: 2.25rem; background: #FFFFFF;
            border: 1px solid var(--ms-border); border-radius: var(--ms-radius-lg);
            box-shadow: var(--ms-shadow-lg);
          }
          .ms-preview-bars {
            display: flex; flex-direction: column; gap: .75rem;
            padding: 1.25rem 0;
          }
          .ms-preview-bar {
            height: 11px; border-radius: 6px;
            transition: opacity .2s ease;
          }
          .ms-preview-bar:nth-child(1) { width: 78%; background: var(--ms-lavender); opacity: .9; }
          .ms-preview-bar:nth-child(2) { width: 52%; background: var(--ms-slate-600); margin-left: 6%; opacity: .35; }
          .ms-preview-bar:nth-child(3) { width: 88%; background: var(--ms-green); margin-left: 3%; opacity: .85; }
          .ms-preview-bar:nth-child(4) { width: 62%; background: var(--ms-slate-600); margin-left: 10%; opacity: .4; }

          /* Trust bar */
          .ms-trust-bar {
            display: flex; align-items: center; justify-content: space-between;
            gap: 2rem; flex-wrap: wrap;
            width: 100%; max-width: 920px; margin: 3rem auto 0;
            padding: 2rem 0 0; border-top: 1px solid var(--ms-border-light);
          }
          .ms-trust-label {
            color: var(--ms-muted); font-size: .9375rem; font-weight: 500;
          }
          .ms-trust-tags {
            display: flex; align-items: center; gap: 1.75rem; flex-wrap: wrap;
          }
          .ms-trust-tags span {
            color: var(--ms-slate-600); font-size: .9375rem; font-weight: 500;
          }

          /* Section dividers */
          .ms-section-divider {
            width: 100%; max-width: 1160px; margin: 0 auto;
            border: 0; border-top: 1px solid var(--ms-border-light);
          }

          /* Feature sections */
          .ms-feature-section {
            max-width: 1160px; margin: 0 auto;
            padding: 3.25rem 0;
          }
          .ms-feature-grid {
            display: grid; grid-template-columns: 1fr 1fr;
            gap: 4rem; align-items: center;
          }
          .ms-feature-grid.reverse { direction: rtl; }
          .ms-feature-grid.reverse > * { direction: ltr; }
          .ms-feature-label {
            color: var(--ms-muted); font-size: .85rem; font-weight: 500;
            margin-bottom: 1rem; letter-spacing: .02em;
          }
          .ms-feature-heading {
            font-family: var(--ms-font-serif); font-size: 2rem;
            line-height: 1.1875; letter-spacing: -.01em; font-weight: 440;
            color: var(--ms-text); margin: 0 0 1.25rem;
          }
          .ms-feature-body {
            color: var(--ms-muted); font-size: 1.0625rem; line-height: 1.5; margin: 0;
          }
          .ms-feature-link {
            display: inline-flex; align-items: center; gap: .35rem;
            margin-top: 1.5rem; color: var(--ms-text); font-size: .925rem;
            font-weight: 600; text-decoration: none !important;
            border-bottom: 1px solid var(--ms-text); padding-bottom: 2px;
            transition: opacity .2s ease;
          }
          .ms-feature-link:hover { opacity: .7; }

          /* Feature preview cards */
          .ms-feature-card-wrap {
            padding: 1.75rem; background: var(--ms-sand);
            border-radius: var(--ms-radius-lg);
          }
          .ms-feature-card-inner {
            background: var(--ms-warm-white); border-radius: var(--ms-radius);
            padding: 1.5rem; box-shadow: var(--ms-shadow-sm);
          }
          .ms-speaker-pill {
            display: flex; align-items: center; gap: .75rem;
            padding: .85rem 1rem; border-radius: var(--ms-radius-sm);
            background: var(--ms-warm-white); border: 1px solid var(--ms-border-light);
            margin-bottom: .65rem;
          }
          .ms-speaker-pill:last-child { margin-bottom: 0; }
          .ms-speaker-avatar-sm {
            width: 36px; height: 36px; border-radius: 50%;
            display: grid; place-items: center; font-size: .7rem; font-weight: 700;
            color: white; background: var(--ms-lavender); flex-shrink: 0;
          }
          .ms-speaker-avatar-sm.gray { background: var(--ms-slate-600); opacity: .6; }
          .ms-speaker-pill-info { flex: 1; min-width: 0; }
          .ms-speaker-pill-name {
            font-size: .9rem; font-weight: 600; color: var(--ms-text);
          }
          .ms-speaker-pill-meta {
            font-size: .78rem; color: var(--ms-muted); margin-top: .15rem;
          }
          .ms-speaker-pill-badge {
            font-size: .7rem; font-weight: 600; padding: .25rem .55rem;
            border-radius: var(--ms-radius-pill); background: #F5EDE4;
            color: #9A7355;
          }
          .ms-structured-item {
            display: flex; align-items: center; justify-content: space-between;
            gap: 1rem; padding: .9rem 0;
            border-bottom: 1px solid var(--ms-border-light);
          }
          .ms-structured-item:last-child { border-bottom: 0; padding-bottom: 0; }
          .ms-structured-item:first-child { padding-top: 0; }
          .ms-structured-text { font-size: .925rem; color: var(--ms-text); font-weight: 500; }
          .ms-badge-decided {
            font-size: .72rem; font-weight: 600; padding: .3rem .65rem;
            border-radius: var(--ms-radius-pill); background: var(--ms-success);
            color: var(--ms-green); white-space: nowrap;
          }
          .ms-badge-action {
            font-size: .72rem; font-weight: 600; padding: .3rem .65rem;
            border-radius: var(--ms-radius-pill); background: rgba(109,140,172,.12);
            color: var(--ms-muted-blue); white-space: nowrap;
          }
          .ms-badge-question {
            font-size: .72rem; font-weight: 600; padding: .3rem .65rem;
            border-radius: var(--ms-radius-pill); background: #F5EDE4;
            color: #9A7355; white-space: nowrap;
          }
          .ms-edit-line {
            display: flex; align-items: flex-start; gap: 1rem;
            padding: .75rem 0;
          }
          .ms-edit-time {
            font-family: var(--ms-font-mono); font-size: .75rem;
            color: var(--ms-muted); white-space: nowrap; padding-top: .15rem;
          }
          .ms-edit-text { font-size: .925rem; color: var(--ms-text); line-height: 1.6; }
          .ms-edit-text.highlighted {
            background: rgba(162,150,199,.15); padding: .5rem .75rem;
            border-radius: var(--ms-radius-sm); margin: -.25rem 0;
          }

          /* Workflow section */
          .ms-workflow-section {
            max-width: 1160px; margin: 0 auto; padding: 3.5rem 0;
          }
          .ms-workflow-card {
            background: var(--ms-sand); border-radius: var(--ms-radius-lg);
            padding: 4.25rem 3rem; box-shadow: var(--ms-shadow-sm);
          }
          .ms-workflow-header { text-align: center; margin-bottom: 2.25rem; }
          .ms-workflow-header h2 {
            font-family: var(--ms-font-serif); font-size: 2.375rem;
            line-height: 1.2; font-weight: 440; letter-spacing: -.01em; margin: 0 0 .85rem;
            color: var(--ms-text);
          }
          .ms-workflow-header p {
            color: var(--ms-muted); font-size: 1.05rem; margin: 0;
            max-width: 540px; margin-left: auto; margin-right: auto;
          }
          .ms-workflow-grid {
            display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1rem;
          }
          .ms-workflow-step {
            display: flex; flex-direction: column; align-items: flex-start; justify-content: flex-start;
            min-width: 0; min-height: 220px;
            padding: 1.5rem; background: rgba(255,255,255,.72);
            border: 1px solid var(--ms-border); border-radius: var(--ms-radius);
            box-shadow: 0 8px 20px rgba(51,58,63,.04);
          }
          .ms-workflow-step-num {
            align-self: flex-start; padding: .28rem .55rem; border-radius: var(--ms-radius-pill);
            background: rgba(109,140,172,.12); color: var(--ms-muted-blue);
            font-family: var(--ms-font-mono); font-size: .7rem; font-weight: 600;
            line-height: 1; margin-bottom: 1.15rem; letter-spacing: .05em;
          }
          .ms-workflow-step h3 {
            width: 100%; font-size: 1rem; font-weight: 650; color: var(--ms-text);
            margin: 0 0 .65rem; line-height: 1.35; overflow-wrap: normal; word-break: normal;
          }
          .ms-workflow-step p {
            width: 100%; font-size: .9rem; color: var(--ms-muted); line-height: 1.6; margin: 0;
            overflow-wrap: normal; word-break: normal; hyphens: none;
          }

          /* Comparison table */
          .ms-compare-section {
            max-width: 1160px; margin: 0 auto; padding: 3.5rem 0;
          }
          .ms-compare-header { text-align: center; margin-bottom: 3rem; }
          .ms-compare-header h2 {
            font-family: var(--ms-font-serif); font-size: clamp(1.75rem, 3vw, 2.25rem);
            font-weight: 500; letter-spacing: -.025em; margin: 0 0 .85rem;
          }
          .ms-compare-header p { color: var(--ms-muted); font-size: 1.05rem; margin: 0; }
          .ms-compare-table {
            width: 100%; border-collapse: collapse;
          }
          .ms-compare-table th {
            text-align: left; padding: 1rem 1.25rem; font-size: .72rem;
            font-weight: 700; letter-spacing: .06em; text-transform: uppercase;
            color: var(--ms-muted); border-bottom: 1px solid var(--ms-border-light);
          }
          .ms-compare-table th:last-child { background: rgba(109,140,172,.08); }
          .ms-compare-table td {
            padding: 1.15rem 1.25rem; font-size: .925rem;
            border-bottom: 1px solid var(--ms-border-light);
            color: var(--ms-muted); vertical-align: top;
          }
          .ms-compare-table td:first-child {
            font-weight: 600; color: var(--ms-text); font-size: .875rem;
          }
          .ms-compare-table td:last-child {
            background: rgba(109,140,172,.08); color: var(--ms-text); font-weight: 600;
          }
          .ms-compare-table tr:last-child td { border-bottom: 0; }

          /* FAQ */
          .ms-faq-section {
            max-width: 1160px; margin: 0 auto; padding: 3.5rem 0;
          }
          .ms-faq-header { text-align: center; margin-bottom: 3rem; }
          .ms-faq-header h2 {
            font-family: var(--ms-font-serif); font-size: clamp(1.75rem, 3vw, 2.25rem);
            font-weight: 500; letter-spacing: -.025em; margin: 0;
          }
          .ms-faq-list { max-width: 720px; margin: 0 auto; border-top: 1px solid var(--ms-border-light); }
          .ms-faq-item {
            border-bottom: 1px solid var(--ms-border-light);
          }
          .ms-faq-item summary {
            display: flex; align-items: center; justify-content: space-between;
            gap: 1.5rem; padding: 1.35rem 0; cursor: pointer; list-style: none;
            font-family: var(--ms-font-serif); font-size: 1.125rem; font-weight: 440; color: var(--ms-text);
            transition: opacity .2s ease;
          }
          .ms-faq-item summary::-webkit-details-marker { display: none; }
          .ms-faq-item summary:hover { opacity: .75; }
          .ms-faq-icon {
            font-size: 1.25rem; font-weight: 400; color: var(--ms-muted);
            flex-shrink: 0; width: 24px; text-align: center;
            transition: transform .2s ease;
          }
          .ms-faq-item[open] .ms-faq-icon { transform: rotate(45deg); }
          .ms-faq-answer {
            padding: 0 0 1.35rem; color: var(--ms-muted);
            font-size: .975rem; line-height: 1.75;
          }

          /* Footer CTA */
          .ms-footer-cta {
            max-width: 1160px; margin: 0 auto 0; padding: 0 0 3.5rem;
          }
          .ms-footer-cta-inner {
            background: var(--ms-slate-900); border-radius: var(--ms-radius-lg);
            padding: 4rem 3rem; text-align: center;
          }
          .ms-footer-cta-inner h2 {
            font-family: var(--ms-font-serif); font-size: clamp(1.75rem, 3vw, 2.5rem);
            font-weight: 500; color: #FFFFFF !important; margin: 0 0 1rem;
            letter-spacing: -.025em;
          }
          .ms-footer-cta-inner p {
            color: rgba(255,255,255,.86) !important; font-size: 1.05rem;
            margin: 0 0 2rem; max-width: 480px; margin-left: auto; margin-right: auto;
          }
          .ms-footer-cta-inner .ms-btn-primary {
            background: rgba(255,255,255,.12); color: #FFFFFF !important;
            border: 1px solid rgba(255,255,255,.7);
          }
          .ms-footer-cta-inner .ms-btn-primary:hover {
            background: rgba(255,255,255,.2); color:#FFFFFF !important;
          }

          /* Site footer */
          .ms-site-footer {
            max-width: 1160px; margin: 0 auto;
            padding: 3rem 0 2rem;
            border-top: 1px solid var(--ms-border-light);
          }
          .ms-footer-grid {
            display: grid; grid-template-columns: 1.4fr repeat(3, 1fr);
            gap: 3rem; margin-bottom: 3rem;
          }
          .ms-footer-brand p {
            color: var(--ms-muted); font-size: .925rem; line-height: 1.65;
            margin: .85rem 0 0; max-width: 260px;
          }
          .ms-footer-col h4 {
            font-size: .72rem; font-weight: 700; letter-spacing: .06em;
            text-transform: uppercase; color: var(--ms-muted); margin: 0 0 1rem;
          }
          .ms-footer-col a {
            display: block; color: var(--ms-text); font-size: .925rem;
            text-decoration: none !important; margin-bottom: .65rem;
            transition: opacity .2s ease;
          }
          .ms-footer-col a:hover { opacity: .65; }
          .ms-footer-bottom {
            display: flex; align-items: center; justify-content: space-between;
            padding-top: 1.75rem; border-top: 1px solid var(--ms-border-light);
            color: var(--ms-muted); font-size: .85rem;
          }
          .ms-footer-bottom-links { display: flex; gap: 1.5rem; }
          .ms-footer-bottom-links a {
            color: var(--ms-muted); text-decoration: none !important;
            transition: color .2s ease;
          }
          .ms-footer-bottom-links a:hover { color: var(--ms-text); }

          /* Legacy marketing classes (kept for compatibility) */
          .ms-marketing { margin: 0; }
          .ms-section-intro { max-width:680px; margin-bottom:2rem; }
          .ms-section-intro h2 {
            font-family: var(--ms-font-serif); color:var(--ms-text);
            font-size:clamp(2rem,3.3vw,3rem); line-height:1.08;
            letter-spacing:-.045em; margin:0;
          }
          .ms-section-intro p { color:var(--ms-muted); line-height:1.65; margin:.85rem 0 0; }
          .ms-process-grid { display:grid; grid-template-columns:repeat(5,1fr); gap:.7rem; }
          .ms-process-card, .ms-feature-card {
            position:relative; padding:1.35rem; border:1px solid var(--ms-border);
            border-radius:var(--ms-radius); background:var(--ms-surface); box-shadow:var(--ms-shadow-sm);
            transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease;
          }
          .ms-process-card:hover, .ms-feature-card:hover {
            transform:translateY(-4px); box-shadow:var(--ms-shadow);
          }
          .ms-card-icon {
            width:38px; height:38px; border-radius:11px; display:grid; place-items:center;
            color:var(--ms-muted-blue); background:var(--ms-accent); margin-bottom:1.1rem;
          }
          .ms-card-icon svg { width:18px; height:18px; transition:transform .18s ease; }
          .ms-process-card:hover svg, .ms-feature-card:hover svg { transform:scale(1.08); }
          .ms-process-card small { color:var(--ms-muted); font-size:.66rem; font-weight:700; letter-spacing:.08em; }
          .ms-process-card h3, .ms-feature-card h3 { color:var(--ms-text); font-size:.95rem; margin:.45rem 0 .4rem; }
          .ms-process-card p, .ms-feature-card p { color:var(--ms-muted); font-size:.78rem; line-height:1.55; margin:0; }
          .ms-benefit-grid { display:grid; grid-template-columns:repeat(5,1fr); gap:.7rem; }
          .ms-benefit {
            min-height:170px; padding:1.25rem; border:1px solid var(--ms-border);
            border-radius:var(--ms-radius); background:var(--ms-surface);
            transition:transform .18s ease, box-shadow .18s ease;
          }
          .ms-benefit:hover { transform:translateY(-3px); box-shadow:var(--ms-shadow); }
          .ms-benefit strong { display:block; margin:.9rem 0 .35rem; color:var(--ms-text); font-size:.9rem; }
          .ms-benefit p { margin:0; color:var(--ms-muted); font-size:.76rem; line-height:1.55; }
          .ms-faq-grid { display:grid; grid-template-columns:1fr 1fr; gap:.75rem; }

          /* Interaction workspace */
          #workspace { scroll-margin-top:88px; }
          div[data-testid="stVerticalBlockBorderWrapper"] {
            border:1px solid var(--ms-border) !important;
            border-radius:24px !important;
            background:var(--ms-surface) !important;
            box-shadow:var(--ms-shadow-sm) !important;
          }
          .ms-upload-title-row { justify-content:center !important; gap:.75rem !important; margin-top:.35rem !important; }
          .ms-upload-title { color:var(--ms-text) !important; font-size:1.3rem !important; letter-spacing:-.02em; }
          .ms-upload-desc { max-width:580px; margin:.55rem auto 1.5rem !important; text-align:center; color:var(--ms-muted) !important; }
          .ms-upload-icon-badge {
            width:44px !important; height:44px !important; border:0 !important;
            border-radius:14px !important; color:var(--ms-primary) !important;
            background:var(--ms-accent) !important; box-shadow:none !important;
          }
          .ms-sub-label { color:var(--ms-text) !important; font-size:.9rem !important; text-align:center; }
          .ms-sub-fmt { color:var(--ms-muted) !important; text-align:center; margin-bottom:1rem !important; }
          [data-testid="stFileUploaderDropzone"] {
            min-height:190px !important; border:1.5px dashed #BCC7EA !important;
            border-radius:19px !important; background:#F8FAFF !important;
            padding:2rem !important; transition:border-color .18s ease, background .18s ease, transform .18s ease, box-shadow .18s ease !important;
          }
          [data-testid="stFileUploaderDropzone"]:hover,
          [data-testid="stFileUploaderDropzone"]:focus-within {
            transform:translateY(-2px); border-color:var(--ms-primary) !important;
            background:#F4F6FF !important; box-shadow:0 12px 28px rgba(91,110,245,.11) !important;
          }
          [data-testid="stFileUploaderDropzone"] svg { color:var(--ms-primary) !important; }
          [data-testid="stFileUploaderDropzone"] button {
            border-radius:10px !important; color:var(--ms-primary) !important;
            border-color:#CCD4F8 !important; background:white !important; font-weight:650 !important;
          }
          .ms-or-wrap { min-height:100%; display:grid !important; place-items:center; }
          .ms-or-divider { color:#9CA3AF !important; background:var(--ms-bg) !important; border:1px solid var(--ms-border); }
          .ms-file-card {
            min-height:72px !important; padding:.85rem !important; border:1px solid #DDE3F2 !important;
            border-radius:15px !important; background:#FCFDFF !important; box-shadow:var(--ms-shadow-sm);
            transition:transform .18s ease, box-shadow .18s ease !important;
          }
          .ms-file-card:hover { transform:translateY(-2px); box-shadow:var(--ms-shadow) !important; }
          .ms-file-icon { width:38px !important; height:38px !important; color:var(--ms-primary) !important; background:var(--ms-accent) !important; }
          .ms-file-name { color:var(--ms-text) !important; font-weight:650 !important; }
          .ms-file-size { color:var(--ms-muted) !important; }
          .ms-upload-success {
            display:inline-flex; align-items:center; gap:.35rem; margin-left:auto;
            color:#287A54; background:var(--ms-success); border-radius:999px;
            padding:.3rem .55rem; font-size:.65rem; font-weight:700;
          }
          .ms-file-meta {
            display:flex; align-items:center; gap:.35rem; flex-wrap:wrap; margin-top:.25rem;
            color:var(--ms-muted); font-size:.64rem;
          }
          .ms-file-meta span {
            display:inline-flex; align-items:center; padding:.18rem .4rem;
            border:1px solid var(--ms-border); border-radius:999px; background:#FFFFFF;
          }
          div[data-testid="stHorizontalBlock"]:has(.ms-file-card):not(:has(div[data-testid="stHorizontalBlock"] .ms-file-card)) {
            align-items:center; gap:.65rem !important; margin-top:.75rem; padding:.65rem;
            border:1px solid #DDE3F2; border-radius:15px; background:#FCFDFF;
            box-shadow:var(--ms-shadow-sm); transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease;
          }
          div[data-testid="stHorizontalBlock"]:has(.ms-file-card):not(:has(div[data-testid="stHorizontalBlock"] .ms-file-card)):hover {
            transform:translateY(-1px); border-color:#CDD5EA; box-shadow:var(--ms-shadow);
          }
          div[data-testid="stHorizontalBlock"]:has(.ms-file-card):not(:has(div[data-testid="stHorizontalBlock"] .ms-file-card)) .ms-file-card {
            min-height:54px !important; padding:.15rem !important; border:0 !important;
            border-radius:0 !important; background:transparent !important; box-shadow:none !important;
          }
          div[data-testid="stHorizontalBlock"]:has(.ms-file-card):not(:has(div[data-testid="stHorizontalBlock"] .ms-file-card)) .ms-file-card:hover {
            transform:none !important; box-shadow:none !important;
          }
          div[data-testid="stHorizontalBlock"]:has(.ms-file-card):not(:has(div[data-testid="stHorizontalBlock"] .ms-file-card)) .stButton > button {
            min-height:38px !important; padding:.48rem .65rem !important;
            border:1px solid #E0E3E9 !important; border-radius:10px !important;
            color:#6B7280 !important; background:#FFFFFF !important; box-shadow:none !important;
          }
          div[data-testid="stHorizontalBlock"]:has(.ms-file-card):not(:has(div[data-testid="stHorizontalBlock"] .ms-file-card)) .stButton > button:hover {
            color:#9B3F3F !important; border-color:#E7CACA !important;
            background:#FFF8F7 !important; box-shadow:0 5px 12px rgba(31,41,55,.06) !important;
          }

          .stButton > button, [data-testid="stFormSubmitButton"] button {
            min-height:44px !important; border-radius:11px !important; font-weight:650 !important;
            transition:transform .16s ease, box-shadow .16s ease !important;
          }
          .stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover {
            transform:translateY(-1px) !important; box-shadow:0 8px 20px rgba(31,41,55,.09) !important;
          }
          .stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] {
            color:white !important; border-color:var(--ms-primary) !important; background:var(--ms-primary) !important;
          }
          textarea, input {
            border-radius:11px !important; border-color:var(--ms-border) !important;
            background:#FDFDFD !important; color:var(--ms-text) !important;
          }
          textarea { padding:1rem !important; line-height:1.7 !important; font-size:.9rem !important; }
          textarea:focus, input:focus {
            border-color:var(--ms-primary) !important; box-shadow:0 0 0 3px rgba(91,110,245,.11) !important;
          }

          .ms-speaker-row {
            margin-top:.8rem !important; padding:1rem !important; border:1px solid var(--ms-border) !important;
            border-radius:15px !important; background:#FCFDFF !important; box-shadow:var(--ms-shadow-sm);
          }
          .ms-speaker-avatar {
            background:var(--ms-accent) !important; color:var(--ms-primary) !important;
            border:1px solid #DDE3FB !important;
          }
          .ms-speaker-label { color:var(--ms-text) !important; }
          .ms-speaker-badge-text { color:var(--ms-muted) !important; }
          .ms-speaker-stats { display:flex; gap:.35rem; flex-wrap:wrap; margin-top:.4rem; }
          .ms-speaker-stats span {
            display:inline-flex; padding:.2rem .45rem; border-radius:999px;
            color:#536079; background:var(--ms-accent); font-size:.62rem; font-weight:650;
          }
          [data-testid="stForm"] {
            padding:1rem !important; border:1px solid var(--ms-border) !important;
            border-radius:18px !important; background:#FBFCFF !important;
          }

          .ms-transcript-toolbar {
            position:sticky; top:78px; z-index:20; display:flex; align-items:center;
            justify-content:space-between; gap:1rem; flex-wrap:wrap; margin:.9rem 0;
            padding:.75rem .85rem; border:1px solid var(--ms-border); border-radius:14px;
            background:rgba(255,255,255,.94); backdrop-filter:blur(14px); box-shadow:var(--ms-shadow-sm);
          }
          .ms-toolbar-title { color:var(--ms-text); font-size:.78rem; font-weight:700; }
          .ms-toolbar-stats { display:flex; gap:.45rem; flex-wrap:wrap; }
          .ms-toolbar-chip {
            display:inline-flex; align-items:center; gap:.3rem; padding:.35rem .6rem;
            border:1px solid var(--ms-border); border-radius:999px; color:var(--ms-muted);
            background:#F9FAFB; font-size:.68rem; font-weight:600;
          }
          .ms-conversation-preview {
            display:grid !important; gap:.7rem !important; max-height:430px; overflow:auto;
            padding:.2rem .25rem .4rem .05rem !important;
          }
          .ms-convo-row {
            position:relative; padding:1rem 1.05rem 1.05rem 4.2rem !important;
            border:1px solid var(--ms-border) !important; border-radius:16px !important;
            background:#FCFDFF !important; box-shadow:var(--ms-shadow-sm) !important;
            transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease !important;
          }
          .ms-convo-row:hover { transform:translateY(-2px); border-color:#D4DAEB !important; box-shadow:var(--ms-shadow) !important; }
          .ms-convo-row::before {
            content:attr(data-initial); position:absolute; left:1rem; top:1rem;
            width:36px; height:36px; border-radius:12px; display:grid; place-items:center;
            color:var(--ms-primary); background:var(--ms-accent); font-size:.75rem; font-weight:750;
          }
          .ms-convo-speaker { color:var(--ms-text) !important; font-weight:700 !important; }
          .ms-time-chip {
            border:1px solid var(--ms-border) !important; border-radius:999px !important;
            color:var(--ms-muted) !important; background:#F8F9FB !important; font-size:.65rem !important;
          }
          .ms-convo-text { color:#4B5563 !important; font-size:.84rem !important; line-height:1.7 !important; }
          .ms-convo-meta {
            margin-top:.7rem; color:#9CA3AF; font-size:.62rem; font-weight:600;
            letter-spacing:.01em;
          }
          .ms-editor-label {
            display:flex; justify-content:space-between; align-items:center; margin:1.1rem 0 .55rem;
            color:var(--ms-text); font-size:.78rem; font-weight:700;
          }
          .ms-editor-label span { color:var(--ms-muted); font-size:.67rem; font-weight:500; }

          .ms-proc-wrap {
            border:1px solid var(--ms-border) !important; border-radius:20px !important;
            background:white !important; box-shadow:var(--ms-shadow) !important;
          }
          .ms-bar-track { background:#EEF1F5 !important; }
          .ms-bar-fill { background:var(--ms-primary) !important; }
          .ms-step.active .ms-step-circle, .ms-step.done .ms-step-circle { background:var(--ms-primary) !important; }
          .ms-skeleton-line { background:#EEF2FF !important; }
          [data-testid="stToast"] {
            border:1px solid #CAE8D7 !important; border-radius:14px !important;
            background:var(--ms-success) !important; box-shadow:var(--ms-shadow) !important;
          }
          [data-testid="stAlert"] {
            border:1px solid #F1D8C5 !important; border-radius:14px !important;
            background:var(--ms-warning) !important;
          }

          .ms-report-shell { margin-top:1.25rem; }
          .ms-report-legacy-intro { display:none; }
          .ms-report-hero {
            position:relative; overflow:hidden; padding:1.65rem 1.7rem; border:1px solid #DDE3F2;
            border-radius:22px; background:#FFFFFF; box-shadow:var(--ms-shadow); margin-bottom:1rem;
          }
          .ms-report-hero::after {
            content:""; position:absolute; right:-48px; top:-52px; width:180px; height:180px;
            border-radius:50%; background:var(--ms-accent); opacity:.75; pointer-events:none;
          }
          .ms-report-eyebrow { display:flex; align-items:center; gap:.55rem; flex-wrap:wrap; margin-bottom:.8rem; }
          .ms-success-badge {
            display:inline-flex; align-items:center; gap:.4rem; padding:.38rem .65rem;
            border:1px solid #C9E8D7; border-radius:999px; color:#287A54; background:var(--ms-success);
            font-size:.68rem; font-weight:750; animation:ms-report-arrive .42s ease both;
          }
          .ms-generated-time { color:var(--ms-muted); font-size:.68rem; }
          .ms-report-title {
            position:relative; z-index:1; max-width:760px; margin:0; color:var(--ms-text);
            font-size:clamp(1.75rem,4vw,2.65rem); line-height:1.08; letter-spacing:-.045em; font-weight:740;
          }
          .ms-report-subtitle { position:relative; z-index:1; margin:.7rem 0 0; color:var(--ms-muted); font-size:.84rem; }
          .ms-report-actions { display:flex; gap:.5rem; flex-wrap:wrap; margin-top:1.2rem; position:relative; z-index:1; }
          .ms-report-action {
            display:inline-flex; align-items:center; gap:.4rem; padding:.58rem .75rem;
            border:1px solid var(--ms-border); border-radius:10px; color:#4B5563 !important;
            background:#FFFFFF; text-decoration:none !important; font-size:.7rem; font-weight:650;
            transition:transform .16s ease, box-shadow .16s ease, border-color .16s ease;
          }
          .ms-report-action:hover { transform:translateY(-1px); border-color:#CCD4F8; box-shadow:var(--ms-shadow-sm); }
          .ms-info-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.65rem; margin:1rem 0 1.2rem; }
          .ms-info-card {
            padding:.9rem; border:1px solid var(--ms-border); border-radius:14px; background:#FFFFFF;
            box-shadow:var(--ms-shadow-sm); min-width:0;
          }
          .ms-info-label { color:var(--ms-muted); font-size:.63rem; font-weight:650; text-transform:uppercase; letter-spacing:.055em; }
          .ms-info-value { margin-top:.35rem; color:var(--ms-text); font-size:.8rem; font-weight:680; overflow-wrap:anywhere; }
          .ms-report-section { margin:1.2rem 0; }
          .ms-report-section-head { display:flex; align-items:end; justify-content:space-between; gap:1rem; margin:0 0 .65rem; }
          .ms-report-section-head h3 { margin:0; color:var(--ms-text); font-size:1.05rem; letter-spacing:-.025em; }
          .ms-report-section-head span { color:var(--ms-muted); font-size:.68rem; }
          .ms-summary-editorial {
            padding:1.55rem clamp(1.2rem,4vw,2.25rem); border:1px solid #D9E0FB; border-radius:20px;
            background:var(--ms-accent); box-shadow:var(--ms-shadow-sm);
          }
          .ms-summary-editorial p { max-width:820px; margin:.7rem 0 0; color:#374151; font-size:1rem; line-height:1.85; }
          .ms-summary-kicker { color:var(--ms-primary); font-size:.67rem; font-weight:750; letter-spacing:.06em; text-transform:uppercase; }
          .ms-summary-topics { display:flex; flex-wrap:wrap; gap:.4rem; margin-top:1rem; }
          .ms-summary-topics span { padding:.3rem .55rem; border-radius:999px; background:#FFFFFF; color:#536079; font-size:.65rem; font-weight:650; }
          .ms-discussion-list, .ms-decision-list, .ms-task-list { display:grid; gap:.65rem; }
          .ms-discussion-card {
            border:1px solid var(--ms-border); border-radius:16px; background:#FFFFFF;
            box-shadow:var(--ms-shadow-sm); transition:transform .18s ease, box-shadow .18s ease;
          }
          .ms-discussion-card:hover { transform:translateY(-1px); box-shadow:var(--ms-shadow); }
          .ms-discussion-card summary {
            display:flex; align-items:center; gap:.7rem; padding:1rem; cursor:pointer; list-style:none;
            color:var(--ms-text); font-size:.82rem; font-weight:700;
          }
          .ms-discussion-card summary::-webkit-details-marker { display:none; }
          .ms-discussion-index { display:grid; place-items:center; flex:0 0 30px; height:30px; border-radius:10px; background:var(--ms-accent); color:var(--ms-primary); font-size:.66rem; }
          .ms-discussion-chevron { margin-left:auto; color:#9CA3AF; transition:transform .18s ease; }
          .ms-discussion-card[open] .ms-discussion-chevron { transform:rotate(180deg); }
          .ms-discussion-body { padding:0 1rem 1rem 3.45rem; color:#4B5563; font-size:.8rem; line-height:1.7; }
          .ms-detail-meta { display:flex; gap:.4rem; flex-wrap:wrap; margin-top:.65rem; }
          .ms-detail-meta span, .ms-task-chip {
            padding:.28rem .5rem; border:1px solid var(--ms-border); border-radius:999px;
            color:var(--ms-muted); background:#FAFAFA; font-size:.62rem; font-weight:620;
          }
          .ms-decision-card {
            display:grid; grid-template-columns:34px minmax(0,1fr) auto; gap:.8rem; align-items:start;
            padding:1rem; border:1px solid #CDE7D8; border-radius:16px; background:#F8FCFA; box-shadow:var(--ms-shadow-sm);
          }
          .ms-decision-check { display:grid; place-items:center; width:34px; height:34px; border-radius:11px; color:#287A54; background:var(--ms-success); font-weight:800; }
          .ms-decision-text { color:#374151; font-size:.84rem; line-height:1.65; font-weight:600; }
          .ms-confirmed-chip { padding:.3rem .52rem; border-radius:999px; color:#287A54; background:var(--ms-success); font-size:.62rem; font-weight:700; }
          .ms-task-card {
            display:grid; grid-template-columns:minmax(0,1.5fr) minmax(130px,.55fr); gap:1rem;
            padding:1rem; border:1px solid var(--ms-border); border-radius:16px; background:#FFFFFF; box-shadow:var(--ms-shadow-sm);
          }
          .ms-task-title { color:var(--ms-text); font-size:.84rem; font-weight:680; line-height:1.55; }
          .ms-task-meta { display:flex; gap:.4rem; flex-wrap:wrap; margin-top:.65rem; }
          .ms-owner { display:flex; align-items:center; gap:.55rem; color:#4B5563; font-size:.7rem; font-weight:650; }
          .ms-owner-avatar { display:grid; place-items:center; width:32px; height:32px; border-radius:11px; background:var(--ms-warning); color:#9A5B30; font-size:.68rem; font-weight:750; }
          .ms-priority-high { color:#A84D45 !important; background:#FFF0EE !important; border-color:#F1D2CE !important; }
          .ms-priority-medium { color:#8A6724 !important; background:#FFF8E8 !important; border-color:#EFDFC0 !important; }
          .ms-priority-low { color:#287A54 !important; background:var(--ms-success) !important; border-color:#CDE7D8 !important; }
          .ms-export-wrap { border-color:#DDE3F2 !important; background:#FFFFFF !important; box-shadow:var(--ms-shadow) !important; }
          .ms-export-option { min-height:112px; border:1px solid var(--ms-border); border-radius:14px; background:#FCFDFF; padding:.9rem; }
          .ms-export-option-icon.pdf,
          .ms-export-option-icon.docx,
          .ms-export-option-icon.email {
            color:var(--ms-primary) !important; background:var(--ms-accent) !important;
            border:1px solid #DCE2FA !important;
          }
          @keyframes ms-report-arrive { from { opacity:0; transform:translateY(5px); } to { opacity:1; transform:none; } }

          /* Final production polish: normalize native Streamlit surfaces. */
          *, *::before, *::after { box-sizing:border-box; }
          ::selection { color:var(--ms-text); background:rgba(162,150,199,.25); }
          html { scroll-padding-top:88px; }
          [id] { scroll-margin-top:88px; }
          body { text-rendering:optimizeLegibility; -webkit-font-smoothing:antialiased; }
          body::-webkit-scrollbar, .ms-conversation-preview::-webkit-scrollbar { width:10px; height:10px; }
          body::-webkit-scrollbar-track, .ms-conversation-preview::-webkit-scrollbar-track { background:transparent; }
          body::-webkit-scrollbar-thumb, .ms-conversation-preview::-webkit-scrollbar-thumb {
            border:3px solid transparent; border-radius:999px; background-clip:padding-box; background:#CDD2DC;
          }
          body::-webkit-scrollbar-thumb:hover, .ms-conversation-preview::-webkit-scrollbar-thumb:hover { background:#AEB5C2; }
          [data-testid="stHeaderActionElements"], a[data-testid="stHeaderAction"] { display:none !important; }
          [data-testid="stMarkdownContainer"] > :first-child { margin-top:0; }
          [data-testid="stMarkdownContainer"] > :last-child { margin-bottom:0; }
          p { text-wrap:pretty; }
          h1, h2, h3, h4 { text-wrap:balance; }

          .stButton > button,
          .stDownloadButton > button,
          [data-testid="stFormSubmitButton"] button,
          button[data-baseweb="button"] {
            min-height:44px !important; padding:.62rem .9rem !important;
            border-radius:var(--ms-radius-sm) !important; font-size:.78rem !important;
            font-weight:680 !important; letter-spacing:-.005em !important;
            transition:transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease,
              background-color 160ms ease, color 160ms ease !important;
          }
          .stButton > button:hover,
          .stDownloadButton > button:hover,
          [data-testid="stFormSubmitButton"] button:hover,
          button[data-baseweb="button"]:hover {
            transform:translateY(-1px) !important; box-shadow:0 8px 20px rgba(31,41,55,.09) !important;
          }
          .stButton > button:active,
          .stDownloadButton > button:active,
          [data-testid="stFormSubmitButton"] button:active,
          button[data-baseweb="button"]:active {
            transform:translateY(0) scale(.985) !important; box-shadow:none !important;
          }
          .stButton > button:focus-visible,
          .stDownloadButton > button:focus-visible,
          [data-testid="stFormSubmitButton"] button:focus-visible,
          button[data-baseweb="button"]:focus-visible {
            outline:3px solid rgba(91,110,245,.2) !important; outline-offset:2px !important;
          }
          .stButton > button:disabled,
          .stDownloadButton > button:disabled,
          [data-testid="stFormSubmitButton"] button:disabled,
          button[data-baseweb="button"]:disabled {
            transform:none !important; color:#9CA3AF !important; background:#F3F4F6 !important;
            border-color:#E5E7EB !important; box-shadow:none !important; cursor:not-allowed !important; opacity:.82 !important;
          }
          .stButton > button p, .stDownloadButton > button p,
          [data-testid="stFormSubmitButton"] button p { line-height:1.2 !important; }

          div[data-testid="stTextInput"] input,
          div[data-testid="stTextArea"] textarea,
          div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
            border:1px solid var(--ms-border) !important; border-radius:var(--ms-radius-sm) !important;
            background:#FDFDFD !important; box-shadow:none !important;
            transition:border-color 170ms ease, box-shadow 170ms ease, background-color 170ms ease !important;
          }
          div[data-testid="stTextInput"] input:hover,
          div[data-testid="stTextArea"] textarea:hover,
          div[data-testid="stSelectbox"] [data-baseweb="select"] > div:hover {
            border-color:#CFD4DE !important; background:#FFFFFF !important;
          }
          div[data-testid="stTextInput"] input:focus,
          div[data-testid="stTextArea"] textarea:focus,
          div[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within {
            border-color:var(--ms-primary) !important; background:#FFFFFF !important;
            box-shadow:0 0 0 3px rgba(91,110,245,.11) !important; outline:none !important;
          }
          div[data-testid="stTextInput"] label,
          div[data-testid="stTextArea"] label,
          div[data-testid="stSelectbox"] label {
            color:#4B5563 !important; font-size:.72rem !important; font-weight:680 !important; letter-spacing:0 !important;
          }
          input::placeholder, textarea::placeholder { color:#9CA3AF !important; opacity:1 !important; }

          [data-testid="stVerticalBlockBorderWrapper"] {
            border-color:var(--ms-border) !important; border-radius:var(--ms-radius) !important;
            background:var(--ms-surface); box-shadow:var(--ms-shadow-sm);
          }
          [data-testid="stVerticalBlockBorderWrapper"]:hover {
            border-color:#D9DDE5 !important;
          }
          [data-testid="stExpander"] {
            border:1px solid var(--ms-border) !important; border-radius:var(--ms-radius) !important;
            background:#FFFFFF !important; box-shadow:var(--ms-shadow-sm) !important;
            overflow:hidden; transition:border-color 170ms ease, box-shadow 170ms ease !important;
          }
          [data-testid="stExpander"]:hover { border-color:#D7DCE6 !important; box-shadow:var(--ms-shadow) !important; }

          [data-testid="stProgressBar"] > div {
            height:8px !important; overflow:hidden; border:0 !important; border-radius:999px !important;
            background:#EDF0F5 !important; box-shadow:none !important;
          }
          [data-testid="stProgressBar"] > div > div {
            border-radius:999px !important; background:var(--ms-primary) !important;
            box-shadow:none !important; transition:width 240ms ease !important;
          }
          .ms-skeleton-line {
            position:relative; overflow:hidden; border-radius:999px;
          }
          .ms-skeleton-line::after {
            content:""; position:absolute; inset:0; transform:translateX(-100%);
            background:linear-gradient(90deg,transparent,rgba(255,255,255,.72),transparent);
            animation:ms-shimmer 1.45s ease-in-out infinite;
          }
          @keyframes ms-shimmer { to { transform:translateX(100%); } }

          [data-testid="stToastContainer"] { right:1.25rem !important; bottom:1.25rem !important; }
          [data-testid="stToast"] {
            min-width:280px; padding:.15rem !important; border:1px solid #DDE3EA !important;
            border-radius:14px !important; background:rgba(255,255,255,.98) !important;
            box-shadow:0 18px 42px rgba(17,24,39,.13) !important;
            animation:ms-toast-in 220ms ease-out both;
          }
          [data-testid="stToast"] [data-testid="stMarkdownContainer"] { color:var(--ms-text) !important; font-size:.76rem !important; }
          @keyframes ms-toast-in { from { opacity:0; transform:translateY(8px) scale(.985); } to { opacity:1; transform:none; } }
          [data-testid="stAlert"] {
            padding:.8rem .9rem !important; border-radius:14px !important;
            box-shadow:var(--ms-shadow-sm) !important;
          }
          [data-testid="stCodeBlock"] {
            overflow:hidden; border:1px solid var(--ms-border) !important;
            border-radius:12px !important; background:#F8F9FB !important;
            box-shadow:none !important;
          }
          [data-testid="stCodeBlock"] pre {
            padding:.75rem .85rem !important; color:#4B5563 !important;
            background:#F8F9FB !important; font-size:.68rem !important; line-height:1.55 !important;
          }
          [data-testid="stExpander"] summary {
            min-height:46px; padding:.72rem .85rem !important;
            color:var(--ms-text) !important; font-size:.76rem !important; font-weight:680 !important;
          }
          [data-testid="stExpander"] summary:hover { background:#FAFBFC !important; }

          .stDownloadButton > button {
            color:#FFFFFF !important; border-color:var(--ms-primary) !important;
            background:var(--ms-primary) !important; box-shadow:0 5px 14px rgba(91,110,245,.16) !important;
          }
          .stDownloadButton > button:hover {
            color:#FFFFFF !important; border-color:var(--ms-primary-hover) !important;
            background:var(--ms-primary-hover) !important;
          }
          [data-testid="stFileUploaderDropzone"] button {
            min-height:38px !important; padding:.5rem .72rem !important;
            border:1px solid #CCD4F8 !important; border-radius:10px !important;
            color:var(--ms-primary) !important; background:#FFFFFF !important; box-shadow:none !important;
          }
          [data-testid="stFileUploaderDropzone"] button:active { transform:scale(.985) !important; }

          .ms-email-compose { margin:0 0 .8rem; padding:0 !important; border:0 !important; background:transparent !important; }
          .ms-email-compose-hdr {
            color:var(--ms-text) !important; font-size:1rem !important; font-weight:720 !important;
            letter-spacing:-.02em !important;
          }
          .ms-email-compose-sub { margin:.28rem 0 .8rem !important; color:var(--ms-muted) !important; font-size:.75rem !important; line-height:1.55 !important; }
          .ms-email-field-label {
            margin:.65rem 0 .32rem !important; color:#4B5563 !important;
            font-size:.69rem !important; font-weight:680 !important; letter-spacing:0 !important;
          }
          .ms-attachment-preview {
            display:flex; align-items:center; gap:.7rem; padding:.78rem !important;
            border:1px solid var(--ms-border) !important; border-radius:13px !important;
            background:#FAFBFF !important; box-shadow:none !important;
          }
          .ms-attachment-preview-icon {
            display:grid; place-items:center; flex:0 0 38px; height:38px;
            border-radius:11px !important; color:var(--ms-primary) !important;
            background:var(--ms-accent) !important; font-size:.62rem !important; font-weight:780 !important;
          }
          .ms-attachment-preview-name { color:var(--ms-text) !important; font-size:.74rem !important; font-weight:680 !important; }
          .ms-attachment-preview-meta { color:var(--ms-muted) !important; font-size:.63rem !important; }
          .ms-attachment-pill {
            color:#287A54 !important; background:var(--ms-success) !important;
            border:1px solid #CDE7D8 !important; border-radius:999px !important;
            padding:.28rem .5rem !important; font-size:.61rem !important; font-weight:700 !important;
          }
          iframe[title="streamlit.components.v1.html"] { border:0 !important; border-radius:10px; }
          .ms-empty {
            padding:1.5rem 1.2rem !important; border:1px dashed #CDD5EA !important;
            border-radius:var(--ms-radius) !important; color:var(--ms-muted) !important;
            background:#FAFBFF !important; box-shadow:none !important; font-size:.78rem !important;
          }
          .ms-empty::before {
            width:40px !important; height:40px !important; border-radius:12px !important;
            background:var(--ms-accent) !important; box-shadow:inset 0 0 0 1px #DCE2FA !important;
          }
          .ms-empty-state {
            min-height:180px; padding:2.25rem 1.5rem !important; margin:2rem 0 4.5rem;
            border:1px solid var(--ms-border) !important; border-radius:20px !important;
            background:var(--ms-sand) !important; box-shadow:none !important;
          }
          .ms-empty-icon {
            width:48px !important; height:48px !important; border:1px solid #DCE2FA !important;
            border-radius:14px !important; color:var(--ms-primary) !important;
            background:var(--ms-accent) !important; box-shadow:none !important;
          }
          .ms-empty-title { color:var(--ms-text) !important; font-family:var(--ms-font-serif)!important; font-size:1.5rem !important; line-height:1.3!important; font-weight:440 !important; }
          .ms-empty-copy { max-width:560px;margin:.5rem auto 0!important;color:var(--ms-muted) !important; font-size:.9375rem !important; line-height:1.55 !important; }

          .ms-premium-section, .ms-process-card, .ms-feature-card, .ms-benefit-card,
          .ms-file-card, .ms-speaker-row, .ms-convo-row, .ms-report-hero,
          .ms-summary-editorial, .ms-discussion-card, .ms-decision-card,
          .ms-task-card, .ms-export-wrap {
            animation:ms-surface-in 260ms ease-out both;
          }
          @keyframes ms-surface-in { from { opacity:0; transform:translateY(4px); } to { opacity:1; transform:none; } }

          .ms-footer {
            display:none !important;
          }

          @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after { animation-duration:.01ms !important; animation-iteration-count:1 !important; transition-duration:.01ms !important; }
          }
          @media (max-width: 960px) {
            .ms-nav-links { gap: 1.25rem; }
            .ms-nav-links a { font-size: .85rem; }
            .ms-feature-grid { grid-template-columns: 1fr; gap: 2.5rem; }
            .ms-feature-grid.reverse { direction: ltr; }
            .ms-workflow-grid { grid-template-columns: repeat(2, 1fr); gap: 2rem; }
            .ms-footer-grid { grid-template-columns: 1fr 1fr; gap: 2rem; }
            .ms-compare-table { font-size: .85rem; }
            .ms-compare-table th, .ms-compare-table td { padding: .85rem .75rem; }
            .ms-process-grid, .ms-benefit-grid { grid-template-columns:repeat(2,1fr); }
            .ms-info-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
            .ms-report-hero { padding:1.4rem; }
          }
          @media (max-width: 700px) {
            .main .block-container { padding:7rem 1.25rem 2rem !important; }
            .ms-topnav { min-height: 96px; height: auto; padding: .7rem 1.25rem; grid-template-columns: 1fr; gap: .45rem; }
            .ms-nav-brand { justify-self: center; }
            .ms-nav-links { display: flex; width: 100%; justify-self: stretch; justify-content: center; gap: 1rem; overflow-x: auto; scrollbar-width: none; }
            .ms-nav-links::-webkit-scrollbar { display: none; }
            .ms-nav-links a { font-size: .75rem; white-space: nowrap; }
            .ms-hero { padding: 2.75rem 0 2.5rem !important; }
            .ms-hero h1 { font-size: clamp(2.25rem, 10vw, 2.75rem) !important; }
            .ms-hero-sub { font-size: 1rem !important; }
            .ms-hero-actions { flex-direction: column; width: 100%; }
            .ms-btn-primary, .ms-btn-secondary { width: 100%; }
            .ms-hero-preview { padding: 1.25rem 1.5rem; margin-top: 2.5rem; }
            .ms-trust-bar { flex-direction: column; align-items: flex-start; gap: 1rem; }
            .ms-trust-tags { gap: 1rem; }
            .ms-feature-section, .ms-workflow-section, .ms-compare-section, .ms-faq-section {
              padding: 3.5rem 0;
            }
            .ms-workflow-card { padding: 2rem 1.5rem; }
            .ms-workflow-grid { grid-template-columns: 1fr; }
            .ms-compare-table { display: block; overflow-x: auto; -webkit-overflow-scrolling: touch; }
            .ms-footer-cta-inner { padding: 2.5rem 1.5rem; }
            .ms-footer-grid { grid-template-columns: 1fr; gap: 1.5rem; }
            .ms-footer-bottom { flex-direction: column; gap: 1rem; align-items: flex-start; }
            .ms-process-grid, .ms-benefit-grid, .ms-faq-grid { grid-template-columns:1fr; }
            .ms-feature-card:nth-child(1), .ms-feature-card:nth-child(6) { grid-column:span 1; }
            .ms-process-card:not(:last-child)::after { display:none; }
            .ms-transcript-toolbar { top:68px; align-items:flex-start; flex-direction:column; }
            .ms-convo-row { padding-left:3.8rem !important; }
            .ms-info-grid { grid-template-columns:1fr; }
            .ms-task-card { grid-template-columns:1fr; }
            .ms-decision-card { grid-template-columns:34px minmax(0,1fr); }
            .ms-confirmed-chip { grid-column:2; justify-self:start; }
            .ms-report-hero { padding:1.15rem; border-radius:18px; }
            .ms-report-actions { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); }
            .ms-report-action { justify-content:center; text-align:center; }
            .ms-summary-editorial { padding:1.2rem; border-radius:17px; }
            .ms-discussion-body { padding-left:1rem; }
            [data-testid="stToastContainer"] { inset:auto .75rem .75rem !important; }
            [data-testid="stToast"] { min-width:0; width:100%; }
            .stButton > button, .stDownloadButton > button,
            [data-testid="stFormSubmitButton"] button { min-height:46px !important; }
            div[data-testid="stHorizontalBlock"]:has(.ms-file-card):not(:has(div[data-testid="stHorizontalBlock"] .ms-file-card)) {
              display:flex !important; flex-wrap:nowrap !important; padding:.55rem !important;
            }
            div[data-testid="stHorizontalBlock"]:has(.ms-file-card):not(:has(div[data-testid="stHorizontalBlock"] .ms-file-card))
              > div[data-testid="column"]:first-child { flex:1 1 auto !important; min-width:0 !important; }
            div[data-testid="stHorizontalBlock"]:has(.ms-file-card):not(:has(div[data-testid="stHorizontalBlock"] .ms-file-card))
              > div[data-testid="column"]:last-child { flex:0 0 auto !important; width:auto !important; min-width:76px !important; }
            .ms-file-name { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
            .ms-upload-success { padding:.26rem .46rem; }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_landing_interactivity() -> None:
    """Inject lightweight JS for navbar scroll state and active section highlighting."""

    st.components.v1.html(
        """
        <script>
          (function () {
            const doc = window.parent.document;
            const nav = doc.querySelector('.ms-topnav');
            const links = doc.querySelectorAll('.ms-nav-links a[data-section]');
            const scrollRoot = doc.querySelector('[data-testid="stMain"]') || window.parent;
            if (!nav || !links.length) return;

            const sections = Array.from(links).map(function (link) {
              const id = link.getAttribute('data-section');
              const el = id === 'top' ? doc.getElementById('top') : doc.getElementById(id);
              return { link: link, el: el };
            }).filter(function (item) { return item.el; });

            function setActive() {
              const scrollY = scrollRoot.scrollTop || window.parent.scrollY || 0;
              if (scrollY > 12) nav.classList.add('scrolled');
              else nav.classList.remove('scrolled');

              let current = sections[0];
              sections.forEach(function (item) {
                const rect = item.el.getBoundingClientRect();
                if (rect.top <= 120) current = item;
              });
              links.forEach(function (link) { link.classList.remove('active'); });
              if (current && current.link) current.link.classList.add('active');
            }

            links.forEach(function (link) {
              link.addEventListener('click', function (event) {
                const id = link.getAttribute('data-section');
                const target = id === 'top' ? doc.getElementById('top') : doc.getElementById(id);
                if (!target || !scrollRoot.scrollTo) return;
                event.preventDefault();
                const navOffset = window.parent.innerWidth <= 700 ? 106 : 86;
                const top = scrollRoot.scrollTop + target.getBoundingClientRect().top - navOffset;
                scrollRoot.scrollTo({ top: Math.max(0, top), behavior: 'smooth' });
                window.parent.history.replaceState(null, '', '#' + id);
              });
            });
            scrollRoot.addEventListener('scroll', setActive, { passive: true });
            setActive();
          })();
        </script>
        """,
        height=0,
    )


def render_top_navigation() -> None:
    """Render the static, accessible application navigation."""

    st.markdown(
        f"""
        <nav class="ms-topnav" aria-label="Primary navigation">
          <a class="ms-nav-brand" href="#top" aria-label="MeetScribe home">
            <span class="ms-logo">
              {brand_logo_html("ms-brand-image ms-brand-image-nav", alt="")}
            </span>
            <span>MeetScribe</span>
          </a>
          <div class="ms-nav-links">
            <a href="#top" data-section="top">Home</a>
            <a href="#features" data-section="features">Features</a>
            <a href="#workflow" data-section="workflow">Workflow</a>
            <a href="#workspace" data-section="workspace">Upload</a>
            <a href="#faq" data-section="faq">FAQ</a>
          </div>
        </nav>
        <div id="top"></div>
        """,
        unsafe_allow_html=True,
    )


def experimental_mom_to_analysis_result(
    transcript_text: str,
    generated: GeneratedMomResult,
    audio_quality_mode: bool = False,
) -> MeetingAnalysisResult:
    """Adapt Experimental Formatter V4 output to the stable UI/export contract.

    Args:
        transcript_text: Reviewed transcript text used for generation.
        generated: Result returned by ``ml_mom.experimental.integration.generate_mom``.

    Returns:
        ``MeetingAnalysisResult`` consumed by the existing Streamlit UI, PDF,
        DOCX, and email flows.
    """

    if generated.experimental_mom is None:
        raise ValueError("Experimental MoM output is unavailable.")

    # The UI and exporters are intentionally left untouched. This adapter keeps
    # their existing data contract stable while swapping only the final backend
    # minutes generator to Formatter V4.
    info = st.session_state.get("meeting_info", {})
    experimental_mom = generated.experimental_mom
    title = (
        str(info.get("meeting_title") or "").strip()
        or str(experimental_mom.meeting_title or "").strip()
        or default_meeting_title()
    )
    summary_text = str(experimental_mom.summary or "").strip()
    objective = str(experimental_mom.objective or "").strip()
    summary_sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", summary_text)
        if sentence.strip()
    ]
    unique_summary_sentences: list[str] = []
    seen_summary: set[str] = set()
    for sentence in summary_sentences:
        key = re.sub(r"\W+", " ", sentence).strip().casefold()
        objective_key = re.sub(r"\W+", " ", objective).strip().casefold()
        if key and key not in seen_summary and key != objective_key:
            unique_summary_sentences.append(sentence)
            seen_summary.add(key)
    short_summary = " ".join(unique_summary_sentences[:5]) or objective or "No summary could be generated."
    if audio_quality_mode:
        objective_key = re.sub(r"\W+", " ", objective).strip().casefold()
        short_key = re.sub(r"\W+", " ", short_summary).strip().casefold()
        detailed_summary = objective if objective_key and objective_key != short_key else short_summary
    else:
        detailed_parts = [part for part in (objective, short_summary) if part]
        detailed_summary = "\n\n".join(detailed_parts) or short_summary

    topics = []
    seen_topics: set[str] = set()
    for topic in [objective, *experimental_mom.discussion_points]:
        topic = str(topic or "").strip()
        if not topic or topic.casefold() in seen_topics:
            continue
        topics.append(topic)
        seen_topics.add(topic.casefold())

    return MeetingAnalysisResult(
        cleaned_transcript=transcript_text,
        summary=MeetingSummary(
            title=title,
            short_summary=short_summary,
            detailed_summary=detailed_summary,
            topics_discussed=topics,
        ),
        key_discussion_points=[
            KeyDiscussionPoint(
                point=point,
                speakers=evidence_for_report_item(transcript_text, point)[0],
                timestamp=evidence_for_report_item(transcript_text, point)[1],
            )
            for point in experimental_mom.discussion_points
            if str(point).strip()
        ],
        decisions=[
            Decision(
                decision=decision,
                owner=(
                    evidence_for_report_item(transcript_text, decision)[0][0]
                    if evidence_for_report_item(transcript_text, decision)[0]
                    else None
                ),
                timestamp=evidence_for_report_item(transcript_text, decision)[1],
                confidence="ML",
            )
            for decision in experimental_mom.decisions
            if str(decision).strip()
        ],
        action_items=[
            ActionItem(
                task=item.task,
                owner=None if item.owner in {"", "-", "Unassigned"} else item.owner,
                due_date=extract_due_date_text(item.deadline, item.task),
                timestamp=evidence_for_report_item(transcript_text, item.task)[1],
                status=action_priority(item.task, item.deadline),
            )
            for item in experimental_mom.action_items
            if item.task.strip()
        ],
    )


def analysis_cache_key(transcript_text: str, *, audio_quality_mode: bool = False) -> str:
    # Include the formatter version so an existing Streamlit session cannot
    # reuse a cached report produced by the previous rule-based MoM generator.
    refinement = OLLAMA_GEMMA_MODEL if USE_LOCAL_GEMMA else "disabled"
    audio_marker = "/audio_quality_v1" if audio_quality_mode else ""
    cache_payload = f"experimental_formatter_v4/local_gemma={refinement}{audio_marker}\n{transcript_text.strip()}"
    return hashlib.sha256(cache_payload.encode("utf-8")).hexdigest()


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
        "Preparing Report",
        "Building Summary",
        "Organizing Minutes",
        "Finalizing Document",
        "Completed",
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
            <div class="ms-proc-title">Preparing your meeting <span class="ms-badge queue">In progress</span></div>
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
        f"""
        <div class="ms-side-shell">
          <div class="ms-side-brand">
            {brand_logo_html("ms-side-brand-logo", alt="")}
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
        f"""
        <div class="ms-empty-state">
          <div class="ms-empty-icon">{brand_logo_html("ms-empty-brand-logo", alt="")}</div>
          <h3 class="ms-empty-title">Your meeting workspace is ready</h3>
          <p class="ms-empty-copy">
            Upload audio or an existing transcript above. You will review speakers and transcript
            content before generating professional Minutes of Meeting.
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
    if st.session_state.get("recording_meeting_open"):
        render_record_meeting_card()
        return
    st.markdown(
        """
        <style>
          .ms-hero-action-anchor{height:0;margin:0}
          .st-key-hero_actions{width:min(100%,920px)!important;margin:.2rem auto 0!important}
          .st-key-hero_actions>div[data-testid="stHorizontalBlock"]{gap:10px!important}
          .st-key-hero_actions [data-testid="stColumn"]{border:1px solid #e4ddd2!important;border-radius:22px!important;background:#fffdfa!important;padding:18px 14px 14px!important;box-shadow:0 10px 24px rgba(50,45,35,.06)!important;text-align:center!important;display:flex!important;flex-direction:column!important;align-items:center!important;min-height:208px!important}
          .st-key-hero_actions [data-testid="stColumn"]>div[data-testid="stVerticalBlock"]{width:100%!important;height:100%!important;display:flex!important;flex-direction:column!important;align-items:center!important;text-align:center!important}
          .st-key-hero_actions [data-testid="stMarkdownContainer"],.st-key-hero_actions [data-testid="stMarkdownContainer"]>*{width:100%!important;text-align:center!important}
          .st-key-hero_actions .stButton{width:100%!important;margin-top:auto!important}
          .ms-home-input-icon{width:44px;height:44px;border-radius:14px;background:#f1eef6;color:#30343a;display:grid;place-items:center;margin:0 auto 10px}
          .ms-home-input-icon svg{width:21px;height:21px;display:block}
          .ms-home-input-title{font:500 18px/1.2 Fraunces,serif;color:#1f2937;margin-bottom:5px}
          .ms-home-input-sub{min-height:34px;color:#6b7280;font:400 12px/1.45 Inter,sans-serif;margin-bottom:12px}
          .st-key-hero_actions .st-key-open_audio_workflow button,.st-key-hero_actions .st-key-open_transcript_workflow button,.st-key-hero_actions .st-key-open_recording_meeting button{
            width:100%!important;height:46px!important;min-height:46px!important;border-radius:999px!important;padding:0 22px!important;
            display:flex!important;align-items:center!important;justify-content:center!important;line-height:1!important;
            font:600 13px/1 Inter,sans-serif!important;box-shadow:none!important;background:#30343a!important;border:1px solid #30343a!important;color:#fff!important;
            transition:transform 160ms ease,box-shadow 160ms ease,background-color 160ms ease!important
          }
          .st-key-hero_actions .stButton button,.st-key-hero_actions .stButton button p,.st-key-hero_actions .stButton button span,
          .st-key-hero_actions .st-key-open_audio_workflow button *,.st-key-hero_actions .st-key-open_transcript_workflow button *,.st-key-hero_actions .st-key-open_recording_meeting button *{
            color:#fff!important;-webkit-text-fill-color:#fff!important;opacity:1!important;visibility:visible!important;
          }
          .st-key-hero_actions .stButton button p{margin:0!important;font:600 13px/1 Inter,sans-serif!important;display:block!important;text-align:center!important}
          .st-key-hero_actions .stButton button:hover,.st-key-hero_actions .stButton button:focus,.st-key-hero_actions .stButton button:focus-visible,
          .st-key-hero_actions .stButton button:active,.st-key-hero_actions .stButton button:visited{
            background:#30343a!important;background-color:#30343a!important;border-color:#30343a!important;color:#fff!important;
            -webkit-text-fill-color:#fff!important;opacity:1!important;transform:translateY(-1px)!important;
            box-shadow:0 9px 20px rgba(48,52,58,.12)!important}
          .ms-hero:not(.ms-hero-after-actions){padding:2.25rem 0 .9rem!important}
          .ms-hero-after-actions{padding:1rem 0 1.75rem!important}
          .ms-hero-after-actions .ms-hero-preview{margin:1rem auto 0!important;padding:0!important;max-width:920px!important;
            background:transparent!important;border:0!important;border-radius:0!important;box-shadow:none!important}
          .ms-waveform-art{display:block;width:100%;height:150px;color:#7897b7}
          .ms-trust-bar{display:flex!important;flex-direction:column!important;align-items:center!important;justify-content:center!important;
            gap:.8rem!important;text-align:center!important;margin:1.5rem auto 0!important;padding:1.15rem 0 0!important}
          .ms-trust-tags{justify-content:center!important;gap:1rem 1.8rem!important}
          @media(max-width:700px){.st-key-open_audio_workflow button,.st-key-open_transcript_workflow button,.st-key-open_recording_meeting button{width:100%!important}
            .ms-hero:not(.ms-hero-after-actions){padding:1.7rem 0 .8rem!important}.ms-hero-after-actions{padding:.75rem 0 1.4rem!important}
            .ms-waveform-art{height:110px}}
        </style>
        <div class="ms-hero">
          <h1>The minutes write themselves.</h1>
          <p class="ms-hero-sub">
            Drop in a recording or a transcript. MeetScribe identifies speakers, extracts decisions
            and action items, and prepares professional minutes ready to review and share.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="ms-hero-action-anchor" id="workspace"></div>', unsafe_allow_html=True)
    with st.container(key="hero_actions"):
        audio_col, transcript_col, recording_col = st.columns(3, gap="small")
        with audio_col:
            st.markdown('<div class="ms-home-input-icon" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18V5l10-2v13"></path><circle cx="6" cy="18" r="3"></circle><circle cx="16" cy="16" r="3"></circle></svg></div><div class="ms-home-input-title">Upload Audio</div><div class="ms-home-input-sub">MP3, WAV, M4A, AAC, MP4</div>', unsafe_allow_html=True)
            if st.button("Upload Audio", type="primary", use_container_width=True, key="open_audio_workflow"):
                open_workflow("audio")
        with transcript_col:
            st.markdown('<div class="ms-home-input-icon" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"></path><path d="M14 2v6h6"></path></svg></div><div class="ms-home-input-title">Upload Transcript</div><div class="ms-home-input-sub">PDF, DOCX, TXT</div>', unsafe_allow_html=True)
            if st.button("Upload Transcript", use_container_width=True, key="open_transcript_workflow"):
                open_workflow("transcript")
        with recording_col:
            st.markdown('<div class="ms-home-input-icon" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="3" width="6" height="11" rx="3"></rect><path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21M8.5 21h7"></path></svg></div><div class="ms-home-input-title">Record Meeting</div><div class="ms-home-input-sub">Up to 60 minutes, right in your browser</div>', unsafe_allow_html=True)
            if st.button("Record Meeting", use_container_width=True, key="open_recording_meeting"):
                st.session_state.recording_meeting_open = True
                st.rerun()
    if st.session_state.get("recording_meeting_open"):
        render_record_meeting_card()
    st.markdown(
        """
        <div class="ms-hero ms-hero-after-actions">
          <div class="ms-hero-preview" role="img" aria-label="Meeting audio waveform">
            <svg class="ms-waveform-art" viewBox="0 0 900 150" fill="none" aria-hidden="true">
              <g stroke="currentColor" stroke-width="7" stroke-linecap="round">
                <path d="M32 66v18M54 52v46M76 35v80M98 59v32M120 23v104M142 48v54M164 31v88M186 56v38M208 38v74M230 25v100M252 49v52M274 65v20M296 40v70M318 28v94M340 55v40M362 35v80M384 50v50M406 30v90M428 58v34M450 44v62M472 21v108M494 48v54M516 62v26M538 39v72M560 52v46M582 30v90M604 57v36M626 42v66M648 27v96M670 54v42M692 37v76M714 49v52M736 24v102M758 47v56M780 61v28M802 40v70M824 31v88M846 56v38M868 45v60"/>
              </g>
            </svg>
          </div>
          <div class="ms-trust-bar">
            <span class="ms-trust-label">Built for teams that need reliable meeting records</span>
            <div class="ms-trust-tags">
              <span>Design studios</span>
              <span>Consulting firms</span>
              <span>Product teams</span>
              <span>Legal practices</span>
              <span>Research labs</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_marketing_sections() -> None:
    """Render static product education below the empty workspace."""

    st.markdown(
        """
        <section class="ms-marketing" id="workflow">
          <div class="ms-section-intro">
            <div class="ms-eyebrow"><span class="ms-eyebrow-dot"></span>Workflow</div>
            <h2>From conversation to accountable outcomes.</h2>
            <p>A guided review path keeps every stage visible while preserving control over speakers, transcript wording, and final meeting information.</p>
          </div>
          <div class="ms-process-grid" id="how-it-works">
            <article class="ms-process-card"><div class="ms-card-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m17 8-5-5-5 5M12 3v12"/></svg></div><small>STEP 01</small><h3>Upload</h3><p>Add a recording or an existing transcript through the secure workspace.</p></article>
            <article class="ms-process-card"><div class="ms-card-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="9" cy="7" r="4"/><path d="M17 11a4 4 0 1 0 0-8M2 21a7 7 0 0 1 14 0M17 14a6 6 0 0 1 5 7"/></svg></div><small>STEP 02</small><h3>Resolve speakers</h3><p>Review participant labels and establish clear attribution before analysis.</p></article>
            <article class="ms-process-card"><div class="ms-card-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20h9"/><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L8 18l-4 1 1-4Z"/></svg></div><small>STEP 03</small><h3>Review transcript</h3><p>Edit wording and meeting details before generating the official report.</p></article>
            <article class="ms-process-card"><div class="ms-card-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><path d="M14 2v6h6M8 13h8M8 17h5"/></svg></div><small>STEP 04</small><h3>Export minutes</h3><p>Inspect the structured report and download professional PDF or DOCX files.</p></article>
          </div>
        </section>
        <section class="ms-marketing" id="features">
          <div class="ms-section-intro">
            <div class="ms-eyebrow"><span class="ms-eyebrow-dot"></span>Product capabilities</div>
            <h2>Everything needed for reliable meeting records.</h2>
            <p>Purpose-built tools turn long conversations into documentation teams can review, share, and act on.</p>
          </div>
          <div class="ms-feature-grid">
            <article class="ms-feature-card"><div class="ms-card-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/></svg></div><h3>Speaker diarization</h3><p>Keep each contribution connected to the right participant across the reviewed transcript.</p></article>
            <article class="ms-feature-card"><div class="ms-card-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m9 11 3 3L22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg></div><h3>Decision detection</h3><p>Surface confirmed outcomes separately from discussion and follow-up work.</p></article>
            <article class="ms-feature-card"><div class="ms-card-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 11l3 3L22 4M3 6h7M3 12h3M3 18h7"/></svg></div><h3>Action items</h3><p>Organize assigned tasks, owners, deadlines, and priorities into clear rows.</p></article>
            <article class="ms-feature-card"><div class="ms-card-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 6h16M4 12h16M4 18h10"/></svg></div><h3>Meeting summary</h3><p>Capture the meeting’s central themes in concise, professional language.</p></article>
            <article class="ms-feature-card"><div class="ms-card-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20h9M16.5 3.5a2.12 2.12 0 0 1 3 3L8 18l-4 1 1-4Z"/></svg></div><h3>Transcript editing</h3><p>Review and correct the conversation before report generation begins.</p></article>
            <article class="ms-feature-card"><div class="ms-card-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 9V2h12v7M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"/><path d="M6 14h12v8H6z"/></svg></div><h3>PDF and DOCX export</h3><p>Deliver client-ready documents in the formats stakeholders already use.</p></article>
            <article class="ms-feature-card"><div class="ms-card-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3a6 6 0 0 0 0 12c1.7 0 3.2-.7 4.3-1.7L21 18"/><path d="M18 21l3-3-3-3"/></svg></div><h3>Local refinement</h3><p>Polish deterministic minutes locally while preserving factual source content.</p></article>
            <article class="ms-feature-card"><div class="ms-card-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z"/><path d="m9 12 2 2 4-4"/></svg></div><h3>Review-first control</h3><p>Nothing becomes a report until speakers and transcript content are confirmed.</p></article>
          </div>
        </section>
        <div id="about"></div><div id="documentation"></div>
        """,
        unsafe_allow_html=True,
    )


def render_product_landing_sections() -> None:
    """Render customer-facing landing sections matching the editorial design."""

    st.markdown(
        f"""
        <hr class="ms-section-divider">

        <section class="ms-feature-section" id="features">
          <div class="ms-feature-grid">
            <div class="ms-feature-copy">
              <div class="ms-feature-label">01 — Speaker Intelligence</div>
              <h2 class="ms-feature-heading">Identify who said what before minutes are generated.</h2>
              <p class="ms-feature-body">
                Speaker identification separates detected voices and lets you confirm each
                participant before professional minutes are generated.
              </p>
              <a class="ms-feature-link" href="#workspace">Upload a meeting →</a>
            </div>
            <div class="ms-feature-card-wrap">
              <div class="ms-feature-card-inner">
                <div class="ms-speaker-pill">
                  <span class="ms-speaker-avatar-sm">JM</span>
                  <div class="ms-speaker-pill-info">
                    <div class="ms-speaker-pill-name">Jordan Ma</div>
                    <div class="ms-speaker-pill-meta">Identified speaker</div>
                  </div>
                </div>
                <div class="ms-speaker-pill">
                  <span class="ms-speaker-avatar-sm">RP</span>
                  <div class="ms-speaker-pill-info">
                    <div class="ms-speaker-pill-name">Riya Patel</div>
                    <div class="ms-speaker-pill-meta">Identified speaker</div>
                  </div>
                </div>
                <div class="ms-speaker-pill">
                  <span class="ms-speaker-avatar-sm gray">S</span>
                  <div class="ms-speaker-pill-info">
                    <div class="ms-speaker-pill-name">Speaker 3</div>
                    <div class="ms-speaker-pill-meta">Awaiting confirmation</div>
                  </div>
                  <span class="ms-speaker-pill-badge">Unverified</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        <hr class="ms-section-divider">

        <section class="ms-feature-section">
          <div class="ms-feature-grid reverse">
            <div class="ms-feature-copy">
              <div class="ms-feature-label">02 — Structured by default</div>
              <h2 class="ms-feature-heading">Decisions and action items, pulled out on their own.</h2>
              <p class="ms-feature-body">
                No more scrolling a transcript for the one line that mattered. MeetScribe
                separates discussion from decision from next step.
              </p>
              <a class="ms-feature-link" href="#workflow">Preview a sample document →</a>
            </div>
            <div class="ms-feature-card-wrap">
              <div class="ms-feature-card-inner">
                <div class="ms-structured-item">
                  <span class="ms-structured-text">Ship pricing page redesign</span>
                  <span class="ms-badge-decided">Decided</span>
                </div>
                <div class="ms-structured-item">
                  <span class="ms-structured-text">Riya to send contract by Fri</span>
                  <span class="ms-badge-action">Action item</span>
                </div>
                <div class="ms-structured-item">
                  <span class="ms-structured-text">Revisit onboarding copy</span>
                  <span class="ms-badge-question">Open question</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        <hr class="ms-section-divider">

        <section class="ms-feature-section">
          <div class="ms-feature-grid">
            <div class="ms-feature-copy">
              <div class="ms-feature-label">03 — Edit before it ships</div>
              <h2 class="ms-feature-heading">Every line stays editable, right up to export.</h2>
              <p class="ms-feature-body">
                Correct speaker names, wording, and meeting details before the transcript
                becomes the source for your final Minutes of Meeting.
              </p>
              <a class="ms-feature-link" href="#workspace">Try the editor →</a>
            </div>
            <div class="ms-feature-card-wrap">
              <div class="ms-feature-card-inner">
                <div class="ms-edit-line">
                  <span class="ms-edit-time">00:14:32</span>
                  <span class="ms-edit-text">We'll launch the beta to 50 users first</span>
                </div>
                <div class="ms-edit-line">
                  <span class="ms-edit-time">00:14:51</span>
                  <span class="ms-edit-text highlighted">Edited for clarity</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        <hr class="ms-section-divider">

        <section class="ms-feature-section">
          <div class="ms-feature-grid reverse">
            <div class="ms-feature-copy">
              <div class="ms-feature-label">04 — Professional minutes</div>
              <h2 class="ms-feature-heading">A reviewed record, ready to circulate.</h2>
              <p class="ms-feature-body">
                Generate structured Minutes of Meeting and deliver them as PDF, editable DOCX,
                or directly through the existing email workflow.
              </p>
              <a class="ms-feature-link" href="#workflow">See the complete workflow →</a>
            </div>
            <div class="ms-feature-card-wrap">
              <div class="ms-feature-card-inner">
                <div class="ms-structured-item"><span class="ms-structured-text">Professional Minutes of Meeting</span><span class="ms-badge-decided">Ready</span></div>
                <div class="ms-structured-item"><span class="ms-structured-text">PDF export</span><span class="ms-badge-action">PDF</span></div>
                <div class="ms-structured-item"><span class="ms-structured-text">Editable document</span><span class="ms-badge-action">DOCX</span></div>
                <div class="ms-structured-item"><span class="ms-structured-text">Share with attendees</span><span class="ms-badge-question">Email</span></div>
              </div>
            </div>
          </div>
        </section>

        <hr class="ms-section-divider">

        <section class="ms-workflow-section" id="workflow">
          <div class="ms-workflow-card">
            <div class="ms-workflow-header">
              <h2>From recording to document, in four steps</h2>
              <p>A guided path from upload to professional minutes you can share.</p>
            </div>
            <div class="ms-workflow-grid">
              <div class="ms-workflow-step">
                <div class="ms-workflow-step-num">01</div>
                <h3>Upload</h3>
                <p>Drop an audio recording or paste an existing transcript. MP3, WAV, M4A, PDF, DOCX, and TXT supported.</p>
              </div>
              <div class="ms-workflow-step">
                <div class="ms-workflow-step-num">02</div>
                <h3>Confirm speakers</h3>
                <p>Review detected voices and assign names before the transcript is finalized.</p>
              </div>
              <div class="ms-workflow-step">
                <div class="ms-workflow-step-num">03</div>
                <h3>Review the transcript</h3>
                <p>Skim, search, and edit the conversation before generating your minutes.</p>
              </div>
              <div class="ms-workflow-step">
                <div class="ms-workflow-step-num">04</div>
                <h3>Generate &amp; send</h3>
                <p>Export professional minutes as PDF or DOCX, or share directly through email.</p>
              </div>
            </div>
          </div>
        </section>

        <hr class="ms-section-divider">

        <section class="ms-compare-section" id="compare">
          <div class="ms-compare-header">
            <h2>How it stacks up</h2>
            <p>Built for teams that need more than a raw transcript.</p>
          </div>
          <table class="ms-compare-table">
            <thead>
              <tr>
                <th>Capability</th>
                <th>Manual notes</th>
                <th>Generic AI notetaker</th>
                <th>MeetScribe</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Speaker identification</td>
                <td>Manual</td>
                <td>Varies by tool</td>
                <td>Detected and reviewable</td>
              </tr>
              <tr>
                <td>Decision and action extraction</td>
                <td>Manual review</td>
                <td>Varies by tool</td>
                <td>Structured automatically</td>
              </tr>
              <tr>
                <td>Transcript review</td>
                <td>Manual document editing</td>
                <td>Varies by tool</td>
                <td>Edit before generation</td>
              </tr>
              <tr>
                <td>Professional export</td>
                <td>Manual formatting</td>
                <td>Varies by tool</td>
                <td>PDF, DOCX, and email</td>
              </tr>
              <tr>
                <td>Review before minutes</td>
                <td>Manual process</td>
                <td>Varies by tool</td>
                <td>Speakers and transcript reviewed</td>
              </tr>
            </tbody>
          </table>
        </section>

        <hr class="ms-section-divider">

        <section class="ms-faq-section" id="faq">
          <div class="ms-faq-header">
            <h2>Questions worth asking</h2>
          </div>
          <div class="ms-faq-list">
            <details class="ms-faq-item" open>
              <summary>What audio formats does MeetScribe accept?<span class="ms-faq-icon">+</span></summary>
              <div class="ms-faq-answer">
                MeetScribe accepts MP3, WAV, M4A, AAC, and MP4 audio files. You can also upload
                existing transcripts in PDF, DOCX, or TXT format if you already have a written record.
              </div>
            </details>
            <details class="ms-faq-item">
              <summary>Can I edit speakers before generating minutes?<span class="ms-faq-icon">+</span></summary>
              <div class="ms-faq-answer">
                Yes. After upload, you review every detected speaker and assign names before
                the transcript is processed. Changes carry through to the final minutes.
              </div>
            </details>
            <details class="ms-faq-item">
              <summary>Can I edit the transcript before export?<span class="ms-faq-icon">+</span></summary>
              <div class="ms-faq-answer">
                Yes. The transcript review step lets you correct wording, fix names, and adjust
                meeting details before minutes are generated.
              </div>
            </details>
            <details class="ms-faq-item">
              <summary>What export formats are available?<span class="ms-faq-icon">+</span></summary>
              <div class="ms-faq-answer">
                Final minutes can be downloaded as professional PDF or editable DOCX. You can
                also share the document directly through email from within the app.
              </div>
            </details>
            <details class="ms-faq-item">
              <summary>What happens before minutes are generated?<span class="ms-faq-icon">+</span></summary>
              <div class="ms-faq-answer">
                MeetScribe guides you through speaker identification and transcript review first.
                Minutes are generated from the transcript you have reviewed.
              </div>
            </details>
          </div>
        </section>

        <div class="ms-footer-cta">
          <div class="ms-footer-cta-inner">
            <h2>Stop writing minutes by hand.</h2>
            <p>Upload audio or a transcript, review it, and generate a document worth sending.</p>
            <a class="ms-btn-primary" href="#workspace">Upload a meeting</a>
          </div>
        </div>

        <footer class="ms-site-footer">
          <div class="ms-footer-grid">
            <div class="ms-footer-brand">
              <a class="ms-nav-brand" href="#top">
                <span class="ms-logo">
                  {brand_logo_html("ms-brand-image ms-brand-image-footer", alt="")}
                </span>
                <span>MeetScribe</span>
              </a>
              <p>Meeting minutes, written the moment the call ends.</p>
            </div>
            <div class="ms-footer-col">
              <h4>Product</h4>
              <a href="#workspace">Upload</a>
              <a href="#features">Features</a>
              <a href="#workflow">Workflow</a>
            </div>
            <div class="ms-footer-col">
              <h4>Review</h4>
              <a href="#features">Speaker identification</a>
              <a href="#features">Transcript review</a>
              <a href="#faq">FAQ</a>
            </div>
            <div class="ms-footer-col">
              <h4>Export</h4>
              <a href="#features">PDF</a>
              <a href="#features">DOCX</a>
              <a href="#features">Email</a>
            </div>
          </div>
          <div class="ms-footer-bottom">
            <span>© 2026 MeetScribe</span>
            <div class="ms-footer-bottom-links">
              <a href="#privacy">Privacy</a>
              <a href="#terms">Terms</a>
            </div>
          </div>
        </footer>
        <div id="privacy"></div>
        <div id="terms"></div>
        <div id="documentation"></div>
        """,
        unsafe_allow_html=True,
    )


def empty_card(message: str) -> None:
    st.markdown(
        f"<div class='ms-empty'>{html.escape(message)}</div>",
        unsafe_allow_html=True,
    )


def compact_ui_html(markup: str) -> str:
    """Collapse template-only whitespace so Markdown always parses UI markup as HTML."""

    return re.sub(r">\s+<", "><", markup).strip()


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
    metadata.update(ner_metadata_from_transcript(result.transcript))
    return metadata


def ner_metadata_from_transcript(transcript_text: str) -> dict[str, Any]:
    """Extract meeting metadata with BERT NER and safe fallback behavior.

    Args:
        transcript_text: Transcript text from audio or transcript upload.

    Returns:
        Metadata dictionary that can be merged into session state. If BERT NER
        fails, the returned values fall back to existing participant extraction
        evidence and never interrupt the upload or MoM generation flow.
    """

    if not transcript_text or not transcript_text.strip():
        return {}

    try:
        extraction = extract_meeting_metadata(transcript_text)
    except Exception as exc:  # pragma: no cover - defensive UI safety.
        log_stage(
            "BERT NER",
            "NER metadata extraction failed; using existing fallback.",
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        fallback_participants = extract_actual_participants_from_transcript(transcript_text)
        return {
            "participant_list": fallback_participants,
            "ner_used_fallback": True,
            "ner_error": str(exc),
        }

    metadata = extraction.to_metadata_dict()
    if not metadata.get("participant_list"):
        # BERT can fail to load offline or miss speaker-only names. Existing
        # regex speaker extraction remains the non-blocking fallback so meeting
        # information still pre-fills without affecting the ML MoM pipeline.
        fallback_participants = extract_actual_participants_from_transcript(transcript_text)
        if fallback_participants:
            metadata["participant_list"] = fallback_participants
            metadata["ner_used_fallback"] = bool(metadata.get("ner_used_fallback")) or True

    log_ner_extraction(extraction, metadata)
    return metadata


def log_ner_extraction(
    extraction: EntityExtractionResult,
    metadata: dict[str, Any],
) -> None:
    """Log BERT NER metadata extraction details.

    Args:
        extraction: Structured NER extraction result.
        metadata: Session metadata produced from the extraction.
    """

    detected_entities = [
        {
            "text": entity.text,
            "type": entity.entity_group,
            "confidence": entity.confidence,
        }
        for entity in extraction.entities
    ]
    log_stage(
        "BERT NER",
        "Completed metadata extraction.",
        inference_time_seconds=round(extraction.inference_time_seconds, 4),
        detected_participants=metadata.get("participant_list", []),
        detected_entities=detected_entities,
        participant_confidence=[
            {
                "name": participant.name,
                "confidence": participant.confidence,
                "sources": participant.sources,
            }
            for participant in extraction.participants
        ],
        fallback=metadata.get("ner_used_fallback", False),
        error=metadata.get("ner_error", ""),
    )


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


def extract_actual_participants_from_transcript(transcript_text: str) -> list[str]:
    participants: list[str] = []
    seen: set[str] = set()
    speaker_patterns = (
        re.compile(
            r"^\s*(?:\[\s*\d{1,2}:\d{2}(?::\d{2})?\s*\]\s*)"
            r"(?P<speaker>[^:\[\]\(\)\n]{1,80})\s*:",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*(?:\d{1,2}:\d{2}(?::\d{2})?\s+)"
            r"(?P<speaker>[^:\[\]\(\)\n]{1,80})\s*:",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*(?P<speaker>[^:\[\]\(\)\n]{1,80})\s*"
            r"(?:\[\s*\d{1,2}:\d{2}(?::\d{2})?\s*\]|\(\s*\d{1,2}:\d{2}(?::\d{2})?\s*\))\s*:?",
            re.IGNORECASE,
        ),
        re.compile(r"^\s*(?P<speaker>[^:\[\]\(\)\n]{1,80})\s*:", re.IGNORECASE),
    )

    for raw_line in transcript_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        speaker = ""
        for pattern in speaker_patterns:
            match = pattern.match(line)
            if match:
                speaker = match.group("speaker").strip()
                break
        if not speaker:
            continue
        speaker = re.sub(r"\s+", " ", speaker).strip(" -")
        if not speaker or re.fullmatch(r"(?i)speaker\s+[A-Za-z0-9]+", speaker):
            continue
        key = speaker.casefold()
        if key in seen:
            continue
        participants.append(speaker)
        seen.add(key)
    return participants


def participant_value_is_empty_or_generic(value: str) -> bool:
    names = [item.strip() for item in re.split(r"[\n,]+", value or "") if item.strip()]
    if not names:
        return True
    return all(re.fullmatch(r"(?i)speaker\s+[A-Za-z0-9]+", name) for name in names)


def initialize_meeting_info(result: TranscriptionResult | None = None) -> None:
    current = dict(st.session_state.get("meeting_info", {}))
    if st.session_state.get("meeting_info_initialized") and current:
        return

    metadata = st.session_state.get("meeting_metadata", {})
    participants = participants_from_current_context(result)
    info = {
        "meeting_title": current.get("meeting_title") or default_meeting_title(),
        "meeting_date": current.get("meeting_date")
        or metadata.get("meeting_date")
        or time.strftime("%Y-%m-%d"),
        "meeting_time": current.get("meeting_time") or metadata.get("meeting_time", ""),
        "organization": current.get("organization") or metadata.get("organization", ""),
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
    participant_value = info.get("participants", "")
    participants = (
        participant_value if not participant_value_is_empty_or_generic(participant_value) else ""
        or ", ".join(extract_actual_participants_from_transcript(st.session_state.get("transcript_text", "")))
        or ", ".join(participants_from_current_context())
    )
    duration = (
        info.get("duration")
        or metadata.get("meeting_duration")
        or format_duration_for_report(metadata.get("duration_seconds"))
    )
    source_file = info.get("source_file") or st.session_state.get("uploaded_filename", "")
    return {
        "Meeting Title": info.get("meeting_title", ""),
        "Date": info.get("meeting_date") or metadata.get("meeting_date", ""),
        "Time": info.get("meeting_time") or metadata.get("meeting_time", ""),
        "Organization / Company": info.get("organization") or metadata.get("organization", ""),
        "Project Name": info.get("project_name", ""),
        "Prepared By": info.get("prepared_by", ""),
        "Participants": participants,
        "Attendees": participants,
        "Duration": duration or "",
        "Source File": source_file,
        "Generated On": metadata.get("generated_on", "") or generated_on_display(),
    }


def render_meeting_information_panel(result: TranscriptionResult | None = None) -> None:
    initialize_meeting_info(result)
    info = dict(st.session_state.get("meeting_info", {}))
    time_or_duration = info.get("meeting_time") or info.get("duration", "")
    with st.container(key="workflow_meeting_information"):
        title_col, date_col, time_col = st.columns(3)
        with title_col:
            meeting_title = st.text_input(
                "Meeting Title", value=info.get("meeting_title", ""), key="meeting_info_title_input")
        with date_col:
            meeting_date = st.text_input(
                "Meeting Date", value=info.get("meeting_date", ""), key="meeting_info_date_input")
        with time_col:
            meeting_time = st.text_input(
                "Duration", value=time_or_duration, key="meeting_info_time_input")

        participants_col, project_col, prepared_col = st.columns(3)
        with participants_col:
            participants = st.text_area(
                "Participants", value=info.get("participants", ""), height=82,
                key="meeting_info_participants_input")
        with project_col:
            project_name = st.text_input(
                "Project Name", value=info.get("project_name", ""),
                key="meeting_info_project_name_input")
        with prepared_col:
            prepared_by = st.text_input(
                "Prepared By", value=info.get("prepared_by", ""),
                key="meeting_info_prepared_by_input")

    updated = {
        **info,
        "meeting_title": meeting_title.strip(),
        "meeting_date": meeting_date.strip(),
        "project_name": project_name.strip(),
        "prepared_by": prepared_by.strip(),
        "participants": participants.strip(),
        "source_file": st.session_state.get("uploaded_filename", ""),
    }
    if info.get("meeting_time") or not info.get("duration"):
        updated["meeting_time"] = meeting_time.strip()
    else:
        updated["duration"] = meeting_time.strip()
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


def transcript_turns_from_text(transcript_text: str) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    current_speaker = ""
    current_timestamp = ""
    current_lines: list[str] = []
    header_patterns = (
        re.compile(
            r"^\s*\[\s*(?P<time>\d{1,2}:\d{2}(?::\d{2})?)\s*\]\s*(?P<speaker>[^:]+)\s*:\s*(?P<text>.*)$"
        ),
        re.compile(
            r"^\s*(?P<time>\d{1,2}:\d{2}(?::\d{2})?)\s+(?P<speaker>[^:]+)\s*:\s*(?P<text>.*)$"
        ),
        re.compile(
            r"^\s*(?P<speaker>[^:\[\]\(\)\n]+?)\s*(?:\[(?P<time>[^\]]+)\]|\((?P<time_paren>[^\)]+)\))\s*:?\s*(?P<text>.*)$"
        ),
        re.compile(r"^\s*(?P<speaker>[^:\[\]\(\)\n]+?)\s*:\s*(?P<text>.*)$"),
    )

    def flush() -> None:
        nonlocal current_speaker, current_timestamp, current_lines
        body = "\n".join(line.rstrip() for line in current_lines if line.strip()).strip()
        if current_speaker and body:
            turns.append(
                {
                    "speaker": current_speaker,
                    "timestamp": current_timestamp or "--:--",
                    "text": body,
                }
            )
        current_speaker = ""
        current_timestamp = ""
        current_lines = []

    for raw_line in transcript_text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        matched_header = None
        for pattern in header_patterns:
            matched_header = pattern.match(line)
            if matched_header:
                break
        if matched_header:
            flush()
            current_speaker = re.sub(r"\s+", " ", matched_header.group("speaker")).strip()
            current_timestamp = (
                matched_header.groupdict().get("time")
                or matched_header.groupdict().get("time_paren")
                or "--:--"
            ).strip()
            inline_text = (matched_header.groupdict().get("text") or "").strip()
            if inline_text:
                current_lines.append(inline_text)
            continue
        if current_speaker:
            current_lines.append(line)
        elif line.strip():
            current_speaker = "Transcript"
            current_timestamp = "--:--"
            current_lines.append(line)
    flush()
    return turns


def render_transcript_turn_cards(turns: list[dict[str, str]]) -> None:
    seen: dict[str, str] = {}
    colour_cycle = ["s1", "s2", "s3", "s4"]
    rows_html = ""
    for turn in turns:
        label = turn.get("speaker", "").strip() or "Speaker"
        if label not in seen:
            seen[label] = colour_cycle[len(seen) % len(colour_cycle)]
        cls = seen[label]
        timestamp = turn.get("timestamp", "").strip() or "--:--"
        text = turn.get("text", "").strip()
        initial = next((char for char in label if char.isalnum()), "S").upper()
        rows_html += f"""
        <div class="ms-tr-row">
          <div class="ms-tr-left">
            <span class="ms-speaker-avatar {cls}">{html.escape(initial)}</span>
            <span class="ms-speaker-meta">
              <span class="ms-speaker-name">{html.escape(label)}</span>
              <span class="ms-tr-timestamp">{html.escape(timestamp)}</span>
            </span>
          </div>
          <div class="ms-tr-right">
            <span class="ms-tr-text">{html.escape(text)}</span>
          </div>
        </div>"""

    transcript_html = f"""
    <style>
      :root {{
        --pink-soft: #EEF2FF;
        --pink-mid: #E7E7E7;
        --pink-deep: #5B6EF5;
        --green-soft: #E6F6ED;
        --blue-soft: #EAF2FF;
        --lav-soft: #F3F0FF;
        --warm: #1F2937;
        --warm-2: #4B5563;
        --warm-4: #6B7280;
        --border-soft: #E7E7E7;
      }}
      body {{
        margin: 0;
        background: transparent;
        font-family: "Inter", "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      }}
      .ms-transcript-scroll {{
        box-sizing: border-box;
        height: 600px;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        gap: 0.7rem;
        padding: 0 0.35rem 0.35rem 0;
        scroll-behavior: smooth;
        scrollbar-width: thin;
        scrollbar-color: var(--warm-4) transparent;
      }}
      .ms-transcript-scroll::-webkit-scrollbar {{ width: 4px; }}
      .ms-transcript-scroll::-webkit-scrollbar-track {{ background: transparent; }}
      .ms-transcript-scroll::-webkit-scrollbar-thumb {{
        background: var(--warm-4);
        border-radius: 999px;
      }}
      .ms-transcript-sticky-head {{
        position: sticky;
        top: 0;
        z-index: 3;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem 0.9rem;
        border: 1px solid var(--pink-mid);
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.96);
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 22px rgba(17,24,39,0.06);
      }}
      .ms-transcript-sticky-head span:first-child {{
        color: var(--warm);
        font-size: 0.82rem;
        font-weight: 850;
      }}
      .ms-transcript-sticky-head span:last-child {{
        color: var(--warm-4);
        font-size: 0.68rem;
        font-weight: 750;
      }}
      .ms-tr-row {{
        display: grid;
        grid-template-columns: 150px minmax(0, 1fr);
        border: 1px solid var(--pink-mid);
        border-radius: 16px;
        background: #FFFFFF;
        box-shadow: 0 5px 16px rgba(17,24,39,0.045);
        overflow: visible;
        transition: background 180ms, border-color 180ms, box-shadow 180ms, transform 180ms;
      }}
      .ms-tr-row:hover {{
        background: #FCFDFF;
        border-color: #CDD5EA;
        box-shadow: 0 12px 28px rgba(17,24,39,0.075);
        transform: translateY(-1px);
      }}
      .ms-tr-left {{
        padding: 0.95rem 0.8rem 0.95rem 0.95rem;
        border-right: 1px solid var(--border-soft);
        display: grid;
        grid-template-columns: 34px minmax(0, 1fr);
        gap: 0.55rem;
        align-items: start;
        min-width: 0;
      }}
      .ms-speaker-avatar {{
        width: 34px;
        height: 34px;
        border-radius: 12px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.76rem;
        font-weight: 900;
        flex: 0 0 auto;
      }}
      .ms-speaker-avatar.s1 {{ background: var(--pink-soft); color: var(--pink-deep); }}
      .ms-speaker-avatar.s2 {{ background: var(--green-soft); color: #065F46; }}
      .ms-speaker-avatar.s3 {{ background: var(--blue-soft); color: #1E40AF; }}
      .ms-speaker-avatar.s4 {{ background: var(--lav-soft); color: #5B21B6; }}
      .ms-speaker-meta {{
        display: flex;
        flex-direction: column;
        gap: 0.2rem;
        min-width: 0;
      }}
      .ms-speaker-name {{
        color: var(--warm);
        font-size: 0.78rem;
        font-weight: 850;
        line-height: 1.25;
        overflow-wrap: anywhere;
      }}
      .ms-tr-timestamp {{
        color: var(--warm-4);
        font-size: 0.65rem;
        font-variant-numeric: tabular-nums;
        font-weight: 500;
      }}
      .ms-tr-right {{
        padding: 0.9rem 1rem;
        display: block;
        min-width: 0;
      }}
      .ms-tr-text {{
        color: var(--warm-2);
        font-size: 0.88rem;
        line-height: 1.6;
        display: block;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
        word-break: normal;
      }}
      @media (max-width: 720px) {{
        .ms-transcript-scroll {{ height: 560px; gap: 0.62rem; }}
        .ms-tr-row {{ grid-template-columns: 1fr; }}
        .ms-tr-left {{
          border-right: 0;
          border-bottom: 1px solid var(--border-soft);
          grid-template-columns: 34px minmax(0, 1fr);
        }}
        .ms-tr-right {{ padding: 0.85rem 0.95rem 0.95rem; }}
      }}
    </style>
    <div class="ms-transcript-scroll">
      <div class="ms-transcript-sticky-head">
        <span>Reviewed Transcript</span>
        <span>Speaker-wise view</span>
      </div>
      {rows_html}
    </div>
    """
    components.html(transcript_html, height=620, scrolling=False)


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

    parsed_turns = transcript_turns_from_text(result.transcript)
    if parsed_turns and (not result.segments or len(result.segments) <= 1):
        render_transcript_turn_cards(parsed_turns)
        return

    if result.segments:
        turns = [
            {
                "speaker": speaker_label(segment, mapping),
                "timestamp": (
                    f"{format_timestamp(segment.start_time_seconds)} - "
                    f"{format_timestamp(segment.end_time_seconds)}"
                ),
                "text": segment.transcript,
            }
            for segment in result.segments
        ]
        render_transcript_turn_cards(turns)
        return

    if parsed_turns:
        render_transcript_turn_cards(parsed_turns)
        return

    st.text_area(
        "Transcript",
        value=result.transcript,
        height=480,
        label_visibility="collapsed",
    )


DEADLINE_TEXT_PATTERN = re.compile(
    r"(?i)\b(today|tomorrow|monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday|next sprint|this sprint|within two weeks|end of month|"
    r"end of day|eod|next week|this week|upcoming release|before deployment|"
    r"sprint\s+\d+|by\s+[A-Za-z]+(?:\s+[A-Za-z]+)?|"
    r"before\s+[A-Za-z]+(?:\s+[A-Za-z]+)?|"
    r"\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\b"
)
HIGH_PRIORITY_PATTERN = re.compile(
    r"(?i)\b(urgent|critical|immediately|today|asap|release blocker|high priority)\b"
)
MEDIUM_PRIORITY_PATTERN = re.compile(
    r"(?i)\b(next sprint|this week|upcoming release|soon|planned)\b"
)


def generated_on_display() -> str:
    """Return the report generation timestamp in the required display format."""

    return f"{time.strftime('%d %B %Y')}\n{time.strftime('%I:%M %p')}"


def transcript_keywords(value: str) -> set[str]:
    """Extract comparison keywords for deterministic transcript evidence lookup."""

    ignored = {
        "the", "and", "for", "with", "that", "this", "was", "were", "will",
        "has", "have", "been", "team", "meeting", "discussed", "reviewed",
        "evaluated", "approved", "implemented", "implementation", "during",
        "before", "after", "from", "into", "about", "action", "items",
    }
    return {
        token
        for token in re.findall(r"[A-Za-z][A-Za-z0-9]+", value.casefold())
        if len(token) > 2 and token not in ignored
    }


def evidence_for_report_item(
    transcript_text: str,
    report_text: str,
) -> tuple[list[str], str | None]:
    """Find the edited-transcript speaker/timestamp most related to a report item.

    The formatter produces polished bullets that may not exactly copy transcript
    sentences. This deterministic lookup scores final edited transcript turns by
    keyword overlap, keeping speaker names and timestamps aligned with the
    transcript that the user approved.
    """

    target_keywords = transcript_keywords(report_text)
    if not target_keywords:
        return [], None

    best_score = 0
    best_turn: dict[str, str] | None = None
    for turn in transcript_turns_from_text(transcript_text):
        turn_text = " ".join(
            [
                turn.get("speaker", ""),
                turn.get("timestamp", ""),
                turn.get("text", ""),
            ]
        )
        turn_keywords = transcript_keywords(turn_text)
        if not turn_keywords:
            continue
        score = len(target_keywords & turn_keywords)
        if score > best_score:
            best_score = score
            best_turn = turn

    if best_turn is None or best_score == 0:
        return [], None

    speaker = (best_turn.get("speaker") or "").strip()
    timestamp = (best_turn.get("timestamp") or "").strip()
    speakers = [speaker] if speaker and not re.fullmatch(r"(?i)speaker\s+[A-Za-z0-9]+", speaker) else []
    return speakers, None if timestamp in {"", "--:--"} else timestamp


def extract_due_date_text(*values: str | None) -> str:
    """Extract a deterministic due-date phrase from action-related text."""

    text = " ".join(str(value or "") for value in values)
    match = DEADLINE_TEXT_PATTERN.search(text)
    if not match:
        return "Not Mentioned"
    return re.sub(r"\s+", " ", match.group(0)).strip().title()


def action_priority(task: str, due_date: str | None = None) -> str:
    """Return deterministic action-item priority without using AI."""

    text = f"{task or ''} {due_date or ''}"
    if HIGH_PRIORITY_PATTERN.search(text):
        return "High"
    if MEDIUM_PRIORITY_PATTERN.search(text):
        return "Medium"
    return "Low"


def render_generated_report_header(analysis: MeetingAnalysisResult) -> None:
    """Render the generated report as a product workspace using existing data only."""

    report_info = meeting_info_for_export()
    metrics = st.session_state.get("success_metrics") or {}
    title = (
        report_info.get("Meeting Title")
        or analysis.summary.title
        or "Meeting Minutes"
    )
    participants_text = str(report_info.get("Participants") or report_info.get("Attendees") or "")
    participant_count = len(
        [name for name in re.split(r"[,;\n]+", participants_text) if name.strip()]
    )
    generated_on = str(report_info.get("Generated On") or generated_on_display())
    date_text = str(report_info.get("Date") or "Not specified")
    duration = str(report_info.get("Duration") or metrics.get("Duration") or "Not available")
    prepared_by = str(report_info.get("Prepared By") or "MeetScribe")

    st.html(
        compact_ui_html(
            f"""
        <div class="ms-report-shell">
          <header class="ms-report-hero">
            <div class="ms-report-eyebrow">
              <span class="ms-success-badge">&#10003; Meeting minutes generated successfully</span>
              <span class="ms-generated-time">Generated {html.escape(generated_on)}</span>
            </div>
            <h2 class="ms-report-title">{html.escape(str(title))}</h2>
            <p class="ms-report-subtitle">
              {participant_count} participant{"s" if participant_count != 1 else ""} ·
              {html.escape(date_text)} · Ready to review and share
            </p>
            <nav class="ms-report-actions" aria-label="Report actions">
              <a class="ms-report-action" href="#downloads">&#8595; Download PDF</a>
              <a class="ms-report-action" href="#downloads">&#8595; Download DOCX</a>
              <a class="ms-report-action" href="#summary-heading">Read summary</a>
              <a class="ms-report-action" href="#source-transcript">View transcript</a>
            </nav>
          </header>
          <section class="ms-report-section" aria-labelledby="meeting-information-heading">
            <div class="ms-report-section-head">
              <h3 id="meeting-information-heading">Meeting Information</h3>
              <span>Report details</span>
            </div>
            <div class="ms-info-grid">
              <div class="ms-info-card">
                <div class="ms-info-label">Meeting date</div>
                <div class="ms-info-value">{html.escape(date_text)}</div>
              </div>
              <div class="ms-info-card">
                <div class="ms-info-label">Participants</div>
                <div class="ms-info-value">{participant_count} attendee{"s" if participant_count != 1 else ""}</div>
              </div>
              <div class="ms-info-card">
                <div class="ms-info-label">Duration</div>
                <div class="ms-info-value">{html.escape(duration)}</div>
              </div>
              <div class="ms-info-card">
                <div class="ms-info-label">Prepared by</div>
                <div class="ms-info-value">{html.escape(prepared_by)}</div>
              </div>
            </div>
          </section>
        </div>
        """
        ),
    )


def render_summary_tab(analysis: MeetingAnalysisResult) -> None:
    summary = analysis.summary
    topics_html = "".join(
        f"<span>{html.escape(topic)}</span>"
        for topic in summary.topics_discussed
    ) or "<span>No topics captured</span>"

    st.html(
        compact_ui_html(
            f"""
        <section class="ms-report-section" aria-labelledby="summary-heading">
          <div class="ms-report-section-head">
            <h3 id="summary-heading">Executive Summary</h3>
            <span>At a glance</span>
          </div>
          <div class="ms-summary-editorial">
            <div class="ms-summary-kicker">{html.escape(summary.title or "Meeting overview")}</div>
            <p>{html.escape(summary.short_summary)}</p>
            <div class="ms-summary-topics">{topics_html}</div>
          </div>
        </section>
        """
        ),
    )


def render_key_points_tab(analysis: MeetingAnalysisResult) -> None:
    if not analysis.key_discussion_points:
        empty_card("No key discussion points were extracted.")
        return

    cards: list[str] = []
    for index, item in enumerate(analysis.key_discussion_points, start=1):
        speakers = ", ".join(item.speakers)
        timestamp = item.timestamp or "Not Mentioned"
        topic_title = re.split(r"(?<=[.!?])\s+|:\s+", item.point, maxsplit=1)[0].strip()
        if len(topic_title) > 76:
            topic_title = topic_title[:73].rstrip() + "..."
        cards.append(
            compact_ui_html(
                f"""
            <details class="ms-discussion-card">
              <summary>
                <span class="ms-discussion-index">{index:02d}</span>
                <span>{html.escape(topic_title or f"Discussion {index}")}</span>
                <span class="ms-discussion-chevron">&#8964;</span>
              </summary>
              <div class="ms-discussion-body">
                {html.escape(item.point)}
                <div class="ms-detail-meta">
                  <span>Speaker · {html.escape(speakers or "Not Mentioned")}</span>
                  <span>Time · {html.escape(timestamp)}</span>
                </div>
              </div>
            </details>
            """
            ),
        )
    cards_html = "".join(cards)
    st.html(
        compact_ui_html(
            f"""
        <section class="ms-report-section" aria-labelledby="discussion-heading">
          <div class="ms-report-section-head">
            <h3 id="discussion-heading">Discussion Topics</h3>
            <span>{len(cards)} captured</span>
          </div>
          <div class="ms-discussion-list">{cards_html}</div>
        </section>
        """
        ),
    )


def render_decisions_tab(analysis: MeetingAnalysisResult) -> None:
    if not analysis.decisions:
        empty_card("No decisions were extracted.")
        return

    cards: list[str] = []
    for item in analysis.decisions:
        owner = item.owner or "Unassigned"
        timestamp = item.timestamp or "--:--"
        cards.append(
            compact_ui_html(
                f"""
            <article class="ms-decision-card">
              <div class="ms-decision-check">&#10003;</div>
              <div>
                <div class="ms-decision-text">{html.escape(item.decision)}</div>
                <div class="ms-detail-meta">
                  <span>Owner · {html.escape(owner)}</span>
                  <span>Time · {html.escape(timestamp)}</span>
                </div>
              </div>
              <span class="ms-confirmed-chip">Confirmed</span>
            </article>
            """
            ),
        )
    cards_html = "".join(cards)
    st.html(
        compact_ui_html(
            f"""
        <section class="ms-report-section" aria-labelledby="decisions-heading">
          <div class="ms-report-section-head">
            <h3 id="decisions-heading">Decisions</h3>
            <span>{len(cards)} confirmed</span>
          </div>
          <div class="ms-decision-list">{cards_html}</div>
        </section>
        """
        ),
    )


def render_action_items_tab(analysis: MeetingAnalysisResult) -> None:
    if not analysis.action_items:
        empty_card("No action items were extracted.")
        return

    cards: list[str] = []
    for item in analysis.action_items:
        owner = item.owner or "Unassigned"
        due_date = item.due_date or "Not Mentioned"
        timestamp = item.timestamp or "Not Mentioned"
        priority = action_priority(item.task, due_date)
        initials = "".join(part[0] for part in owner.split()[:2]).upper()[:2] or "—"
        priority_class = priority.lower()
        cards.append(
            compact_ui_html(
                f"""
            <article class="ms-task-card">
              <div>
                <div class="ms-task-title">{html.escape(item.task)}</div>
                <div class="ms-task-meta">
                  <span class="ms-task-chip">Due · {html.escape(due_date)}</span>
                  <span class="ms-task-chip ms-priority-{priority_class}">{html.escape(priority)} priority</span>
                  <span class="ms-task-chip">Open</span>
                  <span class="ms-task-chip">Time · {html.escape(timestamp)}</span>
                </div>
              </div>
              <div class="ms-owner">
                <span class="ms-owner-avatar">{html.escape(initials)}</span>
                <span>{html.escape(owner)}</span>
              </div>
            </article>
            """
            ),
        )
    cards_html = "".join(cards)
    st.html(
        compact_ui_html(
            f"""
        <section class="ms-report-section" aria-labelledby="actions-heading">
          <div class="ms-report-section-head">
            <h3 id="actions-heading">Action Items</h3>
            <span>{len(cards)} tasks</span>
          </div>
          <div class="ms-task-list">{cards_html}</div>
        </section>
        """
        ),
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
    download_name: str | None = None,
) -> None:
    st.download_button(
        label,
        data=export_path.read_bytes(),
        file_name=download_name or export_path.name,
        mime=mime,
        key=key,
        use_container_width=True,
        on_click=toast_download_success,
        args=(success_message,),
    )


def safe_mom_download_filename(extension: str) -> str:
    """Return a cross-platform safe MoM download filename from meeting title."""

    title = (
        st.session_state.get("meeting_info", {}).get("meeting_title")
        or meeting_info_for_export().get("Meeting Title")
        or "Meeting"
    )
    cleaned = re.sub(r"[<>:\"/\\|?*\x00-\x1F]+", " ", str(title))
    cleaned = re.sub(r"\s+", "_", cleaned).strip("._ ")
    if not cleaned:
        cleaned = "Meeting"
    return f"{cleaned}_MoM.{extension.lstrip('.')}"


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
              {brand_logo_html("ms-context-brand-logo", alt="")}
              <div class="ms-email-compose-hdr">Send minutes by email</div>
              <p class="ms-email-compose-sub">
                Review the message and attachment before sending.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.form("send_mom_email_form"):
            st.markdown("<div class='ms-email-field-label'>Recipients</div>", unsafe_allow_html=True)
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
            st.markdown("<div class='ms-email-field-label'>Preview</div>", unsafe_allow_html=True)
            message = st.text_area(
                "Message",
                value=default_email_message(),
                height=180,
                key="email_message_input",
                label_visibility="collapsed",
            )
            st.markdown(
                f"""
                <div class="ms-email-field-label">Attachment</div>
                <div class="ms-attachment-preview">
                  <div class="ms-attachment-preview-icon">PDF</div>
                  <div>
                    <div class="ms-attachment-preview-name">{html.escape(attachment_name)}</div>
                    <div class="ms-attachment-preview-meta">PDF attached automatically</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            cancel_col, send_col = st.columns(2, vertical_alignment="center")
            with cancel_col:
                cancel_clicked = st.form_submit_button(
                    "Cancel",
                    use_container_width=False,
                )
            with send_col:
                send_clicked = st.form_submit_button(
                    "Send Minutes →",
                    type="primary",
                    use_container_width=True,
                )

    if cancel_clicked:
        st.session_state.workflow_stage = "export"
        st.rerun()

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
        f"""
        <div class="ms-export-section" id="downloads">
          <div class="ms-export-wrap">
            <div class="ms-export-hdr">
              <div class="ms-export-hdr-icon">&#8659;</div>
              <div>
                {brand_logo_html("ms-context-brand-logo", alt="")}
                <div class="ms-export-hdr-title">Send these minutes anywhere</div>
                <p class="ms-export-sub">Same document, three ways to share it.</p>
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
              <div class="ms-export-option-title">Export as PDF</div>
              <div class="ms-export-option-desc">A print-ready document, formatted exactly as shown here.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_download_button(
            label="Download PDF →",
            export_path=mom_pdf_path,
            mime="application/pdf",
            key="download_mom_pdf",
            success_message="Download started",
            download_name=safe_mom_download_filename("pdf"),
        )

    with docx_col:
        st.markdown(
            """
            <div class="ms-export-option">
              <div class="ms-export-option-icon docx">DOC</div>
              <div class="ms-export-option-title">Export as Word</div>
              <div class="ms-export-option-desc">An editable .docx you can adapt in your own template.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_download_button(
            label="Download DOCX →",
            export_path=mom_docx_path,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            key="download_mom_docx",
            success_message="Download started",
            download_name=safe_mom_download_filename("docx"),
        )

    with email_col:
        st.markdown(
            """
            <div class="ms-export-option ms-export-email-wrap">
              <div class="ms-export-option-icon email">✉</div>
              <div class="ms-export-option-title">Send by email</div>
              <div class="ms-export-option-desc">Deliver the minutes directly to everyone who attended.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(
            "Compose Email →",
            key="show_send_email_form",
            use_container_width=True,
        ):
            st.session_state.workflow_stage = "email"
            st.rerun()


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
    audio_quality_mode = st.session_state.get("workflow_source") == "audio"
    cache_key = analysis_cache_key(
        transcript_text,
        audio_quality_mode=audio_quality_mode,
    )

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

        def stop_with_analysis_error(message: str) -> None:
            if progress is None:
                analysis_progress.empty()
            st.session_state.analysis_result = None
            st.session_state.analysis_error = message
            log_stage("Meeting analysis", "ML analysis stopped.", error=message)
            st.error(message)

        log_stage(
            "Meeting analysis",
            "Initializing ML meeting analysis pipeline.",
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

        # Parse once here only for existing meeting metadata enrichment. The
        # reusable experimental integration performs its own parser ->
        # preprocessing -> feature extraction -> MiniLM -> ANN path internally.
        parsed_transcript = parse_transcript(transcript_text)
        if not parsed_transcript.is_valid:
            stop_with_analysis_error(
                parsed_transcript.validation_message
                or "We could not identify speaker information in this transcript."
            )
            return None

        analysis_progress.progress(75, text="Generating Meeting Notes")
        if status_placeholder is not None and started_at is not None:
            render_stage_status(
                status_placeholder,
                active_index=3,
                started_at=started_at,
                note="Organizing your reviewed meeting into a clear, professional report.",
            )
        update_elapsed(elapsed_placeholder, started_at)

        current_info_for_generation = st.session_state.get("meeting_info", {})
        participant_text = str(current_info_for_generation.get("participants", "")).strip()
        manual_participants = [
            participant.strip()
            for participant in re.split(r"[,;\n]+", participant_text)
            if participant.strip()
        ]
        generated = generate_mom(
            transcript_text,
            meeting_title=(
                str(current_info_for_generation.get("meeting_title") or "").strip()
                or default_meeting_title()
            ),
            meeting_date=str(current_info_for_generation.get("meeting_date") or "").strip(),
            participants=manual_participants or None,
            audio_quality_mode=audio_quality_mode,
        )
        if not generated.is_valid:
            stop_with_analysis_error(
                generated.error_message
                or "The experimental MoM formatter could not generate meeting notes."
            )
            return None

        analysis = experimental_mom_to_analysis_result(
            transcript_text=transcript_text,
            generated=generated,
            audio_quality_mode=audio_quality_mode,
        )

        log_stage(
            "Meeting analysis",
            "Completed ML meeting analysis pipeline.",
            sentences=generated.sentence_count,
            predictions=len(generated.predictions or []),
            formatter="experimental_formatter_v4",
            local_gemma_applied=generated.gemma_refinement_applied,
            local_gemma_model=generated.gemma_model,
            local_gemma_fallback=bool(generated.gemma_error),
        )
        meeting_metadata = {
            **st.session_state.get("meeting_metadata", {}),
            "source_file": st.session_state.get("uploaded_filename", ""),
            "ml_pipeline": "parser_preprocessing_embeddings_ann_experimental_formatter_v4",
            "ml_sentence_count": generated.sentence_count,
            "ml_prediction_count": len(generated.predictions or []),
            "generated_on": generated_on_display(),
        }
        parsed_participants = participants_from_transcript_turns(parsed_transcript.turns)
        formatter_participants = (
            generated.experimental_mom.participants if generated.experimental_mom else []
        )
        participant_candidates = parsed_participants or formatter_participants
        if participant_candidates:
            meeting_metadata["participant_list"] = participant_candidates
        st.session_state.meeting_metadata = meeting_metadata
        initialize_meeting_info(st.session_state.get("transcript_result"))
        current_info = dict(st.session_state.get("meeting_info", {}))
        if not current_info.get("duration") and meeting_metadata.get("meeting_duration"):
            current_info["duration"] = meeting_metadata["meeting_duration"]
        if participant_value_is_empty_or_generic(current_info.get("participants", "")):
            transcript_participants = extract_actual_participants_from_transcript(transcript_text)
            if transcript_participants:
                current_info["participants"] = ", ".join(transcript_participants)
            elif meeting_metadata.get("participant_list"):
                current_info["participants"] = ", ".join(
                    str(item).strip()
                    for item in meeting_metadata.get("participant_list", [])
                    if str(item).strip()
                    and not re.fullmatch(r"(?i)speaker\s+[A-Za-z0-9]+", str(item).strip())
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
                min-height: 36px;
                border: 1px solid #D6DCF8;
                border-radius: 10px;
                background: #F7F8FF;
                color: #4A5DDE;
                cursor: pointer;
                font: 650 12px Inter, system-ui, sans-serif;
                padding: 0.48rem 0.75rem;
                letter-spacing: 0.01em;
                transition: border-color 160ms ease, background 160ms ease, transform 160ms ease, box-shadow 160ms ease;
                display: inline-flex; align-items: center; gap: 6px;
            "
            onmouseover="this.style.borderColor='#BBC5F5';this.style.background='#EEF2FF';this.style.transform='translateY(-1px)';this.style.boxShadow='0 6px 14px rgba(31,41,55,.07)';"
            onmouseout="this.style.borderColor='#D6DCF8';this.style.background='#F7F8FF';this.style.transform='none';this.style.boxShadow='none';"
            onmousedown="this.style.transform='scale(.985)';"
            onmouseup="this.style.transform='translateY(-1px)';"
            type="button"
        >
            Copy transcript
        </button>
        <span
            id="copy-status"
            style="
                color: #4A5DDE;
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
                speaker_segments = [
                    segment
                    for segment in result.segments
                    if speaker_label(segment) == label
                ]
                speaker_words = sum(
                    len(re.findall(r"\b\w+\b", segment.transcript))
                    for segment in speaker_segments
                )
                st.markdown(
                    f"""
                    <div class="ms-speaker-row">
                      <div class="ms-speaker-avatar">{html.escape(initials)}</div>
                      <div>
                        <div class="ms-speaker-label">{html.escape(str(label))}</div>
                        <div class="ms-speaker-badge-text">Editable participant name</div>
                        <div class="ms-speaker-stats">
                          <span>{len(speaker_segments)} contributions</span>
                          <span>{speaker_words} words</span>
                        </div>
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
        initial = next((character for character in speaker if character.isalnum()), "S").upper()
        displayed_text = body or header
        rows.append(
            f"""
            <div class="ms-convo-row" data-initial="{html.escape(initial)}">
              <div class="ms-convo-head">
                <span class="ms-convo-speaker">{html.escape(speaker)}</span>
                <span class="ms-toolbar-chip">Speaker</span>
                <span class="ms-time-chip">{html.escape(timestamp or "--:--")}</span>
              </div>
              <div class="ms-convo-text">{html.escape(displayed_text)}</div>
              <div class="ms-convo-meta">{len(displayed_text)} characters</div>
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

    transcript_turns = transcript_turns_from_text(transcript_text)
    transcript_speakers = {
        turn.get("speaker", "").strip()
        for turn in transcript_turns
        if turn.get("speaker", "").strip()
    }
    word_count = len(re.findall(r"\b\w+\b", transcript_text))

    with st.container(border=True):
        st.markdown(
            """
            <div class="ms-section-kicker">Transcript Review</div>
            <h3 class="ms-section-heading">Review Transcript</h3>
            <p class="ms-section-subcopy">Make any final edits in the transcript editor below.</p>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="ms-transcript-toolbar">
              <span class="ms-toolbar-title">Transcript workspace</span>
              <span class="ms-toolbar-stats">
                <span class="ms-toolbar-chip">{len(transcript_turns)} conversation blocks</span>
                <span class="ms-toolbar-chip">{len(transcript_speakers)} speakers</span>
                <span class="ms-toolbar-chip">{word_count} words</span>
              </span>
            </div>
            {transcript_review_preview_html(transcript_text)}
            <div class="ms-editor-label">
              <strong>Edit full transcript</strong>
              <span>Changes are applied when you generate the report</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_copy_button(transcript_text)
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
    st.session_state.transcript_review_required = True


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
        raw_audio_transcript = result.transcript
        raw_audio_segments = "\n".join(segment.transcript for segment in result.segments)
        original_segment_count = len(result.segments)
        result = coalesce_contiguous_audio_segments(result)
        confirmed_audio_speakers = saved_speaker_mapping()
        result = repair_audio_transcription(
            result,
            confirmed_speaker_mapping=confirmed_audio_speakers or None,
        )
        raw_digest = hashlib.sha256(raw_audio_transcript.encode("utf-8")).hexdigest()[:12]
        cleaned_digest = hashlib.sha256(result.transcript.encode("utf-8")).hexdigest()[:12]
        cleaned_audio_segments = "\n".join(segment.transcript for segment in result.segments)
        raw_segments_digest = hashlib.sha256(raw_audio_segments.encode("utf-8")).hexdigest()[:12]
        cleaned_segments_digest = hashlib.sha256(cleaned_audio_segments.encode("utf-8")).hexdigest()[:12]
        log_stage(
            "Audio transcript cleanup",
            "Compared raw STT with the cleaned audio transcript.",
            raw_chars=len(raw_audio_transcript),
            cleaned_chars=len(result.transcript),
            raw_sha256=raw_digest,
            cleaned_sha256=cleaned_digest,
            text_changed=raw_digest != cleaned_digest,
            raw_segment_chars=len(raw_audio_segments),
            cleaned_segment_chars=len(cleaned_audio_segments),
            raw_segments_sha256=raw_segments_digest,
            cleaned_segments_sha256=cleaned_segments_digest,
            segments_changed=raw_segments_digest != cleaned_segments_digest,
            removed_segments=original_segment_count - len(result.segments),
        )
        log_stage(
            "Diarization parsing",
            "Validated and deterministically repaired audio transcription.",
            segment_count=len(result.segments),
            original_segment_count=original_segment_count,
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
        audio_speaker_review_required = bool(resolution.detected_speakers)
        st.session_state.speaker_review_required = audio_speaker_review_required
        st.session_state.transcript_review_required = not audio_speaker_review_required
        st.session_state.edited_transcript_text = transcript_text
        log_stage(
            "Speaker mapping",
            "Resolved speakers for audio transcript.",
            detected=resolution.detected_speakers,
            automatic=resolution.names_available,
            reused=resolution.reused_labels,
            review_required=st.session_state.speaker_review_required,
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


def open_workflow(source: str) -> None:
    """Open the single workflow shell without changing the processing pipeline."""
    st.session_state.workflow_open = True
    st.session_state.workflow_source = source
    st.session_state.workflow_stage = "upload"
    st.session_state.workflow_file = None
    st.rerun()


def _recording_payload_to_upload(recording: dict[str, Any]) -> io.BytesIO:
    """Convert the browser recording payload into an upload-like file object."""

    data = base64.b64decode(str(recording.get("data_base64", "")))
    audio_file = io.BytesIO(data)
    name = str(recording.get("name") or "meeting-recording.webm")
    audio_file.name = name  # type: ignore[attr-defined]
    audio_file.size = len(data)  # type: ignore[attr-defined]
    audio_file.type = str(recording.get("mime_type") or "audio/webm")  # type: ignore[attr-defined]
    audio_file.seek(0)
    return audio_file


def clear_recording_state(*, rerun: bool = True) -> None:
    st.session_state.recording_meeting_state = None
    st.session_state.recording_meeting_version += 1
    if rerun:
        st.rerun()


def start_recorded_meeting_workflow(recording: dict[str, Any]) -> None:
    """Send a browser recording into the existing audio processing pipeline."""

    audio_file = _recording_payload_to_upload(recording)
    st.session_state.workflow_open = True
    st.session_state.workflow_source = "audio"
    st.session_state.workflow_stage = "processing"
    st.session_state.workflow_pending_action = "prepare"
    st.session_state.workflow_file = audio_file
    clear_recording_state(rerun=False)
    st.rerun()


def workflow_stage() -> str:
    current_stage = st.session_state.get("workflow_stage", "upload")
    pending_action = st.session_state.get("workflow_pending_action")
    if pending_action == "prepare" and st.session_state.get("workflow_source") == "audio":
        return "processing"
    if current_stage in {"minutes", "export", "email"} and st.session_state.get("analysis_result") is not None:
        return current_stage
    if current_stage == "processing" and pending_action == "analyze":
        return "processing"
    if current_stage == "processing":
        return "processing"
    if current_stage == "speakers" and st.session_state.get("transcript_result") is not None:
        return "speakers"
    if current_stage == "transcript" and st.session_state.get("transcript_result") is not None:
        return "transcript"
    if current_stage == "upload":
        return "upload"
    if st.session_state.get("workflow_file") is not None:
        return "uploaded"
    return "upload"


def inject_workflow_shell_styles() -> None:
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500;600;700&display=swap');
          :root{--wf-bg:#f1eadb;--wf-paper:#fff;--wf-ink:#171b22;--wf-muted:#737982;
            --wf-line:#e2d8c6;--wf-blue:#7897b7;--wf-blue-soft:#e7eff6;--wf-green:#7d9c7d;
            --wf-lav:#a79bca;--wf-rust:#c9826e;--wf-radius:22px;--wf-card-radius:16px;
            --wf-shadow:0 24px 55px rgba(50,45,35,.12)}
          [data-testid="stAppViewContainer"], [data-testid="stApp"]{background:var(--wf-bg)!important}
          [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], footer{display:none!important}
          .block-container{max-width:1140px!important;padding:1.35rem 1rem 2rem!important}
          .st-key-workflow_shell{background:var(--wf-paper);border:1px solid var(--wf-line);border-radius:var(--wf-radius);
            box-shadow:var(--wf-shadow);overflow:hidden;min-height:0;font-family:Inter,sans-serif;scroll-margin-top:18px;
            opacity:1!important;filter:none!important;backdrop-filter:none!important;position:relative!important}
          .st-key-workflow_shell::before,.st-key-workflow_shell::after{display:none!important;content:none!important}
          .st-key-workflow_shell [data-testid="stStatusWidget"],.st-key-workflow_shell [data-testid="stProgress"],
          .st-key-workflow_shell .ms-proc-wrap{display:none!important;opacity:1!important;filter:none!important}
          .st-key-workflow_shell [data-testid="stAppViewBlockContainer"]{opacity:1!important;filter:none!important}
          body:has(.st-key-workflow_shell) [data-testid="stAppViewBlockContainer"],
          body:has(.st-key-workflow_shell) [data-testid="stAppViewContainer"] > .main{opacity:1!important;filter:none!important;backdrop-filter:none!important}
          body:has(.st-key-workflow_shell) [data-testid="stStatusWidget"]{display:none!important}
          .st-key-workflow_shell iframe{opacity:1!important;filter:none!important;backdrop-filter:none!important;background:transparent!important}
          .st-key-workflow_shell>div[data-testid="stVerticalBlock"]{gap:0!important}
          .st-key-workflow_stage_content{padding:22px 40px 20px!important}
          .st-key-workflow_stage_content:has([class*="st-key-meeting_recorder_"]){min-height:560px!important}
          .st-key-recording_stage_frame{min-height:560px!important;display:flex!important;align-items:flex-start!important}
          .st-key-workflow_stage_content>div[data-testid="stVerticalBlock"]{gap:.62rem!important}
          .st-key-workflow_stage_content [data-testid="stForm"]{border:0!important;border-radius:0!important;
            background:transparent!important;box-shadow:none!important;padding:0!important}
          .ms-wf-bar{height:64px;border-bottom:1px solid var(--wf-line);display:grid;
            grid-template-columns:180px 1fr;align-items:center;padding:0 32px}
          .ms-wf-brand{display:flex;align-items:center;gap:8px;font-family:Fraunces,serif;font-weight:600;color:var(--wf-ink)}
          .ms-wf-brand-logo{display:block;width:38px;height:38px;object-fit:contain;background:transparent;border:0;flex:0 0 auto}
          .ms-wf-steps{display:flex;align-items:center;justify-content:flex-end;gap:10px;color:var(--wf-muted);font-size:13px}
          .ms-wf-step{display:flex;align-items:center;gap:7px;white-space:nowrap}.ms-wf-dot{width:17px;height:17px;border-radius:50%;
            background:#e9e0ce;display:grid;place-items:center;font:10px monospace}.ms-wf-step.now{background:var(--wf-blue-soft);
            color:#274b6d;border-radius:999px;padding:7px 12px}.ms-wf-step.now .ms-wf-dot{background:var(--wf-blue);color:white}
          .ms-wf-step.done{color:var(--wf-ink)}.ms-wf-step.done .ms-wf-dot{background:var(--wf-green);color:white}
          .ms-wf-sep{width:14px;height:1px;background:#d5c9b5}
          .ms-wf-body{padding:0}.ms-wf-body h2{font:500 28px Fraunces,serif!important;margin:0 0 6px!important}
          .ms-wf-sub{margin:0 0 20px;color:var(--wf-muted)!important}.ms-drop-copy{text-align:center;padding:18px 10px 10px}
          .ms-drop-copy h3{font:500 22px Fraunces,serif!important;margin:14px 0 6px!important}.ms-drop-copy p{font-size:13px}
          .ms-upload-stage{width:100%!important;text-align:center!important}.ms-upload-stage>h2,.ms-upload-stage>.ms-wf-sub{width:100%!important;text-align:center!important}
          .ms-upload-stage>.ms-wf-sub{max-width:620px;margin:0 auto 12px!important}
          .st-key-workflow_stage_content:has(.ms-upload-stage) .ms-wf-body h2,
          .st-key-workflow_stage_content:has(.ms-upload-stage) .ms-wf-body>.ms-wf-sub{text-align:center!important;margin-left:auto!important;margin-right:auto!important}
          .ms-upload-stage .ms-drop-copy{padding-top:8px!important}
          .ms-wf-card{border:1px solid var(--wf-line);border-radius:var(--wf-card-radius);padding:22px 28px;background:#fffdfa}
          .ms-wf-success{max-width:640px;margin:18px auto 10px;text-align:center}.ms-wf-success-head{display:flex;flex-direction:column;gap:10px;align-items:center;justify-content:center}
          .ms-wf-success-head h3,.ms-wf-success-head p{margin-left:auto!important;margin-right:auto!important;text-align:center!important}
          .ms-wf-check{width:48px;height:48px;border-radius:13px;background:#eaf2e7;color:#426a49;display:grid;place-items:center;font-size:23px}
          .ms-wf-wave{display:flex;align-items:center;justify-content:center;gap:3px;height:48px;margin:12px 0;border-bottom:1px solid var(--wf-line)}
          .ms-wf-wave i{display:block;width:3px;border-radius:4px;background:var(--wf-blue)}
          .ms-wf-meta{display:flex;justify-content:center;gap:26px;font-size:13px;color:var(--wf-muted);padding-bottom:16px;border-bottom:1px solid var(--wf-line);text-align:center}
          .ms-meeting-meta{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1px;background:var(--wf-line);
            border:1px solid var(--wf-line);border-radius:var(--wf-card-radius);overflow:hidden;margin:0 0 28px}
          .ms-meeting-meta-item{background:#fbfaf6;padding:16px 18px;min-height:76px}.ms-meeting-meta-label{display:block;
            margin-bottom:7px;color:#85827b;font:500 10px 'IBM Plex Mono',monospace;text-transform:uppercase;letter-spacing:.08em}
          .ms-meeting-meta-value{display:block;color:var(--wf-ink);font:500 14px Inter,sans-serif;line-height:1.4;overflow-wrap:anywhere}
          .ms-speaker-shell{border:0;padding:10px 2px 6px;background:transparent;margin:0;box-shadow:none}
          .ms-speaker-shell b{font:600 14px Inter,sans-serif;color:var(--wf-ink)}
          .ms-speaker-shell small{font:400 11px 'IBM Plex Mono',monospace;color:var(--wf-muted)}
          .st-key-workflow_meeting_information{padding:0;margin:0 0 10px;background:transparent;border:0;box-shadow:none}
          .st-key-workflow_meeting_information>div[data-testid="stVerticalBlock"]{gap:.45rem!important}
          .st-key-workflow_meeting_information label p{color:#77736c!important;font:500 10px 'IBM Plex Mono',monospace!important;
            letter-spacing:.06em;text-transform:uppercase}
          .st-key-workflow_shell div[data-baseweb="input"],
          .st-key-workflow_shell div[data-baseweb="textarea"]{border:1px solid #ded6c8!important;border-radius:11px!important;
            background:#fffdfa!important;box-shadow:none!important;overflow:hidden!important}
          .st-key-workflow_shell div[data-baseweb="input"]:focus-within,
          .st-key-workflow_shell div[data-baseweb="textarea"]:focus-within{border-color:var(--wf-blue)!important;
            box-shadow:0 0 0 3px rgba(120,151,183,.13)!important}
          .st-key-workflow_shell input,.st-key-workflow_shell textarea{background:#fffdfa!important;color:var(--wf-ink)!important;
            caret-color:var(--wf-ink)!important;border:0!important;box-shadow:none!important;font:500 14px/1.5 Inter,sans-serif!important;padding:9px 12px!important}
          .st-key-workflow_shell input::selection,.st-key-workflow_shell textarea::selection{background:rgba(120,151,183,.28)!important;color:var(--wf-ink)!important}
          .st-key-workflow_speaker_actions{display:flex!important;width:100%!important;margin-top:18px!important;padding-top:14px!important;border-top:1px solid var(--wf-line)!important}
          .st-key-workflow_speaker_actions>div[data-testid="stVerticalBlock"]{display:flex!important;align-items:flex-end!important;width:100%!important}
          .st-key-workflow_speaker_actions [data-testid="stFormSubmitButton"]{display:flex!important;justify-content:flex-end!important;width:fit-content!important;margin-left:auto!important}
          .st-key-workflow_speaker_actions [data-testid="stFormSubmitButton"]>div{display:flex!important;justify-content:flex-end!important;width:auto!important;margin-left:auto!important}
          .st-key-workflow_speaker_actions [data-testid="stFormSubmitButton"] button{width:auto!important;min-width:0!important;padding:0 22px!important;
            background:#30343a!important;border:1px solid #30343a!important;color:#fff!important;box-shadow:none!important}
          .st-key-workflow_speaker_actions [data-testid="stFormSubmitButton"] button:hover{background:#24282d!important;border-color:#24282d!important;color:#fff!important}
          .ms-transcript-preview{display:grid;gap:3px;max-height:410px;overflow-y:auto;padding:6px 8px 6px 0;
            border-top:1px solid var(--wf-line);border-bottom:1px solid var(--wf-line);scrollbar-width:thin;scrollbar-color:#b9ad9a transparent}
          .ms-transcript-row{display:grid;grid-template-columns:92px 132px minmax(0,1fr);align-items:start;
            gap:18px;padding:15px 16px;border-radius:12px;color:var(--wf-ink);font:400 15px/1.62 Inter,sans-serif}
          .ms-transcript-row:nth-child(3n){background:#f2eef8}.ms-transcript-preview::-webkit-scrollbar,
          .st-key-workflow_transcript_editor textarea::-webkit-scrollbar{width:8px}.ms-transcript-preview::-webkit-scrollbar-thumb,
          .st-key-workflow_transcript_editor textarea::-webkit-scrollbar-thumb{background:#b9ad9a;border-radius:999px;border:2px solid transparent;background-clip:padding-box}
          .ms-transcript-time{padding-top:3px;font:400 11px 'IBM Plex Mono',monospace;color:#61707e}.ms-transcript-speaker{font-weight:600;color:#536c5b}
          .st-key-workflow_transcript_editor label p{font:600 12px 'IBM Plex Mono',monospace!important;color:#6e6a63!important;letter-spacing:.03em}
          .st-key-workflow_transcript_editor{margin-top:24px!important;padding-top:22px!important;border-top:1px solid var(--wf-line)!important}
          .st-key-workflow_transcript_editor div[data-baseweb="textarea"]{border:1px solid #ded6c8!important;border-radius:var(--wf-card-radius)!important;
            background:#fbfaf6!important;box-shadow:0 8px 22px rgba(50,45,35,.045)!important;overflow:hidden!important}
          .st-key-workflow_shell .st-key-workflow_transcript_editor div[data-baseweb="textarea"]>div[data-baseweb="base-input"]{
            background:#fbfaf6!important;border:0!important;border-radius:var(--wf-card-radius)!important;box-shadow:none!important}
          .st-key-workflow_transcript_editor div[data-baseweb="textarea"]:focus-within{border-color:var(--wf-blue)!important;
            box-shadow:0 0 0 3px rgba(120,151,183,.14)!important}
          .st-key-workflow_transcript_editor textarea{min-height:300px!important;padding:20px 22px!important;background:#fbfaf6!important;
            color:var(--wf-ink)!important;font:400 15px/1.72 Inter,sans-serif!important;border:0!important;outline:0!important;box-shadow:none!important;
            scrollbar-width:thin;scrollbar-color:#b9ad9a transparent}
          .st-key-workflow_shell .st-key-workflow_stage_content .st-key-workflow_transcript_editor div[data-baseweb="textarea"] textarea{
            background-color:#fbfaf6!important;background-image:none!important;border:0!important;border-radius:var(--wf-card-radius)!important;
            outline:0!important;box-shadow:none!important}
          .ms-process-center{max-width:560px;margin:8px auto 4px;text-align:center}.ms-process-list{text-align:left;margin:24px auto 0;max-width:420px}
          .ms-process-item{display:grid;grid-template-columns:28px 1fr;gap:13px;padding:9px 0 22px}.ms-process-bullet{width:26px;height:26px;
            border-radius:50%;background:#eaf2e7;color:#426a49;display:grid;place-items:center}.ms-process-item.active .ms-process-bullet{background:var(--wf-blue);color:white}
          .ms-audio-preparing{max-width:470px;margin:22px auto 10px}.ms-audio-preparing .ms-process-list{margin-top:28px;max-width:360px}
          .ms-audio-preparing .ms-process-item{padding:7px 0 15px}.ms-loading-wave{height:42px;display:flex;align-items:center;justify-content:center;gap:5px;margin:2px auto 18px}
          .ms-loading-wave i{display:block;width:4px;height:16px;border-radius:999px;background:var(--wf-blue);animation:ms-wave-pulse 1s ease-in-out infinite}
          .ms-loading-wave i:nth-child(2),.ms-loading-wave i:nth-child(4){animation-delay:.12s}.ms-loading-wave i:nth-child(3){animation-delay:.24s}
          .ms-loading-wave i:nth-child(5){animation-delay:.36s}@keyframes ms-wave-pulse{0%,100%{height:14px;opacity:.55}50%{height:36px;opacity:1}}
          .ms-processing-spinner{width:38px;height:38px;margin:2px auto 16px;border:3px solid #dfe7ee;border-top-color:var(--wf-blue);
            border-radius:50%;animation:ms-spinner-turn .85s linear infinite}@keyframes ms-spinner-turn{to{transform:rotate(360deg)}}
          .ms-processing-brand-logo{display:block;width:58px;height:58px;object-fit:contain;margin:0 auto 10px;background:transparent;border:0}
          .ms-estimated-progress{margin-top:8px;color:var(--wf-muted);font:400 11px 'IBM Plex Mono',monospace}
          .st-key-minutes_document{max-width:860px!important;margin:0 auto!important}.st-key-minutes_document>div[data-testid="stVerticalBlock"]{gap:.8rem!important}
          .ms-minutes-document{max-width:860px;margin:0 auto;padding:4px 10px 8px}
          .ms-minutes-head{display:flex;justify-content:space-between;align-items:flex-start;gap:24px;margin-bottom:22px}
          .ms-minutes-title{font:440 30px/1.15 Fraunces,serif;color:var(--wf-ink);letter-spacing:-.015em}
          .ms-minutes-sub{margin-top:6px;color:var(--wf-muted);font:400 12px/1.5 Inter,sans-serif}
          .ms-minutes-actions{display:flex;align-items:center;justify-content:flex-end;gap:8px;padding-top:2px}
          .ms-minutes-export-link{display:inline-flex;align-items:center;height:34px;padding:0 16px;border-radius:999px;background:#30343a;
            color:#fff!important;font:600 12px Inter,sans-serif;text-decoration:none!important}
          .st-key-minutes_regenerate button{height:34px!important;min-height:34px!important;width:auto!important;padding:0 15px!important;
            border:1px solid var(--wf-line)!important;border-radius:999px!important;background:#fff!important;color:var(--wf-ink)!important;font-size:12px!important}
          .ms-minutes-info{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:0 0 30px}
          .ms-minutes-info-card{min-height:86px;padding:15px 16px;border-radius:14px;background:#efe8d9}
          .ms-minutes-info-icon{display:block;margin-bottom:10px;color:#3d5670;font-size:17px}
          .ms-minutes-info-label{margin-bottom:4px;color:var(--wf-muted);font:400 9px 'IBM Plex Mono',monospace;text-transform:uppercase;letter-spacing:.07em}
          .ms-minutes-info-value{color:var(--wf-ink);font:500 13px/1.4 Inter,sans-serif;overflow-wrap:anywhere}
          .ms-minutes-section{margin:0 0 30px}.ms-minutes-section-head{display:flex;align-items:center;gap:9px;margin-bottom:12px}
          .ms-minutes-section-icon{width:26px;height:26px;border-radius:8px;display:grid;place-items:center;font-size:12px;flex:0 0 auto}
          .ms-minutes-section-head h3{margin:0!important;font:440 17px Fraunces,serif!important;color:var(--wf-ink)!important}
          .ms-minutes-summary{padding:20px 22px;border:1px solid var(--wf-line);border-radius:14px;background:#fbfaf7}
          .ms-minutes-summary p{margin:0;color:var(--wf-ink)!important;font:400 14px/1.65 Inter,sans-serif}
          .ms-minutes-summary p+p{margin-top:10px}
          .ms-minutes-discussion{padding:17px 19px;border:1px solid var(--wf-line);border-radius:14px;background:#fbfaf7;margin-bottom:9px}
          .ms-minutes-discussion-title{display:flex;align-items:center;gap:9px;margin-bottom:8px;color:var(--wf-ink);font:600 13px Inter,sans-serif}
          .ms-minutes-discussion-mark{width:25px;height:25px;border-radius:8px;background:#eeeaf4;color:#5c5077;display:grid;place-items:center}
          .ms-minutes-discussion p{margin:0;color:var(--wf-muted)!important;font:400 12px/1.6 Inter,sans-serif}
          .ms-minutes-chips{display:flex;gap:5px;flex-wrap:wrap;margin-top:9px}.ms-minutes-chip{padding:3px 8px;border-radius:999px;
            background:#efe8d9;color:var(--wf-muted);font:500 9px Inter,sans-serif}
          .ms-minutes-decision{display:flex;gap:11px;padding:13px 16px;border-radius:13px;background:#e8efe5;margin-bottom:8px}
          .ms-minutes-decision-check{width:20px;height:20px;border-radius:50%;background:#7c9b79;color:#fff;display:grid;place-items:center;flex:0 0 auto;font-size:11px}
          .ms-minutes-decision-text{color:#40573e;font:500 13px/1.45 Inter,sans-serif}.ms-minutes-meta{margin-top:3px;color:#60735d;
            font:400 9px 'IBM Plex Mono',monospace}
          .ms-minutes-table-wrap{border:1px solid var(--wf-line);border-radius:14px;overflow-x:auto;background:#fff}
          .ms-minutes-table{width:100%;border-collapse:collapse;min-width:620px}.ms-minutes-table th{padding:10px 14px;background:#efe8d9;
            color:var(--wf-muted);font:500 9px 'IBM Plex Mono',monospace;text-transform:uppercase;letter-spacing:.05em;text-align:left}
          .ms-minutes-table td{padding:11px 14px;border-top:1px solid var(--wf-line);color:var(--wf-ink);font:400 12px/1.4 Inter,sans-serif}
          .ms-minutes-owner{display:flex;align-items:center;gap:7px}.ms-minutes-owner-avatar{width:20px;height:20px;border-radius:50%;background:#eeeaf4;
            color:#5c5077;display:grid;place-items:center;font:600 8px Inter,sans-serif}.ms-minutes-status{display:inline-flex;padding:3px 8px;border-radius:999px;
            background:#e5ecf2;color:#3d5670;font:500 9px Inter,sans-serif}
          .ms-export-section{max-width:760px;margin:34px auto 12px;text-align:center}.ms-export-hdr-title{font:440 24px Fraunces,serif;color:var(--wf-ink)}
          .ms-export-wrap{border:0!important;background:transparent!important;box-shadow:none!important;padding:0!important;margin:0!important}
          .ms-export-hdr{display:block!important}.ms-export-sub{margin-top:5px!important;color:var(--wf-muted)!important;font:400 13px Inter,sans-serif!important}.ms-export-hdr-icon{display:none!important}
          div[data-testid="stHorizontalBlock"]:has(.ms-export-option){max-width:760px;margin:22px auto 0;gap:16px!important}
          div[data-testid="stHorizontalBlock"]:has(.ms-export-option)>div[data-testid="stColumn"]{border:1px solid var(--wf-line);border-radius:18px;
            padding:22px 18px 18px!important;background:#fbfaf7!important;box-shadow:none!important;min-height:220px!important;overflow:visible!important;
            position:relative!important;transition:transform .18s ease,box-shadow .18s ease}
          div[data-testid="stHorizontalBlock"]:has(.ms-export-option)>div[data-testid="stColumn"]:hover{transform:translateY(-3px);box-shadow:0 6px 20px rgba(51,54,59,.08)}
          .ms-export-option{text-align:center!important;padding:0!important;min-height:136px!important;border:0!important;background:transparent!important;box-shadow:none!important}.ms-export-option-icon{width:40px;height:40px;border-radius:11px;margin:0 auto 14px;display:grid;place-items:center;
            font:600 10px Inter,sans-serif;border:0!important}.ms-export-option-icon.pdf{background:#f5e8e3!important;color:#7a4331!important}.ms-export-option-icon.docx{background:#e5ecf2!important;color:#3d5670!important}
          .ms-export-option-icon.email{background:#eeeaf4!important;color:#5c5077!important}.ms-export-option-title{font:440 16px Fraunces,serif!important;color:var(--wf-ink)!important}
          .ms-export-option-desc{min-height:38px;margin:6px 0 14px;color:var(--wf-muted);font:400 11px/1.5 Inter,sans-serif}
          div[data-testid="stHorizontalBlock"]:has(.ms-export-option) .stDownloadButton,
          div[data-testid="stHorizontalBlock"]:has(.ms-export-option) .stButton{position:static!important;inset:auto!important;margin:0!important;padding:0!important}
          div[data-testid="stHorizontalBlock"]:has(.ms-export-option) button{position:static!important;opacity:1!important;width:100%!important;height:42px!important;
            min-height:42px!important;padding:0 18px!important;background:#30343a!important;color:#fff!important;border:1px solid #30343a!important;border-radius:999px!important;
            font:600 12px Inter,sans-serif!important}
          div[data-testid="stHorizontalBlock"]:has(.ms-export-option) button p{color:#fff!important;opacity:1!important}
          div[data-testid="stVerticalBlockBorderWrapper"]:has(.ms-email-compose){width:min(100%,520px)!important;margin:30px auto!important;
            background:#fff!important;border:1px solid var(--wf-line)!important;border-radius:22px!important;padding:0!important;
            box-shadow:0 14px 36px rgba(51,54,59,.09)!important}
          div[data-testid="stVerticalBlockBorderWrapper"]:has(.ms-email-compose)>div{width:100%!important;overflow:visible;
            background:transparent!important;border-radius:22px!important;box-shadow:none!important;padding:0!important}
          div[data-testid="stVerticalBlockBorderWrapper"]:has(.ms-email-compose) [data-testid="stForm"]{padding:0 28px 30px!important;border:0!important}
          div[data-testid="stVerticalBlockBorderWrapper"]:has(.ms-email-compose) [data-testid="stForm"] div[data-testid="stHorizontalBlock"]{margin-top:30px!important;padding-top:4px!important;gap:14px!important;align-items:center!important}
          div[data-testid="stVerticalBlockBorderWrapper"]:has(.ms-email-compose) [data-testid="stForm"] div[data-testid="stHorizontalBlock"]>div:first-child button{
            width:auto!important;background:#fff!important;color:var(--wf-ink)!important;border:1px solid var(--wf-line)!important}
          div[data-testid="stVerticalBlockBorderWrapper"]:has(.ms-email-compose) [data-testid="stForm"] div[data-testid="stHorizontalBlock"]>div:last-child button{
            background:#30343a!important;color:#fff!important;border:1px solid #30343a!important}
          .ms-context-brand-logo{display:block;width:48px;height:48px;object-fit:contain;margin:0 auto 9px;background:transparent;border:0}
          .ms-email-compose{padding:24px 28px 2px}.ms-email-compose .ms-context-brand-logo{margin-left:0}.ms-email-compose-hdr{font:440 18px Fraunces,serif;color:var(--wf-ink);padding-bottom:14px;border-bottom:1px solid var(--wf-line)}
          .ms-email-compose-sub{margin:14px 0 18px!important;font-size:12px!important}.ms-attachment-preview{display:flex;align-items:center;gap:10px;padding:11px 12px;
            border:1px solid var(--wf-line);border-radius:9px;background:#fbfaf7}.ms-attachment-preview-icon{width:28px;height:28px;border-radius:7px;background:#f5e8e3;
            color:#7a4331;display:grid;place-items:center;font:600 9px Inter,sans-serif}.ms-attachment-preview-name{font:500 12px Inter,sans-serif}.ms-attachment-preview-meta{font-size:10px;color:var(--wf-muted)}
          .ms-attachment-pill{display:none}.ms-email-field-label{margin:14px 0 6px;color:var(--wf-muted);font:500 11px Inter,sans-serif}
          @media(max-width:800px){.ms-minutes-head{flex-direction:column}.ms-minutes-info{grid-template-columns:repeat(2,minmax(0,1fr))}
            div[data-testid="stHorizontalBlock"]:has(.ms-export-option){display:grid!important;grid-template-columns:1fr!important}.ms-minutes-document{padding-inline:0}}
          .st-key-workflow_shell .ms-proc-wrap{display:none!important}
          .st-key-workflow_shell [data-testid="stProgress"],.st-key-workflow_shell [data-testid="stStatusWidget"]{display:none!important}
          .st-key-transcript_prepare_output [data-testid="stProgress"],
          .st-key-transcript_prepare_output .ms-proc-wrap{display:none!important}
          .st-key-workflow_shell div[data-testid="stFileUploader"]{border:1px dashed var(--wf-blue)!important;border-radius:var(--wf-card-radius)!important;background:var(--wf-blue-soft)!important;padding:24px!important}
          .st-key-workflow_shell div[data-testid="stFileUploader"] section{display:flex!important;flex-direction:column!important;align-items:center!important;justify-content:center!important;
            gap:6px!important;background:transparent!important;border:0!important;min-height:132px!important;text-align:center!important}
          .st-key-workflow_shell div[data-testid="stFileUploader"] section>div{align-items:center!important;justify-content:center!important;text-align:center!important}
          .st-key-workflow_shell div[data-testid="stFileUploader"] section button{margin:4px auto 0!important}
          .st-key-workflow_shell div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"]{margin-bottom:2px!important}
          .st-key-workflow_shell div[data-testid="stFileUploader"] small{margin-top:0!important}
          .st-key-workflow_shell [data-testid="stCaptionContainer"]{text-align:center!important;margin-top:8px!important}
          .st-key-workflow_shell .stButton button,.st-key-workflow_shell [data-testid="stFormSubmitButton"] button{border-radius:999px!important;
            min-height:40px!important;padding:0 18px!important;font:600 13px Inter,sans-serif!important;box-shadow:none!important}
          .st-key-workflow_shell .stButton button[kind="primary"],.st-key-workflow_shell [data-testid="stFormSubmitButton"] button[kind="primary"]{
            background:#30343a!important;color:#fff!important;border-color:#30343a!important}
          .st-key-workflow_shell .stButton button:not([kind="primary"]),.st-key-workflow_shell [data-testid="stFormSubmitButton"] button:not([kind="primary"]){
            background:#fff!important;color:var(--wf-ink)!important;border:1px solid #cfc3b0!important}
          .st-key-workflow_shell .stButton button:not([kind="primary"]):hover,.st-key-workflow_shell [data-testid="stFormSubmitButton"] button:not([kind="primary"]):hover{
            background:#f8f5ee!important;border-color:#9e927f!important;color:var(--wf-ink)!important}
          .st-key-workflow_shell div[data-testid="stHorizontalBlock"]:has(.ms-export-option) .stButton button,
          .st-key-workflow_shell div[data-testid="stHorizontalBlock"]:has(.ms-export-option) .stDownloadButton button{
            background:#30343a!important;color:#fff!important;border-color:#30343a!important}
          .st-key-workflow_shell div[data-testid="stHorizontalBlock"]:has(.ms-export-option) .stButton button:hover,
          .st-key-workflow_shell div[data-testid="stHorizontalBlock"]:has(.ms-export-option) .stDownloadButton button:hover{
            background:#24282d!important;color:#fff!important;border-color:#24282d!important}
          .st-key-workflow_shell div[data-testid="stVerticalBlockBorderWrapper"]:has(.ms-email-compose) [data-testid="stForm"] div[data-testid="stHorizontalBlock"]>div:first-child button{
            width:auto!important;background:#fff!important;color:var(--wf-ink)!important;border:1px solid #bcae98!important}
          .st-key-workflow_shell div[data-testid="stVerticalBlockBorderWrapper"]:has(.ms-email-compose) [data-testid="stForm"] div[data-testid="stHorizontalBlock"]>div:first-child button:hover{
            background:#f8f5ee!important;color:var(--wf-ink)!important;border-color:#938571!important}
          .st-key-workflow_shell div[data-testid="stVerticalBlockBorderWrapper"]:has(.ms-email-compose) [data-testid="stForm"] div[data-testid="stHorizontalBlock"]>div:last-child button{
            width:auto!important;min-width:148px!important;background:#30343a!important;color:#fff!important;border:1px solid #30343a!important;float:right!important}
          .st-key-workflow_shell div[data-testid="stVerticalBlockBorderWrapper"]:has(.ms-email-compose) [data-testid="stForm"] div[data-testid="stHorizontalBlock"]>div:last-child button:hover{
            background:#24282d!important;color:#fff!important;border-color:#24282d!important}
          .st-key-uploaded_actions{max-width:480px;margin:16px auto 0}.st-key-uploaded_actions div[data-testid="stHorizontalBlock"]{gap:12px!important}
          .st-key-uploaded_actions div[data-testid="stColumn"] button{width:100%!important}
          .ms-shell-close button{border:0!important;background:transparent!important}
          @media(max-width:760px){.ms-wf-bar{grid-template-columns:1fr;padding:0 18px}.ms-wf-steps{display:none}
            .st-key-workflow_stage_content{padding:22px 18px 24px!important}.ms-wf-meta{flex-wrap:wrap;gap:10px}
            .ms-meeting-meta{grid-template-columns:1fr}.ms-transcript-row{grid-template-columns:68px 90px minmax(0,1fr);gap:10px;padding:13px 9px}
            .ms-wf-success{padding:20px 16px}.st-key-uploaded_actions{max-width:100%}.ms-process-list{margin-top:18px}
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.ms-email-compose) [data-testid="stForm"]{padding:0 18px 24px!important}
            .ms-email-compose{padding:20px 18px 2px}}
          @media(max-width:520px){.ms-transcript-row{grid-template-columns:1fr;gap:4px}.ms-minutes-info{grid-template-columns:1fr}
            .st-key-uploaded_actions div[data-testid="stHorizontalBlock"]{display:grid!important;grid-template-columns:1fr!important}
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.ms-email-compose) [data-testid="stForm"] div[data-testid="stHorizontalBlock"]{display:grid!important;grid-template-columns:1fr!important}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_record_meeting_card() -> None:
    st.markdown(
        """
        <style>
          .ms-record-shell{max-width:920px;margin:1rem auto 0}
          .ms-record-shell .ms-record-hdr{display:flex;align-items:flex-start;justify-content:space-between;gap:18px;margin:0 0 14px}
          .ms-record-shell .ms-record-title{font:440 20px Fraunces,serif;color:var(--ms-text,#1f2937);margin:0}
          .ms-record-shell .ms-record-desc{margin:.25rem 0 0;color:#5f6674;font:400 13px/1.55 Inter,sans-serif;max-width:620px}
          .ms-record-shell .ms-record-note{margin-top:8px;color:#6c7481;font:500 11px 'IBM Plex Mono',monospace;letter-spacing:.03em}
          .ms-record-shell .ms-record-card{border:1px solid #e4dbc8;border-radius:22px;background:#fffdfa;box-shadow:0 16px 38px rgba(50,45,35,.08);padding:18px 18px 16px}
          .ms-record-shell .ms-record-state{display:flex;align-items:center;gap:10px;margin-bottom:12px}
          .ms-record-shell .ms-record-pulse{width:12px;height:12px;border-radius:50%;background:#c9826e;box-shadow:0 0 0 7px rgba(201,130,110,.14)}
          .ms-record-shell .ms-record-state-text{font:600 12px Inter,sans-serif;color:#39414d}
          .ms-record-shell .ms-record-row{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:14px;align-items:center}
          .ms-record-shell .ms-record-preview{margin-top:14px;padding:14px;border-radius:16px;background:#fbfaf6;border:1px solid #e8dfcf}
          .ms-record-shell .ms-record-meta{display:flex;flex-wrap:wrap;gap:8px 14px;margin-top:10px;color:#66707d;font:400 12px/1.5 Inter,sans-serif}
          .ms-record-shell .ms-record-actions{display:flex;gap:10px;justify-content:flex-end;align-items:center;margin-top:14px}
          .ms-record-shell .ms-record-actions button{border-radius:999px!important;height:40px!important;min-height:40px!important;padding:0 18px!important;font:600 13px Inter,sans-serif!important}
          .ms-record-shell .ms-record-actions .ms-record-clear button{background:#fff!important;border:1px solid #d9cfbd!important;color:#30343a!important}
          .ms-record-shell .ms-record-actions .ms-record-process button{background:#30343a!important;border:1px solid #30343a!important;color:#fff!important}
          .ms-record-shell .ms-record-actions .ms-record-process button:hover{background:#24282d!important;border-color:#24282d!important;color:#fff!important}
          .ms-record-shell .ms-record-empty{display:grid;place-items:center;min-height:156px;border:1px dashed #d7cdbb;border-radius:16px;background:#f8f5ef;color:#6f756f}
          .st-key-close_recording_meeting button{border:1px solid #cfc3b0!important;border-radius:999px!important;background:#fff!important;color:#30343a!important;font:600 12px/1 Inter,sans-serif!important;padding:0 16px!important;min-height:36px!important;display:flex!important;align-items:center!important;justify-content:center!important;text-align:center!important}
          .st-key-close_recording_meeting button p{margin:0!important;line-height:1!important;color:#30343a!important}
          .st-key-close_recording_meeting button:hover{background:#f8f5ee!important;border-color:#9e927f!important}
          @media(max-width:760px){
            .ms-record-shell .ms-record-row{grid-template-columns:1fr}
            .ms-record-shell .ms-record-actions{justify-content:stretch;flex-direction:column}
            .ms-record-shell .ms-record-actions > div{width:100%}
            .ms-record-shell .ms-record-actions button{width:100%!important}
          }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Back to input options", key="close_recording_meeting"):
        clear_recording_state(rerun=False)
        st.session_state.recording_meeting_open = False
        st.rerun()
    with st.container(key="recording_stage_frame"):
        payload = MEETING_RECORDER_COMPONENT(
            state=st.session_state.get("recording_meeting_state"),
            max_minutes=60,
            key=f"meeting_recorder_{st.session_state.recording_meeting_version}",
        )
    if isinstance(payload, dict):
        event = str(payload.get("event") or "")
        if event == "recorded":
            st.session_state.recording_meeting_state = payload.get("recording")
            st.rerun()
        if event == "clear":
            clear_recording_state()
        if event == "process" and st.session_state.get("recording_meeting_state"):
            start_recorded_meeting_workflow(st.session_state.recording_meeting_state)
def render_workflow_header(stage: str) -> None:
    order = ["upload", "speakers", "transcript", "processing", "minutes"]
    active = {"uploaded": 0, "export": 4, "email": 4}.get(stage, order.index(stage) if stage in order else 0)
    labels = ["Upload", "Speakers", "Transcript", "Processing", "Minutes"]
    steps = []
    for index, label in enumerate(labels):
        state = "done" if index < active else "now" if index == active else ""
        dot = "&#10003;" if index < active else str(index + 1)
        steps.append(f'<div class="ms-wf-step {state}"><span class="ms-wf-dot">{dot}</span>{label}</div>')
        if index < len(labels) - 1:
            steps.append('<span class="ms-wf-sep"></span>')
    st.markdown(
        '<div class="ms-wf-bar">'
        f'<div class="ms-wf-brand">{brand_logo_html("ms-wf-brand-logo", alt="")}<span>MeetScribe</span></div>'
        f'<div class="ms-wf-steps">{"".join(steps)}</div>'
        '</div>', unsafe_allow_html=True)


def render_workflow_meeting_metadata(result: TranscriptionResult) -> None:
    """Render the existing meeting-info source as a read-only workflow summary."""
    initialize_meeting_info(result)
    info = meeting_info_for_export()
    time_and_duration = " · ".join(
        value for value in (info.get("Time", ""), info.get("Duration", "")) if value
    )
    fields = (
        ("Meeting Title", info.get("Meeting Title", "")),
        ("Meeting Date", info.get("Date", "")),
        ("Meeting Time / Duration", time_and_duration),
        ("Participants", info.get("Participants", "")),
        ("Project Name", info.get("Project Name", "")),
        ("Prepared By", info.get("Prepared By", "")),
    )
    items = "".join(
        '<div class="ms-meeting-meta-item">'
        f'<span class="ms-meeting-meta-label">{html.escape(label)}</span>'
        f'<span class="ms-meeting-meta-value">{html.escape(str(value or "—"))}</span>'
        '</div>'
        for label, value in fields
    )
    st.markdown(f'<div class="ms-meeting-meta">{items}</div>', unsafe_allow_html=True)


def render_upload_stage() -> None:
    source = st.session_state.workflow_source
    noun = "recording" if source == "audio" else "transcript"
    intro = (
        "Upload a recording for speaker identification and automatic transcription."
        if source == "audio"
        else "Upload a transcript for speaker confirmation, transcript review, and minutes generation."
    )
    drop_label = "Drop your recording here" if source == "audio" else "Drop your transcript here"
    types = SUPPORTED_FILE_TYPES if source == "audio" else SUPPORTED_TRANSCRIPT_TYPES
    version = st.session_state.audio_upload_version if source == "audio" else st.session_state.transcript_upload_version
    st.markdown(f'<div class="ms-wf-body ms-upload-stage"><h2>New meeting</h2><p class="ms-wf-sub">{intro}</p><div class="ms-drop-copy"><h3>{drop_label}</h3><p>or browse from your computer</p></div>', unsafe_allow_html=True)
    selected = st.file_uploader(f"Choose a {noun}", type=list(types), accept_multiple_files=False,
        key=f"workflow_{source}_{version}", label_visibility="collapsed")
    st.caption("Accepts one file at a time · " + " · ".join(f".{item}" for item in types))
    if selected is not None:
        st.session_state.workflow_file = selected
        st.session_state.workflow_stage = "uploaded"
        st.rerun()
    back, _ = st.columns([0.2, 0.8])
    with back:
        if st.button("Back to home", use_container_width=True):
            clear_current_report()
            st.session_state.workflow_file = None
            st.session_state.workflow_pending_action = ""
            st.session_state.workflow_stage = "upload"
            st.session_state.workflow_open = False
            st.session_state.recording_meeting_open = False
            st.session_state.recording_meeting_state = None
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


def render_uploaded_stage() -> None:
    uploaded = st.session_state.workflow_file
    name = html.escape(getattr(uploaded, "name", "Uploaded file"))
    size = (getattr(uploaded, "size", 0) or 0) / (1024 * 1024)
    bars = "".join(f'<i style="height:{h}px"></i>' for h in (14,28,36,20,31,17,34,23,29,18,35,25,32,16,27,21,34,19))
    st.markdown(f'''<div class="ms-wf-body"><div class="ms-wf-card ms-wf-success">
      <div class="ms-wf-success-head"><div class="ms-wf-check">✓</div><div><h3>File uploaded</h3><p>{name} is ready to process.</p></div></div>
      <div class="ms-wf-wave">{bars}</div><div class="ms-wf-meta"><span><b>{size:.2f} MB</b> file size</span><span><b>{Path(name).suffix.lstrip('.').upper()}</b> format</span><span><b>Ready</b> to process</span></div></div>''', unsafe_allow_html=True)
    if st.button("← Upload", use_container_width=False):
        st.session_state.workflow_stage = "upload"
        st.rerun()
    with st.container(key="uploaded_actions"):
        remove_col, continue_col = st.columns(2)
        with remove_col:
            if st.button("Remove", use_container_width=True):
                if st.session_state.workflow_source == "audio": st.session_state.audio_upload_version += 1
                else: st.session_state.transcript_upload_version += 1
                st.session_state.workflow_file = None
                clear_current_report(); st.rerun()
        with continue_col:
            if st.button("Continue to Review →", type="primary", use_container_width=True):
                st.session_state.workflow_pending_action = "prepare"
                st.session_state.workflow_stage = "uploaded"
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


def render_speaker_stage(result: TranscriptionResult) -> None:
    labels = list(current_speaker_mapping()) or detect_speaker_labels(result)
    mapping = current_speaker_mapping() or {label: label for label in labels}
    if st.button("← Upload", use_container_width=False):
        st.session_state.workflow_stage = "upload"
        st.rerun()
    st.markdown(f'<div class="ms-wf-body"><h2>Confirm your speakers</h2><p class="ms-wf-sub">Confirm the {len(labels)} detected speakers before reviewing the transcript.</p>', unsafe_allow_html=True)
    with st.form("workflow_speaker_form"):
        render_meeting_information_panel(result)
        values = {}
        cols = st.columns(2)
        for index, label in enumerate(labels):
            segments = [s for s in result.segments if speaker_label(s) == label]
            first = format_timestamp(segments[0].start_time_seconds if segments else None)
            with cols[index % 2]:
                st.markdown(f'<div class="ms-speaker-shell"><b>{html.escape(label)}</b><br><small>Speaks first at {first}</small></div>', unsafe_allow_html=True)
                values[label] = st.text_input("Speaker name", value=mapping.get(label, label), key=f"wf_speaker_{index}", label_visibility="collapsed")
        with st.container(key="workflow_speaker_actions"):
            submitted = st.form_submit_button(
                "Continue to Transcript →", type="primary", use_container_width=False)
    if submitted:
        updated = update_mapping(labels, values); store_speaker_mapping(updated)
        mapped = apply_speaker_resolution(result, updated); transcript = format_transcript(mapped, {})
        st.session_state.transcript_result = mapped; st.session_state.transcript_text = transcript
        st.session_state.meeting_metadata = source_meeting_metadata(
            mapped, source_file=st.session_state.get("uploaded_filename", ""))
        st.session_state.meeting_info_initialized = False; initialize_meeting_info(mapped)
        st.session_state.edited_transcript_text = transcript; st.session_state.speaker_review_required = False
        st.session_state.transcript_review_required = True; st.session_state.analysis_result = None
        st.session_state.analysis_error = ""; st.session_state.success_metrics = None
        st.session_state.workflow_stage = "transcript"
        reset_export_state(); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


def render_transcript_stage(result: TranscriptionResult) -> None:
    transcript = st.session_state.get("edited_transcript_text") or st.session_state.transcript_text
    if st.button("← Speakers", use_container_width=False):
        st.session_state.workflow_stage = "speakers"
        st.rerun()
    turns = transcript_turns_from_text(transcript)
    rows = []
    for turn in turns:
        rows.append(f'<div class="ms-transcript-row"><span class="ms-transcript-time">{html.escape(turn.get("timestamp", "--:--"))}</span><span class="ms-transcript-speaker">{html.escape(turn.get("speaker", "Speaker"))}</span><span>{html.escape(turn.get("text", ""))}</span></div>')
    st.markdown('<div class="ms-wf-body"><h2>Review transcript</h2><p class="ms-wf-sub">Edit the transcript before generating the meeting minutes.</p><div class="ms-transcript-preview">'+''.join(rows)+'</div>', unsafe_allow_html=True)
    edited = st.text_area("Edit transcript", value=transcript, height=220, key="workflow_transcript_editor")
    if st.button("Continue to Minutes →", type="primary", use_container_width=True):
        if not edited.strip(): st.warning("Transcript cannot be empty.")
        else:
            st.session_state.edited_transcript_text = edited.strip()
            st.session_state.workflow_pending_action = "analyze"
            st.session_state.workflow_stage = "processing"; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


def render_processing_stage() -> None:
    action = st.session_state.get("workflow_pending_action")
    if st.button("← Transcript", use_container_width=False):
        st.session_state.workflow_stage = "transcript"
        st.rerun()
    if action == "prepare":
        st.markdown(f'''<div class="ms-wf-body"><div class="ms-process-center ms-audio-preparing">
          {brand_logo_html("ms-processing-brand-logo", alt="")}
          <div class="ms-processing-spinner" role="status" aria-label="Processing audio"></div>
          <h2>Preparing speaker review</h2><p class="ms-wf-sub">Your recording is being transcribed and organized by speaker.</p>
          <div class="ms-process-list">
            <div class="ms-process-item"><span class="ms-process-bullet">✓</span><div><b>Audio uploaded</b></div></div>
            <div class="ms-process-item active"><span class="ms-process-bullet">●</span><div><b>Transcribing audio</b></div></div>
            <div class="ms-process-item"><span class="ms-process-bullet">3</span><div><b>Detecting speakers</b></div></div>
            <div class="ms-process-item"><span class="ms-process-bullet">4</span><div><b>Preparing speaker review</b></div></div>
          </div>
        </div></div>''', unsafe_allow_html=True)
        st.session_state.workflow_pending_action = ""
        process_upload(st.session_state.workflow_file)
        if st.session_state.get("transcript_result") is not None:
            st.session_state.workflow_stage = "speakers"
            st.rerun()
    elif action == "analyze":
        st.markdown(f'''<div class="ms-wf-body"><div class="ms-process-center">{brand_logo_html("ms-processing-brand-logo", alt="")}<div class="ms-processing-spinner" role="status" aria-label="Generating minutes"></div><h2>Writing your minutes</h2><p class="ms-wf-sub">The reviewed transcript is being analyzed and prepared as structured minutes.</p><div class="ms-process-list">
          <div class="ms-process-item"><span class="ms-process-bullet">✓</span><div><b>Preparing reviewed transcript</b><br><small>Applying confirmed edits and speaker names</small></div></div>
          <div class="ms-process-item"><span class="ms-process-bullet">✓</span><div><b>Extracting meeting context</b><br><small>Reading participants and meeting information</small></div></div>
          <div class="ms-process-item active"><span class="ms-process-bullet">•</span><div><b>Generating structured minutes</b><br><small>Building summary, decisions, and action items</small></div></div>
          <div class="ms-process-item"><span class="ms-process-bullet">4</span><div><b>Organizing the final report</b></div></div>
          <div class="ms-process-item"><span class="ms-process-bullet">5</span><div><b>Preparing export-ready output</b></div></div>
          </div></div></div>''', unsafe_allow_html=True)
        started_at = time.perf_counter()
        result = st.session_state.get("transcript_result")
        edited = str(st.session_state.get("edited_transcript_text") or "").strip()
        if result is None or not edited:
            st.session_state.analysis_error = "The reviewed transcript was not available for meeting analysis."
            st.error(st.session_state.analysis_error)
            return

        # Keep the exact reviewed result available to the existing analysis
        # pipeline.  If the editor text cannot be reparsed into segments, the
        # editing helper preserves the original diarized segments; do not let a
        # transient parsing mismatch discard the recording transcript.
        try:
            edited_result = apply_transcript_edits(result, edited)
        except Exception as exc:
            log_stage(
                "Transcript editing",
                "Could not reparse reviewed transcript; preserving diarized result.",
                error=str(exc),
                traceback=traceback.format_exc(),
            )
            edited_result = result
        if not edited_result.segments and result.segments:
            edited_result = result
        st.session_state.transcript_result = edited_result
        st.session_state.transcript_text = edited
        st.session_state.transcript_review_required = False
        reset_export_state()
        st.session_state.workflow_pending_action = ""
        analysis = run_meeting_analysis(
            edited,
            started_at=started_at,
            estimate_note="Generating your meeting minutes.",
        )
        if analysis is not None:
            store_success_metrics(result=edited_result, analysis=analysis, started_at=started_at)
            st.session_state.workflow_stage = "minutes"
        else:
            # run_meeting_analysis records the real failure in analysis_error.
            # Keep the user on Processing so the failure is visible instead of
            # silently sending the recording back to Transcript Review.
            st.session_state.workflow_stage = "processing"
            return
        st.rerun()
    elif st.session_state.get("analysis_error"):
        st.markdown('''<div class="ms-wf-body"><div class="ms-process-center">
          <h2>Minutes could not be completed</h2>
          <p class="ms-wf-sub">The existing meeting-analysis pipeline reported an error. Your reviewed transcript is still preserved.</p>
        </div></div>''', unsafe_allow_html=True)
        render_analysis_error()
    else:
        st.markdown('''<div class="ms-wf-body"><div class="ms-process-center">
          <h2>Minutes prepared</h2><p class="ms-wf-sub">Processing completed successfully.</p>
          <div class="ms-process-list">
            <div class="ms-process-item"><span class="ms-process-bullet">✓</span><div><b>Transcript reviewed</b></div></div>
            <div class="ms-process-item"><span class="ms-process-bullet">✓</span><div><b>Meeting minutes generated</b></div></div>
          </div></div></div>''', unsafe_allow_html=True)


def render_minutes_stage(result: TranscriptionResult, analysis: MeetingAnalysisResult) -> None:
    report_info = meeting_info_for_export()
    title = str(report_info.get("Meeting Title") or analysis.summary.title or "Meeting Minutes")
    source_file = str(report_info.get("Source File") or st.session_state.get("uploaded_filename", ""))
    date_text = str(report_info.get("Date") or "Not specified")
    duration = str(report_info.get("Duration") or "Not available")
    participants = str(report_info.get("Participants") or report_info.get("Attendees") or "Not specified")
    project_name = str(report_info.get("Project Name") or report_info.get("Prepared By") or "MeetScribe")

    if st.button("← Processing", use_container_width=False):
        st.session_state.workflow_stage = "processing"
        st.rerun()

    with st.container(key="minutes_document"):
        title_col, actions_col = st.columns([0.72, 0.28], vertical_alignment="top")
        with title_col:
            st.markdown(
                f'''<div class="ms-minutes-title">{html.escape(title)}</div>
                <div class="ms-minutes-sub">Generated from {html.escape(source_file or "reviewed transcript")}</div>''',
                unsafe_allow_html=True,
            )
        with actions_col:
            regenerate_col, export_col = st.columns(2, gap="small")
            with regenerate_col:
                with st.container(key="minutes_regenerate"):
                    regenerate_clicked = st.button(
                        "Regenerate", key="continue_to_analysis", use_container_width=False)
            with export_col:
                if st.button("Export ↓", type="primary", use_container_width=False):
                    st.session_state.workflow_stage = "export"
                    st.rerun()

        if regenerate_clicked:
            regenerated = run_meeting_analysis(
                st.session_state.get("transcript_text", ""),
                started_at=time.perf_counter(),
                estimate_note="Regenerating meeting minutes.",
            )
            if regenerated is not None:
                st.rerun()

        info_cards = (
            ("▣", "Date", date_text),
            ("◷", "Duration", duration),
            ("♙", "Attendees", participants),
            ("▤", "Project", project_name),
        )
        info_html = "".join(
            '<div class="ms-minutes-info-card">'
            f'<span class="ms-minutes-info-icon">{icon}</span>'
            f'<div class="ms-minutes-info-label">{html.escape(label)}</div>'
            f'<div class="ms-minutes-info-value">{html.escape(value)}</div></div>'
            for icon, label, value in info_cards
        )

        summary_paragraphs = [analysis.summary.short_summary]
        if analysis.summary.detailed_summary and analysis.summary.detailed_summary.strip() != analysis.summary.short_summary.strip():
            summary_paragraphs.append(analysis.summary.detailed_summary)
        summary_html = "".join(f'<p>{html.escape(text)}</p>' for text in summary_paragraphs if text)

        discussion_cards = []
        for index, item in enumerate(analysis.key_discussion_points, start=1):
            topic = re.split(r"(?<=[.!?])\s+|:\s+", item.point, maxsplit=1)[0].strip()
            if len(topic) > 76:
                topic = topic[:73].rstrip() + "..."
            chips = "".join(f'<span class="ms-minutes-chip">{html.escape(name)}</span>' for name in item.speakers)
            if item.timestamp:
                chips += f'<span class="ms-minutes-chip">{html.escape(item.timestamp)}</span>'
            discussion_cards.append(
                '<article class="ms-minutes-discussion">'
                f'<div class="ms-minutes-discussion-title"><span class="ms-minutes-discussion-mark">{index:02d}</span>{html.escape(topic or f"Discussion {index}")}</div>'
                f'<p>{html.escape(item.point)}</p><div class="ms-minutes-chips">{chips}</div></article>'
            )
        discussions_html = "".join(discussion_cards) or '<div class="ms-minutes-summary"><p>No discussion points were extracted.</p></div>'

        decision_cards = []
        for item in analysis.decisions:
            meta = " · ".join(value for value in (item.owner or "Unassigned", item.timestamp or "") if value)
            decision_cards.append(
                '<article class="ms-minutes-decision"><span class="ms-minutes-decision-check">✓</span><div>'
                f'<div class="ms-minutes-decision-text">{html.escape(item.decision)}</div>'
                f'<div class="ms-minutes-meta">{html.escape(meta)}</div></div></article>'
            )
        decisions_html = "".join(decision_cards) or '<div class="ms-minutes-summary"><p>No decisions were extracted.</p></div>'

        action_rows = []
        for item in analysis.action_items:
            owner = item.owner or "Unassigned"
            initials = "".join(part[0] for part in owner.split()[:2]).upper()[:2] or "—"
            action_rows.append(
                '<tr>'
                f'<td>{html.escape(item.task)}</td>'
                f'<td><span class="ms-minutes-owner"><span class="ms-minutes-owner-avatar">{html.escape(initials)}</span>{html.escape(owner)}</span></td>'
                f'<td class="mono">{html.escape(item.due_date or "Not specified")}</td>'
                f'<td><span class="ms-minutes-status">{html.escape(item.status or "Open")}</span></td></tr>'
            )
        actions_html = (
            '<div class="ms-minutes-table-wrap"><table class="ms-minutes-table"><thead><tr>'
            '<th>Task</th><th>Assignee</th><th>Due</th><th>Status</th></tr></thead><tbody>'
            + ("".join(action_rows) or '<tr><td colspan="4">No action items were extracted.</td></tr>')
            + '</tbody></table></div>'
        )

        st.markdown(
            f'''<div class="ms-minutes-document">
              <div class="ms-minutes-info">{info_html}</div>
              <section class="ms-minutes-section"><div class="ms-minutes-section-head"><span class="ms-minutes-section-icon" style="background:#e5ecf2;color:#3d5670">☷</span><h3>Executive summary</h3></div><div class="ms-minutes-summary">{summary_html}</div></section>
              <section class="ms-minutes-section"><div class="ms-minutes-section-head"><span class="ms-minutes-section-icon" style="background:#eeeaf4;color:#5c5077">▧</span><h3>Discussion</h3></div>{discussions_html}</section>
              <section class="ms-minutes-section"><div class="ms-minutes-section-head"><span class="ms-minutes-section-icon" style="background:#e8efe5;color:#40573e">✓</span><h3>Decisions</h3></div>{decisions_html}</section>
              <section class="ms-minutes-section"><div class="ms-minutes-section-head"><span class="ms-minutes-section-icon" style="background:#f5e8e3;color:#7a4331">▤</span><h3>Action items</h3></div>{actions_html}</section>
            </div>''',
            unsafe_allow_html=True,
        )
        render_analysis_error()


def render_export_stage(analysis: MeetingAnalysisResult) -> None:
    if st.button("← Minutes", use_container_width=False):
        st.session_state.workflow_stage = "minutes"
        st.rerun()
    render_export_card(analysis)


def render_email_stage(analysis: MeetingAnalysisResult) -> None:
    if st.button("← Export", use_container_width=False):
        st.session_state.workflow_stage = "export"
        st.rerun()
    try:
        mom_pdf_path = prepared_export_path("pdf_export_path", export_to_pdf, analysis)
    except Exception as exc:
        log_stage(
            "Export",
            "Could not prepare email attachment.",
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        st.error("The email attachment could not be prepared. Please try again.")
        return
    render_email_form(analysis=analysis, mom_pdf_path=mom_pdf_path)


def render_workflow_shell() -> None:
    inject_workflow_shell_styles()
    stage = workflow_stage()
    with st.container(key="workflow_shell"):
        render_workflow_header(stage)
        with st.container(key="workflow_stage_content"):
            if stage == "upload": render_upload_stage()
            elif stage == "uploaded": render_uploaded_stage()
            elif stage == "speakers": render_speaker_stage(st.session_state.transcript_result)
            elif stage == "transcript": render_transcript_stage(st.session_state.transcript_result)
            elif stage == "processing": render_processing_stage()
            elif stage == "minutes": render_minutes_stage(st.session_state.transcript_result, st.session_state.analysis_result)
            elif stage == "export": render_export_stage(st.session_state.analysis_result)
            elif stage == "email": render_email_stage(st.session_state.analysis_result)
            if (st.session_state.get("workflow_pending_action") == "prepare"
                    and st.session_state.workflow_source == "transcript"):
                st.session_state.workflow_pending_action = ""
                with st.container(key="transcript_prepare_output"):
                    process_transcript_upload(st.session_state.workflow_file)
                if st.session_state.get("transcript_result") is not None:
                    st.session_state.workflow_stage = "speakers"
                    st.rerun()
        components.html(
            """
            <script>
              requestAnimationFrame(() => setTimeout(() => {
                const shell = window.parent.document.querySelector('.st-key-workflow_shell');
                if (shell) shell.scrollIntoView({ behavior: 'smooth', block: 'start' });
              }, 40));
            </script>
            """,
            height=0,
        )


def main() -> None:
    st.set_page_config(page_title="MeetScribe", page_icon=str(BRAND_LOGO_PATH), layout="wide")
    initialize_session_state()
    inject_processing_styles()
    inject_premium_redesign_styles()

    if st.session_state.get("workflow_open", False):
        render_workflow_shell()
        return

    # The public landing page remains intact. Its two CTA buttons open the
    # workflow shell; the legacy upload/review workspace below is intentionally
    # bypassed so none of the previous widgets remain visible.
    render_top_navigation()
    render_hero()
    if st.session_state.get("recording_meeting_open"):
        return
    render_product_landing_sections()
    render_landing_interactivity()
    return

    render_top_navigation()
    render_hero()
    st.markdown("<div id='workspace'></div>", unsafe_allow_html=True)
    render_landing_interactivity()

    # ── UPLOAD PANEL ──────────────────────────────────────────────
    with st.container(border=True):
        st.markdown(
            """
            <div class="ms-upload-title-row">
              <div class="ms-upload-icon-badge">↑</div>
              <span class="ms-upload-title">Bring your meeting into MeetScribe</span>
            </div>
            <p class="ms-upload-desc">
              Drag and drop a recording or transcript, or choose a file to begin the guided review.
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
                    audio_file_type = Path(uploaded_file.name).suffix.lstrip(".").upper() or "AUDIO"
                    upload_signature = f"{uploaded_file.name}:{uploaded_file.size}"
                    if st.session_state.last_logged_upload != upload_signature:
                        log_stage("File upload", "File selected in UI.",
                                  filename=uploaded_file.name, size=uploaded_file.size)
                        st.session_state.last_logged_upload = upload_signature

                    file_col, remove_col = st.columns([1, 0.26], gap="small", vertical_alignment="center")
                    with file_col:
                        st.markdown(
                            f"""<div class="ms-file-card">
                                    <div class="ms-file-icon">
                                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg>
                                    </div>
                                  <div class="ms-file-info">
                                    <div class="ms-file-name">{html.escape(uploaded_file.name)}</div>
                                    <div class="ms-file-meta">
                                      <span>{file_size_mb:.2f} MB</span>
                                      <span>{html.escape(audio_file_type)}</span>
                                      <span>Uploaded this session</span>
                                    </div>
                                  </div>
                                  <span class="ms-upload-success">✓ Ready</span>
                                </div>""",
                            unsafe_allow_html=True,
                        )
                    with remove_col:
                        if st.button("Remove", key="remove_audio_file", type="secondary", help="Remove file"):
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
                    transcript_file_type = Path(transcript_file.name).suffix.lstrip(".").upper() or "DOCUMENT"
                    file_col, remove_col = st.columns([1, 0.26], gap="small", vertical_alignment="center")
                    with file_col:
                        st.markdown(
                            f"""<div class="ms-file-card">
                                    <div class="ms-file-icon">
                                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
                                    </div>
                                  <div class="ms-file-info">
                                    <div class="ms-file-name">{html.escape(transcript_file.name)}</div>
                                    <div class="ms-file-meta">
                                      <span>{tf_size_mb:.2f} MB</span>
                                      <span>{html.escape(transcript_file_type)}</span>
                                      <span>Uploaded this session</span>
                                    </div>
                                  </div>
                                  <span class="ms-upload-success">✓ Ready</span>
                                </div>""",
                            unsafe_allow_html=True,
                        )
                    with remove_col:
                        if st.button("Remove", key="remove_transcript_file", type="secondary", help="Remove file"):
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
        render_product_landing_sections()

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
                <div class="ms-report-legacy-intro">
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
            render_generated_report_header(analysis)

            render_summary_tab(analysis)

            render_key_points_tab(analysis)

            render_decisions_tab(analysis)

            render_action_items_tab(analysis)

            with st.container(border=True):
                st.markdown(
                    """
                    <div id="source-transcript" class="ms-report-section-head">
                      <h3>Source Transcript</h3>
                      <span>Reviewed conversation</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                render_copy_button(transcript_text)
                render_transcript(result, current_speaker_mapping())

            render_export_card(analysis)

    if not has_session_input:
        render_landing_interactivity()


if __name__ == "__main__":
    main()
