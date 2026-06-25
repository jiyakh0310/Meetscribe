"""Streamlit UI for MeetScribe audio transcription — refined UI."""

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
from transcription.speaker_mapping import (
    SpeakerMapping,
    apply_mapping_to_result,
    default_mapping_for_result,
    display_speaker_label,
    has_participant_names,
    named_mapping_for_result,
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
    st.session_state.setdefault("speaker_names_available", False)
    st.session_state.setdefault("speaker_review_required", False)


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
          @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

          :root {
            --bg:        #050505;
            --bg2:       #0A0A0A;
            --bg3:       #111111;
            --card:      #121212;
            --card2:     #171717;
            --card3:     #1C1C1C;
            --border:    rgba(255,255,255,0.07);
            --border2:   rgba(255,255,255,0.12);
            --border-p:  rgba(109,93,246,0.40);
            --ink:       #F0F0F0;
            --ink2:      #A0A0A0;
            --muted:     #5A5A5A;
            --p:         #6D5DF6;
            --p2:        #7C6CFF;
            --p3:        #8A7BFF;
            --pg:        rgba(109,93,246,0.10);
            --pglow:     rgba(109,93,246,0.18);
            --emerald:   #10B981;
            --eg:        rgba(16,185,129,0.10);
            --amber:     #F59E0B;
            --ag:        rgba(245,158,11,0.10);
            --violet:    #8B5CF6;
            --vg:        rgba(139,92,246,0.10);
            --red:       #EF4444;
            --blue:      #3B82F6;
            --pink:      #EC4899;
            --pnkg:      rgba(236,72,153,0.10);
          }

          @keyframes fadein { from{opacity:0;transform:translateY(5px)} to{opacity:1;transform:none} }
          @keyframes pulse  { 0%,100%{opacity:.4} 50%{opacity:1} }
          @keyframes shimmer {
            0%  { background-position: -400% 0 }
            100%{ background-position:  400% 0 }
          }

          *,*::before,*::after { box-sizing:border-box; margin:0; padding:0; }

          html,body,[data-testid="stAppViewContainer"],[data-testid="stApp"] {
            background: var(--bg) !important;
            color: var(--ink);
            font-family: Inter, system-ui, -apple-system, sans-serif;
            -webkit-font-smoothing: antialiased;
            overflow-x: hidden !important;
          }

          [data-testid="stHeader"]     { background: transparent !important; }
          [data-testid="stDecoration"] { display: none !important; }
          [data-testid="stToolbar"]    { display: none !important; }

          .block-container {
            max-width: 860px !important;
            margin: 0 auto !important;
            padding-top: 0 !important;
            padding-bottom: 7rem !important;
            padding-left:  clamp(1rem, 3vw, 2rem) !important;
            padding-right: clamp(1rem, 3vw, 2rem) !important;
          }

          h1,h2,h3,h4,h5,h6 { color: var(--ink); letter-spacing: -0.02em; }
          p,li,label,[data-testid="stMarkdownContainer"] { color: var(--ink2); }

          /* ══ NAVBAR ══════════════════════════════ */
          .ms-navbar {
            display: flex;
            align-items: center;
            justify-content: flex-start;
            padding: 0.85rem 0 0.85rem;
            border-bottom: 1px solid var(--border);
            margin-bottom: 0;
          }

          .ms-navbar-brand {
            display: flex; align-items: center; gap: 0.45rem;
          }

          .ms-waveicon {
            display: flex; align-items: flex-end; gap: 2px; height: 15px;
          }
          .ms-waveicon span {
            display: block; width: 3px; border-radius: 2px;
            background: var(--p2);
          }
          .ms-waveicon span:nth-child(1) { height: 6px; }
          .ms-waveicon span:nth-child(2) { height: 12px; }
          .ms-waveicon span:nth-child(3) { height: 15px; }
          .ms-waveicon span:nth-child(4) { height: 9px; }
          .ms-waveicon span:nth-child(5) { height: 15px; }
          .ms-waveicon span:nth-child(6) { height: 8px; }

          .ms-navbar-name {
            color: var(--ink); font-size: 0.87rem; font-weight: 700;
            letter-spacing: -0.01em;
          }

          /* ══ HERO ════════════════════════════════ */
          .ms-hero {
            text-align: center;
            padding: 1.15rem 1rem 1.25rem;
            animation: fadein 0.4s ease both;
            display: flex;
            flex-direction: column;
            align-items: center;
          }

          .ms-hero h1 {
            font-size: clamp(2.4rem, 7vw, 4rem);
            font-weight: 900;
            letter-spacing: -0.045em;
            line-height: 1.0;
            color: var(--ink);
            margin-bottom: 0.45rem;
            text-align: center;
            width: 100%;
          }

          .ms-hero-sub {
            color: var(--ink2); font-size: 0.98rem; font-weight: 400;
            line-height: 1.5; max-width: 520px;
            margin: 0 auto 1.0rem;
            width: 100%; text-align: center;
          }

          .ms-pill-row {
            display: flex; flex-wrap: wrap;
            justify-content: center; align-items: center;
            gap: 0.42rem; width: 100%;
          }

          .ms-pill {
            display: inline-flex; align-items: center; gap: 0.38rem;
            border: 1px solid var(--border2); border-radius: 999px;
            background: var(--card); color: var(--ink2);
            font-size: 0.77rem; font-weight: 500; padding: 0.35rem 0.85rem;
            transition: border-color 180ms, color 180ms, box-shadow 180ms;
            cursor: default;
          }
          .ms-pill:hover {
            border-color: var(--border-p); color: var(--ink);
            box-shadow: 0 0 12px var(--pglow);
          }
          .ms-pill-icon { font-size: 0.80rem; }

          /* ══ UPLOAD PANEL ════════════════════════ */
          .ms-upload-card {
            border: 1px solid var(--border); border-radius: 16px;
            background: var(--card); padding: 1.4rem 1.4rem 1.4rem;
            margin-bottom: 1.0rem;
            animation: fadein 0.35s ease both;
          }

          .ms-upload-title-row {
            display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.2rem;
          }

          .ms-upload-icon-badge {
            width: 26px; height: 26px; border-radius: 7px;
            background: var(--pg); border: 1px solid var(--border-p);
            display: flex; align-items: center; justify-content: center;
            font-size: 0.78rem; color: var(--p2); flex-shrink: 0;
          }

          .ms-upload-title {
            color: var(--ink); font-size: 0.95rem; font-weight: 700;
            letter-spacing: -0.01em;
          }

          .ms-upload-desc {
            color: var(--muted); font-size: 0.78rem; margin: 0 0 0.85rem;
          }

          /* sub-card headers inside upload */
          .ms-sub-label { color: var(--ink); font-size: 0.84rem; font-weight: 600; margin-bottom: 0.15rem; }
          /* FIX #3: Add breathing room between formats text and upload box */
          .ms-sub-fmt   { color: var(--muted); font-size: 0.73rem; margin-bottom: 1.1rem; }

          div[data-testid="column"] div[data-testid="stVerticalBlockBorderWrapper"] {
            min-height: 238px;
            overflow: hidden !important;
            border-radius: 16px !important;
          }

          div[data-testid="column"] div[data-testid="stVerticalBlockBorderWrapper"] > div {
            height: 100%;
          }

          /* ══ UNIFIED UPLOAD ZONE ══ */
          .ms-upload-outer {
            position: relative;
            width: 100%;
          }

          div[data-testid="stFileUploader"] {
            width: 100% !important;
            margin-top: 0 !important;
          }
          div[data-testid="stFileUploader"] > div {
            width: 100% !important;
          }
          div[data-testid="stFileUploader"] section {
            position: relative;
            min-height: 142px !important;
            width: 100% !important;
            border: 1.5px dashed var(--border-p) !important;
            border-radius: 12px !important;
            background:
              linear-gradient(135deg, rgba(109,93,246,0.055), rgba(255,255,255,0.012)),
              var(--bg2) !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            padding: 1.35rem 1rem !important;
            overflow: hidden !important;
            cursor: pointer !important;
            transition: border-color 180ms ease, background 180ms ease, box-shadow 180ms ease !important;
          }
          div[data-testid="stFileUploader"] section:hover {
            border-color: var(--p2) !important;
            background:
              linear-gradient(135deg, rgba(109,93,246,0.11), rgba(255,255,255,0.018)),
              var(--bg2) !important;
            box-shadow: 0 0 0 1px rgba(109,93,246,0.10), 0 12px 28px rgba(0,0,0,0.16) !important;
          }
          div[data-testid="stFileUploader"] section button {
            position: static !important;
            opacity: 1 !important;
            border: 1px solid rgba(109,93,246,0.28) !important;
            border-radius: 999px !important;
            background: rgba(109,93,246,0.10) !important;
            color: var(--p2) !important;
            box-shadow: none !important;
            font-family: Inter, sans-serif !important;
            font-size: 0.74rem !important;
            font-weight: 700 !important;
            cursor: pointer !important;
            transition: border-color 160ms ease, background 160ms ease, color 160ms ease !important;
          }
          div[data-testid="stFileUploader"] section button:hover {
            border-color: rgba(109,93,246,0.48) !important;
            background: rgba(109,93,246,0.16) !important;
            color: var(--ink) !important;
          }
          div[data-testid="stFileUploader"] section [data-testid="stFileUploaderDropzoneInstructions"] {
            align-items: center !important;
            text-align: center !important;
            gap: 0.28rem !important;
          }
          div[data-testid="stFileUploader"] section [data-testid="stFileUploaderDropzoneInstructions"] ~ div:not([data-testid="stFileUploaderDropzoneInstructions"]) {
            display: none !important;
          }
          div[data-testid="stFileUploader"] section svg {
            color: var(--p2) !important;
            opacity: 0.9 !important;
          }

          /* ── file card + delete button row ── */
          /* FIX #1 & #2: Perfect alignment of file card and delete button */
          div[data-testid="stHorizontalBlock"]:has(button[data-testid="stBaseButton-secondary"]) {
            display: flex !important;
            align-items: center !important;
            flex-wrap: nowrap !important;
            gap: 0.5rem !important;
            margin-top: 0.55rem !important;
            width: 100% !important;
          }
          div[data-testid="stHorizontalBlock"]:has(button[data-testid="stBaseButton-secondary"]) [data-testid="stVerticalBlock"] {
            gap: 0 !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: stretch !important;
          }
          div[data-testid="stHorizontalBlock"]:has(button[data-testid="stBaseButton-secondary"]) .ms-file-card {
            margin-top: 0 !important;
            height: 100% !important;
            transform: translateY(-3px) !important;
          }
          div[data-testid="stHorizontalBlock"]:has(button[data-testid="stBaseButton-secondary"]) > div[data-testid="column"] {
            min-width: 0 !important;
            padding: 0 !important;
            display: flex !important;
            flex-direction: column !important;
          }
          div[data-testid="stHorizontalBlock"]:has(button[data-testid="stBaseButton-secondary"]) > div[data-testid="column"]:has(button[data-testid="stBaseButton-secondary"]) {
            flex: 0 0 50px !important;
            width: 50px !important;
            min-width: 50px !important;
          }
          div[data-testid="stHorizontalBlock"]:has(button[data-testid="stBaseButton-secondary"]) > div[data-testid="column"]:not(:has(button[data-testid="stBaseButton-secondary"])) {
            flex: 1 1 auto !important;
          }
          /* Ensure the stButton wrapper inside the delete column stretches */
          div[data-testid="stHorizontalBlock"]:has(button[data-testid="stBaseButton-secondary"]) > div[data-testid="column"]:has(button[data-testid="stBaseButton-secondary"]) .stButton {
            height: 100% !important;
            display: flex !important;
            flex-direction: column !important;
          }
          div[data-testid="stHorizontalBlock"]:has(button[data-testid="stBaseButton-secondary"]) > div[data-testid="column"]:has(button[data-testid="stBaseButton-secondary"]) .stButton > button {
            height: 100% !important;
            min-height: 48px !important;
            flex: 1 !important;
            transform: translateY(3px) !important;
          }

          .ms-file-card {
            display: flex; align-items: center; gap: 0.65rem;
            border: 1px solid rgba(109,93,246,0.22); border-radius: 10px;
            background: rgba(109,93,246,0.06);
            padding: 0 0.80rem;
            width: 100%; max-width: 100%; box-sizing: border-box;
            height: 44px; min-height: 44px; max-height: 44px; overflow: hidden;
          }
          .ms-file-icon {
            width: 26px; height: 26px; border-radius: 6px; flex-shrink: 0;
            background: var(--pg); border: 1px solid var(--border-p);
            display: flex; align-items: center; justify-content: center;
            color: var(--p2);
          }
          .ms-file-info { flex: 1; min-width: 0; }
          .ms-file-name {
            color: var(--ink); font-size: 0.82rem; font-weight: 600;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
          }
          .ms-file-size { color: var(--muted); font-size: 0.70rem; }

          .ms-or-wrap {
            min-height: 238px;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
          }
          .ms-or-wrap::before {
            content: "";
            position: absolute;
            top: 0.35rem;
            bottom: 0.35rem;
            width: 1px;
            background: linear-gradient(
              180deg,
              transparent, 
              rgba(255,255,255,0.14),
              transparent
            );
          }
          .ms-or-divider {
            display: flex; align-items: center; justify-content: center;
            color: var(--ink2); font-size: 0.68rem; font-weight: 700;
            letter-spacing: 0.04em;
            width: 34px; height: 34px; border-radius: 50%;
            border: 1px solid var(--border2); background: var(--card2);
            margin: 0 auto; position: relative; z-index: 1;
            box-shadow: 0 8px 20px rgba(0,0,0,0.18);
          }

          /* Override Streamlit file uploader appearance globally */
          div[data-testid="stFileUploader"] label,
          div[data-testid="stFileUploader"] small,
          div[data-testid="stFileUploader"] p { color: var(--ink2) !important; }
          div[data-testid="stFileUploader"] small {
            display: none !important;
          }
          [data-testid="stFileUploaderFile"],
          [data-testid="stFileUploaderFileName"],
          [data-testid="stFileUploaderDeleteBtn"],
          [data-testid="stFileUploaderFileData"],
          [data-testid="stFileUploaderFileStatus"],
          [data-testid="stFileUploaderUploadedFile"],
          div[data-testid="stFileUploader"] section [data-testid="stFileUploaderFile"],
          div[data-testid="stFileUploader"] section [data-testid="stFileUploaderFileName"],
          div[data-testid="stFileUploader"] section [data-testid="stFileUploaderDeleteBtn"],
          div[data-testid="stFileUploader"] section [data-testid="stFileUploaderFileData"],
          div[data-testid="stFileUploader"] section [data-testid="stFileUploaderFileStatus"],
          div[data-testid="stFileUploader"] section [data-testid="stFileUploaderUploadedFile"],
          div[data-testid="stFileUploader"] section + div,
          div[data-testid="stFileUploader"] section ~ div,
          div[data-testid="stFileUploader"] ul,
          div[data-testid="stFileUploader"] li {
            display: none !important;
          }

          /* Panels */
          div[data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid var(--border) !important;
            border-radius: 16px !important;
            background: var(--card) !important;
            box-shadow: none !important;
            backdrop-filter: none !important;
            transition: border-color 200ms;
          }
          div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            border-color: var(--border2) !important;
          }

          /* ══ BUTTONS ══════════════════════════════ */
          /* FIX #4: Reduce glow, improve text contrast on Generate button */
          .stButton > button[data-testid="stBaseButton-primary"] {
            border: 1px solid var(--p) !important;
            border-radius: 12px !important;
            background: linear-gradient(135deg, #5B4EE8 0%, var(--p) 40%, var(--p2) 100%) !important;
            color: #FFFFFF !important;
            font-family: Inter, sans-serif !important;
            font-weight: 700 !important; font-size: 0.96rem !important;
            min-height: 3.1rem !important;
            box-shadow: 0 2px 8px rgba(109,93,246,0.20), 0 1px 0 rgba(255,255,255,0.10) inset !important;
            transition: transform 160ms, box-shadow 160ms, opacity 160ms !important;
            letter-spacing: 0.01em !important;
            text-shadow: 0 1px 2px rgba(0,0,0,0.40) !important;
          }
          .stButton > button[data-testid="stBaseButton-primary"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 16px rgba(109,93,246,0.28), 0 1px 0 rgba(255,255,255,0.10) inset !important;
          }
          .stButton > button[data-testid="stBaseButton-primary"]:active { transform: none !important; }
          .stButton > button[data-testid="stBaseButton-primary"]:disabled {
            opacity: 0.45 !important;
            transform: none !important;
          }
          /* Ensure text inside primary button is white and visible */
          .stButton > button[data-testid="stBaseButton-primary"] p,
          .stButton > button[data-testid="stBaseButton-primary"] span {
            color: #FFFFFF !important;
            font-weight: 700 !important;
          }

          /* FIX #1 & #2: Delete button — red destructive styling, same height as file card */
          .stButton > button[data-testid="stBaseButton-secondary"] {
            width: 100% !important;
            min-width: 44px !important;
            height: 44px !important;
            min-height: 44px !important;
            max-height: 44px !important;
            padding: 0 !important;
            margin: 0 !important;
            border: 1.5px solid rgba(239,68,68,0.35) !important;
            border-radius: 10px !important;
            background: rgba(127,29,29,0.22) !important;
            color: #ef4444 !important;
            font-family: Inter, sans-serif !important;
            font-size: 1.05rem !important;
            font-weight: 700 !important;
            line-height: 1 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            box-shadow: none !important;
            transition: color 160ms ease, border-color 160ms ease, background 160ms ease !important;
          }
          .stButton > button[data-testid="stBaseButton-secondary"]:hover {
            color: #ef4444 !important;
            border-color: #ef4444 !important;
            background: rgba(239,68,68,0.12) !important;
            box-shadow: 0 4px 14px rgba(127,29,29,0.28) !important;
          }
          .stButton > button[data-testid="stBaseButton-secondary"] p,
          .stButton > button[data-testid="stBaseButton-secondary"] span {
            color: #ef4444 !important;
            line-height: 1 !important;
            margin: 0 !important;
          }
          /* Hide emoji text and replace with SVG trash icon via background */
          .stButton > button[data-testid="stBaseButton-secondary"] p {
            font-size: 0 !important;
            line-height: 0 !important;
            width: 16px !important;
            height: 16px !important;
            background-color: #ef4444 !important;
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
          .stButton > button[data-testid="stBaseButton-secondary"]:hover p {
            background-color: #ef4444 !important;
          }


          /* ══ PROCESSING ════════════════════════════ */
          .ms-proc-wrap { margin: 1.5rem 0; animation: fadein 0.3s ease both; }

          .ms-proc-top {
            display: flex; align-items: center;
            justify-content: space-between; margin-bottom: 0.75rem;
          }

          /* FIX #6: Processing title — no icon placeholder */
          .ms-proc-title {
            display: flex; align-items: center;
            font-size: 0.92rem; font-weight: 700; color: var(--ink);
          }

          .ms-elapsed-col { text-align: right; }
          .ms-elapsed-label {
            display: block; color: var(--muted);
            font-size: 0.60rem; font-weight: 700;
            letter-spacing: 0.10em; text-transform: uppercase;
          }
          .ms-elapsed-value {
            display: block; color: var(--ink);
            font-size: 1.20rem; font-weight: 800; letter-spacing: -0.03em;
          }

          .ms-bar-track {
            height: 3px; border-radius: 999px;
            background: rgba(255,255,255,0.05);
            margin-bottom: 1.4rem; overflow: hidden;
          }
          .ms-bar-fill {
            height: 100%; border-radius: 999px;
            background: linear-gradient(90deg, var(--p), var(--p2), var(--p3), var(--p2), var(--p));
            background-size: 300% auto;
            animation: shimmer 2s linear infinite;
          }

          .ms-steps {
            display: grid; grid-template-columns: repeat(5, 1fr);
            position: relative;
          }
          .ms-steps::before {
            content: ""; position: absolute;
            top: 14px; left: 10%; right: 10%;
            height: 1px; background: var(--border);
          }

          .ms-step {
            display: flex; flex-direction: column;
            align-items: center; gap: 0.42rem; position: relative; z-index: 1;
          }

          .ms-step-circle {
            width: 28px; height: 28px; border-radius: 50%;
            border: 1.5px solid rgba(255,255,255,0.10);
            background: var(--bg3);
            display: flex; align-items: center; justify-content: center;
            font-size: 0.70rem; font-weight: 700; color: var(--muted);
            transition: all 200ms;
          }
          .ms-step.done .ms-step-circle {
            border-color: var(--p); background: var(--pg); color: var(--p2);
          }
          .ms-step.active .ms-step-circle {
            border-color: var(--p2); background: var(--p); color: #fff;
            box-shadow: 0 0 0 4px rgba(109,93,246,0.16);
          }

          .ms-step-label {
            color: var(--muted); font-size: 0.67rem; font-weight: 500;
            text-align: center; line-height: 1.3;
          }
          .ms-step.done  .ms-step-label { color: var(--ink2); }
          .ms-step.active .ms-step-label { color: var(--ink); font-weight: 600; }

          /* ══ METRICS ═══════════════════════════════ */
          .ms-metrics-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0,1fr));
            gap: 0.85rem; margin: 1.1rem 0 0.65rem;
          }

          /* FIX #5: Stat cards — no icon placeholder, adjusted padding */
          .ms-stat-card {
            border: 1px solid var(--border); border-radius: 14px;
            background: var(--card2); padding: 1.05rem 1.05rem 1.0rem;
            transition: border-color 200ms, transform 220ms, box-shadow 220ms;
          }
          .ms-stat-card:hover {
            border-color: var(--border2); transform: translateY(-3px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.28);
          }

          /* Remove icon wrap — kept in HTML for color accent only, hidden via display:none */
          .ms-stat-icon-wrap { display: none !important; }

          .ms-stat-label {
            display: block; color: var(--muted);
            font-size: 0.68rem; font-weight: 600;
            letter-spacing: 0.04em; margin-bottom: 0.2rem;
          }
          .ms-stat-value {
            display: block; color: var(--ink);
            font-size: 1.65rem; font-weight: 800;
            letter-spacing: -0.04em; line-height: 1.0;
          }
          .ms-stat-sub {
            display: block; font-size: 0.64rem; font-weight: 700;
            letter-spacing: 0.06em; text-transform: uppercase; margin-top: 0.38rem;
          }
          .ms-stat-card.c-purple  .ms-stat-sub { color: var(--p2); }
          .ms-stat-card.c-emerald .ms-stat-sub { color: var(--emerald); }
          .ms-stat-card.c-amber   .ms-stat-sub { color: var(--amber); }
          .ms-stat-card.c-violet  .ms-stat-sub { color: var(--violet); }

          /* ══ SECURITY NOTE ═════════════════════════ */
          .ms-security {
            display: flex; align-items: center; justify-content: center;
            gap: 0.4rem; color: var(--muted); font-size: 0.74rem;
            margin: 0.4rem 0 1.2rem;
          }

          /* ══ EXPORT ════════════════════════════════ */
          .ms-export-wrap { margin: 1.15rem 0 0.9rem; }

          .ms-export-hdr {
            display: flex; align-items: center; gap: 0.50rem; margin-bottom: 0.12rem;
          }
          .ms-export-hdr-icon {
            width: 26px; height: 26px; border-radius: 8px;
            background: var(--pg); border: 1px solid var(--border-p);
            display: flex; align-items: center; justify-content: center;
            font-size: 0.78rem; color: var(--p2);
          }
          .ms-export-hdr-title { color: var(--ink); font-size: 0.95rem; font-weight: 700; }
          .ms-export-sub { color: var(--muted); font-size: 0.78rem; margin: 0.10rem 0 0.9rem; }

          .stDownloadButton > button {
            min-height: 3.9rem !important;
            border: 1px solid rgba(255,255,255,0.11) !important;
            border-radius: 14px !important;
            background:
              linear-gradient(135deg, rgba(255,255,255,0.055), rgba(255,255,255,0.018)),
              var(--card) !important;
            color: var(--ink) !important;
            font-family: Inter, sans-serif !important;
            text-align: left !important;
            box-shadow: 0 8px 22px rgba(0,0,0,0.18) !important;
            transition: transform 180ms ease, border-color 180ms ease, box-shadow 180ms ease, background 180ms ease !important;
          }
          .stDownloadButton > button:hover {
            transform: translateY(-2px);
            border-color: rgba(20,184,166,0.42) !important;
            background:
              linear-gradient(135deg, rgba(20,184,166,0.10), rgba(34,211,238,0.04)),
              var(--card) !important;
            box-shadow: 0 14px 32px rgba(0,0,0,0.24), 0 0 0 1px rgba(20,184,166,0.08) !important;
          }
          .stDownloadButton > button p {
            color: var(--ink) !important;
            font-weight: 700 !important;
            font-size: 0.86rem !important;
          }

          /* ══ TABS ══════════════════════════════════ */
          .ms-tabs-wrap {
            border: 1px solid var(--border); border-radius: 14px;
            background: var(--card); overflow: hidden; margin-bottom: 1.5rem;
          }

          .stTabs [data-baseweb="tab-list"] {
            gap: 0; border-bottom: 1px solid var(--border);
            padding: 0; margin: 0 0 0.55rem; background: transparent;
          }
          .stTabs [data-baseweb="tab"] {
            border: none; border-bottom: 2px solid transparent;
            border-radius: 0; background: transparent;
            color: var(--muted); font-family: Inter, sans-serif;
            font-weight: 500; font-size: 0.82rem;
            padding: 0.70rem 1.05rem; white-space: nowrap;
            transition: color 150ms;
          }
          .stTabs [data-baseweb="tab"]:hover { color: var(--ink2); }
          .stTabs [aria-selected="true"] {
            color: var(--ink) !important; font-weight: 600 !important;
            background: transparent !important;
          }
          .stTabs [data-baseweb="tab-highlight"] {
            background: var(--p2) !important; height: 2px !important;
          }
          .stTabs [data-baseweb="tab-panel"] {
            padding: 0 !important;
            margin: 0 !important;
         }

          .stTabs [data-baseweb="tab-panel"] [data-testid="stVerticalBlock"] {
            gap: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
          }

          /* FIX #4: Kill the huge empty space above tab content */
          .stTabs [data-baseweb="tab-panel"] > div,
          .stTabs [data-baseweb="tab-panel"] > div > div,
          .stTabs [data-baseweb="tab-panel"] [data-testid="stVerticalBlock"] > div,
          .stTabs [data-baseweb="tab-panel"] [data-testid="stMarkdownContainer"] {
            margin-top: 0 !important;
            padding-top: 0 !important;
          }
          /* Target the element directly wrapping ms-tab-scroll */
          .stTabs [data-baseweb="tab-panel"] [data-testid="stMarkdownContainer"]:has(.ms-tab-scroll) {
            margin: 0 !important;
            padding: 0 !important;
          }
          /* Remove any gap/margin Streamlit injects above first child of tab panel */
          .stTabs [data-baseweb="tab-panel"] [data-testid="stVerticalBlock"] > [data-testid="element-container"]:first-child,
          .stTabs [data-baseweb="tab-panel"] [data-testid="stVerticalBlock"] > div:first-child {
            margin-top: 0 !important;
            padding-top: 0 !important;
          }

          /* per-tab accent colours */
          .stTabs [data-baseweb="tab-list"] button:nth-of-type(1)[aria-selected="true"] { border-bottom-color: var(--p2) !important; color: var(--p2) !important; }
          .stTabs [data-baseweb="tab-list"] button:nth-of-type(2)[aria-selected="true"] { border-bottom-color: #60A5FA !important; color: #60A5FA !important; }
          .stTabs [data-baseweb="tab-list"] button:nth-of-type(3)[aria-selected="true"] { border-bottom-color: #34D399 !important; color: #34D399 !important; }
          .stTabs [data-baseweb="tab-list"] button:nth-of-type(4)[aria-selected="true"] { border-bottom-color: #FCD34D !important; color: #FCD34D !important; }
          .stTabs [data-baseweb="tab-list"] button:nth-of-type(5)[aria-selected="true"] { border-bottom-color: #F472B6 !important; color: #F472B6 !important; }
          .stTabs [data-baseweb="tab-list"] button:nth-of-type(1):hover { color: var(--p2) !important; }
          .stTabs [data-baseweb="tab-list"] button:nth-of-type(2):hover { color: #60A5FA !important; }
          .stTabs [data-baseweb="tab-list"] button:nth-of-type(3):hover { color: #34D399 !important; }
          .stTabs [data-baseweb="tab-list"] button:nth-of-type(4):hover { color: #FCD34D !important; }
          .stTabs [data-baseweb="tab-list"] button:nth-of-type(5):hover { color: #F472B6 !important; }

          .stTabs [data-baseweb="tab"] { border-bottom: 2px solid transparent !important; }

          /* ══ SCROLLABLE CONTENT ══════════════════════ */
          .ms-tab-scroll {
            height: 440px;
            overflow-y: auto;
            padding: 0 !important;
            margin-top: 0 !important;
            scrollbar-width: thin;
            scrollbar-color: rgba(109,93,246,0.30) transparent;
          }
          .ms-tab-scroll::-webkit-scrollbar { width: 5px; }
          .ms-tab-scroll::-webkit-scrollbar-track { background: transparent; }
          .ms-tab-scroll::-webkit-scrollbar-thumb {
            background: rgba(109,93,246,0.25); border-radius: 999px;
          }
          .ms-tab-scroll::-webkit-scrollbar-thumb:hover {
            background: rgba(109,93,246,0.45);
          }

          /* per-tab scroll accent */
          .ms-tab-scroll.accent-blue {
            scrollbar-color: rgba(59,130,246,0.30) transparent;
          }
          .ms-tab-scroll.accent-green {
            scrollbar-color: rgba(16,185,129,0.30) transparent;
          }
          .ms-tab-scroll.accent-amber {
            scrollbar-color: rgba(245,158,11,0.30) transparent;
          }
          .ms-tab-scroll.accent-pink {
            scrollbar-color: rgba(236,72,153,0.30) transparent;
          }

          /* ══ TRANSCRIPT ════════════════════════════ */
          .ms-transcript-scroll {
            height: 480px; overflow-y: auto; margin-top: 0;
            scrollbar-width: thin;
            scrollbar-color: rgba(109,93,246,0.30) transparent;
          }
          .ms-transcript-scroll::-webkit-scrollbar { width: 5px; }
          .ms-transcript-scroll::-webkit-scrollbar-track { background: transparent; }
          .ms-transcript-scroll::-webkit-scrollbar-thumb {
            background: rgba(109,93,246,0.25); border-radius: 999px;
          }
          .ms-transcript-scroll::-webkit-scrollbar-thumb:hover {
            background: rgba(109,93,246,0.45);
          }

          .ms-tr-row {
            display: grid; grid-template-columns: 116px 1fr;
            border-bottom: 1px solid var(--border);
            transition: background 130ms;
          }
          .ms-tr-row:last-child { border-bottom: none; }
          .ms-tr-row:hover { background: rgba(109,93,246,0.04); }

          .ms-tr-left {
            padding: 0.80rem 0.70rem 0.80rem 0.90rem;
            border-right: 1px solid var(--border);
            display: flex; flex-direction: column; gap: 0.22rem;
          }

          .ms-speaker-badge {
            display: inline-block; border-radius: 4px;
            font-size: 0.62rem; font-weight: 700;
            letter-spacing: 0.06em; text-transform: uppercase;
            padding: 0.15rem 0.40rem;
          }
          .ms-speaker-badge.s1 { background: rgba(109,93,246,0.14); color: var(--p2); }
          .ms-speaker-badge.s2 { background: rgba(16,185,129,0.12); color: #34D399; }
          .ms-speaker-badge.s3 { background: rgba(59,130,246,0.12);  color: #60A5FA; }
          .ms-speaker-badge.s4 { background: rgba(245,158,11,0.12);  color: #FCD34D; }

          .ms-tr-timestamp {
            color: var(--muted); font-size: 0.65rem;
            font-variant-numeric: tabular-nums; font-weight: 500;
          }

          .ms-tr-right {
            padding: 0.80rem 0.90rem;
            display: flex; align-items: center;
          }
          .ms-tr-text { color: var(--ink2); font-size: 0.87rem; line-height: 1.55; }

          /* ══ CONTENT CARDS ═════════════════════════ */
          /* Summary (blue) */
          .ms-output-card {
            border: 1px solid rgba(59,130,246,0.18); border-radius: 12px;
            background: rgba(59,130,246,0.04); padding: 1.1rem 1.15rem;
            margin: 0 0 1.1rem; transition: border-color 180ms;
          }
          .ms-output-card:first-child,
          .ms-item-card:first-child,
          .ms-empty:first-child {
            margin-top: 0 !important;
          }
          .ms-output-card:hover { border-color: rgba(59,130,246,0.32); }

          .ms-card-label {
            color: #60A5FA; font-size: 0.63rem; font-weight: 700;
            letter-spacing: 0.09em; text-transform: uppercase; margin-bottom: 0.22rem;
          }
          .ms-card-title {
            color: var(--ink); font-size: 1.05rem; font-weight: 700;
            margin: 0 0 0.65rem; letter-spacing: -0.015em;
          }
          .ms-card-body {
            color: var(--ink2); font-size: 0.89rem; line-height: 1.68;
          }

          .ms-chip-row { display:flex;flex-wrap:wrap;gap:0.35rem;margin-top:0.7rem; }
          .ms-chip {
            border: 1px solid rgba(59,130,246,0.22); border-radius: 999px;
            background: rgba(59,130,246,0.08); color: #93C5FD;
            font-size: 0.73rem; font-weight: 500; padding: 0.22rem 0.55rem;
          }

          /* Discussion (green) */
          .ms-item-card {
            border: 1px solid var(--border); border-radius: 11px;
            background: var(--card); padding: 0.85rem 0.95rem;
            margin: 0 0 1.1rem;
            transition: border-color 180ms, transform 180ms;
          }
          .ms-item-card:hover { transform: translateY(-2px); }

          .ms-item-card.discussion {
            border-color: rgba(16,185,129,0.18);
            background: rgba(16,185,129,0.04);
          }
          .ms-item-card.discussion:hover { border-color: rgba(16,185,129,0.32); }
          .ms-item-card.discussion h4 { color: #34D399; }

          .ms-item-card.decision {
            border-color: rgba(245,158,11,0.18);
            background: rgba(245,158,11,0.04);
          }
          .ms-item-card.decision:hover { border-color: rgba(245,158,11,0.32); }
          .ms-item-card.decision h4 { color: #FCD34D; }

          .ms-item-card.action {
            border-color: rgba(236,72,153,0.18);
            background: rgba(236,72,153,0.04);
          }
          .ms-item-card.action:hover { border-color: rgba(236,72,153,0.32); }
          .ms-item-card.action h4 { color: #F472B6; }

          .ms-item-card h4 {
            font-size: 0.62rem; font-weight: 700;
            letter-spacing: 0.09em; text-transform: uppercase; margin-bottom: 0.30rem;
          }
          .ms-item-card p {
            color: var(--ink2); font-size: 0.88rem; line-height: 1.60; margin-bottom: 0.50rem;
          }
          .ms-meta-row { display:flex;flex-wrap:wrap;gap:0.32rem; }

          /* per-card meta pill colors */
          .ms-meta {
            border: 1px solid var(--border); border-radius: 999px;
            color: var(--muted); background: var(--card2);
            font-size: 0.66rem; font-weight: 500; padding: 0.16rem 0.44rem;
          }
          .ms-item-card.discussion .ms-meta { border-color: rgba(16,185,129,0.15); color: #6EE7B7; background: rgba(16,185,129,0.06); }
          .ms-item-card.decision .ms-meta   { border-color: rgba(245,158,11,0.15); color: #FDE68A; background: rgba(245,158,11,0.06); }
          .ms-item-card.action .ms-meta     { border-color: rgba(236,72,153,0.15); color: #FBCFE8; background: rgba(236,72,153,0.06); }

          /* ══ MISC ═══════════════════════════════════ */
          .ms-empty {
            border: 1px dashed var(--border2); border-radius: 11px;
            background: var(--card2); color: var(--muted);
            font-size: 0.85rem; padding: 2rem 1.5rem; text-align: center;
          }

          .ms-section-header { display:flex;align-items:center;gap:0.5rem;margin-bottom:0.10rem; }
          .ms-section-icon   { color: var(--p2); font-size: 0.95rem; }
          .ms-section-title  {
            color: var(--ink); font-size: 0.95rem; font-weight: 700;
            margin: 0; letter-spacing: -0.015em;
          }
          .ms-section-copy   { color: var(--muted); font-size: 0.80rem; margin: 0.10rem 0 1.1rem; }

          .stTextArea textarea {
            border: 1px solid var(--border) !important; border-radius: 10px !important;
            background: var(--card2) !important; color: var(--ink2) !important;
            font-size: 0.87rem !important; height: 480px !important;
          }
          .stTextArea textarea:focus {
            border-color: var(--border-p) !important;
            box-shadow: 0 0 0 3px rgba(109,93,246,0.08) !important;
          }

          div[data-testid="stAlert"] {
            border-radius: 10px !important; border: 1px solid var(--border) !important;
            background: var(--card2) !important; color: var(--ink) !important;
          }
          div[data-testid="stInfoAlert"] {
            background: var(--pg) !important; border-color: var(--border-p) !important;
          }

          div[data-testid="stProgressBar"] > div {
            background: rgba(255,255,255,0.05) !important; border-radius:999px !important;
          }
          div[data-testid="stProgressBar"] > div > div {
            background: linear-gradient(90deg, var(--p), var(--p2)) !important;
            border-radius: 999px !important;
          }

          [data-testid="stExpander"] {
            border: 1px solid var(--border) !important;
            border-radius: 10px !important; background: var(--card) !important;
          }
          [data-testid="stCaptionContainer"] { color: var(--muted) !important; font-size:0.74rem !important; }

          hr { border-color: var(--border) !important; }

          .ms-footer {
            text-align: center; color: var(--muted); font-size: 0.70rem;
            padding: 1.75rem 0 0.5rem; border-top: 1px solid var(--border); margin-top: 2rem;
          }

          ::-webkit-scrollbar { width: 4px; height: 4px; }
          ::-webkit-scrollbar-track { background: transparent; }
          ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.09); border-radius:999px; }
          ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.16); }

          /* ══ COPY BUTTON ROW ══════════════════════ */
          .ms-copy-row {
            display: flex; align-items: center; gap: 0.5rem;
            padding: 0.55rem 0 0.65rem;
            border-bottom: 1px solid var(--border);
            margin-bottom: 0;
          }

          /* ══ RESPONSIVE ═════════════════════════════ */
          @media (max-width: 860px) {
            .block-container { max-width:100% !important; }
            .ms-metrics-grid { grid-template-columns: repeat(2,minmax(0,1fr)); }
            .ms-steps { grid-template-columns: repeat(3,1fr); }
            .ms-steps > .ms-step:nth-child(n+4) { display:none; }
            .ms-or-wrap {
              min-height: 48px;
              margin: 0.1rem 0;
            }
            .ms-or-wrap::before {
              left: 0.75rem;
              right: 0.75rem;
              top: 50%;
              bottom: auto;
              width: auto;
              height: 1px;
              background: linear-gradient(90deg, transparent, rgba(255,255,255,0.14), transparent);
            }
            div[data-testid="column"] div[data-testid="stVerticalBlockBorderWrapper"] {
              min-height: auto;
            }
          }
          @media (min-width: 769px) and (max-width: 1024px) {
            .block-container {
              padding-left: 1.1rem !important;
              padding-right: 1.1rem !important;
            }
            div[data-testid="stFileUploader"] section {
              min-height: 132px !important;
            }
            .ms-metrics-grid {
              grid-template-columns: repeat(2, minmax(0, 1fr));
            }
          }
          @media (max-width: 768px) {
            .block-container {
              max-width: 100% !important;
              padding-left: 0.9rem !important;
              padding-right: 0.9rem !important;
            }
            div[data-testid="stHorizontalBlock"]:not(:has(button[data-testid="stBaseButton-secondary"])) {
              flex-direction: column !important;
              gap: 0.8rem !important;
            }
            div[data-testid="stHorizontalBlock"]:not(:has(button[data-testid="stBaseButton-secondary"])) > div[data-testid="column"] {
              width: 100% !important;
              min-width: 0 !important;
              flex: 1 1 auto !important;
            }
            div[data-testid="stHorizontalBlock"]:has(button[data-testid="stBaseButton-secondary"]) {
              flex-direction: row !important;
              align-items: center !important;
              gap: 0.5rem !important;
            }
            div[data-testid="stHorizontalBlock"]:has(button[data-testid="stBaseButton-secondary"]) > div[data-testid="column"]:has(button[data-testid="stBaseButton-secondary"]) {
              flex: 0 0 50px !important;
              width: 50px !important;
              min-width: 50px !important;
            }
            div[data-testid="stFileUploader"] section {
              min-height: 128px !important;
              padding: 1.15rem 0.85rem !important;
            }
            .ms-or-wrap {
              min-height: 42px;
              margin: 0;
            }
            .ms-metrics-grid {
              grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .stDownloadButton > button {
              min-height: 3.5rem !important;
            }
          }
          @media (max-width: 640px) {
            .block-container { padding-left:0.85rem !important; padding-right:0.85rem !important; }
            .ms-hero { padding: 0.95rem 0.5rem 1.0rem; }
            .ms-hero h1 { font-size: 2.4rem; }
            .ms-metrics-grid { grid-template-columns: repeat(2,minmax(0,1fr)); gap:0.55rem; }
            .ms-tr-row { grid-template-columns: 96px 1fr; }
            .ms-transcript-scroll { height: 360px; }
            .stTabs [data-baseweb="tab-list"] {
              overflow-x: auto;
              scrollbar-width: none;
            }
            .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
            .stTabs [data-baseweb="tab"] {
              padding: 0.65rem 0.85rem;
              font-size: 0.78rem;
            }
            .ms-file-card {
              padding: 0 0.65rem;
            }
            .ms-file-icon {
              width: 24px;
              height: 24px;
            }
          }
          @media (max-width: 420px) {
            .ms-metrics-grid { grid-template-columns: 1fr; }
            .ms-pill-row { flex-direction: column; align-items: center; }
            .ms-hero h1 { font-size: 2.2rem; }
            .ms-file-name { font-size: 0.76rem; }
            .ms-file-size { font-size: 0.66rem; }
            .stTabs [data-baseweb="tab"] {
              padding: 0.6rem 0.75rem;
              font-size: 0.74rem;
            }
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
        "Uploading",
        "Preparing Audio / File",
        "Identifying Speakers",
        "Generating Notes",
        "Exporting",
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
    note_html = f"<p style='color:var(--muted);font-size:.82rem;margin-top:.75rem;text-align:center'>{html.escape(note)}</p>" if note else ""

    # FIX #6: Removed empty ms-proc-icon div — title is now clean text only
    placeholder.markdown(
        f"""
        <div class="ms-proc-wrap">
          <div class="ms-proc-top">
            <div class="ms-proc-title">
              Preparing Your Report
            </div>
            <div class="ms-elapsed-col">
              <span class="ms-elapsed-label">Elapsed</span>
              <span class="ms-elapsed-value">{elapsed}</span>
            </div>
          </div>
          <div class="ms-bar-track">
            <div class="ms-bar-fill" style="width:{pct}%"></div>
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


def render_hero() -> None:
    # FIX #7: Updated hero description to reflect both audio and transcript upload capabilities
    st.markdown(
        """
        <div class="ms-navbar">
          <div class="ms-navbar-brand">
            <div class="ms-waveicon">
              <span></span><span></span><span></span><span></span><span></span><span></span>
            </div>
            <span class="ms-navbar-name">MeetScribe</span>
          </div>
        </div>
        <div class="ms-hero">
          <h1>MeetScribe</h1>
          <p class="ms-hero-sub">Upload meeting recordings or transcripts and instantly generate structured reports, summaries, decisions, and action items.</p>
          <div class="ms-pill-row">
            <span class="ms-pill">Audio Upload</span>
            <span class="ms-pill">Transcript Upload</span>
            <span class="ms-pill">Meeting Summary</span>
            <span class="ms-pill">Action Items</span>
            <span class="ms-pill">Export Ready</span>
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
    st.markdown(
        """
        <div class="ms-export-wrap">
          <div class="ms-export-hdr">
            <div class="ms-export-hdr-icon">&#8659;</div>
            <span class="ms-export-hdr-title">Export Center</span>
          </div>
          <p class="ms-export-sub">Download polished MoM documents or the full transcript report.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    try:
        mom_pdf_path        = prepared_export_path("pdf_export_path",              export_to_pdf,             analysis)
        mom_docx_path       = prepared_export_path("docx_export_path",             export_to_docx,            analysis)
        transcript_pdf_path = prepared_export_path("transcript_pdf_export_path",   export_transcript_to_pdf,  analysis)
        transcript_docx_path= prepared_export_path("transcript_docx_export_path",  export_transcript_to_docx, analysis)
    except Exception as exc:
        log_stage("Export", "Could not prepare export documents.", error=str(exc), traceback=traceback.format_exc())
        st.error("Downloads could not be prepared. Please try again.")
        return

    top_left, top_right = st.columns(2)
    with top_left:
        render_download_button(
            label="&#128196;  Download MoM (.pdf)   →",
            export_path=mom_pdf_path,
            mime="application/pdf",
            key="download_mom_pdf",
            success_message="Download started",
        )
    with top_right:
        render_download_button(
            label="&#128196;  Download MoM (.docx)  →",
            export_path=mom_docx_path,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            key="download_mom_docx",
            success_message="Download started",
        )

    bot_left, bot_right = st.columns(2)
    with bot_left:
        render_download_button(
            label="&#128196;  Download Transcript (.pdf)   →",
            export_path=transcript_pdf_path,
            mime="application/pdf",
            key="download_transcript_pdf",
            success_message="Download started",
        )
    with bot_right:
        render_download_button(
            label="&#128196;  Download Transcript (.docx)  →",
            export_path=transcript_docx_path,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            key="download_transcript_docx",
            success_message="Download started",
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
        log_stage("Meeting analysis", "Calling analyze_meeting() with single Gemini request.")
        analysis = summarizer.analyze_meeting(transcript_text)

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


def reset_export_state() -> None:
    st.session_state.docx_export_path = ""
    st.session_state.docx_export_error = ""
    st.session_state.pdf_export_path = ""
    st.session_state.pdf_export_error = ""
    st.session_state.transcript_docx_export_path = ""
    st.session_state.transcript_docx_export_error = ""
    st.session_state.transcript_pdf_export_path = ""
    st.session_state.transcript_pdf_export_error = ""


def reset_speaker_mapping_state() -> None:
    st.session_state.speaker_mapping = {}
    st.session_state.speaker_names_available = False
    st.session_state.speaker_review_required = False


def clear_current_report() -> None:
    st.session_state.transcript_text = ""
    st.session_state.transcript_result = None
    st.session_state.uploaded_filename = ""
    reset_speaker_mapping_state()
    reset_report_state()


def render_speaker_review(result: TranscriptionResult) -> None:
    if not st.session_state.get("speaker_review_required", False):
        return
    if not result.segments:
        return

    mapping = current_speaker_mapping() or default_mapping_for_result(result)
    if not mapping:
        return

    with st.container(border=True):
        st.markdown("#### Review speaker names")
        st.caption("Rename speakers before generating the meeting analysis.")
        with st.form("speaker_name_review_form"):
            updated_mapping: SpeakerMapping = {}
            for index, label in enumerate(mapping):
                value = st.text_input(
                    label,
                    value=mapping.get(label, label),
                    key=f"speaker_name_{index}",
                )
                updated_mapping[label] = value.strip() or label

            submitted = st.form_submit_button(
                "Generate Meeting Analysis",
                type="primary",
                use_container_width=True,
            )

    if not submitted:
        return

    st.session_state.speaker_mapping = updated_mapping
    mapped_result = apply_mapping_to_result(result, updated_mapping)
    transcript_text = format_transcript(mapped_result, {})
    st.session_state.transcript_result = mapped_result
    st.session_state.transcript_text = transcript_text
    st.session_state.speaker_review_required = False
    st.session_state.analysis_result = None
    st.session_state.analysis_error = ""
    st.session_state.success_metrics = None
    reset_export_state()
    log_stage(
        "Speaker mapping",
        "Stored reviewed speaker names.",
        speaker_count=len(updated_mapping),
    )

    started_at = time.perf_counter()
    analysis = run_meeting_analysis(
        transcript_text,
        started_at=started_at,
        estimate_note="Generating summary, discussion points, decisions, and action items.",
    )
    if analysis is not None:
        store_success_metrics(
            result=mapped_result,
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
        transcript_text = format_transcript(result)
        if not transcript_text.strip():
            raise ValueError("Formatted transcript is empty.")

        speaker_mapping = named_mapping_for_result(result)
        speaker_names_available = has_participant_names(result)
        if speaker_names_available:
            speaker_mapping = speaker_mapping or default_mapping_for_result(result)
        else:
            speaker_mapping = default_mapping_for_result(result)

        st.session_state.transcript_result = result
        st.session_state.transcript_text = transcript_text
        st.session_state.uploaded_filename = getattr(uploaded_file, "name", "")
        st.session_state.speaker_mapping = speaker_mapping
        st.session_state.speaker_names_available = speaker_names_available
        st.session_state.speaker_review_required = bool(
            result.segments and not speaker_names_available
        )
        log_stage(
            "Session state update",
            "Stored transcript in session state.",
            transcript_chars=len(transcript_text),
            segment_count=len(result.segments),
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

        progress.progress(65, text="Generating Meeting Notes")
        render_stage_status(
            status_placeholder,
            active_index=3,
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
        speaker_mapping = named_mapping_for_result(result)
        speaker_names_available = has_participant_names(result)
        if not speaker_mapping:
            speaker_mapping = default_mapping_for_result(result)
        transcript_text = format_transcript(result, speaker_mapping) if result.segments else result.transcript.strip()
        if not transcript_text.strip():
            raise TranscriptFileError("No transcript text was found in the uploaded file.")

        st.session_state.transcript_result = result
        st.session_state.transcript_text = transcript_text
        st.session_state.uploaded_filename = getattr(transcript_file, "name", "")
        st.session_state.speaker_mapping = speaker_mapping
        st.session_state.speaker_names_available = speaker_names_available
        st.session_state.speaker_review_required = False
        log_stage(
            "Session state update",
            "Stored uploaded transcript in session state.",
            transcript_chars=len(transcript_text),
            segment_count=len(result.segments),
            speaker_names_available=speaker_names_available,
        )

        progress.progress(55, text="Generating Meeting Notes")
        render_stage_status(
            status_placeholder,
            active_index=3,
            started_at=started_at,
            note="Generating structured notes from the uploaded transcript.",
        )
        analysis = run_meeting_analysis(
            transcript_text,
            progress=progress,
            status_placeholder=status_placeholder,
            started_at=started_at,
            estimate_note="Generating summary, discussion points, decisions, and action items.",
        )
        if analysis is not None:
            store_success_metrics(
                result=result,
                analysis=analysis,
                started_at=started_at,
            )
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
            "Generate Meeting Report",
            type="primary",
            disabled=uploaded_file is None and transcript_file is None,
            use_container_width=True,
        )

        if process_clicked:
            if transcript_file is not None:
                process_transcript_upload(transcript_file)
            elif uploaded_file is not None:
                process_upload(uploaded_file)

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
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        render_analysis_error()
        render_success_metrics()
        render_speaker_review(result)

        # Export above tabs
        if analysis is not None:
            render_export_card(analysis)

        # Tabs
        (
            transcript_tab,
            summary_tab,
            key_points_tab,
            decisions_tab,
            action_items_tab,
        ) = st.tabs([
            "Transcript",
            "Summary",
            "Discussion Points",
            "Decisions",
            "Action Items",
        ])

        with transcript_tab:
            render_copy_button(transcript_text)
            render_transcript(result, current_speaker_mapping())

        with summary_tab:
            if analysis is None:
                empty_card("Meeting notes are not ready yet.")
            else:
                render_summary_tab(analysis)

        with key_points_tab:
            if analysis is None:
                empty_card("Meeting notes are not ready yet.")
            else:
                render_key_points_tab(analysis)

        with decisions_tab:
            if analysis is None:
                empty_card("Meeting notes are not ready yet.")
            else:
                render_decisions_tab(analysis)

        with action_items_tab:
            if analysis is None:
                empty_card("Meeting notes are not ready yet.")
            else:
                render_action_items_tab(analysis)

    st.markdown(
        "<div class='ms-footer'>&#169; 2026 MeetScribe. All rights reserved.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
