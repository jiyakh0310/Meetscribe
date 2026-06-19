"""PDF export for professional Minutes of Meeting documents."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from summarization.base_summarizer import MeetingAnalysisResult

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXPORT_DIR = PROJECT_ROOT / "exports" / "generated"

INK_BLUE = colors.HexColor("#0B2545")
HEADING_BLUE = colors.HexColor("#2E74B5")
DARK_BLUE = colors.HexColor("#1F4D78")
MUTED = colors.HexColor("#667085")
LIGHT_GRAY = colors.HexColor("#F2F4F7")
BORDER_GRAY = colors.HexColor("#D9E2EC")
CONTENT_WIDTH = 6.5 * inch


def export_to_pdf(
    analysis: MeetingAnalysisResult | dict[str, Any],
    *,
    output_path: str | Path | None = None,
    output_dir: str | Path = DEFAULT_EXPORT_DIR,
    meeting_info: dict[str, str] | None = None,
) -> Path:
    """Create a concise Minutes of Meeting PDF and return its file path."""
    payload = _analysis_to_dict(analysis)
    export_path = _resolve_output_path(output_path, output_dir)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    styles = _build_styles()
    story: list[Any] = []

    _add_mom_title_block(story, styles, payload, meeting_info=meeting_info)
    _add_executive_summary(story, styles, payload)
    _add_concise_topics(story, styles, payload)
    _add_concise_key_discussion_points(story, styles, payload)
    _add_concise_decisions(story, styles, payload)
    _add_action_items(story, styles, payload)
    _add_next_steps(story, styles, payload)
    _add_end_marker(story, styles)

    document = SimpleDocTemplate(
        str(export_path),
        pagesize=LETTER,
        rightMargin=inch,
        leftMargin=inch,
        topMargin=inch,
        bottomMargin=inch,
        title=str(payload.get("summary", {}).get("title") or "Minutes of Meeting"),
        author="MeetScribe",
    )
    document.build(story, onFirstPage=_draw_footer, onLaterPages=_draw_footer)
    return export_path


def export_transcript_to_pdf(
    analysis: MeetingAnalysisResult | dict[str, Any],
    *,
    output_path: str | Path | None = None,
    output_dir: str | Path = DEFAULT_EXPORT_DIR,
    meeting_info: dict[str, str] | None = None,
) -> Path:
    """Create a standalone cleaned transcript PDF and return its file path."""
    payload = _analysis_to_dict(analysis)
    export_path = _resolve_transcript_output_path(output_path, output_dir, "pdf")
    export_path.parent.mkdir(parents=True, exist_ok=True)

    styles = _build_styles()
    story: list[Any] = []
    _add_transcript_title_block(story, styles, payload, meeting_info=meeting_info)
    _add_cleaned_transcript(story, styles, payload, force_page_break=False)

    document = SimpleDocTemplate(
        str(export_path),
        pagesize=LETTER,
        rightMargin=inch,
        leftMargin=inch,
        topMargin=inch,
        bottomMargin=inch,
        title=str(payload.get("summary", {}).get("title") or "Transcript Report"),
        author="MeetScribe",
    )
    document.build(story, onFirstPage=_draw_footer, onLaterPages=_draw_footer)
    return export_path


def _analysis_to_dict(
    analysis: MeetingAnalysisResult | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(analysis, MeetingAnalysisResult):
        return analysis.to_dict()
    return analysis


def _resolve_output_path(
    output_path: str | Path | None,
    output_dir: str | Path,
) -> Path:
    if output_path is not None:
        return Path(output_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(output_dir) / f"minutes_of_meeting_{timestamp}.pdf"


def _resolve_transcript_output_path(
    output_path: str | Path | None,
    output_dir: str | Path,
    extension: str,
) -> Path:
    if output_path is not None:
        return Path(output_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(output_dir) / f"transcript_report_{timestamp}.{extension}"


def _build_styles() -> dict[str, ParagraphStyle]:
    base_styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "MeetScribeTitle",
            parent=base_styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=INK_BLUE,
            alignment=TA_LEFT,
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "MeetScribeSubtitle",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=13,
            textColor=MUTED,
            spaceAfter=14,
        ),
        "subtitle_bold": ParagraphStyle(
            "MeetScribeSubtitleBold",
            parent=base_styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=DARK_BLUE,
            spaceAfter=8,
        ),
        "h1": ParagraphStyle(
            "MeetScribeHeading1",
            parent=base_styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            textColor=HEADING_BLUE,
            spaceBefore=12,
            spaceAfter=7,
        ),
        "body": ParagraphStyle(
            "MeetScribeBody",
            parent=base_styles["BodyText"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=14,
            textColor=colors.black,
            spaceAfter=7,
        ),
        "bullet": ParagraphStyle(
            "MeetScribeBullet",
            parent=base_styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=13,
            leftIndent=16,
            firstLineIndent=-8,
            spaceAfter=5,
        ),
        "meta_indent": ParagraphStyle(
            "MeetScribeMetaIndent",
            parent=base_styles["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=8.5,
            leading=11,
            textColor=MUTED,
            leftIndent=24,
            spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "MeetScribeSmall",
            parent=base_styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=MUTED,
            spaceAfter=2,
        ),
        "speaker": ParagraphStyle(
            "MeetScribeSpeaker",
            parent=base_styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=DARK_BLUE,
            spaceBefore=5,
            spaceAfter=2,
        ),
        "end": ParagraphStyle(
            "MeetScribeEnd",
            parent=base_styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12,
            textColor=MUTED,
            alignment=1,
            spaceBefore=8,
        ),
    }


def _add_mom_title_block(
    story: list[Any],
    styles: dict[str, ParagraphStyle],
    payload: dict[str, Any],
    *,
    meeting_info: dict[str, str] | None,
) -> None:
    title = payload.get("summary", {}).get("title") or "Minutes of Meeting"
    story.append(Paragraph("MINUTES OF MEETING", styles["title"]))
    story.append(Paragraph(_escape(str(title)), styles["subtitle_bold"]))

    _add_compact_meeting_information(story, styles, payload, meeting_info=meeting_info)
    story.append(Spacer(1, 10))


def _add_transcript_title_block(
    story: list[Any],
    styles: dict[str, ParagraphStyle],
    payload: dict[str, Any],
    *,
    meeting_info: dict[str, str] | None,
) -> None:
    title = payload.get("summary", {}).get("title") or "Transcript Report"
    story.append(Paragraph(_escape(str(title)), styles["title"]))
    story.append(Paragraph("Transcript Report", styles["subtitle"]))
    if meeting_info and meeting_info.get("Source File"):
        story.append(Paragraph(
            f"<b>Source Audio File:</b> {_escape(meeting_info['Source File'])}",
            styles["small"],
        ))
    story.append(Spacer(1, 8))


def _add_meeting_information(
    story: list[Any],
    styles: dict[str, ParagraphStyle],
    *,
    meeting_info: dict[str, str] | None,
) -> None:
    _add_heading(story, styles, "Meeting Information")
    info = {
        "Generated On": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Prepared By": "MeetScribe",
    }
    if meeting_info:
        info.update({key: value for key, value in meeting_info.items() if value})

    rows = [
        [
            Paragraph("<b>Field</b>", styles["body"]),
            Paragraph("<b>Value</b>", styles["body"]),
        ]
    ]
    for key, value in info.items():
        rows.append([
            Paragraph(_escape(key), styles["body"]),
            Paragraph(_escape(value), styles["body"]),
        ])

    story.append(_table(rows, [1.55 * inch, 4.95 * inch]))
    story.append(Spacer(1, 8))


def _add_compact_meeting_information(
    story: list[Any],
    styles: dict[str, ParagraphStyle],
    payload: dict[str, Any],
    *,
    meeting_info: dict[str, str] | None,
) -> None:
    _add_heading(story, styles, "Meeting Information")
    metadata = _meeting_metadata(payload, meeting_info=meeting_info)
    rows = []
    for key, value in metadata.items():
        rows.append([
            Paragraph(f"<b>{_escape(key)}</b>", styles["body"]),
            Paragraph(_escape(value), styles["body"]),
        ])
    story.append(_table(rows, [1.7 * inch, 4.8 * inch], header=False))
    story.append(Spacer(1, 6))


def _add_summary_sections(
    story: list[Any],
    styles: dict[str, ParagraphStyle],
    payload: dict[str, Any],
) -> None:
    summary = payload.get("summary", {})
    _add_heading(story, styles, "Executive Summary")
    story.append(Paragraph(_escape(summary.get("short_summary") or "N/A"), styles["body"]))

    _add_heading(story, styles, "Detailed Summary")
    story.append(Paragraph(_escape(summary.get("detailed_summary") or "N/A"), styles["body"]))


def _add_executive_summary(
    story: list[Any],
    styles: dict[str, ParagraphStyle],
    payload: dict[str, Any],
) -> None:
    _add_heading(story, styles, "EXECUTIVE SUMMARY")
    for paragraph_text in _executive_summary_paragraphs(payload):
        story.append(Paragraph(_escape(paragraph_text), styles["body"]))


def _add_topics(
    story: list[Any],
    styles: dict[str, ParagraphStyle],
    payload: dict[str, Any],
) -> None:
    _add_heading(story, styles, "Topics Discussed")
    topics = payload.get("summary", {}).get("topics_discussed", [])
    _add_bullets(story, styles, topics)


def _add_concise_topics(
    story: list[Any],
    styles: dict[str, ParagraphStyle],
    payload: dict[str, Any],
) -> None:
    _add_heading(story, styles, "TOPICS DISCUSSED")
    topics = payload.get("summary", {}).get("topics_discussed", [])
    _add_bullets(story, styles, topics[:8])


def _add_key_discussion_points(
    story: list[Any],
    styles: dict[str, ParagraphStyle],
    payload: dict[str, Any],
) -> None:
    _add_heading(story, styles, "Key Discussion Points")
    points = payload.get("key_discussion_points", [])
    if not points:
        story.append(Paragraph("No key discussion points were extracted.", styles["body"]))
        return

    for item in points:
        timestamp = item.get("timestamp") or "N/A"
        speakers = ", ".join(item.get("speakers") or []) or "N/A"
        text = (
            f"<b>[{_escape(timestamp)}]</b> {_escape(item.get('point') or '')} "
            f"<i>({_escape(speakers)})</i>"
        )
        story.append(Paragraph(f"- {text}", styles["bullet"]))


def _add_concise_key_discussion_points(
    story: list[Any],
    styles: dict[str, ParagraphStyle],
    payload: dict[str, Any],
) -> None:
    _add_heading(story, styles, "KEY DISCUSSION POINTS")
    points = payload.get("key_discussion_points", [])
    if not points:
        story.append(Paragraph("No key discussion points were extracted.", styles["body"]))
        return

    for item in points[:6]:
        point = _escape(item.get("point") or "")
        timestamp = _escape(item.get("timestamp") or "N/A")
        speakers = _escape(", ".join(item.get("speakers") or []) or "N/A")
        story.append(Paragraph(f"- <b>{point}</b>", styles["bullet"]))
        story.append(Paragraph(
            f"Timestamp: {timestamp} | Speaker(s): {speakers}",
            styles["meta_indent"],
        ))


def _add_decisions(
    story: list[Any],
    styles: dict[str, ParagraphStyle],
    payload: dict[str, Any],
) -> None:
    _add_heading(story, styles, "Decisions")
    decisions = payload.get("decisions", [])
    if not decisions:
        story.append(Paragraph("No decisions were extracted.", styles["body"]))
        return

    for item in decisions:
        timestamp = item.get("timestamp") or "N/A"
        owner = item.get("owner") or "Unassigned"
        confidence = item.get("confidence") or "N/A"
        text = (
            f"<b>[{_escape(timestamp)}]</b> {_escape(item.get('decision') or '')} "
            f"<i>Owner: {_escape(owner)}; confidence: {_escape(confidence)}.</i>"
        )
        story.append(Paragraph(f"- {text}", styles["bullet"]))


def _add_concise_decisions(
    story: list[Any],
    styles: dict[str, ParagraphStyle],
    payload: dict[str, Any],
) -> None:
    _add_heading(story, styles, "DECISIONS TAKEN")
    decisions = payload.get("decisions", [])
    if not decisions:
        story.append(Paragraph("No decisions were extracted.", styles["body"]))
        return

    for index, item in enumerate(decisions, start=1):
        story.append(Paragraph(
            f"{index}. {_escape(_professionalize_sentence(item.get('decision') or ''))}",
            styles["bullet"],
        ))


def _add_action_items(
    story: list[Any],
    styles: dict[str, ParagraphStyle],
    payload: dict[str, Any],
) -> None:
    _add_heading(story, styles, "ACTION ITEMS")
    action_items = payload.get("action_items", [])
    if not action_items:
        story.append(Paragraph("No action items were extracted.", styles["body"]))
        return

    rows = [[
        Paragraph("<b>Task</b>", styles["body"]),
        Paragraph("<b>Owner</b>", styles["body"]),
        Paragraph("<b>Due Date</b>", styles["body"]),
        Paragraph("<b>Priority</b>", styles["body"]),
        Paragraph("<b>Status</b>", styles["body"]),
    ]]
    for item in action_items:
        rows.append([
            Paragraph(_escape(item.get("task") or ""), styles["body"]),
            Paragraph(_escape(item.get("owner") or "Unassigned"), styles["body"]),
            Paragraph(_escape(item.get("due_date") or "N/A"), styles["body"]),
            Paragraph(_escape(_priority_for_action_item(item)), styles["body"]),
            Paragraph(_escape(item.get("status") or "N/A"), styles["body"]),
        ])

    story.append(_table(
        rows,
        [2.45 * inch, 1.0 * inch, 1.0 * inch, 0.85 * inch, 0.8 * inch],
    ))


def _add_next_steps(
    story: list[Any],
    styles: dict[str, ParagraphStyle],
    payload: dict[str, Any],
) -> None:
    _add_heading(story, styles, "NEXT STEPS")
    action_items = payload.get("action_items", [])
    if not action_items:
        story.append(Paragraph("No next steps were identified.", styles["body"]))
        return

    for item in action_items:
        story.append(Paragraph(f"- {_escape(_next_step_sentence(item))}", styles["bullet"]))
    for item in payload.get("decisions", [])[:3]:
        decision = _professionalize_sentence(item.get("decision", "")).rstrip(".")
        if decision:
            story.append(Paragraph(
                f"- Team will proceed in line with: {_escape(decision)}.",
                styles["bullet"],
            ))


def _add_end_marker(story: list[Any], styles: dict[str, ParagraphStyle]) -> None:
    story.append(Spacer(1, 10))
    story.append(Paragraph("Generated by MeetScribe", styles["end"]))


def _add_cleaned_transcript(
    story: list[Any],
    styles: dict[str, ParagraphStyle],
    payload: dict[str, Any],
    *,
    force_page_break: bool = True,
) -> None:
    if force_page_break:
        story.append(PageBreak())
    _add_heading(story, styles, "Cleaned Transcript")
    transcript = str(payload.get("cleaned_transcript", "")).strip()
    if not transcript:
        story.append(Paragraph("No cleaned transcript was available.", styles["body"]))
        return

    for block in re.split(r"\n\s*\n", transcript):
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        story.append(Paragraph(_escape(lines[0].strip()), styles["speaker"]))
        body = " ".join(line.strip() for line in lines[1:] if line.strip())
        if body:
            story.append(Paragraph(_escape(body), styles["body"]))


def _add_heading(
    story: list[Any],
    styles: dict[str, ParagraphStyle],
    text: str,
) -> None:
    story.append(Paragraph(_escape(text), styles["h1"]))


def _add_bullets(
    story: list[Any],
    styles: dict[str, ParagraphStyle],
    items: list[Any],
) -> None:
    if not items:
        story.append(Paragraph("No items were extracted.", styles["body"]))
        return

    for item in items:
        story.append(Paragraph(f"- {_escape(str(item))}", styles["bullet"]))


def _table(rows: list[list[Any]], col_widths: list[float], *, header: bool = True) -> Table:
    table = Table(rows, colWidths=col_widths, hAlign="LEFT", repeatRows=1)
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.45, BORDER_GRAY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        commands.append(("BACKGROUND", (0, 0), (-1, 0), LIGHT_GRAY))
    table.setStyle(TableStyle(commands))
    return table


def _draw_footer(canvas: Any, document: SimpleDocTemplate) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawRightString(
        document.pagesize[0] - document.rightMargin,
        0.55 * inch,
        f"Generated by MeetScribe | Page {document.page}",
    )
    canvas.restoreState()


def _next_step_sentence(item: dict[str, Any]) -> str:
    owner = item.get("owner") or "Team"
    task = str(item.get("task") or "complete the assigned task").rstrip(".")
    due_date = item.get("due_date")
    action = task[0].lower() + task[1:] if task else "complete the assigned task"
    if due_date and str(due_date).upper() != "N/A":
        return f"{owner} will {action} by {due_date}."
    return f"{owner} will {action}."


def _meeting_metadata(
    payload: dict[str, Any],
    *,
    meeting_info: dict[str, str] | None,
) -> dict[str, str]:
    meeting_info = meeting_info or {}
    transcript = str(payload.get("cleaned_transcript", ""))
    return {
        "Date": meeting_info.get("Date")
        or meeting_info.get("Generated Date")
        or datetime.now().strftime("%Y-%m-%d"),
        "Duration": meeting_info.get("Duration") or _duration_from_transcript(transcript),
        "Speakers Detected": meeting_info.get("Speakers Detected")
        or str(len(_speakers_from_transcript(transcript)) or "N/A"),
        "Generated By": "MeetScribe",
    }


def _duration_from_transcript(transcript: str) -> str:
    timestamp_ranges = re.findall(
        r"\[([0-9]{1,2}:[0-9]{2}(?::[0-9]{2})?)\s*-\s*([0-9]{1,2}:[0-9]{2}(?::[0-9]{2})?)\]",
        transcript,
    )
    if not timestamp_ranges:
        return "N/A"

    start_seconds = _timestamp_to_seconds(timestamp_ranges[0][0])
    end_seconds = _timestamp_to_seconds(timestamp_ranges[-1][1])
    if start_seconds is None or end_seconds is None or end_seconds <= start_seconds:
        return "N/A"

    total_minutes = max(1, round((end_seconds - start_seconds) / 60))
    return f"{total_minutes} min"


def _timestamp_to_seconds(value: str) -> int | None:
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    return None


def _speakers_from_transcript(transcript: str) -> set[str]:
    return set(re.findall(r"^(Speaker\s+\S+)", transcript, flags=re.MULTILINE))


def _executive_summary_paragraphs(payload: dict[str, Any]) -> list[str]:
    summary = payload.get("summary", {})
    detailed = _sanitize_business_text(
        str(summary.get("detailed_summary") or summary.get("short_summary") or "").strip()
    )
    short = _sanitize_business_text(str(summary.get("short_summary") or "").strip())
    decisions = payload.get("decisions", [])
    actions = payload.get("action_items", [])

    paragraphs: list[str] = []
    if detailed:
        paragraphs.append(detailed)
    elif short:
        paragraphs.append(short)
    else:
        paragraphs.append("The meeting covered the listed agenda items and produced the outcomes captured in this report.")

    decision_text = "; ".join(
        _professionalize_sentence(item.get("decision", "")).rstrip(".")
        for item in decisions[:3]
        if item.get("decision")
    )
    if decision_text:
        paragraphs.append(f"Key decisions included {decision_text}.")

    if actions:
        owners = sorted({str(item.get("owner")) for item in actions if item.get("owner")})
        owner_text = ", ".join(owners[:4]) if owners else "the assigned owners"
        paragraphs.append(
            f"Follow-up activities were assigned to {owner_text}, with progress to be tracked through the action items below."
        )

    return paragraphs[:3]


def _sanitize_business_text(text: str) -> str:
    replacements = {
        "The discussion revolves around": "The meeting addressed",
        "The discussion centered around": "The meeting addressed",
        "The discussion focused on": "The meeting focused on",
        "The conversation focused on": "The meeting focused on",
        "They talked about": "Participants reviewed",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def _professionalize_sentence(value: Any) -> str:
    text = _sanitize_business_text(str(value or "").strip())
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    if text[-1] not in ".?!":
        text += "."
    return text


def _priority_for_action_item(item: dict[str, Any]) -> str:
    text = " ".join(
        str(item.get(key) or "")
        for key in ("task", "due_date", "status")
    ).lower()
    if any(token in text for token in ("urgent", "today", "tomorrow", "blocked", "critical")):
        return "High"
    if any(token in text for token in ("this week", "friday", "next demo", "review", "validate")):
        return "Medium"
    return "Low"


def _escape(value: Any) -> str:
    text = str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
