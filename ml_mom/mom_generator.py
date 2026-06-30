"""Generate rule-based Minutes of Meeting from ANN prediction output.

Purpose:
    Convert sentence-level ANN predictions into a structured Minutes of Meeting
    document without using LLMs, Agentic AI, Gemini, OpenAI, or external APIs.

Responsibilities:
    - Read prediction output from JSON or CSV.
    - Group sentences by predicted label.
    - Build deterministic MoM sections from available predicted labels.
    - Extract simple owner and deadline hints for action-item rows.
    - Save generated MoM as JSON and plain text.

Inputs:
    ``datasets/predictions/prediction.json`` or
    ``datasets/predictions/prediction.csv``.

Outputs:
    - ``datasets/output/generated_mom.json``
    - ``datasets/output/generated_mom.txt``

Safety:
    This generator is offline and deterministic. It does not touch production
    app code, workflows, UI, email, exports, parser, training, or prediction
    modules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
import json
from pathlib import Path
import re
import sys
from typing import Any


PREDICTION_JSON_PATH = Path("datasets") / "predictions" / "prediction.json"
PREDICTION_CSV_PATH = Path("datasets") / "predictions" / "prediction.csv"
OUTPUT_DIR = Path("datasets") / "output"
OUTPUT_JSON_PATH = OUTPUT_DIR / "generated_mom.json"
OUTPUT_TEXT_PATH = OUTPUT_DIR / "generated_mom.txt"

SUPPORTED_LABELS = {
    "Discussion",
    "Decision",
    "Action_Item",
    "Summary",
    "Information",
}
DEADLINE_PATTERN = re.compile(
    r"(?i)\b(today|tomorrow|monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday|next week|this week|eod|end of day|"
    r"\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\b"
)


@dataclass(slots=True)
class PredictionRecord:
    """Single ANN prediction record used to generate MoM sections.

    Attributes:
        sentence: Sentence text.
        speaker: Speaker associated with the sentence.
        timestamp: Optional timestamp in seconds.
        predicted_label: ANN-predicted label.
        confidence_score: Prediction confidence.
    """

    sentence: str
    speaker: str
    timestamp: int | None
    predicted_label: str
    confidence_score: float


@dataclass(slots=True)
class ActionItem:
    """Action item row for the generated MoM.

    Attributes:
        owner: Responsible speaker or ``-`` when unavailable.
        task: Action sentence.
        status: Current status placeholder.
        deadline: Extracted deadline or ``-`` when unavailable.
    """

    owner: str
    task: str
    status: str
    deadline: str


@dataclass(slots=True)
class MeetingMinutes:
    """Structured Minutes of Meeting output.

    Attributes:
        meeting_title: Meeting title.
        meeting_date: Meeting date.
        participants: Participant names inferred from prediction speakers.
        prepared_by: Static prepared-by value.
        executive_summary: Summary bullets.
        discussion_points: Discussion bullets.
        key_decisions: Decision bullets.
        action_items: Action item table rows.
        additional_information: Optional information bullets.
        next_meeting: Optional next-meeting bullets.
        warnings: Non-crashing generation warnings.
    """

    meeting_title: str
    meeting_date: str
    participants: list[str]
    prepared_by: str
    executive_summary: list[str] = field(default_factory=list)
    discussion_points: list[str] = field(default_factory=list)
    key_decisions: list[str] = field(default_factory=list)
    action_items: list[ActionItem] = field(default_factory=list)
    additional_information: list[str] = field(default_factory=list)
    next_meeting: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def read_predictions(
    json_path: Path = PREDICTION_JSON_PATH,
    csv_path: Path = PREDICTION_CSV_PATH,
) -> tuple[list[PredictionRecord], list[str]]:
    """Read prediction output from JSON or CSV.

    Args:
        json_path: Preferred prediction JSON path.
        csv_path: Fallback prediction CSV path.

    Returns:
        Tuple of prediction records and warning messages.
    """

    if json_path.exists():
        return read_prediction_json(json_path)
    if csv_path.exists():
        return read_prediction_csv(csv_path)
    return [], ["Prediction output file was not found."]


def read_prediction_json(path: Path) -> tuple[list[PredictionRecord], list[str]]:
    """Read prediction records from JSON safely.

    Args:
        path: Prediction JSON path.

    Returns:
        Tuple of prediction records and warnings.
    """

    try:
        raw_rows = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [], [f"Prediction JSON is corrupted: {exc}"]
    except OSError as exc:
        return [], [f"Prediction JSON could not be read: {exc}"]

    if not isinstance(raw_rows, list):
        return [], ["Prediction JSON must contain a list of records."]
    return normalize_prediction_rows(raw_rows)


def read_prediction_csv(path: Path) -> tuple[list[PredictionRecord], list[str]]:
    """Read prediction records from CSV safely.

    Args:
        path: Prediction CSV path.

    Returns:
        Tuple of prediction records and warnings.
    """

    try:
        with path.open("r", newline="", encoding="utf-8") as csv_file:
            raw_rows = [dict(row) for row in csv.DictReader(csv_file)]
    except csv.Error as exc:
        return [], [f"Prediction CSV is corrupted: {exc}"]
    except OSError as exc:
        return [], [f"Prediction CSV could not be read: {exc}"]
    return normalize_prediction_rows(raw_rows)


def normalize_prediction_rows(raw_rows: list[Any]) -> tuple[list[PredictionRecord], list[str]]:
    """Normalize raw prediction rows into typed records.

    Args:
        raw_rows: Raw JSON or CSV records.

    Returns:
        Tuple of usable prediction records and warnings.
    """

    records: list[PredictionRecord] = []
    warnings: list[str] = []
    for index, row in enumerate(raw_rows, start=1):
        if not isinstance(row, dict):
            warnings.append(f"Prediction row {index} was skipped because it is not an object.")
            continue

        sentence = str(row.get("sentence") or "").strip()
        label = str(row.get("predicted_label") or "").strip()
        if not sentence:
            warnings.append(f"Prediction row {index} was skipped because sentence is empty.")
            continue
        if label not in SUPPORTED_LABELS:
            warnings.append(f"Prediction row {index} has missing or unsupported label: {label}")
            label = "Information"

        records.append(
            PredictionRecord(
                sentence=sentence,
                speaker=str(row.get("speaker") or "").strip() or "-",
                timestamp=parse_optional_int(row.get("timestamp")),
                predicted_label=label,
                confidence_score=parse_optional_float(row.get("confidence_score")),
            )
        )
    return records, warnings


def group_by_label(records: list[PredictionRecord]) -> dict[str, list[PredictionRecord]]:
    """Group prediction records by predicted label.

    Args:
        records: Prediction records.

    Returns:
        Mapping from label to records.
    """

    grouped = {label: [] for label in SUPPORTED_LABELS}
    for record in records:
        grouped.setdefault(record.predicted_label, []).append(record)
    return grouped


def generate_minutes(
    records: list[PredictionRecord],
    warnings: list[str] | None = None,
) -> MeetingMinutes:
    """Generate structured Minutes of Meeting from predictions.

    Args:
        records: Prediction records.
        warnings: Existing non-fatal warnings.

    Returns:
        Structured ``MeetingMinutes`` object.
    """

    warnings = list(warnings or [])
    if not records:
        warnings.append("No prediction records were available for MoM generation.")

    grouped = group_by_label(records)
    participants = sorted({record.speaker for record in records if record.speaker != "-"})

    summary_records = grouped.get("Summary", [])
    if summary_records:
        executive_summary = unique_sentences(summary_records)
    else:
        # When the ANN predicts no Summary rows, build a concise extractive
        # summary from Information and Discussion only, as required.
        fallback_records = grouped.get("Information", []) + grouped.get("Discussion", [])
        executive_summary = unique_sentences(fallback_records[:5])

    return MeetingMinutes(
        meeting_title="Meeting Title",
        meeting_date="Meeting Date",
        participants=participants,
        prepared_by="MeetScribe",
        executive_summary=executive_summary,
        discussion_points=unique_sentences(grouped.get("Discussion", [])),
        key_decisions=unique_sentences(grouped.get("Decision", [])),
        action_items=[build_action_item(record) for record in grouped.get("Action_Item", [])],
        additional_information=unique_sentences(grouped.get("Information", [])),
        next_meeting=extract_next_meeting_items(records),
        warnings=warnings,
    )


def build_action_item(record: PredictionRecord) -> ActionItem:
    """Build an action-item table row from an action prediction.

    Args:
        record: Prediction record labeled ``Action_Item``.

    Returns:
        Action item row.
    """

    return ActionItem(
        owner=record.speaker if record.speaker else "-",
        task=record.sentence,
        status="-",
        deadline=extract_deadline(record.sentence),
    )


def extract_deadline(sentence: str) -> str:
    """Extract a simple deadline phrase from an action sentence.

    Args:
        sentence: Action item sentence.

    Returns:
        Deadline text or ``-`` when unavailable.
    """

    match = DEADLINE_PATTERN.search(sentence)
    return match.group(0) if match else "-"


def extract_next_meeting_items(records: list[PredictionRecord]) -> list[str]:
    """Extract optional next-meeting items using deterministic keyword rules.

    Args:
        records: Prediction records.

    Returns:
        Sentences that appear to mention a next meeting.
    """

    next_meeting_items = []
    for record in records:
        lowered = record.sentence.lower()
        if "next meeting" in lowered or "next call" in lowered:
            next_meeting_items.append(record.sentence)
    return sorted(set(next_meeting_items))


def unique_sentences(records: list[PredictionRecord]) -> list[str]:
    """Return unique sentence text while preserving order.

    Args:
        records: Prediction records.

    Returns:
        Unique sentence strings.
    """

    seen: set[str] = set()
    sentences: list[str] = []
    for record in records:
        key = record.sentence.strip().lower()
        if key and key not in seen:
            seen.add(key)
            sentences.append(record.sentence)
    return sentences


def save_minutes(minutes: MeetingMinutes) -> None:
    """Save generated MoM to JSON and text files.

    Args:
        minutes: Structured minutes object.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON_PATH.write_text(
        json.dumps(minutes_to_dict(minutes), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    OUTPUT_TEXT_PATH.write_text(render_minutes_text(minutes), encoding="utf-8")
    print(f"Generated MoM JSON saved to: {OUTPUT_JSON_PATH}")
    print(f"Generated MoM text saved to: {OUTPUT_TEXT_PATH}")


def minutes_to_dict(minutes: MeetingMinutes) -> dict[str, Any]:
    """Convert minutes dataclass into a JSON-serializable dictionary.

    Args:
        minutes: Structured minutes object.

    Returns:
        JSON-serializable dictionary.
    """

    data = asdict(minutes)
    data["action_items"] = [asdict(action_item) for action_item in minutes.action_items]
    return data


def render_minutes_text(minutes: MeetingMinutes) -> str:
    """Render structured minutes as plain text.

    Args:
        minutes: Structured minutes object.

    Returns:
        Plain-text MoM.
    """

    lines: list[str] = []
    lines.append(minutes.meeting_title)
    lines.append(minutes.meeting_date)
    lines.append("Participants: " + (", ".join(minutes.participants) if minutes.participants else "-"))
    lines.append(f"Prepared By {minutes.prepared_by}")
    lines.append("")
    lines.append("Executive Summary")
    lines.extend(render_bullets(minutes.executive_summary))
    lines.append("")
    lines.append("Discussion Points")
    lines.extend(render_bullets(minutes.discussion_points))
    lines.append("")
    lines.append("Key Decisions")
    lines.extend(render_bullets(minutes.key_decisions))
    lines.append("")
    lines.append("Action Items")
    lines.append("Owner | Task | Status | Deadline")
    lines.append("--- | --- | --- | ---")
    if minutes.action_items:
        for item in minutes.action_items:
            lines.append(f"{item.owner} | {item.task} | {item.status} | {item.deadline}")
    else:
        lines.append("- | - | - | -")
    if minutes.additional_information:
        lines.append("")
        lines.append("Additional Information")
        lines.extend(render_bullets(minutes.additional_information))
    if minutes.next_meeting:
        lines.append("")
        lines.append("Next Meeting")
        lines.extend(render_bullets(minutes.next_meeting))
    if minutes.warnings:
        lines.append("")
        lines.append("Warnings")
        lines.extend(render_bullets(minutes.warnings))
    return "\n".join(lines) + "\n"


def render_bullets(items: list[str]) -> list[str]:
    """Render a list as text bullets.

    Args:
        items: Section items.

    Returns:
        Bullet lines, or a placeholder bullet when empty.
    """

    if not items:
        return ["-"]
    return [f"- {item}" for item in items]


def parse_optional_int(value: Any) -> int | None:
    """Parse optional integer values from prediction rows.

    Args:
        value: Raw timestamp value.

    Returns:
        Parsed integer or ``None``.
    """

    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def parse_optional_float(value: Any) -> float:
    """Parse optional confidence values from prediction rows.

    Args:
        value: Raw confidence value.

    Returns:
        Parsed float, defaulting to ``0.0``.
    """

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def main() -> MeetingMinutes:
    """Run MoM generation from prediction output files.

    Returns:
        Structured ``MeetingMinutes`` object.
    """

    records, warnings = read_predictions()
    minutes = generate_minutes(records, warnings)
    save_minutes(minutes)
    print(render_minutes_text(minutes))
    return minutes


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    main()
