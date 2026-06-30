"""Lightweight dataset annotation tool for ML-based MoM training data.

Purpose:
    Convert parsed meeting transcript sentences into supervised training rows
    for a future ANN sentence classifier.

Responsibilities:
    - Load transcript text files through the existing transcript parser.
    - Accept already parsed ``TranscriptTurn`` objects for test and tooling use.
    - Present sentence records one at a time through session navigation methods.
    - Store human-selected labels, optional cluster IDs, and annotation notes.
    - Export labeled data to CSV and JSON for model training workflows.
    - Warn about common dataset quality issues without crashing.

Inputs:
    A transcript text file path or a list of parsed ``TranscriptTurn`` objects.

Outputs:
    Annotation records containing sentence text, speaker metadata, selected
    labels, optional cluster IDs, and notes.

Future Implementation Notes:
    This module is not part of the production app. It is intentionally kept as a
    small programmatic tool so a future GUI or Streamlit annotation interface can
    reuse the same session model.
"""

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import sys
import tempfile

try:
    from ml_mom.transcript_parser import TranscriptTurn, parse_transcript
except ModuleNotFoundError:  # pragma: no cover - supports direct script demos.
    from transcript_parser import TranscriptTurn, parse_transcript


SUPPORTED_LABELS = frozenset(
    {
        "Discussion",
        "Decision",
        "Action_Item",
        "Summary",
        "Information",
    }
)
INVALID_SPEAKER_VALUES = frozenset({"", "unknown", "none", "null"})


@dataclass(slots=True)
class AnnotationRecord:
    """Single sentence annotation row for supervised training.

    Attributes:
        sentence_id: Sequential sentence identifier inside the annotation
            session.
        turn_id: Source transcript turn identifier.
        speaker: Normalized speaker label from the parser.
        timestamp: Source turn timestamp in seconds, when available.
        sentence: Sentence text to be labeled.
        selected_label: Human-selected training label.
        cluster_id: Optional cluster identifier from a prior clustering pass.
        notes: Optional annotator notes for ambiguity or cleanup.
    """

    sentence_id: int
    turn_id: int
    speaker: str
    timestamp: int | None
    sentence: str
    selected_label: str | None = None
    cluster_id: int | None = None
    notes: str = ""


@dataclass(slots=True)
class AnnotationValidationResult:
    """Validation result for annotation dataset quality checks.

    Attributes:
        warnings: Non-blocking data quality warnings.
        is_valid: Whether no warnings were found.
    """

    warnings: list[str]
    is_valid: bool


class AnnotationSession:
    """Stateful annotation session for sentence-level MoM labels."""

    def __init__(self) -> None:
        """Initialize an empty annotation session."""

        self.records: list[AnnotationRecord] = []
        self.current_index = 0
        self.validation_warnings: list[str] = []

    def load_transcript(
        self,
        transcript_path: str | Path | None = None,
        turns: list[TranscriptTurn] | None = None,
    ) -> AnnotationValidationResult:
        """Load transcript data into the annotation session.

        Args:
            transcript_path: Optional path to a transcript text file.
            turns: Optional parsed transcript turns. When provided, these are
                used directly instead of reading a file.

        Returns:
            Validation result with non-crashing warnings.
        """

        self.records = []
        self.current_index = 0
        self.validation_warnings = []

        try:
            if turns is None:
                turns = self._load_turns_from_file(transcript_path)
        except Exception as exc:  # pragma: no cover - defensive file safety.
            self.validation_warnings.append(f"Transcript could not be loaded: {exc}")
            return self.validate_records()

        self.records = self._records_from_turns(turns or [])
        return self.validate_records()

    def next_sentence(self) -> AnnotationRecord | None:
        """Move to the next sentence in the session.

        Returns:
            Current ``AnnotationRecord`` after moving forward, or ``None`` when
            the session has no records.
        """

        if not self.records:
            return None

        # Clamping the index keeps navigation safe for CLI and future GUI usage.
        self.current_index = min(self.current_index + 1, len(self.records) - 1)
        return self.records[self.current_index]

    def previous_sentence(self) -> AnnotationRecord | None:
        """Move to the previous sentence in the session.

        Returns:
            Current ``AnnotationRecord`` after moving backward, or ``None`` when
            the session has no records.
        """

        if not self.records:
            return None

        self.current_index = max(self.current_index - 1, 0)
        return self.records[self.current_index]

    def assign_label(
        self,
        label: str,
        notes: str = "",
        cluster_id: int | None = None,
    ) -> AnnotationValidationResult:
        """Assign a label to the current sentence.

        Args:
            label: One of the supported MoM training labels.
            notes: Optional annotator notes.
            cluster_id: Optional cluster identifier for this sentence.

        Returns:
            Validation result after applying the label.
        """

        if not self.records:
            self.validation_warnings.append("No sentence is available for labeling.")
            return self.validate_records()

        if label not in SUPPORTED_LABELS:
            self.validation_warnings.append(f"Invalid label ignored: {label}")
            return self.validate_records()

        record = self.records[self.current_index]
        record.selected_label = label
        record.notes = notes
        record.cluster_id = cluster_id
        return self.validate_records()

    def export_csv(self, output_path: str | Path) -> AnnotationValidationResult:
        """Export annotation records to CSV.

        Args:
            output_path: Destination CSV path.

        Returns:
            Validation result after export attempt.
        """

        path = Path(output_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=list(asdict(self.records[0]).keys()))
                writer.writeheader()
                for record in self.records:
                    writer.writerow(asdict(record))
        except IndexError:
            self.validation_warnings.append("No annotation records are available for CSV export.")
        except Exception as exc:  # pragma: no cover - file-system dependent.
            self.validation_warnings.append(f"CSV export failed: {exc}")
        return self.validate_records()

    def export_json(self, output_path: str | Path) -> AnnotationValidationResult:
        """Export annotation records to JSON.

        Args:
            output_path: Destination JSON path.

        Returns:
            Validation result after export attempt.
        """

        path = Path(output_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as json_file:
                json.dump(
                    [asdict(record) for record in self.records],
                    json_file,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception as exc:  # pragma: no cover - file-system dependent.
            self.validation_warnings.append(f"JSON export failed: {exc}")
        return self.validate_records()

    def dataset_statistics(self) -> dict[str, int | dict[str, int]]:
        """Calculate annotation dataset statistics.

        Returns:
            Dictionary with total sentences, labeled count, unlabeled count,
            duplicate count, and per-label counts.
        """

        label_counts: dict[str, int] = {}
        for record in self.records:
            if record.selected_label:
                label_counts[record.selected_label] = label_counts.get(record.selected_label, 0) + 1

        duplicate_count = len(self.records) - len(
            {record.sentence.strip().lower() for record in self.records if record.sentence.strip()}
        )
        labeled_count = sum(1 for record in self.records if record.selected_label)

        return {
            "total_sentences": len(self.records),
            "labeled_sentences": labeled_count,
            "unlabeled_sentences": len(self.records) - labeled_count,
            "duplicate_sentences": max(0, duplicate_count),
            "label_counts": label_counts,
        }

    def validate_records(self) -> AnnotationValidationResult:
        """Validate annotation records for dataset quality warnings.

        Returns:
            Validation result containing warnings for unlabeled, duplicate,
            empty, or invalid-speaker rows.
        """

        warnings = list(self.validation_warnings)
        seen_sentences: set[str] = set()

        for record in self.records:
            normalized_sentence = record.sentence.strip().lower()
            if not record.sentence.strip():
                warnings.append(f"Sentence {record.sentence_id} is empty.")
            if not record.selected_label:
                warnings.append(f"Sentence {record.sentence_id} is unlabeled.")
            if normalized_sentence in seen_sentences:
                warnings.append(f"Duplicate sentence found at sentence {record.sentence_id}.")
            if normalized_sentence:
                seen_sentences.add(normalized_sentence)
            if record.speaker.strip().lower() in INVALID_SPEAKER_VALUES:
                warnings.append(f"Sentence {record.sentence_id} has an invalid speaker.")

        deduplicated = deduplicate_warnings(warnings)
        return AnnotationValidationResult(
            warnings=deduplicated,
            is_valid=not deduplicated,
        )

    def _load_turns_from_file(
        self,
        transcript_path: str | Path | None,
    ) -> list[TranscriptTurn]:
        """Load and parse transcript turns from a text file.

        Args:
            transcript_path: Path to a transcript text file.

        Returns:
            Parsed transcript turns.
        """

        if transcript_path is None:
            self.validation_warnings.append("No transcript path was provided.")
            return []

        path = Path(transcript_path)
        transcript_text = path.read_text(encoding="utf-8")
        parsed = parse_transcript(transcript_text)
        if not parsed.is_valid and parsed.validation_message:
            self.validation_warnings.append(parsed.validation_message)
        self.validation_warnings.extend(parsed.warnings)
        return parsed.turns

    def _records_from_turns(self, turns: list[TranscriptTurn]) -> list[AnnotationRecord]:
        """Create annotation records from parsed transcript turns.

        Args:
            turns: Parsed transcript turns.

        Returns:
            Sentence-level annotation records.
        """

        records: list[AnnotationRecord] = []
        for turn in turns:
            sentences = turn.sentence_list or ([turn.text] if turn.text else [])
            for sentence in sentences:
                # One record per sentence is required because the ANN classifier
                # will learn sentence-level labels, not turn-level labels.
                records.append(
                    AnnotationRecord(
                        sentence_id=len(records) + 1,
                        turn_id=turn.turn_id,
                        speaker=turn.speaker_normalized,
                        timestamp=turn.timestamp_seconds,
                        sentence=sentence.strip(),
                    )
                )
        return records


def deduplicate_warnings(warnings: list[str]) -> list[str]:
    """Deduplicate warnings while preserving order.

    Args:
        warnings: Warning messages to deduplicate.

    Returns:
        Ordered warning list without repeated messages.
    """

    deduplicated: list[str] = []
    for warning in warnings:
        if warning not in deduplicated:
            deduplicated.append(warning)
    return deduplicated


def annotation_demo() -> None:
    """Demonstrate loading, labeling, exporting, and statistics.

    Returns:
        None. Prints annotation progress, export paths, and dataset statistics.
    """

    demo_transcript = (
        "Rahul [00:10]\n"
        "We approved the budget today. I will prepare the launch report by Friday.\n\n"
        "Priya [00:35]\n"
        "Please review the proposal before tomorrow. The customer feedback was positive."
    )
    demo_dir = Path(tempfile.gettempdir()) / "meetscribe_annotation_demo"
    transcript_path = demo_dir / "demo_transcript.txt"
    csv_path = demo_dir / "annotations.csv"
    json_path = demo_dir / "annotations.json"
    demo_dir.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(demo_transcript, encoding="utf-8")

    session = AnnotationSession()
    load_validation = session.load_transcript(transcript_path=transcript_path)
    print("Loaded records:", len(session.records))
    if load_validation.warnings:
        print("Initial warnings:", load_validation.warnings)

    labels = ["Decision", "Action_Item", "Discussion", "Information"]
    for label in labels:
        current_record = session.records[session.current_index]
        print(f"Labeling sentence {current_record.sentence_id}: {current_record.sentence}")
        session.assign_label(label=label, notes=f"Demo label: {label}")
        session.next_sentence()

    session.export_csv(csv_path)
    session.export_json(json_path)

    print("CSV export:", csv_path)
    print("JSON export:", json_path)
    print("Statistics:", session.dataset_statistics())

    # TODO:
    # Add GUI controls for label selection, notes, and review navigation after
    # the dataset schema stabilizes.


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    annotation_demo()
