"""Merge annotation CSV files into one supervised training dataset.

Purpose:
    Combine all manually annotated transcript sentence CSV files from
    ``datasets/annotations`` into a single cleaned master dataset.

Responsibilities:
    - Read every annotation CSV in the annotation directory.
    - Validate required columns before using a file.
    - Skip corrupted or invalid files without crashing.
    - Remove duplicate rows, missing labels, and empty sentences.
    - Print detailed merge statistics for dataset review.
    - Save the merged dataset to ``datasets/processed/master_dataset.csv``.

Inputs:
    Annotation CSV files created by ``ml_mom.annotation_app``.

Outputs:
    A cleaned master CSV for later dataset preparation and ANN training.

Future Implementation Notes:
    This utility is dataset tooling only. It is not imported by the production
    Streamlit app, export flow, email flow, SMTP layer, or existing workflow.
"""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass, field
from pathlib import Path
import sys


ANNOTATION_DIR = Path("datasets") / "annotations"
PROCESSED_DIR = Path("datasets") / "processed"
MASTER_DATASET_PATH = PROCESSED_DIR / "master_dataset.csv"
REQUIRED_COLUMNS = [
    "sentence_id",
    "turn_id",
    "speaker",
    "timestamp",
    "sentence",
    "selected_label",
    "notes",
]


@dataclass(slots=True)
class MergeStatistics:
    """Statistics collected during annotation CSV merging.

    Attributes:
        total_files: Number of CSV files discovered.
        valid_files: Number of CSV files accepted for merging.
        skipped_files: Number of CSV files skipped because validation failed.
        total_rows: Number of rows read from valid CSV files before cleanup.
        rows_removed: Number of rows removed during cleanup.
        duplicate_rows_removed: Number of duplicate rows removed.
        missing_label_rows_removed: Number of rows removed for missing labels.
        empty_sentence_rows_removed: Number of rows removed for empty sentences.
        label_distribution: Count of remaining rows per label.
        speaker_count: Number of unique speakers in the cleaned dataset.
        average_sentence_length: Average cleaned sentence length in words.
        skipped_file_messages: Details about skipped invalid or corrupted files.
    """

    total_files: int = 0
    valid_files: int = 0
    skipped_files: int = 0
    total_rows: int = 0
    rows_removed: int = 0
    duplicate_rows_removed: int = 0
    missing_label_rows_removed: int = 0
    empty_sentence_rows_removed: int = 0
    label_distribution: dict[str, int] = field(default_factory=dict)
    speaker_count: int = 0
    average_sentence_length: float = 0.0
    skipped_file_messages: list[str] = field(default_factory=list)


def discover_annotation_files(annotation_dir: Path = ANNOTATION_DIR) -> list[Path]:
    """Discover annotation CSV files.

    Args:
        annotation_dir: Directory containing annotation CSV files.

    Returns:
        Sorted list of CSV paths. Returns an empty list when the directory does
        not exist.
    """

    if not annotation_dir.exists():
        return []
    return sorted(annotation_dir.glob("*.csv"))


def validate_csv_columns(path: Path, fieldnames: list[str] | None) -> list[str]:
    """Validate that a CSV contains all required annotation columns.

    Args:
        path: CSV file path.
        fieldnames: Header fields reported by ``csv.DictReader``.

    Returns:
        Validation messages. An empty list means the file is usable.
    """

    if not fieldnames:
        return [f"{path.name}: CSV has no header row."]

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    if missing_columns:
        return [
            f"{path.name}: missing required columns: {', '.join(missing_columns)}."
        ]
    return []


def read_annotation_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Read and validate one annotation CSV.

    Args:
        path: Annotation CSV path.

    Returns:
        Tuple of usable row dictionaries and validation messages. When messages
        are present, the caller should skip the file.
    """

    try:
        with path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            validation_messages = validate_csv_columns(path, reader.fieldnames)
            if validation_messages:
                return [], validation_messages

            rows = []
            for row in reader:
                # Keep only required columns so the master dataset schema stays
                # stable even if future annotation files add extra metadata.
                rows.append({column: (row.get(column) or "") for column in REQUIRED_COLUMNS})
            return rows, []
    except UnicodeDecodeError as exc:
        return [], [f"{path.name}: could not decode as UTF-8: {exc}."]
    except csv.Error as exc:
        return [], [f"{path.name}: corrupted CSV content: {exc}."]
    except OSError as exc:
        return [], [f"{path.name}: could not be read: {exc}."]


def row_deduplication_key(row: dict[str, str]) -> tuple[str, ...]:
    """Build a normalized key used to identify duplicate rows.

    Args:
        row: Annotation row.

    Returns:
        Tuple containing normalized row values.
    """

    # Duplicates are identified across the fields that define one annotation
    # sample, so repeated exports do not overweight the same sentence.
    return tuple((row.get(column) or "").strip().lower() for column in REQUIRED_COLUMNS)


def clean_rows(rows: list[dict[str, str]], stats: MergeStatistics) -> list[dict[str, str]]:
    """Clean merged annotation rows.

    Args:
        rows: Raw rows from valid annotation CSV files.
        stats: Statistics object updated with removal counts.

    Returns:
        Cleaned row dictionaries.
    """

    cleaned_rows: list[dict[str, str]] = []
    seen_keys: set[tuple[str, ...]] = set()

    for row in rows:
        sentence = (row.get("sentence") or "").strip()
        selected_label = (row.get("selected_label") or "").strip()

        if not sentence:
            stats.empty_sentence_rows_removed += 1
            continue
        if not selected_label:
            stats.missing_label_rows_removed += 1
            continue

        cleaned_row = {column: (row.get(column) or "").strip() for column in REQUIRED_COLUMNS}
        duplicate_key = row_deduplication_key(cleaned_row)
        if duplicate_key in seen_keys:
            stats.duplicate_rows_removed += 1
            continue

        seen_keys.add(duplicate_key)
        cleaned_rows.append(cleaned_row)

    stats.rows_removed = (
        stats.duplicate_rows_removed
        + stats.missing_label_rows_removed
        + stats.empty_sentence_rows_removed
    )
    return cleaned_rows


def update_output_statistics(rows: list[dict[str, str]], stats: MergeStatistics) -> None:
    """Populate label, speaker, and sentence-length statistics.

    Args:
        rows: Cleaned annotation rows.
        stats: Statistics object to update.
    """

    label_counter = Counter(row["selected_label"] for row in rows)
    speakers = {row["speaker"] for row in rows if row["speaker"]}
    sentence_lengths = [len(row["sentence"].split()) for row in rows if row["sentence"]]

    stats.label_distribution = dict(sorted(label_counter.items()))
    stats.speaker_count = len(speakers)
    stats.average_sentence_length = (
        round(sum(sentence_lengths) / len(sentence_lengths), 2)
        if sentence_lengths
        else 0.0
    )


def save_master_dataset(
    rows: list[dict[str, str]],
    output_path: Path = MASTER_DATASET_PATH,
) -> None:
    """Save cleaned rows to the master dataset CSV.

    Args:
        rows: Cleaned annotation rows.
        output_path: Destination master dataset path.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def merge_annotation_csvs(
    annotation_dir: Path = ANNOTATION_DIR,
    output_path: Path = MASTER_DATASET_PATH,
) -> tuple[list[dict[str, str]], MergeStatistics]:
    """Merge every valid annotation CSV into one cleaned dataset.

    Args:
        annotation_dir: Directory containing source annotation CSV files.
        output_path: Destination master dataset CSV path.

    Returns:
        Tuple containing cleaned rows and merge statistics.
    """

    stats = MergeStatistics()
    all_rows: list[dict[str, str]] = []
    csv_files = discover_annotation_files(annotation_dir)
    stats.total_files = len(csv_files)

    for csv_path in csv_files:
        rows, validation_messages = read_annotation_csv(csv_path)
        if validation_messages:
            # Bad files are skipped so one corrupted annotation export does not
            # block the entire dataset merge.
            stats.skipped_files += 1
            stats.skipped_file_messages.extend(validation_messages)
            continue

        stats.valid_files += 1
        stats.total_rows += len(rows)
        all_rows.extend(rows)

    cleaned_rows = clean_rows(all_rows, stats)
    update_output_statistics(cleaned_rows, stats)
    save_master_dataset(cleaned_rows, output_path)
    return cleaned_rows, stats


def print_statistics(stats: MergeStatistics, output_path: Path = MASTER_DATASET_PATH) -> None:
    """Print detailed merge statistics.

    Args:
        stats: Merge statistics to print.
        output_path: Master dataset output path.
    """

    print("Annotation Merge Statistics")
    print(f"Total files: {stats.total_files}")
    print(f"Valid files: {stats.valid_files}")
    print(f"Skipped files: {stats.skipped_files}")
    print(f"Total rows: {stats.total_rows}")
    print(f"Rows removed: {stats.rows_removed}")
    print(f"Duplicate rows removed: {stats.duplicate_rows_removed}")
    print(f"Missing-label rows removed: {stats.missing_label_rows_removed}")
    print(f"Empty-sentence rows removed: {stats.empty_sentence_rows_removed}")
    print(f"Label distribution: {stats.label_distribution}")
    print(f"Speaker count: {stats.speaker_count}")
    print(f"Average sentence length: {stats.average_sentence_length} words")
    print(f"Saved merged dataset: {output_path}")

    if stats.skipped_file_messages:
        print("Skipped file details:")
        for message in stats.skipped_file_messages:
            print(f"- {message}")


def main() -> None:
    """Run the annotation CSV merge demo from the command line."""

    rows, stats = merge_annotation_csvs()
    print_statistics(stats)
    print(f"Cleaned rows saved: {len(rows)}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    main()
