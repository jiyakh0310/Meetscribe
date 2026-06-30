"""Prepare annotated MoM sentence datasets for future ANN training.

Purpose:
    Load CSV files exported by ``ml_mom.annotation_tool`` and prepare clean,
    label-encoded train/validation/test splits for a future ANN classifier.

Responsibilities:
    - Load annotated CSV files safely.
    - Validate required annotation columns.
    - Remove duplicate rows and rows without labels.
    - Encode supported labels into stable numeric class IDs.
    - Split records into training, validation, and testing sets.
    - Save processed datasets for later model-training workflows.
    - Report friendly validation messages instead of crashing.

Inputs:
    Annotated CSV exported by ``annotation_tool.py``.

Outputs:
    ``PreparedTrainingDataset`` containing cleaned rows, encoded labels,
    train/validation/test splits, statistics, and validation messages.

Future Implementation Notes:
    This module does not train a model. Data augmentation, class balancing, and
    feature-vector assembly should be added only after dataset quality is
    reviewed.
"""

from dataclasses import asdict, dataclass, field
import csv
import json
from pathlib import Path
import random
import sys
import tempfile
from typing import Any


SUPPORTED_LABELS = (
    "Discussion",
    "Decision",
    "Action_Item",
    "Summary",
    "Information",
)
LABEL_TO_ID = {label: index for index, label in enumerate(SUPPORTED_LABELS)}
REQUIRED_COLUMNS = frozenset(
    {
        "sentence_id",
        "turn_id",
        "speaker",
        "timestamp",
        "sentence",
        "selected_label",
        "cluster_id",
        "notes",
    }
)
DEFAULT_SPLIT_RATIOS = (0.70, 0.15, 0.15)


@dataclass(slots=True)
class TrainingSample:
    """Cleaned and label-encoded training sample.

    Attributes:
        sentence_id: Source annotation sentence identifier.
        turn_id: Source transcript turn identifier.
        speaker: Normalized speaker label.
        timestamp: Source timestamp in seconds, when available.
        sentence: Annotated sentence text.
        selected_label: Human-selected label.
        label_id: Numeric class ID for the label.
        cluster_id: Optional cluster ID from annotation data.
        notes: Optional annotator notes.
    """

    sentence_id: int
    turn_id: int
    speaker: str
    timestamp: int | None
    sentence: str
    selected_label: str
    label_id: int
    cluster_id: int | None = None
    notes: str = ""


@dataclass(slots=True)
class PreparedTrainingDataset:
    """Prepared dataset container for future ANN training.

    Attributes:
        samples: Cleaned and label-encoded samples.
        train_samples: Training split.
        validation_samples: Validation split.
        test_samples: Testing split.
        label_mapping: Mapping from label names to numeric class IDs.
        statistics: Dataset statistics for audit and review.
        validation_messages: Friendly warnings or errors collected while
            preparing the dataset.
    """

    samples: list[TrainingSample] = field(default_factory=list)
    train_samples: list[TrainingSample] = field(default_factory=list)
    validation_samples: list[TrainingSample] = field(default_factory=list)
    test_samples: list[TrainingSample] = field(default_factory=list)
    label_mapping: dict[str, int] = field(default_factory=lambda: dict(LABEL_TO_ID))
    statistics: dict[str, Any] = field(default_factory=dict)
    validation_messages: list[str] = field(default_factory=list)


class TrainingDatasetManager:
    """Manager for loading, validating, cleaning, encoding, and splitting data."""

    def __init__(
        self,
        train_ratio: float = DEFAULT_SPLIT_RATIOS[0],
        validation_ratio: float = DEFAULT_SPLIT_RATIOS[1],
        test_ratio: float = DEFAULT_SPLIT_RATIOS[2],
        random_seed: int = 42,
    ) -> None:
        """Initialize the dataset manager.

        Args:
            train_ratio: Fraction of samples assigned to training.
            validation_ratio: Fraction of samples assigned to validation.
            test_ratio: Fraction of samples assigned to testing.
            random_seed: Deterministic seed for reproducible split shuffling.
        """

        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.raw_rows: list[dict[str, str]] = []
        self.cleaned_rows: list[dict[str, str]] = []
        self.samples: list[TrainingSample] = []
        self.train_samples: list[TrainingSample] = []
        self.validation_samples: list[TrainingSample] = []
        self.test_samples: list[TrainingSample] = []
        self.validation_messages: list[str] = []
        self.duplicate_count = 0
        self.missing_label_count = 0
        self.invalid_label_count = 0

    def load_dataset(self, csv_path: str | Path) -> PreparedTrainingDataset:
        """Load an annotated CSV file safely.

        Args:
            csv_path: Path to the CSV exported by ``annotation_tool.py``.

        Returns:
            Prepared dataset with raw rows loaded or validation messages when
            loading fails.
        """

        self._reset_state()
        path = Path(csv_path)

        try:
            with path.open("r", newline="", encoding="utf-8") as csv_file:
                reader = csv.DictReader(csv_file)
                self.raw_rows = [dict(row) for row in reader]
        except FileNotFoundError:
            self.validation_messages.append(f"Annotated CSV was not found: {path}")
        except UnicodeDecodeError:
            self.validation_messages.append(
                f"Annotated CSV could not be decoded as UTF-8: {path}"
            )
        except csv.Error as exc:
            self.validation_messages.append(f"Annotated CSV is corrupted: {exc}")
        except Exception as exc:  # pragma: no cover - defensive file safety.
            self.validation_messages.append(f"Annotated CSV could not be loaded: {exc}")

        return self._prepared_dataset()

    def validate_dataset(self) -> list[str]:
        """Validate loaded raw rows before cleaning and encoding.

        Returns:
            Friendly validation messages. An empty list means no blocking issue
            was found at this stage.
        """

        messages: list[str] = []
        if not self.raw_rows:
            messages.append("Dataset is empty or could not be loaded.")
            self.validation_messages = deduplicate_messages(
                self.validation_messages + messages
            )
            return self.validation_messages

        missing_columns = REQUIRED_COLUMNS.difference(self.raw_rows[0].keys())
        if missing_columns:
            messages.append(
                "Dataset is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        # Invalid labels are reported before cleaning so annotators can fix the
        # source dataset instead of silently losing rows.
        invalid_labels = sorted(
            {
                row.get("selected_label", "").strip()
                for row in self.raw_rows
                if row.get("selected_label", "").strip()
                and row.get("selected_label", "").strip() not in LABEL_TO_ID
            }
        )
        if invalid_labels:
            messages.append("Dataset contains invalid labels: " + ", ".join(invalid_labels))

        self.validation_messages = deduplicate_messages(
            self.validation_messages + messages
        )
        return self.validation_messages

    def clean_dataset(self) -> list[dict[str, str]]:
        """Remove duplicate rows and rows without labels.

        Returns:
            Cleaned raw row dictionaries.
        """

        self.cleaned_rows = []
        seen_keys: set[tuple[str, str, str]] = set()
        self.duplicate_count = 0
        self.missing_label_count = 0
        self.invalid_label_count = 0

        for row in self.raw_rows:
            label = row.get("selected_label", "").strip()
            sentence = row.get("sentence", "").strip()
            key = (
                row.get("turn_id", "").strip(),
                row.get("speaker", "").strip().lower(),
                sentence.lower(),
            )

            if not label:
                # Rows without labels cannot supervise an ANN class target.
                self.missing_label_count += 1
                continue
            if label not in LABEL_TO_ID:
                self.invalid_label_count += 1
                continue
            if key in seen_keys:
                # Duplicate rows can bias class counts and model gradients.
                self.duplicate_count += 1
                continue
            if not sentence:
                self.validation_messages.append("Empty sentence row removed during cleaning.")
                continue

            seen_keys.add(key)
            self.cleaned_rows.append(row)

        return self.cleaned_rows

    def encode_labels(self) -> list[TrainingSample]:
        """Encode cleaned labels into numeric class IDs.

        Returns:
            Training samples with stable numeric labels.
        """

        self.samples = []
        for row in self.cleaned_rows:
            label = row["selected_label"].strip()
            self.samples.append(
                TrainingSample(
                    sentence_id=parse_optional_int(row.get("sentence_id")) or len(self.samples) + 1,
                    turn_id=parse_optional_int(row.get("turn_id")) or 0,
                    speaker=row.get("speaker", "").strip(),
                    timestamp=parse_optional_int(row.get("timestamp")),
                    sentence=row.get("sentence", "").strip(),
                    selected_label=label,
                    label_id=LABEL_TO_ID[label],
                    cluster_id=parse_optional_int(row.get("cluster_id")),
                    notes=row.get("notes", "").strip(),
                )
            )

        # TODO:
        # Add data augmentation hooks after class imbalance is measured on real
        # annotated meeting datasets.
        return self.samples

    def split_dataset(self) -> tuple[list[TrainingSample], list[TrainingSample], list[TrainingSample]]:
        """Split encoded samples into train, validation, and test sets.

        Returns:
            Tuple of training, validation, and testing sample lists.
        """

        ratio_error = validate_split_ratios(
            self.train_ratio,
            self.validation_ratio,
            self.test_ratio,
        )
        if ratio_error:
            self.validation_messages.append(ratio_error)
            return [], [], []

        samples = list(self.samples)
        random.Random(self.random_seed).shuffle(samples)
        total = len(samples)
        train_count, validation_count, _ = calculate_split_counts(
            total,
            self.train_ratio,
            self.validation_ratio,
            self.test_ratio,
        )
        train_end = train_count
        validation_end = train_end + validation_count

        # Small datasets can round oddly, so the test split receives the
        # remainder to guarantee every sample appears in exactly one split.
        self.train_samples = samples[:train_end]
        self.validation_samples = samples[train_end:validation_end]
        self.test_samples = samples[validation_end:]
        return self.train_samples, self.validation_samples, self.test_samples

    def dataset_statistics(self) -> dict[str, Any]:
        """Generate dataset statistics for audit and training readiness.

        Returns:
            Dictionary containing sample counts, class counts, split sizes,
            duplicate count, and missing-label count.
        """

        class_counts: dict[str, int] = {}
        for sample in self.samples:
            class_counts[sample.selected_label] = class_counts.get(sample.selected_label, 0) + 1

        return {
            "total_samples": len(self.samples),
            "samples_per_class": class_counts,
            "training_samples": len(self.train_samples),
            "validation_samples": len(self.validation_samples),
            "testing_samples": len(self.test_samples),
            "duplicate_count": self.duplicate_count,
            "missing_labels": self.missing_label_count,
            "invalid_labels": self.invalid_label_count,
        }

    def save_processed_dataset(self, output_dir: str | Path) -> PreparedTrainingDataset:
        """Save processed splits and metadata to JSON files.

        Args:
            output_dir: Directory where processed dataset files should be saved.

        Returns:
            Prepared dataset with validation messages updated if saving fails.
        """

        output_path = Path(output_dir)
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            write_json(output_path / "train.json", self.train_samples)
            write_json(output_path / "validation.json", self.validation_samples)
            write_json(output_path / "test.json", self.test_samples)
            metadata = {
                "label_mapping": LABEL_TO_ID,
                "statistics": self.dataset_statistics(),
            }
            (output_path / "metadata.json").write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:  # pragma: no cover - file-system dependent.
            self.validation_messages.append(f"Processed dataset could not be saved: {exc}")

        return self._prepared_dataset()

    def prepare_from_csv(self, csv_path: str | Path) -> PreparedTrainingDataset:
        """Run the complete preparation workflow for one annotated CSV.

        Args:
            csv_path: Path to annotated CSV.

        Returns:
            Prepared training dataset.
        """

        self.load_dataset(csv_path)
        self.validate_dataset()
        self.clean_dataset()
        self.encode_labels()
        self.split_dataset()
        return self._prepared_dataset()

    def _prepared_dataset(self) -> PreparedTrainingDataset:
        """Build the current prepared dataset snapshot.

        Returns:
            Prepared dataset container.
        """

        return PreparedTrainingDataset(
            samples=list(self.samples),
            train_samples=list(self.train_samples),
            validation_samples=list(self.validation_samples),
            test_samples=list(self.test_samples),
            label_mapping=dict(LABEL_TO_ID),
            statistics=self.dataset_statistics(),
            validation_messages=deduplicate_messages(self.validation_messages),
        )

    def _reset_state(self) -> None:
        """Reset manager state before loading a new dataset."""

        self.raw_rows = []
        self.cleaned_rows = []
        self.samples = []
        self.train_samples = []
        self.validation_samples = []
        self.test_samples = []
        self.validation_messages = []
        self.duplicate_count = 0
        self.missing_label_count = 0
        self.invalid_label_count = 0


def validate_split_ratios(
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> str | None:
    """Validate split ratios before creating dataset splits.

    Args:
        train_ratio: Training split ratio.
        validation_ratio: Validation split ratio.
        test_ratio: Testing split ratio.

    Returns:
        Friendly validation message, or ``None`` when ratios are usable.
    """

    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(ratio < 0 for ratio in ratios):
        return "Split ratios must be non-negative."
    if round(sum(ratios), 6) != 1.0:
        return "Split ratios must add up to 1.0."
    return None


def calculate_split_counts(
    total_samples: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> tuple[int, int, int]:
    """Calculate split counts while preserving small non-empty splits.

    Args:
        total_samples: Number of encoded samples.
        train_ratio: Training split ratio.
        validation_ratio: Validation split ratio.
        test_ratio: Testing split ratio.

    Returns:
        Tuple containing training, validation, and testing counts.
    """

    if total_samples <= 0:
        return 0, 0, 0

    raw_counts = [
        total_samples * train_ratio,
        total_samples * validation_ratio,
        total_samples * test_ratio,
    ]
    counts = [int(count) for count in raw_counts]
    remainder = total_samples - sum(counts)
    fractional_order = sorted(
        range(3),
        key=lambda index: raw_counts[index] - counts[index],
        reverse=True,
    )

    for index in fractional_order[:remainder]:
        counts[index] += 1

    positive_ratio_indices = [
        index
        for index, ratio in enumerate((train_ratio, validation_ratio, test_ratio))
        if ratio > 0
    ]
    if total_samples >= len(positive_ratio_indices):
        for index in positive_ratio_indices:
            if counts[index] == 0:
                donor_index = max(range(3), key=lambda item: counts[item])
                if counts[donor_index] > 1:
                    counts[donor_index] -= 1
                    counts[index] += 1

    return counts[0], counts[1], counts[2]


def parse_optional_int(value: str | None) -> int | None:
    """Parse an optional integer value from CSV text.

    Args:
        value: CSV cell value.

    Returns:
        Parsed integer, or ``None`` when the value is empty or invalid.
    """

    if value is None or str(value).strip() == "":
        return None
    try:
        return int(float(str(value).strip()))
    except ValueError:
        return None


def write_json(path: Path, samples: list[TrainingSample]) -> None:
    """Write training samples to a JSON file.

    Args:
        path: Output file path.
        samples: Samples to serialize.
    """

    path.write_text(
        json.dumps([asdict(sample) for sample in samples], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def deduplicate_messages(messages: list[str]) -> list[str]:
    """Deduplicate validation messages while preserving order.

    Args:
        messages: Validation messages.

    Returns:
        Ordered list without duplicates.
    """

    deduplicated: list[str] = []
    for message in messages:
        if message not in deduplicated:
            deduplicated.append(message)
    return deduplicated


def training_dataset_demo() -> None:
    """Demonstrate loading and preparing an annotated CSV dataset.

    Returns:
        None. Prints dataset statistics, label mapping, and split sizes.
    """

    demo_dir = Path(tempfile.gettempdir()) / "meetscribe_training_dataset_demo"
    demo_csv = demo_dir / "annotated_dataset.csv"
    processed_dir = demo_dir / "processed"
    demo_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        ["1", "1", "Rahul", "10", "We approved the budget today.", "Decision", "", ""],
        ["2", "1", "Rahul", "10", "I will prepare the report.", "Action_Item", "", ""],
        ["3", "2", "Priya", "25", "Please review the proposal.", "Discussion", "", ""],
        ["4", "2", "Priya", "25", "The customer feedback was positive.", "Information", "", ""],
        ["5", "3", "Speaker A", "", "This is the overall summary.", "Summary", "", ""],
        ["5", "3", "Speaker A", "", "This is the overall summary.", "Summary", "", "duplicate"],
        ["6", "4", "Speaker B", "", "This row is missing a label.", "", "", ""],
    ]
    with demo_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(sorted(REQUIRED_COLUMNS))
        column_order = sorted(REQUIRED_COLUMNS)
        row_map_keys = [
            "sentence_id",
            "turn_id",
            "speaker",
            "timestamp",
            "sentence",
            "selected_label",
            "cluster_id",
            "notes",
        ]
        for row in rows:
            row_map = dict(zip(row_map_keys, row))
            writer.writerow([row_map[column] for column in column_order])

    manager = TrainingDatasetManager()
    prepared = manager.prepare_from_csv(demo_csv)
    manager.save_processed_dataset(processed_dir)

    print("Dataset statistics")
    print(prepared.statistics)
    print("Label mapping")
    print(prepared.label_mapping)
    print("Train/Validation/Test sizes")
    print(
        len(prepared.train_samples),
        len(prepared.validation_samples),
        len(prepared.test_samples),
    )
    if prepared.validation_messages:
        print("Validation messages")
        print(prepared.validation_messages)

    # TODO:
    # Add class balancing and data augmentation once real annotation volume and
    # class imbalance are measured.


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    training_dataset_demo()
