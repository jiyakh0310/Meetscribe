"""Standalone Streamlit annotation app for ML MoM dataset creation.

Purpose:
    Provide a manual sentence-labeling interface for building supervised
    training data from real meeting transcripts.

Responsibilities:
    - Load transcript text files through the existing annotation backend.
    - Display one sentence at a time with speaker and timestamp context.
    - Save labels and notes to CSV after each annotation.
    - Resume progress when an annotation CSV already exists.
    - Show progress, completion counts, remaining counts, and export location.

Inputs:
    A transcript text file uploaded by the annotator.

Outputs:
    Annotation CSV files stored under ``datasets/annotations/`` with filename
    ``meeting_name_annotations.csv``.

Future Implementation Notes:
    This app is a dataset creation tool only. It is intentionally not imported
    by the production MeetScribe Streamlit app or existing workflow.
"""

from dataclasses import asdict
from pathlib import Path
import csv
import re
import tempfile

import streamlit as st

try:
    from ml_mom.annotation_tool import AnnotationRecord, AnnotationSession, SUPPORTED_LABELS
except ModuleNotFoundError:  # pragma: no cover - supports direct Streamlit execution.
    from annotation_tool import AnnotationRecord, AnnotationSession, SUPPORTED_LABELS


ANNOTATION_DIR = Path("datasets") / "annotations"
LABELS = ["Discussion", "Decision", "Action_Item", "Summary", "Information"]
CSV_COLUMNS = [
    "sentence_id",
    "turn_id",
    "speaker",
    "timestamp",
    "sentence",
    "selected_label",
    "notes",
]


def sanitize_meeting_name(filename: str) -> str:
    """Create a filesystem-safe meeting name from an uploaded filename.

    Args:
        filename: Original uploaded transcript filename.

    Returns:
        Safe meeting name without extension.
    """

    stem = Path(filename).stem or "meeting"
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
    return safe_name or "meeting"


def annotation_csv_path(meeting_name: str) -> Path:
    """Build the annotation CSV path for a meeting.

    Args:
        meeting_name: Sanitized meeting name.

    Returns:
        Path inside ``datasets/annotations``.
    """

    return ANNOTATION_DIR / f"{meeting_name}_annotations.csv"


def initialize_state() -> None:
    """Initialize Streamlit session state for annotation.

    Returns:
        None.
    """

    if "annotation_session" not in st.session_state:
        st.session_state.annotation_session = AnnotationSession()
    if "meeting_name" not in st.session_state:
        st.session_state.meeting_name = ""
    if "export_path" not in st.session_state:
        st.session_state.export_path = None
    if "loaded_transcript_name" not in st.session_state:
        st.session_state.loaded_transcript_name = ""
    if "status_message" not in st.session_state:
        st.session_state.status_message = ""


def load_uploaded_transcript(uploaded_file: object) -> None:
    """Load an uploaded transcript into the annotation session.

    Args:
        uploaded_file: Streamlit uploaded file object.

    Returns:
        None.
    """

    meeting_name = sanitize_meeting_name(uploaded_file.name)
    export_path = annotation_csv_path(meeting_name)
    transcript_text = uploaded_file.getvalue().decode("utf-8", errors="replace")

    # The backend expects a file path, so a temporary file lets us reuse parser
    # and annotation logic without duplicating it in the UI layer.
    with tempfile.NamedTemporaryFile(
        "w",
        delete=False,
        encoding="utf-8",
        suffix=".txt",
    ) as temp_file:
        temp_file.write(transcript_text)
        temp_path = Path(temp_file.name)

    session = AnnotationSession()
    validation = session.load_transcript(transcript_path=temp_path)
    if export_path.exists():
        apply_existing_annotations(session, export_path)
        st.session_state.status_message = "Existing annotation CSV loaded for resume."
    elif validation.warnings:
        st.session_state.status_message = "Transcript loaded with validation warnings."
    else:
        st.session_state.status_message = "Transcript loaded."

    st.session_state.annotation_session = session
    st.session_state.meeting_name = meeting_name
    st.session_state.export_path = export_path
    st.session_state.loaded_transcript_name = uploaded_file.name


def apply_existing_annotations(session: AnnotationSession, csv_path: Path) -> None:
    """Apply saved labels and notes from an existing annotation CSV.

    Args:
        session: Active annotation session.
        csv_path: Existing annotation CSV path.

    Returns:
        None.
    """

    saved_rows = read_existing_annotation_rows(csv_path)
    saved_by_sentence_id = {
        row.get("sentence_id", ""): row
        for row in saved_rows
        if row.get("sentence_id")
    }

    # Matching by sentence ID keeps resume behavior predictable as long as the
    # same transcript file is used for the same meeting name.
    for record in session.records:
        saved_row = saved_by_sentence_id.get(str(record.sentence_id))
        if not saved_row:
            continue
        saved_label = saved_row.get("selected_label", "")
        if saved_label in SUPPORTED_LABELS:
            record.selected_label = saved_label
        record.notes = saved_row.get("notes", "")


def read_existing_annotation_rows(csv_path: Path) -> list[dict[str, str]]:
    """Read an existing annotation CSV safely.

    Args:
        csv_path: Annotation CSV path.

    Returns:
        Existing rows, or an empty list when the file cannot be read.
    """

    try:
        with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
            return [dict(row) for row in csv.DictReader(csv_file)]
    except Exception:
        return []


def autosave_annotations() -> None:
    """Write current annotation records to the configured CSV path.

    Returns:
        None.
    """

    session = st.session_state.annotation_session
    export_path = st.session_state.export_path
    if export_path is None or not session.records:
        return

    export_path.parent.mkdir(parents=True, exist_ok=True)
    with export_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in session.records:
            row = asdict(record)
            writer.writerow({column: row.get(column) for column in CSV_COLUMNS})


def current_record() -> AnnotationRecord | None:
    """Return the active annotation record.

    Returns:
        Current ``AnnotationRecord`` or ``None`` when no transcript is loaded.
    """

    session = st.session_state.annotation_session
    if not session.records:
        return None
    return session.records[session.current_index]


def save_current_label(label: str, notes: str) -> None:
    """Save the selected label and notes for the current sentence.

    Args:
        label: Selected annotation label.
        notes: Optional annotation notes.

    Returns:
        None.
    """

    session = st.session_state.annotation_session
    session.assign_label(label=label, notes=notes)
    autosave_annotations()
    st.session_state.status_message = "Label saved and annotation CSV updated."


def move_previous() -> None:
    """Move the active sentence pointer backward."""

    st.session_state.annotation_session.previous_sentence()


def move_next() -> None:
    """Move the active sentence pointer forward."""

    st.session_state.annotation_session.next_sentence()


def render_sidebar() -> None:
    """Render annotation progress and export metadata in the sidebar."""

    session = st.session_state.annotation_session
    stats = session.dataset_statistics()
    total = stats["total_sentences"]
    completed = stats["labeled_sentences"]
    remaining = stats["unlabeled_sentences"]

    st.sidebar.header("Annotation")
    st.sidebar.write("Current transcript")
    st.sidebar.code(st.session_state.loaded_transcript_name or "Not loaded")
    st.sidebar.metric("Total sentences", total)
    st.sidebar.metric("Completed", completed)
    st.sidebar.metric("Remaining", remaining)
    st.sidebar.write("Export location")
    st.sidebar.code(str(st.session_state.export_path or ANNOTATION_DIR))


def render_progress() -> None:
    """Render progress bar and percentage."""

    session = st.session_state.annotation_session
    stats = session.dataset_statistics()
    total = stats["total_sentences"]
    completed = stats["labeled_sentences"]
    progress = completed / total if total else 0
    st.progress(progress)
    st.caption(f"{progress:.0%} complete")


def render_annotation_controls(record: AnnotationRecord) -> None:
    """Render the one-sentence annotation controls.

    Args:
        record: Active sentence annotation record.

    Returns:
        None.
    """

    session = st.session_state.annotation_session
    label_key = f"label_{record.sentence_id}"
    notes_key = f"notes_{record.sentence_id}"
    current_label = record.selected_label if record.selected_label in LABELS else LABELS[0]

    st.subheader(f"Sentence {record.sentence_id} of {len(session.records)}")
    meta_cols = st.columns(3)
    meta_cols[0].metric("Speaker", record.speaker or "Unknown")
    meta_cols[1].metric("Timestamp", record.timestamp if record.timestamp is not None else "N/A")
    meta_cols[2].metric("Turn", record.turn_id)

    st.text_area(
        "Sentence",
        value=record.sentence,
        height=120,
        disabled=True,
    )
    selected_label = st.radio(
        "Label",
        LABELS,
        index=LABELS.index(current_label),
        key=label_key,
        horizontal=True,
    )
    notes = st.text_area(
        "Notes",
        value=record.notes,
        key=notes_key,
        placeholder="Optional annotation notes",
    )

    button_cols = st.columns(3)
    with button_cols[0]:
        st.button("Previous", on_click=move_previous, disabled=session.current_index == 0)
    with button_cols[1]:
        st.button("Save Label", type="primary", on_click=save_current_label, args=(selected_label, notes))
    with button_cols[2]:
        st.button(
            "Next",
            on_click=move_next,
            disabled=session.current_index >= len(session.records) - 1,
        )

    st.caption("Keyboard shortcuts: use Tab/Shift+Tab to move between controls and Enter/Space to activate focused buttons.")


def main() -> None:
    """Run the standalone Streamlit annotation application."""

    st.set_page_config(
        page_title="MeetScribe Annotation Tool",
        layout="wide",
    )
    initialize_state()

    st.title("MeetScribe Annotation Tool")
    st.caption("Standalone dataset creation app. Not connected to production MeetScribe.")

    uploaded_file = st.file_uploader(
        "Select transcript",
        type=["txt", "md"],
        accept_multiple_files=False,
    )
    if uploaded_file is not None and uploaded_file.name != st.session_state.loaded_transcript_name:
        load_uploaded_transcript(uploaded_file)

    render_sidebar()

    if st.session_state.status_message:
        st.info(st.session_state.status_message)

    record = current_record()
    if record is None:
        st.warning("Upload a speaker-labeled transcript to start annotation.")
        st.code("streamlit run ml_mom/annotation_app.py")
        return

    render_progress()
    render_annotation_controls(record)


if __name__ == "__main__":
    main()
