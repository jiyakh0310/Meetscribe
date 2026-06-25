"""Speaker resolution helpers for transcript display and analysis."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from transcription.speaker_mapping import (
    SpeakerMapping,
    apply_mapping_to_result,
    default_mapping_for_result,
    has_participant_names,
    named_mapping_for_result,
)

if TYPE_CHECKING:
    from transcription.sarvam_client import TranscriptionResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SpeakerResolution:
    """Resolved speaker mapping state for a transcript."""

    mapping: SpeakerMapping
    detected_speakers: list[str]
    names_available: bool
    review_required: bool
    reused_labels: list[str]


_GENERIC_SPEAKER_LINE_RE = re.compile(
    r"(?m)^\s*(?P<label>Speaker\s+\d+)(?=\s*(?:\[|:|\n))",
    re.IGNORECASE,
)


def detect_speaker_labels(result: TranscriptionResult) -> list[str]:
    """Return speaker labels detected in transcript order."""

    mapping = default_mapping_for_result(result)
    labels = list(mapping)
    if labels:
        logger.info("Speakers detected from segments: %s", labels)
        return labels

    labels = _ordered_unique(
        _normalize_generic_label(match.group("label"))
        for match in _GENERIC_SPEAKER_LINE_RE.finditer(result.transcript or "")
    )
    logger.info("Speakers detected from transcript text: %s", labels)
    return labels


def resolve_speakers(
    result: TranscriptionResult,
    *,
    previous_mapping: SpeakerMapping | None = None,
) -> SpeakerResolution:
    """Resolve automatic and reusable mappings for a transcript."""

    detected_speakers = detect_speaker_labels(result)
    automatic_mapping = named_mapping_for_result(result)
    names_available = has_participant_names(result)
    if automatic_mapping:
        logger.info("Automatic speaker mappings detected: %s", automatic_mapping)

    base_mapping = automatic_mapping or {label: label for label in detected_speakers}
    if not base_mapping:
        base_mapping = default_mapping_for_result(result)

    mapping, reused_labels = merge_saved_mapping(
        base_mapping,
        previous_mapping or {},
        detected_speakers=detected_speakers or list(base_mapping),
    )
    if reused_labels:
        logger.info("Speaker mapping reuse applied for labels: %s", reused_labels)

    saved_labels = {
        label
        for label in detected_speakers
        if label in (previous_mapping or {})
    }
    saved_mapping_available = bool(detected_speakers) and len(saved_labels) == len(
        detected_speakers
    )
    review_required = bool(
        detected_speakers and not names_available and not saved_mapping_available
    )
    logger.info(
        "Speaker resolution completed: detected=%s names_available=%s "
        "saved_mapping_available=%s review_required=%s",
        detected_speakers,
        names_available,
        saved_mapping_available,
        review_required,
    )
    return SpeakerResolution(
        mapping=mapping,
        detected_speakers=detected_speakers,
        names_available=names_available,
        review_required=review_required,
        reused_labels=reused_labels,
    )


def merge_saved_mapping(
    base_mapping: SpeakerMapping,
    previous_mapping: SpeakerMapping,
    *,
    detected_speakers: list[str],
) -> tuple[SpeakerMapping, list[str]]:
    """Reuse saved mappings for labels that still exist in the transcript."""

    mapping = dict(base_mapping)
    reused_labels: list[str] = []
    for label in detected_speakers:
        if label not in previous_mapping:
            continue
        saved_name = str(previous_mapping.get(label, "")).strip() or label
        if saved_name != mapping.get(label):
            mapping[label] = saved_name
        reused_labels.append(label)
    return mapping, reused_labels


def apply_speaker_resolution(
    result: TranscriptionResult,
    mapping: SpeakerMapping,
) -> TranscriptionResult:
    """Return a transcription result with speaker names applied."""

    return apply_mapping_to_result(result, mapping)


def update_mapping(
    labels: list[str],
    submitted_values: SpeakerMapping,
) -> SpeakerMapping:
    """Normalize user submitted speaker names while preserving empty labels."""

    mapping: SpeakerMapping = {}
    for label in labels:
        value = str(submitted_values.get(label, "")).strip()
        mapping[label] = value or label
    logger.info("Manual speaker mappings submitted: %s", mapping)
    return mapping


def _ordered_unique(values: Iterable[str]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for value in values:
        label = str(value).strip()
        if label and label.lower() not in seen:
            labels.append(label)
            seen.add(label.lower())
    return labels


def _normalize_generic_label(value: str) -> str:
    match = re.search(r"speaker\s+(\d+)", value, re.IGNORECASE)
    if not match:
        return re.sub(r"\s+", " ", value).strip()
    return f"Speaker {int(match.group(1))}"
