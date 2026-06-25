"""Speaker-name detection and label mapping helpers."""

from __future__ import annotations

import re
from dataclasses import replace

from transcription.sarvam_client import TranscriptionResult, TranscriptionSegment


SpeakerMapping = dict[str, str]

_GENERIC_SPEAKER_RE = re.compile(r"^speaker\s+\S+$", re.IGNORECASE)
_TIMESTAMP_RE = re.compile(
    r"(?:\b\d{1,2}:\d{2}\s*(?:AM|PM)\b|\b\d{1,2}:\d{2}(?::\d{2})?\b)",
    re.IGNORECASE,
)
_INLINE_NAME_RE = re.compile(
    r"^\s*(?:\d{1,2}:\d{2}(?::\d{2})?\s+)?"
    r"(?P<label>[A-Z][A-Za-z0-9 .,'_-]{1,59}?)"
    r"(?:\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?)?"
    r"\s*:\s+(?P<rest>.+)$",
    re.IGNORECASE,
)

_NON_NAME_LABELS = {
    "action item",
    "action items",
    "agenda",
    "am",
    "decision",
    "decisions",
    "discussion",
    "discussion points",
    "key points",
    "meeting notes",
    "notes",
    "pm",
    "summary",
    "transcript",
}


def default_speaker_label(segment: TranscriptionSegment) -> str:
    """Return the stable generic label used for a transcription segment."""

    if segment.speaker_id is None:
        return "Speaker"

    speaker_id = str(segment.speaker_id).strip()
    try:
        return f"Speaker {int(speaker_id) + 1}"
    except ValueError:
        if speaker_id.lower().startswith("speaker"):
            return re.sub(r"\s+", " ", speaker_id).strip()
        return speaker_id


def display_speaker_label(
    segment: TranscriptionSegment,
    mapping: SpeakerMapping | None = None,
) -> str:
    """Return the mapped display label for a transcription segment."""

    fallback = default_speaker_label(segment)
    if not mapping:
        return fallback
    return mapping.get(fallback, fallback).strip() or fallback


def detected_speaker_labels(result: TranscriptionResult) -> list[str]:
    """Return unique speaker labels in transcript order."""

    labels: list[str] = []
    seen: set[str] = set()
    for segment in result.segments:
        label = default_speaker_label(segment)
        if label not in seen:
            labels.append(label)
            seen.add(label)
    return labels


def default_mapping_for_result(result: TranscriptionResult) -> SpeakerMapping:
    """Create an identity mapping for every speaker in a result."""

    return {label: label for label in detected_speaker_labels(result)}


def named_mapping_for_result(result: TranscriptionResult) -> SpeakerMapping:
    """Create an identity mapping when uploaded transcript segments already use names."""

    mapping: SpeakerMapping = {}
    for label in detected_speaker_labels(result):
        if is_participant_name(label):
            mapping[label] = label
    return mapping


def has_participant_names(result: TranscriptionResult) -> bool:
    """Return True when a parsed transcript appears to contain participant names."""

    return bool(named_mapping_for_result(result))


def apply_mapping_to_transcript_text(text: str, mapping: SpeakerMapping) -> str:
    """Replace generic speaker labels in transcript text with mapped display names."""

    updated = text
    for label in sorted(mapping, key=len, reverse=True):
        display_name = mapping[label].strip() or label
        if display_name == label:
            continue
        pattern = re.compile(rf"(?m)^(?P<label>{re.escape(label)})(?=\s*(?:\[|:|\n))")
        updated = pattern.sub(display_name, updated)
    return updated


def apply_mapping_to_result(
    result: TranscriptionResult,
    mapping: SpeakerMapping,
) -> TranscriptionResult:
    """Return a result whose transcript text and segment speaker ids use mapped labels."""

    if not mapping:
        return result

    segments = [
        replace(segment, speaker_id=display_speaker_label(segment, mapping))
        for segment in result.segments
    ]
    return replace(
        result,
        transcript=apply_mapping_to_transcript_text(result.transcript, mapping),
        segments=segments,
    )


def extract_named_speaker_segments(text: str) -> list[TranscriptionSegment]:
    """Parse common Meet/Zoom/Teams speaker-name transcript lines."""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    segments: list[TranscriptionSegment] = []
    current_label: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_label, current_lines
        transcript = " ".join(line.strip() for line in current_lines if line.strip()).strip()
        if current_label and transcript:
            segments.append(
                TranscriptionSegment(
                    transcript=transcript,
                    speaker_id=current_label,
                )
            )
        current_label = None
        current_lines = []

    for index, stripped in enumerate(lines):
        inline_match = _INLINE_NAME_RE.match(stripped)
        if inline_match and is_participant_name(inline_match.group("label")):
            flush()
            current_label = _clean_name(inline_match.group("label"))
            current_lines.append(inline_match.group("rest").strip())
            continue

        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        has_timestamp_evidence = bool(
            _TIMESTAMP_RE.search(stripped) or _TIMESTAMP_RE.fullmatch(next_line)
        )
        if has_timestamp_evidence and is_participant_name(stripped):
            flush()
            current_label = _clean_name(stripped)
            continue

        if _TIMESTAMP_RE.fullmatch(stripped):
            continue

        if current_label:
            current_lines.append(stripped)

    flush()
    return segments if _has_enough_named_evidence(segments) else []


def is_participant_name(value: str) -> bool:
    """Return True for plausible participant labels and False for generic headings."""

    label = _clean_name(value)
    if not label or len(label) > 60:
        return False
    if _GENERIC_SPEAKER_RE.match(label):
        return False
    if label.lower() in _NON_NAME_LABELS:
        return False
    if not re.search(r"[A-Za-z]", label):
        return False
    if any(char in label for char in "?!{}[]|/\\"):
        return False
    words = label.split()
    if len(words) > 5:
        return False
    alpha_words = re.findall(r"[A-Za-z]+", label)
    if any(len(word) > 1 and not word[0].isupper() for word in alpha_words):
        return False
    return bool(re.match(r"^[A-Za-z][A-Za-z0-9 .,'_-]*$", label))


def _clean_name(value: str) -> str:
    value = _TIMESTAMP_RE.sub("", value)
    value = value.strip().strip(":").strip()
    return re.sub(r"\s+", " ", value)


def _has_enough_named_evidence(segments: list[TranscriptionSegment]) -> bool:
    if not segments:
        return False

    names = {str(segment.speaker_id) for segment in segments if segment.speaker_id}
    if len(names) >= 2:
        return True
    return len(segments) >= 2
