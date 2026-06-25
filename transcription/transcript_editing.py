"""Helpers for applying user transcript edits before analysis."""

from __future__ import annotations

import logging
import re
from dataclasses import replace

from transcription.sarvam_client import TranscriptionResult, TranscriptionSegment

logger = logging.getLogger(__name__)


_SEGMENT_HEADER_RE = re.compile(
    r"^\s*(?P<label>[A-Za-z][A-Za-z0-9 .,'_-]{0,59}?)"
    r"(?:\s*\[(?P<start>[^\]-]+)\s*-\s*(?P<end>[^\]]+)\])?"
    r"\s*:?\s*$"
)
_INLINE_SEGMENT_RE = re.compile(
    r"^\s*(?P<label>[A-Za-z][A-Za-z0-9 .,'_-]{0,59}?)"
    r"(?:\s*\[(?P<start>[^\]-]+)\s*-\s*(?P<end>[^\]]+)\])?"
    r"\s*:\s+(?P<rest>.+)$"
)


def apply_transcript_edits(
    result: TranscriptionResult,
    edited_transcript: str,
) -> TranscriptionResult:
    """Return a transcription result updated with edited transcript text."""

    transcript = _normalize_transcript(edited_transcript)
    segments = parse_transcript_segments(transcript)
    if not segments and result.segments:
        logger.info(
            "Transcript edits did not parse into segments; preserving existing segments"
        )
        segments = result.segments

    logger.info(
        "Transcript edits applied: chars=%d parsed_segments=%d",
        len(transcript),
        len(segments),
    )
    return replace(result, transcript=transcript, segments=segments)


def parse_transcript_segments(text: str) -> list[TranscriptionSegment]:
    """Parse speaker/timestamp blocks from editable transcript text."""

    segments: list[TranscriptionSegment] = []
    current_label: str | None = None
    current_start: float | None = None
    current_end: float | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_label, current_start, current_end, current_lines
        body = " ".join(line.strip() for line in current_lines if line.strip()).strip()
        if current_label and body:
            segments.append(
                TranscriptionSegment(
                    transcript=body,
                    speaker_id=current_label,
                    start_time_seconds=current_start,
                    end_time_seconds=current_end,
                )
            )
        current_label = None
        current_start = None
        current_end = None
        current_lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        inline_match = _INLINE_SEGMENT_RE.match(line)
        if inline_match:
            flush()
            current_label = _clean_label(inline_match.group("label"))
            current_start = _parse_timestamp(inline_match.group("start"))
            current_end = _parse_timestamp(inline_match.group("end"))
            current_lines.append(inline_match.group("rest").strip())
            continue

        header_match = _SEGMENT_HEADER_RE.match(line)
        if header_match and (
            header_match.group("start") is not None
            or line.lower().startswith("speaker ")
            or not current_label
        ):
            flush()
            current_label = _clean_label(header_match.group("label"))
            current_start = _parse_timestamp(header_match.group("start"))
            current_end = _parse_timestamp(header_match.group("end"))
            continue

        if current_label:
            current_lines.append(line)

    flush()
    return segments


def speaker_mapping_from_segments(result: TranscriptionResult) -> dict[str, str]:
    """Build an identity mapping for the currently parsed speaker labels."""

    mapping: dict[str, str] = {}
    for segment in result.segments:
        label = str(segment.speaker_id or "").strip()
        if label and label not in mapping:
            mapping[label] = label
    return mapping


def _normalize_transcript(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _clean_label(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().strip(":").strip()


def _parse_timestamp(value: str | None) -> float | None:
    if not value:
        return None
    value = value.strip()
    parts = value.split(":")
    try:
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        return float(value)
    except ValueError:
        return None
