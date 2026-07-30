"""Audio-only normalization for diarized speech-to-text segments.

The transcript-upload workflow already provides coherent speaker turns and is
intentionally not routed through this module. Audio providers can emit one
spoken sentence as several adjacent diarization entries; coalescing only
timestamp-contiguous entries from the same speaker restores that sentence
context before the shared transcript review and deterministic MoM pipeline.
"""

from __future__ import annotations

from dataclasses import replace

from transcription.sarvam_client import TranscriptionResult, TranscriptionSegment


MAX_CONTIGUOUS_GAP_SECONDS = 1.5


def coalesce_contiguous_audio_segments(
    result: TranscriptionResult,
    *,
    max_gap_seconds: float = MAX_CONTIGUOUS_GAP_SECONDS,
) -> TranscriptionResult:
    """Merge adjacent audio fragments when speaker and timing evidence agree.

    Every transcript token is preserved verbatim apart from whitespace at the
    fragment boundary. Different speakers, missing timestamps, reversed timing,
    and pauses longer than ``max_gap_seconds`` remain separate.
    """

    if len(result.segments) < 2:
        return result

    coalesced: list[TranscriptionSegment] = []
    for segment in result.segments:
        if not coalesced or not _can_merge(
            coalesced[-1],
            segment,
            max_gap_seconds=max_gap_seconds,
        ):
            coalesced.append(segment)
            continue

        previous = coalesced[-1]
        coalesced[-1] = replace(
            previous,
            transcript=_join_fragment_text(previous.transcript, segment.transcript),
            end_time_seconds=segment.end_time_seconds,
        )

    if len(coalesced) == len(result.segments):
        return result
    return replace(result, segments=coalesced)


def _can_merge(
    previous: TranscriptionSegment,
    current: TranscriptionSegment,
    *,
    max_gap_seconds: float,
) -> bool:
    previous_speaker = str(previous.speaker_id or "").strip()
    current_speaker = str(current.speaker_id or "").strip()
    if not previous_speaker or previous_speaker != current_speaker:
        return False

    if previous.end_time_seconds is None or current.start_time_seconds is None:
        return False

    gap = current.start_time_seconds - previous.end_time_seconds
    return 0 <= gap <= max(0.0, max_gap_seconds)


def _join_fragment_text(previous: str, current: str) -> str:
    return " ".join(part for part in (previous.strip(), current.strip()) if part)
