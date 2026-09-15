"""Canonical timeline representation -- SELaD Phase 0.

This module builds a read-only, additive view of a meeting's timed events
from the existing diarized transcript. It exists purely as a foundation for
future conversation-behaviour analytics (speaking speed, turn-taking,
keyword/topic timing, Voxels/video alignment, playback, ...) -- Phase 0
itself does not compute or expose any of those features.

Authoritative source (integration point)
-----------------------------------------
The richest timestamped data in the project already lives on
``transcription.sarvam_client.TranscriptionSegment`` (``start_time_seconds``
/ ``end_time_seconds``, float precision). This module builds directly from
that structure -- specifically from ``st.session_state.transcript_result``
as it exists *after* the user has completed transcript/speaker review in the
Streamlit workflow (``render_speaker_stage`` -> speaker-mapping submit ->
``st.session_state.transcript_result = mapped`` in ``app/main.py``).

By that point speaker names have already been resolved via
``transcription.speaker_mapping.apply_mapping_to_result`` -- the segment
list is the same one the deterministic MoM path (``format_transcript``) and
Voxels alignment (``align_voxels_windows_to_transcript``) both already treat
as authoritative, so the timeline never introduces a second, competing
transcript representation. It deliberately does NOT go through the
plain-text ``format_transcript`` / ``ml_mom.transcript_parser`` round trip,
which is the exact place the prior architecture audit found end-timestamps
get discarded.

Nothing here mutates a ``TranscriptionSegment``, calls into Sarvam, MiniLM,
the ANN, the deterministic formatter, Gemma, or Voxels, or writes anything
back to session state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from transcription.sarvam_client import TranscriptionResult, TranscriptionSegment
from transcription.speaker_mapping import (
    SpeakerMapping,
    default_speaker_label,
    display_speaker_label,
)


class TimingStatus(str, Enum):
    """How trustworthy an event's timing is, so callers never have to guess
    from field values alone whether a missing/odd duration means "no timing
    was ever available" versus "timing was present but malformed"."""

    TIMED = "timed"
    """start/end are both present and end >= start; duration_seconds is a
    reliable, non-negative float."""

    UNTIMED = "untimed"
    """start and/or end were not available (e.g. a plain-text transcript
    upload, or the Sarvam real-time/no-diarization path). No timestamp was
    fabricated; duration_seconds is None."""

    INVALID = "invalid"
    """start/end were both present but malformed (end earlier than start).
    The original values are kept on the event for inspection, but
    duration_seconds is None rather than a negative number."""


@dataclass(frozen=True, slots=True)
class TimelineEvent:
    """One timed (or explicitly untimed) unit of a meeting.

    Field names intentionally mirror ``TranscriptionSegment`` where they
    overlap, so this reads as an additive view over that data, not a
    competing shape.
    """

    event_id: int
    """Stable 0-based position of this event within the timeline. Equal to
    ``source_index`` for Phase 0 (one event per source segment); kept as a
    separate field so later phases can introduce events that do not map
    1:1 to a transcript segment (e.g. a merged/derived event) without
    disturbing this identifier's meaning."""

    speaker_id: str
    """Stable generic speaker key (e.g. ``"Speaker 1"``), independent of
    whatever display name a user has assigned. See
    ``transcription.speaker_mapping.default_speaker_label``."""

    speaker_name: str
    """Display name for the speaker if a mapping resolves one, otherwise the
    same value as ``speaker_id``. Never blank/None -- degrades to the
    generic label rather than omitting the speaker."""

    transcript: str
    """The segment's text, verbatim."""

    start_time_seconds: float | None
    end_time_seconds: float | None
    """Verbatim from the source segment, including when malformed (see
    ``timing_status``). Never fabricated: both are None together, or both
    are present together, matching how ``TranscriptionSegment`` itself is
    populated."""

    duration_seconds: float | None
    """``end_time_seconds - start_time_seconds`` when that is a reliable,
    non-negative value; otherwise None. Never negative."""

    timing_status: TimingStatus

    source_index: int
    """Index of the originating segment in the source
    ``TranscriptionResult.segments`` list, for tracing an event back to its
    transcript segment."""


@dataclass(frozen=True, slots=True)
class Timeline:
    """The full ordered timeline for one meeting, plus an explicit status so
    callers do not have to infer "no timing available" from an event list
    that merely happens to be empty or all-untimed."""

    events: tuple[TimelineEvent, ...]
    has_timed_events: bool
    """True if at least one event has ``timing_status == TimingStatus.TIMED``.
    False for transcript-only uploads without timestamps and for the
    short/segment-less Sarvam real-time path -- callers should treat that as
    "timing unavailable for this meeting", not as an error."""


def _classify_timing(
    start: float | None, end: float | None
) -> tuple[float | None, TimingStatus]:
    """Safely derive (duration, status) from a pair of timestamps.

    Never returns a negative duration and never fabricates a missing value.
    """

    if start is None or end is None:
        return None, TimingStatus.UNTIMED
    if end < start:
        return None, TimingStatus.INVALID
    return max(0.0, end - start), TimingStatus.TIMED


def _event_from_segment(
    index: int,
    segment: TranscriptionSegment,
    mapping: SpeakerMapping | None,
) -> TimelineEvent:
    duration, status = _classify_timing(
        segment.start_time_seconds, segment.end_time_seconds
    )
    return TimelineEvent(
        event_id=index,
        speaker_id=default_speaker_label(segment),
        speaker_name=display_speaker_label(segment, mapping),
        transcript=segment.transcript,
        start_time_seconds=segment.start_time_seconds,
        end_time_seconds=segment.end_time_seconds,
        duration_seconds=duration,
        timing_status=status,
        source_index=index,
    )


def build_timeline(
    result: TranscriptionResult | None,
    mapping: SpeakerMapping | None = None,
) -> Timeline:
    """Build the additive timeline view for one meeting.

    ``result`` should be the reviewed/mapped ``TranscriptionResult`` (see the
    module docstring for the exact integration point) -- but this function
    places no other requirement on it and never raises for missing/odd
    input:

    - ``result is None`` or ``result.segments`` is empty (e.g. the Sarvam
      real-time/segment-less path, or before a transcript exists yet)
      returns an empty, ``has_timed_events=False`` timeline.
    - Segments with missing or malformed timestamps become ``UNTIMED`` /
      ``INVALID`` events rather than being dropped or given fabricated
      timing -- every segment still gets exactly one event, in order.
    - The source ``TranscriptionSegment`` objects are only read, never
      mutated or replaced.
    """

    if result is None or not result.segments:
        return Timeline(events=(), has_timed_events=False)

    events = tuple(
        _event_from_segment(index, segment, mapping)
        for index, segment in enumerate(result.segments)
    )
    has_timed_events = any(event.timing_status is TimingStatus.TIMED for event in events)
    return Timeline(events=events, has_timed_events=has_timed_events)


def timed_events(events: Sequence[TimelineEvent]) -> tuple[TimelineEvent, ...]:
    """Convenience filter: only the events with reliable timing.

    Small, intentionally trivial helper -- later phases (speaking speed,
    turn-taking, Voxels/video alignment) will all need exactly this filter,
    so it is established once here rather than re-implemented per phase.
    """

    return tuple(event for event in events if event.timing_status is TimingStatus.TIMED)
