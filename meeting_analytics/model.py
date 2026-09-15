"""SELaD Phase 1 -- deterministic conversation-behaviour analytics.

Consumes only the Phase 0 ``timeline.Timeline`` (already built from the
reviewed transcript). This module never calls Sarvam, MiniLM, the ANN, the
deterministic MoM formatter, Gemma, or Voxels, and never imports from
``ml_mom`` -- the dependency direction is strictly one-way:

    Timeline -> meeting_analytics -> UI

Nothing here can influence ANN predictions, Decision/Action-Item
extraction, owners/deadlines, the deterministic formatter, Gemma's input or
output, or Voxels. Given the same ``Timeline``, ``analyze_conversation``
always returns the same result (pure function, no I/O, no randomness).

This module intentionally reimplements two trivial helpers (word counting,
question counting) rather than importing them from ``ml_mom`` -- both are
one-line rules, and importing from the factual pipeline package would create
exactly the coupling the one-way dependency direction above is meant to
avoid. See ``_word_count``/``_question_count`` for the exact, documented
rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from timeline import Timeline, TimelineEvent, TimingStatus


@dataclass(frozen=True, slots=True)
class SpeakerAnalytics:
    """Conversation-behaviour metrics for one speaker."""

    speaker_id: str
    """Stable generic speaker key -- see ``timeline.TimelineEvent.speaker_id``."""

    speaker_name: str
    """Reviewed display name (or the generic label if unmapped)."""

    total_event_count: int
    """Number of timeline events attributed to this speaker, timed or not."""

    timed_event_count: int
    """Of those, how many had reliable (``TimingStatus.TIMED``) timing."""

    speaking_duration_seconds: float | None
    """Sum of ``duration_seconds`` over this speaker's TIMED events only.
    None (not 0.0) when this speaker has no TIMED events -- a real zero
    duration and "unavailable" must never be confused."""

    timed_word_count: int
    """Word count summed over this speaker's TIMED events only (the
    denominator's numerator for WPM). 0 when there are no TIMED events."""

    speaking_speed_wpm: float | None
    """``timed_word_count / (speaking_duration_seconds / 60)``. None when
    ``speaking_duration_seconds`` is None or 0 -- never a fabricated 0 or
    an infinite value."""

    turn_count: int
    """Number of speaking initiations by this speaker -- see
    ``_compute_turns`` for the exact rule. Computed from the full
    chronological event sequence regardless of timing status, since
    turn-taking depends on speaker order, not on duration."""

    conversation_proportion: float | None
    """This speaker's share (0..1) of ``total_timed_speaking_duration_seconds``
    across all speakers. None when no valid timed speech exists anywhere in
    the meeting, or when this speaker has no timed speech."""

    question_count: int
    """Deterministic question count across ALL of this speaker's events
    (timed or not) -- see ``_question_count``."""


@dataclass(frozen=True, slots=True)
class ConversationAnalytics:
    """Meeting-level conversation-behaviour analytics."""

    per_speaker: tuple[SpeakerAnalytics, ...]
    """One entry per distinct speaker, in first-appearance order."""

    speaker_count: int

    total_timed_speaking_duration_seconds: float | None
    """Sum of every speaker's ``speaking_duration_seconds``. None when no
    TIMED events exist anywhere in the meeting (matches
    ``timeline.Timeline.has_timed_events`` being False, or all TIMED events
    happening to have zero duration)."""

    total_turns: int
    total_questions: int

    timing_available: bool
    """Mirrors ``Timeline.has_timed_events`` for convenience -- duration/WPM/
    proportion metrics are meaningful only when this is True."""

    timed_event_count: int
    total_event_count: int

    timing_coverage_ratio: float | None
    """``timed_event_count / total_event_count``, for surfacing "based on N
    of M segments" when coverage is partial. None when there are no events
    at all (nothing to compute a ratio over)."""


def _word_count(text: str) -> int:
    """Deterministic word count: whitespace-separated tokens.

    Matches the convention already used for the same purpose in
    ``ml_mom/feature_extraction.py`` and ``ml_mom/embeddings.py``
    (``len(text.split())``), reimplemented locally rather than imported --
    see the module docstring for why.
    """

    return len(text.split())


def _question_count(text: str) -> int:
    """Deterministic question count: one question per literal ``?``
    character in the event's text.

    This handles both a single trailing ``?`` (the minimum required rule)
    and multiple explicit questions packed into one timeline event (e.g.
    "Are you done? And QA?" -> 2) without attempting sentence-boundary
    detection or any NLP model. It is intentionally simple and exact:
    "?" characters are counted, nothing is inferred.
    """

    return text.count("?")


def _compute_turns(events: Sequence[TimelineEvent]) -> dict[str, int]:
    """Count speaking initiations per speaker.

    Rule (applied over ALL events in chronological/source order, regardless
    of timing status -- turn-taking depends on who spoke, not on whether
    duration is known):

    A new speaking initiation is counted for a speaker when either
      - it is the first valid (non-blank-text) event in the meeting, or
      - the speaker differs from the previous valid event's speaker.

    Consecutive events from the same speaker (e.g. one continuous
    contribution the STT/diarization split into several adjacent segments)
    contribute to only ONE turn, not one turn each. Events with blank/
    whitespace-only text are skipped entirely -- they neither start a turn
    nor break the run of a surrounding speaker's turn.
    """

    turns: dict[str, int] = {}
    previous_speaker: str | None = None
    for event in events:
        if not event.transcript.strip():
            continue
        if event.speaker_id != previous_speaker:
            turns[event.speaker_id] = turns.get(event.speaker_id, 0) + 1
            previous_speaker = event.speaker_id
    return turns


def analyze_conversation(timeline: Timeline) -> ConversationAnalytics:
    """Compute conversation-behaviour analytics from a Phase 0 ``Timeline``.

    Pure and deterministic: reads ``timeline.events`` only, never mutates
    it, and never touches Sarvam/MiniLM/ANN/the formatter/Gemma/Voxels.
    Safe for an empty timeline (returns an empty, ``timing_available=False``
    result) and for any mix of TIMED/UNTIMED/INVALID events.
    """

    events = timeline.events

    speaker_order: list[str] = []
    speaker_name_by_id: dict[str, str] = {}
    for event in events:
        if event.speaker_id not in speaker_name_by_id:
            speaker_order.append(event.speaker_id)
        speaker_name_by_id[event.speaker_id] = event.speaker_name

    turns_by_speaker = _compute_turns(events)

    timed_duration_by_speaker: dict[str, float] = {}
    timed_words_by_speaker: dict[str, int] = {}
    timed_event_count_by_speaker: dict[str, int] = {}
    total_event_count_by_speaker: dict[str, int] = {}
    questions_by_speaker: dict[str, int] = {}

    for event in events:
        speaker_id = event.speaker_id
        total_event_count_by_speaker[speaker_id] = total_event_count_by_speaker.get(speaker_id, 0) + 1
        questions_by_speaker[speaker_id] = questions_by_speaker.get(speaker_id, 0) + _question_count(
            event.transcript
        )
        if event.timing_status is TimingStatus.TIMED:
            timed_event_count_by_speaker[speaker_id] = timed_event_count_by_speaker.get(speaker_id, 0) + 1
            timed_duration_by_speaker[speaker_id] = timed_duration_by_speaker.get(
                speaker_id, 0.0
            ) + (event.duration_seconds or 0.0)
            timed_words_by_speaker[speaker_id] = timed_words_by_speaker.get(
                speaker_id, 0
            ) + _word_count(event.transcript)

    total_timed_duration = sum(timed_duration_by_speaker.values())
    # A meeting can have TIMED events that all happen to have zero duration
    # (e.g. instant markers); guard against treating that as real coverage
    # for proportion/WPM purposes, not just checking has_timed_events.
    has_meaningful_timed_duration = total_timed_duration > 0

    per_speaker: list[SpeakerAnalytics] = []
    for speaker_id in speaker_order:
        speaking_duration = timed_duration_by_speaker.get(speaker_id)
        words = timed_words_by_speaker.get(speaker_id, 0)

        wpm: float | None = None
        if speaking_duration is not None and speaking_duration > 0:
            wpm = words / (speaking_duration / 60.0)

        proportion: float | None = None
        if has_meaningful_timed_duration and speaking_duration is not None:
            proportion = speaking_duration / total_timed_duration

        per_speaker.append(
            SpeakerAnalytics(
                speaker_id=speaker_id,
                speaker_name=speaker_name_by_id[speaker_id],
                total_event_count=total_event_count_by_speaker.get(speaker_id, 0),
                timed_event_count=timed_event_count_by_speaker.get(speaker_id, 0),
                speaking_duration_seconds=speaking_duration,
                timed_word_count=words,
                speaking_speed_wpm=wpm,
                turn_count=turns_by_speaker.get(speaker_id, 0),
                conversation_proportion=proportion,
                question_count=questions_by_speaker.get(speaker_id, 0),
            )
        )

    total_events = len(events)
    timed_event_count = sum(1 for event in events if event.timing_status is TimingStatus.TIMED)
    coverage = (timed_event_count / total_events) if total_events else None

    return ConversationAnalytics(
        per_speaker=tuple(per_speaker),
        speaker_count=len(speaker_order),
        total_timed_speaking_duration_seconds=(
            total_timed_duration if has_meaningful_timed_duration else None
        ),
        total_turns=sum(turns_by_speaker.values()),
        total_questions=sum(questions_by_speaker.values()),
        timing_available=timeline.has_timed_events,
        timed_event_count=timed_event_count,
        total_event_count=total_events,
        timing_coverage_ratio=coverage,
    )
