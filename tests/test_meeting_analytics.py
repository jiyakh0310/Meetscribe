"""Tests for the SELaD Phase 1 conversation-behaviour analytics
(meeting_analytics/model.py).

Style matches tests/test_timeline.py: plain pytest-style functions with
bare asserts, no unittest.TestCase.
"""

from timeline import Timeline, TimelineEvent, TimingStatus, build_timeline
from meeting_analytics import ConversationAnalytics, SpeakerAnalytics, analyze_conversation
from transcription.sarvam_client import TranscriptionResult, TranscriptionSegment


def _segment(text: str, speaker: str | None, start: float | None, end: float | None) -> TranscriptionSegment:
    return TranscriptionSegment(transcript=text, speaker_id=speaker, start_time_seconds=start, end_time_seconds=end)


def _speaker(analytics: ConversationAnalytics, speaker_id: str) -> SpeakerAnalytics:
    for speaker in analytics.per_speaker:
        if speaker.speaker_id == speaker_id:
            return speaker
    raise AssertionError(f"no speaker analytics for {speaker_id!r}")


# --- A: WPM basic calculation -----------------------------------------------


def test_wpm_basic_calculation_twenty_words_over_ten_seconds() -> None:
    text = " ".join(f"word{i}" for i in range(20))
    result = TranscriptionResult(transcript="ignored", segments=[_segment(text, "0", 0.0, 10.0)])

    analytics = analyze_conversation(build_timeline(result))

    speaker = _speaker(analytics, "Speaker 1")
    assert speaker.timed_word_count == 20
    assert speaker.speaking_duration_seconds == 10.0
    assert speaker.speaking_speed_wpm == 120.0


# --- B: multiple timed segments aggregate correctly -------------------------


def test_multiple_timed_segments_aggregate_duration_and_words() -> None:
    result = TranscriptionResult(
        transcript="ignored",
        segments=[
            _segment("five word segment goes here", "0", 0.0, 5.0),   # 5 words, 5s
            _segment("another six word segment right here", "0", 5.0, 10.0),  # 6 words, 5s
        ],
    )

    analytics = analyze_conversation(build_timeline(result))

    speaker = _speaker(analytics, "Speaker 1")
    assert speaker.speaking_duration_seconds == 10.0
    assert speaker.timed_word_count == 11
    assert speaker.speaking_speed_wpm == 66.0  # 11 words / (10s/60)


# --- C: consecutive same-speaker segments (turn-taking) ---------------------


def test_consecutive_same_speaker_segments_do_not_inflate_turns() -> None:
    # A A B B A -> A=2, B=1 (per the spec's own example)
    result = TranscriptionResult(
        transcript="ignored",
        segments=[
            _segment("a1", "0", 0.0, 1.0),
            _segment("a2", "0", 1.0, 2.0),
            _segment("b1", "1", 2.0, 3.0),
            _segment("b2", "1", 3.0, 4.0),
            _segment("a3", "0", 4.0, 5.0),
        ],
    )

    analytics = analyze_conversation(build_timeline(result))

    a = _speaker(analytics, "Speaker 1")
    b = _speaker(analytics, "Speaker 2")
    assert a.turn_count == 2
    assert b.turn_count == 1
    assert analytics.total_turns == 3


# --- D: talk-time proportion --------------------------------------------------


def test_talk_time_proportion_three_speakers() -> None:
    result = TranscriptionResult(
        transcript="ignored",
        segments=[
            _segment("a", "0", 0.0, 30.0),
            _segment("b", "1", 30.0, 50.0),
            _segment("c", "2", 50.0, 60.0),
        ],
    )

    analytics = analyze_conversation(build_timeline(result))

    a = _speaker(analytics, "Speaker 1")
    b = _speaker(analytics, "Speaker 2")
    c = _speaker(analytics, "Speaker 3")

    assert round(a.conversation_proportion * 100, 2) == 50.0
    assert round(b.conversation_proportion * 100, 2) == 33.33
    assert round(c.conversation_proportion * 100, 2) == 16.67

    total_pct = sum(speaker.conversation_proportion for speaker in analytics.per_speaker) * 100
    assert abs(total_pct - 100.0) < 0.01


# --- E: question count -------------------------------------------------------


def test_question_count_handles_single_and_multiple_questions() -> None:
    result = TranscriptionResult(
        transcript="ignored",
        segments=[
            _segment("Is this ready?", "0", 0.0, 2.0),  # 1 question
            _segment("It is ready.", "1", 2.0, 4.0),  # 0 questions
            _segment("Are we on track? Can we ship Friday?", "0", 4.0, 8.0),  # 2 questions
        ],
    )

    analytics = analyze_conversation(build_timeline(result))

    speaker0 = _speaker(analytics, "Speaker 1")
    speaker1 = _speaker(analytics, "Speaker 2")
    assert speaker0.question_count == 3  # 1 + 2
    assert speaker1.question_count == 0
    assert analytics.total_questions == 3


# --- F: untimed transcript ----------------------------------------------------


def test_untimed_transcript_marks_timing_metrics_unavailable_but_keeps_text_metrics() -> None:
    result = TranscriptionResult(
        transcript="ignored",
        segments=[
            _segment("Good afternoon everyone.", "0", None, None),
            _segment("Is the report ready?", "1", None, None),
            _segment("Yes it is.", "0", None, None),
        ],
    )

    analytics = analyze_conversation(build_timeline(result))

    assert analytics.timing_available is False
    assert analytics.total_timed_speaking_duration_seconds is None

    speaker0 = _speaker(analytics, "Speaker 1")
    speaker1 = _speaker(analytics, "Speaker 2")

    # Timing-dependent metrics are unavailable, not fabricated as zero.
    assert speaker0.speaking_duration_seconds is None
    assert speaker0.speaking_speed_wpm is None
    assert speaker0.conversation_proportion is None

    # Text-based metrics still work without any timing at all.
    assert speaker1.question_count == 1
    assert speaker0.turn_count == 2  # 0 -> 1 -> 0
    assert speaker1.turn_count == 1
    assert analytics.total_turns == 3


# --- G: invalid timing --------------------------------------------------------


def test_invalid_timing_excluded_from_wpm_and_talk_time_totals() -> None:
    result = TranscriptionResult(
        transcript="ignored",
        segments=[
            _segment("malformed timing here", "0", 10.0, 4.0),  # end < start -> INVALID
            _segment("valid segment after it", "1", 10.0, 12.0),  # TIMED
        ],
    )

    analytics = analyze_conversation(build_timeline(result))

    bad = _speaker(analytics, "Speaker 1")
    good = _speaker(analytics, "Speaker 2")

    # The invalid segment contributes no duration/words/WPM/proportion...
    assert bad.speaking_duration_seconds is None
    assert bad.speaking_speed_wpm is None
    assert bad.conversation_proportion is None
    assert bad.timed_event_count == 0
    # ...but it still exists as a real speaking turn (turn-taking is
    # independent of timing validity).
    assert bad.turn_count == 1

    # The invalid segment must not contaminate the meeting-level total either.
    assert analytics.total_timed_speaking_duration_seconds == 2.0
    assert good.speaking_duration_seconds == 2.0
    assert good.conversation_proportion == 1.0


# --- H: mixed timed/untimed ---------------------------------------------------


def test_mixed_timed_and_untimed_events_use_only_the_valid_timed_subset() -> None:
    result = TranscriptionResult(
        transcript="ignored",
        segments=[
            _segment("timed segment with four words", "0", 0.0, 10.0),  # TIMED, 5 words
            _segment("an untimed segment with words here too", "0", None, None),  # UNTIMED
        ],
    )

    analytics = analyze_conversation(build_timeline(result))
    speaker = _speaker(analytics, "Speaker 1")

    # WPM/duration reflect only the TIMED segment (5 words / (10s/60) = 30).
    assert speaker.speaking_duration_seconds == 10.0
    assert speaker.timed_word_count == 5
    assert speaker.speaking_speed_wpm == 30.0

    # Coverage information is truthful: 1 of 2 events had valid timing.
    assert analytics.timed_event_count == 1
    assert analytics.total_event_count == 2
    assert analytics.timing_coverage_ratio == 0.5

    # Question/turn counting still reflects both events (text-based).
    assert speaker.total_event_count == 2
    assert speaker.turn_count == 1  # same speaker throughout -> one turn


# --- I: empty meeting ----------------------------------------------------------


def test_empty_meeting_returns_empty_analytics_without_crash() -> None:
    analytics = analyze_conversation(build_timeline(None))

    assert analytics.per_speaker == ()
    assert analytics.speaker_count == 0
    assert analytics.total_timed_speaking_duration_seconds is None
    assert analytics.total_turns == 0
    assert analytics.total_questions == 0
    assert analytics.timing_available is False
    assert analytics.timed_event_count == 0
    assert analytics.total_event_count == 0
    assert analytics.timing_coverage_ratio is None


# --- J: speaker mapping ---------------------------------------------------------


def test_reviewed_speaker_mapping_names_attach_to_correct_stable_speaker() -> None:
    result = TranscriptionResult(
        transcript="ignored",
        segments=[
            _segment("Priya's line.", "0", 0.0, 3.0),
            _segment("Amit's line.", "1", 3.0, 6.0),
        ],
    )
    mapping = {"Speaker 1": "Priya", "Speaker 2": "Amit"}

    analytics = analyze_conversation(build_timeline(result, mapping=mapping))

    priya = _speaker(analytics, "Speaker 1")
    amit = _speaker(analytics, "Speaker 2")
    # Stable identity (speaker_id) is untouched by the mapping...
    assert priya.speaker_id == "Speaker 1"
    assert amit.speaker_id == "Speaker 2"
    # ...while the display name reflects review, correctly associated with
    # its own speaker's metrics (not swapped).
    assert priya.speaker_name == "Priya"
    assert priya.speaking_duration_seconds == 3.0
    assert amit.speaker_name == "Amit"
    assert amit.speaking_duration_seconds == 3.0


# --- K: Timeline immutability ----------------------------------------------------


def test_analyze_conversation_does_not_mutate_timeline_or_events() -> None:
    result = TranscriptionResult(
        transcript="ignored",
        segments=[_segment("Do not touch.", "0", 0.0, 2.0)],
    )
    tl = build_timeline(result)
    original_event = tl.events[0]

    analyze_conversation(tl)

    # Timeline/TimelineEvent are frozen dataclasses, so any attempted
    # mutation would raise; this additionally confirms object identity and
    # field values are completely untouched by analysis.
    assert tl.events[0] is original_event
    assert tl.events[0].transcript == "Do not touch."
    assert tl.events[0].speaker_id == "Speaker 1"


# --- Single-speaker edge case (also exercised via AppTest in Phase 1 UI tests) --


def test_single_speaker_meeting_gets_full_proportion_no_divide_by_zero() -> None:
    result = TranscriptionResult(
        transcript="ignored",
        segments=[_segment("Only one speaker here.", "0", 0.0, 5.0)],
    )

    analytics = analyze_conversation(build_timeline(result))
    speaker = _speaker(analytics, "Speaker 1")
    assert speaker.conversation_proportion == 1.0
    assert analytics.speaker_count == 1
