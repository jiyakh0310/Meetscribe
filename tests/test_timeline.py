"""Tests for the SELaD Phase 0 shared timeline foundation (timeline/model.py).

Style matches the existing transcript-domain tests in this directory
(tests/test_audio_transcript_normalization.py,
tests/test_audio_transcript_repair.py): plain pytest-style functions with
bare asserts, no unittest.TestCase.
"""

from timeline import Timeline, TimelineEvent, TimingStatus, build_timeline, timed_events
from transcription.sarvam_client import TranscriptionResult, TranscriptionSegment


def _segment(
    text: str,
    speaker: str | None,
    start: float | None,
    end: float | None,
) -> TranscriptionSegment:
    return TranscriptionSegment(
        transcript=text,
        speaker_id=speaker,
        start_time_seconds=start,
        end_time_seconds=end,
    )


# --- Test A: normal diarized segments -------------------------------------


def test_normal_diarized_segments_preserve_speaker_text_timing_and_order() -> None:
    result = TranscriptionResult(
        transcript="ignored",
        segments=[
            _segment("Let's start the meeting.", "0", 0.0, 4.5),
            _segment("Sure, I'm ready.", "1", 4.5, 6.25),
            _segment("QA testing is at seventy percent.", "1", 6.25, 11.0),
        ],
    )

    timeline = build_timeline(result)

    assert len(timeline.events) == 3
    assert timeline.has_timed_events is True

    first, second, third = timeline.events

    assert first.transcript == "Let's start the meeting."
    assert first.speaker_id == "Speaker 1"
    assert first.start_time_seconds == 0.0
    assert first.end_time_seconds == 4.5
    assert first.duration_seconds == 4.5
    assert first.timing_status is TimingStatus.TIMED

    assert second.transcript == "Sure, I'm ready."
    assert second.speaker_id == "Speaker 2"
    assert second.duration_seconds == 1.75

    assert third.transcript == "QA testing is at seventy percent."
    assert third.speaker_id == "Speaker 2"
    assert third.duration_seconds == 4.75

    # Chronological/source order preserved exactly.
    assert [event.event_id for event in timeline.events] == [0, 1, 2]
    assert [event.source_index for event in timeline.events] == [0, 1, 2]
    assert [event.start_time_seconds for event in timeline.events] == [0.0, 4.5, 6.25]


# --- Test B: same-speaker consecutive segments -----------------------------


def test_same_speaker_consecutive_segments_are_not_merged_or_reordered() -> None:
    result = TranscriptionResult(
        transcript="ignored",
        segments=[
            _segment("First fragment.", "0", 0.0, 2.0),
            _segment("Second fragment, same speaker.", "0", 2.0, 5.0),
            _segment("Third fragment, same speaker.", "0", 5.0, 7.5),
        ],
    )

    timeline = build_timeline(result)

    # Phase 0 builds one event per source segment; it must never silently
    # merge same-speaker segments -- any merging is the job of the existing,
    # separate normalization step (transcription/audio_transcript_normalization.py)
    # applied earlier in the pipeline, not the timeline builder.
    assert len(timeline.events) == 3
    assert [event.transcript for event in timeline.events] == [
        "First fragment.",
        "Second fragment, same speaker.",
        "Third fragment, same speaker.",
    ]
    assert [event.speaker_id for event in timeline.events] == ["Speaker 1"] * 3
    # Timing stays monotonic and untouched.
    assert [event.start_time_seconds for event in timeline.events] == [0.0, 2.0, 5.0]
    assert [event.end_time_seconds for event in timeline.events] == [2.0, 5.0, 7.5]
    assert [event.duration_seconds for event in timeline.events] == [2.0, 3.0, 2.5]


# --- Test C: missing timestamps ---------------------------------------------


def test_missing_timestamps_degrade_gracefully_without_fabrication() -> None:
    result = TranscriptionResult(
        transcript="ignored",
        segments=[
            _segment("Uploaded transcript line with no timing.", "0", None, None),
            _segment("Another line, still no timing.", "1", None, None),
        ],
    )

    timeline = build_timeline(result)

    assert len(timeline.events) == 2
    assert timeline.has_timed_events is False
    for event in timeline.events:
        assert event.start_time_seconds is None
        assert event.end_time_seconds is None
        assert event.duration_seconds is None
        assert event.timing_status is TimingStatus.UNTIMED
    # Text/speaker are still preserved even without timing.
    assert timeline.events[0].transcript == "Uploaded transcript line with no timing."
    assert timeline.events[0].speaker_id == "Speaker 1"


# --- Test D: segment-less transcript ----------------------------------------


def test_segment_less_transcript_result_returns_empty_timeline_without_crash() -> None:
    # Mirrors the short/no-diarization Sarvam real-time path, which returns a
    # flat transcript with segments == [].
    result = TranscriptionResult(transcript="Just a flat transcript, no segments.", segments=[])

    timeline = build_timeline(result)

    assert timeline.events == ()
    assert timeline.has_timed_events is False


def test_none_result_returns_empty_timeline_without_crash() -> None:
    timeline = build_timeline(None)

    assert timeline.events == ()
    assert timeline.has_timed_events is False


# --- Test E: invalid timing --------------------------------------------------


def test_end_before_start_never_produces_negative_duration() -> None:
    result = TranscriptionResult(
        transcript="ignored",
        segments=[
            _segment("Malformed timing segment.", "0", 10.0, 4.0),
            _segment("Normal segment after it.", "1", 10.0, 12.0),
        ],
    )

    timeline = build_timeline(result)

    bad, good = timeline.events

    # The raw (malformed) values are preserved for inspection...
    assert bad.start_time_seconds == 10.0
    assert bad.end_time_seconds == 4.0
    # ...but duration is never negative -- it degrades to None instead, and
    # the status makes the reason explicit rather than looking like a
    # plain "no timing available" case.
    assert bad.duration_seconds is None
    assert bad.timing_status is TimingStatus.INVALID

    # A malformed segment does not corrupt the ones around it.
    assert good.duration_seconds == 2.0
    assert good.timing_status is TimingStatus.TIMED

    # timed_events() must exclude the invalid one alongside untimed ones.
    assert timed_events(timeline.events) == (good,)


def test_equal_start_and_end_is_a_valid_zero_duration_event() -> None:
    result = TranscriptionResult(
        transcript="ignored",
        segments=[_segment("Instant marker.", "0", 5.0, 5.0)],
    )

    timeline = build_timeline(result)

    assert timeline.events[0].duration_seconds == 0.0
    assert timeline.events[0].timing_status is TimingStatus.TIMED


# --- Test F: speaker mapping --------------------------------------------------


def test_reviewed_speaker_mapping_is_reflected_in_speaker_name() -> None:
    result = TranscriptionResult(
        transcript="ignored",
        segments=[
            _segment("I'll finish the QA testing.", "0", 0.0, 3.0),
            _segment("I'll handle the marketing collateral.", "1", 3.0, 6.0),
        ],
    )
    mapping = {"Speaker 1": "Priya", "Speaker 2": "Amit"}

    timeline = build_timeline(result, mapping=mapping)

    priya_event, amit_event = timeline.events
    # The stable generic key is untouched by the mapping...
    assert priya_event.speaker_id == "Speaker 1"
    assert amit_event.speaker_id == "Speaker 2"
    # ...while the display name reflects the reviewed mapping, and stays
    # correctly associated with its own timed segment (not swapped/shifted).
    assert priya_event.speaker_name == "Priya"
    assert priya_event.start_time_seconds == 0.0
    assert amit_event.speaker_name == "Amit"
    assert amit_event.start_time_seconds == 3.0


def test_unmapped_speaker_falls_back_to_generic_label_not_blank() -> None:
    result = TranscriptionResult(
        transcript="ignored",
        segments=[_segment("No mapping entry for this speaker.", "0", 0.0, 2.0)],
    )
    # A partial/incomplete mapping (as could exist mid-review) must not
    # blank out the unmapped speaker's display name.
    mapping = {"Speaker 2": "Amit"}

    timeline = build_timeline(result, mapping=mapping)

    assert timeline.events[0].speaker_id == "Speaker 1"
    assert timeline.events[0].speaker_name == "Speaker 1"


def test_already_display_named_segments_are_handled_without_a_mapping() -> None:
    # Mirrors calling build_timeline on the *post-review* TranscriptionResult,
    # where transcription.speaker_mapping.apply_mapping_to_result has already
    # replaced segment.speaker_id with the reviewed display name -- the
    # authoritative integration point documented in timeline/model.py.
    result = TranscriptionResult(
        transcript="ignored",
        segments=[_segment("Already-resolved display name on the segment.", "Priya", 0.0, 2.0)],
    )

    timeline = build_timeline(result)

    assert timeline.events[0].speaker_id == "Priya"
    assert timeline.events[0].speaker_name == "Priya"


# --- Non-mutation guarantee ---------------------------------------------------


def test_source_segments_are_not_mutated() -> None:
    segment = _segment("Do not touch me.", "0", 1.0, 2.0)
    result = TranscriptionResult(transcript="ignored", segments=[segment])

    build_timeline(result, mapping={"Speaker 1": "Someone Else"})

    # TranscriptionSegment is frozen, so any attempted mutation would raise;
    # this additionally confirms the original object identity and values are
    # completely untouched by the call.
    assert result.segments[0] is segment
    assert segment.speaker_id == "0"
    assert segment.transcript == "Do not touch me."
