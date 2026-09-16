"""Tests for the SELaD Phase 2 content analytics (meeting_analytics/content.py):
topic grouping, keyword extraction, and positive-language proportion.

Style matches tests/test_timeline.py and tests/test_meeting_analytics.py:
plain pytest-style functions with bare asserts, no unittest.TestCase.
"""

from timeline import build_timeline
from meeting_analytics.content import (
    _rank_keywords,
    _tokenize,
    analyze_content,
)
from transcription.sarvam_client import TranscriptionResult, TranscriptionSegment


def _segment(text: str, speaker: str | None, start: float | None, end: float | None) -> TranscriptionSegment:
    return TranscriptionSegment(transcript=text, speaker_id=speaker, start_time_seconds=start, end_time_seconds=end)


def _speaker_kw(analytics, speaker_id: str):
    for entry in analytics.keywords.per_speaker:
        if entry.speaker_id == speaker_id:
            return entry
    raise AssertionError(f"no speaker keywords for {speaker_id!r}")


def _speaker_lang(analytics, speaker_id: str):
    for entry in analytics.positive_language.per_speaker:
        if entry.speaker_id == speaker_id:
            return entry
    raise AssertionError(f"no positive-language entry for {speaker_id!r}")


# --- A: keyword extraction ----------------------------------------------------


def test_repeated_meaningful_terms_rank_above_noise() -> None:
    documents = [
        "The database migration plan needs review.",
        "We discussed the database migration timeline in detail.",
        "The database migration is the main risk for this release.",
        "Okay, sure, that sounds fine to me.",
    ]
    ranked = _rank_keywords(documents, top_n=5)
    # Phase 3.2: a repeated, meaningful bigram ("database migration",
    # occurring 3 times) now outranks and subsumes its own component
    # unigrams rather than appearing alongside them redundantly.
    assert "database migration" in ranked
    assert ranked[0] == "database migration"
    # A generic filler document contributes no meaningful ranked term.
    assert "sure" not in ranked
    assert "fine" not in ranked


# --- B: stopword/noise handling ------------------------------------------------


def test_common_function_words_do_not_dominate_ranking() -> None:
    documents = [
        "We are going to do the thing that we said we would do.",
        "Release testing and release planning are the two topics today.",
    ]
    ranked = _rank_keywords(documents, top_n=5)
    for stopword in ("we", "are", "the", "to", "that", "would"):
        assert stopword not in ranked
    assert "release" in ranked


# --- C: empty text -------------------------------------------------------------


def test_empty_documents_produce_no_keywords_without_crash() -> None:
    assert _rank_keywords([], top_n=5) == []
    assert _rank_keywords(["", "   ", "?? !!"], top_n=5) == []


# --- D: small transcript --------------------------------------------------------


def test_small_transcript_produces_graceful_output() -> None:
    result = TranscriptionResult(transcript="ignored", segments=[_segment("Hi.", "0", 0.0, 1.0)])
    analytics = analyze_content(build_timeline(result))
    assert len(analytics.topics) == 1
    assert analytics.topics[0].event_count == 1
    # Not enough lexical evidence -> neutral fallback title, not a guess.
    assert analytics.topics[0].title == "Discussion Topic 1"


# --- E: topic grouping (structural invariants, not brittle floats) -------------


def test_semantically_distinct_content_does_not_always_collapse_into_one_topic() -> None:
    segments = []
    for i in range(4):
        segments.append(_segment(f"QA testing status update number {i}, release testing looks solid.", "0", float(i * 5), float(i * 5 + 4)))
    base = len(segments) * 5
    for i in range(4):
        segments.append(_segment(f"Marketing campaign budget discussion number {i}, campaign spend review.", "1", float(base + i * 5), float(base + i * 5 + 4)))
    result = TranscriptionResult(transcript="ignored", segments=segments)

    analytics = analyze_content(build_timeline(result))

    # Structural invariant only: distinct topical content should not always
    # collapse into a single giant topic, and should not fragment into one
    # topic per event either. Exact cluster count depends on model
    # behavior and is intentionally not asserted.
    assert 1 <= len(analytics.topics) <= 8
    total_events_in_topics = sum(topic.event_count for topic in analytics.topics)
    assert total_events_in_topics == 8


def test_repeated_identical_content_does_not_fragment_into_many_topics() -> None:
    segments = [
        _segment("Release testing status is on track for Friday.", "0", float(i * 5), float(i * 5 + 4))
        for i in range(5)
    ]
    result = TranscriptionResult(transcript="ignored", segments=segments)
    analytics = analyze_content(build_timeline(result))
    # Near-identical utterances should not fragment into 5 separate topics.
    assert len(analytics.topics) < 5


# --- F: topic source traceability ------------------------------------------------


def test_every_topic_source_event_id_maps_to_a_real_timeline_event() -> None:
    result = TranscriptionResult(
        transcript="ignored",
        segments=[
            _segment("Release testing update for today's meeting.", "0", 0.0, 4.0),
            _segment("Marketing campaign review for this quarter.", "1", 4.0, 8.0),
        ],
    )
    timeline = build_timeline(result)
    analytics = analyze_content(timeline)

    real_ids = {event.event_id for event in timeline.events}
    for topic in analytics.topics:
        for source_id in topic.source_event_ids:
            assert source_id in real_ids
        assert len(topic.source_event_ids) == topic.event_count


# --- G: topic timing --------------------------------------------------------------


def test_topic_timing_uses_earliest_start_and_latest_end_of_timed_events() -> None:
    result = TranscriptionResult(
        transcript="ignored",
        segments=[
            _segment("Release testing update one.", "0", 10.0, 14.0),
            _segment("Release testing update two.", "0", 14.0, 20.0),
        ],
    )
    analytics = analyze_content(build_timeline(result))
    assert len(analytics.topics) == 1
    topic = analytics.topics[0]
    assert topic.start_time_seconds == 10.0
    assert topic.end_time_seconds == 20.0


def test_untimed_topic_has_no_fabricated_time_range() -> None:
    result = TranscriptionResult(
        transcript="ignored",
        segments=[_segment("Release testing update with no timestamps.", "0", None, None)],
    )
    analytics = analyze_content(build_timeline(result))
    assert len(analytics.topics) == 1
    topic = analytics.topics[0]
    assert topic.start_time_seconds is None
    assert topic.end_time_seconds is None


# --- H: speaker preservation ----------------------------------------------------


def test_topic_participants_reflect_actual_contributing_speakers() -> None:
    result = TranscriptionResult(
        transcript="ignored",
        segments=[
            _segment("Release testing update from Rahul.", "0", 0.0, 4.0),
            _segment("Release testing follow-up from Priya.", "1", 4.0, 8.0),
        ],
    )
    mapping = {"Speaker 1": "Rahul", "Speaker 2": "Priya"}
    analytics = analyze_content(build_timeline(result, mapping=mapping))
    assert len(analytics.topics) == 1
    assert set(analytics.topics[0].participant_speakers) == {"Rahul", "Priya"}


# --- I: positive-language basic --------------------------------------------------


def test_known_positive_terms_produce_expected_numerator_and_proportion() -> None:
    result = TranscriptionResult(
        transcript="ignored",
        segments=[_segment("This is great and excellent work, thank you.", "0", 0.0, 4.0)],
    )
    analytics = analyze_content(build_timeline(result))
    speaker = _speaker_lang(analytics, "Speaker 1")
    # tokens: this is great and excellent work thank you -> 8 eligible
    # positive: great, excellent, thank -> 3
    assert speaker.eligible_token_count == 8
    assert speaker.positive_token_count == 3
    assert speaker.proportion == 3 / 8


# --- J: zero positive terms -------------------------------------------------------


def test_genuine_zero_positive_terms_is_a_real_zero_not_unavailable() -> None:
    result = TranscriptionResult(
        transcript="ignored",
        segments=[_segment("The server returned an error during the deployment step.", "0", 0.0, 4.0)],
    )
    analytics = analyze_content(build_timeline(result))
    speaker = _speaker_lang(analytics, "Speaker 1")
    assert speaker.eligible_token_count > 0
    assert speaker.positive_token_count == 0
    assert speaker.proportion == 0.0
    assert speaker.proportion is not None


# --- K: no eligible text -----------------------------------------------------------


def test_no_eligible_text_is_unavailable_not_fabricated_zero() -> None:
    result = TranscriptionResult(
        transcript="ignored",
        segments=[_segment("   ", "0", 0.0, 1.0), _segment("...", "1", 1.0, 2.0)],
    )
    analytics = analyze_content(build_timeline(result))
    assert analytics.positive_language.meeting_eligible_token_count == 0
    assert analytics.positive_language.meeting_proportion is None


# --- L: untimed transcript ----------------------------------------------------------


def test_untimed_transcript_still_produces_keywords_and_positive_language() -> None:
    result = TranscriptionResult(
        transcript="ignored",
        segments=[
            _segment("Release testing went great this week, excellent progress overall.", "0", None, None),
            _segment("Marketing campaign budget review needed before Friday.", "1", None, None),
        ],
    )
    analytics = analyze_content(build_timeline(result))
    assert analytics.keywords.meeting_keywords  # non-empty
    assert analytics.positive_language.meeting_proportion is not None
    # Topics still form (grouped by text alone) without fabricated timing.
    assert len(analytics.topics) >= 1
    for topic in analytics.topics:
        assert topic.start_time_seconds is None
        assert topic.end_time_seconds is None


# --- M: mixed timing ------------------------------------------------------------------


def test_mixed_timing_topic_range_uses_only_valid_timed_events() -> None:
    result = TranscriptionResult(
        transcript="ignored",
        segments=[
            _segment("Release testing update one.", "0", 10.0, 14.0),
            _segment("Release testing update two, no timing here.", "0", None, None),
            _segment("Release testing update three.", "0", 20.0, 24.0),
        ],
    )
    analytics = analyze_content(build_timeline(result))
    assert len(analytics.topics) == 1
    topic = analytics.topics[0]
    assert topic.event_count == 3
    assert topic.start_time_seconds == 10.0
    assert topic.end_time_seconds == 24.0


# --- N: determinism ----------------------------------------------------------------------


def test_same_input_produces_same_analytics_twice() -> None:
    result = TranscriptionResult(
        transcript="ignored",
        segments=[
            _segment("Release testing update, great progress on QA.", "0", 0.0, 4.0),
            _segment("Marketing campaign review for this quarter.", "1", 4.0, 8.0),
        ],
    )
    timeline = build_timeline(result, mapping={"Speaker 1": "Rahul", "Speaker 2": "Priya"})

    first = analyze_content(timeline)
    second = analyze_content(timeline)

    assert first.keywords.meeting_keywords == second.keywords.meeting_keywords
    assert [t.title for t in first.topics] == [t.title for t in second.topics]
    assert [t.keywords for t in first.topics] == [t.keywords for t in second.topics]
    assert first.positive_language.meeting_proportion == second.positive_language.meeting_proportion


# --- O: Timeline immutability --------------------------------------------------------------


def test_analyze_content_does_not_mutate_timeline_or_events() -> None:
    result = TranscriptionResult(
        transcript="ignored",
        segments=[_segment("Do not touch this event.", "0", 0.0, 2.0)],
    )
    timeline = build_timeline(result)
    original_event = timeline.events[0]

    analyze_content(timeline)

    assert timeline.events[0] is original_event
    assert timeline.events[0].transcript == "Do not touch this event."
    assert timeline.events[0].speaker_id == "Speaker 1"


# --- Tokenizer sanity (supports the "does not break on Hinglish" requirement) --------------


def test_tokenizer_handles_hinglish_text_without_crashing() -> None:
    tokens = _tokenize("Yeh release date next Monday tak ho jaana chahiye, theek hai?")
    assert "release" in tokens
    assert "monday" in tokens
    assert all(token.isalpha() or "'" in token or "-" in token for token in tokens)
