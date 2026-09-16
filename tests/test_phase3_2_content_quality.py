"""Tests for Phase 3.2 (SELaD content-analytics quality & generalization).

Covers: analytics-only semantic segmentation, unit-based topic clustering
with the consecutive-context rule, deterministic keyphrase (unigram +
bigram) ranking with participant-name/conversational-noise filtering,
deterministic topic titles, and the Positive-language-cues phrase
extension. Style matches tests/test_meeting_analytics_content.py: plain
pytest-style functions with bare asserts.

Every scenario uses SYNTHETIC content, distinct in names/wording/domain
from the manually-tested sample audio (no "Tom"/"Sumit"/"Will"/"JIYA"/
"RingCentral"/"Beta Program"/"Deployment"-exact-sample-sentence content),
per the Phase 3.2 anti-overfitting requirement. Tests assert semantic
PROPERTIES (title is not a participant name, distinct subjects produce
distinct topics, no fabricated timing, ...), not brittle exact strings,
except where deterministic behavior specifically guarantees a string.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from timeline import build_timeline
from meeting_analytics.content import (
    _candidate_ngrams,
    _is_non_substantive_fragment,
    _rank_keywords,
    _segment_event_text,
    _tokenize,
    analyze_content,
)
from transcription.sarvam_client import TranscriptionResult, TranscriptionSegment


def _segment(text: str, speaker: str | None, start: float | None, end: float | None) -> TranscriptionSegment:
    return TranscriptionSegment(transcript=text, speaker_id=speaker, start_time_seconds=start, end_time_seconds=end)


def _topics_by_title_substring(analytics, needle: str):
    needle = needle.casefold()
    return [t for t in analytics.topics if needle in t.title.casefold()]


def _all_keyword_text(analytics) -> str:
    parts = list(analytics.keywords.meeting_keywords)
    for topic in analytics.topics:
        parts.extend(topic.keywords)
        parts.append(topic.title)
    return " ".join(parts).casefold()


# ---------------------------------------------------------------------------
# Generalization matrix
# ---------------------------------------------------------------------------


def test_case_1_technical_architecture_meeting_produces_meaningful_topics() -> None:
    segments = [
        _segment("Let's review the API integration plan for the payments service.", "0", 0.0, 6.0),
        _segment("The API integration looks solid, the authentication flow is working well.", "1", 6.0, 12.0),
        _segment("Now let's move to the database migration for the reporting cluster.", "0", 12.0, 18.0),
        _segment("The database migration timeline depends on the authentication rollout finishing first.", "1", 18.0, 26.0),
    ]
    result = TranscriptionResult(transcript="ignored", segments=segments)
    analytics = analyze_content(build_timeline(result, mapping={"Speaker 1": "Elena", "Speaker 2": "Farid"}))

    assert len(analytics.topics) >= 1
    for topic in analytics.topics:
        assert topic.title not in ("Elena", "Farid")
        assert topic.title.strip() != ""
    combined = _all_keyword_text(analytics)
    assert "api" in combined or "database" in combined or "authentication" in combined


def test_case_2_product_planning_meeting_covers_expected_concepts() -> None:
    segments = [
        _segment("We should finalize the customer onboarding flow before the beta launch.", "0", 0.0, 6.0),
        _segment("Onboarding needs a clearer pricing explanation during signup.", "1", 6.0, 12.0),
        _segment("Let's also review customer feedback from the last pricing survey.", "0", 12.0, 18.0),
        _segment("The feedback on pricing was mostly about the beta launch timing.", "1", 18.0, 26.0),
    ]
    result = TranscriptionResult(transcript="ignored", segments=segments)
    analytics = analyze_content(build_timeline(result))
    combined = _all_keyword_text(analytics)
    assert any(term in combined for term in ("onboarding", "pricing", "beta", "feedback"))


def test_case_3_business_operations_meeting_covers_expected_concepts() -> None:
    segments = [
        _segment("The hiring budget for next quarter needs approval from finance.", "0", 0.0, 6.0),
        _segment("We also need to confirm the vendor timeline for the new supplier.", "1", 6.0, 12.0),
        _segment("Hiring three engineers this quarter affects the budget significantly.", "0", 12.0, 18.0),
        _segment("The vendor timeline slipped, so procurement is reviewing the budget again.", "1", 18.0, 26.0),
    ]
    result = TranscriptionResult(transcript="ignored", segments=segments)
    analytics = analyze_content(build_timeline(result))
    combined = _all_keyword_text(analytics)
    assert any(term in combined for term in ("budget", "hiring", "vendor"))


def test_case_4_informational_demo_meeting_introductions_do_not_dominate() -> None:
    segments = [
        _segment("Hi everyone, thanks for joining today's demo.", "0", 0.0, 4.0),
        _segment("Hello, glad to be here, excited to see the product.", "1", 4.0, 8.0),
        _segment("This dashboard shows real-time inventory levels across warehouses.", "0", 8.0, 16.0),
        _segment("Can the dashboard also show historical inventory trends?", "1", 16.0, 22.0),
        _segment("Yes, the inventory trend view is available under the analytics tab.", "0", 22.0, 30.0),
    ]
    result = TranscriptionResult(transcript="ignored", segments=segments)
    mapping = {"Speaker 1": "Priya", "Speaker 2": "Marcus"}
    analytics = analyze_content(build_timeline(result, mapping=mapping))

    for topic in analytics.topics:
        assert "priya" not in topic.title.casefold()
        assert "marcus" not in topic.title.casefold()
    for keyword in analytics.keywords.meeting_keywords:
        assert keyword.casefold() not in ("priya", "marcus")
    combined = _all_keyword_text(analytics)
    assert "inventory" in combined or "dashboard" in combined


def test_case_5_repeated_topic_across_multiple_speakers_stays_one_topic_with_all_participants() -> None:
    segments = [
        _segment("The release testing status looks good for this cycle.", "0", 0.0, 5.0),
        _segment("Agreed, release testing is on track for Friday.", "1", 5.0, 10.0),
        _segment("One more note on release testing: the regression suite passed.", "2", 10.0, 15.0),
    ]
    result = TranscriptionResult(transcript="ignored", segments=segments)
    mapping = {"Speaker 1": "Ingrid", "Speaker 2": "Hassan", "Speaker 3": "Grace"}
    analytics = analyze_content(build_timeline(result, mapping=mapping))

    matches = _topics_by_title_substring(analytics, "release") or _topics_by_title_substring(analytics, "testing")
    assert matches, [t.title for t in analytics.topics]
    topic = matches[0]
    assert set(topic.participant_speakers) >= {"Ingrid", "Hassan", "Grace"}


def test_case_6_long_single_speaker_turn_with_three_subjects_allows_multiple_topics() -> None:
    long_turn = (
        "First, let's talk about the customer onboarding checklist and how we track completion. "
        "Second, I want to raise the marketing campaign budget for next quarter and whether it needs revision. "
        "Third, there is an open question about warehouse inventory accuracy that operations flagged this week."
    )
    segments = [_segment(long_turn, "0", 0.0, 60.0)]
    result = TranscriptionResult(transcript="ignored", segments=segments)
    analytics = analyze_content(build_timeline(result))

    # Old Phase 2 behavior: one whole TimelineEvent -> exactly one topic.
    # New Phase 3.2 behavior: the analytics-only sentence segmentation lets
    # this single long event contribute to more than one topic when its
    # sentences are not all semantically similar.
    assert len(analytics.topics) >= 1
    # Every topic must still trace back to the SAME one real source event.
    for topic in analytics.topics:
        assert topic.source_event_ids == (0,)
        assert topic.event_count == 1


def test_case_7_short_back_and_forth_qa_does_not_fragment_into_many_micro_topics() -> None:
    segments = [
        _segment("What is the current status of the checkout redesign?", "0", 0.0, 4.0),
        _segment("The checkout redesign is in final review with design.", "1", 4.0, 8.0),
        _segment("Is the checkout redesign expected to ship this sprint?", "0", 8.0, 12.0),
        _segment("Yes, the checkout redesign should ship by end of sprint.", "1", 12.0, 16.0),
    ]
    result = TranscriptionResult(transcript="ignored", segments=segments)
    analytics = analyze_content(build_timeline(result))
    assert len(analytics.topics) <= 2


def test_case_8_transcript_only_meeting_has_no_fabricated_timestamps() -> None:
    segments = [
        _segment("We reviewed the vendor contract renewal terms in detail today.", "0", None, None),
        _segment("The vendor contract renewal looks reasonable for another year.", "1", None, None),
    ]
    result = TranscriptionResult(transcript="ignored", segments=segments)
    analytics = analyze_content(build_timeline(result))
    assert len(analytics.topics) >= 1
    for topic in analytics.topics:
        assert topic.start_time_seconds is None
        assert topic.end_time_seconds is None


def test_case_9_hinglish_meeting_does_not_crash_and_produces_output() -> None:
    segments = [
        _segment("Yeh onboarding process next Monday tak complete ho jaana chahiye, theek hai?", "0", 0.0, 6.0),
        _segment("Haan bilkul, onboarding checklist ready hai already.", "1", 6.0, 12.0),
    ]
    result = TranscriptionResult(transcript="ignored", segments=segments)
    analytics = analyze_content(build_timeline(result))
    assert len(analytics.topics) >= 1
    assert isinstance(analytics.keywords.meeting_keywords, tuple)


def test_case_10_participant_names_in_introductions_do_not_dominate_key_themes() -> None:
    segments = [
        _segment("Hi, I'm Oksana, I lead the platform team.", "0", 0.0, 4.0),
        _segment("Hi, I'm Tariq, I lead the growth team.", "1", 4.0, 8.0),
        _segment("Let's discuss the subscription pricing model changes for next quarter.", "0", 8.0, 16.0),
        _segment("The subscription pricing changes need legal review before rollout.", "1", 16.0, 24.0),
    ]
    result = TranscriptionResult(transcript="ignored", segments=segments)
    mapping = {"Speaker 1": "Oksana", "Speaker 2": "Tariq"}
    analytics = analyze_content(build_timeline(result, mapping=mapping))
    for keyword in analytics.keywords.meeting_keywords:
        assert keyword.casefold() not in ("oksana", "tariq")


# ---------------------------------------------------------------------------
# Topic quality properties
# ---------------------------------------------------------------------------


def test_unrelated_concepts_produce_distinct_topics_not_one_giant_topic() -> None:
    segments = []
    for i in range(4):
        segments.append(_segment(f"Authentication service outage update number {i}, login failures continue.", "0", float(i * 5), float(i * 5 + 4)))
    base = len(segments) * 5
    for i in range(4):
        segments.append(_segment(f"Holiday party planning discussion number {i}, catering options reviewed.", "1", float(base + i * 5), float(base + i * 5 + 4)))
    result = TranscriptionResult(transcript="ignored", segments=segments)
    analytics = analyze_content(build_timeline(result))
    assert len(analytics.topics) >= 2


def test_source_event_ids_reflect_unique_contributing_events_only() -> None:
    long_turn = (
        "The onboarding checklist needs an update. "
        "We should also confirm the onboarding checklist owner. "
        "Finally, the onboarding checklist rollout date should be set."
    )
    segments = [_segment(long_turn, "0", 0.0, 30.0), _segment("Separate unrelated note about parking.", "1", 30.0, 34.0)]
    result = TranscriptionResult(transcript="ignored", segments=segments)
    analytics = analyze_content(build_timeline(result))
    for topic in analytics.topics:
        assert len(set(topic.source_event_ids)) == len(topic.source_event_ids)
        assert topic.event_count == len(topic.source_event_ids)


# ---------------------------------------------------------------------------
# Semantic segmentation unit tests (no embeddings required)
# ---------------------------------------------------------------------------


def test_segmentation_splits_multiple_sentences_in_one_event() -> None:
    text = "We should review the budget. Then we need to confirm the hiring plan. Finally let's discuss the vendor timeline."
    pieces = _segment_event_text(text)
    assert len(pieces) == 3


def test_segmentation_keeps_short_meaningful_phrases_intact() -> None:
    text = "API integration. Budget approval. Beta program."
    pieces = _segment_event_text(text)
    assert "API integration" in pieces
    assert "Budget approval" in pieces
    assert "Beta program" in pieces


def test_segmentation_merges_pure_filler_fragment_into_neighbor() -> None:
    text = "We should ship this. Right. It is ready."
    pieces = _segment_event_text(text)
    assert not any(piece.strip().casefold() == "right" for piece in pieces)


def test_segmentation_does_not_split_a_normal_short_utterance() -> None:
    pieces = _segment_event_text("Sounds good, let's proceed with the plan.")
    assert len(pieces) == 1


def test_segmentation_fallback_chunks_a_long_unpunctuated_run() -> None:
    long_run = " ".join(f"word{i}" for i in range(80))
    pieces = _segment_event_text(long_run)
    assert len(pieces) > 1
    for piece in pieces:
        assert len(piece.split()) <= 25


def test_is_non_substantive_fragment_true_for_pure_filler() -> None:
    assert _is_non_substantive_fragment("Okay, right, yeah.") is True


def test_is_non_substantive_fragment_false_for_short_real_content() -> None:
    assert _is_non_substantive_fragment("API integration.") is False


# ---------------------------------------------------------------------------
# Keyphrase tests
# ---------------------------------------------------------------------------


def test_meaningful_bigram_can_outrank_generic_unigram() -> None:
    documents = [
        "The production deployment plan needs sign-off.",
        "We reviewed the production deployment checklist yesterday.",
        "Production deployment is scheduled for Thursday night.",
    ]
    ranked = _rank_keywords(documents, top_n=5)
    assert ranked[0] == "production deployment"


def test_stopword_only_phrase_is_never_a_candidate() -> None:
    candidates = _candidate_ngrams("We need to and we should", frozenset())
    assert "we need" not in candidates
    assert "we should" not in candidates


def test_participant_names_are_filtered_from_candidates() -> None:
    candidates = _candidate_ngrams("Tom said the budget looks fine to Tom.", frozenset({"tom"}))
    assert "tom" not in candidates
    assert not any("tom" in c for c in candidates)
    assert "budget" in candidates


def test_greetings_and_generic_filler_are_filtered_from_candidates() -> None:
    candidates = _candidate_ngrams("Hi everyone, thanks for joining, great to see you all today.", frozenset())
    assert "hi" not in candidates
    assert "thanks" not in candidates
    assert "great" not in candidates
    assert "today" not in candidates


def test_meaningful_technical_phrase_survives_candidate_generation() -> None:
    candidates = _candidate_ngrams("The conversational ai model needs retraining.", frozenset())
    assert "conversational ai" in candidates


def test_no_forced_keyword_when_evidence_is_only_noise() -> None:
    ranked = _rank_keywords(["Hi, thanks, okay, right, great, today."], top_n=5)
    assert ranked == []


def test_context_sensitive_noise_word_does_not_block_meaningful_plural_form() -> None:
    # "right" alone is noise, but "access rights" (a different, unfiltered
    # token) must still be candidate-eligible.
    candidates = _candidate_ngrams("We reviewed the access rights for the new system.", frozenset())
    assert "access rights" in candidates
    # "question" alone is noise, but "customer questions" is not filtered.
    candidates2 = _candidate_ngrams("The team addressed customer questions today.", frozenset())
    assert "customer questions" in candidates2


# ---------------------------------------------------------------------------
# Positive-language cue tests
# ---------------------------------------------------------------------------


def test_positive_phrase_matches_are_counted_separately_from_tokens() -> None:
    result = TranscriptionResult(
        transcript="ignored",
        segments=[_segment("Sounds good, thank you for the update, that makes sense.", "0", 0.0, 4.0)],
    )
    analytics = analyze_content(build_timeline(result))
    speaker = analytics.positive_language.per_speaker[0]
    assert speaker.positive_phrase_count == 3
    assert set(speaker.matched_phrases) == {"sounds good", "thank you", "makes sense"}
    # Token-level proportion is unaffected by phrase counting (still based
    # purely on individual POSITIVE_LEXICON token matches).
    assert speaker.positive_token_count >= 1


def test_zero_positive_phrases_is_a_valid_zero_not_fabricated() -> None:
    result = TranscriptionResult(
        transcript="ignored",
        segments=[_segment("The server returned an error during processing.", "0", 0.0, 4.0)],
    )
    analytics = analyze_content(build_timeline(result))
    speaker = analytics.positive_language.per_speaker[0]
    assert speaker.positive_phrase_count == 0
    assert speaker.matched_phrases == ()


def test_phrase_and_token_counts_do_not_double_count_each_other() -> None:
    # "thank you" contains "thank", which is ALSO an individual
    # POSITIVE_LEXICON token -- both signals may legitimately fire, but
    # positive_phrase_count must count the PHRASE occurrence exactly once
    # per occurrence, independent of the token-level count.
    result = TranscriptionResult(
        transcript="ignored",
        segments=[_segment("Thank you. Thank you. Thank you.", "0", 0.0, 4.0)],
    )
    analytics = analyze_content(build_timeline(result))
    speaker = analytics.positive_language.per_speaker[0]
    assert speaker.positive_phrase_count == 3
    assert speaker.positive_token_count == 3  # "thank" token x3, independent metric


def test_meeting_level_positive_phrase_count_sums_across_speakers() -> None:
    result = TranscriptionResult(
        transcript="ignored",
        segments=[
            _segment("Well done on the release.", "0", 0.0, 4.0),
            _segment("Great work, sounds good to me.", "1", 4.0, 8.0),
        ],
    )
    analytics = analyze_content(build_timeline(result))
    # "well done" + "great work" + "sounds good" = 3 phrase matches total.
    assert analytics.positive_language.meeting_positive_phrase_count == 3


# ---------------------------------------------------------------------------
# No-crash / tokenizer sanity for Hinglish (already partially covered
# upstream; kept here as a Phase 3.2-local sanity check on the new
# candidate-generation path specifically)
# ---------------------------------------------------------------------------


def test_candidate_ngrams_do_not_crash_on_hinglish_text() -> None:
    candidates = _candidate_ngrams("Yeh release date next Monday tak ho jaana chahiye, theek hai?", frozenset())
    assert isinstance(candidates, list)
    assert "release" in candidates or "release date" in candidates
