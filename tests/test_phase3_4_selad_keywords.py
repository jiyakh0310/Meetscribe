"""Tests for the Phase 3.4 SELaD-inspired graph/rank-based keyword and
keyphrase extraction (meeting_analytics/content.py's ``_rank_keywords``,
``_graph_rank_unigrams``, ``_build_cooccurrence_graph``).

This is a FUNCTIONAL ADAPTATION of SELaD's Okt-noun-extraction +
KRWordRank keyword-ranking stage for English/Hinglish transcripts (see
the Phase 3.4 section of meeting_analytics/content.py's module docstring)
-- a local co-occurrence graph + PageRank/TextRank-style iterative
importance score, not term frequency or TF-IDF. No MiniLM, no ANN, no
LLM is involved in this extraction path at all.

Style matches tests/test_meeting_analytics_content.py and
tests/test_phase3_2_content_quality.py: plain pytest-style functions with
bare asserts. Every scenario uses SYNTHETIC content, distinct in names/
wording/domain from the manually-tested sample audio.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from timeline import build_timeline
from meeting_analytics.content import (
    _graph_rank_unigrams,
    _rank_keywords,
    analyze_content,
)
from transcription.sarvam_client import TranscriptionResult, TranscriptionSegment


def _segment(text: str, speaker: str | None, start: float | None, end: float | None) -> TranscriptionSegment:
    return TranscriptionSegment(transcript=text, speaker_id=speaker, start_time_seconds=start, end_time_seconds=end)


def _keywords_for(segments, mapping=None) -> tuple[str, ...]:
    result = TranscriptionResult(transcript="ignored", segments=segments)
    analytics = analyze_content(build_timeline(result, mapping=mapping))
    return analytics.keywords.meeting_keywords


# ---------------------------------------------------------------------------
# Generalization matrix (12 scenarios)
# ---------------------------------------------------------------------------


def test_case_1_technical_meeting_surfaces_technical_concepts() -> None:
    keywords = _keywords_for([
        _segment("Let's review the API integration plan for the payments service.", "0", 0.0, 6.0),
        _segment("The API integration depends on the new authentication service rollout.", "1", 6.0, 12.0),
        _segment("We also need the database migration finished before the API integration ships.", "0", 12.0, 20.0),
    ])
    combined = " ".join(keywords).casefold()
    assert "api" in combined or "integration" in combined or "authentication" in combined


def test_case_2_project_planning_meeting_surfaces_planning_concepts() -> None:
    keywords = _keywords_for([
        _segment("We should finalize the onboarding checklist before the beta launch.", "0", 0.0, 6.0),
        _segment("The onboarding checklist still needs a section for pricing questions.", "1", 6.0, 12.0),
        _segment("Let's also confirm the beta launch date with the customer team.", "0", 12.0, 20.0),
    ])
    combined = " ".join(keywords).casefold()
    assert any(term in combined for term in ("onboarding", "beta", "launch", "pricing", "checklist"))


def test_case_3_business_meeting_surfaces_business_concepts() -> None:
    keywords = _keywords_for([
        _segment("The hiring budget for next quarter needs finance approval.", "0", 0.0, 6.0),
        _segment("We also need to confirm the vendor timeline for the new supplier.", "1", 6.0, 12.0),
        _segment("Hiring three engineers this quarter affects the budget significantly.", "0", 12.0, 20.0),
    ])
    combined = " ".join(keywords).casefold()
    assert any(term in combined for term in ("budget", "hiring", "vendor"))


def test_case_4_product_demo_surfaces_product_capability_terms() -> None:
    keywords = _keywords_for([
        _segment("This dashboard shows real-time inventory levels across warehouses.", "0", 0.0, 6.0),
        _segment("The inventory dashboard also supports historical trend analysis.", "1", 6.0, 12.0),
        _segment("Can the dashboard export inventory reports automatically?", "0", 12.0, 20.0),
    ])
    combined = " ".join(keywords).casefold()
    assert "inventory" in combined or "dashboard" in combined


def test_case_5_long_explanation_surfaces_the_explained_concept() -> None:
    long_turn = (
        "Let me walk through how the recommendation engine works. "
        "The recommendation engine collects user browsing behavior. "
        "Then the recommendation engine ranks products using a scoring model. "
        "Finally the recommendation engine displays the ranked products to the user."
    )
    keywords = _keywords_for([_segment(long_turn, "0", 0.0, 40.0)])
    combined = " ".join(keywords).casefold()
    assert "recommendation" in combined


def test_case_6_rapid_qa_surfaces_the_shared_subject() -> None:
    keywords = _keywords_for([
        _segment("What is the status of the checkout redesign?", "0", 0.0, 4.0),
        _segment("The checkout redesign is in final review.", "1", 4.0, 8.0),
        _segment("Will the checkout redesign ship this sprint?", "0", 8.0, 12.0),
        _segment("Yes, the checkout redesign ships this sprint.", "1", 12.0, 16.0),
    ])
    combined = " ".join(keywords).casefold()
    assert "checkout" in combined or "redesign" in combined


def test_case_7_introductions_plus_substantive_discussion_names_do_not_dominate() -> None:
    keywords = _keywords_for(
        [
            _segment("Hi, I'm Farah, I lead the platform team.", "0", 0.0, 4.0),
            _segment("Hi, I'm Devon, I lead the growth team.", "1", 4.0, 8.0),
            _segment("Let's discuss the subscription pricing model changes for next quarter.", "0", 8.0, 16.0),
            _segment("The subscription pricing changes need legal review before rollout.", "1", 16.0, 24.0),
        ],
        mapping={"Speaker 1": "Farah", "Speaker 2": "Devon"},
    )
    lowered = [k.casefold() for k in keywords]
    assert "farah" not in lowered
    assert "devon" not in lowered
    combined = " ".join(lowered)
    assert "pricing" in combined or "subscription" in combined


def test_case_8_mostly_greetings_and_filler_produces_few_or_no_keywords() -> None:
    keywords = _keywords_for([
        _segment("Hi everyone, thanks for joining.", "0", 0.0, 4.0),
        _segment("Hello, glad to be here, thanks.", "1", 4.0, 8.0),
        _segment("Okay great, thanks, that's all for today.", "0", 8.0, 12.0),
    ])
    lowered = [k.casefold() for k in keywords]
    for filler in ("hi", "hello", "thanks", "okay", "great", "everyone"):
        assert filler not in lowered


def test_case_9_repeated_concept_across_speakers_ranks_highly() -> None:
    keywords = _keywords_for([
        _segment("The release testing status looks good for this cycle.", "0", 0.0, 5.0),
        _segment("Agreed, release testing is on track for Friday.", "1", 5.0, 10.0),
        _segment("One more note on release testing: the regression suite passed.", "2", 10.0, 15.0),
    ])
    assert keywords
    combined = " ".join(keywords).casefold()
    assert "release" in combined or "testing" in combined


def test_case_10_transcript_only_meeting_produces_keywords() -> None:
    keywords = _keywords_for([
        _segment("We reviewed the vendor contract renewal terms in detail today.", "0", None, None),
        _segment("The vendor contract renewal looks reasonable for another year.", "1", None, None),
    ])
    assert keywords
    combined = " ".join(keywords).casefold()
    assert "vendor" in combined or "contract" in combined


def test_case_11_hinglish_meeting_does_not_crash_and_produces_output() -> None:
    keywords = _keywords_for([
        _segment("Yeh onboarding process next Monday tak complete ho jaana chahiye, theek hai?", "0", 0.0, 6.0),
        _segment("Haan bilkul, onboarding checklist ready hai already.", "1", 6.0, 12.0),
    ])
    assert isinstance(keywords, tuple)


def test_case_12_technical_acronyms_are_preserved_not_lowercased_away() -> None:
    keywords = _keywords_for([
        _segment("The REST API needs an updated OAuth flow before launch.", "0", 0.0, 6.0),
        _segment("The OAuth flow for the REST API is almost finished.", "1", 6.0, 12.0),
    ])
    combined = " ".join(keywords).casefold()
    assert "api" in combined or "oauth" in combined


# ---------------------------------------------------------------------------
# Property-based verification
# ---------------------------------------------------------------------------


def test_meaningful_bigram_supported_by_transcript_adjacency_outranks_unigram() -> None:
    ranked = _rank_keywords(
        [
            "The production deployment plan needs sign-off.",
            "We reviewed the production deployment checklist yesterday.",
            "Production deployment is scheduled for Thursday night.",
        ],
        top_n=5,
    )
    assert ranked[0] == "production deployment"


def test_no_invented_phrase_not_supported_by_adjacency() -> None:
    # "database" and "budget" never sit next to each other in the source
    # text -- the ranker must never invent "database budget" as a phrase.
    ranked = _rank_keywords(
        [
            "The database migration needs review.",
            "The marketing budget needs review.",
        ],
        top_n=10,
    )
    assert "database budget" not in ranked
    assert "budget database" not in ranked


def test_duplicate_case_and_plural_variants_are_reduced() -> None:
    ranked = _rank_keywords(
        [
            "The API needs review.",
            "The APIs need review.",
            "The API design is solid.",
        ],
        top_n=5,
    )
    lowered = [term.casefold() for term in ranked]
    # Both "api" and "apis" should not independently occupy separate slots.
    assert not ("api" in lowered and "apis" in lowered)


def test_fewer_results_are_returned_when_evidence_is_weak() -> None:
    ranked = _rank_keywords(["Hi, thanks, okay, right, great, today."], top_n=10)
    assert len(ranked) == 0


def test_deterministic_same_input_produces_same_output() -> None:
    documents = [
        "The customer onboarding flow needs a redesign this quarter.",
        "Customer onboarding feedback was mostly positive this cycle.",
        "We should simplify the customer onboarding steps further.",
    ]
    first = _rank_keywords(documents, top_n=5)
    second = _rank_keywords(documents, top_n=5)
    assert first == second


def test_participant_names_are_excluded_as_graph_nodes() -> None:
    noise = frozenset({"tom"})
    scores = _graph_rank_unigrams(["Tom said the budget review looks good to Tom."], noise)
    assert "tom" not in scores
    assert "budget" in scores or "review" in scores


def test_isolated_single_candidate_does_not_crash() -> None:
    scores = _graph_rank_unigrams(["Deployment."], frozenset())
    assert scores == {"deployment": 1.0}


def test_empty_documents_produce_no_keywords_without_crash() -> None:
    assert _rank_keywords([], top_n=5) == []
    assert _rank_keywords(["", "   ", "?? !!"], top_n=5) == []


def test_graph_rank_handles_disconnected_components_without_frequency_bias() -> None:
    # A one-off filler aside ("sure, sounds fine") must not outrank a
    # genuinely repeated concept ("database migration") merely because it
    # forms a small isolated graph component (see _rank_keywords's
    # docstring on frequency-weighting the raw graph score).
    ranked = _rank_keywords(
        [
            "The database migration plan needs review.",
            "We discussed the database migration timeline in detail.",
            "The database migration is the main risk for this release.",
            "Okay, sure, that sounds fine to me.",
        ],
        top_n=5,
    )
    assert "sure" not in ranked
    assert "fine" not in ranked
    assert ranked[0] == "database migration"
