"""Phase 3.1 mandatory synthetic regression matrix.

Eight scenarios (technical / planning / business / Hinglish / informational
demo / multi-speaker / single-speaker / transcript-only), each built from
SYNTHETIC content distinct from the manually-tested sample audio. Exercises
the deterministic Decision-evidence gate and the speaker/participant/evidence
helpers fixed in Phase 3.1, at the same function-level granularity already
used by tests/test_deterministic_formatter_quality.py (no ANN/MiniLM
inference is required to validate these deterministic code paths).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_mom.experimental.mom_formatter import has_explicit_decision_evidence
from ml_mom.transcript_parser import TranscriptTurn


def _turn(turn_id: int, speaker: str, text: str) -> TranscriptTurn:
    return TranscriptTurn(
        turn_id=turn_id,
        speaker_raw=speaker,
        speaker_normalized=speaker,
        timestamp_raw=None,
        timestamp_seconds=None,
        text=text,
        sentence_list=[text],
    )


# --- Case A: technical meeting (architecture discussion + explicit Decision + Action + questions) ---


def test_case_a_technical_meeting_separates_discussion_decision_and_questions() -> None:
    from app.main import participants_from_transcript_turns

    turns = [
        _turn(1, "Elena", "Should we split the monolith into services now or wait?"),
        _turn(2, "Farid", "We agreed to split the monolith into services starting next sprint."),
        _turn(3, "Elena", "What is the current API latency baseline?"),
    ]
    assert participants_from_transcript_turns(turns) == ["Elena", "Farid"]
    assert has_explicit_decision_evidence("Should we split the monolith into services now or wait?") is False
    assert has_explicit_decision_evidence("We agreed to split the monolith into services starting next sprint.") is True
    assert has_explicit_decision_evidence("What is the current API latency baseline?") is False


# --- Case B: planning meeting (suggestions/possibilities, NO final decision) ---


def test_case_b_planning_meeting_produces_no_fabricated_decision() -> None:
    candidates = (
        "Maybe we could push the launch to next quarter.",
        "We should consider a phased rollout.",
        "Could we use a smaller pilot group first?",
        "What are the available options for the rollout plan?",
    )
    for sentence in candidates:
        assert has_explicit_decision_evidence(sentence) is False, sentence


# --- Case C: business meeting (explicit approval + budget discussion + real Decision) ---


def test_case_c_business_meeting_retains_the_real_decision() -> None:
    assert has_explicit_decision_evidence("We discussed the marketing budget for the next two quarters.") is False
    assert has_explicit_decision_evidence("The additional marketing budget was approved by the finance committee.") is True


# --- Case D: Hinglish meeting (suggestion/question vs actual agreement + assigned action) ---


def test_case_d_hinglish_meeting_distinguishes_suggestion_from_agreement() -> None:
    assert has_explicit_decision_evidence("Kya hum naya CRM tool try kar sakte hain?") is False
    assert has_explicit_decision_evidence("Humne decided kar liya hai ki naya CRM tool use karenge.") is True


# --- Case E: informational/demo meeting (mostly explanations/questions, expect empty Decision) ---


def test_case_e_informational_demo_meeting_yields_no_decisions() -> None:
    candidates = (
        "This is how the export button currently works.",
        "How does the retry logic handle timeouts?",
        "The dashboard refreshes every five minutes.",
        "Can someone explain the caching layer?",
        "We walked through the current onboarding flow.",
    )
    for sentence in candidates:
        assert has_explicit_decision_evidence(sentence) is False, sentence


# --- Case F: multi-speaker metadata (>=4 speakers, some renamed, some generic) ---


def test_case_f_four_speakers_partially_renamed_all_appear() -> None:
    from app.main import participants_from_transcript_turns

    turns = [
        _turn(1, "Grace", "Let's start with the hiring update."),
        _turn(2, "Speaker 2", "We have three candidates in the final round."),
        _turn(3, "Hassan", "I can do the reference checks this week."),
        _turn(4, "Speaker 4", "I'll sit in on the next interview."),
    ]
    assert participants_from_transcript_turns(turns) == ["Grace", "Speaker 2", "Hassan", "Speaker 4"]


# --- Case G: single speaker (no crash, correct participant metadata) ---


def test_case_g_single_speaker_meeting_is_handled_without_crash() -> None:
    from app.main import participants_from_transcript_turns

    turns = [_turn(1, "Ingrid", "This is a solo status update covering last week's progress.")]
    assert participants_from_transcript_turns(turns) == ["Ingrid"]


# --- Case H: transcript-only (no timestamps; speaker mapping/grounding still work) ---


def test_case_h_transcript_only_meeting_grounds_decisions_without_timestamps() -> None:
    from app.main import evidence_for_report_item

    transcript = (
        "Speaker 1:\nWe reviewed the vendor contract renewal terms in detail.\n\n"
        "Speaker 2:\nWe agreed to renew the vendor contract for another year.\n"
    )
    speakers, timestamp = evidence_for_report_item(
        transcript,
        "Vendor Contract Renewal: the team agreed to renew the vendor contract for another year.",
    )
    assert "Speaker 2" in speakers
    assert timestamp is None
    assert has_explicit_decision_evidence("We agreed to renew the vendor contract for another year.") is True
    assert has_explicit_decision_evidence("We reviewed the vendor contract renewal terms in detail.") is False
