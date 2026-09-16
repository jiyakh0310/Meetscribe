"""Tests for Phase 3.1 (real-audio grounding and speaker integrity) fixes.

Style matches the other tests/test_*.py files in this repo: plain
pytest-style functions with bare asserts, no unittest.TestCase.

Every scenario here uses SYNTHETIC content that is intentionally
different from the manually-tested sample audio (no "Tom"/"Sumit"/"Will"/
"JIYA"/"RingCentral"/"Deployment"/"Beta Program" names or sentences), per
the Phase 3.1 anti-overfitting requirement: fixes must generalize across
different meetings, speaker counts, names, and domains.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_mom.mom_generator import PredictionRecord
from ml_mom.transcript_parser import TranscriptTurn
from transcription.sarvam_client import TranscriptionResult, TranscriptionSegment


# --- Bug 1: participant propagation must not drop generic-but-real speakers ----


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


def test_partially_renamed_speakers_all_appear_in_transcript_turn_participants() -> None:
    from app.main import participants_from_transcript_turns

    # Four speakers, only two renamed -- Speaker 2 and Speaker 4 remain
    # generic. All four spoke and must all be listed.
    turns = [
        _turn(1, "Nora", "Let's review the quarterly budget numbers."),
        _turn(2, "Speaker 2", "I have some concerns about the marketing spend."),
        _turn(3, "Devraj", "We can revisit that after the demo."),
        _turn(4, "Speaker 4", "Sounds good to me."),
    ]
    participants = participants_from_transcript_turns(turns)
    assert participants == ["Nora", "Speaker 2", "Devraj", "Speaker 4"]


def test_all_generic_speakers_are_still_listed_when_none_renamed() -> None:
    from app.main import participants_from_transcript_turns

    turns = [
        _turn(1, "Speaker 1", "Good morning everyone."),
        _turn(2, "Speaker 2", "Morning, let's get started."),
    ]
    participants = participants_from_transcript_turns(turns)
    assert participants == ["Speaker 1", "Speaker 2"]


def test_duplicate_speaker_turns_are_deduplicated_case_insensitively() -> None:
    from app.main import participants_from_transcript_turns

    turns = [
        _turn(1, "Anya", "First point."),
        _turn(2, "anya", "Second point, same speaker different casing."),
        _turn(3, "Speaker 3", "Third speaker chimes in."),
    ]
    participants = participants_from_transcript_turns(turns)
    assert participants == ["Anya", "Speaker 3"]


def test_infer_participants_keeps_unrenamed_generic_speaker() -> None:
    from ml_mom.experimental.mom_formatter import infer_participants

    records = [
        PredictionRecord(sentence="We reviewed the roadmap.", speaker="Fatima", timestamp=0, predicted_label="Discussion", confidence_score=0.9),
        PredictionRecord(sentence="Any objections?", speaker="Speaker 2", timestamp=5, predicted_label="Discussion", confidence_score=0.9),
        PredictionRecord(sentence="None from me.", speaker="Speaker 3", timestamp=10, predicted_label="Discussion", confidence_score=0.9),
    ]
    participants = infer_participants(records)
    assert participants == ["Fatima", "Speaker 2", "Speaker 3"]


def test_infer_participants_still_skips_blank_and_placeholder_speakers() -> None:
    from ml_mom.experimental.mom_formatter import infer_participants

    records = [
        PredictionRecord(sentence="Hello.", speaker="", timestamp=0, predicted_label="Discussion", confidence_score=0.9),
        PredictionRecord(sentence="Hi.", speaker="-", timestamp=1, predicted_label="Discussion", confidence_score=0.9),
        PredictionRecord(sentence="Ready?", speaker="Speaker 1", timestamp=2, predicted_label="Discussion", confidence_score=0.9),
    ]
    participants = infer_participants(records)
    assert participants == ["Speaker 1"]


# --- Bug 2: first-speaking timestamp must use the stable generic key -----------


def test_default_speaker_label_matches_generic_key_regardless_of_prior_rename() -> None:
    from app.main import default_speaker_label, speaker_label

    segments = [
        TranscriptionSegment("Kickoff remarks.", "0", 0.0, 3.0),
        TranscriptionSegment("Follow-up remarks.", "1", 3.0, 6.0),
        TranscriptionSegment("Closing remarks.", "2", 6.0, 9.0),
    ]
    # Simulate a mapping where "Speaker 1" (segment id "0") has already been
    # renamed to "Wren" -- the exact condition that broke the old
    # `speaker_label(s) == label` comparison for generic-key lookups.
    mapping = {"Speaker 1": "Wren"}

    labels = ["Speaker 1", "Speaker 2", "Speaker 3"]
    for label in labels:
        matched = [s for s in segments if default_speaker_label(s) == label]
        assert len(matched) == 1, f"expected exactly one segment for {label}"

    # The renamed speaker's display label differs from its generic key --
    # this is exactly why comparing against speaker_label() (not
    # default_speaker_label()) breaks the generic-key lookup.
    renamed_segment = segments[0]
    assert speaker_label(renamed_segment, mapping) == "Wren"
    assert default_speaker_label(renamed_segment) == "Speaker 1"


def test_first_speaker_begins_at_exactly_zero_is_not_treated_as_untimed() -> None:
    from app.main import default_speaker_label, format_timestamp

    segments = [
        TranscriptionSegment("Starts right at zero.", "0", 0.0, 2.0),
        TranscriptionSegment("Starts a bit later.", "1", 5.0, 7.0),
    ]
    first_speaker_segments = [s for s in segments if default_speaker_label(s) == "Speaker 1"]
    first_timestamp = format_timestamp(first_speaker_segments[0].start_time_seconds if first_speaker_segments else None)
    assert first_timestamp == "00:00"
    assert first_timestamp != "--:--"


def test_first_speaker_begins_at_one_second() -> None:
    from app.main import default_speaker_label, format_timestamp

    segments = [TranscriptionSegment("Starts at one second.", "0", 1.0, 3.0)]
    matched = [s for s in segments if default_speaker_label(s) == "Speaker 1"]
    assert format_timestamp(matched[0].start_time_seconds) == "00:01"


def test_second_speaker_begins_later_than_first() -> None:
    from app.main import default_speaker_label, format_timestamp

    segments = [
        TranscriptionSegment("First speaker opens.", "0", 0.0, 4.0),
        TranscriptionSegment("Second speaker responds.", "1", 4.0, 9.0),
    ]
    second_speaker_segments = [s for s in segments if default_speaker_label(s) == "Speaker 2"]
    assert format_timestamp(second_speaker_segments[0].start_time_seconds) == "00:04"


def test_untimed_speaker_reports_dashes_not_a_fabricated_time() -> None:
    from app.main import default_speaker_label, format_timestamp

    segments = [TranscriptionSegment("No timestamps available for this upload.", "0", None, None)]
    matched = [s for s in segments if default_speaker_label(s) == "Speaker 1"]
    assert format_timestamp(matched[0].start_time_seconds if matched else None) == "--:--"


def test_renamed_speaker_still_resolves_first_timestamp_via_generic_key() -> None:
    from app.main import default_speaker_label, format_timestamp

    # Three speakers: first two already renamed by a prior review pass,
    # third left generic -- all three must still resolve correctly.
    segments = [
        TranscriptionSegment("Opening from speaker one.", "0", 0.0, 3.0),
        TranscriptionSegment("Response from speaker two.", "1", 3.0, 6.0),
        TranscriptionSegment("Comment from speaker three.", "2", 6.0, 9.0),
    ]
    expected_firsts = {"Speaker 1": "00:00", "Speaker 2": "00:03", "Speaker 3": "00:06"}
    for label, expected in expected_firsts.items():
        matched = [s for s in segments if default_speaker_label(s) == label]
        assert format_timestamp(matched[0].start_time_seconds) == expected


# --- Bug 3: Decision evidence gate must reject discussion/question framing ----


def test_question_about_a_process_is_not_a_decision() -> None:
    from ml_mom.experimental.mom_formatter import has_explicit_decision_evidence

    assert has_explicit_decision_evidence("How should we onboard the new vendor?") is False


def test_possible_technology_choice_is_not_a_decision() -> None:
    from ml_mom.experimental.mom_formatter import has_explicit_decision_evidence

    assert has_explicit_decision_evidence("Maybe we could use a message queue for this.") is False


def test_discussion_of_a_budget_is_not_a_decision() -> None:
    from ml_mom.experimental.mom_formatter import has_explicit_decision_evidence

    assert has_explicit_decision_evidence("We discussed the marketing budget for next quarter.") is False


def test_explanation_of_existing_process_is_not_a_decision() -> None:
    from ml_mom.experimental.mom_formatter import has_explicit_decision_evidence

    assert has_explicit_decision_evidence("The onboarding process works like this: fill the form, then wait for approval.") is False


def test_evaluation_planned_for_later_is_not_a_decision() -> None:
    from ml_mom.experimental.mom_formatter import has_explicit_decision_evidence

    assert has_explicit_decision_evidence("We need to evaluate the two vendors before choosing one.") is False


def test_recommendation_without_acceptance_is_not_a_decision() -> None:
    from ml_mom.experimental.mom_formatter import has_explicit_decision_evidence

    assert has_explicit_decision_evidence("I recommend we switch to the new supplier.") is False


def test_explicit_agreement_is_a_decision() -> None:
    from ml_mom.experimental.mom_formatter import has_explicit_decision_evidence

    assert has_explicit_decision_evidence("We agreed to onboard the new vendor starting next month.") is True


def test_explicit_approval_is_a_decision() -> None:
    from ml_mom.experimental.mom_formatter import has_explicit_decision_evidence

    assert has_explicit_decision_evidence("The budget increase was approved by the finance team.") is True


def test_explicit_selection_is_a_decision() -> None:
    from ml_mom.experimental.mom_formatter import has_explicit_decision_evidence

    assert has_explicit_decision_evidence("We selected the second vendor for the contract.") is True


def test_explicit_commitment_is_a_decision() -> None:
    from ml_mom.experimental.mom_formatter import has_explicit_decision_evidence

    assert has_explicit_decision_evidence("We will proceed with the message queue architecture.") is True


def test_supported_hinglish_agreement_is_still_a_decision() -> None:
    from ml_mom.experimental.mom_formatter import has_explicit_decision_evidence

    assert has_explicit_decision_evidence("Hum log ne decided kar liya hai ki naya vendor onboard karenge.") is True


def test_deployment_discussion_question_is_not_a_decision() -> None:
    # Generic regression for the exact bug class reported: a question about
    # deploying/shipping something must not be accepted as a Decision just
    # because it contains a decision-adjacent verb.
    from ml_mom.experimental.mom_formatter import has_explicit_decision_evidence

    assert has_explicit_decision_evidence("How do we ship this feature into production?") is False


def test_sentence_can_both_discuss_and_report_a_commitment() -> None:
    # A sentence may use discussion framing AND state a commitment in the
    # same breath -- the discussion marker alone must not veto it when a
    # commitment phrase is also present.
    from ml_mom.experimental.mom_formatter import has_explicit_decision_evidence

    assert has_explicit_decision_evidence("We discussed the options and decided to go with the second vendor.") is True


# --- Bug 4: objective fallback must not fabricate a "required follow-up" ------


def test_objective_fallback_does_not_invent_a_required_follow_up() -> None:
    from ml_mom.experimental.mom_formatter import (
        ExtractedEntities,
        TopicBlueprint,
        TopicCluster,
        TopicEvidence,
        build_objective,
    )

    cluster = TopicCluster(
        title="Shared Calendar Tool",
        records=[],
        contexts=["Participants asked about the shared calendar tool."],
        representative_sentence="Participants asked about the shared calendar tool.",
        centroid_embedding=[],
    )
    evidence = TopicEvidence(
        topic="Shared Calendar Tool",
        cluster=cluster,
        discussion_sentences=["Participants asked about the shared calendar tool."],
    )
    blueprint = TopicBlueprint(
        title="Shared Calendar Tool",
        evidence=evidence,
        entities=ExtractedEntities(),
        importance_score=1.0,
    )
    objective = build_objective([blueprint])
    assert "follow-up" not in objective.lower()
    assert "shared calendar tool" in objective.lower()


# --- Bug 5: evidence attribution must preserve every genuine contributor -----


def test_evidence_attribution_preserves_multiple_contributing_speakers() -> None:
    from app.main import evidence_for_report_item

    transcript = (
        "Priya [00:00 - 00:05]\n"
        "We should review the onboarding checklist and note the equipment requests early.\n\n"
        "Speaker 3 [00:05 - 00:10]\n"
        "The onboarding checklist needs a section for equipment requests.\n\n"
        "Marcus [00:10 - 00:15]\n"
        "Unrelated note about the parking garage closure today.\n"
    )
    speakers, timestamp = evidence_for_report_item(
        transcript,
        "Onboarding Checklist: the team reviewed equipment requests for the onboarding checklist.",
    )
    assert speakers == ["Priya", "Speaker 3"]
    assert timestamp == "00:00 - 00:05"


def test_evidence_attribution_keeps_generic_speaker_when_it_is_the_only_match() -> None:
    from app.main import evidence_for_report_item

    transcript = (
        "Speaker 2 [00:20 - 00:25]\n"
        "The invoicing system migration timeline was reviewed today.\n\n"
        "Speaker 4 [00:30 - 00:35]\n"
        "Completely unrelated remark about the coffee machine.\n"
    )
    speakers, timestamp = evidence_for_report_item(
        transcript,
        "Invoicing System Migration: the team reviewed the migration timeline.",
    )
    assert speakers == ["Speaker 2"]
    assert timestamp == "00:20 - 00:25"


def test_evidence_attribution_excludes_weakly_related_turns() -> None:
    from app.main import evidence_for_report_item

    transcript = (
        "Aisha [00:00 - 00:05]\n"
        "We reviewed the customer support ticket backlog in detail today.\n\n"
        "Ben [00:40 - 00:45]\n"
        "Lunch is at noon.\n"
    )
    speakers, timestamp = evidence_for_report_item(
        transcript,
        "Customer Support Ticket Backlog: the team reviewed the ticket backlog in detail.",
    )
    assert speakers == ["Aisha"]
    assert "Ben" not in speakers
