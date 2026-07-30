"""Generalized quality contracts for deterministic MoM post-processing."""

from __future__ import annotations

import unittest

from ml_mom.experimental.formatter_utils import extract_action_parts
from ml_mom.experimental.mom_formatter import (
    action_sentence_has_evidence,
    cleanup_decision_output,
    decision_from_sentence,
    is_decision_only_statement,
    is_non_action_utterance,
    semantic_topic_key,
    summary_opening_phrase,
)


class DecisionQualityTests(unittest.TestCase):
    def test_imperative_approval_becomes_passive_decision(self) -> None:
        self.assertEqual(
            decision_from_sentence(
                "Approve the current build for client demonstration.",
                "Build",
            ),
            "Current Build for Client Demonstration was approved.",
        )

    def test_go_ahead_has_natural_generic_outcome(self) -> None:
        self.assertEqual(
            decision_from_sentence("Go ahead.", "Sprint Planning"),
            "The team approved proceeding with the planned work.",
        )

    def test_finalize_imperative_is_not_approved(self) -> None:
        self.assertEqual(
            decision_from_sentence(
                "Finalize Sprint 5 feature list.",
                "Sprint Feature List",
            ),
            "Sprint 5 Feature List was finalized.",
        )

    def test_duplicate_outcome_phrase_is_removed(self) -> None:
        self.assertEqual(
            cleanup_decision_output(
                "The current build has been approved approved as the agreed direction."
            ),
            "The current build was approved.",
        )


class ActionQualityTests(unittest.TestCase):
    def test_greetings_and_introductions_are_not_actions(self) -> None:
        rejected = (
            "Good morning everyone.",
            "Let's begin.",
            "Today's meeting is about the annual event.",
            "We'll review the client project status.",
            "The project status was discussed.",
            "What is the current deployment status?",
        )
        for sentence in rejected:
            with self.subTest(sentence=sentence):
                self.assertTrue(is_non_action_utterance(sentence))
                self.assertFalse(action_sentence_has_evidence(sentence, "Action_Item"))

    def test_clear_commitments_are_actions(self) -> None:
        accepted = (
            "I will prepare the documentation by Friday.",
            "Please deploy the backend.",
            "Can you verify the dashboard?",
            "Ravi must complete regression testing.",
        )
        for sentence in accepted:
            with self.subTest(sentence=sentence):
                self.assertTrue(action_sentence_has_evidence(sentence, "Action_Item"))

    def test_action_tasks_are_concise(self) -> None:
        self.assertEqual(
            extract_action_parts("I will prepare the documentation by Friday.", "Asha").task,
            "Prepare the documentation.",
        )
        self.assertEqual(
            extract_action_parts("Can you verify the dashboard?", "Asha").task,
            "Verify the dashboard.",
        )

    def test_scope_decision_is_not_an_action(self) -> None:
        sentence = "Automatic email delivery will be included in this release."
        self.assertTrue(is_decision_only_statement(sentence))
        self.assertFalse(action_sentence_has_evidence(sentence, "Action_Item"))


class DiscussionAndSummaryQualityTests(unittest.TestCase):
    def test_overlapping_email_topics_share_a_semantic_key(self) -> None:
        self.assertEqual(
            semantic_topic_key("Automatic Email Delivery"),
            semantic_topic_key("Email Notifications"),
        )

    def test_summary_opener_is_deterministic_and_varied(self) -> None:
        topic_sets = (
            ["Backend Integration"],
            ["College Event Logistics"],
            ["Client Requirements"],
            ["Model Performance", "UI Improvements"],
            ["Release Readiness", "Testing"],
            ["Budget Planning", "Venue Logistics"],
        )
        first_pass = [summary_opening_phrase(topics) for topics in topic_sets]
        second_pass = [summary_opening_phrase(topics) for topics in topic_sets]
        self.assertEqual(first_pass, second_pass)
        self.assertGreaterEqual(len(set(first_pass)), 3)


if __name__ == "__main__":
    unittest.main()
