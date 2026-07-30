"""Contract tests for the optional local Gemma post-formatter layer."""

from __future__ import annotations

import unittest
from unittest.mock import patch
from urllib.error import URLError

from ml_mom.experimental.mom_formatter import ExperimentalActionItem, ExperimentalMom
from ml_mom.local_gemma_rewriter import LocalGemmaRewriter, _permitted_markdown


def sample_mom() -> ExperimentalMom:
    return ExperimentalMom(
        meeting_title="Release Review",
        meeting_date="30 July 2026",
        objective="Review release readiness.",
        participants=["Asha", "Ravi"],
        summary="Team review release and risks.",
        discussion_points=["Team discuss PDF layout.", "Ravi review final checks."],
        decisions=["Release remains scheduled for Friday."],
        action_items=[
            ExperimentalActionItem(
                owner="Ravi",
                task="Complete final checks",
                deadline="31 July 2026",
            )
        ],
        pending_items=["Confirm deployment window."],
        information=["Version 4 is frozen."],
    )


class LocalGemmaRewriterTests(unittest.TestCase):
    def test_disabled_returns_deterministic_object(self) -> None:
        source = sample_mom()
        result = LocalGemmaRewriter(enabled=False).rewrite(source)
        self.assertIs(result.mom, source)
        self.assertFalse(result.applied)

    def test_failure_falls_back_without_raising(self) -> None:
        source = sample_mom()
        rewriter = LocalGemmaRewriter(enabled=True, timeout_seconds=0.01)
        with patch.object(rewriter, "_get_json", side_effect=URLError("offline")):
            result = rewriter.rewrite(source)
        self.assertIs(result.mom, source)
        self.assertFalse(result.applied)
        self.assertIn("offline", result.error or "")

    def test_only_summary_and_discussion_can_change(self) -> None:
        source = sample_mom()
        response = """## Meeting Information
unchanged
## Executive Summary
The team reviewed the release plan and associated risks.
## Discussion
- The team discussed improvements to the PDF layout.
- Ravi reviewed the final verification steps.
## Decisions
- malicious changed decision
## Action Items
| Owner | Task | Deadline |
| --- | --- | --- |
| Other | Invent work | Tomorrow |
"""
        rewriter = LocalGemmaRewriter(enabled=True)
        with (
            patch.object(rewriter, "_select_local_gemma_model", return_value="gemma3:4b"),
            patch.object(
                rewriter,
                "_post_json",
                return_value={"message": {"content": response}},
            ),
        ):
            result = rewriter.rewrite(source)

        self.assertTrue(result.applied)
        self.assertNotEqual(result.mom.summary, source.summary)
        self.assertNotEqual(result.mom.discussion_points, source.discussion_points)
        self.assertEqual(result.mom.decisions, source.decisions)
        self.assertEqual(result.mom.action_items, source.action_items)
        self.assertEqual(result.mom.meeting_date, source.meeting_date)
        self.assertEqual(result.mom.participants, source.participants)
        self.assertEqual(result.mom.pending_items, source.pending_items)

    def test_prompt_payload_excludes_transcript_and_internal_ml_data(self) -> None:
        payload = _permitted_markdown(sample_mom())
        self.assertIn("## Executive Summary", payload)
        self.assertIn("## Discussion", payload)
        self.assertIn("## Decisions", payload)
        self.assertIn("## Action Items", payload)
        self.assertNotIn("objective", payload.casefold())
        self.assertNotIn("pending", payload.casefold())
        self.assertNotIn("embedding", payload.casefold())
        self.assertNotIn("speaker diarization", payload.casefold())


if __name__ == "__main__":
    unittest.main()
