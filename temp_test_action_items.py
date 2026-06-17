"""Temporary local smoke test for action item extraction.

Usage:
    python temp_test_action_items.py
"""

from __future__ import annotations

import json
import re

from summarization.base_summarizer import ActionItemInput
from summarization.llm_summarizer import LLMSummarizer


SAMPLE_CLEANED_TRANSCRIPT = """Speaker 1 [00:00 - 00:06]
Let's review the Streamlit transcript screen today.

Speaker 2 [00:06 - 00:15]
Rahul, please add detailed logging for the Sarvam API call by tomorrow.

Speaker 1 [00:15 - 00:24]
I will update the UI to show exceptions directly before the next demo.

Speaker 2 [00:24 - 00:32]
After that, we should verify the action item extraction test output."""


class DemoActionItemClient:
    """Deterministic demo client with the same interface as a future LLM client."""

    def generate(self, prompt: str, *, system_prompt: str | None = None) -> str:
        transcript = self._extract_transcript(prompt)
        return json.dumps(self._extract_action_items(transcript), indent=2)

    def _extract_transcript(self, prompt: str) -> str:
        match = re.search(r"<transcript>\s*(.*?)\s*</transcript>", prompt, re.S)
        if not match:
            return prompt
        return match.group(1).strip()

    def _extract_action_items(self, transcript: str) -> dict[str, object]:
        return {
            "action_items": [
                {
                    "task": "Add detailed logging for the Sarvam API call.",
                    "owner": "Rahul",
                    "due_date": "tomorrow",
                    "timestamp": "00:06 - 00:15",
                    "status": "open",
                },
                {
                    "task": "Update the UI to show exceptions directly.",
                    "owner": "Speaker 1",
                    "due_date": "before the next demo",
                    "timestamp": "00:15 - 00:24",
                    "status": "open",
                },
                {
                    "task": "Verify the action item extraction test output.",
                    "owner": None,
                    "due_date": None,
                    "timestamp": "00:24 - 00:32",
                    "status": "open",
                },
            ]
        }


def main() -> int:
    summarizer = LLMSummarizer(llm_client=DemoActionItemClient())
    result = summarizer.extract_action_items(
        ActionItemInput(transcript_text=SAMPLE_CLEANED_TRANSCRIPT)
    )

    print("Sample input:")
    print(SAMPLE_CLEANED_TRANSCRIPT)
    print("\nAction item extraction output:")
    print(json.dumps(
        {
            "action_items": [
                {
                    "task": item.task,
                    "owner": item.owner,
                    "due_date": item.due_date,
                    "timestamp": item.timestamp,
                    "status": item.status,
                }
                for item in result.action_items
            ]
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
