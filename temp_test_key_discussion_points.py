"""Temporary local smoke test for key discussion point extraction.

Usage:
    python temp_test_key_discussion_points.py
"""

from __future__ import annotations

import json
import re

from summarization.base_summarizer import KeyDiscussionPointInput
from summarization.llm_summarizer import LLMSummarizer


SAMPLE_CLEANED_TRANSCRIPT = """Speaker 1 [00:00 - 00:08]
The backend transcription pipeline is stable, and we should keep Phase 1 unchanged.

Speaker 2 [00:08 - 00:18]
The Streamlit UI still needs better progress logging because failures are hard to see.

Speaker 1 [00:18 - 00:30]
For Phase 2, we are building transcript cleanup, summary generation, decisions, action items, and discussion point extraction as separate modules.

Speaker 2 [00:30 - 00:40]
We should keep the LLM layer provider-neutral so OpenAI, Claude, Gemini, or local models can be added later."""


class DemoKeyDiscussionPointClient:
    """Deterministic demo client with the same interface as a future LLM client."""

    def generate(self, prompt: str, *, system_prompt: str | None = None) -> str:
        transcript = self._extract_transcript(prompt)
        return json.dumps(self._extract_key_discussion_points(transcript), indent=2)

    def _extract_transcript(self, prompt: str) -> str:
        match = re.search(r"<transcript>\s*(.*?)\s*</transcript>", prompt, re.S)
        if not match:
            return prompt
        return match.group(1).strip()

    def _extract_key_discussion_points(
        self,
        transcript: str,
    ) -> dict[str, object]:
        return {
            "key_discussion_points": [
                {
                    "point": (
                        "Phase 1 transcription is stable and should remain "
                        "unchanged."
                    ),
                    "speakers": ["Speaker 1"],
                    "timestamp": "00:00 - 00:08",
                },
                {
                    "point": (
                        "The Streamlit UI needs clearer progress logging and "
                        "failure visibility."
                    ),
                    "speakers": ["Speaker 2"],
                    "timestamp": "00:08 - 00:18",
                },
                {
                    "point": (
                        "Phase 2 is being split into modular cleanup, summary, "
                        "decision, action item, and discussion point extractors."
                    ),
                    "speakers": ["Speaker 1"],
                    "timestamp": "00:18 - 00:30",
                },
                {
                    "point": (
                        "The LLM layer should stay provider-neutral for future "
                        "OpenAI, Claude, Gemini, or local model integration."
                    ),
                    "speakers": ["Speaker 2"],
                    "timestamp": "00:30 - 00:40",
                },
            ]
        }


def main() -> int:
    summarizer = LLMSummarizer(llm_client=DemoKeyDiscussionPointClient())
    result = summarizer.extract_key_discussion_points(
        KeyDiscussionPointInput(transcript_text=SAMPLE_CLEANED_TRANSCRIPT)
    )

    print("Sample input:")
    print(SAMPLE_CLEANED_TRANSCRIPT)
    print("\nKey discussion point extraction output:")
    print(json.dumps(
        {
            "key_discussion_points": [
                {
                    "point": item.point,
                    "speakers": item.speakers,
                    "timestamp": item.timestamp,
                }
                for item in result.key_discussion_points
            ]
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
