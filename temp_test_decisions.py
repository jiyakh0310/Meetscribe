"""Temporary local smoke test for decision extraction.

Usage:
    python temp_test_decisions.py
"""

from __future__ import annotations

import json
import re

from summarization.base_summarizer import DecisionInput
from summarization.llm_summarizer import LLMSummarizer


SAMPLE_CLEANED_TRANSCRIPT = """Speaker 1 [00:00 - 00:08]
The backend transcription pipeline is working, so we should keep it unchanged.

Speaker 2 [00:08 - 00:18]
Agreed. We will use the existing Phase 1 transcript output as the input for Phase 2.

Speaker 1 [00:18 - 00:28]
Let's not connect OpenAI, Claude, or Gemini yet. For now, we will use mock clients.

Speaker 2 [00:28 - 00:36]
Yes, the next implementation step will be key discussion point extraction later."""


class DemoDecisionClient:
    """Deterministic demo client with the same interface as a future LLM client."""

    def generate(self, prompt: str, *, system_prompt: str | None = None) -> str:
        transcript = self._extract_transcript(prompt)
        return json.dumps(self._extract_decisions(transcript), indent=2)

    def _extract_transcript(self, prompt: str) -> str:
        match = re.search(r"<transcript>\s*(.*?)\s*</transcript>", prompt, re.S)
        if not match:
            return prompt
        return match.group(1).strip()

    def _extract_decisions(self, transcript: str) -> dict[str, object]:
        return {
            "decisions": [
                {
                    "decision": "Keep the backend transcription pipeline unchanged.",
                    "owner": "Speaker 1",
                    "timestamp": "00:00 - 00:08",
                    "confidence": "high",
                },
                {
                    "decision": (
                        "Use the existing Phase 1 transcript output as the input "
                        "for Phase 2."
                    ),
                    "owner": "Speaker 2",
                    "timestamp": "00:08 - 00:18",
                    "confidence": "high",
                },
                {
                    "decision": (
                        "Use mock clients for now instead of connecting OpenAI, "
                        "Claude, or Gemini."
                    ),
                    "owner": "Speaker 1",
                    "timestamp": "00:18 - 00:28",
                    "confidence": "high",
                },
            ]
        }


def main() -> int:
    summarizer = LLMSummarizer(llm_client=DemoDecisionClient())
    result = summarizer.extract_decisions(
        DecisionInput(transcript_text=SAMPLE_CLEANED_TRANSCRIPT)
    )

    print("Sample input:")
    print(SAMPLE_CLEANED_TRANSCRIPT)
    print("\nDecision extraction output:")
    print(json.dumps(
        {
            "decisions": [
                {
                    "decision": item.decision,
                    "owner": item.owner,
                    "timestamp": item.timestamp,
                    "confidence": item.confidence,
                }
                for item in result.decisions
            ]
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
