"""Temporary local smoke test for meeting summary generation.

Usage:
    python temp_test_meeting_summary.py
"""

from __future__ import annotations

import json
import re

from summarization.base_summarizer import MeetingSummaryInput
from summarization.llm_summarizer import LLMSummarizer


SAMPLE_CLEANED_TRANSCRIPT = """Speaker 1 [00:00 - 00:05]
Hello everyone, let's start the API reliability meeting.

Speaker 2 [00:05 - 00:14]
The upload flow is working, but the Streamlit UI needs clearer progress logging.

Speaker 1 [00:14 - 00:22]
We should keep the backend pipeline unchanged and improve UI error visibility.

Speaker 2 [00:22 - 00:30]
The next focus is transcript cleanup and summary generation for Phase 2."""


class DemoMeetingSummaryClient:
    """Deterministic demo client with the same interface as a future LLM client."""

    def generate(self, prompt: str, *, system_prompt: str | None = None) -> str:
        transcript = self._extract_transcript(prompt)
        return json.dumps(self._summarize(transcript), indent=2)

    def _extract_transcript(self, prompt: str) -> str:
        match = re.search(r"<transcript>\s*(.*?)\s*</transcript>", prompt, re.S)
        if not match:
            return prompt
        return match.group(1).strip()

    def _summarize(self, transcript: str) -> dict[str, object]:
        return {
            "title": "API Reliability and Phase 2 Planning",
            "short_summary": (
                "The meeting reviewed the working backend transcription pipeline "
                "and focused on improving Streamlit UI observability before moving "
                "into Phase 2 summarization features."
            ),
            "detailed_summary": (
                "The team confirmed that the upload and transcription backend are "
                "working correctly. They discussed that the Streamlit UI needs "
                "clearer progress logging and better exception visibility while "
                "keeping the backend pipeline unchanged. The meeting then shifted "
                "toward Phase 2, beginning with transcript cleanup and meeting "
                "summary generation."
            ),
            "topics_discussed": [
                "API reliability",
                "Streamlit progress logging",
                "UI error visibility",
                "Transcript cleanup",
                "Meeting summary generation",
            ],
        }


def main() -> int:
    summarizer = LLMSummarizer(llm_client=DemoMeetingSummaryClient())
    summary = summarizer.generate_meeting_summary(
        MeetingSummaryInput(transcript_text=SAMPLE_CLEANED_TRANSCRIPT)
    )

    print("Sample input:")
    print(SAMPLE_CLEANED_TRANSCRIPT)
    print("\nMeeting summary output:")
    print(json.dumps(
        {
            "title": summary.title,
            "short_summary": summary.short_summary,
            "detailed_summary": summary.detailed_summary,
            "topics_discussed": summary.topics_discussed,
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
