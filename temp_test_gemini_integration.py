"""Temporary local smoke test for real Gemini Phase 2 integration.

Usage:
    python temp_test_gemini_integration.py
"""

from __future__ import annotations

import json

from llm_clients.gemini_client import GeminiClient, GeminiClientError
from summarization.base_summarizer import MeetingSummaryInput, TranscriptCleanupInput
from summarization.llm_summarizer import (
    LLMSummarizer,
    MeetingSummaryError,
    TranscriptCleanupError,
)


SAMPLE_TRANSCRIPT = """Speaker 1 [00:00 - 00:05]
hello everyone lets review the phase two plan

Speaker 2 [00:05 - 00:14]
the backend transcription is working but we need clean summary generation

Speaker 1 [00:14 - 00:24]
yes we will use gemini for cleanup and meeting summary first then add other llms later"""


def main() -> int:
    try:
        gemini_client = GeminiClient()
        summarizer = LLMSummarizer(llm_client=gemini_client)

        cleanup_result = summarizer.cleanup_transcript(
            TranscriptCleanupInput(transcript_text=SAMPLE_TRANSCRIPT)
        )
        meeting_summary = summarizer.generate_meeting_summary(
            MeetingSummaryInput(transcript_text=cleanup_result.cleaned_transcript)
        )
    except (GeminiClientError, TranscriptCleanupError, MeetingSummaryError) as exc:
        print(f"Error: {exc}")
        return 1

    print("Sample input:")
    print(SAMPLE_TRANSCRIPT)
    print("\nGemini cleaned transcript:")
    print(cleanup_result.cleaned_transcript)
    print("\nGemini meeting summary:")
    print(json.dumps(
        {
            "title": meeting_summary.title,
            "short_summary": meeting_summary.short_summary,
            "detailed_summary": meeting_summary.detailed_summary,
            "topics_discussed": meeting_summary.topics_discussed,
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
