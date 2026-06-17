"""Temporary local smoke test for the full Gemini meeting analysis pipeline.

Usage:
    python temp_test_full_pipeline.py
"""

from __future__ import annotations

import json

from llm_clients.gemini_client import GeminiClient, GeminiClientError
from summarization.base_summarizer import MeetingAnalysisResult
from summarization.llm_summarizer import (
    ActionItemExtractionError,
    DecisionExtractionError,
    KeyDiscussionPointExtractionError,
    LLMSummarizer,
    MeetingSummaryError,
    TranscriptCleanupError,
)


SAMPLE_TRANSCRIPT = """Speaker 1 [00:00 - 00:08]
hello team lets review phase two of meetscribe today

Speaker 2 [00:08 - 00:18]
the transcription pipeline is working now so we should keep phase one unchanged

Speaker 1 [00:18 - 00:32]
agreed we will use the speaker wise transcript as input for cleanup summary decisions action items and key points

Speaker 2 [00:32 - 00:45]
rahul should connect the streamlit ui to this analysis pipeline by friday

Speaker 1 [00:45 - 00:56]
for llm providers we will use gemini now and keep the architecture open for openai claude and local models later"""


def print_mom_output(analysis: MeetingAnalysisResult) -> None:
    output = analysis.to_dict()
    summary = output["summary"]

    print("Minutes of Meeting")
    print("==================")
    print(f"\nTitle: {summary['title']}")

    print("\nShort Summary")
    print("-------------")
    print(summary["short_summary"])

    print("\nDetailed Summary")
    print("----------------")
    print(summary["detailed_summary"])

    print("\nTopics Discussed")
    print("----------------")
    for topic in summary["topics_discussed"]:
        print(f"- {topic}")

    print("\nKey Discussion Points")
    print("---------------------")
    for item in output["key_discussion_points"]:
        speakers = ", ".join(item["speakers"])
        timestamp = item["timestamp"] or "N/A"
        print(f"- [{timestamp}] {item['point']} ({speakers})")

    print("\nDecisions")
    print("---------")
    for item in output["decisions"]:
        owner = item["owner"] or "N/A"
        timestamp = item["timestamp"] or "N/A"
        print(
            f"- [{timestamp}] {item['decision']} "
            f"(owner: {owner}, confidence: {item['confidence']})"
        )

    print("\nAction Items")
    print("------------")
    for item in output["action_items"]:
        owner = item["owner"] or "N/A"
        due_date = item["due_date"] or "N/A"
        timestamp = item["timestamp"] or "N/A"
        print(
            f"- [{timestamp}] {item['task']} "
            f"(owner: {owner}, due: {due_date}, status: {item['status']})"
        )

    print("\nCleaned Transcript")
    print("------------------")
    print(output["cleaned_transcript"])

    print("\nStructured Output")
    print("-----------------")
    print(json.dumps(output, indent=2))


def main() -> int:
    try:
        gemini_client = GeminiClient()
        summarizer = LLMSummarizer(llm_client=gemini_client)
        analysis = summarizer.analyze_meeting(SAMPLE_TRANSCRIPT)
    except (
        GeminiClientError,
        TranscriptCleanupError,
        MeetingSummaryError,
        KeyDiscussionPointExtractionError,
        DecisionExtractionError,
        ActionItemExtractionError,
    ) as exc:
        print(f"Error: {exc}")
        return 1

    print("Sample input:")
    print(SAMPLE_TRANSCRIPT)
    print()
    print_mom_output(analysis)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
