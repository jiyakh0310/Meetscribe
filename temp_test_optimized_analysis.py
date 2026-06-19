"""Temporary local smoke test for the optimized single-call analysis pipeline.

Usage:
    python temp_test_optimized_analysis.py
"""

from __future__ import annotations

import json

from summarization.llm_summarizer import LLMSummarizer


SAMPLE_TRANSCRIPT = """Speaker 1 [00:00 - 00:08]
hello team lets review phase two of meetscribe today

Speaker 2 [00:08 - 00:18]
the transcription pipeline is working now so we should keep phase one unchanged

Speaker 1 [00:18 - 00:32]
agreed rahul should connect the streamlit ui to this analysis pipeline by friday"""


class CountingFullAnalysisClient:
    """Deterministic demo client that proves analyze_meeting uses one LLM call."""

    def __init__(self) -> None:
        self.call_count = 0

    def generate(self, prompt: str, *, system_prompt: str | None = None) -> str:
        self.call_count += 1
        return json.dumps(
            {
                "cleaned_transcript": (
                    "Speaker 1 [00:00 - 00:08]\n"
                    "Hello team, let's review Phase 2 of MeetScribe today.\n\n"
                    "Speaker 2 [00:08 - 00:18]\n"
                    "The transcription pipeline is working now, so we should keep "
                    "Phase 1 unchanged.\n\n"
                    "Speaker 1 [00:18 - 00:32]\n"
                    "Agreed. Rahul should connect the Streamlit UI to this "
                    "analysis pipeline by Friday."
                ),
                "summary": {
                    "title": "MeetScribe Phase 2 Review",
                    "short_summary": (
                        "The team reviewed the stable transcription pipeline and "
                        "confirmed the next UI integration step."
                    ),
                    "detailed_summary": (
                        "The meeting confirmed that the Phase 1 transcription "
                        "pipeline is working and should remain unchanged. The team "
                        "also identified the Streamlit UI connection to the analysis "
                        "pipeline as the next follow-up activity."
                    ),
                },
                "topics_discussed": [
                    "Phase 1 transcription stability",
                    "Phase 2 analysis pipeline",
                    "Streamlit UI integration",
                ],
                "key_discussion_points": [
                    {
                        "point": "Phase 1 transcription is stable and should remain unchanged.",
                        "speakers": ["Speaker 2"],
                        "timestamp": "00:08 - 00:18",
                    }
                ],
                "decisions": [
                    {
                        "decision": "Keep Phase 1 unchanged.",
                        "owner": "Speaker 2",
                        "timestamp": "00:08 - 00:18",
                        "confidence": "high",
                    }
                ],
                "action_items": [
                    {
                        "task": "Connect the Streamlit UI to the analysis pipeline.",
                        "owner": "Rahul",
                        "due_date": "Friday",
                        "timestamp": "00:18 - 00:32",
                        "status": "open",
                    }
                ],
            },
            indent=2,
        )


def main() -> int:
    client = CountingFullAnalysisClient()
    summarizer = LLMSummarizer(llm_client=client)

    first_result = summarizer.analyze_meeting(SAMPLE_TRANSCRIPT)
    second_result = summarizer.analyze_meeting(SAMPLE_TRANSCRIPT)

    print("LLM calls:", client.call_count)
    print("Cache reused:", first_result is second_result)
    print(json.dumps(first_result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
