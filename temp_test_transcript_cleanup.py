"""Temporary local smoke test for transcript cleanup.

Usage:
    python temp_test_transcript_cleanup.py
"""

from __future__ import annotations

import re

from summarization.base_summarizer import TranscriptCleanupInput
from summarization.llm_summarizer import LLMSummarizer


SAMPLE_TRANSCRIPT = """Speaker 1 [00:00 - 00:05]
hello everyone lets start teh meeting

Speaker 2 [00:05 - 00:10]
yes we should discuss the api issue and action item for rahul

Speaker 1 [00:10 - 00:15]
okay lets fix it today and then we will test again"""


class DemoCleanupClient:
    """Deterministic demo client with the same interface as a future LLM client."""

    def generate(self, prompt: str, *, system_prompt: str | None = None) -> str:
        transcript = self._extract_transcript(prompt)
        return self._cleanup_transcript(transcript)

    def _extract_transcript(self, prompt: str) -> str:
        match = re.search(r"<transcript>\s*(.*?)\s*</transcript>", prompt, re.S)
        if not match:
            return prompt
        return match.group(1).strip()

    def _cleanup_transcript(self, transcript: str) -> str:
        replacements = {
            "lets": "let's",
            "teh": "the",
            "api": "API",
            "rahul": "Rahul",
            "okay": "Okay",
        }

        cleaned_lines: list[str] = []
        for line in transcript.splitlines():
            stripped = line.strip()
            if not stripped:
                cleaned_lines.append("")
                continue

            if stripped.lower().startswith("speaker "):
                cleaned_lines.append(stripped)
                continue

            text = stripped
            for source, target in replacements.items():
                text = re.sub(rf"\b{source}\b", target, text, flags=re.I)

            text = text[0].upper() + text[1:] if text else text
            if text[-1] not in ".?!":
                text = f"{text}."
            cleaned_lines.append(text)

        return "\n".join(cleaned_lines)


def main() -> int:
    summarizer = LLMSummarizer(llm_client=DemoCleanupClient())
    result = summarizer.cleanup_transcript(
        TranscriptCleanupInput(transcript_text=SAMPLE_TRANSCRIPT)
    )

    print("Original transcript:")
    print(result.original_transcript)
    print("\nCleaned transcript:")
    print(result.cleaned_transcript)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
