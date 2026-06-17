"""LLM-backed transcript cleanup orchestration."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Protocol

from summarization.base_summarizer import (
    ActionItem,
    ActionItemExtractionResult,
    ActionItemInput,
    BaseSummarizer,
    Decision,
    DecisionExtractionResult,
    DecisionInput,
    KeyDiscussionPoint,
    KeyDiscussionPointExtractionResult,
    KeyDiscussionPointInput,
    MeetingAnalysisResult,
    MeetingSummary,
    MeetingSummaryInput,
    TranscriptCleanupInput,
    TranscriptCleanupResult,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLEANUP_PROMPT_PATH = (
    PROJECT_ROOT / "prompts" / "cleanup" / "transcript_cleanup.md"
)
DEFAULT_SUMMARY_PROMPT_PATH = (
    PROJECT_ROOT / "prompts" / "summarization" / "meeting_summary.md"
)
DEFAULT_ACTION_ITEMS_PROMPT_PATH = (
    PROJECT_ROOT / "prompts" / "extraction" / "action_items.md"
)
DEFAULT_DECISIONS_PROMPT_PATH = (
    PROJECT_ROOT / "prompts" / "extraction" / "decisions.md"
)
DEFAULT_KEY_DISCUSSION_POINTS_PROMPT_PATH = (
    PROJECT_ROOT / "prompts" / "extraction" / "key_discussion_points.md"
)

SPEAKER_HEADER_PATTERN = re.compile(
    r"^\s*Speaker\s+\S+(?:\s+\[[^\]]+\])?\s*:?\s*$",
    re.IGNORECASE,
)


class TranscriptCleanupError(Exception):
    """Raised when transcript cleanup fails validation."""


class MeetingSummaryError(Exception):
    """Raised when meeting summary generation fails validation."""


class ActionItemExtractionError(Exception):
    """Raised when action item extraction fails validation."""


class DecisionExtractionError(Exception):
    """Raised when decision extraction fails validation."""


class KeyDiscussionPointExtractionError(Exception):
    """Raised when key discussion point extraction fails validation."""


class LLMClientProtocol(Protocol):
    """Minimal client interface for OpenAI, Claude, Gemini, or local LLMs."""

    def generate(self, prompt: str, *, system_prompt: str | None = None) -> str:
        """Return model-generated text for the given prompt."""


class LLMSummarizer(BaseSummarizer):
    """Transcript cleanup implementation using an injected LLM client."""

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        *,
        cleanup_prompt_path: str | Path = DEFAULT_CLEANUP_PROMPT_PATH,
        summary_prompt_path: str | Path = DEFAULT_SUMMARY_PROMPT_PATH,
        action_items_prompt_path: str | Path = DEFAULT_ACTION_ITEMS_PROMPT_PATH,
        decisions_prompt_path: str | Path = DEFAULT_DECISIONS_PROMPT_PATH,
        key_discussion_points_prompt_path: str
        | Path = DEFAULT_KEY_DISCUSSION_POINTS_PROMPT_PATH,
        system_prompt: str | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._cleanup_prompt_path = Path(cleanup_prompt_path)
        self._summary_prompt_path = Path(summary_prompt_path)
        self._action_items_prompt_path = Path(action_items_prompt_path)
        self._decisions_prompt_path = Path(decisions_prompt_path)
        self._key_discussion_points_prompt_path = Path(
            key_discussion_points_prompt_path
        )
        self._system_prompt = system_prompt

    def cleanup_transcript(
        self,
        cleanup_input: TranscriptCleanupInput | str,
    ) -> TranscriptCleanupResult:
        normalized_input = self._normalize_cleanup_input(cleanup_input)
        transcript_text = normalized_input.transcript_text.strip()
        if not transcript_text:
            raise TranscriptCleanupError("Transcript cleanup input is empty.")

        prompt = self._build_cleanup_prompt(transcript_text)
        response = self._llm_client.generate(
            prompt,
            system_prompt=self._system_prompt,
        )
        cleaned_transcript = self._extract_cleaned_transcript(response)
        self._validate_cleaned_transcript(
            original_transcript=transcript_text,
            cleaned_transcript=cleaned_transcript,
        )

        return TranscriptCleanupResult(
            cleaned_transcript=cleaned_transcript,
            original_transcript=transcript_text,
        )

    def generate_meeting_summary(
        self,
        summary_input: MeetingSummaryInput | str,
    ) -> MeetingSummary:
        normalized_input = self._normalize_summary_input(summary_input)
        transcript_text = normalized_input.transcript_text.strip()
        if not transcript_text:
            raise MeetingSummaryError("Meeting summary input is empty.")

        prompt = self._build_summary_prompt(transcript_text)
        response = self._llm_client.generate(
            prompt,
            system_prompt=self._system_prompt,
        )
        payload = self._extract_summary_payload(response)
        summary = MeetingSummary(
            title=str(payload.get("title", "")).strip(),
            short_summary=str(payload.get("short_summary", "")).strip(),
            detailed_summary=str(payload.get("detailed_summary", "")).strip(),
            topics_discussed=[
                str(topic).strip()
                for topic in payload.get("topics_discussed", [])
                if str(topic).strip()
            ],
        )
        self._validate_meeting_summary(summary)
        return summary

    def extract_action_items(
        self,
        action_item_input: ActionItemInput | str,
    ) -> ActionItemExtractionResult:
        normalized_input = self._normalize_action_item_input(action_item_input)
        transcript_text = normalized_input.transcript_text.strip()
        if not transcript_text:
            raise ActionItemExtractionError("Action item extraction input is empty.")

        prompt = self._build_action_items_prompt(transcript_text)
        response = self._llm_client.generate(
            prompt,
            system_prompt=self._system_prompt,
        )
        payload = self._extract_action_items_payload(response)
        result = self._parse_action_items_payload(payload)
        self._validate_action_item_result(result)
        return result

    def extract_decisions(
        self,
        decision_input: DecisionInput | str,
    ) -> DecisionExtractionResult:
        normalized_input = self._normalize_decision_input(decision_input)
        transcript_text = normalized_input.transcript_text.strip()
        if not transcript_text:
            raise DecisionExtractionError("Decision extraction input is empty.")

        prompt = self._build_decisions_prompt(transcript_text)
        response = self._llm_client.generate(
            prompt,
            system_prompt=self._system_prompt,
        )
        payload = self._extract_decisions_payload(response)
        result = self._parse_decisions_payload(payload)
        self._validate_decision_result(result)
        return result

    def extract_key_discussion_points(
        self,
        key_point_input: KeyDiscussionPointInput | str,
    ) -> KeyDiscussionPointExtractionResult:
        normalized_input = self._normalize_key_point_input(key_point_input)
        transcript_text = normalized_input.transcript_text.strip()
        if not transcript_text:
            raise KeyDiscussionPointExtractionError(
                "Key discussion point extraction input is empty."
            )

        prompt = self._build_key_discussion_points_prompt(transcript_text)
        response = self._llm_client.generate(
            prompt,
            system_prompt=self._system_prompt,
        )
        payload = self._extract_key_discussion_points_payload(response)
        result = self._parse_key_discussion_points_payload(payload)
        self._validate_key_discussion_point_result(result)
        return result

    def analyze_meeting(self, transcript_text: str) -> MeetingAnalysisResult:
        cleanup_result = self.cleanup_transcript(
            TranscriptCleanupInput(transcript_text=transcript_text)
        )
        cleaned_transcript = cleanup_result.cleaned_transcript

        summary = self.generate_meeting_summary(
            MeetingSummaryInput(transcript_text=cleaned_transcript)
        )
        key_discussion_points = self.extract_key_discussion_points(
            KeyDiscussionPointInput(transcript_text=cleaned_transcript)
        )
        decisions = self.extract_decisions(
            DecisionInput(transcript_text=cleaned_transcript)
        )
        action_items = self.extract_action_items(
            ActionItemInput(transcript_text=cleaned_transcript)
        )

        return MeetingAnalysisResult(
            cleaned_transcript=cleaned_transcript,
            summary=summary,
            key_discussion_points=key_discussion_points.key_discussion_points,
            decisions=decisions.decisions,
            action_items=action_items.action_items,
        )

    def _normalize_cleanup_input(
        self,
        cleanup_input: TranscriptCleanupInput | str,
    ) -> TranscriptCleanupInput:
        if isinstance(cleanup_input, TranscriptCleanupInput):
            return cleanup_input

        return TranscriptCleanupInput(transcript_text=str(cleanup_input))

    def _normalize_summary_input(
        self,
        summary_input: MeetingSummaryInput | str,
    ) -> MeetingSummaryInput:
        if isinstance(summary_input, MeetingSummaryInput):
            return summary_input

        return MeetingSummaryInput(transcript_text=str(summary_input))

    def _normalize_action_item_input(
        self,
        action_item_input: ActionItemInput | str,
    ) -> ActionItemInput:
        if isinstance(action_item_input, ActionItemInput):
            return action_item_input

        return ActionItemInput(transcript_text=str(action_item_input))

    def _normalize_decision_input(
        self,
        decision_input: DecisionInput | str,
    ) -> DecisionInput:
        if isinstance(decision_input, DecisionInput):
            return decision_input

        return DecisionInput(transcript_text=str(decision_input))

    def _normalize_key_point_input(
        self,
        key_point_input: KeyDiscussionPointInput | str,
    ) -> KeyDiscussionPointInput:
        if isinstance(key_point_input, KeyDiscussionPointInput):
            return key_point_input

        return KeyDiscussionPointInput(transcript_text=str(key_point_input))

    def _build_cleanup_prompt(self, transcript_text: str) -> str:
        template = self._cleanup_prompt_path.read_text(encoding="utf-8")
        return template.replace("{transcript}", transcript_text)

    def _build_summary_prompt(self, transcript_text: str) -> str:
        template = self._summary_prompt_path.read_text(encoding="utf-8")
        return template.replace("{transcript}", transcript_text)

    def _build_action_items_prompt(self, transcript_text: str) -> str:
        template = self._action_items_prompt_path.read_text(encoding="utf-8")
        return template.replace("{transcript}", transcript_text)

    def _build_decisions_prompt(self, transcript_text: str) -> str:
        template = self._decisions_prompt_path.read_text(encoding="utf-8")
        return template.replace("{transcript}", transcript_text)

    def _build_key_discussion_points_prompt(self, transcript_text: str) -> str:
        template = self._key_discussion_points_prompt_path.read_text(encoding="utf-8")
        return template.replace("{transcript}", transcript_text)

    def _extract_cleaned_transcript(self, response: str) -> str:
        text = response.strip()
        if not text:
            raise TranscriptCleanupError("LLM returned an empty cleanup response.")

        text = self._strip_code_fence(text)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return text.strip()

        cleaned_transcript = str(payload.get("cleaned_transcript", "")).strip()
        if not cleaned_transcript:
            raise TranscriptCleanupError(
                "LLM JSON response did not include cleaned_transcript."
            )

        return cleaned_transcript

    def _extract_summary_payload(self, response: str) -> dict[str, object]:
        text = response.strip()
        if not text:
            raise MeetingSummaryError("LLM returned an empty summary response.")

        text = self._strip_code_fence(text)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise MeetingSummaryError(
                "LLM summary response was not valid JSON."
            ) from exc

        if not isinstance(payload, dict):
            raise MeetingSummaryError("LLM summary response must be a JSON object.")

        return payload

    def _extract_key_discussion_points_payload(
        self,
        response: str,
    ) -> dict[str, object]:
        text = response.strip()
        if not text:
            raise KeyDiscussionPointExtractionError(
                "LLM returned an empty key discussion point response."
            )

        text = self._strip_code_fence(text)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise KeyDiscussionPointExtractionError(
                "LLM key discussion point response was not valid JSON."
            ) from exc

        if not isinstance(payload, dict):
            raise KeyDiscussionPointExtractionError(
                "LLM key discussion point response must be a JSON object."
            )

        return payload

    def _extract_decisions_payload(self, response: str) -> dict[str, object]:
        text = response.strip()
        if not text:
            raise DecisionExtractionError("LLM returned an empty decision response.")

        text = self._strip_code_fence(text)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DecisionExtractionError(
                "LLM decision response was not valid JSON."
            ) from exc

        if not isinstance(payload, dict):
            raise DecisionExtractionError(
                "LLM decision response must be a JSON object."
            )

        return payload

    def _extract_action_items_payload(self, response: str) -> dict[str, object]:
        text = response.strip()
        if not text:
            raise ActionItemExtractionError(
                "LLM returned an empty action item response."
            )

        text = self._strip_code_fence(text)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ActionItemExtractionError(
                "LLM action item response was not valid JSON."
            ) from exc

        if not isinstance(payload, dict):
            raise ActionItemExtractionError(
                "LLM action item response must be a JSON object."
            )

        return payload

    def _parse_action_items_payload(
        self,
        payload: dict[str, object],
    ) -> ActionItemExtractionResult:
        raw_items = payload.get("action_items")
        if not isinstance(raw_items, list):
            raise ActionItemExtractionError(
                "Action item response must include an action_items list."
            )

        action_items: list[ActionItem] = []
        for index, raw_item in enumerate(raw_items, start=1):
            if not isinstance(raw_item, dict):
                raise ActionItemExtractionError(
                    f"Action item {index} must be a JSON object."
                )

            action_items.append(
                ActionItem(
                    task=str(raw_item.get("task", "")).strip(),
                    owner=self._optional_string(raw_item.get("owner")),
                    due_date=self._optional_string(raw_item.get("due_date")),
                    timestamp=self._optional_string(raw_item.get("timestamp")),
                    status=str(raw_item.get("status", "")).strip(),
                )
            )

        return ActionItemExtractionResult(action_items=action_items)

    def _parse_decisions_payload(
        self,
        payload: dict[str, object],
    ) -> DecisionExtractionResult:
        raw_decisions = payload.get("decisions")
        if not isinstance(raw_decisions, list):
            raise DecisionExtractionError(
                "Decision response must include a decisions list."
            )

        decisions: list[Decision] = []
        for index, raw_decision in enumerate(raw_decisions, start=1):
            if not isinstance(raw_decision, dict):
                raise DecisionExtractionError(
                    f"Decision {index} must be a JSON object."
                )

            decisions.append(
                Decision(
                    decision=str(raw_decision.get("decision", "")).strip(),
                    owner=self._optional_string(raw_decision.get("owner")),
                    timestamp=self._optional_string(raw_decision.get("timestamp")),
                    confidence=str(raw_decision.get("confidence", "")).strip(),
                )
            )

        return DecisionExtractionResult(decisions=decisions)

    def _parse_key_discussion_points_payload(
        self,
        payload: dict[str, object],
    ) -> KeyDiscussionPointExtractionResult:
        raw_points = payload.get("key_discussion_points")
        if not isinstance(raw_points, list):
            raise KeyDiscussionPointExtractionError(
                "Key discussion point response must include a "
                "key_discussion_points list."
            )

        key_discussion_points: list[KeyDiscussionPoint] = []
        for index, raw_point in enumerate(raw_points, start=1):
            if not isinstance(raw_point, dict):
                raise KeyDiscussionPointExtractionError(
                    f"Key discussion point {index} must be a JSON object."
                )

            raw_speakers = raw_point.get("speakers", [])
            if not isinstance(raw_speakers, list):
                raise KeyDiscussionPointExtractionError(
                    f"Key discussion point {index} speakers must be a list."
                )

            key_discussion_points.append(
                KeyDiscussionPoint(
                    point=str(raw_point.get("point", "")).strip(),
                    speakers=[
                        str(speaker).strip()
                        for speaker in raw_speakers
                        if str(speaker).strip()
                    ],
                    timestamp=self._optional_string(raw_point.get("timestamp")),
                )
            )

        return KeyDiscussionPointExtractionResult(
            key_discussion_points=key_discussion_points
        )

    def _strip_code_fence(self, text: str) -> str:
        if not text.startswith("```"):
            return text

        lines = text.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()

        return text

    def _validate_cleaned_transcript(
        self,
        *,
        original_transcript: str,
        cleaned_transcript: str,
    ) -> None:
        if not cleaned_transcript.strip():
            raise TranscriptCleanupError("Cleaned transcript is empty.")

        original_headers = self._speaker_headers(original_transcript)
        cleaned_headers = self._speaker_headers(cleaned_transcript)
        if original_headers and len(cleaned_headers) < len(original_headers):
            raise TranscriptCleanupError(
                "Cleaned transcript appears to be missing speaker labels "
                "or timestamp headers."
            )

    def _validate_meeting_summary(self, summary: MeetingSummary) -> None:
        if not summary.title:
            raise MeetingSummaryError("Meeting summary is missing a title.")
        if not summary.short_summary:
            raise MeetingSummaryError("Meeting summary is missing a short summary.")
        if not summary.detailed_summary:
            raise MeetingSummaryError("Meeting summary is missing a detailed summary.")
        if not summary.topics_discussed:
            raise MeetingSummaryError("Meeting summary is missing topics discussed.")

    def _validate_action_item_result(
        self,
        result: ActionItemExtractionResult,
    ) -> None:
        for index, action_item in enumerate(result.action_items, start=1):
            if not action_item.task:
                raise ActionItemExtractionError(
                    f"Action item {index} is missing task."
                )
            if not action_item.status:
                raise ActionItemExtractionError(
                    f"Action item {index} is missing status."
                )

    def _validate_decision_result(
        self,
        result: DecisionExtractionResult,
    ) -> None:
        valid_confidence_values = {"high", "medium", "low"}
        for index, decision in enumerate(result.decisions, start=1):
            if not decision.decision:
                raise DecisionExtractionError(
                    f"Decision {index} is missing decision text."
                )
            if not decision.confidence:
                raise DecisionExtractionError(
                    f"Decision {index} is missing confidence."
                )
            if decision.confidence not in valid_confidence_values:
                raise DecisionExtractionError(
                    f"Decision {index} has invalid confidence "
                    f"{decision.confidence!r}; expected high, medium, or low."
                )

    def _validate_key_discussion_point_result(
        self,
        result: KeyDiscussionPointExtractionResult,
    ) -> None:
        for index, key_point in enumerate(result.key_discussion_points, start=1):
            if not key_point.point:
                raise KeyDiscussionPointExtractionError(
                    f"Key discussion point {index} is missing point text."
                )
            if not key_point.speakers:
                raise KeyDiscussionPointExtractionError(
                    f"Key discussion point {index} is missing speakers."
                )

    def _optional_string(self, value: object) -> str | None:
        if value is None:
            return None

        text = str(value).strip()
        if not text:
            return None

        return text

    def _speaker_headers(self, transcript_text: str) -> list[str]:
        return [
            line.strip()
            for line in transcript_text.splitlines()
            if SPEAKER_HEADER_PATTERN.match(line)
        ]
