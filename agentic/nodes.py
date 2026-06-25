"""Meeting analysis agents used by the Python orchestrator."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from math import ceil
from typing import TypeVar

from agentic.state import AgenticMeetingState
from summarization.base_summarizer import (
    ActionItem,
    Decision,
    KeyDiscussionPoint,
    MeetingAnalysisResult,
)

logger = logging.getLogger(__name__)

StateUpdate = dict[str, object]
T = TypeVar("T")


class MetadataAgent:
    """Extract deterministic meeting metadata before LLM analysis."""

    name = "MetadataAgent"

    def run(self, context: AgenticMeetingState) -> StateUpdate:
        return metadata_node(context)


class SummaryAgent:
    """Generate the meeting summary and seed the shared analysis context."""

    name = "SummaryAgent"

    def run(self, context: AgenticMeetingState) -> StateUpdate:
        return summary_node(context)


class DiscussionAgent:
    """Generate key discussion points for the meeting."""

    name = "DiscussionAgent"

    def run(self, context: AgenticMeetingState) -> StateUpdate:
        return discussion_points_node(context)


class DecisionAgent:
    """Generate meeting decisions."""

    name = "DecisionAgent"

    def run(self, context: AgenticMeetingState) -> StateUpdate:
        return decisions_node(context)


class ActionItemAgent:
    """Generate action items and follow-up tasks."""

    name = "ActionItemAgent"

    def run(self, context: AgenticMeetingState) -> StateUpdate:
        return action_items_node(context)


class ValidationAgent:
    """Validate and lightly repair generated meeting analysis sections."""

    name = "ValidationAgent"

    def run(self, context: AgenticMeetingState) -> StateUpdate:
        return validation_node(context)


def metadata_node(state: AgenticMeetingState) -> StateUpdate:
    """Extract fast, deterministic metadata from the resolved transcript."""

    return _run_logged_node(MetadataAgent.name, _metadata_node_impl, state)


def summary_node(state: AgenticMeetingState) -> StateUpdate:
    """Run the existing full analysis once and expose the summary fields."""

    return _run_logged_node(SummaryAgent.name, _summary_node_impl, state)


def discussion_points_node(state: AgenticMeetingState) -> StateUpdate:
    """Expose discussion points from the existing analysis result."""

    return _run_logged_node(
        DiscussionAgent.name,
        _discussion_points_node_impl,
        state,
    )


def decisions_node(state: AgenticMeetingState) -> StateUpdate:
    """Expose decisions from the existing analysis result."""

    return _run_logged_node(DecisionAgent.name, _decisions_node_impl, state)


def action_items_node(state: AgenticMeetingState) -> StateUpdate:
    """Expose action items from the existing analysis result."""

    return _run_logged_node(ActionItemAgent.name, _action_items_node_impl, state)


def validation_node(state: AgenticMeetingState) -> StateUpdate:
    """Validate and lightly repair generated meeting analysis sections."""

    return _run_logged_node(ValidationAgent.name, _validation_node_impl, state)


def merge_results_node(state: AgenticMeetingState) -> StateUpdate:
    """Merge node outputs into the existing MeetingAnalysisResult contract."""

    return _run_logged_node("ResultMerge", _merge_results_node_impl, state)


def _summary_node_impl(state: AgenticMeetingState) -> StateUpdate:
    started_at = time.perf_counter()
    transcript = state["transcript"].strip()
    summarizer = state.get("summarizer")
    if summarizer is None:
        raise ValueError("Agentic meeting workflow requires a summarizer.")

    analysis = summarizer.analyze_meeting(transcript)
    summary_evidence = _summary_evidence(transcript)
    summary_confidence = _confidence_level(
        supported=bool(summary_evidence.get("text")),
        has_timestamp=bool(summary_evidence.get("timestamp")),
        validation_status="success",
    )
    logger.info(
        "Evidence and confidence generated for summary in %.3fs: confidence=%s",
        time.perf_counter() - started_at,
        summary_confidence,
    )
    return {
        "transcript": transcript,
        "cleaned_transcript": analysis.cleaned_transcript,
        "summary": analysis.summary,
        "analysis_result": analysis,
        "evidence": {
            **state.get("evidence", {}),
            "summary": summary_evidence,
        },
        "confidence": {
            **state.get("confidence", {}),
            "summary": summary_confidence,
        },
        "validation_results": {
            **state.get("validation_results", {}),
            "summary": "success",
        },
    }


def _metadata_node_impl(state: AgenticMeetingState) -> StateUpdate:
    started_at = time.perf_counter()
    transcript = state["transcript"].strip()
    incoming_metadata = dict(state.get("meeting_metadata", {}))
    speaker_mapping = state.get("speaker_mapping", {})

    participants = _participant_list(transcript, speaker_mapping)
    word_count = len(re.findall(r"\b[\w']+\b", transcript))
    duration_seconds = _duration_seconds(transcript, incoming_metadata)
    metadata = {
        **incoming_metadata,
        "meeting_duration_seconds": duration_seconds,
        "meeting_duration": _format_duration(duration_seconds),
        "number_of_speakers": len(participants),
        "participant_list": participants,
        "transcript_language": _transcript_language(incoming_metadata),
        "word_count": word_count,
        "estimated_reading_time_minutes": max(1, ceil(word_count / 200))
        if word_count
        else 0,
    }
    elapsed = time.perf_counter() - started_at
    metadata["metadata_processing_time_seconds"] = round(elapsed, 3)
    logger.info("Metadata values extracted: %s", metadata)
    return {
        "meeting_metadata": metadata,
        "extracted_metadata": metadata,
        "validation_results": {
            **state.get("validation_results", {}),
            "metadata": {
                "status": "success",
                "processing_time_seconds": round(elapsed, 3),
            },
        },
    }


def _discussion_points_node_impl(state: AgenticMeetingState) -> StateUpdate:
    started_at = time.perf_counter()
    summarizer = state.get("summarizer")
    transcript = _analysis_transcript(state)
    analysis = _optional_analysis_from_state(state)
    discussion_points = (
        analysis.key_discussion_points
        if analysis is not None
        else summarizer.extract_key_discussion_points(transcript).key_discussion_points
    )
    evidence = [
        _item_evidence(item.point, transcript, timestamp=item.timestamp)
        for item in discussion_points
    ]
    confidence = [
        _confidence_level(
            supported=bool(item_evidence.get("text")),
            has_timestamp=bool(item_evidence.get("timestamp")),
            validation_status="success",
        )
        for item_evidence in evidence
    ]
    logger.info(
        "Evidence and confidence generated for discussion points in %.3fs: count=%d",
        time.perf_counter() - started_at,
        len(discussion_points),
    )
    return {
        "discussion_points": discussion_points,
        "evidence": {
            **state.get("evidence", {}),
            "discussion_points": evidence,
        },
        "confidence": {
            **state.get("confidence", {}),
            "discussion_points": confidence,
        },
        "validation_results": {
            **state.get("validation_results", {}),
            "discussion_points": "success",
        },
    }


def _decisions_node_impl(state: AgenticMeetingState) -> StateUpdate:
    started_at = time.perf_counter()
    summarizer = state.get("summarizer")
    transcript = _analysis_transcript(state)
    analysis = _optional_analysis_from_state(state)
    decisions = (
        analysis.decisions
        if analysis is not None
        else summarizer.extract_decisions(transcript).decisions
    )
    evidence = [
        _item_evidence(item.decision, transcript, timestamp=item.timestamp)
        for item in decisions
    ]
    confidence = [
        _confidence_level(
            supported=bool(item_evidence.get("text")),
            has_timestamp=bool(item_evidence.get("timestamp")),
            validation_status=item.confidence,
        )
        for item, item_evidence in zip(decisions, evidence)
    ]
    logger.info(
        "Evidence and confidence generated for decisions in %.3fs: count=%d",
        time.perf_counter() - started_at,
        len(decisions),
    )
    return {
        "decisions": decisions,
        "evidence": {
            **state.get("evidence", {}),
            "decisions": evidence,
        },
        "confidence": {
            **state.get("confidence", {}),
            "decisions": confidence,
        },
        "validation_results": {
            **state.get("validation_results", {}),
            "decisions": "success",
        },
    }


def _action_items_node_impl(state: AgenticMeetingState) -> StateUpdate:
    started_at = time.perf_counter()
    summarizer = state.get("summarizer")
    transcript = _analysis_transcript(state)
    analysis = _optional_analysis_from_state(state)
    action_items = (
        analysis.action_items
        if analysis is not None
        else summarizer.extract_action_items(transcript).action_items
    )
    evidence = [
        _item_evidence(item.task, transcript, timestamp=item.timestamp)
        for item in action_items
    ]
    confidence = [
        _confidence_level(
            supported=bool(item_evidence.get("text")),
            has_timestamp=bool(item_evidence.get("timestamp")),
            validation_status="success",
        )
        for item_evidence in evidence
    ]
    logger.info(
        "Evidence and confidence generated for action items in %.3fs: count=%d",
        time.perf_counter() - started_at,
        len(action_items),
    )
    return {
        "action_items": action_items,
        "evidence": {
            **state.get("evidence", {}),
            "action_items": evidence,
        },
        "confidence": {
            **state.get("confidence", {}),
            "action_items": confidence,
        },
        "validation_results": {
            **state.get("validation_results", {}),
            "action_items": "success",
        },
    }


def _validation_node_impl(state: AgenticMeetingState) -> StateUpdate:
    original_analysis = _analysis_from_state(state)
    started_at = time.perf_counter()
    logger.info("Validation started")

    try:
        transcript = state["transcript"].strip()
        summarizer = state.get("summarizer")
        summary = _require_state_value(state, "summary")
        discussion_points = list(state.get("discussion_points", []))
        decisions = list(state.get("decisions", []))
        action_items = list(state.get("action_items", []))

        discussion_points, duplicate_discussions = _dedupe_items(
            discussion_points,
            _discussion_key,
        )
        decisions, duplicate_decisions = _dedupe_items(decisions, _decision_key)
        action_items, duplicate_actions = _dedupe_items(action_items, _action_key)

        discussion_points, cross_section_discussions = _remove_cross_section_repeats(
            discussion_points,
            decisions,
            action_items,
        )

        empty_sections = _empty_sections(
            discussion_points=discussion_points,
            decisions=decisions,
            action_items=action_items,
        )
        regeneration_triggered: list[str] = []

        if summarizer is not None:
            if "discussion_points" in empty_sections and _has_discussion_signal(transcript):
                logger.info("Validation regeneration triggered for discussion_points")
                discussion_points = summarizer.extract_key_discussion_points(
                    transcript
                ).key_discussion_points
                regeneration_triggered.append("discussion_points")

            if "decisions" in empty_sections and _has_decision_signal(transcript):
                logger.info("Validation regeneration triggered for decisions")
                decisions = summarizer.extract_decisions(transcript).decisions
                regeneration_triggered.append("decisions")

            if "action_items" in empty_sections and _has_action_signal(transcript):
                logger.info("Validation regeneration triggered for action_items")
                action_items = summarizer.extract_action_items(transcript).action_items
                regeneration_triggered.append("action_items")

        action_support = _action_support_counts(action_items, transcript)
        owner_quality = _action_owner_quality_counts(action_items)

        validation_details = {
            **state.get("validation_results", {}),
            "validation": {
                "status": "success",
                "checks_performed": [
                    "duplicate_discussion_points",
                    "duplicate_decisions",
                    "duplicate_action_items",
                    "empty_sections",
                    "targeted_regeneration",
                    "cross_section_repetition",
                    "action_item_transcript_support",
                    "action_item_owner_or_actionability",
                    "decision_discussion_overlap",
                ],
                "duplicates_removed": {
                    "discussion_points": duplicate_discussions,
                    "decisions": duplicate_decisions,
                    "action_items": duplicate_actions,
                    "cross_section_discussion_points": cross_section_discussions,
                },
                "empty_sections_detected": empty_sections,
                "regeneration_triggered": regeneration_triggered,
                "unsupported_action_items": action_support["unsupported"],
                "action_items_without_owner": owner_quality["without_owner"],
                "processing_time_seconds": round(time.perf_counter() - started_at, 3),
            },
        }

        logger.info(
            "Validation checks performed: duplicates_removed=%s empty_sections=%s "
            "regeneration_triggered=%s unsupported_action_items=%d "
            "action_items_without_owner=%d",
            validation_details["validation"]["duplicates_removed"],
            empty_sections,
            regeneration_triggered,
            action_support["unsupported"],
            owner_quality["without_owner"],
        )
        logger.info(
            "Validation completed in %.3fs",
            validation_details["validation"]["processing_time_seconds"],
        )

        analysis = MeetingAnalysisResult(
            cleaned_transcript=str(state.get("cleaned_transcript", "")).strip()
            or original_analysis.cleaned_transcript,
            summary=summary,
            key_discussion_points=discussion_points,
            decisions=decisions,
            action_items=action_items,
        )
        final_evidence = _analysis_evidence(analysis, transcript)
        final_confidence = _analysis_confidence(
            analysis,
            final_evidence,
            validation_details["validation"]["status"],
        )
        return {
            "cleaned_transcript": analysis.cleaned_transcript,
            "summary": analysis.summary,
            "discussion_points": analysis.key_discussion_points,
            "decisions": analysis.decisions,
            "action_items": analysis.action_items,
            "analysis_result": analysis,
            "evidence": final_evidence,
            "confidence": final_confidence,
            "validation_results": validation_details,
        }
    except Exception:
        elapsed = time.perf_counter() - started_at
        logger.exception(
            "Validation failed unexpectedly after %.3fs; returning original analysis",
            elapsed,
        )
        return {
            "cleaned_transcript": original_analysis.cleaned_transcript,
            "summary": original_analysis.summary,
            "discussion_points": original_analysis.key_discussion_points,
            "decisions": original_analysis.decisions,
            "action_items": original_analysis.action_items,
            "analysis_result": original_analysis,
            "validation_results": {
                **state.get("validation_results", {}),
                "validation": {
                    "status": "failed_fallback_original",
                    "processing_time_seconds": round(elapsed, 3),
                },
            },
        }


def _merge_results_node_impl(state: AgenticMeetingState) -> StateUpdate:
    analysis = MeetingAnalysisResult(
        cleaned_transcript=str(state.get("cleaned_transcript", "")).strip(),
        summary=_require_state_value(state, "summary"),
        key_discussion_points=list(state.get("discussion_points", [])),
        decisions=list(state.get("decisions", [])),
        action_items=list(state.get("action_items", [])),
    )
    return {
        "analysis_result": analysis,
        "validation_results": {
            **state.get("validation_results", {}),
            "merge_results": "success",
        },
    }


def _analysis_from_state(state: AgenticMeetingState) -> MeetingAnalysisResult:
    analysis = state.get("analysis_result")
    if not isinstance(analysis, MeetingAnalysisResult):
        raise ValueError("Agentic meeting workflow requires analysis_result.")
    return analysis


def _optional_analysis_from_state(
    state: AgenticMeetingState,
) -> MeetingAnalysisResult | None:
    analysis = state.get("analysis_result")
    if isinstance(analysis, MeetingAnalysisResult):
        return analysis
    return None


def _analysis_transcript(state: AgenticMeetingState) -> str:
    transcript = str(state.get("cleaned_transcript") or state["transcript"]).strip()
    if not transcript:
        raise ValueError("Agentic meeting workflow requires transcript.")
    summarizer = state.get("summarizer")
    if summarizer is None:
        raise ValueError("Agentic meeting workflow requires a summarizer.")
    return transcript


def _summary_evidence(transcript: str) -> dict[str, object]:
    snippet = _first_transcript_snippet(transcript)
    return {
        "text": snippet,
        "timestamp": _timestamp_for_snippet(snippet),
    }


def _item_evidence(
    item_text: str,
    transcript: str,
    *,
    timestamp: str | None = None,
) -> dict[str, object]:
    started_at = time.perf_counter()
    snippet = _best_supporting_snippet(item_text, transcript)
    evidence = {
        "text": snippet,
        "timestamp": timestamp or _timestamp_for_snippet(snippet),
    }
    logger.info(
        "Evidence extracted in %.3fs: has_text=%s has_timestamp=%s",
        time.perf_counter() - started_at,
        bool(evidence["text"]),
        bool(evidence["timestamp"]),
    )
    return evidence


def _confidence_level(
    *,
    supported: bool,
    has_timestamp: bool,
    validation_status: str,
) -> str:
    normalized_status = validation_status.lower().strip()
    if supported and (has_timestamp or normalized_status == "high"):
        confidence = "High"
    elif supported or normalized_status in {"success", "medium", "high"}:
        confidence = "Medium"
    else:
        confidence = "Low"
    logger.info(
        "Confidence generated: supported=%s has_timestamp=%s validation=%s "
        "confidence=%s",
        supported,
        has_timestamp,
        validation_status,
        confidence,
    )
    return confidence


def _analysis_evidence(
    analysis: MeetingAnalysisResult,
    transcript: str,
) -> dict[str, object]:
    started_at = time.perf_counter()
    evidence = {
        "summary": _summary_evidence(transcript),
        "discussion_points": [
            _item_evidence(item.point, transcript, timestamp=item.timestamp)
            for item in analysis.key_discussion_points
        ],
        "decisions": [
            _item_evidence(item.decision, transcript, timestamp=item.timestamp)
            for item in analysis.decisions
        ],
        "action_items": [
            _item_evidence(item.task, transcript, timestamp=item.timestamp)
            for item in analysis.action_items
        ],
    }
    logger.info(
        "Evidence extraction refreshed for final analysis in %.3fs",
        time.perf_counter() - started_at,
    )
    return evidence


def _analysis_confidence(
    analysis: MeetingAnalysisResult,
    evidence: dict[str, object],
    validation_status: str,
) -> dict[str, object]:
    started_at = time.perf_counter()
    discussion_evidence = list(evidence.get("discussion_points", []))
    decision_evidence = list(evidence.get("decisions", []))
    action_evidence = list(evidence.get("action_items", []))
    confidence = {
        "summary": _confidence_level(
            supported=bool(dict(evidence.get("summary", {})).get("text")),
            has_timestamp=bool(dict(evidence.get("summary", {})).get("timestamp")),
            validation_status=validation_status,
        ),
        "discussion_points": [
            _confidence_level(
                supported=bool(dict(item_evidence).get("text")),
                has_timestamp=bool(dict(item_evidence).get("timestamp")),
                validation_status=validation_status,
            )
            for item_evidence in discussion_evidence
        ],
        "decisions": [
            _confidence_level(
                supported=bool(dict(item_evidence).get("text")),
                has_timestamp=bool(dict(item_evidence).get("timestamp")),
                validation_status=item.confidence,
            )
            for item, item_evidence in zip(analysis.decisions, decision_evidence)
        ],
        "action_items": [
            _confidence_level(
                supported=bool(dict(item_evidence).get("text")),
                has_timestamp=bool(dict(item_evidence).get("timestamp")),
                validation_status=validation_status,
            )
            for item_evidence in action_evidence
        ],
    }
    logger.info(
        "Confidence generation refreshed for final analysis in %.3fs",
        time.perf_counter() - started_at,
    )
    return confidence


def _best_supporting_snippet(item_text: str, transcript: str) -> str:
    item_terms = _important_terms(item_text)
    if not item_terms:
        return ""

    best_snippet = ""
    best_score = 0
    for snippet in _transcript_snippets(transcript):
        normalized_snippet = _normalize_text(snippet)
        score = sum(1 for term in item_terms if term in normalized_snippet)
        if score > best_score:
            best_score = score
            best_snippet = snippet

    minimum_score = max(1, min(3, len(item_terms) // 2))
    return best_snippet if best_score >= minimum_score else ""


def _first_transcript_snippet(transcript: str) -> str:
    for snippet in _transcript_snippets(transcript):
        if snippet:
            return snippet[:500]
    return transcript.strip()[:500]


def _transcript_snippets(transcript: str) -> list[str]:
    blocks = [
        re.sub(r"\s+", " ", block).strip()
        for block in re.split(r"\n\s*\n", transcript)
        if block.strip()
    ]
    if blocks:
        return blocks
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", transcript)
        if sentence.strip()
    ]


def _timestamp_for_snippet(snippet: str) -> str | None:
    match = re.search(r"\[(?P<timestamp>[^\]]+)\]", snippet)
    if match:
        return match.group("timestamp").strip()
    return None


def _important_terms(text: str) -> set[str]:
    return {
        token
        for token in _normalize_text(text).split()
        if len(token) > 3 and token not in _STOPWORDS
    }


def _participant_list(
    transcript: str,
    speaker_mapping: dict[str, str],
) -> list[str]:
    participants: list[str] = []
    seen: set[str] = set()

    for label, mapped_name in speaker_mapping.items():
        participant = (mapped_name or label).strip()
        if participant and participant.lower() not in seen:
            participants.append(participant)
            seen.add(participant.lower())

    if participants:
        return participants

    for line in transcript.splitlines():
        label = _speaker_label_from_line(line)
        if label and label.lower() not in seen:
            participants.append(label)
            seen.add(label.lower())
    return participants


def _speaker_label_from_line(line: str) -> str | None:
    match = re.match(
        r"^\s*(?P<label>[A-Za-z][A-Za-z0-9 .,'_-]{0,59}?)(?:\s*\[[^\]]+\])?\s*:",
        line,
    )
    if match:
        return re.sub(r"\s+", " ", match.group("label")).strip()

    match = re.match(
        r"^\s*(?P<label>[A-Za-z][A-Za-z0-9 .,'_-]{0,59}?)\s*\[[^\]]+\]",
        line,
    )
    if match:
        return re.sub(r"\s+", " ", match.group("label")).strip()
    return None


def _duration_seconds(transcript: str, metadata: dict[str, object]) -> float | None:
    for key in ("duration_seconds", "meeting_duration_seconds"):
        value = metadata.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str) and value.strip():
            parsed = _parse_time_value(value)
            if parsed is not None:
                return parsed

    end_times = [
        parsed
        for parsed in (
            _parse_time_value(match.group("end"))
            for match in re.finditer(r"\[[^\]-]+-\s*(?P<end>[^\]]+)\]", transcript)
        )
        if parsed is not None
    ]
    return max(end_times) if end_times else None


def _parse_time_value(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    parts = value.split(":")
    try:
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        return float(value)
    except ValueError:
        return None


def _format_duration(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    total_seconds = max(0, round(seconds))
    minutes, remaining_seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"
    return f"{minutes:02d}:{remaining_seconds:02d}"


def _transcript_language(metadata: dict[str, object]) -> str:
    value = metadata.get("transcript_language") or metadata.get("language_code")
    return str(value).strip() if value else "unknown"


def _require_state_value(state: AgenticMeetingState, key: str) -> T:
    value = state.get(key)
    if value is None:
        raise ValueError(f"Agentic meeting workflow requires {key}.")
    return value


def _dedupe_items(
    items: list[T],
    key_factory: Callable[[T], str],
) -> tuple[list[T], int]:
    deduped: list[T] = []
    seen: set[str] = set()
    removed = 0
    for item in items:
        key = key_factory(item)
        if key and key in seen:
            removed += 1
            continue
        if key:
            seen.add(key)
        deduped.append(item)
    return deduped, removed


def _remove_cross_section_repeats(
    discussion_points: list[KeyDiscussionPoint],
    decisions: list[Decision],
    action_items: list[ActionItem],
) -> tuple[list[KeyDiscussionPoint], int]:
    reserved = {
        key
        for key in [
            *(_decision_key(item) for item in decisions),
            *(_action_key(item) for item in action_items),
        ]
        if key
    }
    filtered: list[KeyDiscussionPoint] = []
    removed = 0
    for item in discussion_points:
        if _discussion_key(item) in reserved:
            removed += 1
            continue
        filtered.append(item)
    return filtered, removed


def _empty_sections(
    *,
    discussion_points: list[KeyDiscussionPoint],
    decisions: list[Decision],
    action_items: list[ActionItem],
) -> list[str]:
    sections: list[str] = []
    if not discussion_points:
        sections.append("discussion_points")
    if not decisions:
        sections.append("decisions")
    if not action_items:
        sections.append("action_items")
    logger.info("Validation empty section detection: %s", sections)
    return sections


def _action_support_counts(
    action_items: list[ActionItem],
    transcript: str,
) -> dict[str, int]:
    unsupported = 0
    for item in action_items:
        if not _text_supported_by_transcript(item.task, transcript):
            unsupported += 1
    return {"total": len(action_items), "unsupported": unsupported}


def _action_owner_quality_counts(action_items: list[ActionItem]) -> dict[str, int]:
    without_owner = sum(1 for item in action_items if not (item.owner or "").strip())
    return {"total": len(action_items), "without_owner": without_owner}


def _text_supported_by_transcript(text: str, transcript: str) -> bool:
    normalized_text = _normalize_text(text)
    normalized_transcript = _normalize_text(transcript)
    if not normalized_text:
        return False
    if normalized_text in normalized_transcript:
        return True

    terms = {
        token
        for token in normalized_text.split()
        if len(token) > 3 and token not in _STOPWORDS
    }
    if not terms:
        return True
    matches = sum(1 for term in terms if term in normalized_transcript)
    return matches >= max(1, min(3, len(terms) // 2))


def _has_discussion_signal(transcript: str) -> bool:
    return len(_normalize_text(transcript).split()) >= 20


def _has_decision_signal(transcript: str) -> bool:
    return _has_any_signal(
        transcript,
        [
            "agreed",
            "approved",
            "decided",
            "decision",
            "finalized",
            "resolved",
            "signed off",
            "will proceed",
        ],
    )


def _has_action_signal(transcript: str) -> bool:
    return _has_any_signal(
        transcript,
        [
            "action item",
            "assign",
            "follow up",
            "needs to",
            "next step",
            "owner",
            "please",
            "task",
            "todo",
            "will send",
        ],
    )


def _has_any_signal(transcript: str, signals: list[str]) -> bool:
    normalized = _normalize_text(transcript)
    return any(signal in normalized for signal in signals)


def _discussion_key(item: KeyDiscussionPoint) -> str:
    return _normalize_text(item.point)


def _decision_key(item: Decision) -> str:
    return _normalize_text(item.decision)


def _action_key(item: ActionItem) -> str:
    return _normalize_text(item.task)


def _normalize_text(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", " ", value.lower())
    return re.sub(r"\s+", " ", value).strip()


_STOPWORDS = {
    "about",
    "after",
    "also",
    "from",
    "have",
    "into",
    "that",
    "their",
    "them",
    "then",
    "there",
    "this",
    "with",
    "will",
}


def _run_logged_node(
    node_name: str,
    handler: Callable[[AgenticMeetingState], StateUpdate],
    state: AgenticMeetingState,
) -> StateUpdate:
    started_at = time.perf_counter()
    logger.info("%s started", node_name)
    try:
        update = handler(state)
    except Exception:
        elapsed = time.perf_counter() - started_at
        logger.exception("%s failed after %.3fs", node_name, elapsed)
        raise

    elapsed = time.perf_counter() - started_at
    logger.info("%s completed in %.3fs", node_name, elapsed)
    return update
