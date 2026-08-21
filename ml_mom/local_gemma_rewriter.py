"""Fail-safe local Gemma language polishing for deterministic meeting minutes."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, replace
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config.settings import (
    OLLAMA_BASE_URL,
    OLLAMA_GEMMA_MODEL,
    OLLAMA_TIMEOUT_SECONDS,
    USE_LOCAL_GEMMA,
)
from ml_mom.experimental.mom_formatter import ExperimentalMom

LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a professional meeting minutes editor.

Rewrite the following Minutes of Meeting.

Rules:

Do NOT add information.
Do NOT remove information.
Do NOT infer.
Do NOT hallucinate.
Do NOT create new action items.
Do NOT create new decisions.
Do NOT modify names.
Do NOT modify dates.
Do NOT modify owners.
Do NOT modify priorities.
Keep every heading unchanged.
Keep every decision and action-item row exactly unchanged.
Improve only the wording in Executive Summary and Discussion:

grammar
sentence flow
professional tone
readability
natural wording

Return only the improved Minutes of Meeting."""

AUDIO_QUALITY_INSTRUCTIONS = """

Additional rules for minutes derived from audio speech-to-text:

The source may contain Hinglish (mixed Hindi and English), transcription
fillers, or repeated phrases. Express only the supplied meaning as fluent,
professional business English. Translate by business intent, never word for
word. Ignore filler language and repetition without removing real outcomes.
Preserve commitments, ownership, deadlines, uncertainty, and negation exactly.
The final Minutes of Meeting must be written entirely in professional English,
even when the spoken conversation is Hinglish or Hindi-English mixed speech.

Executive Summary must be concise, outcome-focused, and contain no repeated or
near-duplicate sentence or paragraph.

For each existing Discussion bullet, replace only its topic prefix before the
colon when necessary. Use a grounded 3-6 word professional topic that represents
the overall meaning of that bullet, not the first words of a transcript phrase.
Keep the same number and order of Discussion bullets. Do not create topics,
facts, decisions, or actions that are not supported by the supplied minutes.
Section headings such as Executive Summary and Discussion remain unchanged.
"""


@dataclass(frozen=True, slots=True)
class GemmaRewriteResult:
    """Result of a best-effort rewrite; ``mom`` is always safe to consume."""

    mom: ExperimentalMom
    applied: bool = False
    model: str = ""
    error: str | None = None


class LocalGemmaRewriter:
    """Polish formatter output through a local Ollama server."""

    def __init__(
        self,
        *,
        enabled: bool = USE_LOCAL_GEMMA,
        base_url: str = OLLAMA_BASE_URL,
        preferred_model: str = OLLAMA_GEMMA_MODEL,
        timeout_seconds: float = OLLAMA_TIMEOUT_SECONDS,
    ) -> None:
        self.enabled = enabled
        self.base_url = base_url.rstrip("/")
        self.preferred_model = preferred_model
        self.timeout_seconds = timeout_seconds

    def rewrite(
        self,
        deterministic_mom: ExperimentalMom,
        *,
        audio_quality_mode: bool = False,
    ) -> GemmaRewriteResult:
        """Return a refined copy or the original object on every failure."""

        if not self.enabled:
            return GemmaRewriteResult(mom=deterministic_mom)

        try:
            model = self._select_local_gemma_model()
            if not model:
                raise RuntimeError("No local Gemma model is installed in Ollama.")
            source = _permitted_markdown(deterministic_mom)
            prompt = SYSTEM_PROMPT
            if audio_quality_mode:
                prompt += AUDIO_QUALITY_INSTRUCTIONS
            response = self._post_json(
                "/api/chat",
                {
                    "model": model,
                    "stream": False,
                    "options": {"temperature": 0},
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": source},
                    ],
                },
            )
            content = str(response.get("message", {}).get("content", "")).strip()
            refined = _apply_safe_rewrite(
                deterministic_mom,
                content,
                allow_discussion_topic_rewrite=audio_quality_mode,
            )
            if audio_quality_mode:
                refined = replace(
                    refined,
                    summary=_deduplicate_summary_sentences(refined.summary),
                )
            return GemmaRewriteResult(mom=refined, applied=True, model=model)
        except Exception as exc:
            LOGGER.warning("Local Gemma rewrite unavailable; using deterministic MoM: %s", exc)
            return GemmaRewriteResult(mom=deterministic_mom, error=str(exc))

    def _select_local_gemma_model(self) -> str:
        response = self._get_json("/api/tags")
        names = [
            str(model.get("name", "")).strip()
            for model in response.get("models", [])
            if isinstance(model, dict)
        ]
        gemma_names = [name for name in names if "gemma" in name.casefold()]
        if self.preferred_model in gemma_names:
            return self.preferred_model
        return gemma_names[0] if gemma_names else ""

    def _get_json(self, path: str) -> dict[str, Any]:
        request = Request(f"{self.base_url}{path}", method="GET")
        return self._open_json(request)

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._open_json(request)

    def _open_json(self, request: Request) -> dict[str, Any]:
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc
        if not isinstance(result, dict):
            raise RuntimeError("Ollama returned an invalid response.")
        return result


def _permitted_markdown(mom: ExperimentalMom) -> str:
    """Render only fields explicitly permitted to cross the Gemma boundary."""

    participants = ", ".join(mom.participants) if mom.participants else "-"
    lines = [
        "## Meeting Information",
        f"**Title:** {mom.meeting_title}",
        f"**Meeting Date:** {mom.meeting_date or '-'}",
        f"**Participants:** {participants}",
        "",
        "## Executive Summary",
        mom.summary,
        "",
        "## Discussion",
        *[f"- {point}" for point in mom.discussion_points],
        "",
        "## Decisions",
        *[f"- {decision}" for decision in mom.decisions],
        "",
        "## Action Items",
        "| Owner | Task | Deadline |",
        "| --- | --- | --- |",
    ]
    lines.extend(
        f"| {item.owner} | {item.task} | {item.deadline or ''} |"
        for item in mom.action_items
    )
    return "\n".join(lines).strip()


def _apply_safe_rewrite(
    mom: ExperimentalMom,
    markdown: str,
    *,
    allow_discussion_topic_rewrite: bool = False,
) -> ExperimentalMom:
    """Accept only the two language fields and enforce structural invariants."""

    required = (
        "Meeting Information",
        "Executive Summary",
        "Discussion",
        "Decisions",
        "Action Items",
    )
    sections = _markdown_sections(markdown)
    if any(heading not in sections for heading in required):
        raise ValueError("Gemma response changed or omitted a required heading.")

    summary = sections["Executive Summary"].strip()
    discussion = [
        re.sub(r"^\s*[-*+]\s+", "", line).strip()
        for line in sections["Discussion"].splitlines()
        if re.match(r"^\s*[-*+]\s+\S", line)
    ]
    if not summary or len(discussion) != len(mom.discussion_points):
        raise ValueError("Gemma response changed the MoM section structure.")

    if allow_discussion_topic_rewrite:
        discussion = [
            _validated_audio_discussion_rewrite(source, candidate)
            for source, candidate in zip(mom.discussion_points, discussion, strict=True)
        ]

    # Meeting information, decisions, action items, and every unsubmitted
    # formatter field are copied from the deterministic source of truth.
    return replace(mom, summary=summary, discussion_points=discussion)


def _validated_audio_discussion_rewrite(source: str, candidate: str) -> str:
    """Accept a semantic audio topic only when its structure is report-safe."""

    title, separator, body = candidate.partition(":")
    title_words = re.findall(r"[A-Za-z0-9&/-]+", title)
    if (
        not separator
        or not body.strip()
        or not 3 <= len(title_words) <= 6
    ):
        return source
    return f"{title.strip()}: {body.strip()}"


def _deduplicate_summary_sentences(summary: str) -> str:
    """Remove exact and near-duplicate audio-summary sentences."""

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", summary)
        if sentence.strip()
    ]
    unique: list[str] = []
    token_sets: list[set[str]] = []
    for sentence in sentences:
        tokens = set(re.findall(r"[a-z0-9]+", sentence.casefold()))
        if not tokens:
            continue
        duplicate = any(
            tokens == existing
            or len(tokens & existing) / max(1, min(len(tokens), len(existing))) >= 0.85
            for existing in token_sets
        )
        if duplicate:
            continue
        unique.append(sentence)
        token_sets.append(tokens)
    return " ".join(unique)


def _markdown_sections(markdown: str) -> dict[str, str]:
    matches = list(re.finditer(r"(?m)^##\s+(.+?)\s*$", markdown))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        sections[match.group(1).strip()] = markdown[match.end() : end].strip()
    return sections
