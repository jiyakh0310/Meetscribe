"""Participant resolution for BERT NER metadata extraction.

Purpose:
    Convert raw PERSON entities and speaker labels into a clean participant
    list suitable for pre-filling meeting information.

Responsibilities:
    - Remove honorifics and noisy speaker labels.
    - Merge duplicate or partial participant names.
    - Preserve first-seen transcript order.
    - Keep confidence scores for audit logging.

Inputs:
    PERSON entities from BERT NER and optional speaker labels from the
    transcript parser.

Outputs:
    Ordered participant names with associated confidence values.

Future Implementation Notes:
    This resolver is intentionally conservative. It should enrich metadata
    only, never rewrite transcripts or speaker mappings.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from ml_ner.bert_ner import BIONerEntity


HONORIFIC_RE = re.compile(
    r"(?i)^(?:mr|mrs|ms|miss|dr|prof|sir|madam)\.?\s+"
)
GENERIC_SPEAKER_RE = re.compile(r"(?i)^speaker\s+[A-Za-z0-9]+$")
NAME_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
NON_PERSON_LABELS = {
    "agenda",
    "attendees",
    "date",
    "duration",
    "location",
    "meeting",
    "meeting date",
    "meeting time",
    "organisation",
    "organization",
    "participants",
    "platform",
    "project",
    "source",
    "time",
    "title",
    "venue",
}


@dataclass(slots=True)
class ResolvedParticipant:
    """Resolved participant candidate.

    Attributes:
        name: Display name selected for meeting metadata.
        confidence: Highest confidence observed across merged evidence.
        sources: Evidence sources such as ``ner`` or ``speaker_label``.
    """

    name: str
    confidence: float
    sources: list[str]


def resolve_participants(
    person_entities: list[BIONerEntity],
    speaker_labels: list[str] | None = None,
) -> list[ResolvedParticipant]:
    """Resolve participants from NER PERSON entities and speaker labels.

    Args:
        person_entities: PERSON entities detected by the BERT NER model.
        speaker_labels: Optional speaker labels parsed from transcript turns.

    Returns:
        Ordered resolved participant records.
    """

    resolved: list[ResolvedParticipant] = []

    # Speaker labels are strong participant evidence in meeting transcripts.
    # They are added before BERT PERSON entities to preserve transcript order.
    for speaker in speaker_labels or []:
        cleaned = normalize_person_name(speaker)
        if cleaned:
            _merge_participant(
                resolved,
                cleaned,
                confidence=1.0,
                source="speaker_label",
            )

    for entity in person_entities:
        cleaned = normalize_person_name(entity.text)
        if cleaned:
            _merge_participant(
                resolved,
                cleaned,
                confidence=entity.confidence,
                source="ner",
            )

    return resolved


def normalize_person_name(value: str) -> str:
    """Normalize a person name while preserving human-readable casing.

    Args:
        value: Raw PERSON entity or speaker label.

    Returns:
        Cleaned participant name, or an empty string when the value is generic
        or not person-like.
    """

    name = " ".join(str(value or "").replace(":", " ").split())
    name = HONORIFIC_RE.sub("", name).strip(" -")
    if not name or GENERIC_SPEAKER_RE.fullmatch(name):
        return ""
    if name.casefold() in NON_PERSON_LABELS:
        return ""

    tokens = NAME_TOKEN_RE.findall(name)
    if not tokens:
        return ""
    if len(tokens) > 4:
        # Very long "names" are usually parser artifacts or copied headings.
        return ""

    return " ".join(tokens)


def _merge_participant(
    resolved: list[ResolvedParticipant],
    candidate: str,
    *,
    confidence: float,
    source: str,
) -> None:
    """Merge a participant candidate into an ordered participant list.

    Args:
        resolved: Existing participant list to update.
        candidate: Cleaned candidate name.
        confidence: Confidence score for the candidate.
        source: Evidence source name.
    """

    candidate_key = _name_key(candidate)
    candidate_tokens = set(candidate_key.split())
    for participant in resolved:
        existing_key = _name_key(participant.name)
        existing_tokens = set(existing_key.split())

        # Merge when one name is a confident subset of another, such as
        # "Rahul" and "Rahul Sharma". Prefer the more specific display name.
        if (
            candidate_key == existing_key
            or candidate_tokens.issubset(existing_tokens)
            or existing_tokens.issubset(candidate_tokens)
        ):
            if len(candidate_tokens) > len(existing_tokens):
                participant.name = candidate
            participant.confidence = round(max(participant.confidence, confidence), 6)
            if source not in participant.sources:
                participant.sources.append(source)
            return

    resolved.append(
        ResolvedParticipant(
            name=candidate,
            confidence=round(float(confidence), 6),
            sources=[source],
        )
    )


def _name_key(value: str) -> str:
    """Return a comparable key for a participant name.

    Args:
        value: Display name.

    Returns:
        Case-folded token key.
    """

    return " ".join(token.casefold() for token in NAME_TOKEN_RE.findall(value))
