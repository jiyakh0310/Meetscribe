"""Meeting metadata extraction using BERT NER and speaker evidence.

Purpose:
    Extract participants and meeting metadata from transcripts before the
    existing ML Minutes of Meeting pipeline runs.

Responsibilities:
    - Parse transcript speaker labels for strong participant evidence.
    - Run BERT NER with BIO tagging for PERSON, DATE, TIME, ORG, and LOC.
    - Resolve duplicate participant names.
    - Return structured metadata and audit-friendly entity details.
    - Fail safely so upload, review, and MoM generation continue unchanged.

Inputs:
    Transcript text from audio transcription or transcript upload.

Outputs:
    ``EntityExtractionResult`` containing participants, metadata candidates,
    detected entities, BIO tags, confidence scores, and runtime diagnostics.

Future Implementation Notes:
    This module enriches meeting metadata only. It must not call or modify ANN
    training, ANN inference, embeddings, clustering, rule-based MoM generation,
    exports, email, or UI rendering.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any

from ml_mom.transcript_parser import ParsedTranscript, TranscriptTurn, parse_transcript
from ml_ner.bert_ner import BIONerEntity, BIONerToken, BertNerService
from ml_ner.participant_resolver import ResolvedParticipant, resolve_participants


@dataclass(slots=True)
class MeetingMetadataEntities:
    """Structured metadata entities extracted from a transcript.

    Attributes:
        participants: Ordered participant names.
        meeting_date: First DATE entity, when available.
        meeting_time: First TIME entity, when available.
        organization: First ORGANIZATION entity, when available.
        location: First LOCATION entity, when available.
    """

    participants: list[str] = field(default_factory=list)
    meeting_date: str = ""
    meeting_time: str = ""
    organization: str = ""
    location: str = ""


@dataclass(slots=True)
class EntityExtractionResult:
    """BERT NER metadata extraction result.

    Attributes:
        metadata: Meeting metadata values suitable for default UI fields.
        participants: Resolved participant records with confidence/source data.
        entities: Merged BERT entity spans.
        bio_tokens: Token-level BIO predictions retained internally.
        inference_time_seconds: NER inference duration.
        used_fallback: Whether BERT failed and caller should rely on fallback.
        error_message: Friendly non-fatal error message, if any.
    """

    metadata: MeetingMetadataEntities = field(default_factory=MeetingMetadataEntities)
    participants: list[ResolvedParticipant] = field(default_factory=list)
    entities: list[BIONerEntity] = field(default_factory=list)
    bio_tokens: list[BIONerToken] = field(default_factory=list)
    inference_time_seconds: float = 0.0
    used_fallback: bool = False
    error_message: str | None = None

    def to_metadata_dict(self) -> dict[str, Any]:
        """Convert extraction output into session metadata fields.

        Returns:
            Dictionary safe to merge into ``st.session_state.meeting_metadata``.
        """

        entity_payload = [
            {
                "text": entity.text,
                "type": _session_entity_type(entity.entity_group),
                "bio_tags": list(entity.bio_tags),
                "confidence": entity.confidence,
                "start": entity.start,
                "end": entity.end,
            }
            for entity in self.entities
        ]
        bio_payload = [asdict(token) for token in self.bio_tokens]
        participant_payload = [
            {
                "name": participant.name,
                "confidence": participant.confidence,
                "sources": list(participant.sources),
            }
            for participant in self.participants
        ]

        return {
            "participant_list": list(self.metadata.participants),
            "ner_participants": participant_payload,
            "ner_entities": entity_payload,
            "ner_bio_tags": bio_payload,
            "ner_inference_time_seconds": self.inference_time_seconds,
            "ner_used_fallback": self.used_fallback,
            "ner_error": self.error_message or "",
            "meeting_date": self.metadata.meeting_date,
            "meeting_time": self.metadata.meeting_time,
            "organization": self.metadata.organization,
            "location": self.metadata.location,
        }


def extract_meeting_metadata(transcript_text: str) -> EntityExtractionResult:
    """Extract meeting metadata from a transcript using BERT NER.

    Args:
        transcript_text: Raw or reviewed transcript text.

    Returns:
        ``EntityExtractionResult`` with metadata candidates. BERT failures are
        represented as ``used_fallback=True`` instead of exceptions.
    """

    parsed = parse_transcript(transcript_text)
    speaker_labels = speaker_labels_from_parsed_transcript(parsed)

    ner_result = BertNerService().extract_entities(transcript_text)
    if ner_result.error_message:
        participants = resolve_participants([], speaker_labels)
        metadata_hint_entities = extract_metadata_hint_entities(transcript_text)
        metadata = MeetingMetadataEntities(
            participants=[participant.name for participant in participants],
            meeting_date=_first_entity_text(metadata_hint_entities, "DATE"),
            meeting_time=_first_entity_text(metadata_hint_entities, "TIME"),
            organization=_first_entity_text(metadata_hint_entities, "ORG"),
            location=_first_entity_text(metadata_hint_entities, "LOC"),
        )
        return EntityExtractionResult(
            metadata=metadata,
            participants=participants,
            entities=metadata_hint_entities,
            inference_time_seconds=ner_result.inference_time_seconds,
            used_fallback=True,
            error_message=ner_result.error_message,
        )

    metadata_hint_entities = extract_metadata_hint_entities(transcript_text)
    combined_entities = [*ner_result.entities, *metadata_hint_entities]
    person_entities = [
        entity for entity in combined_entities if entity.entity_group == "PER"
    ]
    participants = resolve_participants(person_entities, speaker_labels)
    metadata = build_metadata_from_entities(combined_entities, participants)
    return EntityExtractionResult(
        metadata=metadata,
        participants=participants,
        entities=combined_entities,
        bio_tokens=ner_result.tokens,
        inference_time_seconds=ner_result.inference_time_seconds,
    )


def speaker_labels_from_parsed_transcript(parsed: ParsedTranscript) -> list[str]:
    """Extract ordered speaker labels from parsed transcript turns.

    Args:
        parsed: Parsed transcript result from the existing ML parser.

    Returns:
        Ordered speaker labels, or an empty list when parsing was invalid.
    """

    if not parsed.is_valid:
        return []
    return speaker_labels_from_turns(parsed.turns)


def speaker_labels_from_turns(turns: list[TranscriptTurn]) -> list[str]:
    """Extract ordered speaker labels from transcript turns.

    Args:
        turns: Parsed transcript turns.

    Returns:
        Ordered speaker names without duplicates.
    """

    labels: list[str] = []
    seen: set[str] = set()
    for turn in turns:
        speaker = (turn.speaker_normalized or turn.speaker_raw or "").strip()
        if not speaker:
            continue
        key = speaker.casefold()
        if key in seen:
            continue
        labels.append(speaker)
        seen.add(key)
    return labels


def build_metadata_from_entities(
    entities: list[BIONerEntity],
    participants: list[ResolvedParticipant],
) -> MeetingMetadataEntities:
    """Build meeting metadata from BERT entity spans.

    Args:
        entities: Merged BERT NER entity spans.
        participants: Resolved participant records.

    Returns:
        Metadata values for default meeting information.
    """

    return MeetingMetadataEntities(
        participants=[participant.name for participant in participants],
        meeting_date=_first_entity_text(entities, "DATE"),
        meeting_time=_first_entity_text(entities, "TIME"),
        organization=_first_entity_text(entities, "ORG"),
        location=_first_entity_text(entities, "LOC"),
    )


def _first_entity_text(entities: list[BIONerEntity], entity_group: str) -> str:
    """Return the first entity text for a group.

    Args:
        entities: Merged BERT NER entity spans.
        entity_group: Entity group to search for.

    Returns:
        First matching entity text, or an empty string.
    """

    for entity in entities:
        if entity.entity_group == entity_group and entity.text.strip():
            return entity.text.strip()
    return ""


def extract_metadata_hint_entities(transcript_text: str) -> list[BIONerEntity]:
    """Extract explicit meeting metadata hints not covered by the NER model.

    Args:
        transcript_text: Transcript text to inspect for metadata headings.

    Returns:
        BIONerEntity records for DATE, TIME, ORG, and LOC hints.

    Notes:
        The default ``dslim/bert-base-NER`` model emits PER/ORG/LOC tags but
        typically not DATE/TIME. These explicit heading hints keep metadata
        extraction useful while participant extraction remains BERT-backed and
        speaker-label enriched.
    """

    patterns = (
        ("DATE", re.compile(r"(?im)^\s*(?:meeting\s+date|date)\s*[:\-]\s*(?P<value>.+)$")),
        ("TIME", re.compile(r"(?im)^\s*(?:meeting\s+time|time)\s*[:\-]\s*(?P<value>.+)$")),
        (
            "ORG",
            re.compile(
                r"(?im)^\s*(?:organization|organisation|company)\s*[:\-]\s*(?P<value>.+)$"
            ),
        ),
        (
            "LOC",
            re.compile(
                r"(?im)^\s*(?:location|venue|meeting\s+location)\s*[:\-]\s*(?P<value>.+)$"
            ),
        ),
    )
    entities: list[BIONerEntity] = []
    for entity_group, pattern in patterns:
        match = pattern.search(transcript_text)
        if not match:
            continue
        value = " ".join(match.group("value").strip().split())
        if not value:
            continue
        entities.append(
            BIONerEntity(
                text=value,
                entity_group=entity_group,
                bio_tags=[f"B-{entity_group}"],
                confidence=1.0,
                start=match.start("value"),
                end=match.end("value"),
            )
        )
    return entities


def _session_entity_type(entity_group: str) -> str:
    """Map model entity groups to meeting metadata entity names.

    Args:
        entity_group: Model entity group such as ``PER`` or ``LOC``.

    Returns:
        Metadata-friendly entity name.
    """

    return {
        "PER": "PERSON",
        "ORG": "ORGANIZATION",
        "LOC": "LOCATION",
        "DATE": "DATE",
        "TIME": "TIME",
    }.get(entity_group, entity_group)
