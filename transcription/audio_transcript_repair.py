"""Deterministic cleanup for transcripts produced by audio speech-to-text.

This module is intentionally audio-specific.  It does not summarize, infer
missing words, classify sentences, or call a language model.  Repairs are
limited to surface normalization, unambiguous filler/repetition removal,
configured vocabulary substitutions, and conservative punctuation restoration
inside existing speaker segments.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType

from transcription.sarvam_client import TranscriptionResult, TranscriptionSegment
from transcription.speaker_mapping import default_speaker_label


DEFAULT_TECHNICAL_VOCABULARY: Mapping[str, str] = MappingProxyType(
    {
        "speaker diaries": "speaker diarization",
        "speaker diarisation": "speaker diarization",
        "mini lm": "MiniLM",
        "ollama": "Ollama",
        "gemma": "Gemma",
    }
)

_QUESTION_START = (
    r"(?:how\s+(?:do|does|did|can|could|should|would|will|is|are)|"
    r"what\s+(?:do|does|did|can|could|should|would|will|is|are)|"
    r"why\s+(?:do|does|did|can|could|should|would|will|is|are)|"
    r"when\s+(?:do|does|did|can|could|should|would|will|is|are)|"
    r"where\s+(?:do|does|did|can|could|should|would|will|is|are)|"
    r"who\s+(?:can|could|should|would|will|is|are)|"
    r"which\s+(?:can|could|should|would|will|is|are)|"
    r"can\s+you|could\s+you|should\s+we|would\s+you|"
    r"do\s+we|does\s+it|did\s+we|is\s+it|are\s+we|will\s+we)"
)
_QUESTION_CLAUSE_RE = re.compile(
    rf",\s+(?P<starter>{_QUESTION_START})\b",
    re.IGNORECASE,
)

_UNICODE_PUNCTUATION = str.maketrans(
    {
        "\u00a0": " ",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
    }
)
_UNAMBIGUOUS_FILLER_RE = re.compile(
    r"(?<![\w'-])(?:u+m+|u+h+|e+r+m+|h+m+)(?![\w'-])",
    re.IGNORECASE,
)


def repair_audio_transcription(
    result: TranscriptionResult,
    *,
    technical_vocabulary: Mapping[str, str] | None = None,
    confirmed_speaker_mapping: Mapping[str, str] | None = None,
) -> TranscriptionResult:
    """Return a repaired copy of an audio transcription result.

    Segment timestamps and order are retained exactly. Speaker identifiers are
    changed only by an explicit confirmed mapping. The raw payload is preserved.
    """

    vocabulary = (
        DEFAULT_TECHNICAL_VOCABULARY
        if technical_vocabulary is None
        else technical_vocabulary
    )
    repaired_segments = [
        _repair_segment(
            segment,
            vocabulary=vocabulary,
            confirmed_speaker_mapping=confirmed_speaker_mapping,
        )
        for segment in result.segments
    ]
    repaired_segments = _deduplicate_adjacent_segments(repaired_segments)
    repaired_transcript = repair_audio_text(
        result.transcript,
        technical_vocabulary=vocabulary,
    )

    if (
        repaired_transcript == result.transcript
        and repaired_segments == result.segments
    ):
        return result
    return replace(
        result,
        transcript=repaired_transcript,
        segments=repaired_segments,
    )


def _deduplicate_adjacent_segments(
    segments: list[TranscriptionSegment],
) -> list[TranscriptionSegment]:
    """Drop only adjacent duplicate STT segments from the same speaker.

    Providers occasionally return the same finalized utterance twice at a
    segment boundary.  Restricting this repair to adjacent, same-speaker text
    avoids removing intentional repetition elsewhere in the meeting.
    """

    deduplicated: list[TranscriptionSegment] = []
    for segment in segments:
        if not deduplicated:
            deduplicated.append(segment)
            continue
        previous = deduplicated[-1]
        if (
            str(previous.speaker_id or "").strip()
            == str(segment.speaker_id or "").strip()
            and _segment_text_key(previous.transcript)
            == _segment_text_key(segment.transcript)
        ):
            continue
        deduplicated.append(segment)
    return deduplicated


def _segment_text_key(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def repair_audio_text(
    text: str,
    *,
    technical_vocabulary: Mapping[str, str] | None = None,
) -> str:
    """Apply conservative deterministic repairs to STT text."""

    vocabulary = (
        DEFAULT_TECHNICAL_VOCABULARY
        if technical_vocabulary is None
        else technical_vocabulary
    )
    repaired = _normalize_surface(text)
    repaired = _remove_adjacent_repetitions(repaired)
    repaired = _restore_repeated_question_boundaries(repaired)
    repaired = _normalize_configured_vocabulary(repaired, vocabulary)
    repaired = _remove_unambiguous_fillers(repaired)
    repaired = _normalize_surface(repaired)
    repaired = _capitalize_sentence_starts(repaired)
    repaired = _ensure_terminal_punctuation(repaired)
    return repaired


def _normalize_surface(text: str) -> str:
    """Normalize typography and spacing without changing spoken meaning."""

    repaired = str(text).translate(_UNICODE_PUNCTUATION)
    repaired = re.sub(r"[\t\r\f\v]+", " ", repaired)
    repaired = re.sub(r" *\n+ *", " ", repaired)
    repaired = re.sub(r" {2,}", " ", repaired)
    repaired = re.sub(r"\s+([,.;:!?])", r"\1", repaired)
    repaired = re.sub(r"([,;:!?])(?=[A-Za-z])", r"\1 ", repaired)
    repaired = re.sub(r"([!?])\1+", r"\1", repaired)
    repaired = re.sub(r"\.{4,}", "...", repaired)
    return repaired.strip()


def _remove_unambiguous_fillers(text: str) -> str:
    """Remove only hesitation tokens with no meeting-content meaning.

    Hinglish discourse words such as ``haan``, ``toh``, ``matlab``, ``kal``
    and ``tak`` are intentionally retained because they can carry intent.
    """

    repaired = _UNAMBIGUOUS_FILLER_RE.sub("", text)
    repaired = re.sub(r"(^|[.!?])\s*[,;:]\s*", r"\1 ", repaired)
    repaired = re.sub(r",\s*(?=[.!?]|$)", "", repaired)
    return repaired


def _capitalize_sentence_starts(text: str) -> str:
    repaired = re.sub(r"\bi\b", "I", text)
    return re.sub(
        r"(^|[.!?]\s+)([a-z])",
        lambda match: match.group(1) + match.group(2).upper(),
        repaired,
    )


def _ensure_terminal_punctuation(text: str) -> str:
    if not text or re.search(r"[.!?](?:['\"\)\]]*)$", text):
        return text
    return f"{text}."


def _repair_segment(
    segment: TranscriptionSegment,
    *,
    vocabulary: Mapping[str, str],
    confirmed_speaker_mapping: Mapping[str, str] | None,
) -> TranscriptionSegment:
    speaker_id = segment.speaker_id
    if confirmed_speaker_mapping:
        label = default_speaker_label(segment)
        confirmed_name = str(confirmed_speaker_mapping.get(label, "")).strip()
        if confirmed_name:
            speaker_id = confirmed_name

    repaired_text = repair_audio_text(
        segment.transcript,
        technical_vocabulary=vocabulary,
    )
    if repaired_text == segment.transcript and speaker_id == segment.speaker_id:
        return segment
    return replace(segment, transcript=repaired_text, speaker_id=speaker_id)


def _remove_adjacent_repetitions(text: str) -> str:
    repaired = text
    # Longer repeated phrases are removed before single repeated words. Only
    # whitespace-separated exact repetitions are touched; punctuated emphasis
    # such as "very, very" remains intact.
    for word_count in range(4, 0, -1):
        word = r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*"
        phrase = rf"{word}(?:[ \t]+{word}){{{word_count - 1}}}"
        pattern = re.compile(
            rf"(?<![\w’'-])(?P<phrase>{phrase})[ \t]+(?P=phrase)(?![\w’'-])",
            re.IGNORECASE,
        )
        while True:
            repaired, replacements = pattern.subn(r"\g<phrase>", repaired)
            if not replacements:
                break
    return repaired


def _restore_repeated_question_boundaries(text: str) -> str:
    """Split comma-joined repeated interrogative clauses.

    A comma is converted only when the current sentence already contains an
    interrogative starter and the next clause begins with another one. This
    repairs STT run-ons such as "how do we use it, how do we deploy it" without
    creating wording or splitting ordinary conjunctions.
    """

    output: list[str] = []
    cursor = 0
    for match in _QUESTION_CLAUSE_RE.finditer(text):
        sentence_start = max(
            text.rfind(".", cursor, match.start()),
            text.rfind("?", cursor, match.start()),
            text.rfind("!", cursor, match.start()),
        )
        prefix = text[sentence_start + 1 : match.start()]
        repeated_starter = re.compile(
            rf"\b{re.escape(match.group('starter'))}\b",
            re.IGNORECASE,
        )
        if repeated_starter.search(prefix):
            output.append(text[cursor : match.start()])
            output.append("? ")
            cursor = match.start("starter")
    output.append(text[cursor:])
    return "".join(output)


def _normalize_configured_vocabulary(
    text: str,
    vocabulary: Mapping[str, str],
) -> str:
    repaired = text
    for source, replacement in sorted(
        vocabulary.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        source = str(source).strip()
        if not source:
            continue
        pattern = re.compile(
            rf"(?<!\w){re.escape(source)}(?!\w)",
            re.IGNORECASE,
        )
        repaired = pattern.sub(lambda _match, value=str(replacement): value, repaired)
    return repaired
