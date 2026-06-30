"""Transcript parser for the future ML-based Minutes of Meeting pipeline.

Purpose:
    Convert reviewed meeting transcript text into normalized speaker turns that
    downstream ML components can process safely.

Responsibilities:
    - Detect common speaker label formats using conservative regular
      expressions.
    - Normalize speaker names and timestamps without changing the transcript
      content itself.
    - Split each speaker turn into lightweight sentence units.
    - Return friendly validation results for empty or speakerless transcripts.
    - Never raise user-facing exceptions for malformed transcript content.

Inputs:
    Raw or reviewed transcript text, typically after upload, speaker review, and
    transcript review have already happened elsewhere in the application.

Outputs:
    ``ParsedTranscript`` containing normalized ``TranscriptTurn`` objects with
    fields shaped for later preprocessing, feature extraction, embeddings,
    clustering, and classification.

Future Implementation Notes:
    This module is intentionally not integrated with the Streamlit UI yet.
    Later ML phases can consume ``ParsedTranscript.turns`` as the first stage of
    the MoM pipeline.
"""

from dataclasses import dataclass, field
import re
import sys


NO_SPEAKER_VALIDATION_MESSAGE = (
    "Speaker information could not be identified.\n"
    "Please upload a transcript containing speaker labels such as:\n\n"
    "Speaker 1:\n"
    "Rahul:\n"
    "Priya [00:10]"
)

EMPTY_TRANSCRIPT_VALIDATION_MESSAGE = (
    "Transcript is empty. Please upload a transcript containing speaker labels "
    'such as "Speaker 1:", "Rahul:", or "Priya [00:10]".'
)

_SPEAKER_NAME_PATTERN = (
    r"(?:Speaker\s+[A-Za-z0-9]+|[^\W\d_][\w ._'&-]{0,60})"
)
_TIMESTAMP_CORE_PATTERN = r"\d{1,2}:\d{2}(?::\d{2})?"
_TIMESTAMP_TOKEN_PATTERN = (
    rf"(?:\[\s*{_TIMESTAMP_CORE_PATTERN}\s*\]"
    rf"|\(\s*{_TIMESTAMP_CORE_PATTERN}\s*\)"
    rf"|{_TIMESTAMP_CORE_PATTERN})"
)
_COLON_SPEAKER_RE = re.compile(
    rf"^\s*(?P<speaker>{_SPEAKER_NAME_PATTERN})\s*:\s*(?P<inline_text>.*)$"
)
_SPEAKER_TIMESTAMP_RE = re.compile(
    rf"^\s*(?P<speaker>{_SPEAKER_NAME_PATTERN})\s*"
    rf"(?P<timestamp>{_TIMESTAMP_TOKEN_PATTERN})"
    r"\s*:?\s*(?P<inline_text>.*)$"
)
_TIMESTAMP_SPEAKER_RE = re.compile(
    rf"^\s*(?P<timestamp>{_TIMESTAMP_TOKEN_PATTERN})\s+"
    rf"(?P<speaker>{_SPEAKER_NAME_PATTERN})\s*:?\s*(?P<inline_text>.*)$"
)
_TIMESTAMP_ONLY_RE = re.compile(rf"^\s*(?P<timestamp>{_TIMESTAMP_TOKEN_PATTERN})\s*$")
_GENERIC_SPEAKER_ONLY_RE = re.compile(r"^\s*(?P<speaker>Speaker\s+[A-Za-z0-9]+)\s*$")
_DECORATIVE_SEPARATOR_RE = re.compile(r"^\s*[-_=*#~]{3,}\s*$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(slots=True)
class TranscriptTurn:
    """Normalized speaker turn extracted from a transcript.

    Attributes:
        turn_id: Sequential one-based turn identifier.
        speaker_raw: Speaker label exactly as detected, after edge whitespace
            is trimmed.
        speaker_normalized: Canonical speaker label for downstream processing.
        timestamp_raw: Timestamp exactly as detected, including brackets or
            parentheses when present.
        timestamp_seconds: Timestamp converted to seconds when possible.
        text: Full text associated with the speaker turn.
        sentence_list: Sentence-level text units derived from ``text``.
    """

    turn_id: int
    speaker_raw: str
    speaker_normalized: str
    timestamp_raw: str | None
    timestamp_seconds: int | None
    text: str
    sentence_list: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ParsedTranscript:
    """Container for transcript parsing output and validation state.

    Attributes:
        turns: Parsed and normalized speaker turns.
        has_speaker_information: Whether at least one speaker label was found.
        is_valid: Whether parsing produced usable speaker-aware content.
        validation_message: Friendly message explaining a blocking validation
            issue, if one exists.
        warnings: Non-blocking parser warnings for diagnostics.
    """

    turns: list[TranscriptTurn] = field(default_factory=list)
    has_speaker_information: bool = False
    is_valid: bool = True
    validation_message: str | None = None
    warnings: list[str] = field(default_factory=list)


def parse_transcript(transcript_text: str) -> ParsedTranscript:
    """Parse transcript text into normalized speaker turns.

    Args:
        transcript_text: Transcript text containing supported speaker labels.

    Returns:
        Parsed transcript result. The result contains validation details instead
        of raising exceptions when input is empty, malformed, or speakerless.
    """

    try:
        return _parse_transcript_safely(transcript_text)
    except Exception as exc:  # pragma: no cover - defensive safety net.
        # The parser is the first ML pipeline boundary, so it must degrade into
        # a friendly validation response rather than breaking the application.
        return ParsedTranscript(
            is_valid=False,
            validation_message=NO_SPEAKER_VALIDATION_MESSAGE,
            warnings=[f"Parser recovered from unexpected error: {exc}"],
        )


def parse_transcript_placeholder(transcript_text: str) -> ParsedTranscript:
    """Backward-compatible alias for the skeleton parser entry point.

    Args:
        transcript_text: Transcript text containing supported speaker labels.

    Returns:
        Parsed transcript result from ``parse_transcript``.
    """

    return parse_transcript(transcript_text)


def _parse_transcript_safely(transcript_text: str) -> ParsedTranscript:
    """Parse transcript text without the outer defensive exception wrapper.

    Args:
        transcript_text: Transcript text containing supported speaker labels.

    Returns:
        Parsed transcript with validation information.
    """

    if not isinstance(transcript_text, str):
        transcript_text = "" if transcript_text is None else str(transcript_text)

    transcript_text = normalize_transcript_text(transcript_text)

    if not transcript_text.strip():
        return ParsedTranscript(
            is_valid=False,
            validation_message=EMPTY_TRANSCRIPT_VALIDATION_MESSAGE,
        )

    turns: list[TranscriptTurn] = []
    current_speaker_raw: str | None = None
    current_timestamp_raw: str | None = None
    current_lines: list[str] = []
    pending_timestamp_raw: str | None = None
    warnings: list[str] = []

    for line in transcript_text.splitlines():
        if is_decorative_separator(line):
            # Decorative separators are layout noise from copied transcripts.
            # Ignoring them avoids polluting the previous speaker's dialogue.
            continue

        if not line.strip():
            # Blank lines separate speaker blocks visually, but they should not
            # close a turn because wrapped dialogue can contain blank spacing.
            continue

        timestamp_only = detect_timestamp_only(line)
        if timestamp_only is not None:
            # Some transcripts emit a timestamp on its own line and put the
            # speaker on the next line. Store it for the next detected speaker.
            pending_timestamp_raw = timestamp_only
            continue

        parsed_label = detect_speaker_label(
            line,
            pending_timestamp_raw=pending_timestamp_raw,
        )

        if parsed_label is not None:
            if current_speaker_raw is not None:
                _append_turn(
                    turns=turns,
                    speaker_raw=current_speaker_raw,
                    timestamp_raw=current_timestamp_raw,
                    text_lines=current_lines,
                )

            current_speaker_raw = parsed_label["speaker_raw"]
            current_timestamp_raw = parsed_label["timestamp_raw"]
            current_lines = []
            pending_timestamp_raw = None

            # Some transcripts place speech on the same line as the speaker
            # label; preserving it here prevents quiet data loss.
            if parsed_label["inline_text"]:
                current_lines.append(parsed_label["inline_text"])
            continue

        if current_speaker_raw is not None:
            # Non-label lines belong to the most recent speaker because reviewed
            # transcripts commonly wrap speaker turns across multiple lines.
            current_lines.append(line)
        elif line.strip():
            # Pre-speaker text is ignored for turn creation but noted for later
            # diagnostics, since headers can appear before the first speaker.
            warnings.append("Ignored text before the first detected speaker label.")

    if current_speaker_raw is not None:
        _append_turn(
            turns=turns,
            speaker_raw=current_speaker_raw,
            timestamp_raw=current_timestamp_raw,
            text_lines=current_lines,
        )

    if not turns:
        return ParsedTranscript(
            is_valid=False,
            validation_message=NO_SPEAKER_VALIDATION_MESSAGE,
            warnings=warnings,
        )

    return ParsedTranscript(
        turns=turns,
        has_speaker_information=True,
        is_valid=True,
        warnings=_deduplicate_warnings(warnings),
    )


def detect_speaker_label(
    line: str,
    pending_timestamp_raw: str | None = None,
) -> dict[str, str | None] | None:
    """Detect a supported speaker label at the start of a line.

    Args:
        line: Single transcript line to inspect.
        pending_timestamp_raw: Timestamp captured from a previous timestamp-only
            line, if present.

    Returns:
        A dictionary containing ``speaker_raw``, ``timestamp_raw``, and
        ``inline_text`` when a label is detected; otherwise ``None``.
    """

    # Rule 1: timestamp-first labels, e.g. "[10:00:05] Rahul:" or
    # "10:00:05 Rahul:". This must run before speaker-first matching so the
    # timestamp is preserved rather than treated as dialogue text.
    timestamp_speaker_match = _TIMESTAMP_SPEAKER_RE.match(line)
    if timestamp_speaker_match:
        return {
            "speaker_raw": _clean_speaker_label(timestamp_speaker_match.group("speaker")),
            "timestamp_raw": _clean_timestamp(timestamp_speaker_match.group("timestamp")),
            "inline_text": timestamp_speaker_match.group("inline_text").strip(),
        }

    # Rule 2: speaker-first timestamp labels, e.g. "Rahul [10:00]" and
    # "Priya (10:00)". Inline text after the timestamp is preserved.
    speaker_timestamp_match = _SPEAKER_TIMESTAMP_RE.match(line)
    if speaker_timestamp_match:
        return {
            "speaker_raw": _clean_speaker_label(speaker_timestamp_match.group("speaker")),
            "timestamp_raw": _clean_timestamp(speaker_timestamp_match.group("timestamp")),
            "inline_text": speaker_timestamp_match.group("inline_text").strip(),
        }

    # Rule 3: standard colon labels, e.g. "Speaker 1:" or "Rahul:".
    colon_match = _COLON_SPEAKER_RE.match(line)
    if colon_match:
        return {
            "speaker_raw": _clean_speaker_label(colon_match.group("speaker")),
            "timestamp_raw": pending_timestamp_raw,
            "inline_text": colon_match.group("inline_text").strip(),
        }

    # Rule 4: generic speaker-only labels such as "Speaker A". These appear in
    # diarized transcripts and should start a new turn even without a colon.
    generic_speaker_only_match = _GENERIC_SPEAKER_ONLY_RE.match(line)
    if generic_speaker_only_match:
        return {
            "speaker_raw": _clean_speaker_label(
                generic_speaker_only_match.group("speaker")
            ),
            "timestamp_raw": pending_timestamp_raw,
            "inline_text": "",
        }

    # Rule 5: if a timestamp-only line was just seen, the next plausible line is
    # allowed to be a speaker name even without a colon.
    if pending_timestamp_raw and is_plausible_standalone_speaker(line):
        return {
            "speaker_raw": _clean_speaker_label(line),
            "timestamp_raw": pending_timestamp_raw,
            "inline_text": "",
        }

    return None


def normalize_transcript_text(transcript_text: str) -> str:
    """Normalize file-level transcript text before line parsing.

    Args:
        transcript_text: Raw transcript text.

    Returns:
        Transcript text with BOM removed and line endings normalized.
    """

    # UTF-8 BOMs can appear at the beginning of uploaded files and otherwise
    # break first-speaker detection. Normalizing line endings makes splitlines
    # behavior consistent across Windows and Unix files.
    return transcript_text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")


def is_decorative_separator(line: str) -> bool:
    """Return whether a line is a decorative separator.

    Args:
        line: Transcript line.

    Returns:
        ``True`` when the line is made only of separator characters.
    """

    return bool(_DECORATIVE_SEPARATOR_RE.fullmatch(line))


def detect_timestamp_only(line: str) -> str | None:
    """Detect a line that contains only a timestamp.

    Args:
        line: Transcript line.

    Returns:
        Clean timestamp token, or ``None`` when the line is not timestamp-only.
    """

    match = _TIMESTAMP_ONLY_RE.match(line)
    if not match:
        return None
    return _clean_timestamp(match.group("timestamp"))


def is_plausible_standalone_speaker(line: str) -> bool:
    """Return whether a line can safely be treated as a speaker after a timestamp.

    Args:
        line: Transcript line after a timestamp-only line.

    Returns:
        ``True`` when the line resembles a short speaker label.
    """

    candidate = _clean_speaker_label(line)
    if not candidate or len(candidate.split()) > 4:
        return False
    if any(mark in candidate for mark in ".!?"):
        return False
    return bool(re.fullmatch(_SPEAKER_NAME_PATTERN, candidate))


def normalize_speaker_label(speaker_raw: str) -> str:
    """Normalize a raw speaker label for downstream ML stages.

    Args:
        speaker_raw: Speaker label captured from the transcript.

    Returns:
        Canonical speaker label with compact spacing.
    """

    speaker = _clean_speaker_label(speaker_raw)
    speaker_match = re.fullmatch(r"Speaker\s+([A-Za-z0-9]+)", speaker, re.IGNORECASE)
    if speaker_match:
        # Canonicalizing generic speaker labels makes mixed transcript formats
        # easier to compare without changing real participant names.
        return f"Speaker {speaker_match.group(1).upper()}"
    return speaker


def timestamp_to_seconds(timestamp_raw: str | None) -> int | None:
    """Convert a timestamp string to seconds.

    Args:
        timestamp_raw: Timestamp such as ``[00:10]`` or ``(01:02:03)``.

    Returns:
        Timestamp converted to seconds, or ``None`` when conversion is not
        possible.
    """

    if not timestamp_raw:
        return None

    cleaned = _clean_timestamp(timestamp_raw).strip("[]() ")
    parts = cleaned.split(":")
    if len(parts) not in {2, 3}:
        return None

    try:
        numbers = [int(part) for part in parts]
    except ValueError:
        return None

    if len(numbers) == 2:
        minutes, seconds = numbers
        return minutes * 60 + seconds

    hours, minutes, seconds = numbers
    return hours * 3600 + minutes * 60 + seconds


def split_sentences(text: str) -> list[str]:
    """Split turn text into simple sentence units.

    Args:
        text: Speaker turn text.

    Returns:
        List of non-empty sentence strings.
    """

    # This intentionally remains lightweight because richer segmentation should
    # be evaluated with the future ML preprocessing stage.
    sentence_candidates = _SENTENCE_SPLIT_RE.split(text.strip())
    return [candidate.strip() for candidate in sentence_candidates if candidate.strip()]


def _append_turn(
    turns: list[TranscriptTurn],
    speaker_raw: str,
    timestamp_raw: str | None,
    text_lines: list[str],
) -> None:
    """Append a normalized transcript turn to the turn list.

    Args:
        turns: Mutable list receiving parsed turns.
        speaker_raw: Raw speaker label for this turn.
        timestamp_raw: Optional timestamp label for this turn.
        text_lines: Lines of transcript text associated with the speaker.
    """

    text = _normalize_turn_text(text_lines)
    turns.append(
        TranscriptTurn(
            turn_id=len(turns) + 1,
            speaker_raw=speaker_raw,
            speaker_normalized=normalize_speaker_label(speaker_raw),
            timestamp_raw=timestamp_raw,
            timestamp_seconds=timestamp_to_seconds(timestamp_raw),
            text=text,
            sentence_list=split_sentences(text),
        )
    )


def _normalize_turn_text(text_lines: list[str]) -> str:
    """Normalize internal whitespace for one speaker turn.

    Args:
        text_lines: Lines associated with one speaker.

    Returns:
        Cleaned speaker-turn text.
    """

    # Blank lines are dropped because they are visual separators in transcripts,
    # not semantic content for the ML pipeline.
    cleaned_lines = [line.strip() for line in text_lines if line.strip()]
    return " ".join(cleaned_lines)


def _clean_speaker_label(speaker: str) -> str:
    """Clean spacing around a speaker label.

    Args:
        speaker: Raw speaker label.

    Returns:
        Speaker label with repeated whitespace collapsed.
    """

    return re.sub(r"\s+", " ", speaker.strip())


def _clean_timestamp(timestamp: str) -> str:
    """Clean internal whitespace in a timestamp token.

    Args:
        timestamp: Raw timestamp token.

    Returns:
        Timestamp token with compact internal spacing.
    """

    return re.sub(r"\s+", "", timestamp.strip())


def _deduplicate_warnings(warnings: list[str]) -> list[str]:
    """Deduplicate warnings while preserving their original order.

    Args:
        warnings: Warning messages collected during parsing.

    Returns:
        Ordered warning list without duplicates.
    """

    unique_warnings: list[str] = []
    for warning in warnings:
        if warning not in unique_warnings:
            unique_warnings.append(warning)
    return unique_warnings


def _run_example(name: str, transcript_text: str) -> None:
    """Print parser output for one unit-test style example.

    Args:
        name: Example name.
        transcript_text: Transcript sample to parse.
    """

    result = parse_transcript(transcript_text)
    print(f"\n{name}")
    print(f"valid={result.is_valid} speakers={result.has_speaker_information}")
    if result.validation_message:
        print(result.validation_message)
    for turn in result.turns:
        print(
            {
                "turn_id": turn.turn_id,
                "speaker_raw": turn.speaker_raw,
                "speaker_normalized": turn.speaker_normalized,
                "timestamp_raw": turn.timestamp_raw,
                "timestamp_seconds": turn.timestamp_seconds,
                "text": turn.text,
                "sentence_list": turn.sentence_list,
            }
        )


def _assert_parse_case(
    name: str,
    transcript_text: str,
    expected_speakers: list[str],
    expected_text_fragments: list[str],
    expected_timestamps: list[str | None] | None = None,
) -> None:
    """Assert parser behavior for one transcript format.

    Args:
        name: Test case name.
        transcript_text: Transcript sample to parse.
        expected_speakers: Expected raw speaker labels in order.
        expected_text_fragments: Text fragments expected in matching turns.
        expected_timestamps: Optional expected raw timestamps in order.
    """

    result = parse_transcript(transcript_text)
    assert result.is_valid, f"{name}: expected valid parse, got {result.validation_message}"
    assert [turn.speaker_raw for turn in result.turns] == expected_speakers, name
    for index, expected_fragment in enumerate(expected_text_fragments):
        assert expected_fragment in result.turns[index].text, (
            f"{name}: missing text fragment {expected_fragment!r}"
        )
    if expected_timestamps is not None:
        assert [turn.timestamp_raw for turn in result.turns] == expected_timestamps, name


def _run_parser_tests() -> None:
    """Run unit-test style parser cases for supported transcript formats."""

    cases = [
        (
            "01 numbered speakers with colon",
            "Speaker 1:\nHello\n\nSpeaker 2:\nHi",
            ["Speaker 1", "Speaker 2"],
            ["Hello", "Hi"],
            [None, None],
        ),
        (
            "02 named speakers with colon",
            "Rahul:\nHello\n\nPriya:\nHi",
            ["Rahul", "Priya"],
            ["Hello", "Hi"],
            [None, None],
        ),
        (
            "03 bracket timestamp first",
            "[10:00:05] Rahul:\nHello",
            ["Rahul"],
            ["Hello"],
            ["[10:00:05]"],
        ),
        (
            "04 bare timestamp first",
            "10:00:05 Rahul:\nHello",
            ["Rahul"],
            ["Hello"],
            ["10:00:05"],
        ),
        (
            "05 speaker bracket timestamp",
            "Rahul [10:00]\nHello",
            ["Rahul"],
            ["Hello"],
            ["[10:00]"],
        ),
        (
            "06 speaker parenthesis timestamp",
            "Rahul (10:00)\nHello",
            ["Rahul"],
            ["Hello"],
            ["(10:00)"],
        ),
        (
            "07 timestamp-only then colon speaker",
            "[10:00:05]\nRahul:\nHello",
            ["Rahul"],
            ["Hello"],
            ["[10:00:05]"],
        ),
        (
            "08 timestamp-only then bare speaker",
            "10:00:05\nRahul\nHello",
            ["Rahul"],
            ["Hello"],
            ["10:00:05"],
        ),
        (
            "09 multiline speaker block",
            "Rahul:\nFirst sentence.\nSecond sentence.\nThird sentence.",
            ["Rahul"],
            ["First sentence. Second sentence. Third sentence."],
            [None],
        ),
        (
            "10 blank lines between speakers",
            "Rahul:\nHello\n\n\nPriya:\nHi",
            ["Rahul", "Priya"],
            ["Hello", "Hi"],
            [None, None],
        ),
        (
            "11 mixed formats",
            "Speaker 1:\nHello\n[10:00:05] Rahul:\nHi\nPriya (10:01)\nOkay",
            ["Speaker 1", "Rahul", "Priya"],
            ["Hello", "Hi", "Okay"],
            [None, "[10:00:05]", "(10:01)"],
        ),
        (
            "12 BOM first speaker",
            "\ufeffRahul:\nFirst sentence.",
            ["Rahul"],
            ["First sentence."],
            [None],
        ),
        (
            "13 Windows line endings",
            "Rahul:\r\nHello there.\r\nPriya:\r\nGood morning.",
            ["Rahul", "Priya"],
            ["Hello there.", "Good morning."],
            [None, None],
        ),
        (
            "14 decorative separators ignored",
            "-----\nRahul:\nHello\n=====\nPriya:\nHi",
            ["Rahul", "Priya"],
            ["Hello", "Hi"],
            [None, None],
        ),
        (
            "15 generic speaker-only label",
            "Speaker A\nHello\nSpeaker B\nHi",
            ["Speaker A", "Speaker B"],
            ["Hello", "Hi"],
            [None, None],
        ),
        (
            "16 inline label text preserved",
            "Rahul: Hello first sentence.\nPriya [10:02] Hi second sentence.",
            ["Rahul", "Priya"],
            ["Hello first sentence.", "Hi second sentence."],
            [None, "[10:02]"],
        ),
    ]

    for case in cases:
        _assert_parse_case(*case)

    empty_result = parse_transcript("")
    assert not empty_result.is_valid
    assert empty_result.validation_message == EMPTY_TRANSCRIPT_VALIDATION_MESSAGE

    no_speaker_result = parse_transcript("Hello.\nThis has no speaker names.")
    assert not no_speaker_result.is_valid
    assert no_speaker_result.validation_message == NO_SPEAKER_VALIDATION_MESSAGE

    print(f"\nParser test cases passed: {len(cases) + 2}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    _run_example(
        "Format 1 - numbered speakers",
        "Speaker 1:\nHello everyone.\n\nSpeaker 2:\nGood morning.",
    )
    _run_example(
        "Format 2 - lettered speakers",
        "Speaker A:\nHello.\n\nSpeaker B:\nHi.",
    )
    _run_example(
        "Format 3 - named speakers",
        "Rahul:\nHello.\n\nPriya:\nHi.",
    )
    _run_example(
        "Format 4 - bracket timestamps",
        "Rahul [00:10]\nThe proposal is ready.\n\nPriya [00:22]\nLet's review it.",
    )
    _run_example(
        "Format 5 - parenthesis timestamps",
        "Rahul (00:10)\nHello.\n\nPriya (00:25)\nHi.",
    )
    _run_example(
        "Format 6 - mixed transcript",
        "Speaker 1:\nHello\n\nRahul:\nHi\n\nSpeaker A:\nOkay",
    )
    _run_example(
        "Validation - no speakers",
        "Hello everyone.\nThis transcript has no speaker labels.",
    )

    _run_parser_tests()
