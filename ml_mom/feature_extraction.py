"""Rule-based feature extraction for the future ML MoM pipeline.

Purpose:
    Convert preprocessed transcript turns into sentence-level feature records
    that later clustering and ANN classification stages can consume.

Responsibilities:
    - Produce one ``SentenceFeature`` record per transcript sentence.
    - Extract lexical features such as word count and character count.
    - Extract structural features such as question, exclamation, number,
      currency, percentage, date, and time signals.
    - Extract meeting-specific keyword signals for actions, decisions,
      discussions, and deadlines.
    - Preserve speaker and sentence context for future ML use.

Inputs:
    Preprocessed ``TranscriptTurn`` objects from ``ml_mom.preprocessing``.

Outputs:
    ``SentenceFeature`` objects containing explainable rule-based features.

Future Implementation Notes:
    This module intentionally performs no clustering, classification, embedding
    generation, or ANN inference. Future ML stages can combine these features
    with sentence embeddings.
"""

from dataclasses import dataclass
import re
import sys

try:
    from ml_mom.transcript_parser import TranscriptTurn
except ModuleNotFoundError:  # pragma: no cover - supports direct script demos.
    from transcript_parser import TranscriptTurn


DECISION_KEYWORDS = frozenset(
    {
        "approved",
        "agreed",
        "accepted",
        "confirmed",
        "decided",
    }
)

ACTION_KEYWORDS = frozenset(
    {
        "will",
        "shall",
        "assign",
        "complete",
        "submit",
        "prepare",
        "finish",
        "deliver",
    }
)

DEADLINE_KEYWORDS = frozenset(
    {
        "today",
        "tomorrow",
        "friday",
        "monday",
        "deadline",
        "due",
        "before",
    }
)

DISCUSSION_KEYWORDS = frozenset(
    {
        "discuss",
        "review",
        "consider",
        "propose",
        "brainstorm",
        "analyze",
    }
)

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[.'_-][A-Za-z0-9]+)*")
_NUMBER_RE = re.compile(r"(?<!\w)\d+(?:,\d{2,3})*(?:\.\d+)?(?!\w)")
_CURRENCY_RE = re.compile(
    r"(?i)(?:[$€£₹]\s*\d|\b(?:usd|eur|gbp|inr|rs\.?)\s*\d)"
)
_PERCENTAGE_RE = re.compile(
    r"(?i)(?<!\w)\d+(?:\.\d+)?\s*(?:%|\bpercent\b)"
)
_DATE_RE = re.compile(
    r"(?ix)"
    r"("
    r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b"
    r"|"
    r"\b(?:jan|january|feb|february|mar|march|apr|april|may|jun|june|"
    r"jul|july|aug|august|sep|sept|september|oct|october|nov|november|"
    r"dec|december)\s+\d{1,2}(?:,\s*\d{4})?\b"
    r"|"
    r"\b\d{1,2}\s+(?:jan|january|feb|february|mar|march|apr|april|may|"
    r"jun|june|jul|july|aug|august|sep|sept|september|oct|october|"
    r"nov|november|dec|december)(?:\s+\d{4})?\b"
    r")"
)
_TIME_RE = re.compile(
    r"(?ix)"
    r"("
    r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm)?\b"
    r"|"
    r"\b\d{1,2}\s*(?:am|pm)\b"
    r")"
)


@dataclass(slots=True)
class SentenceFeature:
    """Feature record for one sentence in a transcript.

    Attributes:
        sentence_id: Sequential one-based sentence identifier.
        turn_id: Source speaker turn identifier.
        speaker: Normalized speaker label.
        timestamp: Source turn timestamp in seconds, when available.
        original_sentence: Preprocessed sentence text used for feature
            extraction.
        word_count: Number of word-like tokens in the sentence.
        character_count: Number of characters in the sentence.
        average_word_length: Mean word length across word-like tokens.
        sentence_position: Sequential one-based position in the transcript.
        is_question: Whether the sentence is phrased as a question.
        is_exclamation: Whether the sentence contains an exclamation signal.
        contains_number: Whether numeric content appears.
        contains_currency: Whether currency values or currency names appear.
        contains_percentage: Whether percentages appear.
        contains_date: Whether date-like content appears.
        contains_time: Whether time-like content appears.
        contains_action_keyword: Whether an action keyword appears.
        contains_decision_keyword: Whether a decision keyword appears.
        contains_discussion_keyword: Whether a discussion keyword appears.
        contains_deadline_keyword: Whether a deadline keyword appears.
        speaker_name: Normalized speaker label duplicated for speaker-focused
            feature consumers.
        speaker_turn_index: One-based count of this speaker's turn occurrence.
        previous_sentence: Previous sentence text, if available.
        next_sentence: Next sentence text, if available.
    """

    sentence_id: int
    turn_id: int
    speaker: str
    timestamp: int | None
    original_sentence: str
    word_count: int
    character_count: int
    average_word_length: float
    sentence_position: int
    is_question: bool
    is_exclamation: bool
    contains_number: bool
    contains_currency: bool
    contains_percentage: bool
    contains_date: bool
    contains_time: bool
    contains_action_keyword: bool
    contains_decision_keyword: bool
    contains_discussion_keyword: bool
    contains_deadline_keyword: bool
    speaker_name: str
    speaker_turn_index: int
    previous_sentence: str | None
    next_sentence: str | None


@dataclass(slots=True)
class SentenceRecord:
    """Internal flattened sentence record used during feature extraction.

    Attributes:
        sentence_id: Sequential one-based sentence identifier.
        turn: Source transcript turn.
        sentence: Sentence text from the preprocessed turn.
        sentence_position: Sequential one-based position in the transcript.
        speaker_turn_index: One-based count of this speaker's turn occurrence.
    """

    sentence_id: int
    turn: TranscriptTurn
    sentence: str
    sentence_position: int
    speaker_turn_index: int


def extract_sentence_features(turns: list[TranscriptTurn]) -> list[SentenceFeature]:
    """Extract rule-based features for every sentence in preprocessed turns.

    Args:
        turns: Preprocessed transcript turns.

    Returns:
        Sentence-level feature records.
    """

    sentence_records = flatten_turns_to_sentence_records(turns)
    features: list[SentenceFeature] = []

    for index, record in enumerate(sentence_records):
        previous_sentence, next_sentence = extract_context_features(sentence_records, index)
        basic_features = extract_basic_features(record.sentence, record.sentence_position)
        structural_features = extract_structural_features(record.sentence)
        meeting_features = extract_meeting_features(record.sentence)
        speaker_features = extract_speaker_features(record)

        # Features stay as explicit fields instead of opaque vectors so future
        # ANN behavior can be audited against human-readable evidence.
        features.append(
            SentenceFeature(
                sentence_id=record.sentence_id,
                turn_id=record.turn.turn_id,
                speaker=record.turn.speaker_normalized,
                timestamp=record.turn.timestamp_seconds,
                original_sentence=record.sentence,
                previous_sentence=previous_sentence,
                next_sentence=next_sentence,
                **basic_features,
                **structural_features,
                **meeting_features,
                **speaker_features,
            )
        )

    return features


def flatten_turns_to_sentence_records(turns: list[TranscriptTurn]) -> list[SentenceRecord]:
    """Flatten transcript turns into sentence records.

    Args:
        turns: Preprocessed transcript turns.

    Returns:
        Sentence records with global sentence order and per-speaker turn index.
    """

    records: list[SentenceRecord] = []
    speaker_turn_counts: dict[str, int] = {}

    for turn in turns:
        speaker_turn_counts[turn.speaker_normalized] = (
            speaker_turn_counts.get(turn.speaker_normalized, 0) + 1
        )
        sentences = turn.sentence_list or ([turn.text] if turn.text else [])

        for sentence in sentences:
            cleaned_sentence = sentence.strip()
            if not cleaned_sentence:
                continue

            # Flattening preserves both sentence-level order and source turn
            # metadata, which later models need for context-aware decisions.
            sentence_id = len(records) + 1
            records.append(
                SentenceRecord(
                    sentence_id=sentence_id,
                    turn=turn,
                    sentence=cleaned_sentence,
                    sentence_position=sentence_id,
                    speaker_turn_index=speaker_turn_counts[turn.speaker_normalized],
                )
            )

    return records


def extract_basic_features(sentence: str, sentence_position: int) -> dict[str, int | float]:
    """Extract lexical length features from a sentence.

    Args:
        sentence: Preprocessed sentence text.
        sentence_position: One-based sentence position in the transcript.

    Returns:
        Mapping containing word count, character count, average word length, and
        sentence position.
    """

    words = tokenize_words(sentence)
    word_count = len(words)
    total_word_characters = sum(len(word) for word in words)

    # Length features help later models separate short acknowledgements from
    # content-rich statements that may carry decisions or action items.
    average_word_length = (
        round(total_word_characters / word_count, 3) if word_count else 0.0
    )

    return {
        "word_count": word_count,
        "character_count": len(sentence),
        "average_word_length": average_word_length,
        "sentence_position": sentence_position,
    }


def extract_structural_features(sentence: str) -> dict[str, bool]:
    """Extract structural signals from a sentence.

    Args:
        sentence: Preprocessed sentence text.

    Returns:
        Mapping of boolean structural features.
    """

    # Structural flags capture high-signal surface patterns without making any
    # classification decision in this module.
    return {
        "is_question": sentence.rstrip().endswith("?"),
        "is_exclamation": "!" in sentence,
        "contains_number": bool(_NUMBER_RE.search(sentence)),
        "contains_currency": bool(_CURRENCY_RE.search(sentence)),
        "contains_percentage": bool(_PERCENTAGE_RE.search(sentence)),
        "contains_date": bool(_DATE_RE.search(sentence)),
        "contains_time": bool(_TIME_RE.search(sentence)),
    }


def extract_meeting_features(sentence: str) -> dict[str, bool]:
    """Extract meeting-specific keyword features from a sentence.

    Args:
        sentence: Preprocessed sentence text.

    Returns:
        Mapping of boolean meeting keyword features.
    """

    # These are intentionally rule-based constants so the next ML phase can use
    # transparent, expandable signals before any trained classifier is added.
    return {
        "contains_action_keyword": contains_any_keyword(sentence, ACTION_KEYWORDS),
        "contains_decision_keyword": contains_any_keyword(sentence, DECISION_KEYWORDS),
        "contains_discussion_keyword": contains_any_keyword(
            sentence,
            DISCUSSION_KEYWORDS,
        ),
        "contains_deadline_keyword": contains_any_keyword(sentence, DEADLINE_KEYWORDS),
    }


def extract_speaker_features(record: SentenceRecord) -> dict[str, str | int]:
    """Extract speaker-related features from a sentence record.

    Args:
        record: Flattened sentence record with source turn metadata.

    Returns:
        Mapping containing speaker name and speaker turn index.
    """

    # Speaker features preserve accountability context for later action-item and
    # decision models without changing speaker labels.
    return {
        "speaker_name": record.turn.speaker_normalized,
        "speaker_turn_index": record.speaker_turn_index,
    }


def extract_context_features(
    records: list[SentenceRecord],
    index: int,
) -> tuple[str | None, str | None]:
    """Extract neighboring sentence context.

    Args:
        records: Flattened sentence records.
        index: Current sentence index in ``records``.

    Returns:
        Tuple of previous sentence and next sentence text.
    """

    # Neighboring sentences let future models interpret short sentences like
    # "Approved" or "I will do it" with surrounding context.
    previous_sentence = records[index - 1].sentence if index > 0 else None
    next_sentence = records[index + 1].sentence if index + 1 < len(records) else None
    return previous_sentence, next_sentence


def tokenize_words(sentence: str) -> list[str]:
    """Tokenize a sentence into simple word-like tokens.

    Args:
        sentence: Sentence text to tokenize.

    Returns:
        List of word-like tokens.
    """

    return _WORD_RE.findall(sentence)


def contains_any_keyword(sentence: str, keywords: frozenset[str]) -> bool:
    """Check whether a sentence contains any keyword as a whole token.

    Args:
        sentence: Sentence text to inspect.
        keywords: Keyword set to search for.

    Returns:
        ``True`` when any keyword appears as an independent token.
    """

    tokens = {token.lower() for token in tokenize_words(sentence)}
    return bool(tokens.intersection(keywords))


def extract_features_placeholder(sentence: str) -> dict[str, int | float | bool]:
    """Backward-compatible helper for the original skeleton entry point.

    Args:
        sentence: Preprocessed sentence text.

    Returns:
        Combined basic, structural, and meeting features for one sentence.
    """

    features: dict[str, int | float | bool] = {}
    features.update(extract_basic_features(sentence, sentence_position=1))
    features.update(extract_structural_features(sentence))
    features.update(extract_meeting_features(sentence))

    # TODO:
    # Add vector export helpers when the ANN training format is finalized.
    return features


def feature_extraction_demo() -> None:
    """Run unit-test style examples for feature extraction.

    Returns:
        None. Prints feature dictionaries for manual verification.
    """

    turns = [
        TranscriptTurn(
            turn_id=1,
            speaker_raw="Rahul",
            speaker_normalized="Rahul",
            timestamp_raw="[00:10]",
            timestamp_seconds=10,
            text="We approved the budget of ₹1,50,000 today. I will prepare the report.",
            sentence_list=[
                "We approved the budget of ₹1,50,000 today.",
                "I will prepare the report.",
            ],
        ),
        TranscriptTurn(
            turn_id=2,
            speaker_raw="Priya",
            speaker_normalized="Priya",
            timestamp_raw="[00:25]",
            timestamp_seconds=25,
            text="Can we review the proposal before Friday? The completion is 75%.",
            sentence_list=[
                "Can we review the proposal before Friday?",
                "The completion is 75%.",
            ],
        ),
        TranscriptTurn(
            turn_id=3,
            speaker_raw="Speaker A",
            speaker_normalized="Speaker A",
            timestamp_raw=None,
            timestamp_seconds=None,
            text="Let's brainstorm options at 3:30 pm on March 5.",
            sentence_list=["Let's brainstorm options at 3:30 pm on March 5."],
        ),
    ]

    print("\nFeature extraction demo")
    for feature in extract_sentence_features(turns):
        print(feature)


if __name__ == "__main__":
    # Windows consoles may use a legacy code page; UTF-8 keeps currency symbols
    # in the demo printable without changing extraction behavior.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    feature_extraction_demo()

    # TODO:
    # Move demo cases into formal unit tests when the ML pipeline test suite is
    # introduced.
