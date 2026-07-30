"""Utility functions for the experimental deterministic MoM formatter.

Purpose:
    Provide lightweight NLP helpers for text cleanup, de-duplication, sentence
    ranking, simple owner/deadline extraction, and MiniLM similarity grouping.

Responsibilities:
    - Remove conversational filler without altering production preprocessing.
    - Normalize sentence wording for professional deterministic output.
    - Merge highly similar discussion sentences with cosine similarity when the
      existing MiniLM embedding service is available.
    - Fall back to lexical similarity when embeddings are unavailable.

Inputs:
    ANN prediction records from the existing prediction pipeline.

Outputs:
    Cleaned and ranked text fragments used by ``mom_formatter.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable

try:
    from ml_mom.embeddings import EmbeddingService
    from ml_mom.feature_extraction import SentenceFeature
    from ml_mom.experimental import template_library as templates
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution.
    from embeddings import EmbeddingService
    from feature_extraction import SentenceFeature
    import template_library as templates


FILLER_PATTERNS = (
    r"\bwe have\b",
    r"\bthere is\b",
    r"\bthere are\b",
    r"\blet'?s\b",
    r"\bum\b",
    r"\buh\b",
    r"\berm\b",
    r"\bhmm\b",
    r"\byou know\b",
    r"\bi think\b",
    r"\byeah\b",
    r"\bokay\b",
    r"\bok\b",
    r"\bas discussed\b",
    r"\bactually\b",
    r"\bbasically\b",
    r"\bprobably\b",
    r"\baround\b",
    r"\bkind of\b",
    r"\bsort of\b",
)
WEAK_TOPIC_PATTERNS = (
    r"^\s*(we have|there is|there are|let'?s|okay|yes|no|around|actually|basically|probably|i think)\b",
    r"^\s*(we|i|you|they|it|this|that)\s+(have|had|are|is|was|were)\b",
)
WEAK_TOPIC_WORDS = {
    "we",
    "have",
    "there",
    "is",
    "are",
    "let",
    "lets",
    "okay",
    "yes",
    "no",
    "around",
    "actually",
    "basically",
    "probably",
    "think",
}
AGREEMENT_PATTERNS = (
    r"^\s*(yes|okay|ok|sure|done|exactly|absolutely|correct|fine|perfect|great|nice)\s*[.!]?$",
    r"^\s*(i agree|agreed|sounds good|that works|looks good)\s*[.!]?$",
)
LOW_VALUE_PATTERNS = (
    r"^\s*(hello|hi|good morning|good afternoon|welcome everyone|thanks|thank you)\b",
    r"^\s*(meeting closed|let'?s close|that'?s all)\s*[.!]?$",
    r"^\s*let'?s\s+summari[sz]e\b",
    r"^\s*(yeah|hmm|umm|um|uh|actually|basically)\s*[.!]?$",
    r"^\s*(that'?s exactly what|that is exactly what)\b",
    r"^\s*(i'?ll\s+i'?ll|we'?ll\s+we'?ll)\b",
    *AGREEMENT_PATTERNS,
)
DEADLINE_PATTERN = re.compile(
    r"(?i)\b(today|tomorrow|monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday|next week|this week|eod|end of day|"
    r"end of month|next release|before deployment|sprint\s+\d+|"
    r"by\s+[A-Za-z]+(?:\s+[A-Za-z]+)?|before\s+[A-Za-z]+(?:\s+[A-Za-z]+)?|"
    r"\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\b"
)
ACTION_VERB_PATTERN = re.compile(
    r"(?i)\b(will|shall|need to|needs to|prepare|submit|complete|finish|deliver|"
    r"review|send|share|update|create|finalize|implement|publish|confirm|assign|"
    r"generate|deploy|verify|test|fix)\b"
)
OWNER_ACTION_PATTERN = re.compile(
    r"^\s*(?P<owner>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+"
    r"(?P<verb>(?i:will|shall|needs to|need to|should|must|has to|complete|update|"
    r"publish|confirm|prepare|assign|generate|review|finalize|implement|deploy|"
    r"verify|deliver|submit|finish|test|fix))\s+"
    r"(?P<task>.+)$"
)
FIRST_PERSON_ACTION_PATTERN = re.compile(
    r"(?i)^\s*(i'll|i will|we will|we need to|we should|i)\s+(?P<task>.+)$"
)


@dataclass(slots=True)
class ActionParts:
    """Normalized action item components."""

    owner: str
    task: str
    deadline: str


def deterministic_index(text: str, size: int) -> int:
    """Return a stable template index without random state.

    Args:
        text: Text used as the deterministic seed.
        size: Number of available templates.

    Returns:
        Stable index within the template collection.
    """

    if size <= 0:
        return 0
    return sum(ord(char) for char in text) % size


def choose_template(templates_: tuple[str, ...], seed_text: str) -> str:
    """Choose a template deterministically to avoid repetitive wording."""

    return templates_[deterministic_index(seed_text, len(templates_))]


def prediction_confidence(record: Any, default: float = 1.0) -> float:
    """Return ANN confidence from a prediction-like record.

    Args:
        record: Existing prediction object.
        default: Confidence used for legacy objects without the field.

    Returns:
        Confidence score clamped into the ``0.0`` to ``1.0`` range.
    """

    try:
        value = float(getattr(record, "confidence_score", default))
    except (TypeError, ValueError):
        value = default
    return max(0.0, min(1.0, value))


def confidence_filtered_records(
    records: Iterable[Any],
    threshold: float,
) -> list[Any]:
    """Filter predictions using the experimental confidence threshold."""

    return [
        record
        for record in records
        if prediction_confidence(record) >= threshold
    ]


def record_sentence(record: Any) -> str:
    """Return the sentence text from a prediction-like record."""

    return normalize_whitespace(getattr(record, "sentence", ""))


def context_window_text(
    records: list[Any],
    index: int,
    window_size: int = 2,
) -> str:
    """Build a previous/current/next context window for one record.

    Context helps the experimental formatter connect related utterances such as
    a problem statement, a proposed fix, and a decision that appear in adjacent
    speaker turns.
    """

    start = max(0, index - window_size)
    end = min(len(records), index + window_size + 1)
    return normalize_whitespace(
        " ".join(
            record_sentence(record)
            for record in records[start:end]
            if record_sentence(record)
        )
    )


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and trim a text fragment."""

    return re.sub(r"\s+", " ", text or "").strip()


def remove_fillers(text: str) -> str:
    """Remove lightweight conversational filler from a sentence."""

    cleaned = text or ""
    for pattern in FILLER_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+([,.!?])", r"\1", cleaned)
    cleaned = re.sub(r"^[,.\s]+", "", cleaned)
    return normalize_whitespace(cleaned)


def remove_speech_repetitions(text: str) -> str:
    """Remove adjacent repeated words caused by speech disfluency."""

    words = normalize_whitespace(text).split()
    cleaned_words: list[str] = []
    for word in words:
        comparable = re.sub(r"[^a-z0-9']+", "", word.casefold())
        previous = (
            re.sub(r"[^a-z0-9']+", "", cleaned_words[-1].casefold())
            if cleaned_words
            else ""
        )
        if comparable and comparable == previous and not cleaned_words[-1].endswith(":"):
            continue
        cleaned_words.append(word)
    return normalize_whitespace(" ".join(cleaned_words))


def normalize_sentence(text: str) -> str:
    """Create a readable sentence without changing its factual content."""

    cleaned = remove_speech_repetitions(remove_fillers(text))
    cleaned = cleaned.strip(" -")
    if not cleaned:
        return ""
    cleaned = cleaned[0].upper() + cleaned[1:]
    if cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def is_agreement_sentence(text: str) -> bool:
    """Return whether text is only agreement or acknowledgement.

    Agreement-only replies often receive Decision or Information labels from a
    classifier, but they do not contain decision substance. Filtering them here
    prevents corporate MoM sections from being populated with "Yes" or "Okay".
    """

    normalized = normalize_whitespace(text).casefold()
    return any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in AGREEMENT_PATTERNS)


def is_low_value_sentence(text: str) -> bool:
    """Return whether a sentence is too conversational to keep."""

    normalized = normalize_whitespace(text).casefold()
    if is_agreement_sentence(normalized):
        return True
    has_work_signal = ACTION_VERB_PATTERN.search(normalized) or any(
        re.search(rf"\b{re.escape(keyword)}\b", normalized)
        for keyword in templates.RANKING_KEYWORDS
    )
    has_numeric_business_fact = bool(re.search(r"\d", normalized)) and any(
        re.search(rf"\b{re.escape(keyword)}\b", normalized)
        for keyword in templates.RANKING_KEYWORDS
    )
    if len(normalized.split()) < 4 and not has_work_signal:
        return True
    if has_numeric_business_fact:
        return False
    return any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in LOW_VALUE_PATTERNS)


def is_reportable_sentence(text: str) -> bool:
    """Return whether text is useful enough for the experimental MoM."""

    return bool(normalize_whitespace(text)) and not is_low_value_sentence(text) and not is_agreement_sentence(text)


def quality_score(text: str) -> float:
    """Score report text for completeness, substance, and professionalism.

    The score is deterministic and uses only text quality signals. It does not
    call an ML model or external service.
    """

    normalized = normalize_whitespace(text)
    if not normalized:
        return 0.0
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'%/-]*", normalized)
    meaningful_words = [
        word
        for word in words
        if word.casefold() not in WEAK_TOPIC_WORDS
    ]
    score = min(len(meaningful_words), 18) / 18
    if re.search(r"\d", normalized):
        score += 0.12
    if ACTION_VERB_PATTERN.search(normalized):
        score += 0.12
    if any(re.search(rf"(?i)\b{re.escape(keyword)}\b", normalized) for keyword in templates.RANKING_KEYWORDS):
        score += 0.15
    if re.search(r"(?i)\b(approved|confirmed|agreed|resolved|accepted|finalized|deferred)\b", normalized):
        score += 0.15
    if is_low_value_sentence(normalized) or is_weak_topic(normalized):
        score -= 0.55
    if re.search(r"(?i)\b(i'?ll\s+i'?ll|we'?ll\s+we'?ll|we have around|kind of|sort of)\b", normalized):
        score -= 0.45
    if normalized.rstrip().endswith("?"):
        score -= 0.15
    return max(0.0, min(1.0, score))


def clean_report_sentence(text: str) -> str:
    """Clean a generated candidate before it enters the final report."""

    cleaned = normalize_sentence(text)
    cleaned = re.sub(r"(?i)\bi'?ll\s+i'?ll\b", "I'll", cleaned)
    cleaned = re.sub(r"(?i)\bwe'?ll\s+we'?ll\b", "We'll", cleaned)
    cleaned = re.sub(r"(?i)\bkind of\b|\bsort of\b", "", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return normalize_sentence(cleaned)


def is_quality_report_sentence(text: str, *, min_score: float = 0.34) -> bool:
    """Return whether a candidate sentence is suitable for final MoM content."""

    cleaned = clean_report_sentence(text)
    if not cleaned or is_low_value_sentence(cleaned) or is_agreement_sentence(cleaned):
        return False
    if quality_score(cleaned) < min_score:
        return False
    if is_weak_topic(cleaned):
        return False
    return True


def unique_normalized_sentences(sentences: Iterable[str]) -> list[str]:
    """Remove duplicate sentences while preserving order."""

    unique: list[str] = []
    seen: set[str] = set()
    for sentence in sentences:
        normalized = normalize_sentence(sentence)
        key = re.sub(r"[^a-z0-9]+", " ", normalized.casefold()).strip()
        if not normalized or key in seen:
            continue
        unique.append(normalized)
        seen.add(key)
    return unique


def sentence_score(sentence: str) -> float:
    """Rank sentences by deterministic usefulness signals."""

    words = re.findall(r"[A-Za-z][A-Za-z_-]+", sentence)
    score = min(len(words), 28) / 28
    if ACTION_VERB_PATTERN.search(sentence):
        score += 0.15
    for keyword in templates.RANKING_KEYWORDS:
        if re.search(rf"(?i)\b{re.escape(keyword)}\b", sentence):
            score += 0.08
    if sentence.rstrip().endswith("?"):
        score -= 0.1
    if is_low_value_sentence(sentence):
        score -= 0.5
    return score


def professional_topic(sentence: str) -> str:
    """Convert a raw sentence or question into a corporate topic phrase."""

    cleaned = normalize_sentence(sentence).rstrip(".")
    for keyword, topic in templates.PROFESSIONAL_TOPIC_KEYWORDS.items():
        if re.search(rf"(?i)\b{re.escape(keyword)}\b", cleaned):
            return topic

    # Questions are often useful discussion cues, but raw questions read like a
    # transcript. Removing the interrogative framing turns them into MoM topics.
    cleaned = re.sub(
        r"(?i)^(are there any|is there any|do we have|can we|should we|what are|what is|how do we)\s+",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"(?i)\b(we|the team|team|they|i)\b\s+"
        r"(discussed|reviewed|considered|proposed|analyzed|covered|checked)\s+",
        "",
        cleaned,
    )
    cleaned = re.sub(r"(?i)\b(before|for)\s+release\??$", "release readiness", cleaned)
    cleaned = re.sub(
        r"(?i)^(we have|there is|there are|let'?s|okay|yes|no|around|actually|basically|probably|i think)\s+",
        "",
        cleaned,
    )
    cleaned = cleaned.strip(" ?.")
    words = [
        word
        for word in cleaned.split()
        if re.sub(r"[^a-z0-9]+", "", word.casefold()) not in WEAK_TOPIC_WORDS
    ]
    if len(words) > 9:
        cleaned = " ".join(words[:9])
    else:
        cleaned = " ".join(words)
    return cleaned[0].upper() + cleaned[1:] if cleaned else ""


def is_weak_topic(topic: str) -> bool:
    """Return whether a candidate topic is filler rather than meeting content."""

    normalized = normalize_whitespace(topic).casefold().strip(" .")
    if not normalized:
        return True
    if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in WEAK_TOPIC_PATTERNS):
        return True
    tokens = re.findall(r"[a-z0-9]+", normalized)
    if len(tokens) <= 2 and all(token in WEAK_TOPIC_WORDS for token in tokens):
        return True
    return False


def meaningful_fact(sentence: str) -> str:
    """Return a concise transcript-supported fact for topic discussion bullets."""

    cleaned = normalize_sentence(sentence).rstrip(".")
    cleaned = re.sub(
        r"(?i)^(we have|there is|there are|let'?s|okay|yes|no|actually|basically|probably|i think)\s+",
        "",
        cleaned,
    )
    cleaned = re.sub(r"(?i)\baround\s+(\d)", r"\1", cleaned)
    cleaned = normalize_whitespace(cleaned)
    if not cleaned or is_low_value_sentence(cleaned):
        return ""
    return normalize_sentence(cleaned)


def discussion_context(topic: str) -> str:
    """Return a What/Why discussion bullet for a normalized topic."""

    if topic in templates.DISCUSSION_CONTEXT:
        return templates.DISCUSSION_CONTEXT[topic]
    template = choose_template(templates.DISCUSSION_TEMPLATES, topic)
    return template.format(topic=topic)


def embedding_vectors_for_texts(texts: list[str]) -> list[list[float]]:
    """Generate MiniLM vectors for experimental clustering text.

    The helper uses the existing embedding service but stores nothing. If the
    model is unavailable, callers can fall back to lexical similarity.
    """

    if not texts:
        return []
    try:
        service = EmbeddingService()
        features = [
            sentence_feature(index, text)
            for index, text in enumerate(texts, start=1)
        ]
        result = service.generate_embeddings(features)
        if result.error_message or len(result.embeddings) != len(texts):
            return []
        return [embedding.embedding_vector for embedding in result.embeddings]
    except Exception:
        return []


def normalize_task(task: str) -> str:
    """Convert conversational action wording into professional task wording."""

    cleaned = normalize_sentence(task).rstrip(".")
    cleaned = re.sub(r"(?i)^(?:can|could|would)\s+you\s+", "", cleaned)
    cleaned = re.sub(
        r"(?i)^(?:i|we|you|the\s+team)\s+(?:will|shall|must|should|"
        r"need(?:s)?\s+to|have\s+to)\s+",
        "",
        cleaned,
    )
    cleaned = re.sub(r"(?i)^please\s+", "", cleaned)
    replacements = (
        (r"(?i)^finish\s+lazy loading$", "Implement lazy loading"),
        (r"(?i)^complete\s+lazy loading$", "Implement lazy loading"),
        (r"(?i)^verify everything once the fixes are merged$", "Verify all implemented fixes after code merge"),
        (r"(?i)^complete\s+pdf improvements$", "Complete PDF export improvements"),
        (r"(?i)^finish\s+it$", ""),
        (r"(?i)^finish\s+it\s+evening$", ""),
    )
    for pattern, replacement in replacements:
        if re.search(pattern, cleaned):
            cleaned = replacement
            break
    cleaned = re.sub(r"(?i)\bonce the fixes are merged\b", "after code merge", cleaned)
    cleaned = re.sub(r"(?i)\bpdf\b", "PDF", cleaned)
    return normalize_sentence(cleaned) if cleaned else ""


def factual_information(sentence: str) -> str:
    """Preserve useful factual statements with light professional cleanup."""

    cleaned = normalize_sentence(sentence).rstrip(".")
    cleaned = re.sub(r"(?i)\bapi\b", "API", cleaned)
    cleaned = re.sub(r"(?i)\bapis\b", "APIs", cleaned)
    cleaned = re.sub(r"(?i)\bpdf\b", "PDF", cleaned)
    cleaned = re.sub(r"(?i)\bdocx\b", "DOCX", cleaned)
    cleaned = re.sub(r"(?i)\bui\b", "UI", cleaned)
    return normalize_sentence(cleaned)


def is_pending_sentence(sentence: str) -> bool:
    """Return whether a sentence indicates unresolved or deferred work."""

    normalized = normalize_whitespace(sentence).casefold()
    return any(keyword in normalized for keyword in templates.PENDING_KEYWORDS)


def simple_noun_phrase(sentence: str) -> str:
    """Extract a short deterministic topic phrase from a sentence."""

    return professional_topic(sentence)


def lexical_similarity(left: str, right: str) -> float:
    """Compute simple token-overlap similarity as a no-model fallback."""

    left_tokens = set(re.findall(r"[a-z0-9]+", left.casefold()))
    right_tokens = set(re.findall(r"[a-z0-9]+", right.casefold()))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def sentence_feature(index: int, sentence: str) -> SentenceFeature:
    """Build a minimal feature object for the existing embedding service."""

    words = sentence.split()
    average_word_length = sum(len(word) for word in words) / len(words) if words else 0.0
    return SentenceFeature(
        sentence_id=index,
        turn_id=0,
        speaker="",
        timestamp=None,
        original_sentence=sentence,
        word_count=len(words),
        character_count=len(sentence),
        average_word_length=round(average_word_length, 3),
        sentence_position=index,
        is_question=sentence.rstrip().endswith("?"),
        is_exclamation="!" in sentence,
        contains_number=any(char.isdigit() for char in sentence),
        contains_currency=False,
        contains_percentage="%" in sentence,
        contains_date=False,
        contains_time=False,
        contains_action_keyword=False,
        contains_decision_keyword=False,
        contains_discussion_keyword=False,
        contains_deadline_keyword=False,
        speaker_name="",
        speaker_turn_index=0,
        previous_sentence=None,
        next_sentence=None,
    )


def cosine(left: list[float], right: list[float]) -> float:
    """Compute cosine similarity for two vectors."""

    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def merge_similar_sentences(
    sentences: list[str],
    similarity_threshold: float = 0.72,
) -> list[str]:
    """Merge highly similar sentences using MiniLM cosine similarity.

    The function uses the existing ``EmbeddingService`` when available. If the
    local MiniLM model cannot load, it falls back to lexical overlap so the
    experimental formatter remains non-crashing and deterministic.
    """

    cleaned = unique_normalized_sentences(
        sentence for sentence in sentences if is_reportable_sentence(sentence)
    )
    if not cleaned:
        return []

    vectors: list[list[float]] = []
    try:
        service = EmbeddingService()
        result = service.generate_embeddings(
            [sentence_feature(index, sentence) for index, sentence in enumerate(cleaned, start=1)]
        )
        if not result.error_message and len(result.embeddings) == len(cleaned):
            vectors = [embedding.embedding_vector for embedding in result.embeddings]
    except Exception:
        vectors = []

    selected: list[str] = []
    selected_vectors: list[list[float]] = []
    for index, sentence in sorted(
        enumerate(cleaned),
        key=lambda item: sentence_score(item[1]),
        reverse=True,
    ):
        if vectors:
            is_duplicate = any(
                cosine(vectors[index], selected_vector) >= similarity_threshold
                for selected_vector in selected_vectors
            )
        else:
            is_duplicate = any(
                lexical_similarity(sentence, existing) >= 0.62
                for existing in selected
            )
        if is_duplicate:
            continue
        selected.append(sentence)
        if vectors:
            selected_vectors.append(vectors[index])
    return selected


def merge_text_fragments(
    fragments: list[str],
    similarity_threshold: float = 0.78,
    limit: int | None = None,
) -> list[str]:
    """Merge already-normalized text fragments without rewording them."""

    merged = merge_similar_sentences(fragments, similarity_threshold=similarity_threshold)
    if limit is not None:
        return merged[:limit]
    return merged


def extract_deadline(sentence: str) -> str:
    """Extract a simple deadline phrase from an action sentence."""

    match = DEADLINE_PATTERN.search(sentence)
    return normalize_whitespace(match.group(0)) if match else ""


def extract_action_parts(sentence: str, speaker: str) -> ActionParts:
    """Extract owner, task, and deadline from an action prediction."""

    normalized = normalize_sentence(sentence)
    normalized = re.sub(
        r"(?i)^(yes|okay|ok|sure|done|fine|great|perfect|nice),?\s+",
        "",
        normalized,
    )
    normalized = re.sub(r"(?i)^please\s+", "", normalized)
    owner = "Unassigned"
    deadline = extract_deadline(normalized)

    # A named person at the start of an action sentence should override the
    # current speaker because "Aarav: Rahul will complete..." assigns Rahul.
    owner_match = OWNER_ACTION_PATTERN.match(normalized.rstrip("."))
    if owner_match:
        owner = normalize_whitespace(owner_match.group("owner"))
        verb = owner_match.group("verb")
        task = owner_match.group("task")
        if not re.search(r"(?i)^(will|shall|needs to|need to|should|must|has to)$", verb):
            task = f"{verb} {task}"
    else:
        speaker_name = normalize_whitespace(speaker)
        first_person = FIRST_PERSON_ACTION_PATTERN.match(normalized.rstrip("."))
        if first_person and speaker_name and speaker_name != "-":
            owner = speaker_name
            task = first_person.group("task")
        else:
            task = normalized

    task = DEADLINE_PATTERN.sub("", task).strip(" .?")
    task = re.sub(
        r"(?i)^(will|shall|needs to|need to|should|must|has to|please)\s+",
        "",
        task,
    ).strip()
    task = normalize_task(task)
    return ActionParts(owner=owner, task=task, deadline=deadline)


def extract_next_meeting_date(sentences: Iterable[str]) -> str:
    """Return a possible next-meeting date phrase from information sentences."""

    for sentence in sentences:
        if re.search(r"(?i)\b(next meeting|next review|follow-up|review meeting)\b", sentence):
            deadline = extract_deadline(sentence)
            if deadline:
                return deadline
    return ""


def records_by_label(records: Iterable[Any]) -> dict[str, list[Any]]:
    """Group prediction-like records by ``predicted_label``."""

    grouped: dict[str, list[Any]] = {
        "Discussion": [],
        "Decision": [],
        "Action_Item": [],
        "Summary": [],
        "Information": [],
    }
    for record in records:
        label = getattr(record, "predicted_label", "")
        grouped.setdefault(label, []).append(record)
    return grouped
