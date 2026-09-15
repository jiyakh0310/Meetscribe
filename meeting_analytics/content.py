"""SELaD Phase 2 -- topic grouping, keyword extraction, and positive-language
proportion.

Consumes only the Phase 0 ``timeline.Timeline`` (the same object Phase 1's
``meeting_analytics.model`` consumes). This module never touches Sarvam,
the ANN, the deterministic MoM formatter, Gemma, or Voxels, and never
imports from ``ml_mom.experimental`` (the formatter package). The
dependency direction stays one-way:

    Timeline -> meeting_analytics.content -> UI

Two design decisions worth calling out explicitly (see the audit that
preceded this phase):

1. Topic grouping does NOT reuse ``ml_mom.experimental.mom_formatter``'s
   live clustering (``build_topic_clusters``). That function is private
   formatter machinery tuned for MoM construction (context-window text
   units, a 0.74 similarity threshold chosen for that use, deep coupling to
   ``PredictionResult``/ANN labels). Reusing it here would either require
   invasive coupling to formatter internals or silently diverge from what
   it actually does. Instead this module implements a small,
   analytics-specific single-pass grouping layer (see ``_cluster_events``)
   that is deterministic, documented, and answers only to analytics needs.

2. It DOES reuse the existing MiniLM model-loading/caching infrastructure
   (``ml_mom.embeddings.EmbeddingService``) rather than loading a second
   copy of the model. ``EmbeddingService``'s public embedding methods
   require ``ml_mom.feature_extraction.SentenceFeature`` objects (a
   20-field formatter-shaped dataclass) -- constructing throwaway instances
   of that just to get a vector back would itself be a form of invasive
   coupling to formatter-internal data shapes for no benefit, since none of
   those fields affect the embedding. Instead this module calls
   ``EmbeddingService.load_model()`` (the service's public, documented
   entry point) and then encodes raw text directly through the loaded
   model, which is fetched from the SAME module-level ``_MODEL_CACHE`` the
   factual pipeline already populates -- so no duplicate model instance is
   ever loaded. See ``_embed_texts``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence

from timeline import Timeline, TimelineEvent, TimingStatus

# --------------------------------------------------------------------------
# Result structures
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TopicAnalytics:
    """One analytics-derived conversation topic, traceable to its source
    timeline events."""

    topic_id: int
    title: str
    """Deterministic, keyword-derived title (see ``_topic_title``). Never
    LLM-generated. Falls back to "Discussion Topic N" when there is not
    enough lexical evidence to name the topic."""

    keywords: tuple[str, ...]
    """Up to a handful of representative terms for this topic specifically
    (topic-local ranking, not the meeting-wide keyword list)."""

    participant_speakers: tuple[str, ...]
    """Reviewed display names of speakers who contributed to this topic, in
    first-appearance order."""

    event_count: int
    start_time_seconds: float | None
    """Earliest ``start_time_seconds`` among this topic's TIMED source
    events. None when none of the topic's events have reliable timing --
    never fabricated."""

    end_time_seconds: float | None
    """Latest ``end_time_seconds`` among this topic's TIMED source events.
    None under the same condition as ``start_time_seconds``."""

    source_event_ids: tuple[int, ...]
    """``TimelineEvent.event_id`` values this topic was built from -- the
    traceability link back to the Phase 0 timeline (and, transitively, to
    Voxels/video alignment in later phases)."""

    representative_text: str
    """A short evidence preview: the first non-blank source event's text,
    truncated. Not a summary -- verbatim transcript text."""


@dataclass(frozen=True, slots=True)
class SpeakerKeywords:
    speaker_id: str
    speaker_name: str
    keywords: tuple[str, ...]
    """Empty when this speaker did not contribute enough distinct lexical
    content to rank meaningfully -- never padded with weak/forced terms."""


@dataclass(frozen=True, slots=True)
class KeywordAnalytics:
    meeting_keywords: tuple[str, ...]
    """5-10 meeting-wide keywords (see ``_rank_keywords``)."""

    per_speaker: tuple[SpeakerKeywords, ...]
    """One entry per speaker, in the same first-appearance order used
    elsewhere in ``meeting_analytics``."""


@dataclass(frozen=True, slots=True)
class PositiveLanguageSpeaker:
    speaker_id: str
    speaker_name: str
    eligible_token_count: int
    positive_token_count: int
    proportion: float | None
    """``positive_token_count / eligible_token_count``. None when
    ``eligible_token_count`` is 0 (no eligible text -- "Not available"),
    NOT fabricated as 0.0. A genuine zero (eligible text exists, zero
    matches) is a real, valid 0.0."""


@dataclass(frozen=True, slots=True)
class PositiveLanguageAnalytics:
    per_speaker: tuple[PositiveLanguageSpeaker, ...]
    meeting_eligible_token_count: int
    meeting_positive_token_count: int
    meeting_proportion: float | None


@dataclass(frozen=True, slots=True)
class ContentAnalytics:
    """Phase 2 result: topics, keywords, and positive-language, all derived
    from the same Timeline Phase 1 already consumes."""

    topics: tuple[TopicAnalytics, ...]
    keywords: KeywordAnalytics
    positive_language: PositiveLanguageAnalytics


# --------------------------------------------------------------------------
# Shared text utilities
# --------------------------------------------------------------------------

_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z'-]*")


def _tokenize(text: str) -> list[str]:
    """Lowercase alphabetic word tokens (letters plus internal ``'``/``-``),
    length > 1. Pure regex tokenization -- no language-specific NLP, so it
    does not break on Hinglish text (Latin-script tokens tokenize fine; see
    the module-level limitation note on ``POSITIVE_LEXICON`` for why
    lexicon *matching* is still English-only)."""

    return [token.lower() for token in _TOKEN_PATTERN.findall(text) if len(token) > 1]


# A small, generic English stopword list for keyword ranking. Deliberately
# separate from ml_mom.experimental.formatter_utils's DECISION/ACTION/
# DEADLINE keyword sets -- those exist to detect factual MoM signal words
# and are not a stopword list; reusing them here would be a category error,
# not code reuse.
STOPWORDS = frozenset(
    """
    a an the is are was were be been being to of in on at for with and or
    but if so that this these those it its i we you they he she his her
    their our your my me us them him as by from not no do does did has have had will
    would can could should shall about into than then there here what
    which who whom when where why how all any some just also very more
    most other such only own same too up down out off over under again
    further once s t don now let lets okay ok yeah yes um uh uhh hmm
    actually basically really quite well go going gone get gets getting
    got one two three hi hello hey bye goodbye morning afternoon evening
    everyone everybody
    """.split()
)


def _rank_keywords(documents: Sequence[str], top_n: int) -> list[str]:
    """Deterministic TF-IDF-style keyword ranking across ``documents``
    (each a timeline event's text, i.e. this meeting's own segments treated
    as a small corpus).

    Separate from, and unrelated to, the TF-IDF baseline classifier in
    ``ml_mom/logistic_baseline.py`` -- that scores whole sentences for a
    5-class label; this ranks individual terms for display as keywords.

        idf(term) = ln((1 + N) / (1 + df(term))) + 1   (smoothed)
        score(term) = total_frequency(term) * idf(term)

    Ranked descending by score; ties broken by first-appearance order so
    the result is fully deterministic. Stopwords (``STOPWORDS``) are
    excluded before scoring. Single-word terms only -- multi-word phrase
    extraction was considered and skipped to keep this ranker simple and
    reliable, per the instruction's own allowance for that tradeoff.
    """

    doc_token_lists = [
        [token for token in _tokenize(document) if token not in STOPWORDS]
        for document in documents
    ]
    n_docs = len(doc_token_lists)
    if n_docs == 0:
        return []

    doc_frequency: dict[str, int] = {}
    total_frequency: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    position = 0
    for tokens in doc_token_lists:
        seen_in_doc: set[str] = set()
        for token in tokens:
            total_frequency[token] = total_frequency.get(token, 0) + 1
            if token not in first_seen:
                first_seen[token] = position
                position += 1
            seen_in_doc.add(token)
        for token in seen_in_doc:
            doc_frequency[token] = doc_frequency.get(token, 0) + 1

    if not total_frequency:
        return []

    scores = {
        term: frequency * (math.log((1 + n_docs) / (1 + doc_frequency[term])) + 1)
        for term, frequency in total_frequency.items()
    }
    ranked = sorted(scores.items(), key=lambda item: (-item[1], first_seen[item[0]]))
    return [term for term, _ in ranked[:top_n]]


def _topic_title(keywords: Sequence[str], fallback_index: int) -> str:
    """Deterministic, keyword-derived topic title. Never calls an LLM.
    Falls back to "Discussion Topic N" (N is 1-based) when there is not
    enough lexical evidence to name the topic, rather than guessing."""

    if not keywords:
        return f"Discussion Topic {fallback_index}"
    return " ".join(word.capitalize() for word in keywords[:2])


# --------------------------------------------------------------------------
# MiniLM reuse (see module docstring point 2)
# --------------------------------------------------------------------------


def _embed_texts(texts: Sequence[str]) -> list[list[float]] | None:
    """Encode ``texts`` with the project's existing MiniLM model, reusing
    its shared, module-level model cache. Returns None if the model cannot
    be loaded (e.g. no local files and network access is unavailable) --
    callers must degrade gracefully, never fabricate embeddings."""

    try:
        from ml_mom.embeddings import EmbeddingService
    except Exception:
        return None

    service = EmbeddingService()
    if service.load_model() is not None or service._model is None:
        return None

    try:
        vectors = service._model.encode(list(texts), convert_to_numpy=True)
    except Exception:
        return None

    return [[float(value) for value in vector] for vector in vectors]


# --------------------------------------------------------------------------
# Topic grouping
# --------------------------------------------------------------------------

# Chosen lower than the live formatter's 0.74 (ml_mom/experimental/
# mom_formatter.py's build_topic_clusters), which embeds multi-sentence
# CONTEXT WINDOWS -- longer, semantically richer, more stable vectors.
# Analytics topics group single raw timeline-event utterances instead
# (shorter, noisier embeddings, and deliberately NOT the formatter's own
# clusters -- see module docstring point 1), so a lower threshold is
# needed to avoid fragmenting a normal meeting into dozens of one-event
# "topics". 0.5 was chosen as a moderate middle ground for this shorter,
# noisier text unit; it is a documented heuristic, not empirically tuned
# against a labeled dataset.
TOPIC_SIMILARITY_THRESHOLD = 0.5


def _cluster_events(embeddings: Sequence[Sequence[float]] | None, count: int) -> list[list[int]]:
    """Group event indices [0, count) by embedding similarity.

    Single-pass, deterministic: each item joins the most similar existing
    cluster if that similarity is >= TOPIC_SIMILARITY_THRESHOLD (a cluster's
    centroid is the running mean of its members' unit vectors), otherwise it
    starts a new cluster. When embeddings are unavailable (``embeddings is
    None``), everything falls back to a single cluster rather than
    fabricating semantic grouping or crashing.
    """

    if embeddings is None or count == 0:
        return [list(range(count))] if count else []

    clusters: list[dict[str, object]] = []
    for index in range(count):
        vector = list(embeddings[index])
        norm = math.sqrt(sum(value * value for value in vector))
        unit_vector = [value / norm for value in vector] if norm > 0 else vector

        best_cluster_index = -1
        best_similarity = -1.0
        for cluster_index, cluster in enumerate(clusters):
            centroid = cluster["centroid"]  # type: ignore[index]
            similarity = sum(a * b for a, b in zip(unit_vector, centroid))
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster_index = cluster_index

        if best_cluster_index >= 0 and best_similarity >= TOPIC_SIMILARITY_THRESHOLD:
            cluster = clusters[best_cluster_index]
            indices: list[int] = cluster["indices"]  # type: ignore[assignment]
            indices.append(index)
            centroid = cluster["centroid"]  # type: ignore[assignment]
            member_count = len(indices)
            new_centroid = [
                (c * (member_count - 1) + v) / member_count for c, v in zip(centroid, unit_vector)
            ]
            centroid_norm = math.sqrt(sum(value * value for value in new_centroid))
            cluster["centroid"] = (
                [value / centroid_norm for value in new_centroid] if centroid_norm > 0 else new_centroid
            )
        else:
            clusters.append({"indices": [index], "centroid": unit_vector})

    return [cluster["indices"] for cluster in clusters]  # type: ignore[misc]


def _build_topics(events: Sequence[TimelineEvent]) -> tuple[TopicAnalytics, ...]:
    non_blank = [event for event in events if event.transcript.strip()]
    if not non_blank:
        return ()

    embeddings = _embed_texts([event.transcript for event in non_blank])
    clusters = _cluster_events(embeddings, len(non_blank))

    topics: list[TopicAnalytics] = []
    for topic_index, indices in enumerate(clusters, start=1):
        cluster_events = [non_blank[i] for i in indices]
        cluster_texts = [event.transcript for event in cluster_events]
        keywords = _rank_keywords(cluster_texts, top_n=5)

        timed = [event for event in cluster_events if event.timing_status is TimingStatus.TIMED]
        start = min((event.start_time_seconds for event in timed), default=None)
        end = max((event.end_time_seconds for event in timed), default=None)

        participants: list[str] = []
        for event in cluster_events:
            if event.speaker_name not in participants:
                participants.append(event.speaker_name)

        preview = cluster_events[0].transcript.strip()
        if len(preview) > 140:
            preview = preview[:137].rstrip() + "..."

        topics.append(
            TopicAnalytics(
                topic_id=topic_index,
                title=_topic_title(keywords, topic_index),
                keywords=tuple(keywords[:5]),
                participant_speakers=tuple(participants),
                event_count=len(cluster_events),
                start_time_seconds=start,
                end_time_seconds=end,
                source_event_ids=tuple(event.event_id for event in cluster_events),
                representative_text=preview,
            )
        )
    return tuple(topics)


# --------------------------------------------------------------------------
# Keywords (meeting-level + per-speaker)
# --------------------------------------------------------------------------

_MIN_SPEAKER_TOKENS_FOR_KEYWORDS = 6
"""A speaker needs at least this many non-stopword tokens across their own
events before we rank keywords for them individually -- below this, ranking
is too noisy to be meaningful and we return an empty tuple rather than
force weak keywords onto a speaker with little text."""


def _build_keywords(events: Sequence[TimelineEvent]) -> KeywordAnalytics:
    non_blank = [event for event in events if event.transcript.strip()]
    meeting_keywords = _rank_keywords([event.transcript for event in non_blank], top_n=10)

    speaker_order: list[str] = []
    speaker_name_by_id: dict[str, str] = {}
    texts_by_speaker: dict[str, list[str]] = {}
    for event in non_blank:
        if event.speaker_id not in speaker_name_by_id:
            speaker_order.append(event.speaker_id)
        speaker_name_by_id[event.speaker_id] = event.speaker_name
        texts_by_speaker.setdefault(event.speaker_id, []).append(event.transcript)

    per_speaker: list[SpeakerKeywords] = []
    for speaker_id in speaker_order:
        texts = texts_by_speaker[speaker_id]
        token_count = sum(
            1 for text in texts for token in _tokenize(text) if token not in STOPWORDS
        )
        keywords = _rank_keywords(texts, top_n=5) if token_count >= _MIN_SPEAKER_TOKENS_FOR_KEYWORDS else []
        per_speaker.append(
            SpeakerKeywords(
                speaker_id=speaker_id,
                speaker_name=speaker_name_by_id[speaker_id],
                keywords=tuple(keywords),
            )
        )

    return KeywordAnalytics(
        meeting_keywords=tuple(meeting_keywords),
        per_speaker=tuple(per_speaker),
    )


# --------------------------------------------------------------------------
# Positive-language proportion
# --------------------------------------------------------------------------

# A small, explicitly curated English lexicon of clearly positive
# meeting-communication terms. This is NOT a general sentiment model and
# makes no claim to be one -- it is a transparent, literal word-match rule.
# LIMITATION: English-centric. Hinglish/Hindi positive terms (e.g. "badhiya",
# "theek hai" as agreement) are not recognized -- text in those segments
# still tokenizes (see _tokenize) and counts toward the eligible-token
# denominator, but will rarely contribute to the positive-token numerator.
# This under-coverage is a known, documented limitation, not silently
# assumed-equivalent coverage.
POSITIVE_LEXICON = frozenset(
    """
    good great excellent agree agreed awesome happy glad perfect nice
    helpful appreciate appreciated thanks thank thankful positive
    confident excited pleased wonderful fantastic love loved best success
    successful progress improved improvement solid smooth clear easy
    resolved works working ready done complete completed approve approved
    support supportive welcome encouraging optimistic enjoy enjoyed
    delighted proud impressive impressed
    """.split()
)


def _positive_language_for_texts(texts: Sequence[str]) -> tuple[int, int]:
    """Return (eligible_token_count, positive_token_count) for the given
    texts. Eligible tokens are all tokenized words (see _tokenize) --
    stopwords ARE included in this denominator, since the metric measures
    "share of words spoken", not "share of content words"."""

    eligible = 0
    positive = 0
    for text in texts:
        for token in _tokenize(text):
            eligible += 1
            if token in POSITIVE_LEXICON:
                positive += 1
    return eligible, positive


def _build_positive_language(events: Sequence[TimelineEvent]) -> PositiveLanguageAnalytics:
    non_blank = [event for event in events if event.transcript.strip()]

    speaker_order: list[str] = []
    speaker_name_by_id: dict[str, str] = {}
    texts_by_speaker: dict[str, list[str]] = {}
    for event in non_blank:
        if event.speaker_id not in speaker_name_by_id:
            speaker_order.append(event.speaker_id)
        speaker_name_by_id[event.speaker_id] = event.speaker_name
        texts_by_speaker.setdefault(event.speaker_id, []).append(event.transcript)

    per_speaker: list[PositiveLanguageSpeaker] = []
    for speaker_id in speaker_order:
        eligible, positive = _positive_language_for_texts(texts_by_speaker[speaker_id])
        proportion = (positive / eligible) if eligible > 0 else None
        per_speaker.append(
            PositiveLanguageSpeaker(
                speaker_id=speaker_id,
                speaker_name=speaker_name_by_id[speaker_id],
                eligible_token_count=eligible,
                positive_token_count=positive,
                proportion=proportion,
            )
        )

    meeting_eligible, meeting_positive = _positive_language_for_texts(
        [event.transcript for event in non_blank]
    )
    meeting_proportion = (meeting_positive / meeting_eligible) if meeting_eligible > 0 else None

    return PositiveLanguageAnalytics(
        per_speaker=tuple(per_speaker),
        meeting_eligible_token_count=meeting_eligible,
        meeting_positive_token_count=meeting_positive,
        meeting_proportion=meeting_proportion,
    )


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def analyze_content(timeline: Timeline) -> ContentAnalytics:
    """Compute Phase 2 content/language analytics from a Phase 0 Timeline.

    Deterministic for a fixed Timeline AND a fixed embedding model: topic
    grouping depends on MiniLM's (deterministic, non-random) encode output;
    keyword ranking and positive-language scoring are pure text
    computations with no randomness anywhere. Never mutates ``timeline`` or
    its events. Safe for an empty timeline (returns empty results).
    """

    events = timeline.events
    return ContentAnalytics(
        topics=_build_topics(events),
        keywords=_build_keywords(events),
        positive_language=_build_positive_language(events),
    )
