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

    positive_phrase_count: int = 0
    """Phase 3.2: count of explicit multi-word cue matches (``POSITIVE_PHRASES``,
    e.g. "thank you", "sounds good") found in this speaker's text. Additive
    evidence alongside the single-token count above -- deliberately NOT
    folded into ``positive_token_count``/``proportion`` to avoid double-
    counting a phrase whose individual words may already be lexicon tokens
    (e.g. "thank" in "thank you"), and to keep the existing proportion
    calculation exactly as it was before this field existed."""

    matched_phrases: tuple[str, ...] = ()
    """Distinct matched phrases (evidence), for optional UI display."""


@dataclass(frozen=True, slots=True)
class PositiveLanguageAnalytics:
    per_speaker: tuple[PositiveLanguageSpeaker, ...]
    meeting_eligible_token_count: int
    meeting_positive_token_count: int
    meeting_proportion: float | None
    meeting_positive_phrase_count: int = 0
    """Phase 3.2: meeting-wide sum of ``PositiveLanguageSpeaker.positive_phrase_count``."""


@dataclass(frozen=True, slots=True)
class ContentAnalytics:
    """Phase 2 result: topics, keywords, and positive-language, all derived
    from the same Timeline Phase 1 already consumes."""

    topics: tuple[TopicAnalytics, ...]
    keywords: KeywordAnalytics
    positive_language: PositiveLanguageAnalytics


@dataclass(frozen=True, slots=True)
class AnalyticsSemanticUnit:
    """Phase 3.2: an analytics-only sentence-level slice of one
    ``TimelineEvent``, used ONLY for topic clustering -- never for the
    factual MoM pipeline, never written back to ``timeline`` or
    ``TranscriptionSegment``.

    A single long Sarvam turn (introduction + question + explanation +
    topic change, all in one diarized segment) is too coarse a unit for
    topic clustering: embedding the whole turn as one vector blends
    unrelated concepts into a single noisy point. This type lets one
    ``TimelineEvent`` contribute several independently-embedded semantic
    units while remaining fully traceable back to that one real event --
    it never introduces a new source of truth for timing or transcript
    text (see ``source_start_time_seconds``/``source_end_time_seconds``).
    """

    unit_id: int
    """0-based position across ALL units for this meeting, in chronological
    (event, then within-event) order -- used only for the adjacency rule in
    ``_cluster_units``, not a public identifier."""

    source_event_id: int
    """The originating ``TimelineEvent.event_id``. Multiple units may share
    the same ``source_event_id`` when one event was segmented into several
    sentences -- traceability back to Phase 0 always goes through this
    field, never through a fabricated per-sentence event."""

    speaker_id: str
    speaker_name: str

    text: str
    """This unit's slice of the parent event's verbatim transcript text
    (a sentence, or a deterministic long-run fallback chunk -- see
    ``_segment_event_text``). Still real transcript text, never rewritten,
    summarized, or generated."""

    source_start_time_seconds: float | None
    source_end_time_seconds: float | None
    """Inherited verbatim from the PARENT event's own interval -- never a
    fabricated per-sentence sub-interval. Multiple units from the same
    event carry the identical parent interval; topic-level start/end is
    still derived from genuine per-EVENT timing (see ``_build_topics``),
    not by treating each unit as if it had its own distinct timestamp."""


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
    +
    # Phase 3.2: common contractions. _tokenize keeps internal apostrophes
    # (see its docstring), so "let's"/"don't"/"that's" each tokenize as ONE
    # word distinct from their expanded stopword forms ("let"/"do"/"that")
    # above -- without these, a contraction of an otherwise-stopword phrase
    # slips through as if it were meaningful keyword/title evidence (e.g.
    # "let's" appearing as a topic keyword next to a real term).
    """
    let's don't didn't isn't wasn't weren't aren't that's it's there's
    here's what's who's you're we're they're i'm i've you've we've
    they've i'll you'll we'll they'll can't couldn't shouldn't wouldn't
    haven't hasn't hadn't
    """.split()
)

# Phase 3.2: generic conversational discourse noise that survives
# _tokenize/STOPWORDS but is still not meaningful evidence for a topic
# TITLE or a Key Theme on its own (greetings, introductions, filler
# acknowledgements, generic meeting-process verbs). Deliberately curated
# from broad, general conversational-English categories -- not from any
# one transcript's wording -- and deliberately BARE/SINGULAR forms only:
# a plural or otherwise-inflected sibling (e.g. "questions", "rights",
# "talks") is left untouched because it is far more likely to carry real
# meaning ("customer questions", "access rights") than the bare noise form
# ("question?", "right." as a filler acknowledgement, "let's talk").
# This is the same conservative, phrase-context-aware handling the task
# calls for, achieved without any part-of-speech or semantic analysis.
CONVERSATIONAL_NOISE_WORDS = frozenset(
    """
    hello hi hey thanks thank name person speaker today okay right great
    question talk discuss meeting someone thing
    """.split()
    +
    # Phase 3.2: bare ordinal/sequence discourse markers ("First, let's...",
    # "Second, I want to...", "Finally, ...") are structural signposts for
    # ORGANIZING a turn, not evidence of what that turn's segment is ABOUT --
    # a general pattern in any multi-point speaker turn, in any domain, not
    # specific to one transcript. Left as bare forms only, same conservative
    # rule as the rest of this set (e.g. "third-party" would still tokenize
    # as separate words and is unaffected; a genuinely meaningful ordinal
    # phrase is vanishingly rare in meeting topic evidence).
    """
    first second third finally next
    """.split()
)


def _participant_name_noise_tokens(events: Sequence[TimelineEvent]) -> frozenset[str]:
    """Dynamically derive noise tokens from this meeting's OWN reviewed
    speaker display names -- never a hardcoded name list. A multi-word name
    ("John Smith") contributes each of its component tokens ("john",
    "smith") individually, since either one alone is what would otherwise
    leak into an unrelated topic title or keyword ("Name Tom", "Sumit
    Help"). Generic/common-word names are handled the same conservative way
    as any other noise token: excluded only as a candidate on their own or
    paired with another noise/name token (see ``_candidate_ngrams``), never
    globally deleted from the transcript -- a bigram made of one name token
    and one otherwise-meaningful token still gets excluded too (see
    ``_candidate_ngrams``'s docstring for why that conservative trade-off
    was chosen)."""

    tokens: set[str] = set()
    for event in events:
        for token in _tokenize(event.speaker_name):
            tokens.add(token)
    return frozenset(tokens)


def _is_noise_token(token: str, noise_terms: frozenset[str]) -> bool:
    return token in STOPWORDS or token in CONVERSATIONAL_NOISE_WORDS or token in noise_terms


def _candidate_ngrams(text: str, noise_terms: frozenset[str]) -> list[str]:
    """Generate unigram + bigram keyword/title candidates from ``text``.

    A bigram is only produced from two ADJACENT tokens in the original
    (unfiltered) token sequence, so word order/context is preserved --
    "database migration" is a candidate exactly because those two words sit
    next to each other in the transcript, not because both happen to be
    frequent somewhere in the meeting.

    A bigram is rejected if EITHER token is noise (stopword, conversational
    filler, or a participant-name token). This single rule is what keeps
    "the api", "we need", "right so", and "thanks everyone" out (their
    stopword/filler half fails the check) and is also the conservative
    choice for names: a bigram pairing a real name with an otherwise
    meaningful word (e.g. "tom project") is excluded too, rather than
    attempting to judge whether that specific occurrence is "really about"
    the person -- a false negative here (a legitimately name-relevant
    phrase not being title-eligible) is preferred over a false positive
    (a participant's name dominating a topic title/Key Theme).
    """

    raw_tokens = _tokenize(text)
    candidates: list[str] = []
    for token in raw_tokens:
        if not _is_noise_token(token, noise_terms):
            candidates.append(token)
    for first, second in zip(raw_tokens, raw_tokens[1:]):
        if not _is_noise_token(first, noise_terms) and not _is_noise_token(second, noise_terms):
            candidates.append(f"{first} {second}")
    return candidates


def _rank_keywords(
    documents: Sequence[str],
    top_n: int,
    noise_terms: frozenset[str] = frozenset(),
) -> list[str]:
    """Deterministic TF-IDF-style keyword/keyphrase ranking across
    ``documents`` (each a semantic unit's or event's text, i.e. this
    meeting's own segments treated as a small corpus).

    Separate from, and unrelated to, the TF-IDF baseline classifier in
    ``ml_mom/logistic_baseline.py`` -- that scores whole sentences for a
    5-class label; this ranks individual terms/phrases for display as
    keywords.

        idf(term) = ln((1 + N) / (1 + df(term))) + 1   (smoothed)
        score(term) = total_frequency(term) * idf(term)
        adjusted(term) = score(term) * 1.15 if term is a phrase (2 words)

    Ranked descending by adjusted score; ties broken by first-appearance
    order so the result is fully deterministic. The small fixed multi-word
    bonus (not a separate scoring model) reflects that a concrete phrase
    ("production deployment") is more useful, less ambiguous evidence than
    an equally-frequent bare unigram ("deployment") -- see module-level
    Phase 3.2 notes. A unigram is then dropped from the final ranked list
    if a higher-ranked selected phrase already contains it (or vice
    versa), so a short result never shows both "deployment" and
    "production deployment" as if they were two distinct concepts.

    ``noise_terms`` (typically this meeting's own participant-name tokens,
    see ``_participant_name_noise_tokens``) are excluded the same way
    ``STOPWORDS``/``CONVERSATIONAL_NOISE_WORDS`` are -- see
    ``_candidate_ngrams``.
    """

    doc_candidate_lists = [_candidate_ngrams(document, noise_terms) for document in documents]
    n_docs = len(doc_candidate_lists)
    if n_docs == 0:
        return []

    doc_frequency: dict[str, int] = {}
    total_frequency: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    position = 0
    for candidates in doc_candidate_lists:
        seen_in_doc: set[str] = set()
        for term in candidates:
            total_frequency[term] = total_frequency.get(term, 0) + 1
            if term not in first_seen:
                first_seen[term] = position
                position += 1
            seen_in_doc.add(term)
        for term in seen_in_doc:
            doc_frequency[term] = doc_frequency.get(term, 0) + 1

    if not total_frequency:
        return []

    # A bigram that occurs only once is exactly as likely to be a
    # meaningless adjacent-word accident ("plan needs", "needs review") as
    # a genuine concept -- neither token being a stopword is not, by
    # itself, enough evidence. Requiring at least one REPEATED occurrence
    # (the same two words adjacent more than once, anywhere in the corpus)
    # is the deterministic, frequency-based signal the task calls for;
    # unigrams have no such requirement since a single strong unigram
    # occurrence is still meaningful evidence on its own.
    _MIN_PHRASE_FREQUENCY = 2
    total_frequency = {
        term: frequency
        for term, frequency in total_frequency.items()
        if " " not in term or frequency >= _MIN_PHRASE_FREQUENCY
    }
    if not total_frequency:
        return []

    _PHRASE_BONUS = 1.15
    scores = {
        term: frequency * (math.log((1 + n_docs) / (1 + doc_frequency[term])) + 1)
        for term, frequency in total_frequency.items()
    }
    adjusted_scores = {
        term: (score * _PHRASE_BONUS if " " in term else score) for term, score in scores.items()
    }
    ranked = sorted(adjusted_scores.items(), key=lambda item: (-item[1], first_seen[item[0]]))

    selected: list[str] = []
    selected_word_sets: list[set[str]] = []
    for term, _score in ranked:
        term_words = set(term.split())
        if any(term_words <= existing or existing <= term_words for existing in selected_word_sets):
            continue
        selected.append(term)
        selected_word_sets.append(term_words)
        if len(selected) >= top_n:
            break
    return selected


def _topic_title(keywords: Sequence[str], fallback_index: int) -> str:
    """Deterministic, keyword-derived topic title. Never calls an LLM.
    Falls back to "Discussion Topic N" (N is 1-based) when there is not
    enough lexical evidence to name the topic, rather than guessing.

    ``keywords`` is the already-ranked output of ``_rank_keywords``, so the
    top entry is the strongest evidence. If it is already a two-word phrase
    (e.g. "database migration"), that phrase IS the title -- title-casing
    each of its words rather than appending a second term. Only when the
    top entry is a single word do we fall back to the prior Phase 2
    behavior of combining it with the next single-word candidate for a
    more specific two-word title.
    """

    if not keywords:
        return f"Discussion Topic {fallback_index}"
    top = keywords[0]
    if " " in top:
        return " ".join(word.capitalize() for word in top.split())
    if len(keywords) > 1 and " " not in keywords[1]:
        return " ".join(word.capitalize() for word in keywords[:2])
    return top.capitalize()


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
# Phase 3.2: analytics-only semantic segmentation
# --------------------------------------------------------------------------

_SENTENCE_BOUNDARY_PATTERN = re.compile(r"[.!?\n]+")
"""Deterministic sentence boundaries: '.', '?', '!', or a newline (common
transcript-line separator). No ML model, no language-specific NLP -- this
is intentionally the same lightweight class of tool as ``_tokenize``."""

_MAX_UNSPLIT_WORD_COUNT = 40
"""Above this many words with NO sentence boundary at all, real STT output
is almost certainly under-punctuated run-on text (weak Sarvam punctuation
on a long turn), not one genuinely long sentence -- see
``_split_event_text``'s fallback."""

_FALLBACK_CHUNK_WORD_COUNT = 25
"""Deterministic chunk size for the long-run fallback split. Chosen as a
"few sentences" of English text -- large enough that short utterances are
never affected (they never reach ``_MAX_UNSPLIT_WORD_COUNT``), small enough
that one chunk still embeds as one reasonably coherent semantic unit
instead of the whole multi-topic turn."""


def _split_event_text(text: str) -> list[str]:
    """Split one event's transcript into candidate sentence-level pieces.

    Primary strategy: split on real sentence punctuation/newlines. Only a
    piece with NO such boundary at all AND more than
    ``_MAX_UNSPLIT_WORD_COUNT`` words falls back to a fixed-size word-count
    chunk split -- a last resort for weakly-punctuated long-run STT text,
    never the primary segmentation strategy, and never triggered for
    ordinary short-to-medium utterances.
    """

    pieces = [piece.strip() for piece in _SENTENCE_BOUNDARY_PATTERN.split(text) if piece.strip()]
    if not pieces:
        return []

    result: list[str] = []
    for piece in pieces:
        words = piece.split()
        if len(words) <= _MAX_UNSPLIT_WORD_COUNT:
            result.append(piece)
            continue
        for start in range(0, len(words), _FALLBACK_CHUNK_WORD_COUNT):
            chunk = " ".join(words[start : start + _FALLBACK_CHUNK_WORD_COUNT])
            if chunk:
                result.append(chunk)
    return result


def _is_non_substantive_fragment(text: str) -> bool:
    """A fragment with zero non-noise tokens (e.g. "Okay." "Right." "Yeah
    yeah.") -- pure filler, never a short MEANINGFUL phrase. A genuinely
    short but substantive phrase like "API integration." or "Beta
    program." always has at least one non-noise token and is therefore
    never treated as filler by this check."""

    tokens = _tokenize(text)
    if not tokens:
        return True
    return all(_is_noise_token(token, frozenset()) for token in tokens)


def _segment_event_text(text: str) -> list[str]:
    """Deterministically segment one event's transcript into semantic
    pieces, merging clearly non-substantive filler fragments into an
    adjacent real piece rather than discarding or standalone-emitting them.

    A filler fragment merges into the PRECEDING accepted piece when one
    already exists (e.g. "We should ship this. Right." -> one piece); a
    filler fragment appearing before any substantive content accumulates
    and prefixes the next substantive piece instead (e.g. "Okay, so, the
    database migration is ready." -> "Okay, so, the database migration is
    ready." stays one piece since "so"/"okay" are noise but the sentence as
    a whole has real content and was never split apart from it in the
    first place -- this rule only ever matters when an ENTIRE piece,
    between two boundaries, is pure filler). If literally nothing
    substantive is ever found, the accumulated filler survives as the
    event's one unit rather than silently vanishing.
    """

    raw_pieces = _split_event_text(text)
    merged: list[str] = []
    pending_prefix = ""
    for piece in raw_pieces:
        candidate = f"{pending_prefix} {piece}".strip() if pending_prefix else piece
        if _is_non_substantive_fragment(piece):
            if merged:
                merged[-1] = f"{merged[-1]} {piece}".strip()
            else:
                pending_prefix = candidate
            continue
        merged.append(candidate)
        pending_prefix = ""
    if pending_prefix and not merged:
        merged.append(pending_prefix)
    return merged


def _build_semantic_units(events: Sequence[TimelineEvent]) -> list[AnalyticsSemanticUnit]:
    """Expand each non-blank ``TimelineEvent`` into one or more
    ``AnalyticsSemanticUnit``s, in chronological order. Never touches
    ``events`` -- purely a read-and-derive step."""

    units: list[AnalyticsSemanticUnit] = []
    for event in events:
        if not event.transcript.strip():
            continue
        is_timed = event.timing_status is TimingStatus.TIMED
        for piece in _segment_event_text(event.transcript):
            units.append(
                AnalyticsSemanticUnit(
                    unit_id=len(units),
                    source_event_id=event.event_id,
                    speaker_id=event.speaker_id,
                    speaker_name=event.speaker_name,
                    text=piece,
                    source_start_time_seconds=event.start_time_seconds if is_timed else None,
                    source_end_time_seconds=event.end_time_seconds if is_timed else None,
                )
            )
    return units


# --------------------------------------------------------------------------
# Topic grouping
# --------------------------------------------------------------------------

# Chosen lower than the live formatter's 0.74 (ml_mom/experimental/
# mom_formatter.py's build_topic_clusters), which embeds multi-sentence
# CONTEXT WINDOWS -- longer, semantically richer, more stable vectors.
# Analytics topics group short sentence-level semantic units instead
# (shorter, noisier embeddings, and deliberately NOT the formatter's own
# clusters -- see module docstring point 1), so a lower threshold is
# needed to avoid fragmenting a normal meeting into dozens of one-unit
# "topics". 0.5 was chosen as a moderate middle ground for this shorter,
# noisier text unit; it is a documented heuristic, not empirically tuned
# against a labeled dataset, and is UNCHANGED from Phase 2 -- Phase 3.2
# improves the INPUT UNITS (see above) and adds the adjacency rule below,
# rather than re-tuning this number, per the instruction to justify any
# threshold change with a diverse regression set before touching it (see
# tests/test_phase3_2_content_quality.py's Case F/G for over-/under-
# fragmentation checks against the unchanged threshold).
TOPIC_SIMILARITY_THRESHOLD = 0.5


def _cluster_units(embeddings: Sequence[Sequence[float]] | None, count: int) -> list[list[int]]:
    """Group semantic-unit indices [0, count) by embedding similarity.

    Single-pass, deterministic, and identical in spirit to Phase 2's
    per-event clustering: each item joins the most similar existing
    cluster if that similarity is >= TOPIC_SIMILARITY_THRESHOLD (a
    cluster's centroid is the running mean of its members' unit vectors),
    otherwise it starts a new cluster. When embeddings are unavailable
    (``embeddings is None``), everything falls back to a single cluster
    rather than fabricating semantic grouping or crashing.

    Phase 3.2 addition -- consecutive context: meeting conversation is
    chronological, so before falling back to the pure centroid comparison,
    also compare the current unit directly against the IMMEDIATELY
    PRECEDING unit's own vector (not that cluster's running centroid). If
    that direct adjacent-pair similarity already clears
    TOPIC_SIMILARITY_THRESHOLD, the current unit joins the previous unit's
    cluster. This uses the SAME semantic bar as ordinary clustering -- it
    is not a "merge because adjacent" rule, only a "the immediately
    preceding sentence and this one are still clearly on the same subject,
    even if the cluster's centroid (built from several earlier, possibly
    more tangential members) has drifted enough that the pure centroid
    match alone would have missed it" rule. This is what keeps an extended
    question -> answer -> follow-up exchange as one topic instead of
    fragmenting once a longer back-and-forth pulls the centroid away from
    the newest, still-obviously-related unit.
    """

    if embeddings is None or count == 0:
        return [list(range(count))] if count else []

    clusters: list[dict[str, object]] = []
    unit_vectors: list[list[float]] = []
    cluster_of_unit: dict[int, int] = {}

    for index in range(count):
        vector = list(embeddings[index])
        norm = math.sqrt(sum(value * value for value in vector))
        unit_vector = [value / norm for value in vector] if norm > 0 else vector
        unit_vectors.append(unit_vector)

        best_cluster_index = -1
        best_similarity = -1.0
        for cluster_index, cluster in enumerate(clusters):
            centroid = cluster["centroid"]  # type: ignore[index]
            similarity = sum(a * b for a, b in zip(unit_vector, centroid))
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster_index = cluster_index

        if index - 1 in cluster_of_unit:
            adjacent_similarity = sum(a * b for a, b in zip(unit_vector, unit_vectors[index - 1]))
            if adjacent_similarity >= TOPIC_SIMILARITY_THRESHOLD and adjacent_similarity > best_similarity:
                best_cluster_index = cluster_of_unit[index - 1]
                best_similarity = adjacent_similarity

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
            cluster_of_unit[index] = best_cluster_index
        else:
            clusters.append({"indices": [index], "centroid": unit_vector})
            cluster_of_unit[index] = len(clusters) - 1

    return [cluster["indices"] for cluster in clusters]  # type: ignore[misc]


def _build_topics(events: Sequence[TimelineEvent]) -> tuple[TopicAnalytics, ...]:
    units = _build_semantic_units(events)
    if not units:
        return ()

    event_by_id = {event.event_id: event for event in events}
    noise_terms = _participant_name_noise_tokens(events)

    embeddings = _embed_texts([unit.text for unit in units])
    clusters = _cluster_units(embeddings, len(units))

    topics: list[TopicAnalytics] = []
    for topic_index, indices in enumerate(clusters, start=1):
        cluster_units = [units[i] for i in indices]
        cluster_texts = [unit.text for unit in cluster_units]
        keywords = _rank_keywords(cluster_texts, top_n=5, noise_terms=noise_terms)

        # Unique contributing source events, in first-appearance order.
        # Multiple units from the same event (a long turn segmented into
        # several sentences) contribute exactly ONE entry here -- never a
        # fabricated extra "event" per sentence.
        seen_event_ids: list[int] = []
        for unit in cluster_units:
            if unit.source_event_id not in seen_event_ids:
                seen_event_ids.append(unit.source_event_id)
        contributing_events = [event_by_id[event_id] for event_id in seen_event_ids]

        timed = [event for event in contributing_events if event.timing_status is TimingStatus.TIMED]
        start = min((event.start_time_seconds for event in timed), default=None)
        end = max((event.end_time_seconds for event in timed), default=None)

        participants: list[str] = []
        for event in contributing_events:
            if event.speaker_name not in participants:
                participants.append(event.speaker_name)

        preview = cluster_units[0].text.strip()
        if len(preview) > 140:
            preview = preview[:137].rstrip() + "..."

        topics.append(
            TopicAnalytics(
                topic_id=topic_index,
                title=_topic_title(keywords, topic_index),
                keywords=tuple(keywords[:5]),
                participant_speakers=tuple(participants),
                event_count=len(contributing_events),
                start_time_seconds=start,
                end_time_seconds=end,
                source_event_ids=tuple(seen_event_ids),
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
    noise_terms = _participant_name_noise_tokens(events)
    meeting_keywords = _rank_keywords(
        [event.transcript for event in non_blank], top_n=10, noise_terms=noise_terms
    )

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
        keywords = (
            _rank_keywords(texts, top_n=5, noise_terms=noise_terms)
            if token_count >= _MIN_SPEAKER_TOKENS_FOR_KEYWORDS
            else []
        )
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


# Phase 3.2: a small, explicit, deterministic lexicon of common
# MULTI-WORD positive-communication cues that a single-token lexicon
# necessarily misses (e.g. "sounds good" -- neither "sounds" nor "good"
# alone is as clear a communication-cue signal as the phrase is). Kept
# deliberately short and generic -- not built from, or tuned against, any
# one meeting's transcript. Counted SEPARATELY from the token-level
# eligible/positive counts above (see ``_positive_phrase_matches``) so the
# existing proportion calculation is completely unaffected by this
# addition, and a phrase whose words already happen to be individual
# POSITIVE_LEXICON tokens (e.g. "thank" inside "thank you") is never
# double-counted into a single combined metric.
POSITIVE_PHRASES: tuple[str, ...] = (
    "good job",
    "well done",
    "sounds good",
    "great work",
    "thank you",
    "makes sense",
)

_PHRASE_NORMALIZE_PATTERN = re.compile(r"[^a-z0-9\s]")


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


def _positive_phrase_matches(texts: Sequence[str]) -> tuple[str, ...]:
    """Return every ``POSITIVE_PHRASES`` occurrence across ``texts``, one
    entry per match (so repeated phrases are each counted). Matching is a
    literal, case-insensitive, punctuation-normalized substring search --
    still fully deterministic and explainable, not a sentiment model."""

    normalized = _PHRASE_NORMALIZE_PATTERN.sub(" ", " ".join(texts).lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return ()
    padded = f" {normalized} "
    matches: list[str] = []
    for phrase in POSITIVE_PHRASES:
        needle = f" {phrase} "
        start = 0
        while True:
            found = padded.find(needle, start)
            if found == -1:
                break
            matches.append(phrase)
            start = found + 1
    return tuple(matches)


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
        speaker_texts = texts_by_speaker[speaker_id]
        eligible, positive = _positive_language_for_texts(speaker_texts)
        proportion = (positive / eligible) if eligible > 0 else None
        phrase_matches = _positive_phrase_matches(speaker_texts)
        per_speaker.append(
            PositiveLanguageSpeaker(
                speaker_id=speaker_id,
                speaker_name=speaker_name_by_id[speaker_id],
                eligible_token_count=eligible,
                positive_token_count=positive,
                proportion=proportion,
                positive_phrase_count=len(phrase_matches),
                matched_phrases=tuple(sorted(set(phrase_matches))),
            )
        )

    all_texts = [event.transcript for event in non_blank]
    meeting_eligible, meeting_positive = _positive_language_for_texts(all_texts)
    meeting_proportion = (meeting_positive / meeting_eligible) if meeting_eligible > 0 else None
    meeting_phrase_matches = _positive_phrase_matches(all_texts)

    return PositiveLanguageAnalytics(
        per_speaker=tuple(per_speaker),
        meeting_eligible_token_count=meeting_eligible,
        meeting_positive_token_count=meeting_positive,
        meeting_proportion=meeting_proportion,
        meeting_positive_phrase_count=len(meeting_phrase_matches),
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
