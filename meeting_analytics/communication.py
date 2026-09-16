"""SELaD Phase 3 -- topic-aligned communication insights.

Connects Phase 2 ``TopicAnalytics`` to the ALREADY-COMPUTED Voxels speech-
emotion result at the presentation/alignment layer only. This module never
runs Voxels inference, never modifies it, and is never imported by the
factual MoM pipeline. Dependency direction stays one-way:

    Timeline -> meeting_analytics.content (topics) -+
    integrations.voxels_ser (already-run result)    -+--> meeting_analytics.communication -> UI

Reuse policy (see the Phase 3 audit that preceded this module):

- ``integrations.voxels_ser.run_voxels_emotion()`` and
  ``app.main.align_voxels_windows_to_transcript()`` are NOT called from
  here. Voxels inference and transcript-window alignment already happened
  earlier in the app's normal flow, and this module only reads their
  RESULT (``st.session_state.voxels_emotion_result``, passed in by the
  caller) -- specifically each window's ``start_time``/``end_time``/
  ``probabilities``/``confidence``, which are present on every window
  regardless of whether transcript alignment ran.
- ``app.main.build_topic_emotion_insights()`` is NOT reused as-is: it
  anchors topics using only a single timestamp per factual discussion
  point and infers each topic's END by extending to the NEXT discussion
  point's start. Phase 2's ``TopicAnalytics`` already carries genuine,
  independently-derived ``start_time_seconds``/``end_time_seconds`` for
  each topic, so borrowing that boundary-inference trick would be less
  accurate, not more reusable. What IS reused from it: the exact same
  interval-overlap formula, the "largest overlap wins" tie-break for a
  window spanning two topics, the probability-averaging/dominant-emotion
  math, and ``integrations.voxels_ser._tone_for`` (imported in app/main.py
  as ``voxels_tone_for`` -- the same already-established reuse pattern
  followed here) -- so results stay numerically and textually consistent
  with the existing meeting-level and discussion-point-level insights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from integrations.voxels_ser import _tone_for as voxels_tone_for
from meeting_analytics.content import TopicAnalytics

NO_AUDIO_MESSAGE = "Speech-emotion insights are available for meetings with audio."
"""Exact wording required for transcript-only meetings (no Voxels ever ran)."""


# --------------------------------------------------------------------------
# Phase 3.3 -- shared, presentation-only uncertainty interpretation
# --------------------------------------------------------------------------
#
# Real-audio validation showed a general bug class: the Voxels model's
# argmax class was presented as a confident meeting-level conclusion
# ("Dominant detected speech emotion: Disgust" / "Derived meeting tone:
# Negative") even when the underlying distribution was nearly flat (e.g.
# Disgust 18% barely ahead of Neutral 17%, Sad 16%, ...). This is NOT a
# Voxels inference bug -- argmax is a perfectly correct summary of "the
# single highest-probability class" -- it is a PRESENTATION bug: argmax
# alone does not tell a reader whether that class was a clear signal or a
# statistical coin-flip among several similarly-likely classes.
#
# ``interpret_emotion_distribution`` is the ONE place this judgment is
# made. Every caller that needs to decide "should this be presented as a
# clear pattern or as an uncertain one" -- the meeting-level summary here,
# the per-topic insights here, ``app.main.build_topic_emotion_insights``,
# and the PDF exporter -- calls this SAME function, so the three surfaces
# can never silently diverge into different uncertainty rules. It never
# touches raw Voxels probabilities/confidence/windows; it only reads them
# and returns a separate, additional judgment.
_CLEAR_TOP_PROBABILITY_THRESHOLD = 0.40
"""The leading class must represent a plausible plurality of the acoustic
evidence -- comfortably above the ~14.3% a uniform 7-class distribution
would give each class -- before its class label is presented as the
meeting's/topic's acoustic pattern at all, regardless of margin."""

_CLEAR_MARGIN_THRESHOLD = 0.15
"""The leading class must also clear the runner-up by a comfortable margin
(15 percentage points) -- a 1-point lead (e.g. 18% vs 17%, the real-audio
validation case) is a statistical near-tie, not a "dominant" class, no
matter how the two threshold constants are tuned; both constants were
chosen by reasoning about the general problem (chance level for 7 classes,
what a "comfortable" separation looks like) and validated against
synthetic distributions spanning strongly-dominant, borderline, and
near-uniform cases -- never against the one real validation meeting's
actual numbers."""


@dataclass(frozen=True, slots=True)
class EmotionDistributionInterpretation:
    """A presentation-only judgment derived from a raw Voxels probability
    distribution. Never mutates or replaces the raw distribution -- see
    ``interpret_emotion_distribution``.
    """

    status: str
    """One of "clear", "mixed", or "unavailable" (no usable probabilities
    were supplied at all -- e.g. an empty dict)."""

    highest_class: str | None
    highest_probability: float | None
    second_class: str | None
    second_probability: float | None
    margin: float | None
    """``highest_probability - second_probability``. None when there is no
    second class to compare against (a single-class distribution) or when
    ``status == "unavailable"``."""

    display_pattern: str
    """Human-facing phrase for the acoustic pattern itself, e.g. "Happy-
    associated acoustic pattern" (status == "clear") or "No clearly
    dominant acoustic pattern was detected." (status == "mixed"/
    "unavailable"). Never a claim about a person or the meeting."""

    secondary_tone: str | None
    """Voxels' own Positive/Neutral/Negative/Mixed-Uncertain tone category
    (via ``voxels_tone_for``), populated ONLY when ``status == "clear"`` --
    callers should never headline this for an uncertain distribution, so
    leaving it None for "mixed"/"unavailable" lets a caller show it
    unconditionally without repeating the status check."""


def interpret_emotion_distribution(
    probabilities: dict[str, Any] | None,
) -> EmotionDistributionInterpretation:
    """Judge whether ``probabilities`` (a Voxels class -> probability
    mapping, meeting-level or topic-level, already averaged across
    whichever windows contributed) shows a clear dominant acoustic pattern
    or a mixed/low-separation one.

    Deterministic rule (see module-level threshold docstrings for why):
    "clear" requires BOTH the top class's probability to be
    >= ``_CLEAR_TOP_PROBABILITY_THRESHOLD`` AND its margin over the
    second-highest class to be >= ``_CLEAR_MARGIN_THRESHOLD``. A
    single-class distribution (no second class to compare against) is
    "clear" by construction -- there is nothing for it to be mixed with.
    Ties are broken deterministically by sorting on (-probability, class
    name) so repeated calls with the same input always pick the same
    highest/second class regardless of dict insertion order.

    Never raises on malformed input (missing/non-numeric values, an empty
    dict, ``None``) -- degrades to ``status="unavailable"`` instead.
    """

    if not probabilities:
        return EmotionDistributionInterpretation(
            status="unavailable",
            highest_class=None,
            highest_probability=None,
            second_class=None,
            second_probability=None,
            margin=None,
            display_pattern="No acoustic distribution is available.",
            secondary_tone=None,
        )

    parsed: list[tuple[str, float]] = []
    for label, value in probabilities.items():
        try:
            parsed.append((str(label), float(value)))
        except (TypeError, ValueError):
            continue
    if not parsed:
        return EmotionDistributionInterpretation(
            status="unavailable",
            highest_class=None,
            highest_probability=None,
            second_class=None,
            second_probability=None,
            margin=None,
            display_pattern="No acoustic distribution is available.",
            secondary_tone=None,
        )

    ranked = sorted(parsed, key=lambda item: (-item[1], item[0]))
    highest_class, highest_probability = ranked[0]
    second_class, second_probability = (ranked[1] if len(ranked) > 1 else (None, None))
    margin = (highest_probability - second_probability) if second_probability is not None else None

    is_clear = (
        highest_probability >= _CLEAR_TOP_PROBABILITY_THRESHOLD
        and (margin is None or margin >= _CLEAR_MARGIN_THRESHOLD)
    )

    if is_clear:
        return EmotionDistributionInterpretation(
            status="clear",
            highest_class=highest_class,
            highest_probability=highest_probability,
            second_class=second_class,
            second_probability=second_probability,
            margin=margin,
            display_pattern=f"{highest_class.title()}-associated acoustic pattern",
            secondary_tone=voxels_tone_for(highest_class),
        )

    return EmotionDistributionInterpretation(
        status="mixed",
        highest_class=highest_class,
        highest_probability=highest_probability,
        second_class=second_class,
        second_probability=second_probability,
        margin=margin,
        display_pattern="No clearly dominant acoustic pattern was detected.",
        secondary_tone=None,
    )


@dataclass(frozen=True, slots=True)
class TopicCommunicationInsight:
    """Voxels-derived acoustic pattern aligned to one Phase 2 topic.

    ``available=False`` covers every case where no genuine insight exists
    (untimed topic, no overlapping windows, Voxels unavailable) --
    ``dominant_acoustic_emotion``/``emotion_distribution``/
    ``communication_tone``/``pattern``/``distribution_interpretation`` are
    then all None/empty rather than fabricated.
    """

    topic_id: int
    topic_title: str
    start_time_seconds: float | None
    end_time_seconds: float | None
    participant_speakers: tuple[str, ...]
    source_event_ids: tuple[int, ...]

    available: bool
    unavailable_reason: str | None

    windows_analyzed: int
    dominant_acoustic_emotion: str | None
    """Raw argmax class of this topic's averaged distribution -- UNCHANGED
    Phase 3 semantics, kept for backward compatibility. See
    ``distribution_interpretation`` for the uncertainty-aware judgment;
    prefer it for any human-facing headline."""

    emotion_distribution: tuple[tuple[str, float], ...]
    """(emotion, average probability across this topic's contributing
    windows), sorted descending. Empty when unavailable."""

    average_confidence: float | None
    communication_tone: str | None
    """One of Voxels' own tone categories via ``voxels_tone_for`` (Positive/
    Neutral/Negative/Mixed-Uncertain) -- never a new judgment scale. Raw
    pass-through of the argmax-derived tone, kept for backward
    compatibility; human-facing presentation should prefer
    ``distribution_interpretation.secondary_tone`` (None for a mixed/
    uncertain distribution, unlike this field)."""

    pattern: str | None
    """Phase 3.3: now ``distribution_interpretation.display_pattern`` --
    uncertainty-aware wording (e.g. "No clearly dominant acoustic pattern
    was detected." instead of always naming the argmax class)."""

    distribution_interpretation: "EmotionDistributionInterpretation | None" = None
    """Phase 3.3: the full shared interpretation for this topic's averaged
    distribution -- see ``interpret_emotion_distribution``. None only when
    ``available`` is False."""


@dataclass(frozen=True, slots=True)
class CommunicationInsights:
    """Meeting-level communication insights: a pass-through summary of the
    already-computed Voxels result, plus topic-aligned detail."""

    available: bool
    unavailable_reason: str | None

    dominant_acoustic_emotion: str | None
    """Raw argmax class of the meeting-wide distribution -- UNCHANGED Phase
    3 semantics, kept for backward compatibility. See
    ``distribution_interpretation`` for the uncertainty-aware judgment."""

    communication_tone: str | None
    """Raw pass-through of Voxels' own meeting-level tone. Human-facing
    presentation should prefer ``distribution_interpretation.secondary_tone``
    (None for a mixed/uncertain distribution)."""

    average_confidence: float | None
    """Phase 3.3 note: this is Voxels' top-level ``confidence`` field,
    which is literally the winning class's averaged probability (see
    ``integrations.voxels_ser.run_voxels_emotion``) -- NOT a calibrated
    statistical confidence. Kept under its established name for backward
    compatibility; prefer ``distribution_interpretation.highest_probability``
    (the same number) in new human-facing text, described as "highest
    model probability", and ``average_window_confidence`` (a genuinely
    different, per-window metric) where that is what is actually meant."""

    average_window_confidence: float | None
    """Phase 3.3 addition: Voxels' own ``average_window_confidence`` (the
    mean of each window's OWN model confidence, distinct from the
    meeting-level winning-class probability above) -- surfaced so callers
    that want a genuine confidence figure have one under an accurate name."""

    windows_analyzed: int | None
    quality_warnings: tuple[str, ...]

    distribution_interpretation: "EmotionDistributionInterpretation | None"
    """Phase 3.3: the shared uncertainty-aware interpretation of the
    meeting-wide probability distribution. None only when ``available`` is
    False."""

    per_topic: tuple[TopicCommunicationInsight, ...]
    """Only topics with ``available=True`` genuinely have aligned evidence;
    topics with no timing or no overlapping windows still appear here with
    ``available=False`` so the UI can render them consistently rather than
    silently omitting some topics."""


def _valid_windows(voxels_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Windows with usable float start_time/end_time. Malformed windows are
    skipped, never fabricated -- matches align_voxels_windows_to_transcript's
    own defensive handling of the same fields."""

    valid: list[dict[str, Any]] = []
    for window in voxels_payload.get("windows", []) or []:
        if not isinstance(window, dict):
            continue
        try:
            start = float(window["start_time"])
            end = float(window["end_time"])
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        valid.append({**window, "start_time": start, "end_time": end})
    return valid


def _assign_windows_to_topics(
    topics: Sequence[TopicAnalytics],
    windows: Sequence[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    """Assign each window to at most one topic: the topic it overlaps the
    MOST (interval-overlap = min(ends) - max(starts), same formula as
    align_voxels_windows_to_transcript). Ties keep the first topic
    encountered (topics are processed in their given, stable order).

    This mirrors app.main.build_topic_emotion_insights's own
    "largest-overlap-wins" rule exactly (it is the established precedent
    for window-to-TOPIC aggregation specifically -- as opposed to
    align_voxels_windows_to_transcript's window-to-SEGMENT alignment, which
    intentionally allows one window to match many segments). Using the same
    rule here keeps topic-level communication insights numerically
    consistent with the existing discussion-point-level insights rather
    than introducing a second, contradictory topic-overlap algorithm.
    """

    timed_topics = [topic for topic in topics if topic.start_time_seconds is not None and topic.end_time_seconds is not None]
    groups: dict[int, list[dict[str, Any]]] = {topic.topic_id: [] for topic in timed_topics}

    for window in windows:
        best_topic_id: int | None = None
        best_overlap = 0.0
        for topic in timed_topics:
            overlap = min(window["end_time"], topic.end_time_seconds) - max(
                window["start_time"], topic.start_time_seconds
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_topic_id = topic.topic_id
        if best_topic_id is not None:
            groups[best_topic_id].append(window)

    return groups


def _insight_for_topic(
    topic: TopicAnalytics,
    contributing: Sequence[dict[str, Any]],
) -> TopicCommunicationInsight:
    base = dict(
        topic_id=topic.topic_id,
        topic_title=topic.title,
        start_time_seconds=topic.start_time_seconds,
        end_time_seconds=topic.end_time_seconds,
        participant_speakers=topic.participant_speakers,
        source_event_ids=topic.source_event_ids,
    )

    if topic.start_time_seconds is None or topic.end_time_seconds is None:
        return TopicCommunicationInsight(
            **base,
            available=False,
            unavailable_reason="Topic timing is unavailable; speech-emotion alignment requires timed evidence.",
            windows_analyzed=0,
            dominant_acoustic_emotion=None,
            emotion_distribution=(),
            average_confidence=None,
            communication_tone=None,
            pattern=None,
        )

    if not contributing:
        return TopicCommunicationInsight(
            **base,
            available=False,
            unavailable_reason="No aligned speech-emotion windows for this topic.",
            windows_analyzed=0,
            dominant_acoustic_emotion=None,
            emotion_distribution=(),
            average_confidence=None,
            communication_tone=None,
            pattern=None,
        )

    totals: dict[str, float] = {}
    for window in contributing:
        for emotion, probability in (window.get("probabilities") or {}).items():
            try:
                totals[str(emotion)] = totals.get(str(emotion), 0.0) + float(probability)
            except (TypeError, ValueError):
                continue

    if not totals:
        return TopicCommunicationInsight(
            **base,
            available=False,
            unavailable_reason="Aligned windows did not contain usable emotion probabilities.",
            windows_analyzed=len(contributing),
            dominant_acoustic_emotion=None,
            emotion_distribution=(),
            average_confidence=None,
            communication_tone=None,
            pattern=None,
        )

    averaged = {emotion: value / len(contributing) for emotion, value in totals.items()}
    dominant = max(averaged, key=averaged.get)
    distribution = tuple(sorted(averaged.items(), key=lambda item: item[1], reverse=True))

    confidences: list[float] = []
    for window in contributing:
        try:
            confidences.append(float(window.get("confidence", 0.0)))
        except (TypeError, ValueError):
            continue
    average_confidence = (sum(confidences) / len(confidences)) if confidences else None

    tone = voxels_tone_for(dominant)
    interpretation = interpret_emotion_distribution(averaged)

    return TopicCommunicationInsight(
        **base,
        available=True,
        unavailable_reason=None,
        windows_analyzed=len(contributing),
        dominant_acoustic_emotion=dominant.title(),
        emotion_distribution=distribution,
        average_confidence=average_confidence,
        communication_tone=tone,
        pattern=interpretation.display_pattern,
        distribution_interpretation=interpretation,
    )


def align_topics_with_voxels(
    topics: Sequence[TopicAnalytics],
    voxels_payload: dict[str, Any] | None,
) -> CommunicationInsights:
    """Build meeting-level + topic-aligned communication insights.

    ``voxels_payload`` should be the SAME already-computed
    ``st.session_state.voxels_emotion_result`` dict the Minutes UI's
    Emotion & Communication Insights section already reads -- this
    function never triggers Voxels inference itself. ``None`` (the
    transcript-only case, where Voxels never ran) and
    ``{"available": False, ...}`` (Voxels ran but produced no usable
    result) are both handled without fabricating any insight.

    Pure and deterministic for fixed inputs; never mutates ``topics`` or
    ``voxels_payload``.
    """

    if voxels_payload is None:
        return CommunicationInsights(
            available=False,
            unavailable_reason=NO_AUDIO_MESSAGE,
            dominant_acoustic_emotion=None,
            communication_tone=None,
            average_confidence=None,
            average_window_confidence=None,
            windows_analyzed=None,
            quality_warnings=(),
            distribution_interpretation=None,
            per_topic=(),
        )

    if not voxels_payload.get("available"):
        reason = str(
            voxels_payload.get("alignment_reason")
            or "Speech-emotion insights were unavailable for this recording."
        )
        return CommunicationInsights(
            available=False,
            unavailable_reason=reason,
            dominant_acoustic_emotion=None,
            communication_tone=None,
            average_confidence=None,
            average_window_confidence=None,
            windows_analyzed=None,
            quality_warnings=(),
            distribution_interpretation=None,
            per_topic=(),
        )

    windows = _valid_windows(voxels_payload)
    assigned = _assign_windows_to_topics(topics, windows)
    per_topic = tuple(_insight_for_topic(topic, assigned.get(topic.topic_id, [])) for topic in topics)

    confidence = voxels_payload.get("confidence")
    try:
        average_confidence = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        average_confidence = None

    window_confidence = voxels_payload.get("average_window_confidence")
    try:
        average_window_confidence = float(window_confidence) if window_confidence is not None else None
    except (TypeError, ValueError):
        average_window_confidence = None

    windows_analyzed = voxels_payload.get("windows_analyzed")
    try:
        windows_analyzed = int(windows_analyzed) if windows_analyzed is not None else None
    except (TypeError, ValueError):
        windows_analyzed = None

    quality_warnings = tuple(
        str(warning) for warning in (voxels_payload.get("quality_warnings") or [])
    )

    return CommunicationInsights(
        available=True,
        unavailable_reason=None,
        dominant_acoustic_emotion=(
            str(voxels_payload.get("dominant_emotion")) if voxels_payload.get("dominant_emotion") else None
        ),
        communication_tone=str(voxels_payload.get("tone")) if voxels_payload.get("tone") else None,
        average_confidence=average_confidence,
        average_window_confidence=average_window_confidence,
        windows_analyzed=windows_analyzed,
        quality_warnings=quality_warnings,
        distribution_interpretation=interpret_emotion_distribution(voxels_payload.get("probabilities")),
        per_topic=per_topic,
    )
