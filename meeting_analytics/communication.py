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


@dataclass(frozen=True, slots=True)
class TopicCommunicationInsight:
    """Voxels-derived acoustic pattern aligned to one Phase 2 topic.

    ``available=False`` covers every case where no genuine insight exists
    (untimed topic, no overlapping windows, Voxels unavailable) --
    ``dominant_acoustic_emotion``/``emotion_distribution``/
    ``communication_tone``/``pattern`` are then all None/empty rather than
    fabricated.
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
    emotion_distribution: tuple[tuple[str, float], ...]
    """(emotion, average probability across this topic's contributing
    windows), sorted descending. Empty when unavailable."""

    average_confidence: float | None
    communication_tone: str | None
    """One of Voxels' own tone categories via ``voxels_tone_for`` (Positive/
    Neutral/Negative/Mixed-Uncertain) -- never a new judgment scale."""

    pattern: str | None
    """Short observational phrase (matches the wording already used by
    ``app.main.build_topic_emotion_insights`` for the equivalent
    discussion-point-level insight, e.g. "Mostly neutral speech-emotion
    pattern")."""


@dataclass(frozen=True, slots=True)
class CommunicationInsights:
    """Meeting-level communication insights: a pass-through summary of the
    already-computed Voxels result, plus topic-aligned detail."""

    available: bool
    unavailable_reason: str | None

    dominant_acoustic_emotion: str | None
    communication_tone: str | None
    average_confidence: float | None
    windows_analyzed: int | None
    quality_warnings: tuple[str, ...]

    per_topic: tuple[TopicCommunicationInsight, ...]
    """Only topics with ``available=True`` genuinely have aligned evidence;
    topics with no timing or no overlapping windows still appear here with
    ``available=False`` so the UI can render them consistently rather than
    silently omitting some topics."""


def _pattern_for_tone(tone: str) -> str:
    """Matches app.main.build_topic_emotion_insights's tone->pattern text
    exactly, so meeting-level and topic-level wording never diverges."""

    if tone == "Negative":
        return "Increased negative speech-emotion signals"
    if tone == "Positive":
        return "Mostly positive speech-emotion pattern"
    if tone == "Neutral":
        return "Mostly neutral speech-emotion pattern"
    return "Mixed / uncertain speech-emotion pattern"


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

    return TopicCommunicationInsight(
        **base,
        available=True,
        unavailable_reason=None,
        windows_analyzed=len(contributing),
        dominant_acoustic_emotion=dominant.title(),
        emotion_distribution=distribution,
        average_confidence=average_confidence,
        communication_tone=tone,
        pattern=_pattern_for_tone(tone),
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
            windows_analyzed=None,
            quality_warnings=(),
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
            windows_analyzed=None,
            quality_warnings=(),
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
        windows_analyzed=windows_analyzed,
        quality_warnings=quality_warnings,
        per_topic=per_topic,
    )
