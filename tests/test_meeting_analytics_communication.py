"""Tests for the SELaD Phase 3 topic-aligned communication insights
(meeting_analytics/communication.py).

Style matches tests/test_timeline.py, tests/test_meeting_analytics.py, and
tests/test_meeting_analytics_content.py: plain pytest-style functions with
bare asserts, no unittest.TestCase.
"""

import copy

from meeting_analytics.communication import NO_AUDIO_MESSAGE, align_topics_with_voxels
from meeting_analytics.content import TopicAnalytics


def _topic(
    topic_id: int,
    title: str = "Sample Topic",
    start=None,
    end=None,
    source_event_ids=(0,),
) -> TopicAnalytics:
    return TopicAnalytics(
        topic_id=topic_id,
        title=title,
        keywords=("sample",),
        participant_speakers=("Jiya",),
        event_count=len(source_event_ids),
        start_time_seconds=start,
        end_time_seconds=end,
        source_event_ids=source_event_ids,
        representative_text="Sample text.",
    )


def _window(start: float, end: float, probabilities: dict, confidence: float = 0.6) -> dict:
    return {"start_time": start, "end_time": end, "probabilities": probabilities, "confidence": confidence}


def _voxels_payload(windows: list[dict], **overrides) -> dict:
    payload = {
        "available": True,
        "dominant_emotion": "Neutral",
        "tone": "Neutral",
        "confidence": 0.7,
        "windows_analyzed": len(windows),
        "quality_warnings": [],
        "windows": windows,
    }
    payload.update(overrides)
    return payload


def _by_id(insights, topic_id: int):
    for topic in insights.per_topic:
        if topic.topic_id == topic_id:
            return topic
    raise AssertionError(f"no topic communication insight for id {topic_id!r}")


# --- A: topic + Voxels overlap -----------------------------------------------


def test_topic_voxels_overlap_associates_correct_windows() -> None:
    topic = _topic(1, start=10.0, end=30.0)
    windows = [
        _window(0.0, 4.0, {"neutral": 1.0}),   # before topic -> excluded
        _window(8.0, 12.0, {"happy": 1.0}),    # overlaps [10,12] -> included
        _window(12.0, 16.0, {"happy": 1.0}),   # fully inside -> included
        _window(28.0, 32.0, {"sad": 1.0}),     # overlaps [28,30] -> included
        _window(40.0, 44.0, {"angry": 1.0}),   # after topic -> excluded
    ]
    insights = align_topics_with_voxels([topic], _voxels_payload(windows))
    result = _by_id(insights, 1)
    assert result.available is True
    assert result.windows_analyzed == 3
    assert result.dominant_acoustic_emotion == "Happy"


# --- B: no overlap -------------------------------------------------------------


def test_no_overlapping_window_produces_no_fabricated_insight() -> None:
    topic = _topic(1, start=10.0, end=20.0)
    windows = [_window(100.0, 104.0, {"neutral": 1.0})]
    insights = align_topics_with_voxels([topic], _voxels_payload(windows))
    result = _by_id(insights, 1)
    assert result.available is False
    assert result.dominant_acoustic_emotion is None
    assert result.emotion_distribution == ()
    assert result.windows_analyzed == 0
    assert result.unavailable_reason is not None


# --- C: untimed topic -----------------------------------------------------------


def test_untimed_topic_has_no_voxels_alignment() -> None:
    topic = _topic(1, start=None, end=None)
    windows = [_window(0.0, 4.0, {"neutral": 1.0})]
    insights = align_topics_with_voxels([topic], _voxels_payload(windows))
    result = _by_id(insights, 1)
    assert result.available is False
    assert "timing is unavailable" in result.unavailable_reason.lower()


# --- D: no Voxels ---------------------------------------------------------------


def test_no_voxels_payload_keeps_content_analytics_available_without_crash() -> None:
    topic = _topic(1, start=10.0, end=20.0)
    insights = align_topics_with_voxels([topic], None)
    assert insights.available is False
    assert insights.unavailable_reason == NO_AUDIO_MESSAGE
    assert insights.per_topic == ()  # topics list intentionally not echoed when there's no audio at all


def test_voxels_unavailable_flag_produces_graceful_unavailable_result() -> None:
    topic = _topic(1, start=10.0, end=20.0)
    insights = align_topics_with_voxels([topic], {"available": False})
    assert insights.available is False
    assert insights.per_topic == ()


# --- E: multiple topics, boundary-overlap window --------------------------------


def test_boundary_overlapping_window_goes_to_the_larger_overlap_topic() -> None:
    # A window is assigned to exactly ONE topic: whichever it overlaps the
    # MOST -- the same "largest-overlap-wins" rule already established by
    # app.main.build_topic_emotion_insights for window-to-topic aggregation
    # (distinct from align_voxels_windows_to_transcript's window-to-SEGMENT
    # alignment, which intentionally allows many-to-many matches). A window
    # that overlaps two topics unequally goes to the topic with more overlap.
    topic_a = _topic(1, start=10.0, end=20.0, source_event_ids=(0,))
    topic_b = _topic(2, start=20.0, end=30.0, source_event_ids=(1,))
    # Overlaps topic_a [10,20] by 3s (17-20) and topic_b [20,30] by 1s (20-21) -> topic_a wins.
    boundary_window = _window(17.0, 21.0, {"happy": 1.0})
    other_window_a = _window(10.0, 14.0, {"neutral": 1.0})
    other_window_b = _window(24.0, 28.0, {"sad": 1.0})
    windows = [other_window_a, boundary_window, other_window_b]

    insights = align_topics_with_voxels([topic_a, topic_b], _voxels_payload(windows))
    result_a = _by_id(insights, 1)
    result_b = _by_id(insights, 2)

    assert result_a.windows_analyzed == 2  # other_window_a + boundary_window (larger overlap)
    assert result_b.windows_analyzed == 1  # other_window_b only


def test_multiple_non_overlapping_topics_align_independently() -> None:
    topic_a = _topic(1, "Release Testing", start=10.0, end=20.0, source_event_ids=(0,))
    topic_b = _topic(2, "Marketing Campaign", start=40.0, end=50.0, source_event_ids=(1,))
    windows = [
        _window(10.0, 14.0, {"neutral": 1.0}),
        _window(40.0, 44.0, {"sad": 1.0}),
    ]
    insights = align_topics_with_voxels([topic_a, topic_b], _voxels_payload(windows))
    result_a = _by_id(insights, 1)
    result_b = _by_id(insights, 2)
    assert result_a.dominant_acoustic_emotion == "Neutral"
    assert result_b.dominant_acoustic_emotion == "Sad"


# --- F: invalid/malformed window timing --------------------------------------------


def test_malformed_window_timing_is_skipped_gracefully() -> None:
    topic = _topic(1, start=10.0, end=20.0)
    windows = [
        {"start_time": "not-a-number", "end_time": 14.0, "probabilities": {"neutral": 1.0}},
        {"start_time": 20.0, "end_time": 10.0, "probabilities": {"happy": 1.0}},  # end < start
        {"end_time": 14.0, "probabilities": {"sad": 1.0}},  # missing start_time
        _window(12.0, 16.0, {"happy": 1.0}),  # the only valid one
    ]
    insights = align_topics_with_voxels([topic], _voxels_payload(windows))
    result = _by_id(insights, 1)
    assert result.available is True
    assert result.windows_analyzed == 1
    assert result.dominant_acoustic_emotion == "Happy"


# --- G: source traceability ----------------------------------------------------------


def test_topic_id_and_source_event_ids_are_unchanged_by_alignment() -> None:
    topic = _topic(7, "Traceable Topic", start=10.0, end=20.0, source_event_ids=(3, 4, 5))
    windows = [_window(10.0, 14.0, {"neutral": 1.0})]
    insights = align_topics_with_voxels([topic], _voxels_payload(windows))
    result = _by_id(insights, 7)
    assert result.topic_id == 7
    assert result.source_event_ids == (3, 4, 5)


# --- H: determinism -------------------------------------------------------------------


def test_same_inputs_produce_the_same_aligned_result() -> None:
    topic = _topic(1, start=10.0, end=20.0)
    windows = [_window(10.0, 14.0, {"neutral": 0.6, "happy": 0.4}), _window(14.0, 18.0, {"happy": 0.7, "neutral": 0.3})]
    payload = _voxels_payload(windows)

    first = align_topics_with_voxels([topic], payload)
    second = align_topics_with_voxels([topic], payload)

    assert first.per_topic[0].dominant_acoustic_emotion == second.per_topic[0].dominant_acoustic_emotion
    assert first.per_topic[0].emotion_distribution == second.per_topic[0].emotion_distribution
    assert first.per_topic[0].average_confidence == second.per_topic[0].average_confidence


# --- I: Voxels immutability -----------------------------------------------------------


def test_voxels_payload_is_not_mutated() -> None:
    topic = _topic(1, start=10.0, end=20.0)
    windows = [_window(10.0, 14.0, {"neutral": 1.0})]
    payload = _voxels_payload(windows)
    payload_before = copy.deepcopy(payload)

    align_topics_with_voxels([topic], payload)

    assert payload == payload_before


# --- J: content analytics (TopicAnalytics) immutability --------------------------------


def test_topic_analytics_is_not_mutated() -> None:
    topic = _topic(1, "Immutable Topic", start=10.0, end=20.0, source_event_ids=(0, 1))
    windows = [_window(10.0, 14.0, {"neutral": 1.0})]

    align_topics_with_voxels([topic], _voxels_payload(windows))

    # TopicAnalytics is a frozen dataclass -- any attempted mutation would
    # raise; this additionally confirms field values are untouched.
    assert topic.title == "Immutable Topic"
    assert topic.start_time_seconds == 10.0
    assert topic.end_time_seconds == 20.0
    assert topic.source_event_ids == (0, 1)
