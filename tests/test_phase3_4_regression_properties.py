"""Phase 3.4 Part N: generic regression properties for the real-audio
failure class, WITHOUT encoding the real transcript.

Real-audio manual validation showed a short informational/demo meeting
(introductions, acknowledgements, questions, explanations, a few
substantive concepts, closing remarks) producing dozens of user-facing
"topics" -- greeting fragments, participant introductions, isolated
question fragments, and "next question"-style navigation phrases each
becoming their own card. This file asserts the GENERAL property that no
meeting of this general shape can do that anymore, using entirely
synthetic content and names distinct from the manually-tested sample.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from timeline import build_timeline
from meeting_analytics.content import analyze_content
from transcription.sarvam_client import TranscriptionResult, TranscriptionSegment


def _segment(text: str, speaker: str | None, start: float, end: float) -> TranscriptionSegment:
    return TranscriptionSegment(transcript=text, speaker_id=speaker, start_time_seconds=start, end_time_seconds=end)


# A short ~4-minute informational/demo meeting shape: introductions,
# acknowledgements, questions, explanations, a few substantive concepts,
# closing remarks. Names/content are entirely synthetic.
_DEMO_MEETING_SEGMENTS = [
    _segment("Hi everyone, thanks so much for joining today's product walkthrough.", "0", 0.0, 6.0),
    _segment("Hi, thanks for having me, excited to see what's new.", "1", 6.0, 12.0),
    _segment("Hello, glad to be here too.", "2", 12.0, 16.0),
    _segment("Great, so let's get started with the overview.", "0", 16.0, 20.0),
    _segment("This release focuses on the new analytics export feature.", "0", 20.0, 30.0),
    _segment("The analytics export feature lets users download reports as CSV or PDF.", "0", 30.0, 40.0),
    _segment("That's great, does the export feature support scheduled exports?", "1", 40.0, 46.0),
    _segment("Yes, the analytics export feature supports scheduling on a daily or weekly basis.", "0", 46.0, 56.0),
    _segment("Awesome, thanks for explaining that.", "2", 56.0, 60.0),
    _segment("Next question, is there a limit on export file size?", "1", 60.0, 66.0),
    _segment("Good question, the export feature currently caps files at fifty megabytes.", "0", 66.0, 76.0),
    _segment("Okay, got it, thanks.", "1", 76.0, 80.0),
    _segment("Sounds good.", "2", 80.0, 82.0),
    _segment("Alright, thanks everyone for joining, that wraps up the walkthrough.", "0", 82.0, 90.0),
    _segment("Thanks, bye everyone.", "1", 90.0, 94.0),
    _segment("Bye, thanks.", "2", 94.0, 98.0),
]

_GREETING_OR_FILLER_TITLE_MARKERS = (
    "hi", "hello", "thanks", "thank you", "bye", "goodbye", "okay", "great",
    "awesome", "sounds good", "got it", "next question", "good question",
)


def _analyze() -> object:
    result = TranscriptionResult(transcript="ignored", segments=_DEMO_MEETING_SEGMENTS)
    mapping = {"Speaker 1": "Reese", "Speaker 2": "Alex", "Speaker 3": "Jordan"}
    return analyze_content(build_timeline(result, mapping=mapping))


def test_short_informational_demo_meeting_produces_compact_keywords_not_dozens_of_topics() -> None:
    analytics = _analyze()
    # The internal clustering may still exist for Voxels alignment, but the
    # USER-FACING surface is the keyword/keyphrase list, which must be
    # compact, not dozens of entries.
    assert len(analytics.keywords.meeting_keywords) <= 10


def test_short_informational_demo_meeting_keywords_exclude_greetings_and_names() -> None:
    analytics = _analyze()
    lowered_keywords = [term.casefold() for term in analytics.keywords.meeting_keywords]
    for marker in _GREETING_OR_FILLER_TITLE_MARKERS:
        assert marker not in lowered_keywords
    for name in ("reese", "alex", "jordan"):
        assert name not in lowered_keywords


def test_short_informational_demo_meeting_keywords_surface_the_real_concept() -> None:
    analytics = _analyze()
    combined = " ".join(analytics.keywords.meeting_keywords).casefold()
    assert "export" in combined or "analytics" in combined


def test_internal_topic_clustering_does_not_leak_into_the_primary_content_analytics_surface() -> None:
    # ContentAnalytics.keywords (the SELaD "Conversation Summary") must
    # never itself be a per-fragment topic dump -- verified structurally:
    # far fewer keyword entries than there are raw transcript segments.
    analytics = _analyze()
    assert len(analytics.keywords.meeting_keywords) < len(_DEMO_MEETING_SEGMENTS)


def test_positive_language_cues_still_computed_for_this_meeting_shape() -> None:
    analytics = _analyze()
    assert analytics.positive_language.per_speaker
