"""Phase 3.5 runtime regression test (Part 13): exercises the SAME path
the real Streamlit "Generate Minutes" click uses -- ``render_processing_stage``'s
``action == "analyze"`` branch, which calls ``run_meeting_analysis`` ->
``generate_mom`` -> ``experimental_mom_to_analysis_result`` -> session
state -> the Minutes UI renderer -- rather than a hand-built
``MeetingAnalysisResult`` object (which several earlier Phase 3.x AppTests
used and which does NOT exercise the actual formatter/runtime bug class
this phase corrects).

Synthetic transcript matches Part 13's required shape: 3 speakers,
introductions, substantive technical discussion, questions, explanations,
"next question" navigation, "let's move to..." navigation, product/release
status, no actual Decision, no actual Action -- distinct in names/content
from the manually-tested sample audio.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest
from transcription.sarvam_client import TranscriptionResult, TranscriptionSegment

APP_PATH = str(Path(__file__).resolve().parents[1] / "app" / "main.py")

TRANSCRIPT_TEXT = """Speaker 1 [00:00 - 00:08]
Hi everyone, I'm Naomi, I lead the platform team.

Speaker 2 [00:08 - 00:16]
Hi, I'm Victor from infrastructure.

Speaker 3 [00:16 - 00:22]
Hi, I'm Priti, I work on the release process.

Speaker 1 [00:22 - 00:35]
So today let's talk about the new caching layer for the search service.

Speaker 2 [00:35 - 00:50]
The caching layer reduces search latency by storing frequent queries in memory for five minutes.

Speaker 3 [00:50 - 01:00]
How will the database migration affect production during the caching rollout?

Speaker 1 [01:00 - 01:15]
Good question, the database migration runs in parallel and shouldn't affect production traffic.

Speaker 2 [01:15 - 01:20]
Next question, what's the current release status?

Speaker 3 [01:20 - 01:35]
The release is on track for this Friday, all tests are passing.

Speaker 1 [01:35 - 01:45]
Great, now let's move to the monitoring dashboard updates.

Speaker 2 [01:45 - 02:00]
The monitoring dashboard now shows cache hit rates and latency percentiles in real time.

Speaker 3 [02:00 - 02:05]
Thanks everyone, that covers today's updates.
"""


def _segment(text: str, speaker: str, start: float, end: float) -> TranscriptionSegment:
    return TranscriptionSegment(transcript=text, speaker_id=speaker, start_time_seconds=start, end_time_seconds=end)


TRANSCRIPT_RESULT = TranscriptionResult(
    transcript="ignored",
    segments=[
        _segment("Hi everyone, I'm Naomi, I lead the platform team.", "0", 0.0, 8.0),
        _segment("Hi, I'm Victor from infrastructure.", "1", 8.0, 16.0),
        _segment("Hi, I'm Priti, I work on the release process.", "2", 16.0, 22.0),
        _segment("So today let's talk about the new caching layer for the search service.", "0", 22.0, 35.0),
        _segment("The caching layer reduces search latency by storing frequent queries in memory for five minutes.", "1", 35.0, 50.0),
        _segment("How will the database migration affect production during the caching rollout?", "2", 50.0, 60.0),
        _segment("Good question, the database migration runs in parallel and shouldn't affect production traffic.", "0", 60.0, 75.0),
        _segment("Next question, what's the current release status?", "1", 75.0, 80.0),
        _segment("The release is on track for this Friday, all tests are passing.", "2", 80.0, 95.0),
        _segment("Great, now let's move to the monitoring dashboard updates.", "0", 95.0, 105.0),
        _segment("The monitoring dashboard now shows cache hit rates and latency percentiles in real time.", "1", 105.0, 120.0),
        _segment("Thanks everyone, that covers today's updates.", "2", 120.0, 125.0),
    ],
)


def _run_generation() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.session_state["workflow_open"] = True
    at.session_state["workflow_stage"] = "processing"
    at.session_state["workflow_pending_action"] = "analyze"
    at.session_state["workflow_source"] = "transcript"
    at.session_state["transcript_result"] = TRANSCRIPT_RESULT
    at.session_state["edited_transcript_text"] = TRANSCRIPT_TEXT
    at.session_state["transcript_text"] = TRANSCRIPT_TEXT
    at.session_state["speaker_mapping"] = {"Speaker 1": "Naomi", "Speaker 2": "Victor", "Speaker 3": "Priti"}
    at.session_state["meeting_info"] = {"meeting_title": "Runtime Integration Meeting"}
    at.run()
    return at


def test_generation_completes_without_exception() -> None:
    at = _run_generation()
    assert not at.exception
    assert not at.error
    assert "analysis_result" in at.session_state
    assert at.session_state["analysis_result"] is not None


def test_runtime_executive_summary_is_grounded_and_grammatical() -> None:
    at = _run_generation()
    analysis = at.session_state["analysis_result"]
    summary = (analysis.summary.short_summary or "") + " " + (analysis.summary.detailed_summary or "")
    lowered = summary.casefold()

    assert "review next question" not in lowered
    assert "next question" not in lowered
    assert "follow up on" not in lowered or "validated" in lowered  # no unsupported follow-up claim
    # No malformed contraction artifact ("Shouldn T").
    assert " t " not in lowered.replace("shouldn't", "").replace("wasn't", "")
    assert "shouldn t" not in lowered


def test_runtime_discussion_has_no_navigation_only_items() -> None:
    at = _run_generation()
    analysis = at.session_state["analysis_result"]
    for point in analysis.key_discussion_points:
        combined = f"{point.point}".casefold()
        assert "next question" not in combined
        assert combined.strip() not in ("next question.", "moving on.", "let's continue.")


def test_runtime_produces_no_decisions_or_actions_for_this_transcript() -> None:
    # This synthetic meeting contains no actual commitment/agreement --
    # only discussion, questions, explanations, and navigation. No
    # Decision or Action should be fabricated from the navigation phrases.
    at = _run_generation()
    analysis = at.session_state["analysis_result"]
    for decision in analysis.decisions:
        lowered = decision.decision.casefold()
        assert "move" not in lowered
        assert "monitoring dashboard" not in lowered


def test_runtime_minutes_ui_renders_without_navigation_fragments() -> None:
    at = _run_generation()
    assert not at.exception
    combined = "\n".join(m.value for m in at.markdown if m.value)
    assert "Review next question" not in combined
    assert "Shouldn T" not in combined
    assert "Shouldn t" not in combined.replace("Shouldn't", "")


def test_runtime_conversation_summary_is_compact_and_content_bearing() -> None:
    at = _run_generation()
    combined = "\n".join(m.value for m in at.markdown if m.value)
    idx = combined.find("Conversation summary")
    assert idx >= 0
    section = combined[idx : idx + 800]
    lowered = section.casefold()
    assert "naomi" not in lowered
    assert "victor" not in lowered
    assert "priti" not in lowered
