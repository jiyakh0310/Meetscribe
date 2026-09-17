"""SELaD Phase 4.1 repair #2 -- real browser DOM/layout verification.

Unlike tests/test_phase4_1_recorder_js_contract.py (static source audits),
this file drives the ACTUAL recorder HTML in a real Chromium browser via
Playwright, with getUserMedia/MediaRecorder replaced by a synthetic
canvas+WebAudio MediaStream (real camera/microphone hardware is not
available in this environment, but the browser's own layout/rendering
engine is -- these tests exercise that engine for real, not a simulation
of it). This is what actually proves the invisible-Stop-button bug is
fixed: real computed styles and bounding rects, not string matching.

Skips gracefully (not fails) if Playwright/Chromium cannot be launched in
this environment.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

RECORDER_HTML_PATH = Path(__file__).resolve().parents[1] / "components" / "video_meeting_recorder" / "frontend" / "index.html"

_MOCK_MEDIA_SCRIPT = """
window.__mockStreamActive = true;
navigator.mediaDevices.getUserMedia = async (constraints) => {
  const canvas = document.createElement('canvas');
  canvas.width = 640;
  canvas.height = 360;
  const ctx = canvas.getContext('2d');
  let hue = 0;
  function draw() {
    hue = (hue + 2) % 360;
    ctx.fillStyle = `hsl(${hue}, 70%, 50%)`;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    if (window.__mockStreamActive) requestAnimationFrame(draw);
  }
  draw();
  const videoStream = canvas.captureStream(24);

  const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const dest = audioCtx.createMediaStreamDestination();
  const osc = audioCtx.createOscillator();
  osc.connect(dest);
  osc.start();

  return new MediaStream([
    ...videoStream.getVideoTracks(),
    ...dest.stream.getAudioTracks(),
  ]);
};
"""


@pytest.fixture(scope="module")
def browser_page():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        pytest.skip("Playwright is not installed in this environment.")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                args=["--use-fake-ui-for-media-stream", "--autoplay-policy=no-user-gesture-required"]
            )
            context = browser.new_context(permissions=["camera", "microphone"])
            page = context.new_page()
            page.add_init_script(_MOCK_MEDIA_SCRIPT)
            page.goto(RECORDER_HTML_PATH.as_uri())
            yield page
            browser.close()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Could not launch a real Chromium browser in this environment: {exc}")


def _is_effectively_visible(page, selector: str) -> bool:
    """True if the element is displayed, non-zero-sized, and has a
    non-transparent opacity -- i.e. actually visible, not merely present
    in the DOM (which static source tests cannot distinguish)."""

    return page.evaluate(
        """(sel) => {
            const el = document.querySelector(sel);
            if (!el) return false;
            const style = window.getComputedStyle(el);
            if (style.display === 'none' || style.visibility === 'hidden' || parseFloat(style.opacity) === 0) {
                return false;
            }
            const rect = el.getBoundingClientRect();
            return rect.width > 0 && rect.height > 0;
        }""",
        selector,
    )


def _bounding_rect(page, selector: str) -> dict:
    return page.evaluate(
        """(sel) => {
            const el = document.querySelector(sel);
            const r = el.getBoundingClientRect();
            return { top: r.top, bottom: r.bottom, left: r.left, right: r.right };
        }""",
        selector,
    )


def test_ready_state_shows_start_not_stop(browser_page) -> None:
    page = browser_page
    assert _is_effectively_visible(page, "#startBtn")
    assert not _is_effectively_visible(page, "#stopBtn")
    assert not _is_effectively_visible(page, "#clearBtn")
    assert not _is_effectively_visible(page, "#processBtn")


def test_stop_button_is_actually_visible_while_recording(browser_page) -> None:
    # This is the exact real-browser bug report: recording badge/timer/
    # camera visible, but no reachable Stop Recording control. Click Start
    # and inspect REAL computed layout, not just recorderState.
    page = browser_page
    page.click("#startBtn")
    page.wait_for_selector("#stopBtn:not([style*='display: none'])", timeout=5000)

    assert _is_effectively_visible(page, "#recBadge"), "recording badge should be visible while recording"
    assert _is_effectively_visible(page, "#stopBtn"), "Stop Recording must be visible while recording"

    # The button must also be within the document's rendered content --
    # not positioned/clipped somewhere unreachable.
    stop_rect = _bounding_rect(page, "#stopBtn")
    assert stop_rect["bottom"] > stop_rect["top"]
    assert stop_rect["right"] > stop_rect["left"]


def test_iframe_reported_height_covers_the_stop_button(browser_page) -> None:
    # The specific mechanism behind the original bug: Streamlit sizes the
    # component iframe from postMessage("streamlit:setFrameHeight", ...)
    # and has no scrollbar of its own, so if the reported height is
    # smaller than the Stop button's actual position, it would be
    # invisible in the real embedded (non-standalone) case even though it
    # is "visible" by this standalone page's own scrollable viewport.
    # Assert document.documentElement.scrollHeight -- the exact value
    # postHeight() reports -- reaches past the Stop button's bottom edge.
    page = browser_page
    stop_rect = _bounding_rect(page, "#stopBtn")
    reported_height = page.evaluate("document.documentElement.scrollHeight")
    assert reported_height >= stop_rect["bottom"], (
        f"Reported scrollHeight ({reported_height}) does not cover the Stop button's "
        f"bottom edge ({stop_rect['bottom']}) -- Streamlit would clip it."
    )


def test_live_preview_is_mirrored_while_recording(browser_page) -> None:
    page = browser_page
    transform = page.evaluate("window.getComputedStyle(document.querySelector('#liveVideo')).transform")
    # scaleX(-1) as a computed matrix is matrix(-1, 0, 0, 1, 0, 0).
    assert transform not in ("none", ""), "live preview should have a mirror transform applied"
    assert transform.startswith("matrix(-1")


def test_badges_and_timer_are_not_mirrored(browser_page) -> None:
    page = browser_page
    badge_transform = page.evaluate("window.getComputedStyle(document.querySelector('#recBadge')).transform")
    timer_transform = page.evaluate("window.getComputedStyle(document.querySelector('#timer')).transform")
    assert badge_transform in ("none", "matrix(1, 0, 0, 1, 0, 0)")
    assert timer_transform in ("none", "matrix(1, 0, 0, 1, 0, 0)")


def test_stopping_reaches_recorded_state_with_visible_completion_controls(browser_page) -> None:
    page = browser_page
    # Let MediaRecorder actually accumulate at least one real chunk before
    # stopping -- stopping within milliseconds of starting can produce a
    # zero-byte Blob (a real recorder characteristic, not a component bug),
    # which the component correctly routes to ERROR rather than RECORDED.
    page.wait_for_timeout(1500)
    page.click("#stopBtn")
    # FINALIZING is expected to be brief; wait for RECORDED's controls.
    page.wait_for_selector("#processBtn:not([disabled])", timeout=10000)

    assert _is_effectively_visible(page, "#readyBanner")
    assert _is_effectively_visible(page, "#playbackVideo")
    assert _is_effectively_visible(page, "#clearBtn")
    assert _is_effectively_visible(page, "#processBtn")
    assert not _is_effectively_visible(page, "#stopBtn")
    assert not _is_effectively_visible(page, "#startBtn")

    process_rect = _bounding_rect(page, "#processBtn")
    clear_rect = _bounding_rect(page, "#clearBtn")
    reported_height = page.evaluate("document.documentElement.scrollHeight")
    assert reported_height >= process_rect["bottom"]
    assert reported_height >= clear_rect["bottom"]


def test_playback_video_is_not_mirrored(browser_page) -> None:
    page = browser_page
    transform = page.evaluate("window.getComputedStyle(document.querySelector('#playbackVideo')).transform")
    assert transform in ("none", "matrix(1, 0, 0, 1, 0, 0)")


def test_recorded_duration_is_small_not_max_seconds(browser_page) -> None:
    # The recording in this test ran for well under a minute -- the
    # completion screen must reflect that, never 60:00.
    page = browser_page
    duration_text = page.inner_text("#durationMeta")
    assert "60:00" not in duration_text
    assert "Duration:" in duration_text
