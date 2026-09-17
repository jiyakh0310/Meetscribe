"""SELaD Phase 4 -- video preprocessing (Part 4).

Inspects an uploaded video file and normalizes it to an analysis-
compatible MP4/H.264 file ONLY when necessary, preserving timing. This
is a SEPARATE path from transcription.audio_utils's existing audio
FFmpeg conversion -- it never calls into, imports, or modifies that
module's behavior (Absolute Freeze). It reuses the same underlying
FFmpeg binary-discovery convention (imageio-ffmpeg, then system PATH)
since duplicating that logic would be a real inconsistency risk, but
keeps its own conversion function entirely separate.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2

from video_analytics.model import VideoMetadata

logger = logging.getLogger(__name__)


class VideoProcessingError(Exception):
    """Raised when video preprocessing fails. Always caught by
    video_analytics.pipeline and converted into a graceful
    ``VideoAnalyticsResult(available=False, ...)`` -- never allowed to
    propagate into the factual MoM pipeline (Part 17)."""


SUPPORTED_VIDEO_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".mov", ".m4v", ".avi", ".webm"})
"""Formats accepted for VIDEO analytics specifically. Distinct from
transcription.audio_utils.SUPPORTED_EXTENSIONS (audio-only path), which
is never modified here."""

_H264_MP4_CODECS = frozenset({"h264", "avc1"})


def _ffmpeg_binary() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    raise VideoProcessingError("FFmpeg is unavailable for video normalization.")


def inspect_video_metadata(video_path: Path) -> VideoMetadata:
    """Read basic container metadata via OpenCV, never fabricating a
    missing value.

    ``fps``/``frame_count`` are None when the container reports a
    non-positive value (common for some webm/variable-frame-rate
    sources) rather than defaulting to a guessed number (Part 4: "Do not
    silently fabricate FPS.").
    """

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise VideoProcessingError(f"Could not open video file: {video_path.name}")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        raw_fps = capture.get(cv2.CAP_PROP_FPS)
        raw_frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)

        fps = float(raw_fps) if raw_fps and raw_fps > 0 else None
        frame_count = int(raw_frame_count) if raw_frame_count and raw_frame_count > 0 else None

        if fps is not None and frame_count is not None:
            duration_seconds = frame_count / fps
        else:
            # Conservative fallback: walk the container's own duration
            # via CAP_PROP_POS_MSEC after seeking to the end is unreliable
            # across backends: safer to report 0.0 (never fabricate a
            # plausible-looking number) and let the caller treat this as
            # a quality warning.
            duration_seconds = 0.0

        if width <= 0 or height <= 0:
            raise VideoProcessingError(
                f"Video '{video_path.name}' reported an invalid frame size."
            )

        return VideoMetadata(
            duration_seconds=duration_seconds,
            fps=fps,
            width=width,
            height=height,
            frame_count=frame_count,
        )
    finally:
        capture.release()


def _needs_normalization(video_path: Path, metadata: VideoMetadata) -> bool:
    if video_path.suffix.lower() != ".mp4":
        return True
    # A quick, best-effort codec check via OpenCV's fourcc report. Not
    # all backends populate this reliably, so an unreadable/unknown
    # fourcc is treated as "normalize to be safe" rather than assumed OK.
    capture = cv2.VideoCapture(str(video_path))
    try:
        fourcc_int = int(capture.get(cv2.CAP_PROP_FOURCC) or 0)
    finally:
        capture.release()
    if fourcc_int <= 0:
        return True
    fourcc = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)).strip().lower()
    return fourcc not in _H264_MP4_CODECS


def normalize_video_for_analysis(video_path: Path) -> tuple[Path, bool]:
    """Return (path_to_analysis_ready_mp4, is_temporary_file).

    Only re-encodes when the source is not already MP4/H.264 -- most
    modern phone/webcam recordings already are, so this is the common
    fast path. Timing is preserved (no ``-r`` frame-rate override; only
    container/codec is normalized).
    """

    if not video_path.is_file():
        raise VideoProcessingError(f"Video file not found: {video_path}")
    if video_path.stat().st_size == 0:
        raise VideoProcessingError(f"Video file is empty: {video_path.name}")
    if video_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
        raise VideoProcessingError(
            f"Unsupported video format '{video_path.suffix}'. Supported: {supported}."
        )

    metadata = inspect_video_metadata(video_path)
    if metadata.duration_seconds <= 0:
        raise VideoProcessingError(f"Video '{video_path.name}' has zero duration.")

    if not _needs_normalization(video_path, metadata):
        return video_path, False

    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_file.close()
    output_path = Path(output_file.name)

    ffmpeg_bin = _ffmpeg_binary()
    command = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(video_path),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-an",  # audio is handled entirely by the existing, separate audio path
        str(output_path),
    ]
    try:
        subprocess.run(command, capture_output=True, check=True, timeout=600)
    except subprocess.CalledProcessError as exc:
        output_path.unlink(missing_ok=True)
        stderr = (exc.stderr or b"").decode("utf-8", errors="replace")[-500:]
        raise VideoProcessingError(
            f"Could not normalize video '{video_path.name}' for analysis. {stderr}"
        ) from exc
    except (OSError, subprocess.TimeoutExpired) as exc:
        output_path.unlink(missing_ok=True)
        raise VideoProcessingError(
            f"Video normalization failed for '{video_path.name}'."
        ) from exc

    if not output_path.exists() or output_path.stat().st_size == 0:
        output_path.unlink(missing_ok=True)
        raise VideoProcessingError(f"Video normalization produced an empty file for '{video_path.name}'.")

    return output_path, True
