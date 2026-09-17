"""SELaD Phase 4 -- video analytics package.

A functional adaptation of SELaD's visual-sensing methodology (facial
expression, FaceMesh-derived interaction cues, expression synchrony)
applied to MeetScribe meeting recordings.

ISOLATION POLICY: this package is never imported by, and never
imports from, ``transcription/``, ``ml_mom/``, ``meeting_analytics/``,
or ``integrations/voxels_ser.py``. The only intended dependency
direction is: this package -> ``app/main.py`` (UI), one-way. It has no
effect on the factual MoM pipeline, Sarvam transcription/diarization,
or the existing Voxels acoustic-emotion pipeline.

Public entry point: ``analyze_video``.
"""

from video_analytics.model import VideoAnalyticsResult
from video_analytics.pipeline import STANDARD_LIMITATIONS, analyze_video

__all__ = ["analyze_video", "VideoAnalyticsResult", "STANDARD_LIMITATIONS"]
