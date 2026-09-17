"""SELaD Phase 4 -- approximate visual-interaction / eye-contact cue
(Part 12).

SELaD estimated eye contact / interpersonal visual interaction from
facial landmarks, face geometry, and participant seating position,
under an assumed stable, fixed-camera, side-by-side layout. MeetScribe
video sources are not guaranteed to share that layout (participants
may move, camera may be handheld, framing may vary), so this module
implements a deliberately CONSERVATIVE, coarse approximation rather
than claiming lab-grade eye tracking.

Heuristic: for each pair of tracks with usable landmark-derived yaw
estimates in a window, treat their orientation as "mutually oriented"
when each face's estimated horizontal yaw points toward the other
face's approximate screen position (left/right), which is the same
coarse signal SELaD's own geometry-based approach approximates.
Never described as definitive "eye contact" -- always qualified as an
estimate/cue.
"""

from __future__ import annotations

from statistics import mean

from video_analytics.model import WindowInteractionSummary

_YAW_TOLERANCE_DEGREES = 25.0
"""How close a face's estimated yaw must be to "facing across" (rather
than facing the camera or away) to count as oriented toward another
participant. Deliberately loose -- this is an approximation, not a
precise gaze vector."""


def compute_window_interaction(
    track_positions_and_yaws: dict[str, list[tuple[float, float]]],
) -> WindowInteractionSummary:
    """``track_positions_and_yaws`` maps track_id -> list of
    (horizontal_screen_position, yaw_estimate_degrees) samples observed
    in this window, where horizontal_screen_position is the face's
    normalized x-center (0=left edge, 1=right edge) and yaw_estimate
    comes from landmarks.py's coarse geometry heuristic.

    Returns a conservative WindowInteractionSummary. ``available`` is
    False whenever fewer than 2 tracks have usable geometry -- no
    interaction cue is ever forced onto a single visible participant.
    """

    usable_tracks = {
        track_id: samples for track_id, samples in track_positions_and_yaws.items() if samples
    }

    if len(usable_tracks) < 2:
        return WindowInteractionSummary(
            available=False,
            unavailable_reason="Fewer than two participants had usable facial geometry in this window.",
            tracks_with_geometry=len(usable_tracks),
            mutual_orientation_score=None,
            display_label="Visual interaction cue unavailable for this window.",
        )

    track_ids = list(usable_tracks.keys())
    track_avg_position = {
        track_id: mean(position for position, _yaw in samples)
        for track_id, samples in usable_tracks.items()
    }
    track_avg_yaw = {
        track_id: mean(yaw for _position, yaw in samples)
        for track_id, samples in usable_tracks.items()
    }

    oriented_pair_count = 0
    total_pairs = 0
    for i in range(len(track_ids)):
        for j in range(i + 1, len(track_ids)):
            id_a, id_b = track_ids[i], track_ids[j]
            total_pairs += 1

            # Determine each face's expected "toward the other" yaw
            # sign from their relative screen positions, then check
            # whether the estimated yaw is plausibly consistent with
            # facing that direction (within tolerance of facing
            # forward-across rather than away).
            a_faces_b = (track_avg_position[id_b] - track_avg_position[id_a])
            b_faces_a = -a_faces_b
            a_expected_sign = 1.0 if a_faces_b > 0 else -1.0
            b_expected_sign = 1.0 if b_faces_a > 0 else -1.0

            a_consistent = (track_avg_yaw[id_a] * a_expected_sign) > -_YAW_TOLERANCE_DEGREES
            b_consistent = (track_avg_yaw[id_b] * b_expected_sign) > -_YAW_TOLERANCE_DEGREES

            if a_consistent and b_consistent:
                oriented_pair_count += 1

    mutual_orientation_score = oriented_pair_count / total_pairs if total_pairs > 0 else None

    return WindowInteractionSummary(
        available=True,
        unavailable_reason=None,
        tracks_with_geometry=len(usable_tracks),
        mutual_orientation_score=mutual_orientation_score,
        display_label="Estimated eye-contact / visual-interaction cue",
    )
