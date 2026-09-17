"""SELaD Phase 4 -- expression synchrony (Part 13).

IMPORTANT SEMANTIC DISCLAIMER (must be preserved in any UI-facing
text that surfaces these numbers): higher facial-expression synchrony
does NOT mean agreement, happiness, team health, or psychological
compatibility. It only indicates greater similarity among observed
facial-expression patterns in that time window.

Method: for each 20-second window, compare every pair of sufficiently-
observed participants' aggregated (mean) expression-probability
vectors using Euclidean distance, converted to a bounded similarity
score. Never fabricated for a single visible participant or when
evidence is insufficient.
"""

from __future__ import annotations

import math

from video_analytics.model import PairwiseSynchrony, TrackExpressionSummary

MIN_OBSERVATIONS_FOR_SYNCHRONY = 2
"""A track needs at least this many expression observations within a
window before it is considered "sufficiently observed" for synchrony
comparison -- avoids treating a single noisy frame as evidence."""


def _euclidean_distance(vector_a: tuple[float, ...], vector_b: tuple[float, ...]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vector_a, vector_b)))


def compute_window_synchrony(
    track_summaries: tuple[TrackExpressionSummary, ...],
) -> tuple[tuple[PairwiseSynchrony, ...], float | None, int]:
    """Return (pairwise_synchrony, aggregate_synchrony, participant_count).

    ``aggregate_synchrony`` is None unless at least 2 tracks in this
    window meet MIN_OBSERVATIONS_FOR_SYNCHRONY -- synchrony is never
    fabricated for a single visible participant (Part 13).
    """

    eligible = [
        summary
        for summary in track_summaries
        if summary.observation_count >= MIN_OBSERVATIONS_FOR_SYNCHRONY
        and summary.aggregated_probabilities
    ]

    if len(eligible) < 2:
        return (), None, len(eligible)

    pairwise: list[PairwiseSynchrony] = []
    for i in range(len(eligible)):
        for j in range(i + 1, len(eligible)):
            track_a, track_b = eligible[i], eligible[j]
            distance = _euclidean_distance(
                track_a.aggregated_probabilities, track_b.aggregated_probabilities
            )
            similarity = 1.0 / (1.0 + distance)
            pairwise.append(
                PairwiseSynchrony(
                    track_id_a=track_a.track_id,
                    track_id_b=track_b.track_id,
                    similarity=similarity,
                )
            )

    aggregate = sum(p.similarity for p in pairwise) / len(pairwise)
    return tuple(pairwise), aggregate, len(eligible)
