"""Tests for Phase 3.3's shared uncertainty-aware presentation helper:
``meeting_analytics.communication.interpret_emotion_distribution``.

This is a pure, deterministic PRESENTATION judgment over an already-
computed Voxels probability distribution -- it never touches Voxels
inference/model/checkpoint/classes, and none of these tests exercise or
depend on that model. Style matches tests/test_meeting_analytics_communication.py.

Every distribution here is SYNTHETIC and chosen to test the general rule
(top-1 probability, margin over top-2) across a spread of shapes -- never
the one real validation meeting's actual numbers (which happened to be
Disgust ~18%, Neutral ~17%, ...; case "close_top_two_low_probability"
below is deliberately similarly-SHAPED -- close top-two, low absolute
probability -- to prove the general rule handles that pattern correctly,
without using the real percentages).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meeting_analytics.communication import interpret_emotion_distribution


# --- 1-2: strong dominant classes ------------------------------------------------


def test_strong_dominant_happy_is_clear() -> None:
    result = interpret_emotion_distribution(
        {"happy": 0.70, "sad": 0.10, "angry": 0.05, "neutral": 0.05, "fear": 0.04, "disgust": 0.03, "surprise": 0.03}
    )
    assert result.status == "clear"
    assert result.highest_class == "happy"
    assert result.highest_probability == 0.70
    assert "Happy" in result.display_pattern
    assert result.secondary_tone == "Positive"


def test_strong_dominant_neutral_is_clear() -> None:
    result = interpret_emotion_distribution(
        {"neutral": 0.62, "happy": 0.10, "sad": 0.09, "angry": 0.08, "fear": 0.05, "disgust": 0.03, "surprise": 0.03}
    )
    assert result.status == "clear"
    assert result.highest_class == "neutral"
    assert result.secondary_tone == "Neutral"


# --- 3: close top-two classes -----------------------------------------------------


def test_close_top_two_classes_is_mixed() -> None:
    result = interpret_emotion_distribution(
        {"neutral": 0.22, "sad": 0.21, "happy": 0.20, "angry": 0.13, "fear": 0.10, "disgust": 0.08, "surprise": 0.06}
    )
    assert result.status == "mixed"
    assert result.secondary_tone is None
    assert "No clearly dominant" in result.display_pattern
    # Raw evidence is still exposed, never suppressed.
    assert result.highest_class == "neutral"
    assert round(result.highest_probability, 2) == 0.22


# --- 4: near-uniform distribution --------------------------------------------------


def test_near_uniform_distribution_is_mixed() -> None:
    result = interpret_emotion_distribution({label: 1 / 7 for label in ("happy", "sad", "angry", "neutral", "fear", "disgust", "surprise")})
    assert result.status == "mixed"
    assert result.margin == 0.0


# --- 5: low-probability top class (general shape of the real-audio bug, --------
# --- synthetic numbers only) -------------------------------------------------------


def test_close_top_two_low_probability_is_mixed_not_a_confident_conclusion() -> None:
    # Deliberately NOT 18%/17% -- a different, still low-separation shape,
    # to prove the general rule (not a hardcoded real-audio special case).
    result = interpret_emotion_distribution(
        {"sad": 0.19, "angry": 0.18, "disgust": 0.17, "neutral": 0.16, "happy": 0.12, "fear": 0.09, "surprise": 0.09}
    )
    assert result.status == "mixed"
    assert result.secondary_tone is None


# --- 6: malformed / missing probabilities ------------------------------------------


def test_none_probabilities_is_unavailable() -> None:
    result = interpret_emotion_distribution(None)
    assert result.status == "unavailable"
    assert result.highest_class is None
    assert result.display_pattern


def test_empty_probabilities_is_unavailable() -> None:
    result = interpret_emotion_distribution({})
    assert result.status == "unavailable"


def test_non_numeric_probability_values_are_skipped_not_crashing() -> None:
    result = interpret_emotion_distribution({"happy": "not-a-number", "sad": 0.4})
    assert result.status in ("clear", "mixed")
    assert result.highest_class == "sad"


def test_all_non_numeric_probabilities_is_unavailable() -> None:
    result = interpret_emotion_distribution({"happy": "n/a", "sad": None})
    assert result.status == "unavailable"


# --- 7: all seven classes present --------------------------------------------------


def test_all_seven_classes_present_still_works() -> None:
    probabilities = {"angry": 0.05, "disgust": 0.05, "fear": 0.05, "happy": 0.6, "neutral": 0.1, "sad": 0.1, "surprise": 0.05}
    result = interpret_emotion_distribution(probabilities)
    assert result.status == "clear"
    assert result.highest_class == "happy"


# --- 8: subset / missing class keys ------------------------------------------------


def test_single_class_distribution_is_clear_by_construction() -> None:
    result = interpret_emotion_distribution({"happy": 0.9})
    assert result.status == "clear"
    assert result.second_class is None
    assert result.margin is None


def test_two_class_subset_uses_available_classes_only() -> None:
    result = interpret_emotion_distribution({"happy": 0.55, "sad": 0.45})
    assert result.status == "mixed"
    assert result.second_class == "sad"


# --- 9-10: ties and deterministic tie behavior -------------------------------------


def test_exact_tie_between_top_two_is_mixed() -> None:
    result = interpret_emotion_distribution({"happy": 0.5, "sad": 0.5})
    assert result.status == "mixed"
    assert result.margin == 0.0


def test_tie_break_is_deterministic_across_repeated_calls() -> None:
    probabilities = {"sad": 0.3, "angry": 0.3, "happy": 0.2, "neutral": 0.2}
    first = interpret_emotion_distribution(probabilities)
    second = interpret_emotion_distribution(probabilities)
    assert first.highest_class == second.highest_class
    assert first.second_class == second.second_class
    # Deterministic alphabetical tie-break among equal probabilities.
    assert first.highest_class == "angry"


def test_dict_insertion_order_does_not_affect_result() -> None:
    a = interpret_emotion_distribution({"sad": 0.3, "angry": 0.3, "happy": 0.2, "neutral": 0.2})
    b = interpret_emotion_distribution({"neutral": 0.2, "happy": 0.2, "angry": 0.3, "sad": 0.3})
    assert a.highest_class == b.highest_class
    assert a.second_class == b.second_class


# --- Borderline threshold cases (documenting the exact rule) -----------------------


def test_top_probability_exactly_at_threshold_with_sufficient_margin_is_clear() -> None:
    result = interpret_emotion_distribution({"happy": 0.40, "sad": 0.20, "neutral": 0.15, "angry": 0.10, "fear": 0.05, "disgust": 0.05, "surprise": 0.05})
    assert result.status == "clear"


def test_top_probability_below_threshold_is_mixed_even_with_large_margin() -> None:
    # top1=0.39 (just below the 0.40 floor) with a huge margin over top2 --
    # margin alone is not sufficient; the leading class must ALSO represent
    # a plausible plurality of the evidence.
    result = interpret_emotion_distribution({"happy": 0.39, "sad": 0.05, "neutral": 0.05, "angry": 0.05, "fear": 0.05, "disgust": 0.05, "surprise": 0.36})
    assert result.status == "mixed"


def test_margin_below_threshold_is_mixed_even_with_high_top_probability() -> None:
    # top1=0.45 clears the probability floor, but margin over top2 (0.40)
    # is only 0.05 -- below the 0.15 margin requirement.
    result = interpret_emotion_distribution({"happy": 0.45, "sad": 0.40, "neutral": 0.05, "angry": 0.04, "fear": 0.03, "disgust": 0.02, "surprise": 0.01})
    assert result.status == "mixed"


# --- Raw data preservation ----------------------------------------------------------


def test_interpretation_never_mutates_input_probabilities() -> None:
    probabilities = {"happy": 0.6, "sad": 0.4}
    original = dict(probabilities)
    interpret_emotion_distribution(probabilities)
    assert probabilities == original


def test_highest_and_second_probability_are_exact_raw_values() -> None:
    result = interpret_emotion_distribution({"disgust": 0.18, "neutral": 0.17, "sad": 0.16})
    assert result.highest_probability == 0.18
    assert result.second_probability == 0.17
    assert result.highest_class == "disgust"
    assert result.second_class == "neutral"
