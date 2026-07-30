from transcription.audio_transcript_normalization import (
    coalesce_contiguous_audio_segments,
)
from ml_mom.predict_ann import prepare_sentence_features
from transcription.speaker_mapping import default_speaker_label
from transcription.sarvam_client import TranscriptionResult, TranscriptionSegment


def _segment(
    text: str,
    speaker: str | None,
    start: float | None,
    end: float | None,
) -> TranscriptionSegment:
    return TranscriptionSegment(
        transcript=text,
        speaker_id=speaker,
        start_time_seconds=start,
        end_time_seconds=end,
    )


def test_coalesces_only_contiguous_fragments_from_same_speaker() -> None:
    result = TranscriptionResult(
        transcript="We need to finalize the launch plan. Agreed.",
        language_code="en-IN",
        segments=[
            _segment("We need to", "0", 0.0, 1.0),
            _segment("finalize the launch plan.", "0", 1.2, 2.8),
            _segment("Agreed.", "1", 3.0, 3.6),
        ],
    )

    normalized = coalesce_contiguous_audio_segments(result)

    assert [segment.transcript for segment in normalized.segments] == [
        "We need to finalize the launch plan.",
        "Agreed.",
    ]
    assert normalized.segments[0].speaker_id == "0"
    assert normalized.segments[0].start_time_seconds == 0.0
    assert normalized.segments[0].end_time_seconds == 2.8
    assert normalized.transcript == result.transcript


def test_preserves_long_pauses_different_speakers_and_missing_timestamps() -> None:
    segments = [
        _segment("First point.", "0", 0.0, 1.0),
        _segment("Later point.", "0", 4.0, 5.0),
        _segment("Response.", "1", 5.1, 6.0),
        _segment("Untimed note.", "1", None, None),
    ]
    result = TranscriptionResult(
        transcript="First point. Later point. Response. Untimed note.",
        segments=segments,
    )

    normalized = coalesce_contiguous_audio_segments(result)

    assert normalized is result
    assert normalized.segments == segments


def test_does_not_mutate_the_source_result_or_segments() -> None:
    first = _segment("Prepare", "0", 10.0, 11.0)
    second = _segment("the release notes.", "0", 11.2, 12.5)
    result = TranscriptionResult(transcript="Prepare the release notes.", segments=[first, second])

    normalized = coalesce_contiguous_audio_segments(result)

    assert result.segments == [first, second]
    assert normalized is not result
    assert normalized.segments[0] is not first


def test_normalized_audio_matches_coherent_transcript_sentence_inputs() -> None:
    result = TranscriptionResult(
        transcript="Priya will prepare the release notes by Friday.",
        segments=[
            _segment("Priya will", "0", 0.0, 0.9),
            _segment("prepare the release notes by Friday.", "0", 1.0, 2.8),
        ],
    )
    normalized = coalesce_contiguous_audio_segments(result)
    segment = normalized.segments[0]
    audio_review_text = (
        f"{default_speaker_label(segment)} [00:00 - 00:03]\n"
        f"{segment.transcript}"
    )
    gold_transcript_text = (
        "Speaker 1 [00:00 - 00:03]\n"
        "Priya will prepare the release notes by Friday."
    )

    audio_features = prepare_sentence_features(audio_review_text)
    gold_features = prepare_sentence_features(gold_transcript_text)

    assert [feature.original_sentence for feature in audio_features] == [
        "Priya will prepare the release notes by Friday."
    ]
    assert audio_features == gold_features
