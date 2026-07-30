from transcription.audio_transcript_repair import (
    repair_audio_text,
    repair_audio_transcription,
)
from transcription.sarvam_client import TranscriptionResult, TranscriptionSegment


def _segment(
    text: str,
    speaker: str | None = "0",
    start: float | None = 0.0,
    end: float | None = 1.0,
) -> TranscriptionSegment:
    return TranscriptionSegment(
        transcript=text,
        speaker_id=speaker,
        start_time_seconds=start,
        end_time_seconds=end,
    )


def test_removes_unpunctuated_adjacent_word_and_phrase_repetitions() -> None:
    assert repair_audio_text("How how do we use it?") == "How do we use it?"
    assert repair_audio_text("Is is it ready?") == "Is it ready?"
    assert repair_audio_text("how do we how do we deploy it") == "how do we deploy it"
    assert repair_audio_text("It is very, very useful.") == "It is very, very useful."


def test_restores_only_repeated_interrogative_clause_boundaries() -> None:
    repaired = repair_audio_text(
        "How do we use the APIs, how do we integrate them, how do we deploy them?"
    )
    assert repaired == (
        "How do we use the APIs? how do we integrate them? how do we deploy them?"
    )
    assert repair_audio_text("We reviewed cost, scope, and timing.") == (
        "We reviewed cost, scope, and timing."
    )


def test_normalizes_only_configured_technical_vocabulary() -> None:
    repaired = repair_audio_text(
        "The speaker diaries output feeds mini lm before ollama and gemma."
    )
    assert repaired == (
        "The speaker diarization output feeds MiniLM before Ollama and Gemma."
    )
    assert repair_audio_text(
        "Use product alpha.",
        technical_vocabulary={"product alpha": "Product Alpha"},
    ) == "Use Product Alpha."


def test_preserves_timestamps_order_and_unmapped_speakers() -> None:
    first = _segment("How how do we proceed?", "0", 4.2, 6.8)
    second = _segment("Use speaker diaries.", "1", 7.0, 8.5)
    result = TranscriptionResult(
        transcript="How how do we proceed? Use speaker diaries.",
        segments=[first, second],
    )

    repaired = repair_audio_transcription(result)

    assert [segment.transcript for segment in repaired.segments] == [
        "How do we proceed?",
        "Use speaker diarization.",
    ]
    assert [segment.speaker_id for segment in repaired.segments] == ["0", "1"]
    assert [
        (segment.start_time_seconds, segment.end_time_seconds)
        for segment in repaired.segments
    ] == [(4.2, 6.8), (7.0, 8.5)]
    assert result.segments == [first, second]


def test_uses_only_explicit_confirmed_speaker_mapping() -> None:
    result = TranscriptionResult(
        transcript="Sumit will present.",
        segments=[_segment("Sumit will present.", "0")],
    )

    unchanged = repair_audio_transcription(result)
    mapped = repair_audio_transcription(
        result,
        confirmed_speaker_mapping={"Speaker 1": "Sameer"},
    )

    assert unchanged.segments[0].speaker_id == "0"
    assert unchanged.segments[0].transcript == "Sumit will present."
    assert mapped.segments[0].speaker_id == "Sameer"
    # The spoken name is not guessed from the speaker-label mapping.
    assert mapped.segments[0].transcript == "Sumit will present."
