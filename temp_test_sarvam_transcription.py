"""Temporary local smoke test for Sarvam transcription.

Usage:
    python temp_test_sarvam_transcription.py path/to/sample.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def format_timestamp(seconds: float | None) -> str:
    if seconds is None:
        return ""

    total_milliseconds = round(seconds * 1000)
    minutes, remainder = divmod(total_milliseconds, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def format_speaker_label(speaker_id: str | None) -> str:
    if speaker_id is None:
        return "Speaker"

    try:
        return f"Speaker {int(speaker_id) + 1}"
    except ValueError:
        return f"Speaker {speaker_id}"


def format_transcription_result(result: object) -> str:
    segments = getattr(result, "segments", [])
    if not segments:
        return str(getattr(result, "transcript", "")).strip()

    blocks: list[str] = []
    for segment in segments:
        label = format_speaker_label(getattr(segment, "speaker_id", None))
        start_time = format_timestamp(getattr(segment, "start_time_seconds", None))
        end_time = format_timestamp(getattr(segment, "end_time_seconds", None))
        time_range = f" [{start_time} - {end_time}]" if start_time or end_time else ""
        transcript = str(getattr(segment, "transcript", "")).strip()

        blocks.append(f"{label}{time_range}:\n{transcript}")

    return "\n\n".join(blocks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe one local sample audio file with Sarvam diarization."
    )
    parser.add_argument(
        "audio_file",
        type=Path,
        help="Path to a sample audio file (.wav, .mp3, .mp4, .m4a, or .aac).",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Expected number of speakers, 1-10. Omit to let Sarvam auto-detect.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from config.settings import SettingsError
    from transcription.audio_utils import AudioProcessingError, preprocess_uploaded_audio
    from transcription.sarvam_client import SarvamTranscriptionClient, TranscriptionError

    original_path = args.audio_file
    prepared_path: Path | None = None

    try:
        prepared_path = preprocess_uploaded_audio(original_path)
        client = SarvamTranscriptionClient()
        result = client.transcribe_detailed(
            prepared_path,
            with_diarization=True,
            num_speakers=args.num_speakers,
        )
    except (AudioProcessingError, SettingsError, TranscriptionError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if prepared_path is not None and prepared_path != original_path:
            prepared_path.unlink(missing_ok=True)

    print(format_transcription_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
