import subprocess
import json
import tempfile
from pathlib import Path


def transcribe_audio(
    input_path: str,
    language: str = "zh-Hans",
    output_srt: str | None = None,
) -> list[dict]:
    """Transcribe audio/video to timestamped segments using macOS Speech Framework.

    Calls the Swift CLI helper (genie-speech-cli) which wraps SFSpeechRecognizer.

    Returns list of {"start": float, "end": float, "text": str}.
    If output_srt is provided, also writes an SRT file.
    """
    input_path = str(input_path)

    audio_path = _extract_audio(input_path)

    try:
        cmd = [
            "genie-speech-cli",
            "--input", audio_path,
            "--language", language,
            "--format", "json",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        segments = json.loads(result.stdout)
    finally:
        if audio_path != input_path:
            Path(audio_path).unlink(missing_ok=True)

    if output_srt:
        _write_srt(segments, output_srt)

    return segments


def _extract_audio(input_path: str) -> str:
    """Extract audio from video file to WAV for speech recognition."""
    if input_path.endswith((".wav", ".m4a", ".aac")):
        return input_path

    output = tempfile.mktemp(suffix=".wav")
    cmd = [
        "ffmpeg", "-i", input_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        output, "-y"
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output


def _write_srt(segments: list[dict], output_path: str):
    """Write segments to SRT format."""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start = _format_srt_time(seg["start"])
            end = _format_srt_time(seg["end"])
            f.write(f"{i}\n{start} --> {end}\n{seg['text']}\n\n")


def _format_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
