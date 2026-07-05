import subprocess
import json
import tempfile
from pathlib import Path


def transcribe_audio(
    input_path: str,
    language: str = "zh",
    model: str = "medium",
    output_srt: str | None = None,
) -> list[dict]:
    """Transcribe audio/video using OpenAI Whisper (local).

    Returns list of {"start": float, "end": float, "text": str}.
    If output_srt is provided, also copies the SRT file there.

    For Chinese-English mixed content, use language="zh" — Whisper handles
    code-switching well with the Chinese language hint.
    """
    input_path = str(input_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Whisper can read video files directly but WAV is faster
        audio_path = _extract_audio(input_path, tmpdir)

        cmd = [
            "whisper", audio_path,
            "--model", model,
            "--language", language,
            "--output_format", "json",
            "--output_dir", tmpdir,
        ]
        subprocess.run(cmd, capture_output=True, check=True)

        json_file = Path(tmpdir) / (Path(audio_path).stem + ".json")
        with open(json_file, "r", encoding="utf-8") as f:
            result = json.load(f)

        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
            })

        if output_srt:
            srt_file = Path(tmpdir) / (Path(audio_path).stem + ".srt")
            # Re-run with SRT output if needed
            if not srt_file.exists():
                cmd_srt = [
                    "whisper", audio_path,
                    "--model", model,
                    "--language", language,
                    "--output_format", "srt",
                    "--output_dir", tmpdir,
                ]
                subprocess.run(cmd_srt, capture_output=True, check=True)

            if srt_file.exists():
                Path(output_srt).write_text(srt_file.read_text(encoding="utf-8"), encoding="utf-8")

    return segments


def _extract_audio(input_path: str, output_dir: str) -> str:
    """Extract audio from video file to WAV for faster processing."""
    if input_path.endswith((".wav", ".m4a", ".aac", ".mp3", ".flac")):
        return input_path

    output = str(Path(output_dir) / "audio.wav")
    cmd = [
        "ffmpeg", "-i", input_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        output, "-y"
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output
