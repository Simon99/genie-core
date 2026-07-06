from __future__ import annotations

import subprocess
import json
import tempfile
from pathlib import Path


def transcribe_audio(
    input_path: str,
    language: str = "zh",
    model: str = "medium",
    output_srt: str | None = None,
    backend: str = "auto",
) -> list[dict]:
    """Transcribe audio/video to timestamped segments.

    backend: "auto" (try mlx-whisper → openai-whisper),
             "mlx" (mlx-whisper, Apple Silicon GPU),
             "openai" (openai-whisper CLI),
             "apple" (macOS Speech Framework via local proxy)

    Returns list of {"start": float, "end": float, "text": str}.
    """
    input_path = str(input_path)

    if backend == "auto":
        backend = _detect_backend()

    if backend == "mlx":
        return _transcribe_mlx(input_path, language, model, output_srt)
    elif backend == "apple":
        return _transcribe_apple(input_path, language, output_srt)
    else:
        return _transcribe_openai(input_path, language, model, output_srt)


def _detect_backend() -> str:
    try:
        import mlx_whisper
        return "mlx"
    except ImportError:
        pass

    # Check if Apple speech proxy is running
    try:
        import requests
        r = requests.get("http://localhost:5300/health", timeout=1)
        if r.status_code == 200:
            return "apple"
    except Exception:
        pass

    return "openai"


def _transcribe_mlx(input_path: str, language: str, model: str, output_srt: str = None) -> list[dict]:
    """Transcribe using mlx-whisper (Apple Silicon GPU accelerated)."""
    import mlx_whisper

    audio_path = input_path
    tmp_wav = None

    if not input_path.endswith((".wav", ".m4a", ".aac", ".mp3", ".flac")):
        tmp_wav = tempfile.mktemp(suffix=".wav")
        _extract_audio_to(input_path, tmp_wav)
        audio_path = tmp_wav

    try:
        model_name = "mlx-community/whisper-%s-mlx" % model
        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=model_name,
            language=language,
        )

        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
            })

        if output_srt:
            _write_srt(segments, output_srt)

        return segments
    finally:
        if tmp_wav:
            Path(tmp_wav).unlink(missing_ok=True)


def _transcribe_openai(input_path: str, language: str, model: str, output_srt: str = None) -> list[dict]:
    """Transcribe using openai-whisper CLI."""
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = input_path
        if not input_path.endswith((".wav", ".m4a", ".aac", ".mp3", ".flac")):
            audio_path = str(Path(tmpdir) / "audio.wav")
            _extract_audio_to(input_path, audio_path)

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
            _write_srt(segments, output_srt)

    return segments


def _transcribe_apple(input_path: str, language: str, output_srt: str = None) -> list[dict]:
    """Transcribe using macOS Speech Framework via local HTTP proxy.

    The proxy runs in the GUI session and has TCC speech recognition permission.
    """
    import requests

    audio_path = input_path
    tmp_wav = None

    if not input_path.endswith((".wav", ".m4a", ".aac", ".mp3", ".flac")):
        tmp_wav = tempfile.mktemp(suffix=".wav")
        _extract_audio_to(input_path, tmp_wav)
        audio_path = tmp_wav

    try:
        resp = requests.post(
            "http://localhost:5300/transcribe",
            json={"path": str(Path(audio_path).resolve()), "language": language},
            timeout=600,
        )
        resp.raise_for_status()
        segments = resp.json().get("segments", [])

        if output_srt:
            _write_srt(segments, output_srt)

        return segments
    finally:
        if tmp_wav:
            Path(tmp_wav).unlink(missing_ok=True)


def _extract_audio_to(input_path: str, output_path: str):
    cmd = [
        "ffmpeg", "-i", input_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        output_path, "-y"
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def _write_srt(segments: list[dict], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start = _fmt_srt(seg["start"])
            end = _fmt_srt(seg["end"])
            f.write("%d\n%s --> %s\n%s\n\n" % (i, start, end, seg["text"]))


def _fmt_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return "%02d:%02d:%02d,%03d" % (h, m, s, ms)
