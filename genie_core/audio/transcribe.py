from __future__ import annotations

import os
import subprocess
import json
import tempfile
from pathlib import Path

# Audio file extensions that can be fed to the backends directly
# (anything else is first extracted to wav via ffmpeg).
AUDIO_EXTENSIONS = (".wav", ".m4a", ".aac", ".mp3", ".flac")

# whisper-style ISO 639-1 codes → BCP-47 locales for the Apple Speech backend.
# Unknown codes are passed through unchanged.
WHISPER_TO_BCP47 = {
    "zh": "zh-Hans",
    "en": "en-US",
    "ja": "ja-JP",
    "ko": "ko-KR",
    "de": "de-DE",
    "fr": "fr-FR",
    "es": "es-ES",
    "it": "it-IT",
    "pt": "pt-BR",
    "ru": "ru-RU",
    "nl": "nl-NL",
    "ar": "ar-SA",
    "hi": "hi-IN",
    "th": "th-TH",
    "vi": "vi-VN",
    "id": "id-ID",
    "tr": "tr-TR",
    "pl": "pl-PL",
    "sv": "sv-SE",
    "da": "da-DK",
    "fi": "fi-FI",
    "nb": "nb-NO",
    "no": "nb-NO",
    "uk": "uk-UA",
    "cs": "cs-CZ",
    "el": "el-GR",
    "he": "he-IL",
    "hu": "hu-HU",
    "ro": "ro-RO",
    "sk": "sk-SK",
    "ms": "ms-MY",
    "ca": "ca-ES",
    "hr": "hr-HR",
    "yue": "zh-Hant",
}

# Generous timeout for whisper-class transcription subprocesses.
WHISPER_TIMEOUT = 3600
FFMPEG_TIMEOUT = 600

VALID_BACKENDS = ("auto", "mlx", "openai", "apple")


def _is_audio_file(path: str) -> bool:
    return Path(path).suffix.lower() in AUDIO_EXTENSIONS


def to_bcp47(language: str) -> str:
    """Map a whisper-style ISO language code to a BCP-47 locale.

    Unknown codes (including already-BCP-47 strings like "zh-Hans") are
    returned unchanged.
    """
    return WHISPER_TO_BCP47.get(language, language)


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

    if backend not in VALID_BACKENDS:
        raise ValueError(
            "Unknown backend %r, expected one of %s" % (backend, ", ".join(VALID_BACKENDS))
        )

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
        import mlx_whisper  # noqa: F401
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

    if not _is_audio_file(input_path):
        fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
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
        if not _is_audio_file(input_path):
            audio_path = str(Path(tmpdir) / "audio.wav")
            _extract_audio_to(input_path, audio_path)

        cmd = [
            "whisper", audio_path,
            "--model", model,
            "--language", language,
            "--output_format", "json",
            "--output_dir", tmpdir,
        ]
        _run_subprocess(cmd, timeout=WHISPER_TIMEOUT,
                        install_hint="pip install openai-whisper")

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
    Accepts whisper-style language codes ("zh", "en", ...) and maps them to
    BCP-47 locales the Speech framework understands.
    """
    import requests

    audio_path = input_path
    tmp_wav = None

    if not _is_audio_file(input_path):
        fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        _extract_audio_to(input_path, tmp_wav)
        audio_path = tmp_wav

    try:
        resp = requests.post(
            "http://localhost:5300/transcribe",
            json={"path": str(Path(audio_path).resolve()), "language": to_bcp47(language)},
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


def _run_subprocess(cmd: list[str], timeout: int, install_hint: str = None):
    """Run a subprocess with timeout, surfacing stderr on failure."""
    try:
        return subprocess.run(cmd, capture_output=True, check=True, timeout=timeout)
    except FileNotFoundError:
        msg = "Command not found: %s." % cmd[0]
        if install_hint:
            msg += " Install it with: %s" % install_hint
        raise RuntimeError(msg)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
        raise subprocess.CalledProcessError(
            e.returncode, e.cmd, output=e.output,
            stderr="%s failed (exit %d). stderr tail:\n%s"
                   % (cmd[0], e.returncode, stderr[-2000:]),
        ) from e


def _extract_audio_to(input_path: str, output_path: str):
    cmd = [
        "ffmpeg", "-i", input_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        output_path, "-y"
    ]
    _run_subprocess(cmd, timeout=FFMPEG_TIMEOUT,
                    install_hint="brew install ffmpeg (or apt install ffmpeg)")


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
