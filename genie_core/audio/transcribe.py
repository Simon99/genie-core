from __future__ import annotations

import logging
import os
import subprocess
import json
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)

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

VALID_BACKENDS = ("auto", "mlx", "openai", "apple", "groq")

# Groq cloud whisper (whisper-large-v3). Free tier: 25 MB per request,
# so audio is re-encoded to 64 kbps mono mp3 and split when still too big.
GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_MODEL = "whisper-large-v3"
GROQ_MAX_BYTES = 24 * 1024 * 1024
GROQ_CHUNK_SECONDS = 1500          # ~25 min at 64 kbps ≈ 12 MB
GROQ_TIMEOUT = 900
ENV_FILE = Path.home() / ".env"

# Groq documents a free-tier daily audio cap (28800 s), but it is not
# returned in any header and — measured on 2026-07-08 — is not enforced
# for this account (40k+ audio-seconds served without a 429). Treat it as
# a reference number only; the authoritative signal is a 429 response.
GROQ_DOC_DAILY_AUDIO_SECONDS = 28800
GROQ_USAGE_FILE = Path.home() / ".genie" / "groq_usage.json"


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
    initial_prompt: str | None = None,
    groq_fallback: bool = True,
) -> list[dict]:
    """Transcribe audio/video to timestamped segments.

    backend: "auto" (try mlx-whisper → openai-whisper),
             "mlx" (mlx-whisper, Apple Silicon GPU),
             "openai" (openai-whisper CLI),
             "apple" (macOS Speech Framework via local proxy),
             "groq" (cloud whisper-large-v3; needs GROQ_API_KEY,
                     audio leaves the machine)

    initial_prompt: optional hotword/context string that biases whisper
    decoding toward domain terms (ignored by the apple backend).

    groq_fallback: when the groq backend hits rate limits, quota, server
    errors or network failures, transparently redo the transcription with
    the local backend (cloud -> local is the privacy-safe direction).
    Auth/input errors always raise: a fallback would only hide them.

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
        return _transcribe_mlx(input_path, language, model, output_srt, initial_prompt)
    elif backend == "apple":
        return _transcribe_apple(input_path, language, output_srt)
    elif backend == "groq":
        try:
            return _transcribe_groq(input_path, language, output_srt, initial_prompt)
        except GroqUnavailable as e:
            if not groq_fallback:
                raise
            logger.warning("Groq unavailable (%s) — falling back to local %s",
                           e, _detect_backend())
            return transcribe_audio(input_path, language=language, model=model,
                                    output_srt=output_srt, backend="auto",
                                    initial_prompt=initial_prompt)
    else:
        return _transcribe_openai(input_path, language, model, output_srt, initial_prompt)


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


def _transcribe_mlx(input_path: str, language: str, model: str,
                    output_srt: str = None, initial_prompt: str = None) -> list[dict]:
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
        kwargs = {"path_or_hf_repo": model_name, "language": language}
        if initial_prompt:
            kwargs["initial_prompt"] = initial_prompt
        result = mlx_whisper.transcribe(audio_path, **kwargs)

        segments = _filter_segments(result.get("segments", []))

        if output_srt:
            _write_srt(segments, output_srt)

        return segments
    finally:
        if tmp_wav:
            Path(tmp_wav).unlink(missing_ok=True)


def _transcribe_openai(input_path: str, language: str, model: str,
                       output_srt: str = None, initial_prompt: str = None) -> list[dict]:
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
        if initial_prompt:
            cmd.extend(["--initial_prompt", initial_prompt])
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


def _filter_segments(raw: list, offset: float = 0.0) -> list[dict]:
    """Drop hallucinated segments (openai-whisper conventions).

    Quiet/noise chunks — especially through meeting-codec AGC — produce
    fluent looping text; such segments carry a high no_speech_prob (with
    low avg_logprob) or an extreme compression_ratio from the repetition.
    """
    segments = []
    for seg in raw:
        if (seg.get("no_speech_prob", 0.0) > 0.6
                and seg.get("avg_logprob", 0.0) < -1.0):
            continue
        if seg.get("compression_ratio", 0.0) > 2.4:
            continue
        segments.append({
            "start": seg["start"] + offset,
            "end": seg["end"] + offset,
            "text": seg["text"].strip(),
        })
    return segments


def read_env_value(key: str, env_file=None) -> str | None:
    """Read KEY=value from a dotenv-style file (default ~/.env)."""
    path = Path(env_file or ENV_FILE)
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith(key + "="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def write_env_value(key: str, value: str, env_file=None):
    """Set KEY=value in a dotenv-style file, replacing any existing line."""
    path = Path(env_file or ENV_FILE)
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    out, replaced = [], False
    for line in lines:
        if line.strip().startswith(key + "="):
            out.append("%s=%s" % (key, value))
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append("%s=%s" % (key, value))
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def groq_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY") or read_env_value("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY not set (env var or ~/.env). Get one at "
            "https://console.groq.com/keys")
    return key


def verify_groq_key(key: str) -> tuple:
    """(ok, message) — cheap authenticated call against the models endpoint."""
    import requests
    try:
        r = requests.get("https://api.groq.com/openai/v1/models",
                         headers={"Authorization": "Bearer " + key}, timeout=15)
    except Exception as e:
        return False, "無法連線 Groq:%s" % e
    if r.status_code == 200:
        return True, "金鑰有效"
    if r.status_code in (401, 403):
        return False, "金鑰無效或權限不足(HTTP %d)" % r.status_code
    return False, "Groq 回應 HTTP %d" % r.status_code


def _encode_for_groq(input_path: str, out_path: str):
    """64 kbps mono mp3 — 25 min ≈ 12 MB, well under the 25 MB request cap."""
    cmd = ["ffmpeg", "-i", input_path, "-vn", "-ar", "16000", "-ac", "1",
           "-b:a", "64k", out_path, "-y"]
    _run_subprocess(cmd, timeout=FFMPEG_TIMEOUT,
                    install_hint="brew install ffmpeg (or apt install ffmpeg)")


def _split_audio(src: str, out_dir: str, seconds: int) -> list[str]:
    pattern = str(Path(out_dir) / "part_%03d.mp3")
    cmd = ["ffmpeg", "-i", src, "-f", "segment", "-segment_time", str(seconds),
           "-c", "copy", pattern, "-y"]
    _run_subprocess(cmd, timeout=FFMPEG_TIMEOUT,
                    install_hint="brew install ffmpeg (or apt install ffmpeg)")
    return sorted(str(p) for p in Path(out_dir).glob("part_*.mp3"))


class GroqUnavailable(RuntimeError):
    """Groq could not serve the request, but the request itself is fine.

    Rate limits, quota exhaustion, server errors, network failures — the
    caller may retry later or fall back to a local backend. Contrast with
    plain RuntimeError for auth/input errors, which a fallback would only
    paper over.
    """


def _groq_retry_after(resp) -> float:
    try:
        return min(float(resp.headers.get("retry-after", "")), 120.0)
    except (TypeError, ValueError):
        return 0.0


def _groq_daily_quota_hit(resp) -> bool:
    """Distinguish per-minute throttling from the daily audio-seconds cap."""
    body = (resp.text or "").lower()
    if "audio_seconds per day" in body or "audio-seconds per day" in body:
        return True
    if "requests per day" in body or "rpd" in body:
        return True
    # A retry-after beyond a few minutes can only be a daily window.
    try:
        return float(resp.headers.get("retry-after", "0")) > 300
    except ValueError:
        return False


def _groq_one(audio_path: str, language: str, prompt: str, key: str,
              retries: int = 1) -> dict:
    import requests

    def post():
        with open(audio_path, "rb") as fh:
            data = {"model": GROQ_MODEL, "response_format": "verbose_json"}
            if language:
                data["language"] = language
            if prompt:
                data["prompt"] = prompt[:1000]   # Groq caps the prompt at ~224 tokens
            return requests.post(GROQ_URL, headers={"Authorization": "Bearer " + key},
                                 files={"file": fh}, data=data, timeout=GROQ_TIMEOUT)

    for attempt in range(retries + 1):
        try:
            r = post()
        except Exception as e:                       # network / DNS / timeout
            if attempt < retries:
                time.sleep(5)
                continue
            raise GroqUnavailable("Groq unreachable: %s" % e)

        if r.status_code == 200:
            body = r.json()
            body["_headers"] = dict(r.headers)
            return body

        if r.status_code == 429:
            if _groq_daily_quota_hit(r):
                raise GroqUnavailable("Groq daily quota exhausted: %s" % r.text[:200])
            if attempt < retries:
                time.sleep(_groq_retry_after(r) or 60.0)
                continue
            raise GroqUnavailable("Groq rate limited: %s" % r.text[:200])

        if r.status_code >= 500:
            if attempt < retries:
                time.sleep(_groq_retry_after(r) or 10.0)
                continue
            raise GroqUnavailable("Groq server error (HTTP %d): %s"
                                  % (r.status_code, r.text[:200]))

        # 4xx other than 429: bad key, bad file, unsupported params — a
        # local fallback would only hide the real problem.
        raise RuntimeError("Groq transcription failed (HTTP %d): %s"
                           % (r.status_code, r.text[:300]))
    raise GroqUnavailable("Groq: retries exhausted")


def _read_usage_file() -> dict:
    try:
        return json.loads(GROQ_USAGE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def groq_usage_today() -> dict:
    """Current Groq budget, mixing authoritative and estimated numbers.

    Groq's whisper responses carry only request-count rate-limit headers
    (x-ratelimit-{limit,remaining,reset}-requests) — verified against the
    live API; there is no audio-seconds header and no usage endpoint. So:

    - requests_*   : reported by Groq on the last response (authoritative,
                     as of ``updated_at``); None until the first call.
    - audio_seconds: local tally of what this machine sent today; other
                     machines sharing the key are invisible to it. No
                     remaining-audio figure is derived from it — the
                     documented daily cap is not enforced in practice
                     (see GROQ_DOC_DAILY_AUDIO_SECONDS).
    """
    today = time.strftime("%Y-%m-%d")
    data = _read_usage_file()
    fresh = data.get("date") == today
    used = float(data.get("audio_seconds", 0)) if fresh else 0.0
    return {
        "audio_seconds": round(used),                           # local tally
        "doc_daily_audio_seconds": GROQ_DOC_DAILY_AUDIO_SECONDS,  # reference only
        "requests_remaining": data.get("requests_remaining"),   # from Groq
        "requests_limit": data.get("requests_limit"),
        "requests_reset": data.get("requests_reset"),
        "updated_at": data.get("updated_at"),
    }


def _record_groq_usage(audio_seconds: float, headers=None):
    """Append to the local audio tally; overwrite request counters from Groq."""
    try:
        today = time.strftime("%Y-%m-%d")
        data = _read_usage_file()
        if data.get("date") != today:
            data = {"date": today, "audio_seconds": 0.0}
        data["audio_seconds"] = float(data.get("audio_seconds", 0)) + audio_seconds
        h = headers or {}
        remaining = h.get("x-ratelimit-remaining-requests")
        if remaining is not None:
            data["requests_remaining"] = int(remaining)
            data["requests_limit"] = int(h.get("x-ratelimit-limit-requests") or 0) or None
            data["requests_reset"] = h.get("x-ratelimit-reset-requests")
            data["updated_at"] = time.strftime("%H:%M:%S")
        GROQ_USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        GROQ_USAGE_FILE.write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        logger.exception("failed to update Groq usage ledger")


def _transcribe_groq(input_path: str, language: str, output_srt: str = None,
                     initial_prompt: str = None) -> list[dict]:
    """Cloud transcription via Groq whisper-large-v3.

    More accurate than local medium on noisy audio and ~50x realtime, but
    the audio leaves the machine — pick the backend accordingly.
    """
    key = groq_api_key()
    # Traditional-Chinese hint: Groq's zh output otherwise skews Simplified.
    prompt = initial_prompt or ""
    if language == "zh" and "繁體" not in prompt:
        prompt = ("以下是繁體中文的會議或課程逐字稿。" + prompt).strip()

    with tempfile.TemporaryDirectory() as tmpdir:
        mp3 = str(Path(tmpdir) / "audio.mp3")
        _encode_for_groq(input_path, mp3)

        parts = [mp3]
        if Path(mp3).stat().st_size > GROQ_MAX_BYTES:
            parts = _split_audio(mp3, tmpdir, GROQ_CHUNK_SECONDS)
            if not parts:
                raise RuntimeError("Groq: failed to split oversized audio")

        segments, offset = [], 0.0
        for part in parts:
            data = _groq_one(part, language, prompt, key)
            part_seconds = float(data.get("duration") or 0.0)
            _record_groq_usage(part_seconds, data.pop("_headers", None))
            segments.extend(_filter_segments(data.get("segments", []), offset))
            offset += part_seconds

    if output_srt:
        _write_srt(segments, output_srt)
    return segments


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
