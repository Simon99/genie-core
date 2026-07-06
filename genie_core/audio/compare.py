from __future__ import annotations

import json
import subprocess
import tempfile
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def dual_transcribe(
    input_path: str,
    output_dir: str,
    language: str = "zh",
    whisper_model: str = "medium",
    apple_language: str = "zh-Hans",
    progress_callback=None,
) -> dict:
    """Run mlx-whisper and Apple Speech in parallel, then produce a comparison.

    Uses whisper's segments as the reference timeline, aligning Apple Speech
    output against each whisper chunk for side-by-side comparison.

    Returns {"whisper": path, "apple": path, "comparison": path, "stats": dict}
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Extract audio once
    audio_path = input_path
    tmp_wav = None
    if not input_path.endswith((".wav", ".m4a", ".aac", ".mp3", ".flac")):
        tmp_wav = tempfile.mktemp(suffix=".wav")
        subprocess.run([
            "ffmpeg", "-i", input_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            tmp_wav, "-y"
        ], capture_output=True, check=True)
        audio_path = tmp_wav

    if progress_callback:
        progress_callback("transcribing", 0)

    # Run both in parallel
    with ThreadPoolExecutor(max_workers=2) as pool:
        whisper_future = pool.submit(_run_whisper, audio_path, language, whisper_model)
        apple_future = pool.submit(_run_apple, audio_path, apple_language)

        whisper_result = whisper_future.result()
        if progress_callback:
            progress_callback("whisper_done", 0.5)

        apple_result = apple_future.result()
        if progress_callback:
            progress_callback("apple_done", 0.7)

    if tmp_wav:
        Path(tmp_wav).unlink(missing_ok=True)

    # Save raw outputs
    whisper_path = out / "whisper.json"
    whisper_path.write_text(json.dumps(whisper_result, ensure_ascii=False, indent=2), encoding="utf-8")

    apple_path = out / "apple.json"
    apple_path.write_text(json.dumps(apple_result, ensure_ascii=False, indent=2), encoding="utf-8")

    # Align and compare
    if progress_callback:
        progress_callback("comparing", 0.8)

    comparison = _align_and_compare(whisper_result, apple_result)

    comp_path = out / "comparison.md"
    comp_path.write_text(comparison["markdown"], encoding="utf-8")

    comp_json_path = out / "comparison.json"
    comp_json_path.write_text(json.dumps(comparison["data"], ensure_ascii=False, indent=2), encoding="utf-8")

    if progress_callback:
        progress_callback("done", 1.0)

    return {
        "whisper": str(whisper_path),
        "apple": str(apple_path),
        "comparison": str(comp_path),
        "stats": comparison["stats"],
    }


def _run_whisper(audio_path: str, language: str, model: str) -> list[dict]:
    try:
        import mlx_whisper
        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo="mlx-community/whisper-%s-mlx" % model,
            language=language,
        )
        return [{"start": s["start"], "end": s["end"], "text": s["text"].strip()}
                for s in result.get("segments", [])]
    except ImportError:
        from .transcribe import _transcribe_openai
        return _transcribe_openai(audio_path, language, model)


def _run_apple(audio_path: str, language: str) -> list[dict]:
    try:
        import requests
        resp = requests.post(
            "http://localhost:5300/transcribe",
            json={"path": str(Path(audio_path).resolve()), "language": language},
            timeout=600,
        )
        if resp.status_code == 200:
            return resp.json().get("segments", [])
    except Exception:
        pass
    return []


def _align_and_compare(whisper_segs: list[dict], apple_segs: list[dict]) -> dict:
    """Align Apple Speech output to whisper's timeline for comparison."""

    # Build Apple Speech full text with rough timestamps
    apple_text_map = []
    for seg in apple_segs:
        apple_text_map.append({
            "start": seg.get("start", 0),
            "end": seg.get("end", 0),
            "text": seg.get("text", ""),
        })

    aligned = []
    for ws in whisper_segs:
        # Find overlapping Apple segments
        apple_match = ""
        for ap in apple_text_map:
            if ap["end"] < ws["start"] or ap["start"] > ws["end"]:
                continue
            apple_match += ap["text"]

        aligned.append({
            "start": ws["start"],
            "end": ws["end"],
            "whisper": ws["text"],
            "apple": apple_match.strip() if apple_match else "(no match)",
        })

    # Stats
    whisper_total = sum(len(s["text"]) for s in whisper_segs)
    apple_total = sum(len(s.get("text", "")) for s in apple_segs)
    matched = sum(1 for a in aligned if a["apple"] != "(no match)")

    stats = {
        "whisper_segments": len(whisper_segs),
        "apple_segments": len(apple_segs),
        "whisper_chars": whisper_total,
        "apple_chars": apple_total,
        "matched_segments": matched,
        "match_rate": "%.1f%%" % (matched / max(len(aligned), 1) * 100),
    }

    # Generate markdown
    lines = ["# Whisper vs Apple Speech Comparison\n"]
    lines.append("| Metric | Whisper | Apple Speech |")
    lines.append("|---|---|---|")
    lines.append("| Segments | %d | %d |" % (len(whisper_segs), len(apple_segs)))
    lines.append("| Total chars | %d | %d |" % (whisper_total, apple_total))
    lines.append("| Match rate | — | %s |" % stats["match_rate"])
    lines.append("")
    lines.append("## Side-by-Side\n")
    lines.append("| Time | Whisper | Apple Speech |")
    lines.append("|---|---|---|")

    for a in aligned:
        t = "%02d:%02d" % (int(a["start"] // 60), int(a["start"] % 60))
        w = a["whisper"][:60].replace("|", "\\|")
        ap = a["apple"][:60].replace("|", "\\|")
        lines.append("| %s | %s | %s |" % (t, w, ap))

    return {
        "data": aligned,
        "stats": stats,
        "markdown": "\n".join(lines),
    }
