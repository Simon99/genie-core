from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from .detect import get_video_info, detect_scene_changes

FFMPEG_TIMEOUT = 300


def extract_screenshots(
    video_path: str,
    output_dir: str,
    interval: float = 30.0,
    scene_threshold: float = 0.3,
    min_gap: float = 5.0,
) -> list[dict]:
    """Extract screenshots from video using scene detection + timed interval.

    Returns list of {"time": float, "path": str} sorted by time.

    Strategy:
    1. Detect scene changes (frame content jumps)
    2. Fill gaps with timed captures every `interval` seconds
    3. Merge and deduplicate (min_gap between captures)
    """
    video_path = str(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear stale frames from previous runs so failures can't be masked
    # by leftover files.
    for stale in output_dir.glob("frame_*.png"):
        stale.unlink()

    info = get_video_info(video_path)
    duration = info["duration"]

    scene_times = detect_scene_changes(video_path, threshold=scene_threshold)

    timed_times = []
    t = 0.0
    while t < duration:
        timed_times.append(t)
        t += interval

    all_times = sorted(set(scene_times + timed_times))

    merged = []
    for t in all_times:
        if t > duration:
            break
        if not merged or (t - merged[-1]) >= min_gap:
            merged.append(t)

    results = []
    for i, t in enumerate(merged):
        out_file = output_dir / f"frame_{i:05d}.png"
        cmd = [
            "ffmpeg", "-ss", str(t),
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "2",
            str(out_file),
            "-y"
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=FFMPEG_TIMEOUT)
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
            raise RuntimeError(
                "ffmpeg frame extraction failed at t=%.2fs for %s (exit %d). stderr tail:\n%s"
                % (t, video_path, result.returncode, stderr[-2000:])
            )
        if out_file.exists():
            results.append({"time": t, "path": str(out_file)})

    return results


def _escape_filter_path(path: str) -> str:
    """Escape a filesystem path for use as a drawtext option value.

    Backslashes, colons and single quotes are special in ffmpeg's filter
    option syntax.
    """
    return path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def burn_subtitle(
    video_path: str,
    time: float,
    subtitle_text: str,
    output_file: str,
    font_path: str = "/System/Library/Fonts/PingFang.ttc",
) -> bool:
    """Extract a frame at `time` with subtitle text burned in.

    Supports multi-line subtitles (split by \\n). Each line gets its own
    drawtext filter stacked vertically from the bottom.

    Text is passed via drawtext's textfile= option (a temp file per line),
    which avoids all quote/%{} expansion issues of inline text=.
    """
    info = get_video_info(video_path)
    height = info["height"]
    font_size = int(height / 20)
    line_height = int(font_size * 1.4)

    lines = subtitle_text.split("\n")

    tmp_files = []
    try:
        # Stack lines from bottom: last line at 85% height, previous lines above
        filters = []
        for li, line in enumerate(lines):
            fd, tmp_txt = tempfile.mkstemp(suffix=".txt")
            tmp_files.append(tmp_txt)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(line)

            y_pos = int(height * 0.85) - (len(lines) - 1 - li) * line_height
            filters.append(
                "drawtext=fontfile=%s"
                ":fontsize=%d"
                ":fontcolor=yellow"
                ":box=1:boxcolor=black@0.5:boxborderw=5"
                ":x=(w-tw)/2:y=%d"
                ":textfile=%s" % (
                    _escape_filter_path(font_path), font_size, y_pos,
                    _escape_filter_path(tmp_txt),
                )
            )

        vf = ",".join(filters)

        cmd = [
            "ffmpeg", "-ss", str(time),
            "-i", video_path,
            "-vf", vf,
            "-vframes", "1",
            "-q:v", "2",
            output_file,
            "-y"
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=FFMPEG_TIMEOUT)
        return result.returncode == 0
    finally:
        for tmp_txt in tmp_files:
            Path(tmp_txt).unlink(missing_ok=True)
