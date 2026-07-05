from __future__ import annotations

import subprocess
from pathlib import Path

from .detect import get_video_info, detect_scene_changes


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
        subprocess.run(cmd, capture_output=True)
        if out_file.exists():
            results.append({"time": t, "path": str(out_file)})

    return results


def burn_subtitle(
    video_path: str,
    time: float,
    subtitle_text: str,
    output_file: str,
    font_path: str = "/System/Library/Fonts/PingFang.ttc",
) -> bool:
    """Extract a frame at `time` with subtitle text burned in.

    Uses ffmpeg drawtext filter (same approach as xdite/video2blog).
    """
    info = get_video_info(video_path)
    height = info["height"]
    font_size = int(height / 20)
    y_position = int(height * 0.85)

    escaped_text = subtitle_text.replace("'", "\\'").replace(":", "\\:")

    cmd = [
        "ffmpeg", "-ss", str(time),
        "-i", video_path,
        "-vf", (
            f"drawtext=fontfile={font_path}"
            f":fontsize={font_size}"
            f":fontcolor=yellow"
            f":box=1:boxcolor=black@0.5:boxborderw=5"
            f":x=(w-tw)/2:y={y_position}"
            f":text='{escaped_text}'"
        ),
        "-vframes", "1",
        "-q:v", "2",
        output_file,
        "-y"
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0
