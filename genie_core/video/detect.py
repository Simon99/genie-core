from __future__ import annotations

import subprocess
import json
from pathlib import Path


def get_video_info(video_path: str) -> dict:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,sample_aspect_ratio,duration,r_frame_rate",
        "-show_entries", "format=duration",
        "-of", "json", video_path
    ]
    output = subprocess.check_output(cmd).decode("utf-8")
    info = json.loads(output)

    stream = info["streams"][0]
    width = stream["width"]
    height = stream["height"]
    sar = stream.get("sample_aspect_ratio", "1:1")
    if sar and ":" in sar:
        sar_parts = sar.split(":")
        sar_ratio = float(sar_parts[0]) / float(sar_parts[1])
    else:
        sar_ratio = 1.0

    display_width = int(width * sar_ratio)

    duration = float(info.get("format", {}).get("duration", 0))
    if not duration:
        duration = float(stream.get("duration", 0))

    return {
        "width": display_width,
        "height": height,
        "duration": duration,
    }


def detect_scene_changes(video_path: str, threshold: float = 0.3) -> list[float]:
    """Detect scene changes using ffmpeg's select filter.

    Returns a list of timestamps (seconds) where significant frame changes occur.
    threshold: 0.0-1.0, lower = more sensitive (more cuts detected).
    """
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-vsync", "vfr",
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    timestamps = []
    for line in result.stderr.split("\n"):
        if "pts_time:" in line:
            parts = line.split("pts_time:")
            if len(parts) > 1:
                time_str = parts[1].split()[0]
                try:
                    timestamps.append(float(time_str))
                except ValueError:
                    continue

    return timestamps
