from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Tolerates "," or "." millisecond separators and trailing position/coordinate
# suffixes (e.g. "X1:0 X2:100 ...") after the end timestamp.
TIMING_RE = re.compile(
    r"(\d+:\d+:\d+[,.]\d+)\s*-->\s*(\d+:\d+:\d+[,.]\d+)"
)


def read_srt(path: str) -> list[dict]:
    """Parse an SRT file into [{"start": float, "end": float, "text": str}].

    Tolerates BOM, CRLF line endings and coordinate suffixes on the timing
    line. Malformed blocks are skipped with a warning.
    """
    with open(path, "r", encoding="utf-8-sig") as f:
        content = f.read()

    content = content.replace("\r\n", "\n").replace("\r", "\n")

    segments = []
    for block in re.split(r"\n\s*\n", content):
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n")

        # Find the timing line (usually line 2, after the numeric index)
        timing_idx = None
        match = None
        for i, line in enumerate(lines):
            m = TIMING_RE.search(line)
            if m:
                timing_idx = i
                match = m
                break

        if match is None:
            logger.warning("Skipping malformed SRT block in %s: %r", path, block[:80])
            continue

        text = "\n".join(lines[timing_idx + 1:]).strip()
        segments.append({
            "start": _srt_time_to_seconds(match.group(1)),
            "end": _srt_time_to_seconds(match.group(2)),
            "text": text,
        })

    return segments


def write_srt(segments: list[dict], path: str):
    """Write [{"start", "end", "text"}] segments to an SRT file."""
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write("%d\n%s --> %s\n%s\n\n" % (
                i,
                _seconds_to_srt_time(float(seg["start"])),
                _seconds_to_srt_time(float(seg["end"])),
                seg.get("text", ""),
            ))


def _srt_time_to_seconds(text: str) -> float:
    h, m, rest = text.split(":")
    rest = rest.replace(",", ".")
    return int(h) * 3600 + int(m) * 60 + float(rest)


def _seconds_to_srt_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    if ms >= 1000:
        ms -= 1000
        s += 1
        if s >= 60:
            s -= 60
            m += 1
            if m >= 60:
                m -= 60
                h += 1
    return "%02d:%02d:%02d,%03d" % (h, m, s, ms)
