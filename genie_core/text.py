from __future__ import annotations


def format_time(seconds) -> str:
    """Format a duration/timestamp as "HH:MM:SS".

    Tolerant of int/float and of strings (LLMs often return timestamps as
    strings): numeric strings ("123", "123.4") and colon formats
    ("MM:SS", "HH:MM:SS", with optional ",ms"/".ms" suffix) are accepted.
    """
    secs = _to_seconds(seconds)
    if secs < 0:
        secs = 0
    total = int(secs)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return "%02d:%02d:%02d" % (h, m, s)


def _to_seconds(value) -> float:
    if isinstance(value, bool):
        raise ValueError("Cannot interpret %r as a timestamp" % (value,))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("Cannot interpret empty string as a timestamp")
        if ":" in text:
            # "HH:MM:SS", "MM:SS", optionally with ",mmm" or ".mmm" suffix
            text = text.replace(",", ".")
            parts = text.split(":")
            if len(parts) > 3:
                raise ValueError("Cannot interpret %r as a timestamp" % (value,))
            try:
                nums = [float(p) for p in parts]
            except ValueError:
                raise ValueError("Cannot interpret %r as a timestamp" % (value,))
            secs = 0.0
            for n in nums:
                secs = secs * 60 + n
            return secs
        try:
            return float(text)
        except ValueError:
            raise ValueError("Cannot interpret %r as a timestamp" % (value,))
    raise ValueError("Cannot interpret %r as a timestamp" % (value,))


def format_segments(segments) -> str:
    """Format transcript segments as one "[HH:MM:SS] text" line per segment.

    segments: iterable of {"start": ..., "text": ...} dicts.
    """
    lines = []
    for seg in segments:
        lines.append("[%s] %s" % (format_time(seg.get("start", 0)), seg.get("text", "")))
    return "\n".join(lines)
