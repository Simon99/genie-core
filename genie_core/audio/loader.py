from __future__ import annotations

import json
from pathlib import Path

from .srt import read_srt


def load_transcript(path: str) -> list[dict]:
    """Load a transcript file into [{"start", "end", "text"}] segments.

    Supported formats (by extension):
    - .json: either a list of segment dicts, or a dict with a "segments" key
      (e.g. whisper output) that is unwrapped automatically.
    - .srt: parsed with read_srt.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".json":
        with open(p, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "segments" in data:
            segments = data["segments"]
            if isinstance(segments, list):
                return segments
            raise ValueError(
                '%s: "segments" key is %s, expected a list of segment dicts'
                % (path, type(segments).__name__)
            )
        raise ValueError(
            "%s: unsupported JSON transcript structure (%s). Expected a list "
            'of {"start", "end", "text"} dicts, or a dict with a "segments" list.'
            % (path, type(data).__name__)
        )

    if suffix == ".srt":
        return read_srt(str(p))

    raise ValueError(
        "%s: unsupported transcript format %r. Supported: .json, .srt"
        % (path, suffix)
    )
