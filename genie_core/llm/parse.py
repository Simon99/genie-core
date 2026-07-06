from __future__ import annotations

import json
import re


def extract_json(text: str):
    """Extract a JSON object or array from an LLM response.

    Strategies, in order:
    1. json.loads on the whole text
    2. strip ``` / ```json fences, then json.loads
    3. slice from the first "{" to the last "}" (and "[" ... "]"), then loads
    4. brace-matching scan (string- and escape-aware) for the first complete
       JSON object/array

    Raises ValueError (with the first 200 chars of the input) if all fail.

    Returns dict or list.
    """
    if not isinstance(text, str):
        raise ValueError("extract_json expects a string, got %s" % type(text).__name__)

    stripped = text.strip()

    # 1. Whole text
    result = _try_loads(stripped)
    if result is not None:
        return result

    # 2. Strip code fences
    fenced = _strip_fences(stripped)
    if fenced is not None:
        result = _try_loads(fenced)
        if result is not None:
            return result

    # 3. First-to-last brace/bracket slice
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = stripped.find(open_ch)
        end = stripped.rfind(close_ch)
        if start != -1 and end > start:
            result = _try_loads(stripped[start:end + 1])
            if result is not None:
                return result

    # 4. Brace-matching scan
    result = _brace_match(stripped)
    if result is not None:
        return result

    raise ValueError(
        "Could not extract JSON from LLM response. First 200 chars:\n%s"
        % text[:200]
    )


def _try_loads(text: str):
    """json.loads that returns None on failure or non-dict/list result."""
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(parsed, (dict, list)):
        return parsed
    return None


def _strip_fences(text: str) -> str | None:
    """Extract the contents of the first ``` fenced block, if any."""
    m = re.search(r"```[a-zA-Z0-9_-]*\s*\n(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Unclosed fence: take everything after the opening fence
    m = re.search(r"```[a-zA-Z0-9_-]*\s*\n(.*)", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def _brace_match(text: str):
    """Scan for the first balanced {...} or [...] block, skipping characters
    inside JSON strings and honoring escapes; try to parse each candidate."""
    openers = {"{": "}", "[": "]"}
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in openers:
            end = _find_balanced_end(text, i, ch, openers[ch])
            if end is not None:
                result = _try_loads(text[i:end + 1])
                if result is not None:
                    return result
        i += 1
    return None


def _find_balanced_end(text: str, start: int, open_ch: str, close_ch: str):
    """Return the index of the matching close_ch for the opener at `start`,
    skipping string contents and escaped characters. None if unbalanced."""
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i
    return None
