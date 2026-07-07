import base64
import json
import logging
import os
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# Model auto-selection: ordered substring preferences, overridable via
# GENIE_TEXT_MODELS / GENIE_VISION_MODELS (comma-separated substrings).
TEXT_PREFERENCES = ["qwen3.6", "glm-4.7", "qwen3.5-122b", "qwen3.5-35b", "qwen3.5", "gpt-oss"]
VISION_PREFERENCES = ["qwen3-vl-30b", "qwen3-vl-32b-instruct", "qwen3-vl", "glm-4.6v", "vl"]
_VISION_MARKERS = ("-vl", "vl-", "4.6v", "vision")


def _looks_vision(model_id: str) -> bool:
    low = model_id.lower()
    return any(m in low for m in _VISION_MARKERS)


def pick_model(available: list, kind: str = "text", requested: str = None) -> str:
    """Pick a model id from LM Studio's list by kind and priority.

    Exact ``requested`` id wins. A missing requested model falls back to
    the preference list with a warning (machines carry different stacks —
    hardcoded ids caused instant 404s). Embedding models are never picked;
    vision-looking ids are excluded for kind="text" and required for
    kind="vision".
    """
    if requested and requested in available:
        return requested

    pool = [m for m in available if "embed" not in m.lower()]
    if kind == "vision":
        pool = [m for m in pool if _looks_vision(m)]
    else:
        pool = [m for m in pool if not _looks_vision(m)]
    if not pool:
        raise RuntimeError(
            "No %s model available in LM Studio (models: %s)" % (kind, available))

    env = os.environ.get(
        "GENIE_VISION_MODELS" if kind == "vision" else "GENIE_TEXT_MODELS")
    prefs = ([p.strip() for p in env.split(",") if p.strip()] if env
             else (VISION_PREFERENCES if kind == "vision" else TEXT_PREFERENCES))

    choice = None
    for pref in prefs:
        hits = [m for m in pool if pref.lower() in m.lower()]
        if hits:
            choice = hits[0]
            break
    if choice is None:
        choice = pool[0]
    if requested:
        logger.warning("requested model %r not found, using %r", requested, choice)
    return choice


class LMStudioClient:
    """Client for LM Studio's OpenAI-compatible API (no openai SDK dependency).

    ``model=None`` auto-picks from the server's model list by ``kind``
    ("text" or "vision") and the preference order above.
    """

    def __init__(self, base_url: str = "http://localhost:1234/v1", model: str = None,
                 timeout=(5, 300), kind: str = "text"):
        self.base_url = base_url.rstrip("/")
        self._requested = model
        self._model = None
        self.kind = kind
        self.timeout = timeout

    def list_models(self) -> list:
        resp = requests.get(f"{self.base_url}/models", timeout=self.timeout)
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", [])]

    @property
    def model(self) -> str:
        if self._model:
            return self._model
        available = self.list_models()
        if not available:
            raise RuntimeError("No models loaded in LM Studio")
        self._model = pick_model(available, kind=self.kind, requested=self._requested)
        if self._model != self._requested:
            print("LM Studio model: %s" % self._model)
        return self._model

    @staticmethod
    def _extract_content(data: dict) -> str:
        msg = data["choices"][0]["message"]
        content = msg.get("content") or ""
        if not content.strip() and (msg.get("reasoning_content") or "").strip():
            # Thinking models stream reasoning into a separate channel; when
            # the token budget runs out there, content comes back empty.
            raise RuntimeError(
                "LLM returned empty content but %d chars of reasoning — "
                "a thinking model likely spent the whole token budget on "
                "reasoning. Use a non-thinking model or raise max_tokens."
                % len(msg["reasoning_content"]))
        return content

    def complete(self, prompt: str, system: str = None, temperature: float = 0.3,
                 max_tokens: int = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {"model": self.model, "messages": messages, "temperature": temperature}
        if max_tokens:
            payload["max_tokens"] = max_tokens
        resp = requests.post(
            f"{self.base_url}/chat/completions", json=payload, timeout=self.timeout,
        )
        resp.raise_for_status()
        return self._extract_content(resp.json())

    def vision(self, prompt: str, image_path: str, system: str = None,
               temperature: float = 0.3, max_tokens: int = None) -> str:
        """Send an image + text prompt to a vision model."""
        image_data = Path(image_path).read_bytes()
        b64 = base64.b64encode(image_data).decode("utf-8")

        suffix = Path(image_path).suffix.lower()
        mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
        mime = mime_map.get(suffix, "image/png")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ],
        })

        payload = {"model": self.model, "messages": messages, "temperature": temperature}
        if max_tokens:
            payload["max_tokens"] = max_tokens
        resp = requests.post(
            f"{self.base_url}/chat/completions", json=payload, timeout=self.timeout,
        )
        resp.raise_for_status()
        return self._extract_content(resp.json())
