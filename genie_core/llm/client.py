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


def pick_model(available: list, kind: str = "text", requested: str = None,
               info: dict = None) -> str:
    """Pick a model id from LM Studio's list by kind and priority.

    Exact ``requested`` id wins. A missing requested model falls back to
    the preference list with a warning (machines carry different stacks —
    hardcoded ids caused instant 404s).

    ``info`` (optional) maps id -> {"llm": bool, "vision": bool,
    "loaded": bool} from the native models endpoint; real capability
    flags then replace the id-substring heuristics, and already-loaded
    models win ties (no load latency).
    """
    if requested and requested in available:
        return requested

    if info:
        pool = [m for m in available if info.get(m, {}).get("llm", True)]
        if kind == "vision":
            pool = [m for m in pool if info.get(m, {}).get("vision")]
    else:
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

    def rank(m):
        loaded = 0 if (info or {}).get(m, {}).get("loaded") else 1
        for i, pref in enumerate(prefs):
            if pref.lower() in m.lower():
                return (loaded, i)
        return (loaded, len(prefs))

    choice = min(pool, key=lambda m: (rank(m), pool.index(m)))
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
        self._native = None  # tri-state: unknown / True / False

    def list_models(self) -> list:
        resp = requests.get(f"{self.base_url}/models", timeout=self.timeout)
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", [])]

    @property
    def _native_root(self) -> str:
        root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        return root + "/api/v1"

    def list_models_native(self):
        """Model info from the native endpoint, or None if unsupported.

        Returns {id: {"llm": bool, "vision": bool, "loaded": bool,
        "context_length": int}}.
        """
        try:
            resp = requests.get(f"{self._native_root}/models", timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            entries = data if isinstance(data, list) else (
                data.get("data") or data.get("models") or [])
            info = {}
            for m in entries:
                key = m.get("key") or m.get("id")
                if not key:
                    continue
                loaded = m.get("loaded_instances") or []
                ctx = None
                if loaded:
                    ctx = (loaded[0].get("config") or {}).get("context_length")
                info[key] = {
                    "llm": m.get("type") == "llm",
                    "vision": bool((m.get("capabilities") or {}).get("vision")),
                    "loaded": bool(loaded),
                    "context_length": ctx or m.get("max_context_length"),
                }
            return info or None
        except Exception:
            return None

    def get_context_length(self, default: int = 8192) -> int:
        """Context window of the selected model (native endpoint), or default."""
        info = self.list_models_native() or {}
        entry = info.get(self.model) or {}
        return entry.get("context_length") or default

    @property
    def model(self) -> str:
        if self._model:
            return self._model
        info = self.list_models_native()
        available = list(info) if info else self.list_models()
        if not available:
            raise RuntimeError("No models loaded in LM Studio")
        self._model = pick_model(available, kind=self.kind,
                                 requested=self._requested, info=info)
        if self._model != self._requested:
            print("LM Studio model: %s" % self._model)
        return self._model

    @staticmethod
    def _starvation_error(reasoning_len: int):
        # Thinking models stream reasoning into a separate channel; when
        # the token budget runs out there, content comes back empty.
        return RuntimeError(
            "LLM returned empty content but %d chars of reasoning — "
            "a thinking model likely spent the whole token budget on "
            "reasoning. Use a non-thinking model or raise max_tokens."
            % reasoning_len)

    @staticmethod
    def _extract_content(data: dict) -> str:
        msg = data["choices"][0]["message"]
        content = msg.get("content") or ""
        if not content.strip() and (msg.get("reasoning_content") or "").strip():
            raise LMStudioClient._starvation_error(len(msg["reasoning_content"]))
        return content

    @staticmethod
    def _extract_native(data: dict) -> str:
        """Parse LM Studio native /api/v1/chat output (typed item list)."""
        items = data.get("output") or []
        content = "\n".join(
            i.get("content") or "" for i in items if i.get("type") == "message").strip()
        if not content:
            reasoning = sum(
                len(i.get("content") or "") for i in items if i.get("type") == "reasoning")
            if reasoning:
                raise LMStudioClient._starvation_error(reasoning)
        return content

    def _post_native(self, input_value, system: str, temperature: float,
                     max_tokens: int) -> str:
        payload = {"model": self.model, "input": input_value,
                   "temperature": temperature}
        if system:
            payload["system_prompt"] = system
        if max_tokens:
            payload["max_output_tokens"] = max_tokens
        resp = requests.post(f"{self._native_root}/chat", json=payload,
                             timeout=self.timeout)
        resp.raise_for_status()
        return self._extract_native(resp.json())

    def _post_openai(self, messages: list, temperature: float, max_tokens: int) -> str:
        payload = {"model": self.model, "messages": messages, "temperature": temperature}
        if max_tokens:
            payload["max_tokens"] = max_tokens
        resp = requests.post(
            f"{self.base_url}/chat/completions", json=payload, timeout=self.timeout,
        )
        resp.raise_for_status()
        return self._extract_content(resp.json())

    def _post_chat(self, messages: list, temperature: float, max_tokens: int,
                   native_input=None) -> str:
        """Prefer the native REST API; fall back to /v1/chat/completions.

        The native endpoint (LM Studio >= 0.4) returns reasoning and
        message as separate typed items, so thinking output never has to
        be scraped out of the answer. Older servers 404 it — remembered
        in self._native after the first attempt.
        """
        system = None
        if messages and messages[0]["role"] == "system":
            system = messages[0]["content"]
        if self._native is not False and native_input is not None:
            try:
                result = self._post_native(native_input, system, temperature, max_tokens)
                self._native = True
                return result
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 404:
                    logger.info("native /api/v1/chat not available, using /v1/chat/completions")
                    self._native = False
                else:
                    raise
        return self._post_openai(messages, temperature, max_tokens)

    def complete(self, prompt: str, system: str = None, temperature: float = 0.3,
                 max_tokens: int = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            return self._post_chat(messages, temperature, max_tokens,
                                   native_input=prompt)
        except RuntimeError:
            if not max_tokens:
                raise
            # Reasoning ate the whole budget; give it one doubled retry
            # (observed: glm-4.7-flash spent 11k chars thinking about a
            # merge prompt and returned empty content at max_tokens=4096).
            bigger = max(max_tokens * 2, 8192)
            logger.warning("empty content at max_tokens=%d, retrying with %d",
                           max_tokens, bigger)
            return self._post_chat(messages, temperature, bigger,
                                   native_input=prompt)

    def vision(self, prompt: str, image_path: str, system: str = None,
               temperature: float = 0.3, max_tokens: int = None) -> str:
        """Send an image + text prompt to a vision model."""
        image_data = Path(image_path).read_bytes()
        b64 = base64.b64encode(image_data).decode("utf-8")

        suffix = Path(image_path).suffix.lower()
        mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
        mime = mime_map.get(suffix, "image/png")

        data_url = f"data:{mime};base64,{b64}"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        })
        native_input = [
            {"type": "text", "content": prompt},
            {"type": "image", "data_url": data_url},
        ]
        return self._post_chat(messages, temperature, max_tokens,
                               native_input=native_input)
