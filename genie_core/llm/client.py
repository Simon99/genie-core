import base64
import json
from pathlib import Path

import requests


class LMStudioClient:
    """Client for LM Studio's OpenAI-compatible API (no openai SDK dependency)."""

    def __init__(self, base_url: str = "http://localhost:1234/v1", model: str = None,
                 timeout=(5, 300)):
        self.base_url = base_url.rstrip("/")
        self._model = model
        self.timeout = timeout

    @property
    def model(self) -> str:
        if self._model:
            return self._model
        resp = requests.get(f"{self.base_url}/models", timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if data:
            self._model = data[0]["id"]
            return self._model
        raise RuntimeError("No models loaded in LM Studio")

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
