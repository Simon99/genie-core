import base64
import json
from pathlib import Path

import requests


class LMStudioClient:
    """Client for LM Studio's OpenAI-compatible API (no openai SDK dependency)."""

    def __init__(self, base_url: str = "http://localhost:1234/v1", model: str = None):
        self.base_url = base_url.rstrip("/")
        self._model = model

    @property
    def model(self) -> str:
        if self._model:
            return self._model
        resp = requests.get(f"{self.base_url}/models")
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if data:
            return data[0]["id"]
        raise RuntimeError("No models loaded in LM Studio")

    def complete(self, prompt: str, system: str = None, temperature: float = 0.3) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json={"model": self.model, "messages": messages, "temperature": temperature},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def vision(self, prompt: str, image_path: str, system: str = None, temperature: float = 0.3) -> str:
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

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json={"model": self.model, "messages": messages, "temperature": temperature},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
