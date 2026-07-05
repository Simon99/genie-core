import base64
from pathlib import Path
from openai import OpenAI


class LMStudioClient:
    """Client for LM Studio's OpenAI-compatible API."""

    def __init__(self, base_url: str = "http://localhost:1234/v1", model: str = None):
        self.client = OpenAI(base_url=base_url, api_key="lm-studio")
        self._model = model

    @property
    def model(self) -> str:
        if self._model:
            return self._model
        models = self.client.models.list()
        if models.data:
            return models.data[0].id
        raise RuntimeError("No models loaded in LM Studio")

    def complete(self, prompt: str, system: str = None, temperature: float = 0.3) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content

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

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content
