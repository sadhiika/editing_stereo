from typing import Any, Dict

from .base import ModelClient, GenerationResult


class DeepSeekClient(ModelClient):
    @property
    def provider_name(self) -> str:
        return "deepseek"

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> GenerationResult:
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        raw = self._post_json(
            f"{self.base_url}/chat/completions",
            payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        text = raw.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        if not text:
            raise RuntimeError(f"DeepSeek returned empty content for model {self.model_name}")
        return GenerationResult(text=text, raw=raw)
