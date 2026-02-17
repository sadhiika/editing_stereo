from typing import Any, Dict

from .base import ModelClient, GenerationResult


class HuggingFaceClient(ModelClient):
    @property
    def provider_name(self) -> str:
        return "huggingface"

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> GenerationResult:
        payload: Dict[str, Any] = {
            "inputs": prompt,
            "parameters": {
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "return_full_text": False,
            },
        }
        raw = self._post_json(
            f"{self.base_url}/models/{self.model_name}",
            payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

        text = ""
        if isinstance(raw, list) and raw:
            text = raw[0].get("generated_text", "").strip()
        elif isinstance(raw, dict):
            text = raw.get("generated_text", "").strip()

        if not text:
            raise RuntimeError(f"HuggingFace returned empty content for model {self.model_name}")

        return GenerationResult(text=text, raw={"response": raw})
