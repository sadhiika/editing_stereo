import os
from typing import Dict, Tuple

from .base import ModelClient
from .deepseek import DeepSeekClient
from .huggingface import HuggingFaceClient
from .together import TogetherClient

# model_name -> (provider, remote_model_id)
MODEL_REGISTRY: Dict[str, Tuple[str, str]] = {
    # Free-tier focused model set
    "deepseek-v3.2": ("deepseek", "deepseek-chat"),
    "deepseek-r1-distill-llama-70b": ("deepseek", "deepseek-reasoner"),
    "phi-4": ("huggingface", "microsoft/phi-4"),
    "gemma-3-12b": ("huggingface", "google/gemma-3-12b-it"),
}


def available_models() -> Dict[str, Tuple[str, str]]:
    return dict(MODEL_REGISTRY)


def get_model_client(model_name: str) -> ModelClient:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Known models: {', '.join(sorted(MODEL_REGISTRY))}")

    provider, remote_model = MODEL_REGISTRY[model_name]

    if provider == "together":
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("Missing TOGETHER_API_KEY")
        return TogetherClient(model_name=remote_model, api_key=api_key, base_url="https://api.together.xyz/v1")

    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("MOONSHOT_API_KEY") or os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("Missing one of DEEPSEEK_API_KEY / MOONSHOT_API_KEY / MISTRAL_API_KEY")
        # OpenAI-compatible endpoint expected for deployed gateway / compatible proxy.
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        return DeepSeekClient(model_name=remote_model, api_key=api_key, base_url=base_url)

    if provider == "huggingface":
        api_key = os.getenv("HF_TOKEN")
        if not api_key:
            raise ValueError("Missing HF_TOKEN")
        return HuggingFaceClient(model_name=remote_model, api_key=api_key, base_url="https://api-inference.huggingface.co")

    raise ValueError(f"Unsupported provider: {provider}")
