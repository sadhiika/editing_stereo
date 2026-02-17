import json
from pathlib import Path
from typing import Any, Dict


def load_prompts_any_format(path: str) -> Dict[str, Dict[str, str]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompts file not found: {path}")

    raw = json.loads(p.read_text())
    normalized: Dict[str, Dict[str, str]] = {}

    if isinstance(raw, dict):
        for key, value in raw.items():
            if not isinstance(value, dict):
                continue
            text = value.get("text") or value.get("prompt") or value.get("prompt_text")
            category = value.get("category") or value.get("type") or "uncategorized"
            if text:
                normalized[str(key)] = {"text": str(text), "category": str(category)}
    elif isinstance(raw, list):
        for i, item in enumerate(raw, start=1):
            if not isinstance(item, dict):
                continue
            prompt_id = item.get("id") or item.get("prompt_id") or f"prompt_{i:03d}"
            text = item.get("text") or item.get("prompt") or item.get("prompt_text")
            category = item.get("category") or item.get("type") or "uncategorized"
            if text:
                normalized[str(prompt_id)] = {"text": str(text), "category": str(category)}

    if not normalized:
        raise ValueError(f"No valid prompts found in {path}")

    return normalized
