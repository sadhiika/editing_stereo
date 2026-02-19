import abc
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib import request, error


@dataclass
class GenerationResult:
    text: str
    raw: Dict[str, Any]


class ModelClient(abc.ABC):
    def __init__(self, model_name: str, api_key: str, base_url: str, timeout_s: int = 60):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.timeout_s = timeout_s

    @property
    @abc.abstractmethod
    def provider_name(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> GenerationResult:
        raise NotImplementedError

    def _post_json(self, url: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None,
                   retries: int = 3, retry_backoff_s: float = 1.5) -> Dict[str, Any]:
        final_headers = {"Content-Type": "application/json"}
        if headers:
            final_headers.update(headers)

        body = json.dumps(payload).encode("utf-8")
        last_error: Optional[str] = None

        for attempt in range(retries + 1):
            try:
                req = request.Request(url, data=body, headers=final_headers, method="POST")
                with request.urlopen(req, timeout=self.timeout_s) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except error.HTTPError as exc:
                raw = exc.read().decode("utf-8", errors="ignore")
                last_error = f"HTTP {exc.code}: {raw[:400]}"
                if exc.code in (429, 500, 502, 503, 504) and attempt < retries:
                    time.sleep(retry_backoff_s ** (attempt + 1))
                    continue
                raise RuntimeError(f"{self.provider_name} request failed: {last_error}")
            except Exception as exc:  # network/transient
                last_error = str(exc)
                if attempt < retries:
                    time.sleep(retry_backoff_s ** (attempt + 1))
                    continue
                raise RuntimeError(f"{self.provider_name} request failed: {last_error}")

        raise RuntimeError(f"{self.provider_name} request failed: {last_error}")
