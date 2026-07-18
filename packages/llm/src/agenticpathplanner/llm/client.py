"""Ollama client wrapper — thin interface over the ollama Python SDK."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    model: str = "llama3.2"
    temperature: float = 0.2
    top_p: float = 0.9
    num_predict: int = 2048
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: dict = field(default_factory=dict)


class OllamaClient:
    def __init__(self, config: GenerationConfig | None = None, host: str | None = None):
        self.config = config or GenerationConfig()
        self._host = host
        self._client = self._build_client()

    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages)

    def chat(self, messages: list[dict]) -> LLMResponse:
        options = {"temperature": self.config.temperature,
                   "top_p": self.config.top_p,
                   "num_predict": self.config.num_predict,
                   **self.config.options}
        try:
            resp = self._client.chat(model=self.config.model, messages=messages, options=options)
        except Exception as exc:
            raise RuntimeError(
                f"Ollama request failed (model='{self.config.model}'): {exc}\n"
                "Is Ollama running?  Try: ollama serve"
            ) from exc
        return LLMResponse(
            content=resp["message"]["content"],
            model=self.config.model,
            prompt_tokens=resp.get("prompt_eval_count", 0),
            completion_tokens=resp.get("eval_count", 0),
            raw=dict(resp),
        )

    def list_models(self) -> list[str]:
        try:
            return [m["name"] for m in self._client.list().get("models", [])]
        except Exception as exc:
            log.warning("Could not list Ollama models: %s", exc)
            return []

    def _build_client(self):
        try:
            import ollama
        except ImportError as exc:
            raise ImportError("Install ollama: pip install ollama") from exc
        return ollama.Client(host=self._host) if self._host else ollama
