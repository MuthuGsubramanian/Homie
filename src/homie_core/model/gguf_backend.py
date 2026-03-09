from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional


class GGUFBackend:
    def __init__(self):
        self._model = None

    def load(self, model_path: str | Path, n_ctx: int = 4096, n_gpu_layers: int = -1, **kwargs) -> None:
        from llama_cpp import Llama
        self._model = Llama(model_path=str(model_path), n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False, **kwargs)

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7, stop: Optional[list[str]] = None) -> str:
        response = self._model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=temperature, stop=stop,
        )
        return response["choices"][0]["message"]["content"]

    def stream(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7, stop: Optional[list[str]] = None) -> Iterator[str]:
        for chunk in self._model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=temperature, stop=stop, stream=True,
        ):
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            if content:
                yield content

    def unload(self) -> None:
        self._model = None
