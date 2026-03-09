from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

from homie_core.model.registry import ModelEntry


class ModelEngine:
    def __init__(self):
        self._backend = None
        self._loaded = False
        self._current_model: Optional[ModelEntry] = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def current_model(self) -> Optional[ModelEntry]:
        return self._current_model

    def load(self, entry: ModelEntry, n_ctx: int = 4096, n_gpu_layers: int = -1) -> None:
        if entry.format == "gguf":
            from homie_core.model.gguf_backend import GGUFBackend
            backend = GGUFBackend()
            backend.load(entry.path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)
            self._backend = backend
        elif entry.format == "safetensors":
            from homie_core.model.transformers_backend import TransformersBackend
            backend = TransformersBackend()
            backend.load(entry.path)
            self._backend = backend
        else:
            raise ValueError(f"Unsupported format: {entry.format}")
        self._loaded = True
        self._current_model = entry

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7, stop: Optional[list[str]] = None) -> str:
        if not self._loaded:
            raise RuntimeError("No model loaded")
        return self._backend.generate(prompt, max_tokens=max_tokens, temperature=temperature, stop=stop)

    def stream(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7, stop: Optional[list[str]] = None) -> Iterator[str]:
        if not self._loaded:
            raise RuntimeError("No model loaded")
        return self._backend.stream(prompt, max_tokens=max_tokens, temperature=temperature, stop=stop)

    def unload(self) -> None:
        if self._backend:
            self._backend.unload()
        self._backend = None
        self._loaded = False
        self._current_model = None
