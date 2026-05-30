"""Homie plugin: Nomic Embed Text v1.5 â€” long-context Matryoshka embeddings for RAG.

Wraps ``nomic-ai/nomic-embed-text-v1.5`` (https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
as a local embedding backend for Homie's RAG pipeline (ChromaDB). The model produces
768-dimensional sentence embeddings trained with Matryoshka Representation Learning, so
the vectors can be truncated to 512 / 256 / 128 / 64 dims with minimal quality loss â€”
useful when storing long-document corpora locally on constrained desktop/mobile hardware.

Key traits:
    * Long context window (up to 8192 tokens after rope-scaling, default 2048) â€” fits whole
      passages without aggressive chunking, ideal for long-document RAG.
    * Task-prefixed inputs (``search_document:`` / ``search_query:`` / ``clustering:`` /
      ``classification:``) â€” the plugin attaches the correct prefix automatically.
    * Fully local once the model is cached. The first ``activate()`` call may download
      weights from Hugging Face; set ``offline=True`` in the plugin config to refuse any
      network access and require a pre-downloaded snapshot.
    * Exposes a ChromaDB-compatible ``EmbeddingFunction`` plus direct ``embed_documents`` /
      ``embed_query`` helpers for ad-hoc use from other Homie plugins.

This plugin performs **no network calls at runtime** once the model is loaded; only the
optional first-time model download touches the network, and that is gated by the
``offline`` flag.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

PLUGIN_NAME = "nomic_embed_text_v15"
MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
FULL_DIM = 768
ALLOWED_MATRYOSHKA_DIMS = (64, 128, 256, 512, 768)
VALID_TASKS = ("search_document", "search_query", "clustering", "classification")


@dataclass
class NomicEmbedConfig:
    """Resolved plugin configuration."""

    model_id: str = MODEL_ID
    cache_dir: Optional[Path] = None
    device: str = "cpu"  # "cpu" | "cuda" | "mps"
    matryoshka_dim: int = FULL_DIM
    max_seq_length: int = 2048
    normalize: bool = True
    default_task: str = "search_document"
    offline: bool = False
    trust_remote_code: bool = True  # nomic_bert requires custom code
    batch_size: int = 16
    extra_model_kwargs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "NomicEmbedConfig":
        data = dict(data or {})
        cache = data.get("cache_dir")
        if cache is not None:
            data["cache_dir"] = Path(cache).expanduser().resolve()
        dim = int(data.get("matryoshka_dim", FULL_DIM))
        if dim not in ALLOWED_MATRYOSHKA_DIMS:
            raise ValueError(
                f"matryoshka_dim must be one of {ALLOWED_MATRYOSHKA_DIMS}, got {dim}"
            )
        data["matryoshka_dim"] = dim
        task = data.get("default_task", "search_document")
        if task not in VALID_TASKS:
            raise ValueError(f"default_task must be one of {VALID_TASKS}, got {task!r}")
        return cls(**data)


class NomicEmbedTextV15Plugin:
    """Long-context Matryoshka embedding backend for Homie RAG.

    Lifecycle:
        plugin = NomicEmbedTextV15Plugin(config)
        plugin.activate()              # loads the SentenceTransformer model
        vecs = plugin.embed_documents(["..."])
        plugin.deactivate()            # releases model + frees memory
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: NomicEmbedConfig = NomicEmbedConfig.from_dict(config)
        self._model: Any = None  # SentenceTransformer; deferred import
        self._np: Any = None  # numpy; deferred import
        self._active: bool = False

    @property
    def name(self) -> str:
        return PLUGIN_NAME

    @property
    def embedding_dim(self) -> int:
        return self.config.matryoshka_dim

    @property
    def is_active(self) -> bool:
        return self._active

    # ------------------------------------------------------------------ lifecycle
    def activate(self) -> None:
        """Load the SentenceTransformer model into memory.

        Honors ``offline=True`` by setting Hugging Face env flags before import so the
        underlying ``transformers`` / ``sentence_transformers`` libraries refuse network
        access and only use locally cached snapshots.
        """
        if self._active:
            logger.debug("%s already active", PLUGIN_NAME)
            return

        if self.config.offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        try:
            import numpy as np  # type: ignore
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "nomic_embed_text_v15 plugin requires `sentence-transformers` and `numpy`. "
                "Install them into Homie's environment before activating."
            ) from exc

        cache_dir = str(self.config.cache_dir) if self.config.cache_dir else None
        logger.info(
            "Loading %s (device=%s, dim=%d, max_seq=%d, offline=%s)",
            self.config.model_id,
            self.config.device,
            self.config.matryoshka_dim,
            self.config.max_seq_length,
            self.config.offline,
        )
        model = SentenceTransformer(
            self.config.model_id,
            device=self.config.device,
            cache_folder=cache_dir,
            trust_remote_code=self.config.trust_remote_code,
            **self.config.extra_model_kwargs,
        )
        # Allow long-context use; the model card supports up to 8192 with rope scaling.
        model.max_seq_length = int(self.config.max_seq_length)

        self._model = model
        self._np = np
        self._active = True

    def deactivate(self) -> None:
        """Release the model and attempt to free GPU memory."""
        if not self._active:
            return
        self._model = None
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass
        self._active = False
        logger.info("%s deactivated", PLUGIN_NAME)

    # ------------------------------------------------------------------ embedding
    def _ensure_active(self) -> None:
        if not self._active or self._model is None:
            raise RuntimeError(
                f"{PLUGIN_NAME} is not active; call activate() before embedding."
            )

    def _apply_prefix(self, text: str, task: str) -> str:
        if task not in VALID_TASKS:
            raise ValueError(f"task must be one of {VALID_TASKS}, got {task!r}")
        return f"{task}: {text}"

    def _truncate_matryoshka(self, embeddings: Any) -> Any:
        dim = self.config.matryoshka_dim
        if dim >= FULL_DIM:
            return embeddings
        truncated = embeddings[..., :dim]
        if self.config.normalize:
            norms = self._np.linalg.norm(truncated, axis=-1, keepdims=True)
            norms = self._np.where(norms == 0, 1.0, norms)
            truncated = truncated / norms
        return truncated

    def _encode(self, texts: Sequence[str], task: str) -> List[List[float]]:
        self._ensure_active()
        if not texts:
            return []
        prefixed = [self._apply_prefix(t, task) for t in texts]
        vectors = self._model.encode(
            prefixed,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors = self._truncate_matryoshka(vectors)
        return [vec.tolist() for vec in vectors]

    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        """Embed corpus passages for storage/indexing (``search_document`` prefix)."""
        return self._encode(list(texts), task="search_document")

    def embed_query(self, text: str) -> List[float]:
        """Embed a single user query (``search_query`` prefix)."""
        result = self._encode([text], task="search_query")
        return result[0] if result else []

    def embed(self, texts: Iterable[str], task: Optional[str] = None) -> List[List[float]]:
        """Generic embedding entrypoint with explicit task selection."""
        return self._encode(list(texts), task=task or self.config.default_task)

    # ------------------------------------------------------------------ ChromaDB glue
    def as_chroma_embedding_function(self) -> "_ChromaEmbeddingFunction":
        """Return a ChromaDB-compatible embedding function bound to this plugin."""
        return _ChromaEmbeddingFunction(self)


class _ChromaEmbeddingFunction:
    """Adapter so Homie's ChromaDB collections can use this plugin directly.

    Matches the ``chromadb.api.types.EmbeddingFunction`` protocol: a callable that takes a
    list of documents and returns a list of embedding vectors. Treats inputs as documents
    by default; callers wanting query-style prefixes should use ``plugin.embed_query``.
    """

    def __init__(self, plugin: NomicEmbedTextV15Plugin) -> None:
        self._plugin = plugin

    def __call__(self, input: Sequence[str]) -> List[List[float]]:  # noqa: A002 - chroma signature
        return self._plugin.embed_documents(list(input))

    def name(self) -> str:
        return f"{PLUGIN_NAME}:dim={self._plugin.embedding_dim}"


# --------------------------------------------------------------------- registration
_PLUGIN_SINGLETON: Optional[NomicEmbedTextV15Plugin] = None


def register(config: Optional[Dict[str, Any]] = None) -> NomicEmbedTextV15Plugin:
    """Construct (or return the cached) plugin instance.

    Homie's plugin loader calls ``register()`` once at startup and stashes the returned
    object on the plugin registry. ``activate()`` is invoked separately so model loading
    can be deferred until the RAG subsystem is actually needed.
    """
    global _PLUGIN_SINGLETON
    if _PLUGIN_SINGLETON is None:
        _PLUGIN_SINGLETON = NomicEmbedTextV15Plugin(config)
        logger.debug("Registered Homie plugin %s", PLUGIN_NAME)
    return _PLUGIN_SINGLETON


def activate(config: Optional[Dict[str, Any]] = None) -> NomicEmbedTextV15Plugin:
    """Convenience entrypoint: register + activate in one call."""
    plugin = register(config)
    plugin.activate()
    return plugin


def deactivate() -> None:
    """Tear down the singleton plugin instance."""
    global _PLUGIN_SINGLETON
    if _PLUGIN_SINGLETON is not None:
        _PLUGIN_SINGLETON.deactivate()
        _PLUGIN_SINGLETON = None


__all__ = [
    "NomicEmbedConfig",
    "NomicEmbedTextV15Plugin",
    "PLUGIN_NAME",
    "MODEL_ID",
    "FULL_DIM",
    "ALLOWED_MATRYOSHKA_DIMS",
    "VALID_TASKS",
    "register",
    "activate",
    "deactivate",
]
