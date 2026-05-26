"""Homie plugin: nomic-ai/nomic-embed-text-v1.5 Matryoshka embeddings.

Provides local, variable-dimension text embeddings for Homie's RAG layer
(ChromaDB) and any downstream semantic search / clustering features.

The nomic-embed-text-v1.5 model is trained with Matryoshka Representation
Learning (MRL), which means a single forward pass yields a 768-d vector
whose leading prefixes are themselves valid embeddings. Truncating to
512 / 256 / 128 / 64 dims trades a small amount of recall for large
savings in storage and ANN search latency -- useful on memory-constrained
desktop or mobile deployments of Homie.

The model is loaded once on ``activate()`` from a locally cached copy
(``~/.cache/huggingface``) and runs entirely on-device via
sentence-transformers. No outbound network calls are made after the
initial model download, which itself only happens if the weights are
not already on disk and ``allow_download=True`` in plugin config.

Nomic embeddings require task-specific prefixes:
    - 'search_document: '  -- for documents being indexed
    - 'search_query: '     -- for user queries
    - 'clustering: '       -- for grouping / topic discovery
    - 'classification: '   -- for supervised downstream tasks
The plugin applies the correct prefix automatically based on the ``task``
argument.

Source: https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

PLUGIN_NAME = "nomic_embed_text_v15_matryoshka"
PLUGIN_VERSION = "1.0.0"
MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"

# Matryoshka-valid truncation dimensions for nomic-embed-text-v1.5.
# Choosing a dim outside this set is allowed but will degrade quality.
VALID_MATRYOSHKA_DIMS = (64, 128, 256, 512, 768)
FULL_DIM = 768

# Task -> required prompt prefix (see model card).
TASK_PREFIXES: Dict[str, str] = {
    "search_document": "search_document: ",
    "search_query": "search_query: ",
    "clustering": "clustering: ",
    "classification": "classification: ",
}
DEFAULT_TASK = "search_document"


@dataclass
class NomicEmbedConfig:
    """Runtime configuration for the embedding plugin."""

    model_id: str = MODEL_ID
    output_dim: int = FULL_DIM
    device: str = "cpu"
    normalize: bool = True
    batch_size: int = 32
    max_seq_length: int = 8192
    allow_download: bool = False
    cache_dir: Optional[str] = None
    trust_remote_code: bool = True
    extra_st_kwargs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Optional[Dict[str, Any]]) -> "NomicEmbedConfig":
        data = dict(data or {})
        cfg = cls(
            model_id=data.get("model_id", MODEL_ID),
            output_dim=int(data.get("output_dim", FULL_DIM)),
            device=str(data.get("device", "cpu")),
            normalize=bool(data.get("normalize", True)),
            batch_size=int(data.get("batch_size", 32)),
            max_seq_length=int(data.get("max_seq_length", 8192)),
            allow_download=bool(data.get("allow_download", False)),
            cache_dir=data.get("cache_dir"),
            trust_remote_code=bool(data.get("trust_remote_code", True)),
            extra_st_kwargs=dict(data.get("extra_st_kwargs", {})),
        )
        if cfg.output_dim not in VALID_MATRYOSHKA_DIMS:
            logger.warning(
                "output_dim=%d is not a Matryoshka-trained dimension %s; "
                "quality will be degraded.",
                cfg.output_dim,
                VALID_MATRYOSHKA_DIMS,
            )
        if cfg.output_dim > FULL_DIM or cfg.output_dim < 1:
            raise ValueError(
                f"output_dim must be in [1, {FULL_DIM}], got {cfg.output_dim}"
            )
        return cfg


class NomicEmbedTextPlugin:
    """Local sentence-embedding plugin backed by nomic-embed-text-v1.5.

    Use ``embed()`` for document/corpus encoding and ``embed_query()`` for
    user queries -- the two apply different task prefixes as required by
    the underlying model.

    Truncation to ``cfg.output_dim`` is performed in-process (Matryoshka
    slicing) followed by optional L2 re-normalization, which is the
    correct sequence for cosine similarity at reduced dimensions.
    """

    name: str = PLUGIN_NAME
    version: str = PLUGIN_VERSION

    def __init__(self, cfg: Optional[NomicEmbedConfig] = None) -> None:
        self.cfg: NomicEmbedConfig = cfg or NomicEmbedConfig()
        self._model: Any = None  # sentence_transformers.SentenceTransformer
        self._np: Any = None  # lazy numpy import

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def activate(self) -> None:
        """Load the model into memory. Idempotent."""
        if self._model is not None:
            logger.debug("%s already active", self.name)
            return

        try:
            import numpy as np  # noqa: WPS433 -- intentional lazy import
            from sentence_transformers import SentenceTransformer  # noqa: WPS433
        except ImportError as exc:  # noqa: BLE001
            raise RuntimeError(
                "nomic_embed_text_v15_matryoshka requires 'sentence-transformers' "
                "and 'numpy'. Install with: pip install sentence-transformers numpy"
            ) from exc

        self._np = np

        # Enforce offline mode unless the operator explicitly opted in to
        # downloading. Homie is a local-first app -- silent network calls
        # are not acceptable.
        if not self.cfg.allow_download:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        logger.info(
            "Loading %s (device=%s, output_dim=%d, max_seq=%d)",
            self.cfg.model_id,
            self.cfg.device,
            self.cfg.output_dim,
            self.cfg.max_seq_length,
        )
        self._model = SentenceTransformer(
            self.cfg.model_id,
            device=self.cfg.device,
            cache_folder=self.cfg.cache_dir,
            trust_remote_code=self.cfg.trust_remote_code,
            **self.cfg.extra_st_kwargs,
        )
        try:
            self._model.max_seq_length = self.cfg.max_seq_length
        except Exception:  # noqa: BLE001
            logger.debug("Could not override max_seq_length")

    def deactivate(self) -> None:
        """Release the loaded model."""
        if self._model is None:
            return
        logger.info("Unloading %s", self.cfg.model_id)
        try:
            del self._model
        finally:
            self._model = None
            self._np = None

    # ------------------------------------------------------------------ #
    # Public embedding API
    # ------------------------------------------------------------------ #
    def embed(
        self,
        texts: Sequence[str],
        task: str = DEFAULT_TASK,
        output_dim: Optional[int] = None,
    ) -> List[List[float]]:
        """Embed a batch of documents and return list-of-list floats.

        Args:
            texts: Iterable of input strings.
            task: One of ``TASK_PREFIXES`` keys. Defaults to
                ``search_document`` (correct for RAG indexing).
            output_dim: Optional override for Matryoshka truncation; falls
                back to the configured ``cfg.output_dim``.
        """
        return [vec.tolist() for vec in self._encode(texts, task, output_dim)]

    def embed_query(
        self,
        query: str,
        output_dim: Optional[int] = None,
    ) -> List[float]:
        """Embed a single search query (applies the ``search_query`` prefix)."""
        vec = self._encode([query], task="search_query", output_dim=output_dim)[0]
        return vec.tolist()

    def dimension(self) -> int:
        """Return the active output dimension after Matryoshka truncation."""
        return self.cfg.output_dim

    # ------------------------------------------------------------------ #
    # ChromaDB adapter
    # ------------------------------------------------------------------ #
    def as_chroma_embedding_function(self):
        """Return an object compatible with ChromaDB's embedding_function API.

        ChromaDB calls ``ef(input=List[str]) -> List[List[float]]``.
        """
        plugin = self

        class _ChromaEF:
            def __call__(self, input: Sequence[str]) -> List[List[float]]:  # noqa: A002,E501
                return plugin.embed(list(input), task="search_document")

            def name(self) -> str:  # Chroma >=0.5 metadata hook
                return f"{plugin.name}-{plugin.cfg.output_dim}d"

        return _ChromaEF()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _encode(
        self,
        texts: Sequence[str],
        task: str,
        output_dim: Optional[int],
    ) -> Any:
        if self._model is None:
            raise RuntimeError(
                f"{self.name} not active. Call activate() first."
            )
        if not isinstance(texts, (list, tuple)):
            texts = list(texts)
        if any(not isinstance(t, str) for t in texts):
            raise TypeError("All inputs must be strings.")

        prefix = TASK_PREFIXES.get(task)
        if prefix is None:
            raise ValueError(
                f"Unknown task '{task}'. Valid: {sorted(TASK_PREFIXES)}"
            )
        prefixed = [prefix + t for t in texts]

        target_dim = int(output_dim or self.cfg.output_dim)
        if target_dim < 1 or target_dim > FULL_DIM:
            raise ValueError(
                f"output_dim must be in [1, {FULL_DIM}], got {target_dim}"
            )

        # Encode at full dim, then Matryoshka-truncate + renormalize.
        # Doing normalization AFTER truncation is the correct order for
        # MRL embeddings used with cosine similarity.
        full = self._model.encode(
            prefixed,
            batch_size=self.cfg.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        truncated = full[:, :target_dim]
        if self.cfg.normalize:
            norms = self._np.linalg.norm(truncated, axis=1, keepdims=True)
            norms = self._np.where(norms == 0, 1.0, norms)
            truncated = truncated / norms
        return truncated


# ---------------------------------------------------------------------- #
# Module-level plugin entry points (Homie plugin discovery convention)
# ---------------------------------------------------------------------- #
_PLUGIN_INSTANCE: Optional[NomicEmbedTextPlugin] = None


def register(config: Optional[Dict[str, Any]] = None) -> NomicEmbedTextPlugin:
    """Construct and register the plugin instance with Homie.

    Homie's plugin loader is expected to call ``register(plugin_cfg_dict)``
    when wiring up optional capabilities at startup. Returns the live
    plugin instance so the caller can attach it to the RAG layer.
    """
    global _PLUGIN_INSTANCE
    cfg = NomicEmbedConfig.from_mapping(config)
    _PLUGIN_INSTANCE = NomicEmbedTextPlugin(cfg)
    return _PLUGIN_INSTANCE


def activate(config: Optional[Dict[str, Any]] = None) -> NomicEmbedTextPlugin:
    """Idempotently register (if needed) and activate the plugin."""
    global _PLUGIN_INSTANCE
    if _PLUGIN_INSTANCE is None:
        register(config)
    assert _PLUGIN_INSTANCE is not None  # for type checkers
    _PLUGIN_INSTANCE.activate()
    return _PLUGIN_INSTANCE


def deactivate() -> None:
    """Tear down the plugin and free model memory."""
    global _PLUGIN_INSTANCE
    if _PLUGIN_INSTANCE is not None:
        _PLUGIN_INSTANCE.deactivate()
        _PLUGIN_INSTANCE = None


__all__ = [
    "PLUGIN_NAME",
    "PLUGIN_VERSION",
    "MODEL_ID",
    "VALID_MATRYOSHKA_DIMS",
    "FULL_DIM",
    "TASK_PREFIXES",
    "NomicEmbedConfig",
    "NomicEmbedTextPlugin",
    "register",
    "activate",
    "deactivate",
]
