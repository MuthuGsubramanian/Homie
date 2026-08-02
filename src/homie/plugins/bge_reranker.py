"""BGE reranker plugin for HOMIE.

Adds an optional second-stage reranking step to HOMIE's local RAG pipeline
using ``BAAI/bge-reranker-v2-m3``, a state-of-the-art multilingual
cross-encoder reranker (100+ languages, xlm-roberta based).

Typical flow:

1. ChromaDB returns the top-N candidate chunks for a query (fast, approximate
   vector similarity).
2. This plugin re-scores each (query, chunk) pair with the cross-encoder,
   which reads both texts jointly and is far more precise than cosine
   distance alone.
3. The top-K highest scoring chunks are passed to the LLM context window.

Privacy / network policy:

- Inference is 100% local (CPU by default, CUDA/MPS when available).
- The model weights (~2.3 GB) are fetched from Hugging Face ONLY when the
  user opts in via config (``rag.reranker.allow_download: true``) or the
  model is already present in the local Hugging Face cache. Otherwise
  activation fails with a clear message and HOMIE silently falls back to
  vector-only retrieval.

Config example (homie.config.yaml)::

    rag:
      reranker:
        enabled: true
        model: BAAI/bge-reranker-v2-m3
        allow_download: false
        device: auto          # auto|cpu|cuda|mps
        max_length: 512
        top_k: 5
        batch_size: 16

Dependencies: ``sentence-transformers>=2.7`` (pulls in transformers/torch).
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from homie.config import HomieConfig, cfg_get

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "BAAI/bge-reranker-v2-m3"
DEFAULT_MAX_LENGTH = 512
DEFAULT_TOP_K = 5
DEFAULT_BATCH_SIZE = 16


class RerankerError(Exception):
    """Raised when the reranker cannot be activated or used."""


@dataclass
class RerankResult:
    """A single reranked document."""

    index: int  # position in the original candidate list
    text: str
    score: float  # relevance score, higher = more relevant
    metadata: Dict[str, Any] = field(default_factory=dict)


def _resolve_device(requested: str) -> str:
    """Map an 'auto' device request to the best available backend."""
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def _model_is_cached(model_id: str) -> bool:
    """Check whether the model already exists in the local HF cache (no network)."""
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(model_id, local_files_only=True)
        return True
    except Exception:  # noqa: BLE001
        return False


class BGERerankerPlugin:
    """Optional cross-encoder reranker for HOMIE's RAG retrieval."""

    name = "bge_reranker"

    def __init__(self, cfg: Optional[HomieConfig] = None, **overrides: Any):
        self.cfg = cfg
        self.model_id: str = str(self._opt("model", DEFAULT_MODEL_ID, overrides))
        self.allow_download: bool = bool(self._opt("allow_download", False, overrides))
        self.device: str = _resolve_device(str(self._opt("device", "auto", overrides)))
        self.max_length: int = int(self._opt("max_length", DEFAULT_MAX_LENGTH, overrides))
        self.top_k: int = int(self._opt("top_k", DEFAULT_TOP_K, overrides))
        self.batch_size: int = int(self._opt("batch_size", DEFAULT_BATCH_SIZE, overrides))
        self._model: Any = None

    def _opt(self, key: str, default: Any, overrides: Dict[str, Any]) -> Any:
        if key in overrides and overrides[key] is not None:
            return overrides[key]
        if self.cfg is not None:
            return cfg_get(self.cfg, "rag", "reranker", key, default=default)
        return default

    @property
    def is_active(self) -> bool:
        return self._model is not None

    def activate(self) -> None:
        """Load the cross-encoder, respecting the opt-in download policy."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RerankerError(
                "sentence-transformers is not installed. "
                "Install with: pip install 'sentence-transformers>=2.7'"
            ) from exc

        if not _model_is_cached(self.model_id) and not self.allow_download:
            raise RerankerError(
                f"Model '{self.model_id}' is not in the local cache and downloads "
                "are disabled. Set rag.reranker.allow_download: true in "
                "homie.config.yaml to fetch it once (~2.3 GB), or pre-download "
                f"it with: hf download {self.model_id}"
            )

        logger.info("Loading reranker %s on %s", self.model_id, self.device)
        try:
            self._model = CrossEncoder(
                self.model_id,
                max_length=self.max_length,
                device=self.device,
            )
        except Exception as exc:  # noqa: BLE001
            self._model = None
            raise RerankerError(f"Failed to load reranker: {exc}") from exc
        logger.info("Reranker ready (device=%s max_length=%d)", self.device, self.max_length)

    def deactivate(self) -> None:
        """Release the model and free device memory."""
        if self._model is None:
            return
        self._model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        gc.collect()
        logger.info("Reranker deactivated")

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_k: Optional[int] = None,
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[RerankResult]:
        """Re-score candidate documents against a query.

        Args:
            query: The user's search/question text (any supported language).
            documents: Candidate chunks, e.g. from a ChromaDB query.
            top_k: How many results to return (defaults to config top_k).
            metadatas: Optional per-document metadata carried through to results.

        Returns:
            Results sorted by descending relevance, truncated to top_k.

        Raises:
            RerankerError: If the plugin has not been activated.
            ValueError: If metadatas length does not match documents.
        """
        if self._model is None:
            raise RerankerError("Reranker not active. Call activate() first.")
        if not documents:
            return []
        if metadatas is not None and len(metadatas) != len(documents):
            raise ValueError("metadatas length must match documents length")

        k = top_k if top_k is not None else self.top_k
        pairs = [(query, doc) for doc in documents]
        scores = self._model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )

        results = [
            RerankResult(
                index=i,
                text=doc,
                score=float(score),
                metadata=dict(metadatas[i]) if metadatas else {},
            )
            for i, (doc, score) in enumerate(zip(documents, scores))
        ]
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    def rerank_chroma_result(
        self,
        query: str,
        chroma_result: Dict[str, Any],
        top_k: Optional[int] = None,
    ) -> List[RerankResult]:
        """Convenience adapter for a raw ChromaDB ``collection.query()`` result.

        Expects the standard Chroma shape where ``documents``/``metadatas``
        are lists-of-lists (one inner list per query text).
        """
        docs = (chroma_result.get("documents") or [[]])[0]
        metas = (chroma_result.get("metadatas") or [[]])[0] or [{} for _ in docs]
        return self.rerank(query, docs, top_k=top_k, metadatas=metas)


_PLUGIN: Optional[BGERerankerPlugin] = None


def register(cfg: Optional[HomieConfig] = None) -> Optional[BGERerankerPlugin]:
    """Entry point for HOMIE's plugin loader.

    Returns the activated plugin, or None when disabled or unavailable so
    the RAG pipeline falls back to vector-only retrieval.
    """
    global _PLUGIN
    if _PLUGIN is not None:
        return _PLUGIN

    enabled = True
    if cfg is not None:
        enabled = bool(cfg_get(cfg, "rag", "reranker", "enabled", default=False))
    if not enabled:
        logger.info("bge_reranker disabled in config; skipping")
        return None

    plugin = BGERerankerPlugin(cfg)
    try:
        plugin.activate()
    except RerankerError as exc:
        logger.warning("bge_reranker unavailable: %s", exc)
        return None
    _PLUGIN = plugin
    return plugin


def activate(cfg: Optional[HomieConfig] = None) -> Optional[BGERerankerPlugin]:
    """Module-level activation hook (HOMIE plugin contract)."""
    return register(cfg)


def deactivate() -> None:
    """Module-level deactivation hook (HOMIE plugin contract)."""
    global _PLUGIN
    if _PLUGIN is not None:
        _PLUGIN.deactivate()
        _PLUGIN = None


__all__ = [
    "BGERerankerPlugin",
    "RerankResult",
    "RerankerError",
    "register",
    "activate",
    "deactivate",
]
