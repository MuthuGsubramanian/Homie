"""HOMIE reranker plugin using cross-encoder/ms-marco-MiniLM-L6-v2.

Provides two-stage retrieval for HOMIE's local RAG pipeline:

1. ChromaDB returns top-K candidate documents using bi-encoder embeddings
   (fast, approximate semantic match).
2. This plugin re-scores every (query, doc) pair with a cross-encoder, which
   jointly attends to query and document and produces a much sharper
   relevance signal. The top-N from this stage is what the LLM actually
   sees in the prompt.

The ms-marco-MiniLM-L6-v2 model is ~80MB and runs comfortably on CPU, so it
fits HOMIE's local-first, no-cloud constraint. The first call downloads the
model weights from Hugging Face into the local sentence-transformers cache;
every subsequent call runs fully offline.

Integration sketch (called from HOMIE's RAG layer)::

    from homie.plugins.reranker_cross_encoder import get_instance

    candidates = chroma_collection.query(query, n_results=20)
    reranked = get_instance().rerank(
        query=query,
        documents=candidates["documents"][0],
        metadatas=candidates["metadatas"][0],
        top_n=5,
    )
    context = "\n\n".join(d.text for d in reranked)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEFAULT_TOP_N = 5
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LENGTH = 512


class RerankerError(Exception):
    """Raised when reranker model load or inference fails."""


@dataclass
class RerankedDocument:
    """A document paired with its cross-encoder relevance score."""

    text: str
    score: float
    metadata: Dict[str, Any]
    original_index: int


class CrossEncoderReranker:
    """Two-stage retrieval reranker for HOMIE's local RAG pipeline.

    The underlying model is lazy-loaded on the first ``rerank`` call so that
    importing this module is cheap and HOMIE startup is not blocked on model
    weights. All inference runs locally; no network traffic occurs after the
    initial weight download.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        cache_dir: Optional[Path] = None,
        device: Optional[str] = None,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.device = device
        self.max_length = max_length
        self._model: Optional[Any] = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RerankerError(
                "sentence-transformers is required for the reranker plugin. "
                "Install with: pip install sentence-transformers"
            ) from exc

        kwargs: Dict[str, Any] = {"max_length": self.max_length}
        if self.cache_dir is not None:
            kwargs["cache_folder"] = str(self.cache_dir)
        if self.device is not None:
            kwargs["device"] = self.device

        logger.info("Loading cross-encoder reranker %s", self.model_name)
        try:
            self._model = CrossEncoder(self.model_name, **kwargs)
        except Exception as exc:  # noqa: BLE001
            raise RerankerError(
                f"Failed to load reranker model '{self.model_name}': {exc}"
            ) from exc

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
        top_n: int = DEFAULT_TOP_N,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> List[RerankedDocument]:
        """Re-score (query, document) pairs and return the top_n ranked docs.

        Args:
            query: User query string.
            documents: First-stage candidate documents (usually from ChromaDB).
            metadatas: Optional per-document metadata; defaults to empty dicts.
            top_n: Number of documents to keep after reranking.
            batch_size: Pairs per forward pass; tune for memory vs. throughput.

        Returns:
            Documents sorted by descending relevance score, truncated to top_n.
        """
        if not query or not query.strip():
            raise ValueError("query must be non-empty")
        if not documents:
            return []

        if metadatas is None:
            metadatas = [{} for _ in documents]
        if len(metadatas) != len(documents):
            raise ValueError("metadatas length must match documents length")
        if top_n <= 0:
            raise ValueError("top_n must be positive")

        self._load_model()
        assert self._model is not None  # for type checkers

        pairs: List[Tuple[str, str]] = [(query, doc) for doc in documents]
        try:
            scores = self._model.predict(
                pairs,
                batch_size=batch_size,
                show_progress_bar=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise RerankerError(f"Reranker inference failed: {exc}") from exc

        ranked = [
            RerankedDocument(
                text=doc,
                score=float(score),
                metadata=dict(meta),
                original_index=idx,
            )
            for idx, (doc, meta, score) in enumerate(zip(documents, metadatas, scores))
        ]
        ranked.sort(key=lambda d: d.score, reverse=True)
        return ranked[:top_n]

    def score_pairs(
        self,
        pairs: Sequence[Tuple[str, str]],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> List[float]:
        """Return raw cross-encoder scores for arbitrary (query, doc) pairs."""
        if not pairs:
            return []
        self._load_model()
        assert self._model is not None
        try:
            scores = self._model.predict(
                list(pairs),
                batch_size=batch_size,
                show_progress_bar=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise RerankerError(f"Reranker inference failed: {exc}") from exc
        return [float(s) for s in scores]

    def unload(self) -> None:
        """Drop the in-memory model so the process can reclaim RAM."""
        self._model = None


_PLUGIN_INSTANCE: Optional[CrossEncoderReranker] = None


def activate(cfg: Optional[Any] = None) -> CrossEncoderReranker:
    """Initialize the reranker plugin. Called by HOMIE's plugin loader.

    If a ``HomieConfig`` is provided, configuration is read from the
    ``plugins.reranker`` section::

        plugins:
          reranker:
            model: cross-encoder/ms-marco-MiniLM-L6-v2
            cache_dir: ~/.homie/models
            device: cpu          # or 'cuda', 'mps'
            max_length: 512

    Returns the singleton plugin instance.
    """
    global _PLUGIN_INSTANCE

    model_name = DEFAULT_MODEL_NAME
    cache_dir: Optional[Path] = None
    device: Optional[str] = None
    max_length = DEFAULT_MAX_LENGTH

    if cfg is not None:
        try:
            from homie.config import cfg_get

            model_name = cfg_get(
                cfg, "plugins", "reranker", "model", default=DEFAULT_MODEL_NAME
            )
            cache_raw = cfg_get(cfg, "plugins", "reranker", "cache_dir", default=None)
            if cache_raw:
                cache_dir = Path(cache_raw).expanduser()
            device = cfg_get(cfg, "plugins", "reranker", "device", default=None)
            max_length = int(
                cfg_get(
                    cfg, "plugins", "reranker", "max_length", default=DEFAULT_MAX_LENGTH
                )
            )
        except Exception:  # noqa: BLE001
            logger.debug("Reranker config missing or invalid; using defaults")

    _PLUGIN_INSTANCE = CrossEncoderReranker(
        model_name=model_name,
        cache_dir=cache_dir,
        device=device,
        max_length=max_length,
    )
    logger.info(
        "Reranker plugin activated (model=%s, device=%s)", model_name, device or "auto"
    )
    return _PLUGIN_INSTANCE


def deactivate() -> None:
    """Release the reranker and free GPU/CPU memory."""
    global _PLUGIN_INSTANCE
    if _PLUGIN_INSTANCE is not None:
        _PLUGIN_INSTANCE.unload()
        _PLUGIN_INSTANCE = None
        logger.info("Reranker plugin deactivated")


def get_instance() -> CrossEncoderReranker:
    """Return the active plugin instance, activating with defaults if needed."""
    if _PLUGIN_INSTANCE is None:
        return activate()
    return _PLUGIN_INSTANCE


def register() -> Dict[str, Any]:
    """Plugin manifest used by HOMIE's plugin registry."""
    return {
        "name": "reranker_cross_encoder",
        "version": "1.0.0",
        "description": "Cross-encoder reranker for two-stage local RAG retrieval",
        "capabilities": ["rag.rerank"],
        "requires": ["sentence-transformers"],
        "network": "first-run-only",
        "activate": activate,
        "deactivate": deactivate,
    }


__all__ = [
    "CrossEncoderReranker",
    "RerankedDocument",
    "RerankerError",
    "activate",
    "deactivate",
    "get_instance",
    "register",
    "DEFAULT_MODEL_NAME",
]
