"""Cross-encoder reranker plugin for HOMIE RAG retrieval.

Integrates ``cross-encoder/ms-marco-MiniLM-L6-v2`` to perform two-stage
retrieval: a fast bi-encoder (e.g. the existing ChromaDB embeddings)
produces a candidate set, then this cross-encoder reranks the top-N
candidates for substantially higher relevance.

The model is small (~22M params, ~80MB) and runs comfortably on CPU,
making it well suited for a local-first assistant. The first call
downloads the weights into the local Hugging Face cache; subsequent
calls run fully offline. Network access can be disabled via the
``allow_download=False`` flag for strict air-gapped operation.

Configuration (all optional, read under ``plugins.cross_encoder_reranker``)::

    plugins:
      cross_encoder_reranker:
        model_name: cross-encoder/ms-marco-MiniLM-L6-v2
        device: cpu          # cpu | cuda | mps
        batch_size: 32
        allow_download: true
        warmup: false

Usage::

    from homie.plugins.cross_encoder_reranker import activate

    reranker = activate(cfg)
    ranked = reranker.rerank("how do I clean disk cache?", candidates, top_k=5)
    for hit in ranked:
        print(hit.score, hit.document)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from homie.config import HomieConfig, cfg_get

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEFAULT_TOP_K = 5
DEFAULT_BATCH_SIZE = 32


class RerankerError(Exception):
    """Raised when the cross-encoder reranker cannot fulfil a request."""


@dataclass
class RerankResult:
    """A single reranked document with its relevance score."""

    document: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    original_index: int = -1


class CrossEncoderReranker:
    """Two-stage retrieval reranker backed by a local cross-encoder model.

    The reranker accepts a query plus a list of candidate documents
    (typically the top-N hits from a bi-encoder retriever such as
    ChromaDB) and returns them ordered by cross-encoder relevance.

    The underlying ``sentence_transformers.CrossEncoder`` model is
    loaded lazily on first use so plugin activation stays cheap and
    fast.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        allow_download: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self.allow_download = allow_download
        self._model: Optional[Any] = None

    def _load_model(self) -> Any:
        """Load the ``sentence-transformers`` CrossEncoder lazily."""
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RerankerError(
                "cross-encoder reranker requires 'sentence-transformers'. "
                "Install with: pip install sentence-transformers"
            ) from exc

        if not self.allow_download:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        logger.info(
            "Loading cross-encoder model %s (device=%s, offline=%s)",
            self.model_name,
            self.device or "auto",
            not self.allow_download,
        )
        try:
            self._model = CrossEncoder(self.model_name, device=self.device)
        except Exception as exc:  # noqa: BLE001
            raise RerankerError(
                f"Failed to load cross-encoder model '{self.model_name}': {exc}"
            ) from exc
        return self._model

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_k: int = DEFAULT_TOP_K,
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[RerankResult]:
        """Rerank candidate documents by cross-encoder relevance.

        Args:
            query: The user query.
            documents: Candidate document strings produced by an
                upstream retriever (e.g. ChromaDB top-N neighbours).
            top_k: Maximum number of reranked results to return.
            metadatas: Optional metadata dicts aligned with ``documents``.

        Returns:
            A list of :class:`RerankResult` sorted by descending score.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not documents:
            return []

        model = self._load_model()
        pairs: List[Tuple[str, str]] = [(query, doc) for doc in documents]
        try:
            scores = model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise RerankerError(f"Cross-encoder inference failed: {exc}") from exc

        meta_list = list(metadatas) if metadatas is not None else [{} for _ in documents]
        ranked = [
            RerankResult(
                document=doc,
                score=float(score),
                metadata=meta_list[i] if i < len(meta_list) else {},
                original_index=i,
            )
            for i, (doc, score) in enumerate(zip(documents, scores))
        ]
        ranked.sort(key=lambda r: r.score, reverse=True)
        return ranked[: max(1, int(top_k))]

    def score_pairs(self, pairs: Iterable[Tuple[str, str]]) -> List[float]:
        """Score arbitrary ``(query, document)`` pairs without sorting."""
        model = self._load_model()
        try:
            scores = model.predict(
                list(pairs),
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise RerankerError(f"Cross-encoder inference failed: {exc}") from exc
        return [float(s) for s in scores]

    def warmup(self) -> None:
        """Eagerly load the model and run a tiny inference to warm caches."""
        model = self._load_model()
        try:
            model.predict([("warmup", "warmup")], show_progress_bar=False)
        except Exception:  # noqa: BLE001
            logger.exception("Cross-encoder warmup inference failed")

    def close(self) -> None:
        """Release model references so memory can be reclaimed."""
        self._model = None


_INSTANCE: Optional[CrossEncoderReranker] = None


def activate(cfg: Optional[HomieConfig] = None) -> CrossEncoderReranker:
    """Activate the reranker plugin and return the singleton instance.

    Reading from ``cfg`` is best-effort; if ``cfg`` is ``None`` or the
    keys are absent, sensible defaults are used. The model itself is
    loaded lazily unless ``plugins.cross_encoder_reranker.warmup`` is
    truthy.
    """
    global _INSTANCE
    if _INSTANCE is not None:
        return _INSTANCE

    model_name = DEFAULT_MODEL_NAME
    device: Optional[str] = None
    batch_size = DEFAULT_BATCH_SIZE
    allow_download = True
    do_warmup = False

    if cfg is not None:
        model_name = cfg_get(
            cfg, "plugins", "cross_encoder_reranker", "model_name",
            default=DEFAULT_MODEL_NAME,
        )
        device = cfg_get(cfg, "plugins", "cross_encoder_reranker", "device", default=None)
        batch_size = int(
            cfg_get(cfg, "plugins", "cross_encoder_reranker", "batch_size",
                    default=DEFAULT_BATCH_SIZE)
        )
        allow_download = bool(
            cfg_get(cfg, "plugins", "cross_encoder_reranker", "allow_download",
                    default=True)
        )
        do_warmup = bool(
            cfg_get(cfg, "plugins", "cross_encoder_reranker", "warmup", default=False)
        )

    _INSTANCE = CrossEncoderReranker(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        allow_download=allow_download,
    )

    if do_warmup:
        try:
            _INSTANCE.warmup()
        except RerankerError:
            logger.exception("Cross-encoder warmup failed; will retry on first use")

    logger.info(
        "Cross-encoder reranker activated (model=%s, allow_download=%s)",
        model_name,
        allow_download,
    )
    return _INSTANCE


def deactivate() -> None:
    """Release the reranker instance and free model memory."""
    global _INSTANCE
    if _INSTANCE is None:
        return
    try:
        _INSTANCE.close()
    finally:
        _INSTANCE = None
    logger.info("Cross-encoder reranker deactivated")


def register(cfg: Optional[HomieConfig] = None) -> CrossEncoderReranker:
    """Alias for :func:`activate` matching the alternate plugin convention."""
    return activate(cfg)


__all__ = [
    "CrossEncoderReranker",
    "RerankResult",
    "RerankerError",
    "DEFAULT_MODEL_NAME",
    "activate",
    "deactivate",
    "register",
]
