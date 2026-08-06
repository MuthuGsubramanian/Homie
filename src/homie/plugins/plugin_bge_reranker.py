"""BGE reranker plugin for HOMIE.

Second-stage reranking for HOMIE's local RAG pipeline using
BAAI/bge-reranker-v2-m3, a multilingual cross-encoder built on
XLM-RoBERTa. After ChromaDB returns candidate chunks by vector
similarity, this plugin scores each (query, chunk) pair jointly and
reorders them so the truly relevant chunks reach the LLM context first.

Everything runs locally. By default the model is loaded only from the
local Hugging Face cache or an explicit directory path; downloading is
opt-in via config (``rerank.allow_download: true``) because HOMIE
plugins must not touch the network unless the user asks.

Config (homie.config.yaml)::

    rerank:
      enabled: true
      model: BAAI/bge-reranker-v2-m3   # or a local directory path
      device: cpu                      # cpu | cuda | mps
      top_k: 5
      batch_size: 16
      max_length: 512
      allow_download: false

Requires the optional dependency ``sentence-transformers``
(``pip install sentence-transformers``).
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from homie.config import HomieConfig, cfg_get

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_TOP_K = 5
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_LENGTH = 512


class RerankerError(Exception):
    """Raised when the reranker cannot load or score documents."""


@dataclass
class RerankedDocument:
    """A single reranked document with its relevance score."""

    index: int  # position in the original candidate list
    text: str
    score: float  # sigmoid-normalised relevance in [0, 1]
    metadata: Dict[str, Any] = field(default_factory=dict)


class BGERerankerPlugin:
    """Cross-encoder reranker for HOMIE's RAG retrieval stage."""

    name = "bge_reranker"
    version = "1.0.0"

    def __init__(self, cfg: Optional[HomieConfig] = None) -> None:
        self.cfg = cfg
        self._model = None  # lazy: sentence_transformers.CrossEncoder
        self._active = False

    def _cfg(self, key: str, default: Any = None) -> Any:
        if self.cfg is None:
            return default
        return cfg_get(self.cfg, "rerank", key, default=default)

    @property
    def is_active(self) -> bool:
        return self._active and self._model is not None

    def activate(self) -> None:
        """Load the cross-encoder model (local cache first)."""
        if self._active:
            return

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RerankerError(
                "sentence-transformers is required for the BGE reranker. "
                "Install with: pip install sentence-transformers"
            ) from exc

        model_name = self._cfg("model", default=DEFAULT_MODEL)
        device = self._cfg("device", default="cpu")
        max_length = int(self._cfg("max_length", default=DEFAULT_MAX_LENGTH))
        allow_download = bool(self._cfg("allow_download", default=False))

        offline_before = os.environ.get("HF_HUB_OFFLINE")
        if not allow_download and not os.path.isdir(str(model_name)):
            # Cache-only resolution: a cold cache must fail loudly instead
            # of silently reaching the network.
            os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            self._model = CrossEncoder(
                str(model_name), max_length=max_length, device=device
            )
        except Exception as exc:  # noqa: BLE001
            raise RerankerError(
                f"Could not load reranker '{model_name}'. If it is not cached "
                "locally, set rerank.allow_download: true or point "
                "rerank.model at a local directory."
            ) from exc
        finally:
            if offline_before is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = offline_before

        self._active = True
        logger.info("BGE reranker active | model=%s device=%s", model_name, device)

    def deactivate(self) -> None:
        """Release the model and free device memory (best-effort)."""
        self._model = None
        self._active = False
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        logger.info("BGE reranker deactivated")

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_k: Optional[int] = None,
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[RerankedDocument]:
        """Score and reorder ``documents`` by relevance to ``query``.

        Returns the ``top_k`` most relevant documents, best first.
        Scores are sigmoid-normalised logits in [0, 1].
        """
        if not self.is_active:
            raise RerankerError("Reranker not active. Call activate() first.")
        if not query or not query.strip():
            raise RerankerError("Query must be a non-empty string.")
        if not documents:
            return []
        if metadatas is not None and len(metadatas) != len(documents):
            raise RerankerError("metadatas length must match documents length.")

        k = int(top_k if top_k is not None else self._cfg("top_k", default=DEFAULT_TOP_K))
        batch_size = int(self._cfg("batch_size", default=DEFAULT_BATCH_SIZE))

        pairs = [(query, doc) for doc in documents]
        logits = self._model.predict(
            pairs, batch_size=batch_size, show_progress_bar=False
        )

        ranked = [
            RerankedDocument(
                index=i,
                text=documents[i],
                score=_sigmoid(float(logit)),
                metadata=dict(metadatas[i]) if metadatas is not None else {},
            )
            for i, logit in enumerate(logits)
        ]
        ranked.sort(key=lambda d: d.score, reverse=True)
        return ranked[: max(k, 1)]

    def rerank_chroma_result(
        self,
        query: str,
        chroma_result: Dict[str, Any],
        top_k: Optional[int] = None,
    ) -> List[RerankedDocument]:
        """Rerank the output of a ChromaDB ``collection.query`` call.

        Expects the standard Chroma result shape where ``documents`` and
        ``metadatas`` are lists-of-lists (one inner list per query text).
        Only the first query's results are reranked.
        """
        docs_outer = chroma_result.get("documents") or []
        metas_outer = chroma_result.get("metadatas") or []
        documents: List[str] = list(docs_outer[0]) if docs_outer else []
        metadatas = list(metas_outer[0]) if metas_outer else None
        return self.rerank(query, documents, top_k=top_k, metadatas=metadatas)


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


_plugin: Optional[BGERerankerPlugin] = None


def activate(cfg: Optional[HomieConfig] = None) -> BGERerankerPlugin:
    """Create (or reuse) and activate the shared reranker instance."""
    global _plugin
    if _plugin is None:
        _plugin = BGERerankerPlugin(cfg)
    _plugin.activate()
    return _plugin


def deactivate() -> None:
    """Deactivate and drop the shared reranker instance."""
    global _plugin
    if _plugin is not None:
        _plugin.deactivate()
        _plugin = None


def register(cfg: Optional[HomieConfig] = None) -> Dict[str, Any]:
    """Describe the plugin for HOMIE's plugin registry."""
    return {
        "name": BGERerankerPlugin.name,
        "version": BGERerankerPlugin.version,
        "kind": "rag.reranker",
        "activate": lambda: activate(cfg),
        "deactivate": deactivate,
    }


__all__ = [
    "BGERerankerPlugin",
    "RerankedDocument",
    "RerankerError",
    "activate",
    "deactivate",
    "register",
]
