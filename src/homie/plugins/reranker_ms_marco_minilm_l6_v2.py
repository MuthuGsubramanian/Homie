"""HOMIE reranker plugin using cross-encoder/ms-marco-MiniLM-L6-v2.

This plugin adds a second-stage reranker to HOMIE's RAG pipeline. After an
initial bi-encoder embedding search returns a coarse candidate set (e.g.
top-50 from ChromaDB), this cross-encoder scores each (query, passage) pair
jointly and reorders by relevance. Two-stage retrieval (embed + rerank)
typically lifts top-k precision substantially with modest latency cost.

Model: cross-encoder/ms-marco-MiniLM-L6-v2 (~22M params, MS MARCO trained).
Runs fully local on CPU; GPU optional. Model files are cached under the
HuggingFace cache directory after first download (one-time network access);
set HF_HUB_OFFLINE=1 or pre-populate the cache for fully offline operation.

Integration example:

    from homie.plugins.reranker_ms_marco_minilm_l6_v2 import register
    reranker = register(cfg)
    top_docs = reranker.rerank(query, candidate_docs, top_k=5)

Or via the activate()/deactivate() lifecycle hooks for plugin hosts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from homie.config import HomieConfig, cfg_get

logger = logging.getLogger(__name__)

PLUGIN_NAME = "reranker_ms_marco_minilm_l6_v2"
DEFAULT_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEFAULT_TOP_K = 5
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LENGTH = 512


@dataclass
class RerankedDocument:
    """A candidate document paired with its cross-encoder relevance score."""

    text: str
    score: float
    original_rank: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class MsMarcoRerankerPlugin:
    """Cross-encoder reranker for HOMIE's local RAG pipeline.

    Uses ``cross-encoder/ms-marco-MiniLM-L6-v2`` to jointly score
    (query, passage) pairs and reorder candidates from the embedding-based
    first-stage retrieval. The model is lazy-loaded on first use to keep
    HOMIE startup cost low; call :meth:`activate` to preload eagerly.
    """

    name: str = PLUGIN_NAME

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = DEFAULT_MAX_LENGTH,
        cache_folder: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.cache_folder = cache_folder
        self._model: Any = None

    @classmethod
    def from_config(cls, cfg: HomieConfig) -> "MsMarcoRerankerPlugin":
        """Construct the plugin from HOMIE config under ``rag.reranker.*``."""
        return cls(
            model_id=cfg_get(cfg, "rag", "reranker", "model_id", default=DEFAULT_MODEL_ID),
            device=cfg_get(cfg, "rag", "reranker", "device", default=None),
            batch_size=int(cfg_get(cfg, "rag", "reranker", "batch_size", default=DEFAULT_BATCH_SIZE)),
            max_length=int(cfg_get(cfg, "rag", "reranker", "max_length", default=DEFAULT_MAX_LENGTH)),
            cache_folder=cfg_get(cfg, "rag", "reranker", "cache_folder", default=None),
        )

    def _load_model(self) -> Any:
        """Lazy-load the cross-encoder. Raises if sentence-transformers is missing."""
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for the reranker plugin. "
                "Install with: pip install sentence-transformers"
            ) from exc

        logger.info("Loading cross-encoder model %s (device=%s)", self.model_id, self.device or "auto")
        kwargs: Dict[str, Any] = {"max_length": self.max_length}
        if self.device:
            kwargs["device"] = self.device
        if self.cache_folder:
            kwargs["cache_folder"] = self.cache_folder
        self._model = CrossEncoder(self.model_id, **kwargs)
        return self._model

    def activate(self) -> None:
        """Lifecycle hook: preload the model so the first query is fast."""
        self._load_model()
        logger.info("Plugin %s activated", self.name)

    def deactivate(self) -> None:
        """Lifecycle hook: drop the model so memory can be reclaimed."""
        self._model = None
        logger.info("Plugin %s deactivated", self.name)

    def score(self, query: str, passages: Sequence[str]) -> List[float]:
        """Score each (query, passage) pair and return raw relevance logits."""
        if not passages:
            return []
        model = self._load_model()
        pairs: List[Tuple[str, str]] = [(query, p) for p in passages]
        raw = model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [float(s) for s in raw]

    def rerank(
        self,
        query: str,
        candidates: Sequence[Any],
        top_k: int = DEFAULT_TOP_K,
        text_key: str = "text",
    ) -> List[RerankedDocument]:
        """Rerank candidate documents from first-stage retrieval.

        ``candidates`` may be plain strings or dicts containing the passage text
        under ``text_key`` (default ``"text"``). Remaining dict keys are preserved
        in ``RerankedDocument.metadata``. Returns the top ``top_k`` candidates
        sorted by descending relevance.
        """
        if not candidates:
            return []

        passages: List[str] = []
        metadata_list: List[Dict[str, Any]] = []
        for cand in candidates:
            if isinstance(cand, str):
                passages.append(cand)
                metadata_list.append({})
            elif isinstance(cand, dict):
                passages.append(str(cand.get(text_key, "")))
                metadata_list.append({k: v for k, v in cand.items() if k != text_key})
            else:
                passages.append(str(cand))
                metadata_list.append({})

        scores = self.score(query, passages)
        order = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)

        results: List[RerankedDocument] = []
        limit = max(0, int(top_k)) if top_k is not None else len(order)
        for original_index, score in order[:limit]:
            results.append(
                RerankedDocument(
                    text=passages[original_index],
                    score=score,
                    original_rank=original_index,
                    metadata=metadata_list[original_index],
                )
            )
        return results


_PLUGIN_INSTANCE: Optional[MsMarcoRerankerPlugin] = None


def activate(cfg: Optional[HomieConfig] = None) -> MsMarcoRerankerPlugin:
    """Activate the reranker plugin singleton and return it."""
    global _PLUGIN_INSTANCE
    if _PLUGIN_INSTANCE is None:
        _PLUGIN_INSTANCE = (
            MsMarcoRerankerPlugin.from_config(cfg) if cfg is not None else MsMarcoRerankerPlugin()
        )
    _PLUGIN_INSTANCE.activate()
    return _PLUGIN_INSTANCE


def deactivate() -> None:
    """Deactivate the plugin singleton and release its model."""
    global _PLUGIN_INSTANCE
    if _PLUGIN_INSTANCE is not None:
        _PLUGIN_INSTANCE.deactivate()
        _PLUGIN_INSTANCE = None


def register(cfg: Optional[HomieConfig] = None) -> MsMarcoRerankerPlugin:
    """HOMIE plugin entry point: alias for :func:`activate`."""
    return activate(cfg)


__all__ = [
    "MsMarcoRerankerPlugin",
    "RerankedDocument",
    "activate",
    "deactivate",
    "register",
    "PLUGIN_NAME",
]
