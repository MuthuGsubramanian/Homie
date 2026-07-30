"""BGE Reranker plugin for HOMIE.

Adds a local retrieval-reranking stage backed by ``BAAI/bge-reranker-v2-m3``,
a multilingual cross-encoder that scores (query, document) pairs directly.
It is designed to sit between HOMIE's ChromaDB retrieval step and the
llama.cpp generation step in a two-stage RAG pipeline:

    1. ChromaDB (bge-m3 or any embedder) returns the top-N candidates fast.
    2. This plugin rescores those N candidates against the query and keeps
       the top-K, which markedly improves answer grounding, especially for
       multilingual corpora.

Privacy / safety:
    - Inference is fully local (CPU by default, CUDA if requested).
    - The plugin NEVER downloads weights unless ``allow_download=True`` is
      passed explicitly (opt-in network). Otherwise it requires a local
      model directory, e.g. one pre-fetched with:
          hf download BAAI/bge-reranker-v2-m3 --local-dir models/bge-reranker-v2-m3

Dependencies (install once, offline afterwards):
    pip install sentence-transformers

Example:
    from bge_reranker_v2_m3 import BGERerankerPlugin

    plugin = BGERerankerPlugin(model_path="models/bge-reranker-v2-m3")
    plugin.activate()
    ranked = plugin.rerank(
        query="How do I free disk space on the msi box?",
        documents=[doc1, doc2, doc3],
        top_k=2,
    )
    plugin.deactivate()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "BAAI/bge-reranker-v2-m3"
DEFAULT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 16


class RerankerError(Exception):
    """Raised when the reranker cannot load or score."""


@dataclass
class RerankResult:
    """A single reranked document.

    Attributes:
        document: The original document text.
        score: Relevance score from the cross-encoder (higher = more relevant).
        index: Position of the document in the original candidate list.
        metadata: Optional metadata carried through from retrieval (e.g. a
            ChromaDB metadata dict), untouched by the reranker.
    """

    document: str
    score: float
    index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class BGERerankerPlugin:
    """Multilingual cross-encoder reranking stage for HOMIE retrieval.

    The model is loaded lazily on :meth:`activate` and released on
    :meth:`deactivate` so HOMIE only pays the memory cost (~2.2 GB fp32,
    ~1.1 GB fp16 on GPU) while the stage is in use.
    """

    name = "bge_reranker_v2_m3"
    version = "1.0.0"

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cpu",
        max_length: int = DEFAULT_MAX_LENGTH,
        batch_size: int = DEFAULT_BATCH_SIZE,
        allow_download: bool = False,
    ) -> None:
        """Configure the reranker without loading any weights yet.

        Args:
            model_path: Local directory containing the model weights. Required
                unless ``allow_download=True``.
            model_id: Hugging Face repo id, used only when downloading is
                explicitly allowed.
            device: "cpu" or "cuda". CPU is fine for reranking <=50 docs.
            max_length: Token cap per (query, document) pair; pairs longer
                than this are truncated by the tokenizer.
            batch_size: Pairs scored per forward pass.
            allow_download: Opt-in switch for fetching weights from the
                Hugging Face Hub. Defaults to False (fully offline).
        """
        self.model_path = Path(model_path) if model_path else None
        self.model_id = model_id
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.allow_download = allow_download
        self._model: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @property
    def is_active(self) -> bool:
        return self._model is not None

    def activate(self) -> None:
        """Load the cross-encoder into memory (idempotent)."""
        if self._model is not None:
            logger.debug("%s already active", self.name)
            return

        source = self._resolve_model_source()
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RerankerError(
                "sentence-transformers is required for the BGE reranker plugin. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        logger.info("Loading reranker %s on %s", source, self.device)
        try:
            self._model = CrossEncoder(
                str(source),
                max_length=self.max_length,
                device=self.device,
            )
        except Exception as exc:  # noqa: BLE001
            self._model = None
            raise RerankerError(f"Failed to load reranker from '{source}': {exc}") from exc
        logger.info("%s activated", self.name)

    def deactivate(self) -> None:
        """Release the model and free memory (idempotent)."""
        if self._model is None:
            return
        self._model = None
        try:
            import torch

            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover - torch always ships with ST
            pass
        logger.info("%s deactivated", self.name)

    def _resolve_model_source(self) -> str | Path:
        """Pick a local path, or the hub id only when downloads are opted in."""
        if self.model_path is not None:
            if not self.model_path.exists():
                raise RerankerError(
                    f"Local model path '{self.model_path}' does not exist. "
                    f"Pre-fetch it with: hf download {self.model_id} "
                    f"--local-dir {self.model_path}"
                )
            return self.model_path
        if self.allow_download:
            return self.model_id
        raise RerankerError(
            "No local model_path configured and allow_download=False. "
            "Either pass model_path to a pre-downloaded copy of "
            f"{self.model_id}, or opt in with allow_download=True."
        )

    # ------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------
    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_k: int = 5,
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[RerankResult]:
        """Score ``documents`` against ``query`` and return the best ``top_k``.

        Args:
            query: The user question (any of the ~100 languages bge-m3 covers).
            documents: Candidate passages from first-stage retrieval.
            top_k: How many results to keep after reranking.
            metadatas: Optional per-document metadata, carried through to the
                results in the same order as ``documents``.

        Returns:
            Results sorted by descending relevance score.
        """
        if self._model is None:
            raise RerankerError("Plugin is not active. Call activate() first.")
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        if metadatas is not None and len(metadatas) != len(documents):
            raise ValueError("metadatas must match documents in length")
        if not documents:
            return []

        pairs = [(query, doc) for doc in documents]
        scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        results = [
            RerankResult(
                document=doc,
                score=float(score),
                index=i,
                metadata=dict(metadatas[i]) if metadatas else {},
            )
            for i, (doc, score) in enumerate(zip(documents, scores))
        ]
        results.sort(key=lambda r: r.score, reverse=True)
        return results[: max(top_k, 0)]

    def rerank_chroma_results(
        self,
        query: str,
        chroma_result: Dict[str, Any],
        top_k: int = 5,
    ) -> List[RerankResult]:
        """Rerank a raw ChromaDB ``query()`` response for a single query.

        Accepts the standard ChromaDB shape::

            {"documents": [[...]], "metadatas": [[...]], "ids": [[...]]}

        and returns reranked :class:`RerankResult` objects with the original
        metadata (including the Chroma id under ``metadata['chroma_id']``).
        """
        docs_nested = chroma_result.get("documents") or [[]]
        metas_nested = chroma_result.get("metadatas") or [[]]
        ids_nested = chroma_result.get("ids") or [[]]

        documents: List[str] = list(docs_nested[0]) if docs_nested else []
        metadatas: List[Dict[str, Any]] = [
            dict(m) if m else {} for m in (metas_nested[0] if metas_nested else [])
        ]
        while len(metadatas) < len(documents):
            metadatas.append({})
        ids: List[Any] = list(ids_nested[0]) if ids_nested else []
        for i, chroma_id in enumerate(ids[: len(metadatas)]):
            metadatas[i]["chroma_id"] = chroma_id

        return self.rerank(query, documents, top_k=top_k, metadatas=metadatas)


def register(config: Optional[Dict[str, Any]] = None) -> BGERerankerPlugin:
    """Build a plugin instance from a HOMIE-style config mapping.

    Expected (all optional) keys under ``config``::

        reranker:
          model_path: models/bge-reranker-v2-m3
          device: cpu
          top_k: 5
          max_length: 512
          batch_size: 16
          allow_download: false
    """
    cfg = (config or {}).get("reranker", {}) or {}
    return BGERerankerPlugin(
        model_path=cfg.get("model_path"),
        model_id=cfg.get("model_id", DEFAULT_MODEL_ID),
        device=cfg.get("device", "cpu"),
        max_length=int(cfg.get("max_length", DEFAULT_MAX_LENGTH)),
        batch_size=int(cfg.get("batch_size", DEFAULT_BATCH_SIZE)),
        allow_download=bool(cfg.get("allow_download", False)),
    )


__all__ = [
    "BGERerankerPlugin",
    "RerankResult",
    "RerankerError",
    "register",
]
