"""BGE Reranker v2-m3 plugin for HOMIE.

Adds a local cross-encoder reranking stage on top of HOMIE's ChromaDB
retrieval. Vector search (bi-encoder cosine similarity) is fast but
lossy: it embeds the query and each document independently and can only
compare the two summaries. A cross-encoder such as
``BAAI/bge-reranker-v2-m3`` reads the query and each candidate passage
together in one forward pass and emits a much sharper relevance score,
so the final top-k context handed to the LLM is noticeably better.

Typical two-stage flow:
    1. ChromaDB ``query()`` returns ~20 candidates (fast, approximate).
    2. This plugin reranks those 20 and keeps the best 3-5 (slow, exact).

Usage::

    from homie.plugins.bge_reranker_plugin import register

    plugin = register(cfg)  # None if disabled in config
    if plugin:
        hits = plugin.rerank("how do I clean the apt cache?", candidates, top_k=4)
        plugin.deactivate()

Configuration (``homie.config.yaml``)::

    plugins:
      bge_reranker:
        enabled: true
        model_id: BAAI/bge-reranker-v2-m3
        device: cpu            # cpu | cuda | mps
        max_length: 512
        batch_size: 16
        allow_download: false  # opt-in: one-time fetch from Hugging Face

The plugin is offline-first: with ``allow_download: false`` (default)
it only loads weights already present in the local Hugging Face cache
and never touches the network.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from homie.config import HomieConfig, cfg_get

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "BAAI/bge-reranker-v2-m3"
DEFAULT_DEVICE = "cpu"
DEFAULT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 16


class RerankerError(Exception):
    """Raised when the reranker cannot be loaded or fails to score."""


@dataclass
class RerankedDocument:
    """A single candidate after reranking.

    Attributes:
        index: Position of the document in the original candidate list.
        text: The document text.
        score: Raw cross-encoder relevance logit (higher = more relevant).
        metadata: Optional metadata carried through from retrieval.
    """

    index: int
    text: str
    score: float
    metadata: Optional[Dict[str, Any]] = None


class BGERerankerPlugin:
    """Cross-encoder reranking stage for HOMIE's local RAG pipeline."""

    name = "bge_reranker"

    def __init__(self, cfg: Optional[HomieConfig] = None) -> None:
        self.cfg = cfg
        self.model_id: str = self._cfg("model_id", default=DEFAULT_MODEL_ID)
        self.device: str = self._cfg("device", default=DEFAULT_DEVICE)
        self.max_length: int = int(self._cfg("max_length", default=DEFAULT_MAX_LENGTH))
        self.batch_size: int = int(self._cfg("batch_size", default=DEFAULT_BATCH_SIZE))
        self.allow_download: bool = bool(self._cfg("allow_download", default=False))
        self._tokenizer: Any = None
        self._model: Any = None
        self._active = False

    def _cfg(self, key: str, default: Any = None) -> Any:
        if self.cfg is None:
            return default
        return cfg_get(self.cfg, "plugins", self.name, key, default=default)

    @property
    def is_active(self) -> bool:
        return self._active

    def activate(self) -> None:
        """Load tokenizer and model weights (idempotent).

        Raises:
            RerankerError: If dependencies are missing, or the weights are
                not in the local cache and ``allow_download`` is false.
        """
        if self._active:
            return
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RerankerError(
                "bge_reranker requires 'torch' and 'transformers'. "
                "Install with: pip install torch transformers"
            ) from exc

        local_only = not self.allow_download
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, local_files_only=local_only
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id, local_files_only=local_only
            )
        except OSError as exc:
            raise RerankerError(
                f"Model '{self.model_id}' is not in the local Hugging Face cache "
                "and allow_download is false. Set "
                "plugins.bge_reranker.allow_download: true for a one-time fetch, "
                "then switch it back off for fully offline operation."
            ) from exc

        self._model.to(self.device)
        self._model.eval()
        self._active = True
        LOGGER.info(
            "bge_reranker active (model=%s device=%s)", self.model_id, self.device
        )

    def deactivate(self) -> None:
        """Release the model and free memory (idempotent)."""
        if not self._active:
            return
        self._model = None
        self._tokenizer = None
        self._active = False
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        LOGGER.info("bge_reranker deactivated")

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_k: Optional[int] = None,
        metadatas: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    ) -> List[RerankedDocument]:
        """Score every document against the query and return them ranked.

        Args:
            query: The user's question or search string.
            documents: Candidate passages (e.g. from a ChromaDB query).
            top_k: If given, return only the best ``top_k`` documents.
            metadatas: Optional per-document metadata, same length as
                ``documents``; carried through to the results.

        Returns:
            Documents sorted by relevance, best first.

        Raises:
            RerankerError: If the plugin has not been activated.
            ValueError: If ``metadatas`` length does not match ``documents``.
        """
        if not self._active:
            raise RerankerError("Plugin is not active. Call activate() first.")
        docs = list(documents)
        if not docs:
            return []
        if metadatas is not None and len(metadatas) != len(docs):
            raise ValueError("metadatas must have the same length as documents")

        import torch

        scores: List[float] = []
        with torch.no_grad():
            for start in range(0, len(docs), self.batch_size):
                batch = docs[start : start + self.batch_size]
                pairs = [[query, doc] for doc in batch]
                inputs = self._tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self._model(**inputs).logits.view(-1).float()
                scores.extend(logits.cpu().tolist())

        ranked = [
            RerankedDocument(
                index=i,
                text=docs[i],
                score=scores[i],
                metadata=metadatas[i] if metadatas is not None else None,
            )
            for i in range(len(docs))
        ]
        ranked.sort(key=lambda r: r.score, reverse=True)
        if top_k is not None:
            ranked = ranked[: max(0, int(top_k))]
        return ranked

    def rerank_chroma_result(
        self,
        query: str,
        result: Dict[str, Any],
        top_k: Optional[int] = None,
    ) -> List[RerankedDocument]:
        """Rerank the first query's hits from a ChromaDB ``query()`` result.

        Accepts the dict shape returned by ``collection.query()``
        (``{"documents": [[...]], "metadatas": [[...]], ...}``) and
        returns the reranked documents for the first query.
        """
        documents = (result.get("documents") or [[]])[0]
        metadatas_all = result.get("metadatas") or [[]]
        metadatas = metadatas_all[0] if metadatas_all and metadatas_all[0] else None
        return self.rerank(query, documents, top_k=top_k, metadatas=metadatas)


def register(cfg: Optional[HomieConfig] = None) -> Optional[BGERerankerPlugin]:
    """Entry point for HOMIE's plugin loader.

    Returns an activated plugin instance when
    ``plugins.bge_reranker.enabled`` is true (or when no config is
    given), otherwise ``None``.
    """
    enabled = (
        True
        if cfg is None
        else bool(
            cfg_get(cfg, "plugins", BGERerankerPlugin.name, "enabled", default=False)
        )
    )
    if not enabled:
        LOGGER.debug("bge_reranker disabled via config")
        return None
    plugin = BGERerankerPlugin(cfg)
    plugin.activate()
    return plugin


__all__ = [
    "BGERerankerPlugin",
    "RerankedDocument",
    "RerankerError",
    "register",
]
