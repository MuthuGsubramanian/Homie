"""Cross-encoder reranker plugin for HOMIE RAG retrieval.

This plugin wraps the ``cross-encoder/ms-marco-MiniLM-L6-v2`` model from
Hugging Face to rerank candidate documents returned by HOMIE's primary
retriever (e.g. ChromaDB bi-encoder search). Reranking with a cross-encoder
typically yields a large precision-at-k boost over pure bi-encoder retrieval
because the model jointly attends to the query and each candidate.

Usage pattern inside HOMIE:
    1. The RAG layer fetches ``top_k`` candidates from ChromaDB (cheap).
    2. This plugin reranks them down to ``top_n`` (expensive but accurate).
    3. The reranked passages are passed to the local LLM for grounding.

The plugin is fully local. The model is downloaded once via
``sentence-transformers``/``huggingface_hub`` on first use and then cached
under ``~/.cache/huggingface``. No outbound traffic occurs after the cache
is warm. Network access during the initial download is opt-in via the
``allow_download`` flag.

Integration points:
    * ``activate(registry, cfg)`` is called by HOMIE's plugin loader.
    * ``deactivate()`` releases the model from memory.
    * The plugin exposes a ``rerank(query, documents, top_n)`` method that
      the RAG pipeline can call directly, and registers itself under the
      ``rerankers`` namespace of the plugin registry when one is provided.

Configuration (read from ``homie.config.yaml`` under ``plugins.reranker``):
    model_name:     str  (default: 'cross-encoder/ms-marco-MiniLM-L6-v2')
    device:         str  (default: 'cpu' â€” set to 'cuda' or 'mps' if available)
    batch_size:     int  (default: 32)
    max_length:     int  (default: 512)
    top_n:          int  (default: 5)
    allow_download: bool (default: False â€” opt-in network access)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from homie.config import HomieConfig, cfg_get
except ImportError:  # pragma: no cover - allow standalone import for tests
    HomieConfig = Any  # type: ignore[assignment,misc]

    def cfg_get(cfg: Any, *keys: str, default: Any = None) -> Any:  # type: ignore[no-redef]
        cur: Any = getattr(cfg, "raw", cfg)
        for key in keys:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur


LOGGER = logging.getLogger("homie.plugins.reranker")

PLUGIN_NAME = "reranker_ms_marco_minilm"
PLUGIN_VERSION = "1.0.0"
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


@dataclass
class RerankResult:
    """A single reranked document with its cross-encoder relevance score."""

    document: str
    score: float
    original_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class MsMarcoMiniLMReranker:
    """Cross-encoder reranker for HOMIE RAG candidate lists.

    The cross-encoder scores each (query, candidate) pair jointly and returns
    a single relevance logit. We sort by that logit to produce the final
    top-N passages handed to the LLM.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        batch_size: int = 32,
        max_length: int = 512,
        top_n: int = 5,
        allow_download: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.top_n = int(top_n)
        self.allow_download = bool(allow_download)
        self._model: Optional[Any] = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def load(self) -> None:
        """Lazy-load the cross-encoder model.

        Honors ``allow_download``: if False and the model is not already in
        the local Hugging Face cache, we force offline mode so no network
        traffic is attempted.
        """
        if self._model is not None:
            return

        if not self.allow_download:
            # Make sentence-transformers / huggingface_hub strictly offline.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise RuntimeError(
                "sentence-transformers is required for the reranker plugin. "
                "Install with: pip install sentence-transformers"
            ) from exc

        LOGGER.info("Loading reranker model %s on %s", self.model_name, self.device)
        self._model = CrossEncoder(
            self.model_name,
            max_length=self.max_length,
            device=self.device,
        )

    def unload(self) -> None:
        """Drop the model so its weights can be garbage collected."""
        if self._model is None:
            return
        LOGGER.info("Unloading reranker model %s", self.model_name)
        self._model = None
        try:
            import gc

            gc.collect()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ #
    # Core reranking API
    # ------------------------------------------------------------------ #
    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_n: Optional[int] = None,
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[RerankResult]:
        """Rerank ``documents`` against ``query`` and return the top-N hits.

        Args:
            query: The user question or retrieval prompt.
            documents: Candidate passages from the bi-encoder retriever.
            top_n: Override the default top-N cutoff for this call.
            metadatas: Optional per-document metadata (e.g. source path).
                Must align positionally with ``documents`` when provided.

        Returns:
            A list of :class:`RerankResult` sorted by descending score.
        """
        if not query or not query.strip():
            return []
        if not documents:
            return []

        self.load()
        assert self._model is not None  # for type checkers

        if metadatas is not None and len(metadatas) != len(documents):
            raise ValueError("metadatas must align positionally with documents")

        pairs: List[Tuple[str, str]] = [(query, doc) for doc in documents]
        scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        scored: List[RerankResult] = [
            RerankResult(
                document=documents[i],
                score=float(scores[i]),
                original_index=i,
                metadata=dict(metadatas[i]) if metadatas else {},
            )
            for i in range(len(documents))
        ]
        scored.sort(key=lambda r: r.score, reverse=True)

        cutoff = int(top_n) if top_n is not None else self.top_n
        if cutoff > 0:
            scored = scored[:cutoff]
        return scored

    def score(self, query: str, document: str) -> float:
        """Convenience helper to score a single (query, document) pair."""
        results = self.rerank(query, [document], top_n=1)
        return results[0].score if results else 0.0

    # ------------------------------------------------------------------ #
    # Plugin metadata
    # ------------------------------------------------------------------ #
    @property
    def info(self) -> Dict[str, Any]:
        """Plugin self-description for the HOMIE plugin registry/UI."""
        return {
            "name": PLUGIN_NAME,
            "version": PLUGIN_VERSION,
            "kind": "reranker",
            "model": self.model_name,
            "device": self.device,
            "top_n": self.top_n,
            "loaded": self._model is not None,
            "network": "opt-in (allow_download)",
        }


# ---------------------------------------------------------------------- #
# Plugin loader hooks expected by HOMIE
# ---------------------------------------------------------------------- #
_PLUGIN_INSTANCE: Optional[MsMarcoMiniLMReranker] = None


def _build_from_config(cfg: Optional[HomieConfig]) -> MsMarcoMiniLMReranker:
    if cfg is None:
        return MsMarcoMiniLMReranker()
    return MsMarcoMiniLMReranker(
        model_name=cfg_get(cfg, "plugins", "reranker", "model_name", default=DEFAULT_MODEL),
        device=cfg_get(cfg, "plugins", "reranker", "device", default="cpu"),
        batch_size=cfg_get(cfg, "plugins", "reranker", "batch_size", default=32),
        max_length=cfg_get(cfg, "plugins", "reranker", "max_length", default=512),
        top_n=cfg_get(cfg, "plugins", "reranker", "top_n", default=5),
        allow_download=cfg_get(
            cfg, "plugins", "reranker", "allow_download", default=False
        ),
    )


def activate(registry: Any = None, cfg: Optional[HomieConfig] = None) -> MsMarcoMiniLMReranker:
    """Register the reranker with HOMIE's plugin registry.

    Called by the plugin loader. ``registry`` is expected to expose a
    ``register(namespace, name, instance)`` method; if not, the plugin
    instance is returned and the caller is responsible for wiring it up.
    """
    global _PLUGIN_INSTANCE
    _PLUGIN_INSTANCE = _build_from_config(cfg)

    if registry is not None and hasattr(registry, "register"):
        try:
            registry.register("rerankers", PLUGIN_NAME, _PLUGIN_INSTANCE)
            LOGGER.info("Registered reranker plugin %s", PLUGIN_NAME)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to register reranker plugin")

    return _PLUGIN_INSTANCE


def deactivate() -> None:
    """Unload the model and clear the module-level singleton."""
    global _PLUGIN_INSTANCE
    if _PLUGIN_INSTANCE is not None:
        _PLUGIN_INSTANCE.unload()
    _PLUGIN_INSTANCE = None


def register(registry: Any = None, cfg: Optional[HomieConfig] = None) -> MsMarcoMiniLMReranker:
    """Alias for :func:`activate` to match alternative loader conventions."""
    return activate(registry=registry, cfg=cfg)


def get_instance() -> Optional[MsMarcoMiniLMReranker]:
    """Return the active reranker instance, if any."""
    return _PLUGIN_INSTANCE


__all__ = [
    "MsMarcoMiniLMReranker",
    "RerankResult",
    "PLUGIN_NAME",
    "PLUGIN_VERSION",
    "activate",
    "deactivate",
    "register",
    "get_instance",
]
