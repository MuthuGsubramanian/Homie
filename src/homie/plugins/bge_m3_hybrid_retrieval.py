"""BGE-M3 Hybrid Retrieval plugin for Homie.

Integrates BAAI/bge-m3 — a multilingual embedding model that produces
dense, sparse (lexical), and ColBERT-style multi-vector representations
in a single forward pass — as a hybrid retrieval backend for Homie's
ChromaDB-backed RAG layer.

The combined hybrid score (weighted dense + sparse + ColBERT) generally
outperforms single-mode dense retrieval on multilingual and long-document
benchmarks while still running fully on-device.

Source: https://huggingface.co/BAAI/bge-m3

Config (``homie.config.yaml``)::

    plugins:
      bge_m3:
        enabled: true
        model_id: BAAI/bge-m3
        cache_dir: ~/.homie/models/bge-m3
        device: cpu              # "cpu" | "cuda" | "mps"
        max_length: 8192
        use_fp16: true
        allow_model_download: false   # opt-in network use
        weights:
          dense: 0.4
          sparse: 0.2
          colbert: 0.4

Usage::

    from homie.plugins import bge_m3_hybrid_retrieval as bge
    retriever = bge.activate(cfg)
    ranked = retriever.rerank(query, passages, top_k=5)
    bge.deactivate()
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from homie.config import HomieConfig, cfg_get


LOGGER = logging.getLogger(__name__)

PLUGIN_NAME = "bge_m3_hybrid_retrieval"
PLUGIN_VERSION = "1.0.0"

# Module-level singleton populated by activate(); cleared by deactivate().
_RETRIEVER: Optional["BGEM3HybridRetriever"] = None


class BGEM3DependencyError(RuntimeError):
    """Raised when the FlagEmbedding backend is unavailable or blocked."""


@dataclass
class BGEM3Settings:
    """Resolved plugin settings derived from :class:`HomieConfig`."""

    model_id: str = "BAAI/bge-m3"
    cache_dir: Path = field(
        default_factory=lambda: Path.home() / ".homie" / "models" / "bge-m3"
    )
    device: str = "cpu"
    max_length: int = 8192
    use_fp16: bool = True
    allow_model_download: bool = False
    dense_weight: float = 0.4
    sparse_weight: float = 0.2
    colbert_weight: float = 0.4

    @classmethod
    def from_config(cls, cfg: HomieConfig) -> "BGEM3Settings":
        """Build settings from the ``plugins.bge_m3`` block in homie.config.yaml."""
        section = cfg_get(cfg, "plugins", "bge_m3", default={}) or {}
        weights = section.get("weights", {}) or {}
        defaults = cls()
        raw_cache = section.get("cache_dir") or str(defaults.cache_dir)
        cache_dir = Path(os.path.expanduser(raw_cache))
        return cls(
            model_id=section.get("model_id", defaults.model_id),
            cache_dir=cache_dir,
            device=section.get("device", defaults.device),
            max_length=int(section.get("max_length", defaults.max_length)),
            use_fp16=bool(section.get("use_fp16", defaults.use_fp16)),
            allow_model_download=bool(section.get("allow_model_download", False)),
            dense_weight=float(weights.get("dense", defaults.dense_weight)),
            sparse_weight=float(weights.get("sparse", defaults.sparse_weight)),
            colbert_weight=float(weights.get("colbert", defaults.colbert_weight)),
        )


class BGEM3HybridRetriever:
    """Wraps BAAI/bge-m3 to produce dense, sparse, and ColBERT embeddings.

    The model is loaded lazily on first use to keep plugin activation cheap.
    All inference is local; the only network access is the initial model
    download, which is blocked unless ``settings.allow_model_download`` is
    True — honouring Homie's ‘safe to run locally, no network unless opt-in’
    convention.
    """

    def __init__(self, settings: BGEM3Settings) -> None:
        self.settings = settings
        self._model: Any = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def load(self) -> None:
        """Load the BGE-M3 model from local cache (or download if opted in)."""
        if self._model is not None:
            return

        try:
            from FlagEmbedding import BGEM3FlagModel  # type: ignore
        except ImportError as exc:
            raise BGEM3DependencyError(
                "FlagEmbedding is required for the bge_m3 plugin. "
                "Install with: pip install -U FlagEmbedding"
            ) from exc

        cache = self.settings.cache_dir
        cache.mkdir(parents=True, exist_ok=True)
        already_cached = any(cache.iterdir())

        if not already_cached and not self.settings.allow_model_download:
            raise BGEM3DependencyError(
                f"Model '{self.settings.model_id}' is not present in {cache} and "
                "plugins.bge_m3.allow_model_download is False. Enable the flag "
                "in config to permit a one-time download."
            )

        # Scope HF_HOME to the configured cache without polluting the parent env.
        prev_hf_home = os.environ.get("HF_HOME")
        os.environ["HF_HOME"] = str(cache)
        try:
            LOGGER.info(
                "Loading BGE-M3 model '%s' on %s (fp16=%s)",
                self.settings.model_id,
                self.settings.device,
                self.settings.use_fp16,
            )
            self._model = BGEM3FlagModel(
                self.settings.model_id,
                use_fp16=self.settings.use_fp16,
                device=self.settings.device,
            )
        finally:
            if prev_hf_home is None:
                os.environ.pop("HF_HOME", None)
            else:
                os.environ["HF_HOME"] = prev_hf_home

    def unload(self) -> None:
        """Release the underlying model and free GPU/CPU memory."""
        self._model = None

    # ------------------------------------------------------------------ #
    # Encoding
    # ------------------------------------------------------------------ #
    def encode(
        self,
        texts: Sequence[str],
        *,
        return_dense: bool = True,
        return_sparse: bool = True,
        return_colbert: bool = True,
        batch_size: int = 12,
    ) -> Dict[str, Any]:
        """Encode ``texts`` and return the requested representations."""
        if not texts:
            return {"dense_vecs": [], "lexical_weights": [], "colbert_vecs": []}
        self.load()
        return self._model.encode(
            list(texts),
            batch_size=batch_size,
            max_length=self.settings.max_length,
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert_vecs=return_colbert,
        )

    def encode_dense(self, texts: Sequence[str]) -> List[List[float]]:
        """Return only the dense (1024-d) vectors — ideal for ChromaDB inserts."""
        out = self.encode(
            texts, return_dense=True, return_sparse=False, return_colbert=False
        )
        vecs = out.get("dense_vecs", [])
        return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vecs]

    # ------------------------------------------------------------------ #
    # Hybrid scoring / reranking
    # ------------------------------------------------------------------ #
    def hybrid_score(self, query: str, passages: Sequence[str]) -> List[float]:
        """Compute a weighted hybrid similarity score for each passage."""
        if not passages:
            return []
        self.load()
        q_enc = self.encode([query])
        p_enc = self.encode(list(passages))

        dense = self._dense_scores(q_enc, p_enc)
        sparse = self._sparse_scores(q_enc, p_enc)
        colbert = self._colbert_scores(q_enc, p_enc)

        w_d = self.settings.dense_weight
        w_s = self.settings.sparse_weight
        w_c = self.settings.colbert_weight
        norm = max(w_d + w_s + w_c, 1e-9)

        return [
            (w_d * d + w_s * s + w_c * c) / norm
            for d, s, c in zip(dense, sparse, colbert)
        ]

    def rerank(
        self,
        query: str,
        passages: Sequence[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, str, float]]:
        """Rerank ``passages`` by hybrid score; returns ``(index, text, score)``."""
        scores = self.hybrid_score(query, passages)
        order = sorted(range(len(passages)), key=lambda i: scores[i], reverse=True)
        if top_k is not None:
            order = order[:top_k]
        return [(i, passages[i], scores[i]) for i in order]

    # ------------------------------------------------------------------ #
    # Internal score helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _dense_scores(q_enc: Dict[str, Any], p_enc: Dict[str, Any]) -> List[float]:
        import numpy as np  # local import keeps the top-level dependency optional

        q_vecs = q_enc.get("dense_vecs") or []
        p_vecs = p_enc.get("dense_vecs") or []
        if len(q_vecs) == 0 or len(p_vecs) == 0:
            return [0.0] * len(p_vecs)
        q = np.asarray(q_vecs[0])
        p = np.asarray(p_vecs)
        return (p @ q).astype(float).tolist()

    def _sparse_scores(self, q_enc: Dict[str, Any], p_enc: Dict[str, Any]) -> List[float]:
        q_lex = q_enc.get("lexical_weights") or []
        p_lex = p_enc.get("lexical_weights") or []
        if not q_lex or not p_lex:
            return [0.0] * len(p_lex)
        compute = getattr(self._model, "compute_lexical_matching_score", None)
        if compute is None:
            return [0.0] * len(p_lex)
        return [float(compute(q_lex[0], pw)) for pw in p_lex]

    def _colbert_scores(self, q_enc: Dict[str, Any], p_enc: Dict[str, Any]) -> List[float]:
        q_vecs = q_enc.get("colbert_vecs") or []
        p_vecs = p_enc.get("colbert_vecs") or []
        if not q_vecs or not p_vecs:
            return [0.0] * len(p_vecs)
        compute = getattr(self._model, "colbert_score", None)
        if compute is None:
            return [0.0] * len(p_vecs)
        return [float(compute(q_vecs[0], pv)) for pv in p_vecs]


# ---------------------------------------------------------------------- #
# Plugin entry points
# ---------------------------------------------------------------------- #
def activate(cfg: HomieConfig) -> BGEM3HybridRetriever:
    """Initialise the plugin from config and return the retriever singleton."""
    global _RETRIEVER
    if not cfg_get(cfg, "plugins", "bge_m3", "enabled", default=False):
        raise RuntimeError(
            "Plugin 'bge_m3' is disabled. Set plugins.bge_m3.enabled: true "
            "in homie.config.yaml to activate."
        )
    settings = BGEM3Settings.from_config(cfg)
    _RETRIEVER = BGEM3HybridRetriever(settings)
    LOGGER.info(
        "Activated %s v%s (model=%s, device=%s)",
        PLUGIN_NAME,
        PLUGIN_VERSION,
        settings.model_id,
        settings.device,
    )
    return _RETRIEVER


def deactivate() -> None:
    """Tear down the retriever and release model memory."""
    global _RETRIEVER
    if _RETRIEVER is not None:
        _RETRIEVER.unload()
        _RETRIEVER = None
        LOGGER.info("Deactivated %s", PLUGIN_NAME)


def get_retriever() -> BGEM3HybridRetriever:
    """Return the active retriever; raise if :func:`activate` has not run."""
    if _RETRIEVER is None:
        raise RuntimeError(
            "bge_m3 plugin is not active. Call activate(cfg) before use."
        )
    return _RETRIEVER


def register() -> Dict[str, Any]:
    """Return plugin metadata for Homie's plugin registry."""
    return {
        "name": PLUGIN_NAME,
        "version": PLUGIN_VERSION,
        "kind": "embedding",
        "capabilities": [
            "dense",
            "sparse",
            "colbert",
            "hybrid_rerank",
            "multilingual",
        ],
        "activate": activate,
        "deactivate": deactivate,
        "factory": get_retriever,
    }


__all__ = [
    "PLUGIN_NAME",
    "PLUGIN_VERSION",
    "BGEM3Settings",
    "BGEM3HybridRetriever",
    "BGEM3DependencyError",
    "activate",
    "deactivate",
    "get_retriever",
    "register",
]
