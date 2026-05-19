"""BGE-M3 Multilingual Multi-Vector Embeddings plugin for Homie.

This plugin integrates BAAI/bge-m3 into Homie's local RAG stack. BGE-M3 is a
state-of-the-art multilingual embedding model (100+ languages) that uniquely
supports three retrieval modes from a single forward pass:

    * Dense  - standard semantic similarity vectors (1024-d)
    * Sparse - learned lexical weights (BM25-style, term-level)
    * Multi-vec / ColBERT - per-token contextual vectors for late interaction

It is well suited for Homie users who need:
    * Cross-lingual RAG (query in one language, retrieve in another)
    * Hybrid dense+sparse retrieval against ChromaDB collections
    * Long-context document chunks (up to 8192 tokens)

Design notes
------------
* The model is loaded LAZILY on first encode call so plugin activation stays cheap.
* Inference runs fully locally via FlagEmbedding / sentence-transformers. The only
  network access is the initial Hugging Face model download, gated behind the
  ``allow_download`` config flag (opt-in, defaults to False).
* When registered, the plugin exposes itself to Homie as both a Chroma-compatible
  EmbeddingFunction and a stand-alone hybrid retriever.

Dependencies (install on demand):
    pip install FlagEmbedding         # preferred - exposes dense+sparse+colbert
    # OR
    pip install sentence-transformers # dense-only fallback
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

PLUGIN_NAME = "bge_m3_multilingual_embeddings"
PLUGIN_VERSION = "0.1.0"
MODEL_ID = "BAAI/bge-m3"
DEFAULT_DIM = 1024
DEFAULT_MAX_TOKENS = 8192


@dataclass
class BGEM3Settings:
    """Configuration for the BGE-M3 plugin.

    Mirrors the ``plugins.bge_m3`` block of ``homie.config.yaml``.
    """

    model_id: str = MODEL_ID
    device: str = "cpu"          # "cpu" | "cuda" | "mps"
    use_fp16: bool = False        # only honoured on cuda
    max_length: int = DEFAULT_MAX_TOKENS
    batch_size: int = 12
    return_dense: bool = True
    return_sparse: bool = True
    return_colbert_vecs: bool = False
    normalize_embeddings: bool = True
    cache_dir: Optional[str] = None
    allow_download: bool = False  # opt-in network access for model fetch
    hybrid_weights: Tuple[float, float, float] = (1.0, 0.3, 0.0)  # dense, sparse, colbert

    @classmethod
    def from_cfg(cls, cfg: Any) -> "BGEM3Settings":
        """Build settings from a HomieConfig instance (or dict)."""
        try:
            from homie.config import cfg_get  # local import to avoid hard dep at import time
            block = cfg_get(cfg, "plugins", "bge_m3", default={}) or {}
        except Exception:  # noqa: BLE001
            block = cfg.get("plugins", {}).get("bge_m3", {}) if isinstance(cfg, dict) else {}
        return cls(
            model_id=block.get("model_id", MODEL_ID),
            device=block.get("device", "cpu"),
            use_fp16=bool(block.get("use_fp16", False)),
            max_length=int(block.get("max_length", DEFAULT_MAX_TOKENS)),
            batch_size=int(block.get("batch_size", 12)),
            return_dense=bool(block.get("return_dense", True)),
            return_sparse=bool(block.get("return_sparse", True)),
            return_colbert_vecs=bool(block.get("return_colbert_vecs", False)),
            normalize_embeddings=bool(block.get("normalize_embeddings", True)),
            cache_dir=block.get("cache_dir"),
            allow_download=bool(block.get("allow_download", False)),
            hybrid_weights=tuple(block.get("hybrid_weights", (1.0, 0.3, 0.0))),  # type: ignore[arg-type]
        )


class BGEM3EmbeddingPlugin:
    """Homie plugin exposing BGE-M3 multilingual multi-vector embeddings.

    Lifecycle
    ---------
    * ``activate(cfg, registry=None)`` - load settings, register the plugin's
      embedding function with Homie's plugin registry (if provided). The heavy
      model is NOT loaded here; first encode call triggers it.
    * ``deactivate()`` - release the underlying model and free GPU memory.

    Public methods
    --------------
    * ``encode(texts, ...)`` - returns a dict with any of
      ``{'dense', 'sparse', 'colbert'}`` keys depending on settings.
    * ``encode_dense(texts)`` - convenience: just dense vectors (list[list[float]]).
    * ``score(query, passages)`` - hybrid dense+sparse+colbert similarity scores.
    * ``as_chroma_embedding_function()`` - returns a callable compatible with
      ChromaDB's ``EmbeddingFunction`` protocol (Homie's RAG store).
    """

    name = PLUGIN_NAME
    version = PLUGIN_VERSION
    description = "BAAI/bge-m3 multilingual hybrid (dense+sparse+ColBERT) embeddings"

    def __init__(self, settings: Optional[BGEM3Settings] = None) -> None:
        self.settings: BGEM3Settings = settings or BGEM3Settings()
        self._model: Any = None
        self._backend: str = "none"  # "flag_embedding" | "sentence_transformers"
        self._loaded: bool = False

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #
    def activate(self, cfg: Any = None, registry: Any = None) -> "BGEM3EmbeddingPlugin":
        """Register the plugin. Does not download or load the model."""
        if cfg is not None:
            self.settings = BGEM3Settings.from_cfg(cfg)

        if not self.settings.allow_download:
            # Surface to user that they must pre-cache the model or opt in.
            logger.info(
                "%s activated in offline-only mode. Set plugins.bge_m3.allow_download=true "
                "to permit one-time model fetch from Hugging Face.",
                PLUGIN_NAME,
            )
        else:
            logger.info("%s activated. Model download permitted.", PLUGIN_NAME)

        if registry is not None and hasattr(registry, "register_embedding"):
            try:
                registry.register_embedding(self.name, self.as_chroma_embedding_function())
            except Exception:  # noqa: BLE001
                logger.exception("Failed to register %s with plugin registry", self.name)
        return self

    def deactivate(self) -> None:
        """Release model resources."""
        self._model = None
        self._loaded = False
        self._backend = "none"
        try:
            import gc

            gc.collect()
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        logger.info("%s deactivated.", PLUGIN_NAME)

    # ------------------------------------------------------------------ #
    # Model loading                                                      #
    # ------------------------------------------------------------------ #
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        s = self.settings
        cache_dir = s.cache_dir or os.environ.get("HF_HOME")
        if not s.allow_download:
            # Force offline mode so transformers will not attempt to fetch.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        # Preferred: FlagEmbedding gives dense + sparse + ColBERT in one pass.
        try:
            from FlagEmbedding import BGEM3FlagModel  # type: ignore

            kwargs: Dict[str, Any] = {
                "use_fp16": s.use_fp16 and s.device.startswith("cuda"),
            }
            if cache_dir:
                kwargs["cache_dir"] = cache_dir
            if s.device:
                kwargs["devices"] = [s.device]
            self._model = BGEM3FlagModel(s.model_id, **kwargs)
            self._backend = "flag_embedding"
            self._loaded = True
            logger.info("Loaded BGE-M3 via FlagEmbedding (device=%s)", s.device)
            return
        except ImportError:
            logger.info("FlagEmbedding not installed, falling back to sentence-transformers (dense only).")
        except Exception:  # noqa: BLE001
            logger.exception("FlagEmbedding load failed, trying sentence-transformers fallback.")

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            st_kwargs: Dict[str, Any] = {"device": s.device}
            if cache_dir:
                st_kwargs["cache_folder"] = cache_dir
            self._model = SentenceTransformer(s.model_id, **st_kwargs)
            self._backend = "sentence_transformers"
            self._loaded = True
            logger.info("Loaded BGE-M3 via sentence-transformers (dense only, device=%s)", s.device)
            return
        except ImportError as exc:
            raise RuntimeError(
                "Neither FlagEmbedding nor sentence-transformers is installed. "
                "Install one of them to use the BGE-M3 plugin."
            ) from exc

    # ------------------------------------------------------------------ #
    # Encoding                                                           #
    # ------------------------------------------------------------------ #
    def encode(
        self,
        texts: Sequence[str],
        return_dense: Optional[bool] = None,
        return_sparse: Optional[bool] = None,
        return_colbert_vecs: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Encode texts. Returns dict with available representations.

        Keys present in the returned dict depend on backend + flags:
            * ``dense``  : List[List[float]]
            * ``sparse`` : List[Dict[token_id, weight]]  (FlagEmbedding only)
            * ``colbert``: List[List[List[float]]]       (FlagEmbedding only)
        """
        if not texts:
            return {}
        self._ensure_loaded()
        s = self.settings
        rd = s.return_dense if return_dense is None else return_dense
        rs = s.return_sparse if return_sparse is None else return_sparse
        rc = s.return_colbert_vecs if return_colbert_vecs is None else return_colbert_vecs

        if self._backend == "flag_embedding":
            output = self._model.encode(
                list(texts),
                batch_size=s.batch_size,
                max_length=s.max_length,
                return_dense=rd,
                return_sparse=rs,
                return_colbert_vecs=rc,
            )
            result: Dict[str, Any] = {}
            if rd and "dense_vecs" in output:
                result["dense"] = [list(map(float, v)) for v in output["dense_vecs"]]
            if rs and "lexical_weights" in output:
                result["sparse"] = output["lexical_weights"]
            if rc and "colbert_vecs" in output:
                result["colbert"] = output["colbert_vecs"]
            return result

        # sentence-transformers fallback - dense only
        if rs or rc:
            logger.warning(
                "Sparse / ColBERT vectors requested but FlagEmbedding is unavailable; returning dense only."
            )
        vecs = self._model.encode(
            list(texts),
            batch_size=s.batch_size,
            normalize_embeddings=s.normalize_embeddings,
            convert_to_numpy=True,
        )
        return {"dense": [list(map(float, v)) for v in vecs]}

    def encode_dense(self, texts: Sequence[str]) -> List[List[float]]:
        """Convenience wrapper returning just dense vectors."""
        out = self.encode(texts, return_dense=True, return_sparse=False, return_colbert_vecs=False)
        return out.get("dense", [])

    # ------------------------------------------------------------------ #
    # Scoring (hybrid)                                                   #
    # ------------------------------------------------------------------ #
    def score(self, query: str, passages: Sequence[str]) -> List[float]:
        """Hybrid similarity scores between query and each passage.

        Combines dense cosine, sparse lexical, and (optionally) ColBERT
        max-sim using ``settings.hybrid_weights``. Falls back gracefully if
        the backend cannot produce sparse / ColBERT vectors.
        """
        if not passages:
            return []
        self._ensure_loaded()

        if self._backend == "flag_embedding":
            q_out = self.encode([query], return_dense=True, return_sparse=True, return_colbert_vecs=True)
            p_out = self.encode(
                list(passages),
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=True,
            )
            w_dense, w_sparse, w_colbert = self.settings.hybrid_weights
            scores: List[float] = []
            for i in range(len(passages)):
                s_dense = _cosine(q_out["dense"][0], p_out["dense"][i]) if "dense" in q_out else 0.0
                s_sparse = _sparse_dot(q_out["sparse"][0], p_out["sparse"][i]) if "sparse" in q_out else 0.0
                s_col = 0.0
                if "colbert" in q_out and "colbert" in p_out and w_colbert:
                    s_col = _colbert_maxsim(q_out["colbert"][0], p_out["colbert"][i])
                scores.append(float(w_dense * s_dense + w_sparse * s_sparse + w_colbert * s_col))
            return scores

        # Dense-only fallback
        q_vec = self.encode_dense([query])[0]
        p_vecs = self.encode_dense(list(passages))
        return [_cosine(q_vec, pv) for pv in p_vecs]

    # ------------------------------------------------------------------ #
    # ChromaDB integration                                               #
    # ------------------------------------------------------------------ #
    def as_chroma_embedding_function(self):
        """Return a callable compatible with ChromaDB's EmbeddingFunction.

        Chroma calls ``ef(input=["text1", "text2", ...])`` and expects
        ``List[List[float]]`` back. We always return dense vectors here; use
        the hybrid ``score()`` API for sparse/ColBERT reranking on top of
        Chroma's dense recall.
        """
        plugin = self

        class _BGEM3ChromaEF:
            def __call__(self, input: Sequence[str]) -> List[List[float]]:  # noqa: A002
                return plugin.encode_dense(input)

            def name(self) -> str:  # newer Chroma versions call this
                return PLUGIN_NAME

        return _BGEM3ChromaEF()


# ---------------------------------------------------------------------- #
# Math helpers (pure-Python, no numpy dependency at module import time)  #
# ---------------------------------------------------------------------- #
def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


def _sparse_dot(a: Dict[Any, float], b: Dict[Any, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    return float(sum(float(w) * float(b.get(tok, 0.0)) for tok, w in a.items()))


def _colbert_maxsim(q_vecs: Sequence[Sequence[float]], p_vecs: Sequence[Sequence[float]]) -> float:
    if not q_vecs or not p_vecs:
        return 0.0
    total = 0.0
    for qv in q_vecs:
        total += max((_cosine(qv, pv) for pv in p_vecs), default=0.0)
    return total / len(q_vecs)


# ---------------------------------------------------------------------- #
# Module-level activate / deactivate (Homie plugin entry points)         #
# ---------------------------------------------------------------------- #
_PLUGIN_SINGLETON: Optional[BGEM3EmbeddingPlugin] = None


def activate(cfg: Any = None, registry: Any = None) -> BGEM3EmbeddingPlugin:
    """Module-level entry point used by Homie's plugin loader."""
    global _PLUGIN_SINGLETON
    if _PLUGIN_SINGLETON is None:
        _PLUGIN_SINGLETON = BGEM3EmbeddingPlugin()
    _PLUGIN_SINGLETON.activate(cfg=cfg, registry=registry)
    return _PLUGIN_SINGLETON


def deactivate() -> None:
    """Module-level teardown counterpart to :func:`activate`."""
    global _PLUGIN_SINGLETON
    if _PLUGIN_SINGLETON is not None:
        _PLUGIN_SINGLETON.deactivate()
        _PLUGIN_SINGLETON = None


def register(registry: Any, cfg: Any = None) -> BGEM3EmbeddingPlugin:
    """Alternative entry point matching Homie's ``register(registry, cfg)`` shape."""
    return activate(cfg=cfg, registry=registry)


__all__ = [
    "PLUGIN_NAME",
    "PLUGIN_VERSION",
    "BGEM3Settings",
    "BGEM3EmbeddingPlugin",
    "activate",
    "deactivate",
    "register",
]
