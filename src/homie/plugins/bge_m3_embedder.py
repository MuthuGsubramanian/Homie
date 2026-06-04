"""Homie plugin: BAAI/bge-m3 multilingual multi-vector embedding backbone.

This plugin integrates BAAI's bge-m3 embedding model into Homie's local RAG
pipeline. bge-m3 is a state-of-the-art sentence embedding model that supports:

- 100+ languages (multilingual retrieval out of the box)
- Long inputs up to 8192 tokens
- Three retrieval modes from a single forward pass:
    * Dense embeddings (semantic similarity, ChromaDB-compatible)
    * Sparse / lexical weights (BM25-like, exact-match boost)
    * Multi-vector / ColBERT-style (fine-grained late interaction)

Homie uses this plugin as an advanced embedding backbone for the local
ChromaDB knowledge store. The model and tokenizer are downloaded once from
Hugging Face the first time the plugin is activated WITH ``allow_download``
explicitly enabled (the only network call this plugin makes); afterwards the
plugin runs fully offline on CPU or CUDA.

Source: https://huggingface.co/BAAI/bge-m3
License: MIT
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

PLUGIN_NAME = "bge_m3_embedder"
DEFAULT_MODEL_ID = "BAAI/bge-m3"
DEFAULT_CACHE_DIR = Path.home() / ".homie" / "models" / "bge-m3"
DEFAULT_MAX_LENGTH = 8192


@dataclass
class BgeM3Config:
    """Runtime configuration for the bge-m3 embedder."""

    model_id: str = DEFAULT_MODEL_ID
    cache_dir: Path = field(default_factory=lambda: DEFAULT_CACHE_DIR)
    device: str = "cpu"  # "cpu" | "cuda" | "cuda:0"
    max_length: int = DEFAULT_MAX_LENGTH
    use_fp16: bool = False  # Recommended only on CUDA
    return_dense: bool = True
    return_sparse: bool = False
    return_colbert_vecs: bool = False
    allow_download: bool = False  # Opt-in; enables the only network call.
    batch_size: int = 12


class BgeM3Embedder:
    """Multilingual / multi-vector embedder powered by BAAI/bge-m3.

    The model is loaded lazily on the first ``encode()`` call, so importing
    the plugin (or merely calling :func:`activate`) has zero runtime cost.
    The class exposes:

    * :meth:`embed_documents` / :meth:`embed_query` -- ChromaDB-compatible
      dense float vectors for drop-in use in Homie's RAG store.
    * :meth:`encode` -- low-level access to all three vector kinds.
    * :meth:`score_pairs` -- hybrid (dense + sparse + ColBERT) reranking.
    """

    def __init__(self, config: Optional[BgeM3Config] = None) -> None:
        self.config = config or BgeM3Config()
        self._model: Any = None  # FlagEmbedding.BGEM3FlagModel
        self._loaded: bool = False

    @property
    def is_loaded(self) -> bool:
        """True once the underlying model has been instantiated."""
        return self._loaded

    def _ensure_model(self) -> None:
        if self._loaded:
            return

        try:
            from FlagEmbedding import BGEM3FlagModel  # type: ignore
        except ImportError as exc:  # pragma: no cover - import-time only
            raise RuntimeError(
                "bge-m3 plugin requires the 'FlagEmbedding' package. "
                "Install it with: pip install -U FlagEmbedding"
            ) from exc

        cache_dir = Path(self.config.cache_dir).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Strict local-first: unless the user has opted in to a download,
        # force HuggingFace to only resolve the model from the local cache.
        if not self.config.allow_download:
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

        logger.info(
            "Loading bge-m3 model %s on %s (cache=%s, offline=%s)",
            self.config.model_id,
            self.config.device,
            cache_dir,
            not self.config.allow_download,
        )

        self._model = BGEM3FlagModel(
            self.config.model_id,
            use_fp16=self.config.use_fp16,
            cache_dir=str(cache_dir),
            device=self.config.device,
        )
        self._loaded = True

    def encode(
        self,
        texts: Sequence[str],
        *,
        return_dense: Optional[bool] = None,
        return_sparse: Optional[bool] = None,
        return_colbert_vecs: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Encode a batch of texts.

        Returns a dict with optional keys ``dense_vecs``, ``lexical_weights``
        and ``colbert_vecs`` depending on which modes are requested. Per-call
        overrides fall back to the plugin :class:`BgeM3Config` defaults.
        """
        if not texts:
            return {}

        self._ensure_model()

        want_dense = self.config.return_dense if return_dense is None else return_dense
        want_sparse = self.config.return_sparse if return_sparse is None else return_sparse
        want_colbert = (
            self.config.return_colbert_vecs if return_colbert_vecs is None else return_colbert_vecs
        )

        return self._model.encode(
            list(texts),
            batch_size=self.config.batch_size,
            max_length=self.config.max_length,
            return_dense=want_dense,
            return_sparse=want_sparse,
            return_colbert_vecs=want_colbert,
        )

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """Return dense float vectors -- ChromaDB ``embedding_function`` API."""
        out = self.encode(
            texts,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense = out.get("dense_vecs", [])
        return [vec.tolist() if hasattr(vec, "tolist") else list(vec) for vec in dense]

    def embed_query(self, text: str) -> List[float]:
        """Return a single dense float vector for one query string."""
        return self.embed_documents([text])[0]

    def score_pairs(
        self,
        queries: Sequence[str],
        passages: Sequence[str],
        weights: Tuple[float, float, float] = (0.4, 0.2, 0.4),
    ) -> List[float]:
        """Compute hybrid (dense + sparse + ColBERT) similarity scores.

        ``weights`` is the ``(dense, sparse, colbert)`` blending coefficients
        suggested by the bge-m3 authors. ``queries`` and ``passages`` must be
        the same length and are scored pairwise.
        """
        if len(queries) != len(passages):
            raise ValueError("queries and passages must have the same length")

        self._ensure_model()
        q_out = self.encode(
            queries,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
        )
        p_out = self.encode(
            passages,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
        )

        scores: List[float] = []
        for q_dense, q_sparse, q_colbert, p_dense, p_sparse, p_colbert in zip(
            q_out["dense_vecs"],
            q_out["lexical_weights"],
            q_out["colbert_vecs"],
            p_out["dense_vecs"],
            p_out["lexical_weights"],
            p_out["colbert_vecs"],
        ):
            dense_score = float((q_dense * p_dense).sum())
            sparse_score = float(
                self._model.compute_lexical_matching_score(q_sparse, p_sparse)
            )
            colbert_score = float(self._model.colbert_score(q_colbert, p_colbert))
            scores.append(
                weights[0] * dense_score
                + weights[1] * sparse_score
                + weights[2] * colbert_score
            )
        return scores

    def unload(self) -> None:
        """Release the underlying model so memory can be reclaimed."""
        self._model = None
        self._loaded = False
        logger.info("bge-m3 model unloaded")


# ---------------------------------------------------------------------------
# Plugin lifecycle hooks (Homie convention: activate / deactivate / register)
# ---------------------------------------------------------------------------

_INSTANCE: Optional[BgeM3Embedder] = None


def _config_from_homie(cfg: Any) -> BgeM3Config:
    """Build a :class:`BgeM3Config` from a :class:`homie.config.HomieConfig`.

    Reads ``plugins.bge_m3_embedder.*`` keys, falling back to safe defaults.
    """
    from homie.config import cfg_get  # local import keeps activate() cheap

    return BgeM3Config(
        model_id=cfg_get(cfg, "plugins", PLUGIN_NAME, "model_id", default=DEFAULT_MODEL_ID),
        cache_dir=Path(
            cfg_get(cfg, "plugins", PLUGIN_NAME, "cache_dir", default=str(DEFAULT_CACHE_DIR))
        ),
        device=cfg_get(cfg, "plugins", PLUGIN_NAME, "device", default="cpu"),
        max_length=int(
            cfg_get(cfg, "plugins", PLUGIN_NAME, "max_length", default=DEFAULT_MAX_LENGTH)
        ),
        use_fp16=bool(cfg_get(cfg, "plugins", PLUGIN_NAME, "use_fp16", default=False)),
        return_dense=bool(cfg_get(cfg, "plugins", PLUGIN_NAME, "return_dense", default=True)),
        return_sparse=bool(cfg_get(cfg, "plugins", PLUGIN_NAME, "return_sparse", default=False)),
        return_colbert_vecs=bool(
            cfg_get(cfg, "plugins", PLUGIN_NAME, "return_colbert_vecs", default=False)
        ),
        allow_download=bool(
            cfg_get(cfg, "plugins", PLUGIN_NAME, "allow_download", default=False)
        ),
        batch_size=int(cfg_get(cfg, "plugins", PLUGIN_NAME, "batch_size", default=12)),
    )


def activate(cfg: Any = None) -> BgeM3Embedder:
    """Activate the plugin and return the embedder singleton.

    The model is NOT loaded yet -- it lazy-loads on the first ``encode()``
    call so that activation stays cheap and side-effect free.
    """
    global _INSTANCE
    if _INSTANCE is None:
        config = _config_from_homie(cfg) if cfg is not None else BgeM3Config()
        _INSTANCE = BgeM3Embedder(config)
        logger.info("Activated plugin '%s'", PLUGIN_NAME)
    return _INSTANCE


def deactivate() -> None:
    """Deactivate the plugin and release any loaded model state."""
    global _INSTANCE
    if _INSTANCE is not None:
        _INSTANCE.unload()
        _INSTANCE = None
        logger.info("Deactivated plugin '%s'", PLUGIN_NAME)


def register(registry: Any) -> None:
    """Register the plugin with a Homie plugin registry, if one is provided.

    Falls back to a generic ``register(name, factory)`` interface and finally
    logs a warning if neither hook is present -- callers can still use
    :func:`activate` directly.
    """
    if hasattr(registry, "register_embedder"):
        registry.register_embedder(PLUGIN_NAME, activate)
    elif hasattr(registry, "register"):
        registry.register(PLUGIN_NAME, activate)
    else:
        logger.warning(
            "Registry %r exposes no known registration method; "
            "call activate() manually instead.",
            registry,
        )


__all__ = [
    "PLUGIN_NAME",
    "BgeM3Config",
    "BgeM3Embedder",
    "activate",
    "deactivate",
    "register",
]
