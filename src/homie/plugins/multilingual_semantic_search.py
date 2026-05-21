"""Multilingual semantic search plugin for Homie.

Provides cross-lingual sentence embedding and similarity search powered by
``sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2``. The model
is loaded lazily on activation and runs fully offline once the weights are
cached locally (Hugging Face cache, default: ``~/.cache/huggingface``).

Integrates with Homie's ChromaDB-backed RAG layer when available, otherwise
falls back to an in-memory NumPy cosine-similarity index. Designed for
querying a multilingual knowledge base (50+ languages) such as user notes,
local docs, and transcribed voice memos from Whisper.

No network calls are made at query time. The initial model download is the
only network operation and must be triggered explicitly via
``activate(allow_download=True)`` or by pre-populating the HF cache.

Follows Homie plugin conventions: module docstring, single class with
``activate()`` / ``deactivate()`` lifecycle hooks, ``register()`` entry
point, and ``cfg_get``-style configuration access.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from homie.config import HomieConfig, cfg_get

logger = logging.getLogger(__name__)

PLUGIN_NAME = "multilingual_semantic_search"
MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384
DEFAULT_COLLECTION = "homie_multilingual"


@dataclass
class SearchHit:
    """A single semantic search result."""

    doc_id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "score": float(self.score),
            "metadata": self.metadata,
        }


class MultilingualSemanticSearchPlugin:
    """Homie plugin exposing multilingual embedding + similarity search.

    The plugin is intentionally backend-agnostic: it prefers ChromaDB when
    the ``chromadb`` package is installed and a persistent path is
    configured, otherwise it maintains an in-memory NumPy index.
    """

    name: str = PLUGIN_NAME
    model_id: str = MODEL_ID

    def __init__(self, cfg: Optional[HomieConfig] = None) -> None:
        self.cfg = cfg
        self._model: Optional[Any] = None
        self._chroma_client: Optional[Any] = None
        self._collection: Optional[Any] = None
        self._mem_ids: List[str] = []
        self._mem_texts: List[str] = []
        self._mem_embeddings: Optional[Any] = None  # numpy.ndarray
        self._mem_metadata: List[Dict[str, Any]] = []
        self._device: str = "cpu"
        self._persist_dir: Optional[Path] = None
        self._collection_name: str = DEFAULT_COLLECTION
        self._active: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def activate(self, allow_download: bool = False) -> None:
        """Load the embedding model and prepare the vector backend.

        Parameters
        ----------
        allow_download:
            If False (default) and the model weights are not present in the
            local HF cache, activation raises ``RuntimeError`` instead of
            making a network call. Set to True the first time you enable
            the plugin on a new machine.
        """
        if self._active:
            logger.debug("%s already active", PLUGIN_NAME)
            return

        self._load_settings()

        if not allow_download and not self._weights_cached():
            raise RuntimeError(
                f"Model weights for {self.model_id} not found locally. "
                "Re-activate with allow_download=True to fetch them."
            )

        self._model = self._load_model()
        self._init_backend()
        self._active = True
        logger.info(
            "%s activated (model=%s, device=%s, backend=%s)",
            PLUGIN_NAME,
            self.model_id,
            self._device,
            "chromadb" if self._collection is not None else "in-memory",
        )

    def deactivate(self) -> None:
        """Release the model and any open vector-store handles."""
        self._model = None
        self._collection = None
        self._chroma_client = None
        self._mem_ids.clear()
        self._mem_texts.clear()
        self._mem_metadata.clear()
        self._mem_embeddings = None
        self._active = False
        logger.info("%s deactivated", PLUGIN_NAME)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def embed(self, texts: Sequence[str]) -> Any:
        """Return a ``numpy.ndarray`` of shape ``(len(texts), 384)``."""
        self._ensure_active()
        if not texts:
            import numpy as np  # local import keeps activation cheap

            return np.zeros((0, EMBEDDING_DIM), dtype="float32")
        return self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def index(
        self,
        docs: Iterable[Dict[str, Any]],
    ) -> int:
        """Add documents to the vector store.

        Each ``doc`` must contain ``id`` and ``text``; ``metadata`` is
        optional. Returns the number of documents indexed.
        """
        self._ensure_active()
        ids: List[str] = []
        texts: List[str] = []
        metas: List[Dict[str, Any]] = []
        for doc in docs:
            doc_id = str(doc["id"])
            text = str(doc["text"])
            meta = dict(doc.get("metadata") or {})
            ids.append(doc_id)
            texts.append(text)
            metas.append(meta)

        if not ids:
            return 0

        embeddings = self.embed(texts)

        if self._collection is not None:
            self._collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings.tolist(),
                metadatas=metas,
            )
        else:
            import numpy as np

            self._mem_ids.extend(ids)
            self._mem_texts.extend(texts)
            self._mem_metadata.extend(metas)
            if self._mem_embeddings is None:
                self._mem_embeddings = embeddings.astype("float32")
            else:
                self._mem_embeddings = np.vstack(
                    [self._mem_embeddings, embeddings.astype("float32")]
                )
        return len(ids)

    def search(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[SearchHit]:
        """Return up to ``top_k`` semantically similar documents."""
        self._ensure_active()
        if not query or not query.strip():
            return []

        query_emb = self.embed([query])[0]

        if self._collection is not None:
            result = self._collection.query(
                query_embeddings=[query_emb.tolist()],
                n_results=max(1, top_k),
                where=where or None,
            )
            hits: List[SearchHit] = []
            ids = (result.get("ids") or [[]])[0]
            docs = (result.get("documents") or [[]])[0]
            metas = (result.get("metadatas") or [[]])[0]
            distances = (result.get("distances") or [[]])[0]
            for doc_id, text, meta, dist in zip(ids, docs, metas, distances):
                hits.append(
                    SearchHit(
                        doc_id=str(doc_id),
                        text=str(text),
                        score=float(1.0 - dist),
                        metadata=dict(meta or {}),
                    )
                )
            return hits

        return self._search_in_memory(query_emb, top_k)

    def similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between two strings (any supported language)."""
        self._ensure_active()
        import numpy as np

        emb = self.embed([text_a, text_b])
        return float(np.dot(emb[0], emb[1]))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _load_settings(self) -> None:
        if self.cfg is None:
            self._device = os.getenv("HOMIE_EMBED_DEVICE", "cpu")
            self._collection_name = DEFAULT_COLLECTION
            self._persist_dir = None
            return

        self._device = cfg_get(
            self.cfg, "plugins", PLUGIN_NAME, "device", default="cpu"
        )
        self._collection_name = cfg_get(
            self.cfg,
            "plugins",
            PLUGIN_NAME,
            "collection",
            default=DEFAULT_COLLECTION,
        )
        persist = cfg_get(
            self.cfg, "plugins", PLUGIN_NAME, "persist_dir", default=None
        )
        self._persist_dir = Path(persist).expanduser() if persist else None

    def _weights_cached(self) -> bool:
        cache_root = Path(
            os.getenv("HF_HOME")
            or os.getenv("TRANSFORMERS_CACHE")
            or Path.home() / ".cache" / "huggingface"
        )
        fragment = self.model_id.replace("/", "--")
        if not cache_root.exists():
            return False
        for child in cache_root.rglob("*"):
            if fragment in child.name:
                return True
        return False

    def _load_model(self) -> Any:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # noqa: BLE001
            raise RuntimeError(
                "sentence-transformers is required for "
                f"{PLUGIN_NAME}. Install with `pip install sentence-transformers`."
            ) from exc
        return SentenceTransformer(self.model_id, device=self._device)

    def _init_backend(self) -> None:
        if self._persist_dir is None:
            return
        try:
            import chromadb
        except ImportError:
            logger.info(
                "chromadb not installed; falling back to in-memory index for %s",
                PLUGIN_NAME,
            )
            return

        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._chroma_client = chromadb.PersistentClient(
            path=str(self._persist_dir)
        )
        self._collection = self._chroma_client.get_or_create_collection(
            name=self._collection_name,
            metadata={"model": self.model_id, "dim": EMBEDDING_DIM},
        )

    def _search_in_memory(self, query_emb: Any, top_k: int) -> List[SearchHit]:
        if self._mem_embeddings is None or not self._mem_ids:
            return []
        import numpy as np

        scores = self._mem_embeddings @ query_emb.astype("float32")
        k = min(top_k, scores.shape[0])
        top_idx = np.argpartition(-scores, k - 1)[:k]
        ordered = top_idx[np.argsort(-scores[top_idx])]
        return [
            SearchHit(
                doc_id=self._mem_ids[i],
                text=self._mem_texts[i],
                score=float(scores[i]),
                metadata=dict(self._mem_metadata[i]),
            )
            for i in ordered
        ]

    def _ensure_active(self) -> None:
        if not self._active or self._model is None:
            raise RuntimeError(
                f"{PLUGIN_NAME} is not active. Call activate() first."
            )


_PLUGIN_SINGLETON: Optional[MultilingualSemanticSearchPlugin] = None


def register(cfg: Optional[HomieConfig] = None) -> MultilingualSemanticSearchPlugin:
    """Return (and cache) the plugin instance for Homie's plugin loader."""
    global _PLUGIN_SINGLETON
    if _PLUGIN_SINGLETON is None:
        _PLUGIN_SINGLETON = MultilingualSemanticSearchPlugin(cfg=cfg)
    return _PLUGIN_SINGLETON


def activate(cfg: Optional[HomieConfig] = None, allow_download: bool = False) -> MultilingualSemanticSearchPlugin:
    """Module-level convenience wrapper around ``register().activate()``."""
    plugin = register(cfg=cfg)
    plugin.activate(allow_download=allow_download)
    return plugin


def deactivate() -> None:
    """Module-level convenience wrapper around ``register().deactivate()``."""
    global _PLUGIN_SINGLETON
    if _PLUGIN_SINGLETON is not None:
        _PLUGIN_SINGLETON.deactivate()
        _PLUGIN_SINGLETON = None


__all__ = [
    "MultilingualSemanticSearchPlugin",
    "SearchHit",
    "PLUGIN_NAME",
    "MODEL_ID",
    "EMBEDDING_DIM",
    "register",
    "activate",
    "deactivate",
]
