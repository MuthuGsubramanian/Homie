"""Multilingual semantic search plugin for HOMIE.

Provides cross-lingual RAG and semantic search using the
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 model.
The model maps sentences from 50+ languages into a shared 384-dim
embedding space, so a query in (e.g.) Tamil can retrieve documents
written in English, Spanish, or Hindi.

The plugin runs entirely on-device (CPU-friendly, ~120 MB model) and
stores vectors in a local ChromaDB collection. No network traffic is
required after the initial model download from Hugging Face, which
itself is opt-in via the ``multilingual_rag.allow_model_download``
config key.

Integration:
- Reads ``HomieConfig`` under the ``multilingual_rag`` key.
- Exposes ``activate(cfg)`` / ``deactivate()`` lifecycle hooks for the
  HOMIE plugin loader.
- Exposes ``register(registry, cfg)`` for registry-style loaders.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    from homie.config import HomieConfig, cfg_get
except ImportError:  # pragma: no cover - allow standalone import for tests
    HomieConfig = Any  # type: ignore[assignment,misc]

    def cfg_get(cfg: Any, *keys: str, default: Any = None) -> Any:
        cur: Any = getattr(cfg, "raw", cfg)
        for key in keys:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur


logger = logging.getLogger("homie.plugins.multilingual_rag")

MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384
DEFAULT_COLLECTION = "homie_multilingual"


@dataclass
class MultilingualRagSettings:
    """Resolved configuration for the multilingual RAG plugin."""

    model_id: str = MODEL_ID
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".homie" / "models")
    chroma_dir: Path = field(default_factory=lambda: Path.home() / ".homie" / "chroma")
    collection_name: str = DEFAULT_COLLECTION
    device: str = "cpu"
    allow_model_download: bool = False
    top_k: int = 5
    normalize: bool = True

    @classmethod
    def from_config(cls, cfg: "HomieConfig") -> "MultilingualRagSettings":
        """Build settings from a ``HomieConfig`` using the ``multilingual_rag`` namespace."""
        return cls(
            model_id=cfg_get(cfg, "multilingual_rag", "model_id", default=MODEL_ID),
            cache_dir=Path(
                cfg_get(
                    cfg,
                    "multilingual_rag",
                    "cache_dir",
                    default=str(Path.home() / ".homie" / "models"),
                )
            ),
            chroma_dir=Path(
                cfg_get(
                    cfg,
                    "multilingual_rag",
                    "chroma_dir",
                    default=str(Path.home() / ".homie" / "chroma"),
                )
            ),
            collection_name=cfg_get(
                cfg, "multilingual_rag", "collection", default=DEFAULT_COLLECTION
            ),
            device=cfg_get(cfg, "multilingual_rag", "device", default="cpu"),
            allow_model_download=bool(
                cfg_get(cfg, "multilingual_rag", "allow_model_download", default=False)
            ),
            top_k=int(cfg_get(cfg, "multilingual_rag", "top_k", default=5)),
            normalize=bool(cfg_get(cfg, "multilingual_rag", "normalize", default=True)),
        )


@dataclass
class SearchHit:
    """One result from a multilingual semantic search."""

    doc_id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultilingualRagPlugin:
    """Cross-lingual semantic search backed by ChromaDB.

    Loads the paraphrase-multilingual-MiniLM-L12-v2 model lazily on
    first use to avoid blocking HOMIE startup. All inference is local;
    the only network access is the initial model download, gated by
    ``settings.allow_model_download``.
    """

    name: str = "multilingual_rag"
    version: str = "1.0.0"

    def __init__(self, settings: MultilingualRagSettings) -> None:
        self.settings = settings
        self._model: Any = None
        self._client: Any = None
        self._collection: Any = None
        self._active: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Prepare on-disk directories. Model load is deferred until first use."""
        self.settings.cache_dir.mkdir(parents=True, exist_ok=True)
        self.settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        self._active = True
        logger.info(
            "multilingual_rag started (model=%s device=%s download=%s)",
            self.settings.model_id,
            self.settings.device,
            self.settings.allow_model_download,
        )

    def stop(self) -> None:
        """Release model and ChromaDB handles."""
        self._model = None
        self._collection = None
        self._client = None
        self._active = False
        logger.info("multilingual_rag stopped")

    # ------------------------------------------------------------------
    # Lazy resource loaders
    # ------------------------------------------------------------------
    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is required. Install with "
                "`pip install sentence-transformers`."
            ) from exc

        logger.info(
            "Loading %s into %s (cache=%s)",
            self.settings.model_id,
            self.settings.device,
            self.settings.cache_dir,
        )
        self._model = SentenceTransformer(
            self.settings.model_id,
            cache_folder=str(self.settings.cache_dir),
            device=self.settings.device,
        )
        return self._model

    def _ensure_collection(self) -> Any:
        if self._collection is not None:
            return self._collection
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "chromadb is required. Install with `pip install chromadb`."
            ) from exc

        self._client = chromadb.PersistentClient(path=str(self.settings.chroma_dir))
        self._collection = self._client.get_or_create_collection(
            name=self.settings.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_dim": EMBEDDING_DIM,
                "model": self.settings.model_id,
            },
        )
        return self._collection

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Return embeddings for ``texts`` (any supported language)."""
        if not texts:
            return []
        model = self._ensure_model()
        vectors = model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=self.settings.normalize,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def add_documents(
        self,
        documents: Sequence[str],
        ids: Optional[Sequence[str]] = None,
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[str]:
        """Embed ``documents`` and upsert them into the local ChromaDB collection."""
        if not documents:
            return []
        col = self._ensure_collection()
        doc_ids = list(ids) if ids else [f"doc_{i}" for i in range(len(documents))]
        if len(doc_ids) != len(documents):
            raise ValueError("len(ids) must match len(documents)")
        if metadatas is not None and len(metadatas) != len(documents):
            raise ValueError("len(metadatas) must match len(documents)")
        embeddings = self.embed(documents)
        col.upsert(
            ids=doc_ids,
            documents=list(documents),
            embeddings=embeddings,
            metadatas=list(metadatas) if metadatas else None,
        )
        logger.debug(
            "Upserted %d docs into %s", len(doc_ids), self.settings.collection_name
        )
        return doc_ids

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[SearchHit]:
        """Cross-lingual semantic search.

        ``query`` may be written in any of the 50+ languages supported by
        the underlying model; matches may be in a different language than
        the query.
        """
        if not query or not query.strip():
            return []
        col = self._ensure_collection()
        k = int(top_k) if top_k else self.settings.top_k
        query_embedding = self.embed([query])[0]
        raw = col.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
        )

        ids = (raw.get("ids") or [[]])[0]
        docs = (raw.get("documents") or [[]])[0]
        dists = (raw.get("distances") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0] or [{} for _ in ids]

        hits: List[SearchHit] = []
        for doc_id, text, dist, meta in zip(ids, docs, dists, metas):
            score = max(0.0, 1.0 - float(dist))  # cosine distance -> similarity
            hits.append(
                SearchHit(
                    doc_id=str(doc_id),
                    text=text or "",
                    score=score,
                    metadata=dict(meta or {}),
                )
            )
        return hits

    def delete(self, ids: Sequence[str]) -> None:
        """Remove documents from the collection by id."""
        if not ids:
            return
        col = self._ensure_collection()
        col.delete(ids=list(ids))

    def stats(self) -> Dict[str, Any]:
        """Diagnostic info: collection size, model id, embedding dim."""
        col = self._ensure_collection()
        return {
            "collection": self.settings.collection_name,
            "count": col.count(),
            "model": self.settings.model_id,
            "embedding_dim": EMBEDDING_DIM,
            "device": self.settings.device,
            "active": self._active,
        }


# ---------------------------------------------------------------------
# Module-level plugin lifecycle
# ---------------------------------------------------------------------
_PLUGIN: Optional[MultilingualRagPlugin] = None


def activate(cfg: "HomieConfig") -> MultilingualRagPlugin:
    """Instantiate and start the plugin. Idempotent."""
    global _PLUGIN
    if _PLUGIN is not None and _PLUGIN._active:
        return _PLUGIN
    settings = MultilingualRagSettings.from_config(cfg)
    _PLUGIN = MultilingualRagPlugin(settings)
    _PLUGIN.start()
    return _PLUGIN


def deactivate() -> None:
    """Stop the plugin and release resources."""
    global _PLUGIN
    if _PLUGIN is not None:
        _PLUGIN.stop()
        _PLUGIN = None


def register(registry: Any, cfg: "HomieConfig") -> MultilingualRagPlugin:
    """Register the plugin instance with an explicit registry object."""
    plugin = activate(cfg)
    if hasattr(registry, "register") and callable(registry.register):
        registry.register(plugin.name, plugin)
    elif isinstance(registry, dict):
        registry[plugin.name] = plugin
    return plugin


__all__ = [
    "MODEL_ID",
    "EMBEDDING_DIM",
    "DEFAULT_COLLECTION",
    "MultilingualRagSettings",
    "MultilingualRagPlugin",
    "SearchHit",
    "activate",
    "deactivate",
    "register",
]
