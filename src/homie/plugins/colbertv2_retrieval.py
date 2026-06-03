"""ColBERTv2 late-interaction retrieval plugin for Homie.

Provides a high-precision document retrieval backend powered by
`colbert-ir/colbertv2.0`. Unlike single-vector dense retrievers (e.g. the
minilm/mpnet sentence-transformers used by Homie's default RAG), ColBERTv2
produces *token-level* embeddings and ranks documents using late-interaction
MaxSim scoring. This gives noticeably better precision on long-form
document Q&A and is intended to be layered on top of Homie's existing
ChromaDB-backed RAG (e.g. as a re-ranker over a candidate set, or as a
standalone high-precision index for a small knowledge base).

Key properties (kept consistent with the rest of Homie):
- Runs fully locally: model weights are loaded from the local HuggingFace
  cache only. Network downloads are opt-in via ``allow_download=True``.
- Persists its token-embedding index to ``~/.homie/colbertv2_index`` so that
  re-activation does not require re-encoding documents.
- Exposes a Homie-style ``activate()`` / ``deactivate()`` pair plus a
  ``register()`` factory so the plugin loader can use either entry point.

Dependencies (installed by the user, not by this plugin):
    pip install torch transformers
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "colbert-ir/colbertv2.0"
DEFAULT_INDEX_DIR = Path.home() / ".homie" / "colbertv2_index"
DEFAULT_MAX_TOKENS = 180


@dataclass
class ColBERTDocument:
    """A single document to be indexed by the ColBERTv2 plugin."""

    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ColBERTv2Plugin:
    """Late-interaction retrieval plugin backed by ``colbert-ir/colbertv2.0``.

    Encodes each document into a matrix of normalized token embeddings and
    scores queries with the MaxSim late-interaction operator::

        score(Q, D) = sum_i max_j (q_i . d_j)

    where Q is the query token-embedding matrix and D is the document
    token-embedding matrix (both L2-normalized).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        index_dir: Optional[Path] = None,
        device: str = "cpu",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        allow_download: bool = False,
    ) -> None:
        self.model_name = model_name
        self.index_dir = Path(index_dir or DEFAULT_INDEX_DIR)
        self.device = device
        self.max_tokens = max_tokens
        self.allow_download = allow_download
        self._tokenizer: Any = None
        self._model: Any = None
        self._index: List[Tuple[ColBERTDocument, Any]] = []
        self._active: bool = False

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def activate(self) -> None:
        """Load tokenizer + model and any persisted index from disk."""
        try:
            import torch  # noqa: F401
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "ColBERTv2 plugin requires 'torch' and 'transformers'. "
                "Install with: pip install torch transformers"
            ) from exc

        local_files_only = not self.allow_download
        logger.info(
            "Loading ColBERTv2 model %s (local_files_only=%s, device=%s)",
            self.model_name,
            local_files_only,
            self.device,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, local_files_only=local_files_only
        )
        self._model = AutoModel.from_pretrained(
            self.model_name, local_files_only=local_files_only
        ).to(self.device)
        self._model.eval()

        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._load_index()
        self._active = True
        logger.info(
            "ColBERTv2 plugin active. Documents in index: %d", len(self._index)
        )

    def deactivate(self) -> None:
        """Persist the index to disk and release model resources."""
        try:
            self._save_index()
        finally:
            self._tokenizer = None
            self._model = None
            self._index.clear()
            self._active = False
            logger.info("ColBERTv2 plugin deactivated")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def add_documents(self, docs: Iterable[ColBERTDocument]) -> int:
        """Encode and add documents to the in-memory index.

        Returns the number of documents added.
        """
        self._ensure_active()
        added = 0
        for doc in docs:
            emb = self._encode(doc.text, is_query=False)
            self._index.append((doc, emb))
            added += 1
        logger.info(
            "ColBERTv2 indexed %d documents (total=%d)", added, len(self._index)
        )
        return added

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Run a MaxSim late-interaction search and return ranked results."""
        self._ensure_active()
        if not self._index:
            return []

        q_emb = self._encode(query, is_query=True)
        scored: List[Tuple[float, ColBERTDocument]] = []
        for doc, d_emb in self._index:
            sim = q_emb @ d_emb.T  # (Lq, Ld)
            score = float(sim.max(dim=1).values.sum().item())
            scored.append((score, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "doc_id": doc.doc_id,
                "score": score,
                "text": doc.text,
                "metadata": doc.metadata,
            }
            for score, doc in scored[:top_k]
        ]

    def rerank(
        self,
        query: str,
        candidates: Iterable[ColBERTDocument],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Rerank an externally-provided candidate set without polluting the index.

        Useful when used as a precision layer on top of Homie's existing
        ChromaDB retrieval: take the top-N from Chroma, hand them here for
        late-interaction rescoring.
        """
        self._ensure_active()
        q_emb = self._encode(query, is_query=True)
        scored: List[Tuple[float, ColBERTDocument]] = []
        for doc in candidates:
            d_emb = self._encode(doc.text, is_query=False)
            sim = q_emb @ d_emb.T
            score = float(sim.max(dim=1).values.sum().item())
            scored.append((score, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "doc_id": doc.doc_id,
                "score": score,
                "text": doc.text,
                "metadata": doc.metadata,
            }
            for score, doc in scored[:top_k]
        ]

    def clear_index(self) -> None:
        """Drop the in-memory index and remove the persisted index file."""
        self._index.clear()
        index_file = self._index_path()
        if index_file.exists():
            index_file.unlink()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _ensure_active(self) -> None:
        if not self._active:
            raise RuntimeError(
                "ColBERTv2 plugin is not active. Call activate() first."
            )

    def _encode(self, text: str, is_query: bool) -> Any:
        """Produce L2-normalized token embeddings for ``text``."""
        import torch

        marker = "[Q] " if is_query else "[D] "
        inputs = self._tokenizer(
            marker + text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_tokens,
            padding=False,
        ).to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        embeddings = outputs.last_hidden_state[0]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        return embeddings.cpu()

    def _index_path(self) -> Path:
        return self.index_dir / "index.pkl"

    def _save_index(self) -> None:
        if not self._index:
            return
        try:
            payload = [
                {
                    "doc_id": doc.doc_id,
                    "text": doc.text,
                    "metadata": doc.metadata,
                    "embedding": emb.numpy(),
                }
                for doc, emb in self._index
            ]
            with self._index_path().open("wb") as fh:
                pickle.dump(payload, fh)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to persist ColBERTv2 index")

    def _load_index(self) -> None:
        path = self._index_path()
        if not path.exists():
            return
        try:
            import torch

            with path.open("rb") as fh:
                payload = pickle.load(fh)
            self._index = [
                (
                    ColBERTDocument(
                        doc_id=row["doc_id"],
                        text=row["text"],
                        metadata=row.get("metadata", {}),
                    ),
                    torch.from_numpy(row["embedding"]),
                )
                for row in payload
            ]
        except Exception:  # noqa: BLE001
            logger.exception("Failed to load ColBERTv2 index; starting empty")
            self._index = []


# ---------------------------------------------------------------------- #
# Homie plugin entry points
# ---------------------------------------------------------------------- #
_PLUGIN: Optional[ColBERTv2Plugin] = None


def register(
    model_name: str = DEFAULT_MODEL,
    index_dir: Optional[str] = None,
    device: str = "cpu",
    allow_download: bool = False,
) -> ColBERTv2Plugin:
    """Module-level factory used by Homie's plugin loader."""
    global _PLUGIN
    _PLUGIN = ColBERTv2Plugin(
        model_name=model_name,
        index_dir=Path(index_dir) if index_dir else None,
        device=device,
        allow_download=allow_download,
    )
    return _PLUGIN


def activate(**kwargs: Any) -> ColBERTv2Plugin:
    """Construct (if needed) and activate the plugin singleton."""
    plugin = _PLUGIN if _PLUGIN is not None else register(**kwargs)
    plugin.activate()
    return plugin


def deactivate() -> None:
    """Deactivate and tear down the plugin singleton."""
    global _PLUGIN
    if _PLUGIN is not None:
        _PLUGIN.deactivate()
        _PLUGIN = None


__all__ = [
    "ColBERTDocument",
    "ColBERTv2Plugin",
    "register",
    "activate",
    "deactivate",
]
