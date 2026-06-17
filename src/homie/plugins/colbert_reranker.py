"""ColBERT late-interaction reranker plugin for Homie.

Homie's default RAG path retrieves chunks from ChromaDB using a single dense
vector per document (bi-encoder cosine similarity). That is fast but lossy:
collapsing a whole passage into one vector blurs fine-grained term matches, so
the top-k can miss the passage that actually answers the query.

This plugin adds an *optional* second stage: ColBERT-style late interaction
(colbert-ir/colbertv2.0). Instead of one vector per passage, ColBERT keeps one
vector per token and scores a (query, passage) pair with MaxSim --- for every
query token, take its best-matching passage token, then sum those maxima. This
recovers precise term-level evidence while only ever re-ranking the small
candidate set ChromaDB already returned, so the cost stays bounded.

Design notes / Homie conventions:
  * Pure-local by default. The model is loaded ONLY from an on-disk path or the
    local HuggingFace cache. Downloading weights touches the network and is
    therefore strictly opt-in via ``allow_download`` (config or env).
  * Heavy deps (torch / transformers) are imported lazily inside ``activate()``
    so importing this module never drags them in and the plugin degrades to a
    no-op when they are absent.
  * Reads its settings through the existing ``cfg_get`` accessor, mirroring the
    rest of the codebase.
  * ``activate()`` / ``deactivate()`` lifecycle plus a ``register()`` entry
    point for plugin-host discovery.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # Homie config accessor; tolerate standalone use in tests.
    from homie.config import HomieConfig, cfg_get
except Exception:  # noqa: BLE001 - allow import without the full package
    HomieConfig = Any  # type: ignore[assignment,misc]

    def cfg_get(cfg: Any, *keys: str, default: Any = None) -> Any:  # type: ignore[misc]
        cur: Any = getattr(cfg, "raw", cfg)
        for key in keys:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur


logger = logging.getLogger(__name__)

PLUGIN_NAME = "colbert_reranker"
DEFAULT_MODEL = "colbert-ir/colbertv2.0"


class ColbertRerankerError(Exception):
    """Raised when the reranker is asked to work but cannot."""


@dataclass
class RerankResult:
    """A single reranked candidate.

    Attributes:
        index: Position of this document in the *original* candidate list.
        document: The passage text that was scored.
        score: ColBERT MaxSim score (higher is more relevant).
        metadata: Any metadata carried alongside the candidate, untouched.
    """

    index: int
    document: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ColbertReranker:
    """Late-interaction reranker over ChromaDB candidates.

    The reranker never performs first-stage retrieval; it refines an existing
    candidate set. Call :meth:`activate` once, then :meth:`rerank` per query.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        allow_download: bool = False,
        device: Optional[str] = None,
        max_query_tokens: int = 32,
        max_doc_tokens: int = 180,
    ) -> None:
        self.model_name = model_name
        self.allow_download = allow_download
        self.device = device
        self.max_query_tokens = max_query_tokens
        self.max_doc_tokens = max_doc_tokens

        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._active = False

    # -- lifecycle ---------------------------------------------------------

    @property
    def active(self) -> bool:
        """True once the model is loaded and ready to score."""
        return self._active

    def activate(self) -> bool:
        """Load the ColBERT model locally. Returns True on success.

        Honors offline-first behavior: unless ``allow_download`` is set, this
        forces HuggingFace into offline mode so no network call can happen. A
        missing model or missing dependency is logged and yields False rather
        than raising, so the host can fall back to plain dense retrieval.
        """
        if self._active:
            return True
        try:
            import torch  # noqa: PLC0415 - lazy heavy import
            from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415
        except ImportError as exc:
            logger.warning("%s disabled: missing torch/transformers (%s)", PLUGIN_NAME, exc)
            return False

        if not self.allow_download:
            # Belt-and-suspenders: keep transformers fully offline.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, local_files_only=not self.allow_download
            )
            self._model = AutoModel.from_pretrained(
                self.model_name, local_files_only=not self.allow_download
            )
        except Exception as exc:  # noqa: BLE001 - model/cache problems are recoverable
            logger.warning(
                "%s disabled: could not load '%s' locally (%s). "
                "Set allow_download=true to fetch it once.",
                PLUGIN_NAME,
                self.model_name,
                exc,
            )
            return False

        self._torch = torch
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self.device)
        self._model.eval()
        self._active = True
        logger.info("%s active: model=%s device=%s", PLUGIN_NAME, self.model_name, self.device)
        return True

    def deactivate(self) -> None:
        """Release the model and free GPU memory if any was allocated."""
        self._tokenizer = None
        self._model = None
        if self._torch is not None and getattr(self._torch, "cuda", None) is not None:
            try:
                if self._torch.cuda.is_available():
                    self._torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                logger.debug("cuda cache clear skipped")
        self._torch = None
        self._active = False
        logger.info("%s deactivated", PLUGIN_NAME)

    # -- scoring -----------------------------------------------------------

    def _embed(self, text: str, max_tokens: int) -> Any:
        """Return L2-normalized per-token embeddings of shape (n_tokens, dim)."""
        torch = self._torch
        enc = self._tokenizer(
            text,
            truncation=True,
            max_length=max_tokens,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            out = self._model(**enc).last_hidden_state.squeeze(0)
        mask = enc["attention_mask"].squeeze(0).bool()
        out = out[mask]
        return torch.nn.functional.normalize(out, p=2, dim=-1)

    def _maxsim(self, query_emb: Any, doc_emb: Any) -> float:
        """ColBERT MaxSim: sum over query tokens of their best doc-token match."""
        if query_emb.shape[0] == 0 or doc_emb.shape[0] == 0:
            return 0.0
        sim = query_emb @ doc_emb.T  # (q_tokens, d_tokens)
        return float(sim.max(dim=1).values.sum().item())

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
        top_k: Optional[int] = None,
    ) -> List[RerankResult]:
        """Rerank ``documents`` for ``query`` by late-interaction score.

        Args:
            query: The user query / question.
            documents: Candidate passages (e.g. ChromaDB query results).
            metadatas: Optional per-document metadata, aligned with documents.
            top_k: If given, return only the top_k highest-scoring results.

        Returns:
            Results sorted by descending MaxSim score. Falls back to the input
            order (score 0.0) if the model is not active, so callers can wire
            this in unconditionally.
        """
        metas = list(metadatas or [{} for _ in documents])
        if not self._active:
            results = [
                RerankResult(index=i, document=doc, score=0.0, metadata=metas[i])
                for i, doc in enumerate(documents)
            ]
            return results[: top_k or len(results)]

        query_emb = self._embed(query, self.max_query_tokens)
        scored: List[RerankResult] = []
        for i, doc in enumerate(documents):
            doc_emb = self._embed(doc, self.max_doc_tokens)
            scored.append(
                RerankResult(
                    index=i,
                    document=doc,
                    score=self._maxsim(query_emb, doc_emb),
                    metadata=metas[i] if i < len(metas) else {},
                )
            )
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[: top_k or len(scored)]

    def rerank_chroma(
        self, query: str, chroma_results: Dict[str, Any], top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """Convenience wrapper for a raw ChromaDB ``query()`` result dict.

        Accepts the standard ``{"documents": [[...]], "metadatas": [[...]]}``
        shape (batched, first query only) and reranks it.
        """
        docs = (chroma_results.get("documents") or [[]])[0]
        metas = (chroma_results.get("metadatas") or [[]])[0] or [{} for _ in docs]
        return self.rerank(query, docs, metas, top_k=top_k)


def _build_from_config(cfg: Optional[HomieConfig]) -> ColbertReranker:
    """Construct a reranker from Homie config, with env opt-in for download."""
    model = cfg_get(cfg, "rag", "colbert", "model", default=DEFAULT_MODEL) if cfg else DEFAULT_MODEL
    cfg_download = bool(cfg_get(cfg, "rag", "colbert", "allow_download", default=False)) if cfg else False
    env_download = os.getenv("HOMIE_COLBERT_ALLOW_DOWNLOAD", "").lower() in {"1", "true", "yes"}
    device = cfg_get(cfg, "rag", "colbert", "device", default=None) if cfg else None
    return ColbertReranker(
        model_name=model,
        allow_download=cfg_download or env_download,
        device=device,
    )


def activate(cfg: Optional[HomieConfig] = None) -> Optional[ColbertReranker]:
    """Plugin entry point: build, load, and return a ready reranker.

    Returns the instance even if model load failed (it then acts as a
    pass-through), so the host can always hold a stable handle. Returns None
    only if the plugin is explicitly disabled in config.
    """
    if cfg is not None and not cfg_get(cfg, "rag", "colbert", "enabled", default=True):
        logger.info("%s is disabled via config", PLUGIN_NAME)
        return None
    reranker = _build_from_config(cfg)
    reranker.activate()
    return reranker


def deactivate(reranker: Optional[ColbertReranker]) -> None:
    """Plugin teardown: release model resources."""
    if reranker is not None:
        reranker.deactivate()


def register() -> Dict[str, Any]:
    """Describe this plugin for Homie's plugin host / discovery."""
    return {
        "name": PLUGIN_NAME,
        "kind": "rag_reranker",
        "description": "ColBERT v2 late-interaction reranking for ChromaDB candidates.",
        "activate": activate,
        "deactivate": deactivate,
        "factory": ColbertReranker,
        "requires": ["torch", "transformers"],
        "network": "opt-in (model download only)",
    }


__all__ = [
    "ColbertReranker",
    "RerankResult",
    "ColbertRerankerError",
    "activate",
    "deactivate",
    "register",
]
