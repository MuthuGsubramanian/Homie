"""HOMIE plugin: Nomic Embed Text v1.5 — long-context local embeddings.

Provides 768-dim sentence embeddings with up to 8192-token context using
``nomic-ai/nomic-embed-text-v1.5``. The long context lets Homie index whole
documents (READMEs, transcripts, long notes) into its ChromaDB RAG store
without aggressive chunking, and the Matryoshka representation lets callers
trade vector size for memory/disk by truncating to any leading dimension
(≥ 64) and re-normalising.

Design goals
------------
* **Local-first.** Inference runs on the host via ``sentence-transformers``.
  The only network access is the implicit Hugging Face Hub fetch on first
  load; set ``HOMIE_OFFLINE=1`` (or ``NomicEmbedConfig(allow_download=False)``)
  to require pre-cached weights and forbid downloads.
* **Task prefixes.** Nomic v1.5 expects ``search_query`` / ``search_document``
  / ``classification`` / ``clustering`` prefixes; this plugin applies them
  automatically via ``embed_query`` / ``embed_documents`` and validates any
  explicit prefix passed to ``embed``.
* **Plugin lifecycle.** Exposes module-level ``activate()`` / ``deactivate()``
  and a ``register()`` helper for Homie's plugin loader, mirroring the
  conventions used elsewhere in ``homie/`` (e.g. ``controller/automations``).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

PLUGIN_NAME = "nomic_embed_text_v15"
PLUGIN_VERSION = "1.0.0"
MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
MAX_SEQ_LENGTH = 8192
FULL_DIM = 768
MIN_MATRYOSHKA_DIM = 64
ALLOWED_PREFIXES = (
    "search_query",
    "search_document",
    "classification",
    "clustering",
)


@dataclass
class NomicEmbedConfig:
    """Runtime configuration for the Nomic embedding plugin.

    Attributes:
        model_id: Hugging Face model id; pin to a revision if you mirror it.
        device: ``"cpu"``, ``"cuda"``, or ``"mps"``. Defaults to CPU so the
            plugin loads on any Homie host.
        max_seq_length: Tokens per input; capped at the model's 8192.
        output_dim: Matryoshka output size (64–768). Smaller ⇒ cheaper RAG
            index at a small recall cost.
        default_prefix: Prefix used by the generic ``embed()`` entrypoint.
        normalize: L2-normalise vectors (cosine-ready, ChromaDB friendly).
        allow_download: If False, refuse Hub downloads and only use the local
            HF cache. Driven by ``HOMIE_OFFLINE`` by default.
    """

    model_id: str = MODEL_ID
    device: str = "cpu"
    max_seq_length: int = MAX_SEQ_LENGTH
    output_dim: int = FULL_DIM
    default_prefix: str = "search_document"
    normalize: bool = True
    allow_download: bool = field(
        default_factory=lambda: os.getenv("HOMIE_OFFLINE", "0") != "1"
    )

    def __post_init__(self) -> None:
        if self.default_prefix not in ALLOWED_PREFIXES:
            raise ValueError(
                f"default_prefix must be one of {ALLOWED_PREFIXES}, "
                f"got {self.default_prefix!r}"
            )
        if not MIN_MATRYOSHKA_DIM <= self.output_dim <= FULL_DIM:
            raise ValueError(
                f"output_dim must be between {MIN_MATRYOSHKA_DIM} and {FULL_DIM}, "
                f"got {self.output_dim}"
            )
        if self.max_seq_length <= 0 or self.max_seq_length > MAX_SEQ_LENGTH:
            raise ValueError(
                f"max_seq_length must be in (0, {MAX_SEQ_LENGTH}], "
                f"got {self.max_seq_length}"
            )


class NomicEmbedPlugin:
    """Long-context embedding provider backed by nomic-embed-text-v1.5.

    Typical usage from Homie's RAG layer::

        plugin = register()                 # loads model once per process
        doc_vecs = plugin.embed_documents(chunks)
        query_vec = plugin.embed_query("how do I rotate my SSH key?")
    """

    def __init__(self, config: Optional[NomicEmbedConfig] = None) -> None:
        self.config = config or NomicEmbedConfig()
        self._model: Any = None  # sentence_transformers.SentenceTransformer

    # ------------------------------------------------------------------ lifecycle
    def activate(self) -> None:
        """Eagerly load the model. Safe to call multiple times."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "nomic_embed_text_v15 requires `sentence-transformers`. "
                "Install with: pip install sentence-transformers"
            ) from exc

        if not self.config.allow_download:
            # Block any Hub egress; user must have weights cached already.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        logger.info(
            "Loading %s on %s (offline=%s, dim=%d)",
            self.config.model_id,
            self.config.device,
            not self.config.allow_download,
            self.config.output_dim,
        )
        # ``trust_remote_code`` is required by nomic_bert's custom modeling code.
        self._model = SentenceTransformer(
            self.config.model_id,
            device=self.config.device,
            trust_remote_code=True,
        )
        self._model.max_seq_length = self.config.max_seq_length

    def deactivate(self) -> None:
        """Drop the model reference so the GC can reclaim its memory."""
        if self._model is None:
            return
        self._model = None
        logger.info("Unloaded %s", self.config.model_id)

    # ------------------------------------------------------------------ public API
    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        """Encode passages for indexing (uses ``search_document`` prefix)."""
        return self._encode(list(texts), prefix="search_document")

    def embed_query(self, text: str) -> List[float]:
        """Encode a single retrieval query (uses ``search_query`` prefix)."""
        return self._encode([text], prefix="search_query")[0]

    def embed(
        self,
        texts: Iterable[str],
        prefix: Optional[str] = None,
    ) -> List[List[float]]:
        """Encode arbitrary text with an explicit task prefix.

        Args:
            texts: Inputs to embed.
            prefix: One of ``search_query``, ``search_document``,
                ``classification``, ``clustering``. Falls back to
                ``config.default_prefix`` when ``None``.
        """
        return self._encode(list(texts), prefix=prefix or self.config.default_prefix)

    def info(self) -> Dict[str, Any]:
        """Return plugin metadata for ``:status``-style introspection."""
        return {
            "name": PLUGIN_NAME,
            "version": PLUGIN_VERSION,
            "model_id": self.config.model_id,
            "device": self.config.device,
            "max_seq_length": self.config.max_seq_length,
            "output_dim": self.config.output_dim,
            "normalize": self.config.normalize,
            "loaded": self._model is not None,
        }

    # ------------------------------------------------------------------ internals
    def _ensure_loaded(self) -> Any:
        if self._model is None:
            self.activate()
        return self._model

    @staticmethod
    def _apply_prefix(texts: Sequence[str], prefix: str) -> List[str]:
        if prefix not in ALLOWED_PREFIXES:
            raise ValueError(
                f"prefix must be one of {ALLOWED_PREFIXES}, got {prefix!r}"
            )
        return [f"{prefix}: {t}" for t in texts]

    def _encode(self, texts: List[str], prefix: str) -> List[List[float]]:
        if not texts:
            return []
        model = self._ensure_loaded()
        prefixed = self._apply_prefix(texts, prefix)
        vectors = model.encode(
            prefixed,
            normalize_embeddings=self.config.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return self._truncate(vectors)

    def _truncate(self, vectors: Any) -> List[List[float]]:
        """Matryoshka truncation: keep leading ``output_dim`` dims, re-normalise."""
        if self.config.output_dim >= FULL_DIM:
            return vectors.tolist()

        import numpy as np  # local import: numpy ships with sentence-transformers

        truncated = vectors[:, : self.config.output_dim]
        if self.config.normalize:
            norms = np.linalg.norm(truncated, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            truncated = truncated / norms
        return truncated.tolist()


# --------------------------------------------------------------------- module-level
_INSTANCE: Optional[NomicEmbedPlugin] = None


def register(config: Optional[NomicEmbedConfig] = None) -> NomicEmbedPlugin:
    """Create (or return) the process-wide plugin instance and activate it."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = NomicEmbedPlugin(config=config)
        _INSTANCE.activate()
    elif config is not None and config is not _INSTANCE.config:
        logger.warning(
            "nomic_embed_text_v15 already registered; ignoring new config. "
            "Call deactivate() first to swap configs."
        )
    return _INSTANCE


def activate() -> NomicEmbedPlugin:
    """Plugin-loader entrypoint: equivalent to ``register()`` with defaults."""
    return register()


def deactivate() -> None:
    """Plugin-loader entrypoint: unload and forget the registered instance."""
    global _INSTANCE
    if _INSTANCE is not None:
        _INSTANCE.deactivate()
        _INSTANCE = None


__all__ = [
    "PLUGIN_NAME",
    "PLUGIN_VERSION",
    "MODEL_ID",
    "ALLOWED_PREFIXES",
    "NomicEmbedConfig",
    "NomicEmbedPlugin",
    "register",
    "activate",
    "deactivate",
]
