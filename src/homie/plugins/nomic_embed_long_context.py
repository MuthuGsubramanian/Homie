"""Nomic Embed Text v1.5 plugin for Homie.

Provides long-context (up to 8192 tokens) text embeddings using the
``nomic-ai/nomic-embed-text-v1.5`` model. Designed for embedding long
local documents into Homie's ChromaDB RAG store without truncating
far more than needed.

The model is loaded lazily via ``sentence-transformers`` on first use
and runs entirely on-device (CPU or local GPU). The only network
activity is the initial model download from Hugging Face the first
time ``activate()`` is called with ``allow_download=True`` (default
is ``False`` â€” the plugin will refuse to fetch weights unless the
user opts in or the model is already cached locally).

Configuration (``homie.config.yaml``)::

    plugins:
      nomic_embed:
        enabled: true
        model_name: nomic-ai/nomic-embed-text-v1.5
        device: cpu                # or 'cuda' / 'mps'
        max_seq_length: 8192
        allow_download: false      # opt-in network access
        default_task: search_document   # or 'search_query', 'classification', 'clustering'
        chunk_overlap_tokens: 128
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sentence_transformers import SentenceTransformer

    from homie.config import HomieConfig

logger = logging.getLogger(__name__)

PLUGIN_NAME = "nomic_embed"
DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"
DEFAULT_MAX_SEQ_LENGTH = 8192
VALID_TASKS = (
    "search_document",
    "search_query",
    "classification",
    "clustering",
)
_TASK_PREFIX = {
    "search_document": "search_document: ",
    "search_query": "search_query: ",
    "classification": "classification: ",
    "clustering": "clustering: ",
}


@dataclass
class NomicEmbedSettings:
    """Resolved settings for the Nomic embedding plugin."""

    enabled: bool = False
    model_name: str = DEFAULT_MODEL
    device: str = "cpu"
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH
    allow_download: bool = False
    default_task: str = "search_document"
    chunk_overlap_tokens: int = 128
    cache_dir: Optional[Path] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: "HomieConfig") -> "NomicEmbedSettings":
        from homie.config import cfg_get

        raw = cfg_get(cfg, "plugins", PLUGIN_NAME, default={}) or {}
        task = str(raw.get("default_task", "search_document"))
        if task not in VALID_TASKS:
            raise ValueError(
                f"plugins.{PLUGIN_NAME}.default_task must be one of {VALID_TASKS}, got {task!r}"
            )
        cache = raw.get("cache_dir")
        return cls(
            enabled=bool(raw.get("enabled", False)),
            model_name=str(raw.get("model_name", DEFAULT_MODEL)),
            device=str(raw.get("device", "cpu")),
            max_seq_length=int(raw.get("max_seq_length", DEFAULT_MAX_SEQ_LENGTH)),
            allow_download=bool(raw.get("allow_download", False)),
            default_task=task,
            chunk_overlap_tokens=int(raw.get("chunk_overlap_tokens", 128)),
            cache_dir=Path(cache).expanduser().resolve() if cache else None,
            extra={k: v for k, v in raw.items() if k not in {
                "enabled", "model_name", "device", "max_seq_length",
                "allow_download", "default_task", "chunk_overlap_tokens", "cache_dir",
            }},
        )


class NomicEmbedError(RuntimeError):
    """Raised when the Nomic embedding plugin cannot fulfil a request."""


class NomicEmbedPlugin:
    """Long-context text embedding plugin backed by Nomic Embed v1.5.

    The plugin exposes a small, deliberate surface area so it can be
    wired into Homie's RAG layer (ChromaDB) without further glue:

    * :meth:`embed` â€” encode a batch of strings into vectors.
    * :meth:`embed_query` / :meth:`embed_documents` â€” task-prefixed
      convenience wrappers that match Nomic's recommended usage.
    * :meth:`chunk_text` â€” token-aware splitter that respects the
      configured ``max_seq_length`` and ``chunk_overlap_tokens``.
    """

    def __init__(self, settings: NomicEmbedSettings) -> None:
        self.settings = settings
        self._model: Optional["SentenceTransformer"] = None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def activate(self) -> None:
        """Load the model into memory.

        Honours ``allow_download``: if the weights are not already cached
        and the user has not opted in to a download, raises
        :class:`NomicEmbedError` rather than reaching out to the network.
        """
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - import guard
            raise NomicEmbedError(
                "sentence-transformers is required for the nomic_embed plugin. "
                "Install it with `pip install sentence-transformers`."
            ) from exc

        cache_folder = (
            str(self.settings.cache_dir) if self.settings.cache_dir else None
        )
        if not self.settings.allow_download and not self._is_model_cached(cache_folder):
            raise NomicEmbedError(
                f"Model {self.settings.model_name!r} is not cached locally and "
                "plugins.nomic_embed.allow_download is false. Enable it once to fetch, "
                "then turn it off to keep Homie offline."
            )

        logger.info(
            "Loading Nomic embedder %s on %s (max_seq_length=%d)",
            self.settings.model_name,
            self.settings.device,
            self.settings.max_seq_length,
        )
        self._model = SentenceTransformer(
            self.settings.model_name,
            device=self.settings.device,
            cache_folder=cache_folder,
            trust_remote_code=True,
        )
        # Nomic Embed v1.5 supports up to 8192 tokens; sentence-transformers
        # otherwise defaults to a much smaller window.
        self._model.max_seq_length = self.settings.max_seq_length

    def deactivate(self) -> None:
        """Release the model and free associated GPU/CPU memory."""
        if self._model is None:
            return
        logger.info("Unloading Nomic embedder %s", self.settings.model_name)
        self._model = None
        try:
            import gc

            gc.collect()
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - cleanup is best-effort
            logger.debug("Torch cache cleanup skipped", exc_info=True)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def embed(
        self,
        texts: Sequence[str],
        task: Optional[str] = None,
        normalize: bool = True,
        batch_size: int = 8,
    ) -> List[List[float]]:
        """Embed a batch of strings using the configured Nomic model.

        :param texts: Strings to encode.
        :param task: Override the configured task prefix.
        :param normalize: L2-normalise vectors (recommended for cosine
            similarity stores like ChromaDB).
        :param batch_size: Forwarded to ``SentenceTransformer.encode``.
        """
        if not texts:
            return []
        model = self._require_model()
        task_name = self._validated_task(task)
        prefixed = [_TASK_PREFIX[task_name] + (t or "") for t in texts]
        vectors = model.encode(
            prefixed,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [vec.tolist() for vec in vectors]

    def embed_query(self, query: str, normalize: bool = True) -> List[float]:
        """Embed a single retrieval query with the ``search_query`` prefix."""
        result = self.embed([query], task="search_query", normalize=normalize)
        return result[0]

    def embed_documents(
        self,
        documents: Sequence[str],
        normalize: bool = True,
        batch_size: int = 8,
    ) -> List[List[float]]:
        """Embed corpus documents with the ``search_document`` prefix."""
        return self.embed(
            documents,
            task="search_document",
            normalize=normalize,
            batch_size=batch_size,
        )

    def chunk_text(self, text: str) -> List[str]:
        """Split ``text`` into overlapping windows that fit ``max_seq_length``.

        Uses the model's own tokenizer so chunk boundaries respect the real
        token budget. Falls back to a whitespace heuristic if the tokenizer
        is unavailable.
        """
        if not text:
            return []
        model = self._require_model()
        tokenizer = getattr(model, "tokenizer", None)
        window = self.settings.max_seq_length
        overlap = max(0, min(self.settings.chunk_overlap_tokens, window // 2))

        if tokenizer is None:
            return _whitespace_chunks(text, approx_tokens=window, overlap=overlap)

        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= window:
            return [text]

        chunks: List[str] = []
        step = max(1, window - overlap)
        for start in range(0, len(token_ids), step):
            piece_ids = token_ids[start : start + window]
            if not piece_ids:
                break
            chunks.append(tokenizer.decode(piece_ids, skip_special_tokens=True))
            if start + window >= len(token_ids):
                break
        return chunks

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _require_model(self) -> "SentenceTransformer":
        if self._model is None:
            raise NomicEmbedError(
                "NomicEmbedPlugin.activate() must be called before embedding."
            )
        return self._model

    def _validated_task(self, task: Optional[str]) -> str:
        if task is None:
            return self.settings.default_task
        if task not in VALID_TASKS:
            raise ValueError(f"task must be one of {VALID_TASKS}, got {task!r}")
        return task

    def _is_model_cached(self, cache_folder: Optional[str]) -> bool:
        candidates: List[Path] = []
        if cache_folder:
            candidates.append(Path(cache_folder))
        hf_home = os.getenv("HF_HOME")
        if hf_home:
            candidates.append(Path(hf_home) / "hub")
        candidates.append(Path.home() / ".cache" / "huggingface" / "hub")

        slug = "models--" + self.settings.model_name.replace("/", "--")
        return any((base / slug).exists() for base in candidates if base)


def _whitespace_chunks(text: str, approx_tokens: int, overlap: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    # ~0.75 words per token is a conservative approximation for English.
    words_per_chunk = max(1, int(approx_tokens * 0.75))
    step = max(1, words_per_chunk - int(overlap * 0.75))
    chunks: List[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + words_per_chunk]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + words_per_chunk >= len(words):
            break
    return chunks


# ----------------------------------------------------------------------
# Homie plugin entry points
# ----------------------------------------------------------------------
_active_plugin: Optional[NomicEmbedPlugin] = None


def register(cfg: "HomieConfig") -> Optional[NomicEmbedPlugin]:
    """Homie plugin entry point â€” instantiate and activate from config.

    Returns ``None`` when ``plugins.nomic_embed.enabled`` is false, so
    callers can skip wiring without special-casing.
    """
    settings = NomicEmbedSettings.from_config(cfg)
    if not settings.enabled:
        logger.debug("nomic_embed plugin disabled via config")
        return None
    plugin = NomicEmbedPlugin(settings)
    activate(plugin)
    return plugin


def activate(plugin: Optional[NomicEmbedPlugin] = None) -> NomicEmbedPlugin:
    """Activate (and remember) a plugin instance.

    When called without arguments, builds a default instance using the
    plugin's own defaults â€” useful for ad-hoc REPL exploration.
    """
    global _active_plugin
    if plugin is None:
        plugin = NomicEmbedPlugin(NomicEmbedSettings(enabled=True))
    plugin.activate()
    _active_plugin = plugin
    return plugin


def deactivate() -> None:
    """Deactivate the currently registered plugin, if any."""
    global _active_plugin
    if _active_plugin is not None:
        _active_plugin.deactivate()
        _active_plugin = None


def get_active_plugin() -> Optional[NomicEmbedPlugin]:
    """Return the currently active plugin instance, if registered."""
    return _active_plugin


__all__ = [
    "PLUGIN_NAME",
    "DEFAULT_MODEL",
    "VALID_TASKS",
    "NomicEmbedSettings",
    "NomicEmbedError",
    "NomicEmbedPlugin",
    "register",
    "activate",
    "deactivate",
    "get_active_plugin",
]
