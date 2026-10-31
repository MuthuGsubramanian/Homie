"""Multilingual sentence embeddings plugin for Homie.

Wraps ``sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`` to give
Homie's local RAG pipeline non-English document understanding. The model maps
sentences from ~50 languages into a shared 384-dimensional vector space, so a
query in one language can retrieve passages stored in another.

The plugin is fully local once the weights have been cached. The first call to
``activate()`` will download the model via ``sentence-transformers``; pass
``allow_download=False`` (or set ``HOMIE_OFFLINE=1`` in the environment) to
refuse network access and rely on a pre-cached snapshot.

Integration points:
    * Registers an ``embedder`` provider named ``multilingual-minilm-l12`` that
      ChromaDB-backed RAG components can consume.
    * Exposes ``encode()`` / ``similarity()`` helpers callable from other
      Homie modules or the AI planner.
    * Cleanly releases GPU/CPU memory on ``deactivate()``.
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PROVIDER_NAME = "multilingual-minilm-l12"
EMBEDDING_DIM = 384

_REGISTRY: Dict[str, Any] = {}
_REGISTRY_LOCK = threading.Lock()


@dataclass
class MultilingualEmbedderConfig:
    """Runtime configuration for the multilingual embedder.

    Attributes:
        model_id: Hugging Face model identifier. Override only if you have
            mirrored the weights to a different repo.
        device: ``cpu``, ``cuda``, ``mps`` or ``auto`` (default).
        batch_size: Maximum sentences per encode batch.
        normalize: If True, returned vectors are L2-normalized (recommended
            for cosine-similarity search such as ChromaDB defaults).
        allow_download: If False, the loader refuses to fetch the model from
            the network and requires a pre-existing local cache.
        cache_dir: Optional override for the sentence-transformers cache.
    """

    model_id: str = MODEL_ID
    device: str = "auto"
    batch_size: int = 32
    normalize: bool = True
    allow_download: bool = True
    cache_dir: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_homie_config(cls, cfg: Any) -> "MultilingualEmbedderConfig":
        """Build a config from a ``HomieConfig`` instance.

        Reads the ``plugins.multilingual_embeddings`` section if present and
        falls back to defaults otherwise. Never raises if keys are missing.
        """
        try:
            from homie.config import cfg_get  # type: ignore
        except Exception:  # pragma: no cover - homie not importable
            return cls()

        section = cfg_get(cfg, "plugins", "multilingual_embeddings", default={}) or {}
        return cls(
            model_id=section.get("model_id", MODEL_ID),
            device=section.get("device", "auto"),
            batch_size=int(section.get("batch_size", 32)),
            normalize=bool(section.get("normalize", True)),
            allow_download=bool(section.get("allow_download", True)),
            cache_dir=section.get("cache_dir"),
            extra=dict(section.get("extra", {}) or {}),
        )


class MultilingualEmbedderPlugin:
    """Loads and serves the multilingual MiniLM sentence encoder.

    The model is loaded lazily on the first call to ``encode`` (or eagerly via
    ``activate``). All public methods are thread-safe.
    """

    name: str = PROVIDER_NAME
    embedding_dim: int = EMBEDDING_DIM

    def __init__(self, config: Optional[MultilingualEmbedderConfig] = None) -> None:
        self.config = config or MultilingualEmbedderConfig()
        self._model: Any = None
        self._lock = threading.Lock()

    def _resolve_device(self) -> str:
        device = self.config.device
        if device != "auto":
            return device
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except Exception:  # noqa: BLE001
            pass
        return "cpu"

    def load(self) -> None:
        """Load the underlying SentenceTransformer model if not already loaded."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "sentence-transformers is required for the multilingual "
                    "embeddings plugin. Install with: pip install sentence-transformers"
                ) from exc

            offline_env = os.getenv("HOMIE_OFFLINE", "").lower() in {"1", "true", "yes"}
            if offline_env or not self.config.allow_download:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

            device = self._resolve_device()
            logger.info("Loading %s on device=%s", self.config.model_id, device)
            self._model = SentenceTransformer(
                self.config.model_id,
                device=device,
                cache_folder=self.config.cache_dir,
            )

    def unload(self) -> None:
        """Release the model and best-effort free GPU memory."""
        with self._lock:
            self._model = None
        try:
            import gc

            gc.collect()
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def encode(
        self,
        texts: Sequence[str] | str,
        *,
        batch_size: Optional[int] = None,
        normalize: Optional[bool] = None,
    ) -> List[List[float]]:
        """Encode one or more texts into embedding vectors.

        Args:
            texts: A single string or an iterable of strings.
            batch_size: Override the configured batch size.
            normalize: Override the configured L2-normalization flag.

        Returns:
            A list of float lists, one per input text. A bare string input
            still yields a list of length 1 for consistency.
        """
        if isinstance(texts, str):
            inputs: List[str] = [texts]
        else:
            inputs = [str(t) for t in texts]
        if not inputs:
            return []

        self.load()
        bs = int(batch_size or self.config.batch_size)
        norm = self.config.normalize if normalize is None else bool(normalize)
        vectors = self._model.encode(
            inputs,
            batch_size=bs,
            normalize_embeddings=norm,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [list(map(float, v)) for v in vectors]

    def similarity(self, a: str, b: str) -> float:
        """Cosine similarity between two strings in [-1.0, 1.0].

        Forces normalization so the dot product equals cosine similarity.
        """
        vec_a, vec_b = self.encode([a, b], normalize=True)
        return float(sum(x * y for x, y in zip(vec_a, vec_b)))

    def __call__(self, texts: Sequence[str]) -> List[List[float]]:
        """ChromaDB-compatible embedding function signature."""
        return self.encode(texts)


def register(
    cfg: Any = None,
    registry: Optional[Dict[str, Any]] = None,
) -> MultilingualEmbedderPlugin:
    """Register the plugin with Homie.

    Args:
        cfg: Optional ``HomieConfig`` used to source plugin settings.
        registry: Optional registry dict to insert the provider into. When
            omitted, an internal module-level registry is used.

    Returns:
        The registered :class:`MultilingualEmbedderPlugin` instance.
    """
    config = MultilingualEmbedderConfig.from_homie_config(cfg) if cfg is not None else MultilingualEmbedderConfig()
    plugin = MultilingualEmbedderPlugin(config=config)

    target = registry if registry is not None else _REGISTRY
    with _REGISTRY_LOCK:
        target[PROVIDER_NAME] = plugin
    logger.info("Registered embedder provider '%s'", PROVIDER_NAME)
    return plugin


def activate(
    cfg: Any = None,
    *,
    eager_load: bool = False,
    registry: Optional[Dict[str, Any]] = None,
) -> MultilingualEmbedderPlugin:
    """Activate the plugin and optionally pre-load the model.

    Args:
        cfg: Optional ``HomieConfig``.
        eager_load: If True, download/load the model immediately rather than
            on the first ``encode()`` call.
        registry: Optional external registry dict.
    """
    plugin = register(cfg=cfg, registry=registry)
    if eager_load:
        plugin.load()
    return plugin


def deactivate(registry: Optional[Dict[str, Any]] = None) -> None:
    """Deactivate the plugin, releasing the model and registry entry."""
    target = registry if registry is not None else _REGISTRY
    with _REGISTRY_LOCK:
        plugin = target.pop(PROVIDER_NAME, None)
    if isinstance(plugin, MultilingualEmbedderPlugin):
        plugin.unload()
    logger.info("Deactivated embedder provider '%s'", PROVIDER_NAME)


def get_plugin(registry: Optional[Dict[str, Any]] = None) -> Optional[MultilingualEmbedderPlugin]:
    """Return the currently registered plugin instance, if any."""
    target = registry if registry is not None else _REGISTRY
    with _REGISTRY_LOCK:
        return target.get(PROVIDER_NAME)


__all__ = [
    "MODEL_ID",
    "PROVIDER_NAME",
    "EMBEDDING_DIM",
    "MultilingualEmbedderConfig",
    "MultilingualEmbedderPlugin",
    "register",
    "activate",
    "deactivate",
    "get_plugin",
]
