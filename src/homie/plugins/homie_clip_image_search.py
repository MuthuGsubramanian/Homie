"""
homie_clip_image_search.py
==========================

CLIP-powered on-device image search plugin for Homie.

This plugin uses ``openai/clip-vit-base-patch32`` to perform zero-shot
text-to-image search across a local folder of images and zero-shot image
classification against an arbitrary label set. All inference runs locally
via the HuggingFace ``transformers`` library; no network calls are made
after the model weights have been cached on disk.

Typical use cases inside Homie:
  - "find photos of my dog on the couch"
  - "show me screenshots that contain code"
  - "label this image as receipt vs. business card vs. id"

Heavy ML dependencies (``transformers``, ``torch``, ``Pillow``) are imported
lazily inside :meth:`CLIPImageSearchPlugin._ensure_model`, so the rest of
Homie can start without them. A clear ``RuntimeError`` is raised the first
time the plugin is actually used if they are missing.

The plugin keeps an in-process embedding cache keyed by ``(path, mtime_ns)``
to avoid recomputing image embeddings between queries. Persisting embeddings
to ChromaDB is an obvious extension but is intentionally left out of this
single-file plugin to keep its surface area small.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

LOGGER = logging.getLogger("homie.plugins.clip_image_search")

DEFAULT_MODEL_ID = "openai/clip-vit-base-patch32"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}


@dataclass
class ImageMatch:
    """A single image search result."""

    path: Path
    score: float


@dataclass
class CLIPImageSearchConfig:
    """Configuration for :class:`CLIPImageSearchPlugin`."""

    model_id: str = DEFAULT_MODEL_ID
    image_roots: List[Path] = field(default_factory=list)
    cache_dir: Optional[Path] = None
    device: str = "cpu"
    batch_size: int = 16
    top_k: int = 5


class CLIPImageSearchPlugin:
    """On-device text-to-image search powered by CLIP.

    The model is loaded lazily on first use. Image embeddings are cached in
    memory and invalidated automatically when the underlying file changes
    (detected via ``mtime_ns``).
    """

    NAME = "clip_image_search"
    VERSION = "0.1.0"

    def __init__(self, config: Optional[CLIPImageSearchConfig] = None) -> None:
        self.config = config or CLIPImageSearchConfig()
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._embedding_cache: Dict[Path, Tuple[int, Any]] = {}
        self._active = False

    # ---- lifecycle ------------------------------------------------------

    def activate(self) -> None:
        """Mark the plugin as active. The model itself is loaded lazily."""
        if self._active:
            return
        LOGGER.info(
            "Activating CLIP image search plugin (model=%s, device=%s)",
            self.config.model_id,
            self.config.device,
        )
        self._active = True

    def deactivate(self) -> None:
        """Release the model and clear the embedding cache."""
        LOGGER.info("Deactivating CLIP image search plugin")
        self._model = None
        self._processor = None
        self._embedding_cache.clear()
        self._active = False

    # ---- public API -----------------------------------------------------

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        image_roots: Optional[Iterable[Path]] = None,
    ) -> List[ImageMatch]:
        """Return the top-k images matching a natural-language ``query``.

        :param query: Free-text description of what to look for.
        :param top_k: Number of results to return. Defaults to ``config.top_k``.
        :param image_roots: Optional override of the directories to search.
        :raises RuntimeError: If the plugin has not been activated.
        :raises ValueError: If ``query`` is empty or no image roots resolve.
        """
        if not self._active:
            raise RuntimeError("Plugin not activated; call activate() first.")
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        self._ensure_model()

        roots = [Path(p) for p in (image_roots or self.config.image_roots)]
        if not roots:
            raise ValueError(
                "No image roots configured. Set CLIPImageSearchConfig.image_roots "
                "or pass image_roots=... to search()."
            )

        candidate_paths = list(self._collect_images(roots))
        if not candidate_paths:
            return []

        valid_paths, image_embeddings = self._embed_images(candidate_paths)
        if not valid_paths:
            return []

        query_embedding = self._embed_text(query)
        scores = self._cosine_similarity(query_embedding, image_embeddings)

        k = top_k if top_k is not None else self.config.top_k
        ranked = sorted(zip(valid_paths, scores), key=lambda item: item[1], reverse=True)
        return [ImageMatch(path=p, score=float(s)) for p, s in ranked[:k]]

    def classify(self, image_path: Path, labels: List[str]) -> List[Tuple[str, float]]:
        """Zero-shot classify a single image against ``labels``.

        :returns: ``(label, probability)`` pairs sorted by descending probability.
        """
        if not self._active:
            raise RuntimeError("Plugin not activated; call activate() first.")
        if not labels:
            raise ValueError("labels must be a non-empty list")

        self._ensure_model()

        from PIL import Image  # local import; lazy ML dep

        with Image.open(image_path) as raw:
            image = raw.convert("RGB")

        inputs = self._processor(
            text=labels,
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(self.config.device)

        with self._torch.no_grad():
            outputs = self._model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=-1).cpu().tolist()[0]

        return sorted(zip(labels, probs), key=lambda item: item[1], reverse=True)

    def clear_cache(self) -> None:
        """Drop all cached image embeddings."""
        self._embedding_cache.clear()

    # ---- internals ------------------------------------------------------

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError(
                "CLIP plugin requires 'transformers', 'torch', and 'Pillow'. "
                "Install with: pip install transformers torch Pillow"
            ) from exc

        cache_dir = str(self.config.cache_dir) if self.config.cache_dir else None
        LOGGER.info("Loading CLIP model %s on %s", self.config.model_id, self.config.device)
        self._processor = CLIPProcessor.from_pretrained(self.config.model_id, cache_dir=cache_dir)
        model = CLIPModel.from_pretrained(self.config.model_id, cache_dir=cache_dir)
        self._model = model.to(self.config.device).eval()
        self._torch = torch

    def _collect_images(self, roots: Iterable[Path]) -> Iterable[Path]:
        for root in roots:
            if not root.exists():
                LOGGER.warning("Image root does not exist: %s", root)
                continue
            for path in root.rglob("*"):
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    yield path

    def _embed_text(self, text: str) -> Any:
        inputs = self._processor(text=[text], return_tensors="pt", padding=True).to(self.config.device)
        with self._torch.no_grad():
            feats = self._model.get_text_features(**inputs)
        return self._normalize(feats)

    def _embed_images(self, paths: List[Path]) -> Tuple[List[Path], Any]:
        from PIL import Image  # local import; lazy ML dep

        results: Dict[int, Any] = {}
        to_compute: List[Tuple[int, Path]] = []

        for idx, path in enumerate(paths):
            try:
                mtime = path.stat().st_mtime_ns
            except OSError as exc:
                LOGGER.warning("Skipping unreadable image %s: %s", path, exc)
                continue
            cached = self._embedding_cache.get(path)
            if cached is not None and cached[0] == mtime:
                results[idx] = cached[1]
            else:
                to_compute.append((idx, path))

        batch_size = max(1, self.config.batch_size)
        for start in range(0, len(to_compute), batch_size):
            batch = to_compute[start : start + batch_size]
            loaded: List[Tuple[int, Path, Any]] = []
            for orig_idx, path in batch:
                try:
                    with Image.open(path) as raw:
                        loaded.append((orig_idx, path, raw.convert("RGB")))
                except Exception as exc:  # noqa: BLE001 - any decode failure is non-fatal
                    LOGGER.warning("Skipping unreadable image %s: %s", path, exc)

            if not loaded:
                continue

            images = [item[2] for item in loaded]
            inputs = self._processor(images=images, return_tensors="pt").to(self.config.device)
            with self._torch.no_grad():
                feats = self._model.get_image_features(**inputs)
            feats = self._normalize(feats)

            for (orig_idx, path, _), vec in zip(loaded, feats):
                try:
                    mtime = path.stat().st_mtime_ns
                except OSError:
                    mtime = 0
                self._embedding_cache[path] = (mtime, vec)
                results[orig_idx] = vec

        ordered_indices = sorted(results.keys())
        valid_paths = [paths[i] for i in ordered_indices]
        vectors = [results[i] for i in ordered_indices]
        stacked = self._torch.stack(vectors) if vectors else self._torch.empty(0)
        return valid_paths, stacked

    def _normalize(self, tensor: Any) -> Any:
        return tensor / tensor.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)

    def _cosine_similarity(self, query_vec: Any, image_matrix: Any) -> List[float]:
        if image_matrix.shape[0] == 0:
            return []
        sims = (image_matrix @ query_vec.T).squeeze(-1)
        return sims.cpu().tolist()


# ---- module-level entry points --------------------------------------------

_plugin_instance: Optional[CLIPImageSearchPlugin] = None


def register(config: Optional[CLIPImageSearchConfig] = None) -> CLIPImageSearchPlugin:
    """Homie plugin-loader entry point.

    Instantiates (or returns the existing) singleton, activates it, and
    returns the instance so the caller can invoke :meth:`search` directly.
    """
    global _plugin_instance
    if _plugin_instance is None:
        _plugin_instance = CLIPImageSearchPlugin(config=config)
    _plugin_instance.activate()
    return _plugin_instance


def activate(config: Optional[CLIPImageSearchConfig] = None) -> CLIPImageSearchPlugin:
    """Alias of :func:`register` for symmetry with :func:`deactivate`."""
    return register(config)


def deactivate() -> None:
    """Deactivate and discard the singleton plugin instance, if any."""
    global _plugin_instance
    if _plugin_instance is not None:
        _plugin_instance.deactivate()
        _plugin_instance = None


__all__ = [
    "CLIPImageSearchPlugin",
    "CLIPImageSearchConfig",
    "ImageMatch",
    "DEFAULT_MODEL_ID",
    "SUPPORTED_EXTENSIONS",
    "register",
    "activate",
    "deactivate",
]
