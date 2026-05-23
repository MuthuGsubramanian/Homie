"""HOMIE plugin: Local image search powered by OpenAI CLIP (ViT-B/32).

Builds CLIP embeddings for images in a user-specified folder and enables
natural-language search (e.g. "a photo of my dog at the beach") entirely on
the local machine. No images are ever sent to the cloud; the only outbound
traffic is a one-time model download from Hugging Face, which is gated behind
an explicit ``allow_download`` opt-in flag.

Integrates with HOMIE the same way other local-AI components do: configured
via ``homie.config.HomieConfig`` under the ``plugins.clip_image_search`` key,
activated through ``register(cfg)`` or directly via
``CLIPImageSearchPlugin().activate()``.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

PLUGIN_NAME = "clip_image_search"
DEFAULT_MODEL_ID = "openai/clip-vit-base-patch32"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


@dataclass
class ImageIndexEntry:
    """One indexed image: absolute path plus L2-normalized CLIP embedding."""

    path: str
    embedding: Any  # numpy.ndarray, kept untyped to avoid import at module load


@dataclass
class CLIPImageSearchPlugin:
    """Local CLIP-based image search.

    Lifecycle:
        * ``activate()``   - load model + processor and any persisted index.
        * ``index_library(folder)`` - walk *folder*, embed new images, persist.
        * ``search(text, top_k)``   - embed a text query and return top matches.
        * ``deactivate()`` - persist the index and release model handles.

    The plugin is fully offline once the model is cached. Set
    ``allow_download=True`` once to fetch the weights from Hugging Face, then
    flip it back to ``False`` for subsequent runs.
    """

    model_id: str = DEFAULT_MODEL_ID
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".homie" / "clip")
    allow_download: bool = False
    device: str = "cpu"

    _model: Any = field(default=None, init=False, repr=False)
    _processor: Any = field(default=None, init=False, repr=False)
    _index: List[ImageIndexEntry] = field(default_factory=list, init=False, repr=False)

    @property
    def index_path(self) -> Path:
        """Filesystem location of the persisted embedding index."""
        return self.cache_dir / "index.pkl"

    def activate(self) -> None:
        """Load the CLIP model/processor and any cached embedding index."""
        try:
            import torch  # noqa: F401
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError(
                "transformers, torch, and pillow are required for the CLIP plugin. "
                "Install with: pip install transformers torch pillow"
            ) from exc

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        local_only = not self.allow_download
        logger.info(
            "Loading CLIP model %s (local_files_only=%s, device=%s)",
            self.model_id,
            local_only,
            self.device,
        )
        try:
            self._processor = CLIPProcessor.from_pretrained(
                self.model_id, local_files_only=local_only
            )
            self._model = CLIPModel.from_pretrained(
                self.model_id, local_files_only=local_only
            )
        except OSError as exc:
            raise RuntimeError(
                f"CLIP model '{self.model_id}' is not cached locally and "
                "allow_download is False. Set allow_download=True once to fetch "
                "the weights, or pre-download with:\n"
                f"    huggingface-cli download {self.model_id}"
            ) from exc

        self._model.to(self.device).eval()
        self._load_index()

    def deactivate(self) -> None:
        """Persist the current index and release model resources."""
        self._save_index()
        self._model = None
        self._processor = None

    def index_library(self, folder: Path | str, recursive: bool = True) -> int:
        """Embed every supported image under *folder* and persist the index.

        Returns the number of newly added images. Existing entries are skipped
        so re-indexing the same folder is cheap.
        """
        self._require_active()
        folder = Path(folder).expanduser().resolve()
        if not folder.exists() or not folder.is_dir():
            raise ValueError(f"Image folder not found: {folder}")

        existing = {entry.path for entry in self._index}
        added = 0
        for path in self._collect_image_paths(folder, recursive):
            key = str(path)
            if key in existing:
                continue
            try:
                embedding = self._embed_image(path)
            except Exception as exc:  # noqa: BLE001 - per-file resilience
                logger.warning("Skipping %s: %s", path, exc)
                continue
            self._index.append(ImageIndexEntry(path=key, embedding=embedding))
            added += 1

        self._save_index()
        logger.info("Indexed %d new images (total=%d)", added, len(self._index))
        return added

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Return up to *top_k* ``(path, score)`` matches for the text *query*.

        Scores are cosine similarities in [-1, 1]; higher is better.
        """
        self._require_active()
        if not self._index:
            return []

        import numpy as np

        query_vec = self._embed_text(query)
        matrix = np.stack([entry.embedding for entry in self._index])
        scores = matrix @ query_vec  # both sides are L2-normalized

        k = max(1, min(top_k, len(self._index)))
        order = np.argsort(-scores)[:k]
        return [(self._index[int(i)].path, float(scores[int(i)])) for i in order]

    def clear_index(self) -> None:
        """Drop the in-memory index and delete the persisted file."""
        self._index = []
        if self.index_path.exists():
            self.index_path.unlink()

    @property
    def size(self) -> int:
        """Number of images currently in the index."""
        return len(self._index)

    def _collect_image_paths(self, folder: Path, recursive: bool) -> Iterable[Path]:
        walker = folder.rglob("*") if recursive else folder.iterdir()
        for path in walker:
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield path

    def _embed_image(self, path: Path) -> Any:
        import numpy as np
        import torch
        from PIL import Image

        with Image.open(path) as img:
            rgb = img.convert("RGB")
            inputs = self._processor(images=rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self._model.get_image_features(**inputs)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.squeeze(0).cpu().numpy().astype(np.float32)

    def _embed_text(self, text: str) -> Any:
        import numpy as np
        import torch

        inputs = self._processor(
            text=[text], return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            features = self._model.get_text_features(**inputs)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.squeeze(0).cpu().numpy().astype(np.float32)

    def _load_index(self) -> None:
        if not self.index_path.exists():
            self._index = []
            return
        try:
            with self.index_path.open("rb") as fh:
                loaded = pickle.load(fh)
            if isinstance(loaded, list):
                self._index = loaded
                logger.info("Loaded %d cached image embeddings", len(self._index))
            else:
                raise ValueError("unexpected index payload type")
        except Exception as exc:  # noqa: BLE001 - corrupt cache is non-fatal
            logger.warning("Failed to load index (%s); starting fresh", exc)
            self._index = []

    def _save_index(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.index_path.with_suffix(".pkl.tmp")
        with tmp.open("wb") as fh:
            pickle.dump(self._index, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(self.index_path)

    def _require_active(self) -> None:
        if self._model is None or self._processor is None:
            raise RuntimeError(
                "CLIPImageSearchPlugin is not active. Call activate() first."
            )


def _plugin_config(cfg: Any) -> Dict[str, Any]:
    """Extract this plugin's section from a HomieConfig (or plain dict)."""
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg.get("plugins", {}).get(PLUGIN_NAME, {}) or cfg)
    # Duck-type HomieConfig without importing it (keeps the plugin standalone).
    getter = getattr(cfg, "get", None)
    if callable(getter):
        section = getter("plugins", PLUGIN_NAME, default={})
        return dict(section or {})
    return {}


def register(cfg: Optional[Any] = None) -> CLIPImageSearchPlugin:
    """HOMIE plugin entry point: build, activate, and return the plugin.

    Config keys read from ``plugins.clip_image_search``:
        model_id (str)         - Hugging Face repo id (default: openai/clip-vit-base-patch32)
        cache_dir (str)        - where to persist the embedding index
        allow_download (bool)  - permit one-time HF model download
        device (str)           - 'cpu' or 'cuda'
    """
    section = _plugin_config(cfg)
    plugin = CLIPImageSearchPlugin(
        model_id=str(section.get("model_id", DEFAULT_MODEL_ID)),
        cache_dir=Path(section.get("cache_dir", Path.home() / ".homie" / "clip")),
        allow_download=bool(section.get("allow_download", False)),
        device=str(section.get("device", "cpu")),
    )
    plugin.activate()
    return plugin


__all__ = [
    "CLIPImageSearchPlugin",
    "ImageIndexEntry",
    "PLUGIN_NAME",
    "DEFAULT_MODEL_ID",
    "register",
]
