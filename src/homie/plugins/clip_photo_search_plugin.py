"""CLIP zero-shot photo tools for HOMIE.

This plugin wraps ``openai/clip-vit-base-patch32`` to give HOMIE two fully
local capabilities:

* ``classify`` -- zero-shot image classification: score an image against any
  set of free-form text labels (no fine-tuning required).
* ``search`` / ``index_directory`` -- natural-language photo search: embed the
  images in a folder once, then rank them against text queries such as
  "birthday cake" or "dog on a beach".

Everything runs on-device via ``transformers`` + ``torch``.  The model is
loaded lazily on first use and only from the local Hugging Face cache unless
the user opts in to a one-time download:

    plugins:
      clip_photo_search:
        allow_download: true      # opt-in network access (default false)
        model_id: openai/clip-vit-base-patch32
        device: cpu               # or cuda
        photo_dirs:
          - ~/Pictures

Dependencies (optional for HOMIE core, install manually):
    pip install torch transformers pillow
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from homie.config import HomieConfig, cfg_get

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "openai/clip-vit-base-patch32"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class ClipPrediction:
    """A single zero-shot classification result."""

    label: str
    score: float


@dataclass
class PhotoMatch:
    """A single photo-search hit."""

    path: Path
    score: float


class ClipPhotoSearchPlugin:
    """Zero-shot image classification and text-to-photo search using CLIP.

    The model and processor are loaded lazily so importing/activating the
    plugin is cheap; heavy dependencies are only touched on first use.
    """

    name = "clip_photo_search"

    def __init__(self, cfg: Optional[HomieConfig] = None) -> None:
        self.cfg = cfg
        self._model: Any = None
        self._processor: Any = None
        self._device: str = "cpu"
        self._index: Dict[Path, Any] = {}
        self._active = False

    # ------------------------------------------------------------------
    # Configuration / lifecycle
    # ------------------------------------------------------------------
    def _cfg(self, *keys: str, default: Any = None) -> Any:
        if self.cfg is None:
            return default
        return cfg_get(self.cfg, "plugins", self.name, *keys, default=default)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError(
                "CLIP plugin requires optional dependencies: "
                "pip install torch transformers pillow"
            ) from exc

        model_id = self._cfg("model_id", default=DEFAULT_MODEL_ID)
        allow_download = bool(self._cfg("allow_download", default=False))
        device = self._cfg(
            "device", default="cuda" if torch.cuda.is_available() else "cpu"
        )
        try:
            model = CLIPModel.from_pretrained(
                model_id, local_files_only=not allow_download
            )
            processor = CLIPProcessor.from_pretrained(
                model_id, local_files_only=not allow_download
            )
        except OSError as exc:
            raise RuntimeError(
                f"CLIP weights for '{model_id}' are not in the local cache and "
                "network access is off. Set plugins.clip_photo_search."
                "allow_download: true for a one-time download, then disable it."
            ) from exc

        model.eval()
        model.to(device)
        self._model = model
        self._processor = processor
        self._device = device
        LOGGER.info("CLIP plugin loaded %s on %s", model_id, device)

    def activate(self) -> None:
        """Mark the plugin active. Model load stays lazy."""
        self._active = True
        LOGGER.info("clip_photo_search activated (lazy load)")

    def deactivate(self) -> None:
        """Release the model, the embeddings index, and any GPU memory."""
        self._active = False
        self._model = None
        self._processor = None
        self._index.clear()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        LOGGER.info("clip_photo_search deactivated")

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------
    def _embed_images(self, paths: Sequence[Path]) -> Any:
        import torch
        from PIL import Image

        images = [Image.open(p).convert("RGB") for p in paths]
        inputs = self._processor(images=images, return_tensors="pt").to(self._device)
        with torch.no_grad():
            feats = self._model.get_image_features(**inputs)
        return feats / feats.norm(dim=-1, keepdim=True)

    def _embed_text(self, texts: Sequence[str]) -> Any:
        import torch

        inputs = self._processor(
            text=list(texts), return_tensors="pt", padding=True
        ).to(self._device)
        with torch.no_grad():
            feats = self._model.get_text_features(**inputs)
        return feats / feats.norm(dim=-1, keepdim=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def classify(
        self,
        image_path: str | Path,
        labels: Sequence[str],
        top_k: int = 5,
    ) -> List[ClipPrediction]:
        """Zero-shot classify one image against free-form text labels."""
        if not labels:
            raise ValueError("labels must be a non-empty sequence")
        self._ensure_loaded()
        import torch
        from PIL import Image

        path = Path(image_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"No such image: {path}")

        image = Image.open(path).convert("RGB")
        prompts = [f"a photo of {label}" for label in labels]
        inputs = self._processor(
            text=prompts, images=image, return_tensors="pt", padding=True
        ).to(self._device)
        with torch.no_grad():
            logits = self._model(**inputs).logits_per_image
        probs = logits.softmax(dim=-1)[0].tolist()
        ranked = sorted(zip(labels, probs), key=lambda pair: pair[1], reverse=True)
        return [ClipPrediction(label=l, score=float(s)) for l, s in ranked[:top_k]]

    def index_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
        batch_size: int = 8,
    ) -> int:
        """Embed every image under ``directory`` into the in-memory index.

        Returns the number of newly indexed images. Already-indexed paths are
        skipped, so re-running after adding photos is cheap.
        """
        self._ensure_loaded()
        root = Path(directory).expanduser()
        if not root.is_dir():
            raise NotADirectoryError(f"No such directory: {root}")

        pattern = root.rglob("*") if recursive else root.glob("*")
        pending = [
            p
            for p in pattern
            if p.suffix.lower() in IMAGE_EXTENSIONS and p not in self._index
        ]
        added = 0
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            try:
                feats = self._embed_images(batch)
            except OSError:
                LOGGER.warning("Skipping unreadable image batch at %s", batch[0])
                continue
            for path, feat in zip(batch, feats):
                self._index[path] = feat
                added += 1
        LOGGER.info("Indexed %d new images (%d total)", added, len(self._index))
        return added

    def search(
        self,
        query: str,
        directory: Optional[str | Path] = None,
        top_k: int = 5,
    ) -> List[PhotoMatch]:
        """Rank indexed photos against a natural-language query.

        If ``directory`` is given it is indexed first; otherwise the
        configured ``photo_dirs`` are indexed on the first call.
        """
        self._ensure_loaded()
        import torch

        if directory is not None:
            self.index_directory(directory)
        elif not self._index:
            for photo_dir in self._cfg("photo_dirs", default=[]) or []:
                self.index_directory(photo_dir)
        if not self._index:
            return []

        paths = list(self._index.keys())
        image_feats = torch.stack([self._index[p] for p in paths])
        text_feat = self._embed_text([query])[0]
        scores = (image_feats @ text_feat).tolist()
        ranked = sorted(zip(paths, scores), key=lambda pair: pair[1], reverse=True)
        return [PhotoMatch(path=p, score=float(s)) for p, s in ranked[:top_k]]


_PLUGIN: Optional[ClipPhotoSearchPlugin] = None


def activate(cfg: Optional[HomieConfig] = None) -> ClipPhotoSearchPlugin:
    """Create (or reuse) the singleton plugin instance and activate it."""
    global _PLUGIN
    if _PLUGIN is None:
        _PLUGIN = ClipPhotoSearchPlugin(cfg)
    _PLUGIN.activate()
    return _PLUGIN


def deactivate() -> None:
    """Deactivate and release the singleton plugin instance, if any."""
    global _PLUGIN
    if _PLUGIN is not None:
        _PLUGIN.deactivate()
        _PLUGIN = None


def register(cfg: Optional[HomieConfig] = None) -> Dict[str, Any]:
    """Entry point for HOMIE's plugin loader."""
    plugin = activate(cfg)
    return {
        "name": plugin.name,
        "provides": ["classify_image", "search_photos", "index_photos"],
        "classify_image": plugin.classify,
        "search_photos": plugin.search,
        "index_photos": plugin.index_directory,
        "deactivate": deactivate,
    }


__all__ = [
    "ClipPhotoSearchPlugin",
    "ClipPrediction",
    "PhotoMatch",
    "activate",
    "deactivate",
    "register",
]
