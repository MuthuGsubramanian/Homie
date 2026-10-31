"""MobileNetV3 image tagging plugin for HOMIE.

Gives HOMIE fast, fully local image classification/tagging using the
``timm/mobilenetv3_small_100.lamb_in1k`` checkpoint (MobileNet-V3-Small,
~2.5M params, ImageNet-1k, LAMB-trained). The model is small enough to run
comfortably on laptop CPUs and mobile-class hardware with sub-100ms latency
per image, which makes it a good default backend for:

- tagging photos so they can be indexed into HOMIE's ChromaDB RAG store,
- quick "what is in this picture?" answers from the assistant,
- batch-labelling a folder of images without any cloud calls.

Privacy / network policy
------------------------
Inference is 100% on-device. The only network access this plugin can ever
perform is the one-time model weight download from the Hugging Face Hub,
and that is **opt-in**: it happens only when the plugin is activated with
``allow_download=True`` (or config key ``plugins.image_tagger.allow_download``
is truthy). Otherwise the plugin runs in offline mode and requires the
weights to already be present in the local HF cache.

Dependencies (install manually, they are optional for HOMIE core)::

    pip install timm torch pillow

Usage::

    from homie.plugins.mobilenet_image_tagger import MobileNetImageTaggerPlugin

    plugin = MobileNetImageTaggerPlugin(allow_download=True)
    plugin.activate()
    tags = plugin.classify("~/Pictures/cat.jpg", top_k=5)
    for tag in tags:
        print(f"{tag.label}: {tag.confidence:.1%}")
    plugin.deactivate()
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PLUGIN_NAME = "mobilenet_image_tagger"
PLUGIN_VERSION = "1.0.0"
MODEL_ID = "mobilenetv3_small_100.lamb_in1k"

# Common raster formats PIL can open without extra plugins.
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}


class ImageTaggerError(Exception):
    """Raised when the image tagger cannot load its model or process an image."""


@dataclass
class TagResult:
    """A single predicted tag for an image."""

    label: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "confidence": round(self.confidence, 4)}


@dataclass
class ImageTags:
    """All predicted tags for one image, ready for storage or RAG indexing."""

    path: str
    tags: List[TagResult] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "tags": [t.to_dict() for t in self.tags],
            "error": self.error,
        }

    def as_caption(self) -> str:
        """Human/RAG-friendly one-line summary, e.g. for ChromaDB documents."""
        if not self.tags:
            return "unrecognized image"
        return ", ".join(t.label for t in self.tags)


class MobileNetImageTaggerPlugin:
    """On-device image classification for HOMIE via timm MobileNetV3-Small.

    The model and its heavyweight dependencies (torch, timm, PIL) are
    imported lazily inside :meth:`activate` so that HOMIE starts up fast
    even when this plugin is installed but unused.
    """

    def __init__(
        self,
        allow_download: bool = False,
        num_threads: Optional[int] = None,
        top_k: int = 5,
        min_confidence: float = 0.01,
    ) -> None:
        """Create the plugin (does not load the model yet).

        Args:
            allow_download: Opt-in flag permitting a one-time weight download
                from the Hugging Face Hub. When False the plugin only uses
                weights already present in the local cache.
            num_threads: Torch intra-op CPU threads. ``None`` keeps the
                torch default; small values (2-4) are friendlier on mobile.
            top_k: Default number of tags returned per image.
            min_confidence: Tags below this probability are dropped.
        """
        self.allow_download = allow_download
        self.num_threads = num_threads
        self.top_k = top_k
        self.min_confidence = min_confidence

        self._model: Any = None
        self._transform: Any = None
        self._labels: Optional[List[str]] = None
        self._torch: Any = None
        self._active = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def activate(self) -> None:
        """Load the model, preprocessing transform, and ImageNet labels."""
        if self._active:
            logger.debug("%s already active", PLUGIN_NAME)
            return

        if not self.allow_download:
            # Force fully offline behaviour: HF hub will only read the cache.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

        try:
            import timm
            import torch
            from timm.data import ImageNetInfo, infer_imagenet_subset
            from timm.data import create_transform, resolve_model_data_config
        except ImportError as exc:
            raise ImageTaggerError(
                "Missing dependencies for image tagging. "
                "Install with: pip install timm torch pillow"
            ) from exc

        self._torch = torch
        if self.num_threads is not None:
            torch.set_num_threads(int(self.num_threads))

        logger.info("Loading %s (download allowed: %s)", MODEL_ID, self.allow_download)
        try:
            model = timm.create_model(MODEL_ID, pretrained=True)
        except Exception as exc:  # noqa: BLE001 - hub errors vary widely
            hint = (
                "Weights not cached locally and downloads are disabled. "
                "Re-activate with allow_download=True to fetch them once."
                if not self.allow_download
                else "Model download or load failed."
            )
            raise ImageTaggerError(f"{hint} ({exc})") from exc

        model = model.eval()
        data_config = resolve_model_data_config(model)
        self._transform = create_transform(**data_config, is_training=False)
        self._model = model

        subset = infer_imagenet_subset(model) or "imagenet-1k"
        info = ImageNetInfo(subset)
        self._labels = [
            info.index_to_description(i, detailed=False)
            for i in range(model.num_classes)
        ]

        self._active = True
        logger.info("%s active: %d classes, input %s", PLUGIN_NAME, model.num_classes, data_config.get("input_size"))

    def deactivate(self) -> None:
        """Release the model and free memory."""
        self._model = None
        self._transform = None
        self._labels = None
        self._torch = None
        self._active = False
        logger.info("%s deactivated", PLUGIN_NAME)

    @property
    def is_active(self) -> bool:
        return self._active

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def classify(self, image_path: str | Path, top_k: Optional[int] = None) -> List[TagResult]:
        """Classify a single image and return its top tags.

        Args:
            image_path: Path to a local image file.
            top_k: Number of tags to return (defaults to the plugin setting).

        Returns:
            Tags sorted by confidence, filtered by ``min_confidence``.

        Raises:
            ImageTaggerError: If the plugin is inactive or the file is unreadable.
        """
        self._ensure_active()
        from PIL import Image

        path = Path(image_path).expanduser()
        if not path.is_file():
            raise ImageTaggerError(f"Image not found: {path}")

        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:  # noqa: BLE001 - PIL raises many types
            raise ImageTaggerError(f"Cannot read image {path}: {exc}") from exc

        k = top_k or self.top_k
        torch = self._torch
        with torch.inference_mode():
            batch = self._transform(image).unsqueeze(0)
            logits = self._model(batch)
            probs = torch.softmax(logits, dim=1)[0]
            confidences, indices = torch.topk(probs, k=min(k, probs.numel()))

        results: List[TagResult] = []
        for conf, idx in zip(confidences.tolist(), indices.tolist()):
            if conf < self.min_confidence:
                continue
            results.append(TagResult(label=self._labels[idx], confidence=float(conf)))
        return results

    def tag_directory(
        self,
        directory: str | Path,
        recursive: bool = False,
        top_k: Optional[int] = None,
    ) -> List[ImageTags]:
        """Tag every supported image in a directory.

        Per-file failures are recorded in the result rather than raised, so
        one corrupt file cannot abort a batch run.
        """
        self._ensure_active()
        root = Path(directory).expanduser()
        if not root.is_dir():
            raise ImageTaggerError(f"Not a directory: {root}")

        pattern = root.rglob("*") if recursive else root.glob("*")
        results: List[ImageTags] = []
        for path in sorted(pattern):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            try:
                tags = self.classify(path, top_k=top_k)
                results.append(ImageTags(path=str(path), tags=tags))
            except ImageTaggerError as exc:
                logger.warning("Skipping %s: %s", path, exc)
                results.append(ImageTags(path=str(path), tags=[], error=str(exc)))
        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _ensure_active(self) -> None:
        if not self._active or self._model is None:
            raise ImageTaggerError(
                f"{PLUGIN_NAME} is not active. Call activate() first."
            )


def register(config: Optional[Dict[str, Any]] = None) -> MobileNetImageTaggerPlugin:
    """HOMIE plugin entry point.

    Reads optional settings from the ``plugins.image_tagger`` section of the
    HOMIE config dict, builds the plugin, and returns it (not yet activated,
    so HOMIE core controls when the model weights are loaded).
    """
    cfg = (config or {}).get("plugins", {}).get("image_tagger", {})
    return MobileNetImageTaggerPlugin(
        allow_download=bool(cfg.get("allow_download", False)),
        num_threads=cfg.get("num_threads"),
        top_k=int(cfg.get("top_k", 5)),
        min_confidence=float(cfg.get("min_confidence", 0.01)),
    )


__all__ = [
    "MobileNetImageTaggerPlugin",
    "ImageTags",
    "TagResult",
    "ImageTaggerError",
    "register",
    "PLUGIN_NAME",
    "PLUGIN_VERSION",
]


if __name__ == "__main__":
    # Quick local smoke test: python mobilenet_image_tagger.py <image> [image...]
    import sys

    logging.basicConfig(level=logging.INFO)
    plugin = MobileNetImageTaggerPlugin(allow_download=True)
    plugin.activate()
    for arg in sys.argv[1:]:
        print(f"\n{arg}")
        for tag in plugin.classify(arg):
            print(f"  {tag.label:<30} {tag.confidence:.1%}")
    plugin.deactivate()
