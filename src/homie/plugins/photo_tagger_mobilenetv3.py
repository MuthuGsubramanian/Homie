"""MobileNetV3 photo tagger plugin for HOMIE.

Tags local photos with ImageNet-1k labels using the
``timm/mobilenetv3_small_100.lamb_in1k`` checkpoint -- a ~2.5M-parameter
MobileNet-V3-Small trained with the LAMB optimizer.  Small enough for
CPU-only inference on desktop and mobile-class hardware, which fits
HOMIE's local-first design.

Behaviour:
    * All inference runs locally (CPU by default; pass device="cuda" to
      opt in to GPU).
    * Weights are read from the local Hugging Face cache.  Downloading
      is OPT-IN: pass allow_download=True (or set
      plugins.photo_tagger.allow_download: true in homie.config.yaml)
      to permit a one-time ~10 MB fetch.  Without it the plugin runs
      fully offline.
    * Heavy dependencies (torch, timm, Pillow) are imported lazily so
      HOMIE starts fine when the plugin is inactive or extras are
      missing.

Usage:
    from homie.plugins.photo_tagger_mobilenetv3 import activate, get_plugin

    activate(cfg)  # cfg optional
    tags = get_plugin().tag_image("~/Pictures/cat.jpg")
    # [TagResult(label='tabby', score=0.61), ...]
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from homie.config import HomieConfig

LOGGER = logging.getLogger(__name__)

PLUGIN_NAME = "photo_tagger_mobilenetv3"
MODEL_ID = "mobilenetv3_small_100.lamb_in1k"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_TOP_K = 5


class PhotoTaggerError(Exception):
    """Raised when tagging fails (missing deps, weights, or bad image)."""


@dataclass
class TagResult:
    """One predicted tag for an image."""

    label: str
    score: float

    def as_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "score": round(self.score, 4)}


class MobileNetPhotoTagger:
    """ImageNet-1k photo tagger backed by timm MobileNetV3-Small."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        top_k: int = DEFAULT_TOP_K,
        device: str = "cpu",
        allow_download: bool = False,
    ) -> None:
        self.model_id = model_id
        self.top_k = top_k
        self.device = device
        self.allow_download = allow_download
        self._lock = threading.Lock()
        self._model: Any = None
        self._transform: Any = None
        self._labels: List[str] = []

    # -- lifecycle -----------------------------------------------------

    def load(self) -> None:
        """Load weights and the preprocessing pipeline (idempotent)."""
        with self._lock:
            if self._model is not None:
                return
            try:
                import timm
                import torch  # noqa: F401
                from PIL import Image  # noqa: F401
                from timm.data import create_transform, resolve_data_config
            except ImportError as exc:
                raise PhotoTaggerError(
                    "Missing dependencies. Install with: pip install timm torch pillow"
                ) from exc

            if not self.allow_download:
                # Keep timm/huggingface_hub restricted to the local cache.
                os.environ.setdefault("HF_HUB_OFFLINE", "1")

            try:
                model = timm.create_model(self.model_id, pretrained=True)
            except Exception as exc:  # noqa: BLE001
                raise PhotoTaggerError(
                    f"Could not load weights for '{self.model_id}'. If they are "
                    "not cached yet, re-activate with allow_download=True."
                ) from exc

            model.eval()
            model.to(self.device)
            data_config = resolve_data_config({}, model=model)
            self._transform = create_transform(**data_config)
            self._labels = self._load_labels(model)
            self._model = model
            LOGGER.info("%s: loaded %s on %s", PLUGIN_NAME, self.model_id, self.device)

    def unload(self) -> None:
        """Release the model so memory returns to the rest of HOMIE."""
        with self._lock:
            self._model = None
            self._transform = None
            self._labels = []

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # -- inference -----------------------------------------------------

    def tag_image(
        self, image_path: str | Path, top_k: Optional[int] = None
    ) -> List[TagResult]:
        """Return the top-k ImageNet tags for one local image."""
        self.load()
        import torch
        from PIL import Image

        path = Path(image_path).expanduser()
        if not path.is_file():
            raise PhotoTaggerError(f"Image not found: {path}")

        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            raise PhotoTaggerError(f"Unreadable image {path}: {exc}") from exc

        k = min(top_k or self.top_k, len(self._labels))
        tensor = self._transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probs = self._model(tensor).softmax(dim=-1)[0]
        scores, indices = probs.topk(k)
        return [
            TagResult(label=self._labels[int(idx)], score=float(score))
            for score, idx in zip(scores, indices)
        ]

    def tag_directory(
        self, directory: str | Path, top_k: Optional[int] = None
    ) -> Dict[str, List[TagResult]]:
        """Tag every supported image directly inside ``directory``.

        Unreadable files are skipped with a warning rather than aborting
        the whole batch.
        """
        root = Path(directory).expanduser()
        if not root.is_dir():
            raise PhotoTaggerError(f"Not a directory: {root}")
        results: Dict[str, List[TagResult]] = {}
        for path in sorted(root.iterdir()):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            try:
                results[str(path)] = self.tag_image(path, top_k=top_k)
            except PhotoTaggerError as exc:
                LOGGER.warning("%s: skipping %s (%s)", PLUGIN_NAME, path, exc)
        return results

    @staticmethod
    def _load_labels(model: Any) -> List[str]:
        num_classes = int(getattr(model, "num_classes", 1000))
        try:
            from timm.data import ImageNetInfo, infer_imagenet_subset

            info = ImageNetInfo(infer_imagenet_subset(model) or "imagenet-1k")
            return [
                info.label_name_to_description(
                    info.index_to_label_name(i)
                ).split(",")[0]
                for i in range(num_classes)
            ]
        except Exception:  # noqa: BLE001
            LOGGER.debug("%s: label metadata unavailable, using indices", PLUGIN_NAME)
            return [f"class_{i}" for i in range(num_classes)]


_PLUGIN: Optional[MobileNetPhotoTagger] = None


def activate(
    cfg: Optional["HomieConfig"] = None, eager: bool = False
) -> MobileNetPhotoTagger:
    """Create the tagger singleton, reading plugins.photo_tagger.* config."""
    global _PLUGIN
    if _PLUGIN is not None:
        return _PLUGIN

    top_k = DEFAULT_TOP_K
    device = "cpu"
    allow_download = False
    if cfg is not None:
        from homie.config import cfg_get

        top_k = int(cfg_get(cfg, "plugins", "photo_tagger", "top_k", default=DEFAULT_TOP_K))
        device = cfg_get(cfg, "plugins", "photo_tagger", "device", default="cpu")
        allow_download = bool(
            cfg_get(cfg, "plugins", "photo_tagger", "allow_download", default=False)
        )

    _PLUGIN = MobileNetPhotoTagger(top_k=top_k, device=device, allow_download=allow_download)
    if eager:
        _PLUGIN.load()
    LOGGER.info(
        "%s activated (device=%s, allow_download=%s)", PLUGIN_NAME, device, allow_download
    )
    return _PLUGIN


def deactivate() -> None:
    """Tear down the singleton and free model memory."""
    global _PLUGIN
    if _PLUGIN is not None:
        _PLUGIN.unload()
        _PLUGIN = None
        LOGGER.info("%s deactivated", PLUGIN_NAME)


def get_plugin() -> MobileNetPhotoTagger:
    """Return the active tagger, or raise if activate() was never called."""
    if _PLUGIN is None:
        raise PhotoTaggerError("Plugin not active. Call activate() first.")
    return _PLUGIN


__all__ = [
    "MODEL_ID",
    "PLUGIN_NAME",
    "MobileNetPhotoTagger",
    "PhotoTaggerError",
    "TagResult",
    "activate",
    "deactivate",
    "get_plugin",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tag a local photo with MobileNetV3.")
    parser.add_argument("image", help="Path to an image file")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Permit a one-time download of model weights (~10 MB)",
    )
    cli_args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cli_tagger = MobileNetPhotoTagger(
        top_k=cli_args.top_k, allow_download=cli_args.allow_download
    )
    for tag in cli_tagger.tag_image(cli_args.image):
        print(f"{tag.label:30s} {tag.score:.3f}")
