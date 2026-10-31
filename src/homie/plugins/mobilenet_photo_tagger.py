"""MobileNetV3 photo tagger plugin for HOMIE.

Runs the ``timm/mobilenetv3_small_100.lamb_in1k`` checkpoint locally to
attach ImageNet-1k labels to the user's photos. MobileNetV3-Small is a
~2.5M-parameter network designed for mobile CPUs, which makes it a good
fit for HOMIE's local-first photo tagging: single-image inference takes
tens of milliseconds on a laptop CPU, with no GPU and no cloud calls.

Usage (from HOMIE core or another plugin)::

    from homie.config import load_config
    from homie.plugins.mobilenet_photo_tagger import register

    cfg = load_config()
    tagger = register(cfg)
    tagger.activate()
    result = tagger.tag_image("~/Pictures/cat.jpg")
    print(result.labels)   # [("tabby", 0.61), ("tiger cat", 0.22), ...]
    tagger.deactivate()

Configuration (all optional, under ``plugins.photo_tagger`` in
``homie.config.yaml``)::

    plugins:
      photo_tagger:
        model_name: mobilenetv3_small_100.lamb_in1k
        top_k: 5
        device: cpu            # or cuda
        allow_download: false  # opt-in: fetch weights from HF Hub once
        min_confidence: 0.05

The plugin is offline by default: unless ``allow_download`` is true it
only loads weights already present in the local Hugging Face cache and
raises a clear error instead of touching the network.

Dependencies: ``timm``, ``torch``, ``Pillow`` -- imported lazily inside
``activate()`` so HOMIE starts fine without them installed.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from homie.config import HomieConfig, cfg_get

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "mobilenetv3_small_100.lamb_in1k"
DEFAULT_TOP_K = 5
DEFAULT_MIN_CONFIDENCE = 0.05
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class PhotoTaggerError(Exception):
    """Raised when the photo tagger cannot load its model or tag an image."""


@dataclass
class TagResult:
    """Labels predicted for a single image."""

    path: str
    labels: List[Tuple[str, float]] = field(default_factory=list)

    @property
    def top_label(self) -> Optional[str]:
        return self.labels[0][0] if self.labels else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "top_label": self.top_label,
            "labels": [
                {"label": name, "score": round(score, 4)}
                for name, score in self.labels
            ],
        }


class MobileNetPhotoTagger:
    """CPU-friendly local photo tagger backed by timm MobileNetV3-Small."""

    name = "photo_tagger"

    def __init__(self, cfg: HomieConfig) -> None:
        self.cfg = cfg
        self.model_name: str = cfg_get(
            cfg, "plugins", "photo_tagger", "model_name", default=DEFAULT_MODEL_NAME
        )
        self.top_k: int = int(
            cfg_get(cfg, "plugins", "photo_tagger", "top_k", default=DEFAULT_TOP_K)
        )
        self.device: str = cfg_get(
            cfg, "plugins", "photo_tagger", "device", default="cpu"
        )
        self.allow_download: bool = bool(
            cfg_get(cfg, "plugins", "photo_tagger", "allow_download", default=False)
        )
        self.min_confidence: float = float(
            cfg_get(
                cfg,
                "plugins",
                "photo_tagger",
                "min_confidence",
                default=DEFAULT_MIN_CONFIDENCE,
            )
        )
        self._model: Any = None
        self._transform: Any = None
        self._torch: Any = None
        self._labels: List[str] = []

    @property
    def is_active(self) -> bool:
        return self._model is not None

    def activate(self) -> None:
        """Load model weights, preprocessing transform, and label map."""
        if self.is_active:
            logger.debug("%s already active", self.name)
            return

        try:
            import timm
            import torch
            from timm.data import ImageNetInfo, infer_imagenet_subset
        except ImportError as exc:
            raise PhotoTaggerError(
                "photo_tagger requires 'timm', 'torch' and 'Pillow'. "
                "Install with: pip install timm torch Pillow"
            ) from exc

        # Offline by default: force the HF Hub client to cache-only mode so
        # a cache miss fails fast instead of silently downloading weights.
        previous_offline = os.environ.get("HF_HUB_OFFLINE")
        if not self.allow_download:
            os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            model = timm.create_model(self.model_name, pretrained=True)
        except Exception as exc:  # noqa: BLE001
            raise PhotoTaggerError(
                f"Could not load '{self.model_name}'. If the weights are not "
                "cached locally yet, set plugins.photo_tagger.allow_download: "
                f"true for a one-time download. Original error: {exc}"
            ) from exc
        finally:
            if not self.allow_download:
                if previous_offline is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    os.environ["HF_HUB_OFFLINE"] = previous_offline

        model.eval()
        model.to(self.device)

        data_cfg = timm.data.resolve_model_data_config(model)
        self._transform = timm.data.create_transform(**data_cfg, is_training=False)

        try:
            subset = infer_imagenet_subset(model) or "imagenet-1k"
            info = ImageNetInfo(subset)
            self._labels = [
                info.index_to_description(i) for i in range(model.num_classes)
            ]
        except Exception:  # noqa: BLE001
            logger.warning("Could not resolve ImageNet label names, using indices")
            self._labels = [f"class_{i}" for i in range(model.num_classes)]

        self._model = model
        self._torch = torch
        logger.info(
            "photo_tagger active | model=%s device=%s top_k=%d",
            self.model_name,
            self.device,
            self.top_k,
        )

    def deactivate(self) -> None:
        """Release the model and associated resources."""
        self._model = None
        self._transform = None
        self._torch = None
        self._labels = []
        logger.info("photo_tagger deactivated")

    def tag_image(self, image_path: str | Path) -> TagResult:
        """Classify a single image and return its top-k ImageNet labels."""
        self._ensure_active()
        path = Path(image_path).expanduser()
        if not path.is_file():
            raise PhotoTaggerError(f"Image not found: {path}")

        try:
            from PIL import Image
        except ImportError as exc:
            raise PhotoTaggerError(
                "Pillow is required to read images: pip install Pillow"
            ) from exc

        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            raise PhotoTaggerError(f"Cannot read image {path}: {exc}") from exc

        torch = self._torch
        with torch.inference_mode():
            batch = self._transform(image).unsqueeze(0).to(self.device)
            probs = self._model(batch).softmax(dim=-1)[0]
            k = min(self.top_k, probs.shape[-1])
            scores, indices = torch.topk(probs, k)

        labels: List[Tuple[str, float]] = []
        for score, index in zip(scores.tolist(), indices.tolist()):
            if score < self.min_confidence:
                continue
            labels.append((self._labels[index], float(score)))
        return TagResult(path=str(path), labels=labels)

    def tag_directory(
        self, directory: str | Path, recursive: bool = False
    ) -> List[TagResult]:
        """Tag every supported image under *directory*.

        Unreadable or unsupported files are skipped with a warning so one
        corrupt photo does not abort a whole library scan.
        """
        self._ensure_active()
        root = Path(directory).expanduser()
        if not root.is_dir():
            raise PhotoTaggerError(f"Not a directory: {root}")

        candidates = root.rglob("*") if recursive else root.glob("*")
        results: List[TagResult] = []
        for candidate in sorted(candidates):
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            try:
                results.append(self.tag_image(candidate))
            except PhotoTaggerError as exc:
                logger.warning("Skipping %s: %s", candidate, exc)
        return results

    def _ensure_active(self) -> None:
        if not self.is_active:
            raise PhotoTaggerError(
                "photo_tagger is not active. Call activate() first."
            )


def register(cfg: HomieConfig) -> MobileNetPhotoTagger:
    """Entry point used by HOMIE's plugin loader."""
    return MobileNetPhotoTagger(cfg)


__all__ = [
    "MobileNetPhotoTagger",
    "TagResult",
    "PhotoTaggerError",
    "register",
]
