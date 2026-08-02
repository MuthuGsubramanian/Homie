"""MobileNetV3-Small image classification plugin for HOMIE.

Wraps ``timm/mobilenetv3_small_100.lamb_in1k`` -- a ~2.5M-parameter ImageNet
classifier that runs comfortably on CPU -- behind HOMIE's plugin conventions.
The same checkpoint already powers HOMIE's image-tagger plugin; this module
exposes it as a standalone, reusable classifier so other plugins (photo
search, automations, suggestion signals) can share one cached copy of the
weights instead of loading their own.

Features
--------
- Lazy model loading: nothing heavy is imported or downloaded at ``activate()``.
- Opt-in downloads: network access is disabled unless ``allow_download=True``
  (or config key ``plugins.mobilenet_classifier.allow_download``); otherwise
  only a locally cached checkpoint is used (``HF_HUB_OFFLINE`` is enforced).
- Single-image classification and directory tagging with top-k labels.
- Pure-local inference via timm/PyTorch on CPU (GPU used when available).

Dependencies (install manually; the plugin never auto-installs):
    pip install timm torch pillow
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MODEL_ID = "mobilenetv3_small_100.lamb_in1k"
HF_HUB_ID = f"hf-hub:timm/{MODEL_ID}"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff"}


class PluginError(RuntimeError):
    """Raised when the classifier cannot be loaded or run."""


@dataclass
class Prediction:
    """A single top-k class prediction."""

    label: str
    index: int
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "index": self.index, "score": round(self.score, 5)}


@dataclass
class ClassificationResult:
    """Predictions for one image."""

    path: str
    predictions: List[Prediction] = field(default_factory=list)

    @property
    def top1(self) -> Optional[Prediction]:
        return self.predictions[0] if self.predictions else None

    def to_dict(self) -> Dict[str, Any]:
        return {"path": self.path, "predictions": [p.to_dict() for p in self.predictions]}


class MobileNetClassifierPlugin:
    """CPU-friendly ImageNet classifier built on MobileNetV3-Small (timm)."""

    name = "mobilenet_classifier"

    def __init__(
        self,
        allow_download: bool = False,
        top_k: int = 5,
        device: Optional[str] = None,
    ) -> None:
        self.allow_download = allow_download
        self.top_k = max(1, top_k)
        self._requested_device = device
        self._model: Any = None
        self._transform: Any = None
        self._labels: Optional[List[str]] = None
        self._device: Any = None
        self._active = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def activate(self) -> None:
        """Mark the plugin active. Model load is deferred to first use."""
        self._active = True
        logger.info("%s activated (allow_download=%s)", self.name, self.allow_download)

    def deactivate(self) -> None:
        """Release the model, transforms, and label cache."""
        self._model = None
        self._transform = None
        self._labels = None
        self._device = None
        self._active = False
        logger.info("%s deactivated", self.name)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        if not self._active:
            raise PluginError("Plugin is not active; call activate() first.")
        if self._model is not None:
            return

        if not self.allow_download:
            # Force hub libraries to use only the local cache (no network).
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

        try:
            import timm
            import torch
        except ImportError as exc:
            raise PluginError(
                "Missing dependencies. Install with: pip install timm torch pillow"
            ) from exc

        try:
            model = timm.create_model(HF_HUB_ID, pretrained=True)
        except Exception as exc:  # noqa: BLE001 - hub raises many exception types
            if not self.allow_download:
                raise PluginError(
                    f"Checkpoint for {MODEL_ID} is not cached locally and downloads "
                    "are disabled. Re-create the plugin with allow_download=True to "
                    "fetch it once (~10 MB); afterwards it works fully offline."
                ) from exc
            raise PluginError(f"Failed to load {MODEL_ID}: {exc}") from exc

        model.eval()
        device = self._requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = torch.device(device)
        model.to(self._device)

        data_cfg = timm.data.resolve_model_data_config(model)
        self._transform = timm.data.create_transform(**data_cfg, is_training=False)
        self._labels = self._load_labels()
        self._model = model
        logger.info("Loaded %s on %s", MODEL_ID, self._device)

    def _load_labels(self) -> Optional[List[str]]:
        """Best-effort human-readable ImageNet-1k labels."""
        try:
            from timm.data import ImageNetInfo

            info = ImageNetInfo()
            return [info.index_to_description(i, detailed=False) for i in range(1000)]
        except Exception:  # noqa: BLE001 - fall back to bare class indices
            logger.debug("ImageNet label metadata unavailable; using class indices")
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self, image_path: str | Path, top_k: Optional[int] = None
    ) -> ClassificationResult:
        """Classify a single image and return its top-k predictions."""
        self._ensure_model()
        import torch
        from PIL import Image

        path = Path(image_path)
        if not path.is_file():
            raise PluginError(f"Image not found: {path}")

        k = max(1, top_k or self.top_k)
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            raise PluginError(f"Cannot read image {path}: {exc}") from exc

        batch = self._transform(image).unsqueeze(0).to(self._device)
        with torch.inference_mode():
            probs = self._model(batch).softmax(dim=-1)
        scores, indices = probs.topk(min(k, probs.shape[-1]))

        predictions: List[Prediction] = []
        for score, index in zip(scores.squeeze(0).tolist(), indices.squeeze(0).tolist()):
            label = self._labels[index] if self._labels else f"class_{index}"
            predictions.append(Prediction(label=label, index=int(index), score=float(score)))
        return ClassificationResult(path=str(path), predictions=predictions)

    def tag_directory(
        self, directory: str | Path, top_k: Optional[int] = None
    ) -> List[ClassificationResult]:
        """Classify every image directly inside *directory* (non-recursive)."""
        root = Path(directory)
        if not root.is_dir():
            raise PluginError(f"Not a directory: {root}")

        results: List[ClassificationResult] = []
        for entry in sorted(root.iterdir()):
            if not entry.is_file() or entry.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            try:
                results.append(self.classify(entry, top_k=top_k))
            except PluginError as exc:
                logger.warning("Skipping %s: %s", entry.name, exc)
        return results


_plugin: Optional[MobileNetClassifierPlugin] = None


def register(cfg: Optional[Any] = None) -> MobileNetClassifierPlugin:
    """Create (or return) the singleton plugin instance, honoring HOMIE config.

    Reads ``plugins.mobilenet_classifier.{allow_download,top_k,device}`` from a
    :class:`homie.config.HomieConfig` when one is provided; otherwise uses the
    safe defaults (no downloads, top-5, auto device).
    """
    global _plugin
    if _plugin is not None:
        return _plugin

    allow_download = False
    top_k = 5
    device: Optional[str] = None
    if cfg is not None and hasattr(cfg, "get"):
        allow_download = bool(
            cfg.get("plugins", "mobilenet_classifier", "allow_download", default=False)
        )
        top_k = int(cfg.get("plugins", "mobilenet_classifier", "top_k", default=5))
        device = cfg.get("plugins", "mobilenet_classifier", "device", default=None)

    _plugin = MobileNetClassifierPlugin(
        allow_download=allow_download, top_k=top_k, device=device
    )
    _plugin.activate()
    return _plugin


def activate(cfg: Optional[Any] = None) -> MobileNetClassifierPlugin:
    """HOMIE plugin entrypoint."""
    return register(cfg)


def deactivate() -> None:
    """HOMIE plugin exit point."""
    global _plugin
    if _plugin is not None:
        _plugin.deactivate()
        _plugin = None


__all__ = [
    "MobileNetClassifierPlugin",
    "ClassificationResult",
    "Prediction",
    "PluginError",
    "activate",
    "deactivate",
    "register",
]
