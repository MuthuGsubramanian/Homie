"""MobileNetV3 image tagger plugin for HOMIE.

Provides fast, fully local image classification/tagging using the
``timm/mobilenetv3_small_100.lamb_in1k`` checkpoint -- a ~2.5M parameter
efficient CNN that runs comfortably on CPU (no GPU required). Predicted
tags can be fed into HOMIE's RAG store (ChromaDB) so local photos become
searchable by content, or surfaced through the assistant's chat loop
("what is in ~/Pictures/trip?").

Network policy: HOMIE is a local-first app, so downloading the model
weights from the Hugging Face Hub is strictly opt-in via the
``allow_download`` config flag. When the flag is off, only an
already-cached checkpoint is used and the plugin fails with a clear,
actionable error if none is present.

Configuration (homie.config.yaml):

    plugins:
      image_tagger:
        enabled: true
        allow_download: false   # opt-in: fetch weights from HF Hub once
        top_k: 5
        device: cpu             # or cuda / mps if available

Optional dependencies: ``timm``, ``torch``, ``pillow``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MODEL_NAME = "mobilenetv3_small_100.lamb_in1k"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class PluginError(Exception):
    """Raised when the tagger plugin cannot fulfil a request."""


@dataclass(frozen=True)
class ImageTag:
    """A single predicted label with its confidence score."""

    label: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "confidence": round(self.confidence, 4)}


class MobileNetV3TaggerPlugin:
    """Local image tagger backed by timm's MobileNetV3-Small (LAMB, ImageNet-1k).

    The model is loaded lazily on first use so that activating the plugin
    is cheap and HOMIE startup time is unaffected.
    """

    name = "image_tagger"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self.allow_download: bool = bool(cfg.get("allow_download", False))
        self.top_k: int = int(cfg.get("top_k", 5))
        self.device: str = str(cfg.get("device", "cpu"))
        self._model: Any = None
        self._transform: Any = None
        self._labels: Any = None
        self._active = False

    # -- lifecycle -----------------------------------------------------

    def activate(self) -> None:
        """Mark the plugin active; the model itself loads lazily on first use."""
        self._active = True
        logger.info(
            "image_tagger activated (model=%s, allow_download=%s)",
            MODEL_NAME,
            self.allow_download,
        )

    def deactivate(self) -> None:
        """Release model memory and mark the plugin inactive."""
        self._model = None
        self._transform = None
        self._labels = None
        self._active = False
        logger.info("image_tagger deactivated")

    # -- internals -----------------------------------------------------

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if not self._active:
            raise PluginError("image_tagger is not active; call activate() first")
        try:
            import timm
            import torch  # noqa: F401 - verifies the backend is importable
        except ImportError as exc:
            raise PluginError(
                "image_tagger requires optional dependencies: pip install timm torch pillow"
            ) from exc

        if not self.allow_download:
            # Local-first guarantee: never touch the network unless opted in.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

        try:
            model = timm.create_model(MODEL_NAME, pretrained=True)
        except Exception as exc:  # noqa: BLE001 - hub raises many error types
            if self.allow_download:
                raise PluginError(f"Failed to load {MODEL_NAME}: {exc}") from exc
            raise PluginError(
                f"Weights for {MODEL_NAME} are not cached locally and "
                "allow_download is false. Set plugins.image_tagger.allow_download: "
                "true in homie.config.yaml to fetch them once (~10 MB)."
            ) from exc

        model.eval()
        model.to(self.device)
        data_cfg = timm.data.resolve_model_data_config(model)
        self._transform = timm.data.create_transform(**data_cfg, is_training=False)
        self._model = model
        self._labels = self._load_labels()
        logger.info("image_tagger model ready on %s", self.device)

    @staticmethod
    def _load_labels() -> Any:
        try:
            from timm.data import ImageNetInfo

            return ImageNetInfo()
        except Exception:  # noqa: BLE001 - labels are cosmetic; degrade gracefully
            logger.debug("ImageNet label metadata unavailable; using class indices")
            return None

    def _label_for(self, index: int) -> str:
        if self._labels is None:
            return f"class_{index}"
        try:
            return self._labels.index_to_description(index)
        except Exception:  # noqa: BLE001
            return f"class_{index}"

    # -- public API ----------------------------------------------------

    def tag_image(self, image_path: str | Path, top_k: Optional[int] = None) -> List[ImageTag]:
        """Return the top-k ImageNet tags for a single image file."""
        self._ensure_model()
        import torch
        from PIL import Image

        path = Path(image_path).expanduser()
        if not path.is_file():
            raise PluginError(f"Image not found: {path}")

        k = top_k or self.top_k
        image = Image.open(path).convert("RGB")
        batch = self._transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probs = self._model(batch).softmax(dim=-1)[0]
        confidences, indices = probs.topk(k)
        return [
            ImageTag(label=self._label_for(int(idx)), confidence=float(conf))
            for conf, idx in zip(confidences, indices)
        ]

    def tag_directory(
        self, directory: str | Path, top_k: Optional[int] = None
    ) -> Dict[str, List[ImageTag]]:
        """Tag every supported image directly inside ``directory``.

        Unreadable or corrupt files are logged and skipped so a single bad
        file does not abort a batch run.
        """
        root = Path(directory).expanduser()
        if not root.is_dir():
            raise PluginError(f"Not a directory: {root}")
        results: Dict[str, List[ImageTag]] = {}
        for entry in sorted(root.iterdir()):
            if entry.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                results[str(entry)] = self.tag_image(entry, top_k=top_k)
            except PluginError:
                raise
            except Exception as exc:  # noqa: BLE001 - keep the batch going
                logger.warning("Skipping %s: %s", entry, exc)
        return results


_plugin: Optional[MobileNetV3TaggerPlugin] = None


def activate(config: Optional[Dict[str, Any]] = None) -> MobileNetV3TaggerPlugin:
    """Create (or reuse) the module-level plugin instance and activate it."""
    global _plugin
    if _plugin is None:
        _plugin = MobileNetV3TaggerPlugin(config)
    _plugin.activate()
    return _plugin


def deactivate() -> None:
    """Deactivate and drop the module-level plugin instance."""
    global _plugin
    if _plugin is not None:
        _plugin.deactivate()
        _plugin = None


def register(homie: Any, config: Optional[Dict[str, Any]] = None) -> MobileNetV3TaggerPlugin:
    """Register the plugin with a HOMIE host object.

    The host is duck-typed: if it exposes a ``plugins`` dict the instance is
    added under :attr:`MobileNetV3TaggerPlugin.name`; the instance is returned
    either way so callers can hold a direct reference.
    """
    plugin = activate(config)
    registry = getattr(homie, "plugins", None)
    if isinstance(registry, dict):
        registry[plugin.name] = plugin
    return plugin


__all__ = [
    "MobileNetV3TaggerPlugin",
    "ImageTag",
    "PluginError",
    "activate",
    "deactivate",
    "register",
]
