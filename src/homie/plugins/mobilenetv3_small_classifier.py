"""MobileNetV3-Small (LAMB, ImageNet-1k) image classification plugin for HOMIE.

Wraps the ``timm/mobilenetv3_small_100.lamb_in1k`` checkpoint -- a ~2.5M
parameter MobileNet-V3 classifier trained with the LAMB optimizer -- for
fast, fully local, on-device image classification. The model is small and
CPU-friendly, so it suits the same edge targets as HOMIE's existing
MobileNet tagger plugin, including low-power laptops and Android devices.

The plugin is offline-first: model weights are only downloaded from the
Hugging Face Hub when ``allow_download`` is explicitly enabled in config.
Once the weights are cached locally, all inference runs with no network
access at all.

Configuration (homie.config.yaml)::

    plugins:
      mobilenetv3_classifier:
        enabled: true
        allow_download: false   # opt-in: fetch weights from HF Hub once
        device: cpu             # or cuda
        top_k: 5

Usage::

    from homie.plugins.mobilenetv3_small_classifier import activate, deactivate

    plugin = activate(cfg)
    for label, score in plugin.classify("photo.jpg"):
        print(f"{label}: {score:.3f}")
    deactivate()

Dependencies (not installed by default): ``timm``, ``torch``, ``Pillow``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from homie.config import HomieConfig

MODEL_ID = "hf_hub:timm/mobilenetv3_small_100.lamb_in1k"
HF_CACHE_REPO_DIR = "models--timm--mobilenetv3_small_100.lamb_in1k"
DEFAULT_TOP_K = 5
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

logger = logging.getLogger(__name__)


class PluginError(Exception):
    """Raised when the classifier plugin cannot load or run."""


@dataclass
class ClassifierSettings:
    """Resolved plugin settings (config defaults + overrides)."""

    enabled: bool = True
    allow_download: bool = False
    device: str = "cpu"
    top_k: int = DEFAULT_TOP_K

    @classmethod
    def from_config(cls, cfg: Optional["HomieConfig"]) -> "ClassifierSettings":
        if cfg is None:
            return cls()
        from homie.config import cfg_get

        section = cfg_get(cfg, "plugins", "mobilenetv3_classifier", default={}) or {}
        return cls(
            enabled=bool(section.get("enabled", True)),
            allow_download=bool(section.get("allow_download", False)),
            device=str(section.get("device", "cpu")),
            top_k=int(section.get("top_k", DEFAULT_TOP_K)),
        )


class MobileNetV3ClassifierPlugin:
    """Local ImageNet-1k classifier backed by timm's MobileNetV3-Small."""

    name = "mobilenetv3_small_classifier"

    def __init__(self, settings: Optional[ClassifierSettings] = None) -> None:
        self.settings = settings or ClassifierSettings()
        self._model: Any = None
        self._transform: Any = None
        self._labels: List[str] = []
        self._active = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def activate(self) -> None:
        """Load the model into memory. Downloads weights only if opted in."""
        if self._active:
            return
        if not self.settings.enabled:
            raise PluginError("mobilenetv3_classifier is disabled in config")
        try:
            import timm  # noqa: F401  (heavy import kept lazy)
        except ImportError as exc:
            raise PluginError(
                "timm and torch are required for this plugin: pip install timm torch Pillow"
            ) from exc

        if not self.settings.allow_download and not self._weights_cached():
            raise PluginError(
                "Model weights are not cached locally and allow_download is false. "
                "Set plugins.mobilenetv3_classifier.allow_download: true to fetch "
                f"{MODEL_ID} once from the Hugging Face Hub, then it runs fully offline."
            )

        logger.info("Loading %s on %s", MODEL_ID, self.settings.device)
        self._model = timm.create_model(MODEL_ID, pretrained=True)
        self._model.eval()
        self._model.to(self.settings.device)

        data_cfg = timm.data.resolve_model_data_config(self._model)
        self._transform = timm.data.create_transform(**data_cfg, is_training=False)
        self._labels = self._load_labels()
        self._active = True
        logger.info("%s activated (%d labels)", self.name, len(self._labels))

    def deactivate(self) -> None:
        """Release the model and associated buffers."""
        self._model = None
        self._transform = None
        self._labels = []
        self._active = False
        logger.info("%s deactivated", self.name)

    @property
    def is_active(self) -> bool:
        return self._active

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def classify(
        self, image_path: str | Path, top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """Classify a single image and return ``(label, probability)`` pairs."""
        if not self._active:
            raise PluginError("Plugin not active; call activate() first")
        import torch
        from PIL import Image

        path = Path(image_path)
        if not path.is_file():
            raise PluginError(f"Image not found: {path}")
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise PluginError(f"Unsupported image type: {path.suffix}")

        image = Image.open(path).convert("RGB")
        tensor = self._transform(image).unsqueeze(0).to(self.settings.device)
        with torch.inference_mode():
            probs = self._model(tensor).softmax(dim=-1)[0]

        k = min(top_k or self.settings.top_k, probs.numel())
        top = torch.topk(probs, k)
        return [
            (self._label_for(idx), float(score))
            for score, idx in zip(top.values.tolist(), top.indices.tolist())
        ]

    def classify_batch(
        self, image_paths: Iterable[str | Path], top_k: Optional[int] = None
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Classify several images; per-file errors are logged, not raised."""
        results: Dict[str, List[Tuple[str, float]]] = {}
        for path in image_paths:
            try:
                results[str(path)] = self.classify(path, top_k=top_k)
            except PluginError as exc:
                logger.warning("Skipping %s: %s", path, exc)
        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _label_for(self, index: int) -> str:
        if 0 <= index < len(self._labels):
            return self._labels[index]
        return f"class_{index}"

    def _load_labels(self) -> List[str]:
        """Resolve human-readable ImageNet-1k labels; fall back to indices."""
        try:
            from timm.data import ImageNetInfo, infer_imagenet_subset

            info = ImageNetInfo(infer_imagenet_subset(self._model))
            return [
                info.index_to_description(i, detailed=False)
                for i in range(self._model.num_classes)
            ]
        except Exception:  # noqa: BLE001
            logger.debug("Could not resolve ImageNet labels, using indices")
            return []

    @staticmethod
    def _weights_cached() -> bool:
        """Best-effort check for an existing local HF Hub cache of the weights."""
        cache_root = Path(os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
        return (cache_root / "hub" / HF_CACHE_REPO_DIR).exists()


# ----------------------------------------------------------------------
# Module-level plugin lifecycle (HOMIE plugin contract)
# ----------------------------------------------------------------------
_plugin: Optional[MobileNetV3ClassifierPlugin] = None


def activate(cfg: Optional["HomieConfig"] = None) -> MobileNetV3ClassifierPlugin:
    """Create (if needed) and activate the shared plugin instance."""
    global _plugin
    if _plugin is None:
        _plugin = MobileNetV3ClassifierPlugin(ClassifierSettings.from_config(cfg))
    _plugin.activate()
    return _plugin


def deactivate() -> None:
    """Deactivate and drop the shared plugin instance."""
    global _plugin
    if _plugin is not None:
        _plugin.deactivate()
        _plugin = None


__all__ = [
    "MobileNetV3ClassifierPlugin",
    "ClassifierSettings",
    "PluginError",
    "activate",
    "deactivate",
]
