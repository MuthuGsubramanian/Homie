"""MobileNetV3-Small image-classification plugin for HOMIE.

Backed by ``timm/mobilenetv3_small_100.lamb_in1k`` (~2.5M parameters), this
plugin gives HOMIE fast, fully local image classification and tagging that is
comfortable on CPU-only laptops and small edge boxes. Typical uses:

* tag screenshots / photos before they are embedded into ChromaDB for RAG
* let the planner answer "what is in this picture?" without any cloud call
* pre-filter camera or folder-watch automations by image content

Offline-first: weights are loaded from the local Hugging Face cache. The
first-time download is opt-in via ``allow_download`` (config or argument);
without it the plugin refuses to touch the network.

Config (``homie.config.yaml``)::

    plugins:
      image_classifier:
        model: mobilenetv3_small_100.lamb_in1k
        device: cpu             # cpu | cuda | mps
        top_k: 5
        allow_download: false   # opt-in one-time weight fetch

Dependencies: ``pip install timm torch pillow`` (inference itself is local).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # typing only; heavy deps stay lazy at runtime
    from PIL import Image
    from homie.config import HomieConfig

try:
    from homie.config import cfg_get
except ImportError:  # allow standalone use outside a HOMIE checkout
    def cfg_get(cfg: Any, *keys: str, default: Any = None) -> Any:
        cur = getattr(cfg, "raw", cfg) or {}
        for key in keys:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur


logger = logging.getLogger(__name__)

DEFAULT_MODEL = "mobilenetv3_small_100.lamb_in1k"
DEFAULT_TOP_K = 5


class ClassifierError(Exception):
    """Raised when the classifier cannot load weights or run inference."""


@dataclass(frozen=True)
class Prediction:
    """One top-k classification result."""

    label: str
    index: int
    score: float

    def as_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "index": self.index, "score": round(self.score, 6)}


class MobileNetV3Classifier:
    """Local ImageNet-1k classifier backed by timm's MobileNetV3-Small."""

    def __init__(
        self,
        cfg: Optional[HomieConfig] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        allow_download: Optional[bool] = None,
    ) -> None:
        self.model_name = model_name or cfg_get(
            cfg, "plugins", "image_classifier", "model", default=DEFAULT_MODEL
        )
        self.device = device or cfg_get(
            cfg, "plugins", "image_classifier", "device", default="cpu"
        )
        self.top_k = int(
            cfg_get(cfg, "plugins", "image_classifier", "top_k", default=DEFAULT_TOP_K)
        )
        if allow_download is None:
            allow_download = bool(
                cfg_get(cfg, "plugins", "image_classifier", "allow_download", default=False)
            )
        self.allow_download = allow_download
        self._model: Any = None
        self._transform: Any = None
        self._labels: Optional[List[str]] = None

    # ------------------------------------------------------------ lifecycle

    def activate(self) -> None:
        """Load weights and preprocessing. Idempotent; safe to call twice."""
        if self._model is not None:
            return
        try:
            import timm
            import torch  # noqa: F401  (verified importable before use)
        except ImportError as exc:
            raise ClassifierError(
                "timm and torch are required: pip install timm torch pillow"
            ) from exc

        if not self.allow_download:
            # Hard offline: only the local HF/timm cache may be used.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

        try:
            model = timm.create_model(self.model_name, pretrained=True)
        except Exception as exc:  # noqa: BLE001 - hub raises many exception types
            raise ClassifierError(
                f"Could not load '{self.model_name}' from the local cache. "
                "Enable plugins.image_classifier.allow_download for a one-time fetch."
            ) from exc

        model.eval()
        model.to(self.device)
        data_cfg = timm.data.resolve_model_data_config(model)
        self._transform = timm.data.create_transform(**data_cfg, is_training=False)
        self._labels = self._load_labels(timm)
        self._model = model
        logger.info(
            "MobileNetV3 classifier ready (model=%s device=%s)", self.model_name, self.device
        )

    def deactivate(self) -> None:
        """Drop the model and free memory."""
        self._model = None
        self._transform = None
        self._labels = None
        logger.info("MobileNetV3 classifier deactivated")

    # ------------------------------------------------------------ inference

    def classify(
        self, image: str | Path | Image.Image, top_k: Optional[int] = None
    ) -> List[Prediction]:
        """Return the top-k ImageNet-1k predictions for one image."""
        self.activate()
        import torch

        batch = self._transform(self._to_pil(image)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probs = self._model(batch).softmax(dim=-1)[0]
        k = min(top_k or self.top_k, int(probs.shape[-1]))
        scores, indices = torch.topk(probs, k=k)
        return [
            Prediction(label=self._label_for(int(i)), index=int(i), score=float(s))
            for s, i in zip(scores.tolist(), indices.tolist())
        ]

    def tag(self, image: str | Path | Image.Image, threshold: float = 0.15) -> List[str]:
        """Return label names above ``threshold`` â€” handy as RAG metadata."""
        return [p.label for p in self.classify(image) if p.score >= threshold]

    # ------------------------------------------------------------ helpers

    @staticmethod
    def _to_pil(image: str | Path | Image.Image) -> Image.Image:
        from PIL import Image as PILImage

        if isinstance(image, PILImage.Image):
            return image.convert("RGB")
        path = Path(image).expanduser()
        if not path.exists():
            raise ClassifierError(f"Image not found: {path}")
        return PILImage.open(path).convert("RGB")

    def _load_labels(self, timm_module: Any) -> Optional[List[str]]:
        try:
            info = timm_module.data.ImageNetInfo()
            return [info.index_to_description(i) for i in range(1000)]
        except Exception:  # noqa: BLE001 - older timm versions lack ImageNetInfo
            logger.debug("ImageNet label names unavailable; using class indices")
            return None

    def _label_for(self, index: int) -> str:
        if self._labels and 0 <= index < len(self._labels):
            return self._labels[index]
        return f"class_{index}"


_ACTIVE: Optional[MobileNetV3Classifier] = None


def activate(cfg: Optional[HomieConfig] = None) -> MobileNetV3Classifier:
    """Module-level activation hook for the HOMIE plugin loader."""
    global _ACTIVE
    if _ACTIVE is None:
        _ACTIVE = MobileNetV3Classifier(cfg)
    _ACTIVE.activate()
    return _ACTIVE


def deactivate() -> None:
    """Module-level deactivation hook for the HOMIE plugin loader."""
    global _ACTIVE
    if _ACTIVE is not None:
        _ACTIVE.deactivate()
        _ACTIVE = None


def register(cfg: Optional[HomieConfig] = None) -> Dict[str, Any]:
    """Describe the plugin to HOMIE's registry without loading any weights."""
    return {
        "name": "mobilenetv3_classifier",
        "version": "1.0.0",
        "capabilities": ["image_classification", "image_tagging"],
        "model": cfg_get(cfg, "plugins", "image_classifier", "model", default=DEFAULT_MODEL),
        "activate": lambda: activate(cfg),
        "deactivate": deactivate,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Classify an image with MobileNetV3-Small")
    parser.add_argument("image", help="Path to an image file")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--allow-download", action="store_true", help="Permit one-time weight fetch")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    clf = MobileNetV3Classifier(allow_download=args.allow_download)
    for pred in clf.classify(args.image, top_k=args.top_k):
        print(f"{pred.score:6.3f}  {pred.label}")
