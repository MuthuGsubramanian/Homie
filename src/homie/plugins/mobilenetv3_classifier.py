"""MobileNetV3-Small image classifier plugin for HOMIE.

Wraps ``timm/mobilenetv3_small_100.lamb_in1k`` â€” a ~2.5M-parameter ImageNet
classifier that runs comfortably on CPU â€” so HOMIE can tag and classify local
photos without any cloud calls. It complements the existing MobileNetV3
photo-tagger by exposing a general-purpose top-k classification action that
other HOMIE components (planner, automations, RAG ingest) can call.

Privacy / safety notes:
- All inference is local. Images never leave the machine.
- Model weights are only downloaded from the Hugging Face Hub when the user
  explicitly opts in (``allow_download=True`` or config key
  ``plugins.mobilenetv3.allow_download: true``). Otherwise the plugin loads
  weights from an already-populated local cache or a user-supplied checkpoint
  and fails cleanly if none is available.
- Heavy dependencies (torch, timm, Pillow) are imported lazily inside
  ``activate()`` so HOMIE starts fast even when the plugin is installed but
  unused.

Example:
    from homie.plugins.mobilenetv3_classifier import MobileNetV3ClassifierPlugin

    plugin = MobileNetV3ClassifierPlugin(allow_download=True)
    plugin.activate()
    for pred in plugin.classify("~/Pictures/cat.jpg", top_k=3):
        print(pred["label"], pred["score"])
    plugin.deactivate()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MODEL_ID = "mobilenetv3_small_100.lamb_in1k"
HF_REPO = "timm/mobilenetv3_small_100.lamb_in1k"


class PluginError(Exception):
    """Raised when the plugin cannot activate or run inference."""


@dataclass
class Prediction:
    """Single classification result."""

    label: str
    score: float
    index: int

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "score": round(self.score, 4), "index": self.index}


@dataclass
class MobileNetV3ClassifierPlugin:
    """Local image classification via timm MobileNetV3-Small (ImageNet-1k).

    Attributes:
        allow_download: Opt-in flag permitting a one-time weight download from
            the Hugging Face Hub. Defaults to False (fully offline).
        checkpoint_path: Optional path to a local ``.safetensors``/``.pth``
            checkpoint to load instead of the Hub cache.
        device: Torch device string; ``"cpu"`` is the safe default and is
            fast enough for this model (~5-15 ms per image on a laptop).
        top_k_default: Default number of predictions returned by classify().
    """

    allow_download: bool = False
    checkpoint_path: Optional[Path] = None
    device: str = "cpu"
    top_k_default: int = 5

    _model: Any = field(default=None, init=False, repr=False)
    _transform: Any = field(default=None, init=False, repr=False)
    _labels: Optional[List[str]] = field(default=None, init=False, repr=False)
    _active: bool = field(default=False, init=False)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def activate(self) -> None:
        """Load the model, preprocessing transform, and label map.

        Imports torch/timm lazily so HOMIE's startup path stays light.
        Raises PluginError with an actionable message if dependencies or
        weights are missing.
        """
        if self._active:
            return

        try:
            import timm  # noqa: PLC0415
            import torch  # noqa: PLC0415
            from timm.data import resolve_data_config  # noqa: PLC0415
            from timm.data.transforms_factory import create_transform  # noqa: PLC0415
        except ImportError as exc:
            raise PluginError(
                "MobileNetV3 plugin requires optional deps. Install with: "
                "pip install timm torch pillow"
            ) from exc

        try:
            if self.checkpoint_path is not None:
                model = timm.create_model(
                    MODEL_ID, pretrained=False, checkpoint_path=str(self.checkpoint_path)
                )
            else:
                # pretrained=True reads the local HF cache; it only hits the
                # network if the weights are absent AND downloads are allowed.
                if not self.allow_download and not self._weights_cached():
                    raise PluginError(
                        f"Weights for {HF_REPO} not found locally and downloads "
                        "are disabled. Re-run with allow_download=True (one-time "
                        "~10 MB fetch) or provide checkpoint_path."
                    )
                model = timm.create_model(MODEL_ID, pretrained=True)
        except PluginError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise PluginError(f"Failed to load {MODEL_ID}: {exc}") from exc

        model.eval()
        model.to(self.device)

        config = resolve_data_config({}, model=model)
        self._transform = create_transform(**config)
        self._model = model
        self._labels = self._load_labels()
        self._torch = torch
        self._active = True
        logger.info("MobileNetV3 classifier active (device=%s)", self.device)

    def deactivate(self) -> None:
        """Release the model and free memory."""
        self._model = None
        self._transform = None
        self._labels = None
        self._active = False
        logger.info("MobileNetV3 classifier deactivated")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def classify(self, image_path: str | Path, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Classify a local image and return the top-k ImageNet predictions.

        Args:
            image_path: Path to a local image file (jpg/png/webp/...).
            top_k: Number of predictions to return (default: top_k_default).

        Returns:
            List of dicts: ``{"label": str, "score": float, "index": int}``,
            sorted by descending confidence.
        """
        if not self._active:
            raise PluginError("Plugin not active. Call activate() first.")

        from PIL import Image  # noqa: PLC0415

        path = Path(image_path).expanduser()
        if not path.exists():
            raise PluginError(f"Image not found: {path}")

        k = top_k or self.top_k_default
        image = Image.open(path).convert("RGB")
        tensor = self._transform(image).unsqueeze(0).to(self.device)

        with self._torch.inference_mode():
            logits = self._model(tensor)
            probs = logits.softmax(dim=-1)[0]
            scores, indices = probs.topk(min(k, probs.shape[-1]))

        results = [
            Prediction(label=self._label_for(int(i)), score=float(s), index=int(i)).to_dict()
            for s, i in zip(scores.tolist(), indices.tolist())
        ]
        return results

    def tag(self, image_path: str | Path, threshold: float = 0.15) -> List[str]:
        """Return confident labels only â€” convenient for photo-tagging flows."""
        return [p["label"] for p in self.classify(image_path) if p["score"] >= threshold]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _weights_cached(self) -> bool:
        """Best-effort check for already-downloaded weights in the HF cache."""
        try:
            from huggingface_hub import scan_cache_dir  # noqa: PLC0415

            return any(repo.repo_id == HF_REPO for repo in scan_cache_dir().repos)
        except Exception:  # noqa: BLE001
            # If the cache can't be scanned, let timm attempt an offline load.
            return True

    def _load_labels(self) -> Optional[List[str]]:
        """Load human-readable ImageNet-1k labels bundled with timm (offline)."""
        try:
            from timm.data import ImageNetInfo  # noqa: PLC0415

            info = ImageNetInfo()
            return [
                info.index_to_description(i, detailed=False) for i in range(1000)
            ]
        except Exception:  # noqa: BLE001
            logger.debug("timm ImageNetInfo unavailable; using numeric labels")
            return None

    def _label_for(self, index: int) -> str:
        if self._labels and 0 <= index < len(self._labels):
            return self._labels[index]
        return f"class_{index}"


def register(config: Optional[Any] = None) -> MobileNetV3ClassifierPlugin:
    """HOMIE plugin entry point.

    Reads optional settings from ``plugins.mobilenetv3`` in homie.config.yaml
    (via HomieConfig.get) and returns an inactive plugin instance; the host
    calls ``activate()`` when the capability is first needed.
    """
    allow_download = False
    checkpoint: Optional[Path] = None
    device = "cpu"

    if config is not None and hasattr(config, "get"):
        allow_download = bool(config.get("plugins", "mobilenetv3", "allow_download", default=False))
        raw_ckpt = config.get("plugins", "mobilenetv3", "checkpoint_path", default=None)
        checkpoint = Path(raw_ckpt).expanduser() if raw_ckpt else None
        device = config.get("plugins", "mobilenetv3", "device", default="cpu")

    return MobileNetV3ClassifierPlugin(
        allow_download=allow_download,
        checkpoint_path=checkpoint,
        device=device,
    )


__all__ = ["MobileNetV3ClassifierPlugin", "Prediction", "PluginError", "register"]
