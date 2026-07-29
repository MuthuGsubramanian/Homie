"""Zero-shot image classification plugin for HOMIE, built on OpenAI CLIP.

Iteration 2 of the already-shipped CLIP classifier plugin. Per the pipeline
discovery note (promote and iterate rather than rebuild), this version keeps
the shipped capability and layers in:

* Prompt-template ensembling ("a photo of a {}", ...) for higher accuracy.
* Label-embedding caching so repeated classifications skip text encoding.
* Batch classification of multiple images in one forward pass.
* Strictly local-first operation: model weights are loaded from the local
  Hugging Face cache; downloading is opt-in via ``allow_download=True``.

The plugin degrades gracefully when ``torch``/``transformers`` are not
installed and exposes ``register()`` plus ``activate()``/``deactivate()``
per HOMIE plugin conventions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

MODEL_ID = "openai/clip-vit-base-patch32"

DEFAULT_TEMPLATES: Tuple[str, ...] = (
    "a photo of a {}",
    "a close-up photo of a {}",
    "a photo of the {}",
)


class ClipPluginError(RuntimeError):
    """Raised when the CLIP plugin cannot load or run the model."""


@dataclass(frozen=True)
class ClassificationResult:
    """Single label/score pair returned by the classifier."""

    label: str
    score: float


class ClipZeroShotClassifier:
    """Zero-shot image classifier backed by ``openai/clip-vit-base-patch32``."""

    name = "clip_zero_shot_classifier"
    version = "2.0.0"

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: Optional[str] = None,
        allow_download: bool = False,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.allow_download = allow_download
        self.cache_dir = str(cache_dir) if cache_dir else None
        self._model = None
        self._processor = None
        self._label_cache: Dict[Tuple[str, ...], object] = {}

    # -- lifecycle -----------------------------------------------------

    def activate(self) -> None:
        """Load model + processor. Called by the HOMIE plugin manager."""
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise ClipPluginError(
                "CLIP plugin requires 'torch' and 'transformers'. "
                "Install with: pip install torch transformers pillow"
            ) from exc

        kwargs = {
            "cache_dir": self.cache_dir,
            "local_files_only": not self.allow_download,
        }
        try:
            self._model = CLIPModel.from_pretrained(self.model_id, **kwargs)
            self._processor = CLIPProcessor.from_pretrained(self.model_id, **kwargs)
        except OSError as exc:
            raise ClipPluginError(
                f"Model '{self.model_id}' not found in local cache. "
                "Re-activate with allow_download=True to fetch it once "
                "(opt-in network access)."
            ) from exc

        device = self.device or self._pick_device()
        self._model = self._model.to(device).eval()
        self.device = device
        logger.info("CLIP plugin active (model=%s device=%s)", self.model_id, device)

    def deactivate(self) -> None:
        """Release the model and cached label embeddings."""
        self._model = None
        self._processor = None
        self._label_cache.clear()
        logger.info("CLIP plugin deactivated")

    # -- public API ----------------------------------------------------

    def classify(
        self,
        image_path: str | Path,
        labels: Sequence[str],
        top_k: int = 3,
        templates: Sequence[str] = DEFAULT_TEMPLATES,
    ) -> List[ClassificationResult]:
        """Classify a single image against candidate ``labels``."""
        results = self.classify_batch(
            [image_path], labels, top_k=top_k, templates=templates
        )
        return results[0]

    def classify_batch(
        self,
        image_paths: Sequence[str | Path],
        labels: Sequence[str],
        top_k: int = 3,
        templates: Sequence[str] = DEFAULT_TEMPLATES,
    ) -> List[List[ClassificationResult]]:
        """Classify several images in one forward pass.

        Returns one ranked list of :class:`ClassificationResult` per image,
        highest score first.
        """
        self._ensure_active()
        if not labels:
            raise ValueError("labels must be a non-empty sequence")

        import torch
        from PIL import Image

        images = [Image.open(Path(p)).convert("RGB") for p in image_paths]
        text_features = self._embed_labels(tuple(labels), tuple(templates))

        inputs = self._processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self._model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = image_features @ text_features.T * self._model.logit_scale.exp()
        probs = logits.softmax(dim=-1).cpu()

        k = min(top_k, len(labels))
        out: List[List[ClassificationResult]] = []
        for row in probs:
            scores, idx = row.topk(k)
            out.append(
                [
                    ClassificationResult(label=labels[i], score=float(s))
                    for s, i in zip(scores.tolist(), idx.tolist())
                ]
            )
        return out

    # -- internals -----------------------------------------------------

    def _ensure_active(self) -> None:
        if self._model is None or self._processor is None:
            raise ClipPluginError("Plugin not active. Call activate() first.")

    def _embed_labels(self, labels: Tuple[str, ...], templates: Tuple[str, ...]):
        """Return normalized, template-ensembled text embeddings (cached)."""
        import torch

        cache_key = labels + ("|",) + templates
        cached = self._label_cache.get(cache_key)
        if cached is not None:
            return cached

        per_label = []
        for label in labels:
            prompts = [t.format(label) for t in templates]
            inputs = self._processor(
                text=prompts, return_tensors="pt", padding=True
            ).to(self.device)
            with torch.no_grad():
                feats = self._model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            mean = feats.mean(dim=0)
            per_label.append(mean / mean.norm())

        stacked = torch.stack(per_label)
        self._label_cache[cache_key] = stacked
        return stacked

    @staticmethod
    def _pick_device() -> str:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"


def register() -> ClipZeroShotClassifier:
    """Entry point used by the HOMIE plugin loader."""
    return ClipZeroShotClassifier()


__all__ = [
    "ClipZeroShotClassifier",
    "ClassificationResult",
    "ClipPluginError",
    "register",
    "MODEL_ID",
]
