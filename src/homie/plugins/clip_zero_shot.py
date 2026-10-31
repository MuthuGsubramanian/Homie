"""Zero-shot image classification plugin for HOMIE.

Wraps OpenAI's CLIP ViT-B/32 checkpoint
(https://huggingface.co/openai/clip-vit-base-patch32) so HOMIE can classify
local images against arbitrary, user-supplied text labels without any
fine-tuning. It complements the existing CLIP image-search plugin: search
answers "find images like this", while this plugin answers "what is this
image, given these candidate labels?".

Everything runs locally. By default the plugin only uses model weights that
are already present in the local Hugging Face cache; downloading weights over
the network is opt-in via ``allow_download`` in the config.

Configuration (homie.config.yaml)::

    plugins:
      clip_zero_shot:
        enabled: true
        model_id: openai/clip-vit-base-patch32
        allow_download: false   # opt-in one-time weight download
        device: cpu             # "cpu" or "cuda"
        top_k: 5
        default_labels:
          - a photo of a person
          - a photo of an animal
          - a photo of a document
          - a screenshot of a computer program
          - a photo of food

Usage::

    from homie.config import load_config
    from homie.plugins.clip_zero_shot import ClipZeroShotPlugin

    plugin = ClipZeroShotPlugin(load_config())
    plugin.activate()
    for result in plugin.classify(
        "~/Pictures/lunch.jpg",
        labels=["a photo of food", "a photo of a cat"],
    ):
        print(f"{result.label}: {result.score:.3f}")
    plugin.deactivate()

Dependencies (installed manually, kept out of core requirements)::

    pip install torch transformers pillow
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from homie.config import HomieConfig, cfg_get


DEFAULT_MODEL_ID = "openai/clip-vit-base-patch32"
DEFAULT_TOP_K = 5
DEFAULT_LABELS: List[str] = [
    "a photo of a person",
    "a photo of an animal",
    "a photo of a document",
    "a screenshot of a computer program",
    "a photo of food",
]


class ClipZeroShotError(Exception):
    """Raised when the zero-shot classifier cannot load or run."""


@dataclass(frozen=True)
class ClassificationResult:
    """Single label/confidence pair for one image."""

    label: str
    score: float


class ClipZeroShotPlugin:
    """Zero-shot image classifier backed by a local CLIP ViT-B/32 model.

    The model and processor are loaded lazily on :meth:`activate` and
    released on :meth:`deactivate`, so an inactive plugin costs nothing.
    Heavy imports (torch, transformers, PIL) happen only inside methods,
    keeping HOMIE startup unaffected when the plugin is disabled.
    """

    name = "clip_zero_shot"

    def __init__(self, cfg: HomieConfig) -> None:
        self.cfg = cfg
        self._model: Any = None
        self._processor: Any = None
        self._active = False

    # -- configuration helpers ----------------------------------------------

    @property
    def enabled(self) -> bool:
        return bool(cfg_get(self.cfg, "plugins", self.name, "enabled", default=False))

    @property
    def model_id(self) -> str:
        return str(
            cfg_get(self.cfg, "plugins", self.name, "model_id", default=DEFAULT_MODEL_ID)
        )

    @property
    def allow_download(self) -> bool:
        return bool(
            cfg_get(self.cfg, "plugins", self.name, "allow_download", default=False)
        )

    @property
    def device(self) -> str:
        return str(cfg_get(self.cfg, "plugins", self.name, "device", default="cpu"))

    @property
    def default_labels(self) -> List[str]:
        labels = cfg_get(self.cfg, "plugins", self.name, "default_labels", default=None)
        if isinstance(labels, list) and labels:
            return [str(label) for label in labels]
        return list(DEFAULT_LABELS)

    # -- lifecycle -----------------------------------------------------------

    def activate(self) -> None:
        """Load the CLIP model into memory (idempotent)."""
        if self._active:
            return
        if not self.enabled:
            raise ClipZeroShotError(
                f"Plugin '{self.name}' is disabled. Set plugins.{self.name}.enabled: true "
                "in homie.config.yaml."
            )
        self._model, self._processor = self._load_model()
        self._active = True
        logging.info(
            "Plugin %s activated (model=%s device=%s)",
            self.name,
            self.model_id,
            self.device,
        )

    def deactivate(self) -> None:
        """Release the model and free memory (idempotent)."""
        if not self._active:
            return
        self._model = None
        self._processor = None
        self._active = False
        logging.info("Plugin %s deactivated", self.name)

    # -- inference -----------------------------------------------------------

    def classify(
        self,
        image_path: str | Path,
        labels: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
    ) -> List[ClassificationResult]:
        """Classify one local image against candidate text labels.

        Args:
            image_path: Path to a local image file (jpg/png/webp/...).
            labels: Candidate labels. Falls back to the configured
                ``default_labels``. Prompt-style labels ("a photo of a cat")
                score noticeably better than bare nouns ("cat").
            top_k: Number of results to return, best first. Defaults to the
                configured ``plugins.clip_zero_shot.top_k`` (or 5).

        Returns:
            Results sorted by descending score. Scores are softmax
            probabilities over the supplied labels and sum to 1.0.

        Raises:
            ClipZeroShotError: If the plugin is inactive, the image is
                missing/unreadable, or no labels are available.
        """
        if not self._active:
            raise ClipZeroShotError("Plugin is not active. Call activate() first.")

        candidates = [str(label) for label in (labels or self.default_labels)]
        if not candidates:
            raise ClipZeroShotError("No labels supplied and no default_labels configured.")

        path = Path(image_path).expanduser()
        if not path.is_file():
            raise ClipZeroShotError(f"Image not found: {path}")

        import torch
        from PIL import Image

        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            raise ClipZeroShotError(f"Could not read image {path}: {exc}") from exc

        inputs = self._processor(
            text=candidates, images=image, return_tensors="pt", padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0].tolist()

        limit = top_k or int(
            cfg_get(self.cfg, "plugins", self.name, "top_k", default=DEFAULT_TOP_K)
        )
        ranked = sorted(zip(candidates, probs), key=lambda pair: pair[1], reverse=True)
        return [
            ClassificationResult(label=label, score=float(score))
            for label, score in ranked[:limit]
        ]

    # -- internals -----------------------------------------------------------

    def _load_model(self) -> Tuple[Any, Any]:
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise ClipZeroShotError(
                "transformers/torch are not installed. "
                "Run: pip install torch transformers pillow"
            ) from exc

        local_only = not self.allow_download
        try:
            model = CLIPModel.from_pretrained(self.model_id, local_files_only=local_only)
            processor = CLIPProcessor.from_pretrained(
                self.model_id, local_files_only=local_only
            )
        except OSError as exc:
            if local_only:
                raise ClipZeroShotError(
                    f"Model '{self.model_id}' is not in the local cache and downloads "
                    f"are disabled. Set plugins.{self.name}.allow_download: true for a "
                    "one-time download, then set it back to false."
                ) from exc
            raise ClipZeroShotError(
                f"Failed to load model '{self.model_id}': {exc}"
            ) from exc

        model.eval()
        return model.to(self.device), processor


def register(cfg: HomieConfig) -> ClipZeroShotPlugin:
    """Create (but do not activate) the plugin for HOMIE's plugin registry."""
    return ClipZeroShotPlugin(cfg)


__all__ = [
    "ClipZeroShotPlugin",
    "ClassificationResult",
    "ClipZeroShotError",
    "register",
    "DEFAULT_MODEL_ID",
]
