"""Zero-shot image classification plugin for HOMIE.

Deepens HOMIE's existing CLIP image-search capability with zero-shot
classification built on ``openai/clip-vit-base-patch32``: an image is scored
against arbitrary user-supplied text labels with no training step, using
prompt-template ensembling for more robust label embeddings.

The plugin is fully local. Model weights are read from the local Hugging Face
cache by default; downloading from the network must be explicitly opted into
via ``allow_download=True``. Heavy dependencies (torch, transformers, PIL) are
imported lazily so HOMIE starts fast even when the plugin is installed but
unused.

Typical use::

    plugin = ClipZeroShotPlugin()
    plugin.activate()
    result = plugin.classify("photo.jpg", ["a cat", "a dog", "a laptop"])
    print(result.top())
    plugin.deactivate()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "openai/clip-vit-base-patch32"

DEFAULT_PROMPT_TEMPLATES: List[str] = [
    "a photo of a {}",
    "a close-up photo of a {}",
    "a blurry photo of a {}",
    "a drawing of a {}",
]


@dataclass
class LabelScore:
    """Single label with its softmax probability."""

    label: str
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "score": round(self.score, 6)}


@dataclass
class ZeroShotResult:
    """Ranked zero-shot classification result for one image."""

    image_path: str
    model_id: str
    scores: List[LabelScore] = field(default_factory=list)

    def top(self) -> Optional[LabelScore]:
        """Return the best-scoring label, if any."""
        return self.scores[0] if self.scores else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_path": self.image_path,
            "model_id": self.model_id,
            "scores": [s.to_dict() for s in self.scores],
        }


class ClipZeroShotPlugin:
    """Zero-shot image classification on top of CLIP ViT-B/32.

    Complements the existing CLIP image-search plugin: instead of retrieving
    similar images from ChromaDB, this scores a single image against a
    caller-supplied label set (e.g. auto-tagging screenshots before they are
    indexed for RAG).
    """

    name = "clip_zero_shot"
    version = "1.0.0"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: Optional[str] = None,
        allow_download: bool = False,
        prompt_templates: Optional[Sequence[str]] = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.allow_download = allow_download
        self.prompt_templates = list(prompt_templates or DEFAULT_PROMPT_TEMPLATES)
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._active = False

    # -- lifecycle -----------------------------------------------------

    def activate(self) -> None:
        """Mark the plugin active; model loading stays lazy until first use."""
        self._active = True
        logger.info(
            "clip_zero_shot activated (model=%s, allow_download=%s)",
            self.model_id,
            self.allow_download,
        )

    def deactivate(self) -> None:
        """Release the model and free GPU memory if it was loaded."""
        self._active = False
        self._model = None
        self._processor = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        logger.info("clip_zero_shot deactivated")

    # -- internals -----------------------------------------------------

    def _ensure_model(self) -> None:
        """Lazily import heavy deps and load CLIP, local cache first."""
        if self._model is not None:
            return
        if not self._active:
            raise RuntimeError("clip_zero_shot is not active; call activate() first")

        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        local_only = not self.allow_download
        try:
            self._model = CLIPModel.from_pretrained(self.model_id, local_files_only=local_only)
            self._processor = CLIPProcessor.from_pretrained(self.model_id, local_files_only=local_only)
        except OSError as exc:
            raise RuntimeError(
                f"CLIP weights for '{self.model_id}' not found in the local cache. "
                "Re-create the plugin with allow_download=True to fetch them once."
            ) from exc

        self._model.to(self.device)
        self._model.eval()
        logger.info("Loaded %s on %s", self.model_id, self.device)

    def _load_image(self, image_path: str | Path) -> Any:
        from PIL import Image

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return Image.open(path).convert("RGB")

    def _text_features(self, labels: Sequence[str], use_templates: bool) -> Any:
        """Build one normalized text embedding per label.

        With template ensembling each label is rendered through every prompt
        template and the resulting embeddings are averaged, which is markedly
        more robust than embedding a single raw label string.
        """
        torch = self._torch
        templates = self.prompt_templates if use_templates else ["{}"]
        features = []
        with torch.no_grad():
            for label in labels:
                prompts = [t.format(label) for t in templates]
                inputs = self._processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
                emb = self._model.get_text_features(**inputs)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                mean = emb.mean(dim=0)
                features.append(mean / mean.norm())
        return torch.stack(features)

    def _score_image(self, image_path: str | Path, text_emb: Any, labels: Sequence[str], top_k: int) -> ZeroShotResult:
        torch = self._torch
        image = self._load_image(image_path)
        with torch.no_grad():
            inputs = self._processor(images=image, return_tensors="pt").to(self.device)
            emb = self._model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            logits = (emb @ text_emb.T) * self._model.logit_scale.exp()
            probs = logits.softmax(dim=-1).squeeze(0).tolist()
        ranked = sorted(zip(labels, probs), key=lambda pair: pair[1], reverse=True)[:top_k]
        scores = [LabelScore(label=lab, score=float(p)) for lab, p in ranked]
        return ZeroShotResult(image_path=str(image_path), model_id=self.model_id, scores=scores)

    # -- public API ----------------------------------------------------

    def classify(
        self,
        image_path: str | Path,
        labels: Sequence[str],
        top_k: int = 5,
        use_templates: bool = True,
    ) -> ZeroShotResult:
        """Score one image against ``labels`` and return the top-k ranking."""
        if not labels:
            raise ValueError("labels must be a non-empty sequence")
        self._ensure_model()
        text_emb = self._text_features(labels, use_templates)
        return self._score_image(image_path, text_emb, labels, top_k)

    def classify_batch(
        self,
        image_paths: Sequence[str | Path],
        labels: Sequence[str],
        top_k: int = 5,
        use_templates: bool = True,
    ) -> List[ZeroShotResult]:
        """Classify several images against one shared label set.

        Text embeddings are computed once and reused across all images.
        """
        if not labels:
            raise ValueError("labels must be a non-empty sequence")
        self._ensure_model()
        text_emb = self._text_features(labels, use_templates)
        return [self._score_image(p, text_emb, labels, top_k) for p in image_paths]

    def suggest_tags(
        self,
        image_path: str | Path,
        vocabulary: Sequence[str],
        threshold: float = 0.15,
        max_tags: int = 5,
    ) -> List[str]:
        """Return vocabulary labels whose probability clears ``threshold``.

        Handy for producing ChromaDB metadata tags before an image is added
        to the CLIP image-search index.
        """
        result = self.classify(image_path, vocabulary, top_k=len(vocabulary))
        return [s.label for s in result.scores if s.score >= threshold][:max_tags]


def register(plugin_manager: Any = None) -> ClipZeroShotPlugin:
    """Entry point for HOMIE's plugin loader.

    Returns the plugin instance; if a manager with a ``register`` method is
    passed, the plugin is registered with it as well.
    """
    plugin = ClipZeroShotPlugin()
    if plugin_manager is not None and hasattr(plugin_manager, "register"):
        plugin_manager.register(plugin)
    return plugin


__all__ = [
    "ClipZeroShotPlugin",
    "ZeroShotResult",
    "LabelScore",
    "register",
    "DEFAULT_MODEL_ID",
    "DEFAULT_PROMPT_TEMPLATES",
]
