"""Homie Plugin: CLIP Visual Search & Tagging

Provides zero-shot image classification, visual search, and automatic tagging
using OpenAI's CLIP ViT-Large/14 model running locally via transformers.

Capabilities:
- Classify images against arbitrary text labels (zero-shot)
- Tag images with relevant categories without pre-training
- Search a local image collection by text query
- Rank images by similarity to a description

All inference runs locally on CPU/GPU â€” no network calls required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

MODEL_ID = "openai/clip-vit-large-patch14"
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_TOP_K = 5


@dataclass
class TagResult:
    """Result of tagging a single image."""

    path: Path
    tags: List[Tuple[str, float]]  # (label, confidence)


@dataclass
class SearchResult:
    """Result of searching images by text query."""

    path: Path
    score: float


@dataclass
class CLIPVisualSearchPlugin:
    """Zero-shot image classification and visual search plugin for Homie.

    Uses CLIP ViT-Large/14 to embed images and text into a shared space,
    enabling classification and retrieval without task-specific training.
    """

    model_id: str = MODEL_ID
    device: str = "cpu"
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    top_k: int = DEFAULT_TOP_K
    _model: Any = field(default=None, repr=False)
    _processor: Any = field(default=None, repr=False)
    _active: bool = field(default=False, repr=False)

    def activate(self) -> None:
        """Load the CLIP model and processor into memory."""
        if self._active:
            logger.debug("CLIP plugin already active")
            return

        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError(
                "transformers and torch are required. "
                "Install with: pip install transformers torch pillow"
            ) from exc

        logger.info("Loading CLIP model: %s on %s", self.model_id, self.device)
        self._processor = CLIPProcessor.from_pretrained(self.model_id)
        self._model = CLIPModel.from_pretrained(self.model_id).to(self.device)
        self._model.eval()
        self._active = True
        logger.info("CLIP plugin activated")

    def deactivate(self) -> None:
        """Unload model and free memory."""
        self._model = None
        self._processor = None
        self._active = False
        logger.info("CLIP plugin deactivated")

    @property
    def is_active(self) -> bool:
        return self._active

    def _ensure_active(self) -> None:
        if not self._active:
            raise RuntimeError("Plugin not activated. Call activate() first.")

    def _load_image(self, image_path: Path) -> Any:
        """Load an image from disk using PIL."""
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        return img

    def classify(
        self,
        image_path: Path,
        candidate_labels: Sequence[str],
    ) -> List[Tuple[str, float]]:
        """Classify an image against candidate text labels (zero-shot).

        Args:
            image_path: Path to the image file.
            candidate_labels: Text descriptions to score against the image.

        Returns:
            List of (label, probability) tuples sorted by confidence descending.
        """
        self._ensure_active()
        import torch

        image = self._load_image(Path(image_path))
        inputs = self._processor(
            text=list(candidate_labels),
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        logits = outputs.logits_per_image[0]
        probs = logits.softmax(dim=-1).cpu().tolist()

        scored = sorted(
            zip(candidate_labels, probs),
            key=lambda x: x[1],
            reverse=True,
        )
        return scored

    def tag_image(
        self,
        image_path: Path,
        tag_vocabulary: Sequence[str],
        threshold: Optional[float] = None,
    ) -> TagResult:
        """Assign tags to an image based on confidence threshold.

        Args:
            image_path: Path to the image file.
            tag_vocabulary: Possible tags to consider.
            threshold: Minimum confidence to assign a tag.

        Returns:
            TagResult with tags exceeding the threshold.
        """
        threshold = threshold or self.confidence_threshold
        scored = self.classify(image_path, tag_vocabulary)
        tags = [(label, conf) for label, conf in scored if conf >= threshold]
        return TagResult(path=Path(image_path), tags=tags)

    def search_images(
        self,
        query: str,
        image_paths: Sequence[Path],
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """Search a collection of images using a text query.

        Args:
            query: Natural language description of what to find.
            image_paths: Paths to images to search through.
            top_k: Number of top results to return.

        Returns:
            List of SearchResult sorted by relevance descending.
        """
        self._ensure_active()
        import torch

        top_k = top_k or self.top_k
        results: List[SearchResult] = []

        for img_path in image_paths:
            try:
                image = self._load_image(Path(img_path))
                inputs = self._processor(
                    text=[query],
                    images=image,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)

                with torch.no_grad():
                    outputs = self._model(**inputs)

                score = outputs.logits_per_image[0][0].cpu().item()
                results.append(SearchResult(path=Path(img_path), score=score))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping %s: %s", img_path, exc)

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def batch_tag(
        self,
        image_dir: Path,
        tag_vocabulary: Sequence[str],
        extensions: Sequence[str] = (".jpg", ".jpeg", ".png", ".webp", ".bmp"),
        threshold: Optional[float] = None,
    ) -> List[TagResult]:
        """Tag all images in a directory.

        Args:
            image_dir: Directory containing images.
            tag_vocabulary: Possible tags to consider.
            extensions: File extensions to include.
            threshold: Minimum confidence to assign a tag.

        Returns:
            List of TagResult for each image found.
        """
        image_dir = Path(image_dir)
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {image_dir}")

        image_paths = [
            p for p in sorted(image_dir.iterdir())
            if p.suffix.lower() in extensions
        ]

        results = []
        for img_path in image_paths:
            try:
                result = self.tag_image(img_path, tag_vocabulary, threshold)
                results.append(result)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to tag %s: %s", img_path, exc)

        return results


def register() -> Dict[str, Any]:
    """Register this plugin with the Homie plugin system.

    Returns:
        Plugin metadata and factory for the Homie loader.
    """
    return {
        "name": "clip_visual_search",
        "version": "1.0.0",
        "description": "Zero-shot image classification and visual search using CLIP ViT-Large/14",
        "model": MODEL_ID,
        "capabilities": ["image_classification", "image_tagging", "visual_search"],
        "factory": CLIPVisualSearchPlugin,
        "requires": ["transformers", "torch", "pillow"],
    }
