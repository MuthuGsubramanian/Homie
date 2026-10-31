"""CLIP-based zero-shot image classification plugin for HOMIE.

This plugin wraps OpenAI's CLIP ViT-B/32 model (``openai/clip-vit-base-patch32``)
to provide lightweight, on-device zero-shot image classification. Given an
image and a list of candidate text labels, it returns ranked predictions
without requiring any task-specific fine-tuning.

The model is loaded lazily inside :meth:`activate` and cached in memory until
:meth:`deactivate` is called. The weights and processor files are downloaded
from the Hugging Face Hub on first activation (network access is required
once); every subsequent inference runs entirely locally, matching HOMIE's
local-first design.

Typical use cases inside HOMIE:
  - Tagging screenshots captured by automations.
  - Detecting on-screen content categories before routing tasks.
  - Lightweight visual triage for user-supplied images.

Dependencies (install separately, kept out of HOMIE core):
    pip install transformers torch pillow
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "openai/clip-vit-base-patch32"
DEFAULT_PROMPT_TEMPLATE = "a photo of {label}"


@dataclass
class ClassificationResult:
    """One ranked label/score pair returned by :meth:`ClipImageClassifierPlugin.classify`."""

    label: str
    score: float


@dataclass
class ClipPluginState:
    """Runtime state for the CLIP plugin.

    Kept as a dataclass so callers (and tests) can introspect what is loaded
    without poking at private attributes.
    """

    model_id: str = DEFAULT_MODEL_ID
    device: str = "cpu"
    cache_dir: Optional[Path] = None
    model: Any = None
    processor: Any = None
    activated: bool = False


class ClipImageClassifierPlugin:
    """Zero-shot image classifier backed by ``openai/clip-vit-base-patch32``.

    The plugin follows HOMIE's activate/deactivate lifecycle::

        plugin = ClipImageClassifierPlugin()
        plugin.activate()
        results = plugin.classify("photo.jpg", ["a cat", "a dog", "a car"])
        plugin.deactivate()

    All heavy imports (``torch``, ``transformers``, ``PIL``) happen inside
    :meth:`activate` so the module is cheap to import in environments where
    the optional vision stack is not installed.
    """

    name = "clip_image_classifier"
    version = "1.0.0"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Configure the plugin without loading any weights.

        Args:
            model_id: Hugging Face model id. Defaults to the base CLIP model.
            device: ``"cpu"``, ``"cuda"``, or ``"mps"``. Auto-detected if ``None``.
            cache_dir: Optional override for the Hugging Face cache directory.
                Useful for sandboxed HOMIE installs that pin all model data
                under ``~/.homie/models``.
        """
        self.state = ClipPluginState(
            model_id=model_id,
            device=device or self._auto_device(),
            cache_dir=Path(cache_dir) if cache_dir else None,
        )

    @staticmethod
    def _auto_device() -> str:
        """Pick the best available local device without importing torch eagerly."""
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:  # noqa: BLE001
            pass
        return "cpu"

    def activate(self) -> None:
        """Load the CLIP model and processor into memory.

        The first call may download ~600MB from Hugging Face Hub. Subsequent
        activations reuse the local cache and run fully offline.

        Raises:
            RuntimeError: If required optional dependencies are missing.
        """
        if self.state.activated:
            logger.debug("%s already activated", self.name)
            return

        try:
            import torch  # type: ignore
            from transformers import CLIPModel, CLIPProcessor  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "CLIP plugin requires 'transformers', 'torch', and 'pillow'. "
                "Install with: pip install transformers torch pillow"
            ) from exc

        cache_dir = str(self.state.cache_dir) if self.state.cache_dir else None
        logger.info(
            "Loading CLIP model %s on %s", self.state.model_id, self.state.device
        )

        processor = CLIPProcessor.from_pretrained(
            self.state.model_id, cache_dir=cache_dir
        )
        model = CLIPModel.from_pretrained(
            self.state.model_id, cache_dir=cache_dir
        )
        model.eval()
        model.to(self.state.device)

        self.state.model = model
        self.state.processor = processor
        self.state.activated = True
        logger.info("CLIP plugin activated (device=%s)", self.state.device)

    def deactivate(self) -> None:
        """Release model resources from memory."""
        if not self.state.activated:
            return
        self.state.model = None
        self.state.processor = None
        self.state.activated = False
        try:
            import torch  # type: ignore

            if self.state.device == "cuda":
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        logger.info("CLIP plugin deactivated")

    def _ensure_active(self) -> None:
        if not self.state.activated:
            raise RuntimeError(
                f"{self.name} is not active. Call activate() before classify()."
            )

    def _load_image(self, image: Union[str, Path, Any]) -> Any:
        """Coerce ``image`` to a PIL ``Image`` in RGB mode."""
        from PIL import Image  # type: ignore

        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.is_file():
                raise FileNotFoundError(f"Image not found: {path}")
            return Image.open(path).convert("RGB")
        # Assume a PIL Image (or compatible) was passed directly.
        return image

    def classify(
        self,
        image: Union[str, Path, Any],
        candidate_labels: Sequence[str],
        top_k: Optional[int] = None,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    ) -> List[ClassificationResult]:
        """Score ``image`` against ``candidate_labels`` using CLIP.

        Args:
            image: File path, ``pathlib.Path``, or a PIL ``Image.Image``.
            candidate_labels: Free-form text labels to score the image against.
            top_k: If provided, return only the highest-scoring ``top_k`` labels.
            prompt_template: Format string used to build each text prompt.
                ``{label}`` is substituted with each candidate label.

        Returns:
            A list of :class:`ClassificationResult` ordered by descending score.
        """
        self._ensure_active()
        if not candidate_labels:
            raise ValueError("candidate_labels must contain at least one label")

        import torch  # type: ignore

        pil_image = self._load_image(image)
        prompts = [prompt_template.format(label=label) for label in candidate_labels]

        inputs = self.state.processor(
            text=prompts,
            images=pil_image,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.state.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.state.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=-1).squeeze(0).tolist()

        ranked: List[ClassificationResult] = sorted(
            (
                ClassificationResult(label=label, score=float(score))
                for label, score in zip(candidate_labels, probs)
            ),
            key=lambda r: r.score,
            reverse=True,
        )
        if top_k is not None:
            ranked = ranked[: max(1, int(top_k))]
        return ranked

    def info(self) -> Dict[str, Any]:
        """Return a small dict describing the plugin state (for HOMIE dashboards)."""
        return {
            "name": self.name,
            "version": self.version,
            "model_id": self.state.model_id,
            "device": self.state.device,
            "activated": self.state.activated,
        }


# Module-level lifecycle hooks expected by HOMIE's plugin loader.
_REGISTERED_INSTANCE: Optional[ClipImageClassifierPlugin] = None


def register() -> ClipImageClassifierPlugin:
    """Create (or return) the shared plugin instance for HOMIE to consume."""
    global _REGISTERED_INSTANCE
    if _REGISTERED_INSTANCE is None:
        _REGISTERED_INSTANCE = ClipImageClassifierPlugin()
    return _REGISTERED_INSTANCE


def activate() -> ClipImageClassifierPlugin:
    """Module-level activate hook used by HOMIE's plugin loader."""
    instance = register()
    instance.activate()
    return instance


def deactivate() -> None:
    """Module-level deactivate hook used by HOMIE's plugin loader."""
    global _REGISTERED_INSTANCE
    if _REGISTERED_INSTANCE is not None:
        _REGISTERED_INSTANCE.deactivate()


__all__ = [
    "ClipImageClassifierPlugin",
    "ClipPluginState",
    "ClassificationResult",
    "register",
    "activate",
    "deactivate",
    "DEFAULT_MODEL_ID",
    "DEFAULT_PROMPT_TEMPLATE",
]
