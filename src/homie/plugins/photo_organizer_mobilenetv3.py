"""MobileNetV3 batch photo organizer plugin for HOMIE.

Extends HOMIE's existing photo-tagger integration of
``timm/mobilenetv3_small_100.lamb_in1k`` (a ~2.5M-parameter ImageNet-1k
classifier that runs comfortably on CPU) from single-image tagging to
on-device batch photo organization.

Given a directory of photos, the plugin classifies each image locally,
maps the fine-grained ImageNet labels onto a small set of human-friendly
folder categories (animals, food, vehicles, nature, screens_documents,
other), and files the photos into per-category subfolders â€” copying by
default (non-destructive) or moving when explicitly requested.

Privacy / safety:
- All inference happens on-device; no image ever leaves the machine.
- Model weights are only fetched from Hugging Face when the plugin is
  constructed with ``allow_download=True`` (opt-in). Otherwise the
  Hugging Face Hub is forced offline for the load and activation
  requires the weights to already be in the local cache.
- ``organize()`` defaults to ``dry_run=True`` and never overwrites an
  existing destination file.

Optional dependencies (not required by the rest of HOMIE):
    pip install timm torch pillow

Example:
    from homie.plugins.photo_organizer_mobilenetv3 import register

    plugin = register()
    plugin.activate()
    report = plugin.organize("~/Pictures/camera-roll", dry_run=True)
    print(report.summary())
    plugin.deactivate()
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)

MODEL_ID = "mobilenetv3_small_100.lamb_in1k"
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"})
DEFAULT_TOP_K = 5
DEFAULT_MIN_CONFIDENCE = 0.15
FALLBACK_CATEGORY = "other"

# Keyword rules mapping ImageNet-1k label words to folder categories.
# Single-word keywords match whole words only (so "car" does not match
# "cardigan"); multi-word keywords match as substrings.
CATEGORY_RULES: Dict[str, Tuple[str, ...]] = {
    "animals": (
        "dog", "terrier", "retriever", "spaniel", "hound", "cat", "tabby",
        "siamese", "bird", "finch", "owl", "eagle", "fish", "shark",
        "snake", "lizard", "frog", "spider", "beetle", "butterfly", "bee",
        "monkey", "ape", "elephant", "bear", "zebra", "horse", "sheep",
        "pig", "rabbit", "hamster", "fox", "wolf", "tiger", "lion",
        "leopard", "panda",
    ),
    "food": (
        "pizza", "cheeseburger", "hotdog", "pretzel", "bagel", "banana",
        "strawberry", "pineapple", "orange", "lemon", "broccoli",
        "espresso", "wine", "plate", "ice cream", "trifle", "guacamole",
        "carbonara", "burrito", "meat loaf",
    ),
    "vehicles": (
        "car", "cab", "convertible", "jeep", "limousine", "minivan",
        "pickup", "truck", "trailer", "bus", "motorcycle", "moped",
        "scooter", "bicycle", "mountain bike", "train", "locomotive",
        "airliner", "warplane", "boat", "canoe", "catamaran", "ship",
    ),
    "nature": (
        "alp", "volcano", "valley", "cliff", "seashore", "lakeside",
        "sandbar", "coral reef", "geyser", "promontory", "daisy",
        "sunflower", "mushroom",
    ),
    "screens_documents": (
        "monitor", "screen", "laptop", "notebook", "desktop computer",
        "web site", "menu", "book jacket", "envelope", "binder",
    ),
}


def _match_category(label: str) -> Optional[str]:
    """Return the folder category for an ImageNet label, if any rule matches."""
    text = label.lower()
    words = set(re.split(r"[^a-z]+", text))
    for category, keywords in CATEGORY_RULES.items():
        for keyword in keywords:
            if (" " in keyword and keyword in text) or keyword in words:
                return category
    return None


@dataclass
class TagResult:
    """Classification result for a single photo."""

    path: Path
    tags: List[Tuple[str, float]]
    category: str


@dataclass
class OrganizeReport:
    """Outcome of a batch organize run (or dry run)."""

    root: Path
    dry_run: bool
    planned: Dict[str, List[Path]] = field(default_factory=dict)
    skipped: List[Path] = field(default_factory=list)
    errors: List[Tuple[Path, str]] = field(default_factory=list)

    def summary(self) -> str:
        mode = "planned" if self.dry_run else "filed"
        total = sum(len(paths) for paths in self.planned.values())
        lines = [f"{mode} {total} photo(s) under {self.root}"]
        for category in sorted(self.planned):
            lines.append(f"  {category}: {len(self.planned[category])}")
        if self.skipped:
            lines.append(f"  skipped: {len(self.skipped)}")
        if self.errors:
            lines.append(f"  errors: {len(self.errors)}")
        return "\n".join(lines)


class PhotoOrganizerPlugin:
    """On-device batch photo organizer backed by MobileNetV3-Small.

    Loads ``timm/mobilenetv3_small_100.lamb_in1k`` once on ``activate()``
    and releases it on ``deactivate()``. All classification runs locally.
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str = "cpu",
        allow_download: bool = False,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.allow_download = allow_download
        self.top_k = max(1, top_k)
        self._model: Optional[Any] = None
        self._transform: Optional[Any] = None
        self._labels: List[str] = []
        self._torch: Optional[Any] = None
        self._image_cls: Optional[Any] = None
        self._prev_offline: Optional[str] = None
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    def activate(self) -> None:
        """Load the model and preprocessing pipeline (idempotent)."""
        if self._active:
            LOGGER.debug("PhotoOrganizerPlugin already active")
            return
        if not self.allow_download:
            # Force the HF Hub offline so activation can never trigger a
            # network call unless the user opted in.
            self._prev_offline = os.environ.get("HF_HUB_OFFLINE")
            os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            import timm
            import torch
            from PIL import Image
        except ImportError as exc:
            self._restore_offline_flag()
            raise RuntimeError(
                "PhotoOrganizerPlugin requires 'timm', 'torch' and 'pillow'. "
                "Install them with: pip install timm torch pillow"
            ) from exc

        try:
            model = timm.create_model(self.model_id, pretrained=True)
        except Exception as exc:  # noqa: BLE001
            self._restore_offline_flag()
            raise RuntimeError(
                f"Could not load weights for '{self.model_id}'. If this is the "
                "first run, re-create the plugin with allow_download=True to "
                "fetch them once from Hugging Face, then switch back to "
                "offline mode."
            ) from exc

        model.eval()
        model.to(self.device)
        data_cfg = timm.data.resolve_model_data_config(model)
        self._transform = timm.data.create_transform(**data_cfg, is_training=False)
        self._labels = self._load_labels(model)
        self._torch = torch
        self._image_cls = Image
        self._model = model
        self._active = True
        LOGGER.info(
            "PhotoOrganizerPlugin active (model=%s, device=%s)",
            self.model_id,
            self.device,
        )

    def deactivate(self) -> None:
        """Release the model and restore environment state (idempotent)."""
        self._model = None
        self._transform = None
        self._labels = []
        self._torch = None
        self._image_cls = None
        self._active = False
        self._restore_offline_flag()
        LOGGER.info("PhotoOrganizerPlugin deactivated")

    def tag_image(self, path: str | Path) -> TagResult:
        """Classify a single photo and return its top-k tags and category."""
        self._ensure_active()
        image = self._image_cls.open(path).convert("RGB")
        tensor = self._transform(image).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            probabilities = self._model(tensor).softmax(dim=1)[0]
        confidences, indices = probabilities.topk(
            min(self.top_k, probabilities.numel())
        )
        tags = [
            (self._labels[int(idx)], float(conf))
            for conf, idx in zip(confidences, indices)
        ]
        return TagResult(path=Path(path), tags=tags, category=self._categorize(tags))

    def organize(
        self,
        directory: str | Path,
        dry_run: bool = True,
        move: bool = False,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ) -> OrganizeReport:
        """Classify every photo in ``directory`` and file it by category.

        Non-recursive by design: only files directly inside ``directory``
        are considered, so already-organized subfolders are left alone.
        With ``dry_run=True`` (the default) nothing on disk changes; the
        returned report shows what would happen. Copies unless ``move``
        is True, and never overwrites an existing destination file.
        """
        self._ensure_active()
        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Not a directory: {root}")

        report = OrganizeReport(root=root, dry_run=dry_run)
        for path in sorted(root.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                result = self.tag_image(path)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Failed to classify %s: %s", path, exc)
                report.errors.append((path, str(exc)))
                continue

            top_confidence = result.tags[0][1] if result.tags else 0.0
            if top_confidence < min_confidence:
                report.skipped.append(path)
                continue

            destination_dir = root / result.category
            report.planned.setdefault(result.category, []).append(path)
            if dry_run:
                continue

            destination = destination_dir / path.name
            if destination.exists():
                report.planned[result.category].remove(path)
                report.skipped.append(path)
                continue
            destination_dir.mkdir(exist_ok=True)
            if move:
                shutil.move(str(path), str(destination))
            else:
                shutil.copy2(path, destination)
        return report

    def _categorize(self, tags: List[Tuple[str, float]]) -> str:
        for label, _confidence in tags:
            category = _match_category(label)
            if category:
                return category
        return FALLBACK_CATEGORY

    def _load_labels(self, model: Any) -> List[str]:
        num_classes = int(getattr(model, "num_classes", 1000))
        try:
            from timm.data import ImageNetInfo, infer_imagenet_subset

            subset = infer_imagenet_subset(model) or "imagenet-1k"
            info = ImageNetInfo(subset)
            return [
                info.index_to_description(i, detailed=False)
                for i in range(num_classes)
            ]
        except Exception:  # noqa: BLE001
            LOGGER.debug("ImageNet label lookup unavailable, using indices")
            return [f"class_{i}" for i in range(num_classes)]

    def _ensure_active(self) -> None:
        if not self._active:
            raise RuntimeError(
                "PhotoOrganizerPlugin is not active. Call activate() first."
            )

    def _restore_offline_flag(self) -> None:
        if self.allow_download:
            return
        if self._prev_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = self._prev_offline
        self._prev_offline = None


def register(cfg: Any = None) -> PhotoOrganizerPlugin:
    """Factory used by HOMIE's plugin loader.

    Reads optional settings from ``plugins.photo_organizer`` in the HOMIE
    config when a :class:`homie.config.HomieConfig` (or anything with the
    same ``get(*keys, default=...)`` signature) is supplied. The plugin is
    returned inactive; call ``activate()`` to load the model.
    """
    options: Dict[str, Any] = {}
    if cfg is not None and hasattr(cfg, "get"):
        options = {
            "device": cfg.get("plugins", "photo_organizer", "device", default="cpu"),
            "allow_download": bool(
                cfg.get("plugins", "photo_organizer", "allow_download", default=False)
            ),
            "top_k": int(
                cfg.get("plugins", "photo_organizer", "top_k", default=DEFAULT_TOP_K)
            ),
        }
    return PhotoOrganizerPlugin(**options)


__all__ = [
    "PhotoOrganizerPlugin",
    "TagResult",
    "OrganizeReport",
    "register",
    "MODEL_ID",
]
