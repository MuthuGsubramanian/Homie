"""MobileNetV3 batch photo organizer plugin for HOMIE.

Extends HOMIE's existing MobileNet tagger with on-device *batch* photo
organization, powered by ``timm/mobilenetv3_small_100.lamb_in1k`` (an
ImageNet-1k classifier, ~2.5M params, fast on CPU -- ideal for edge and
laptop use).

Capabilities:

* ``tag_image``  -- top-k ImageNet labels for a single photo.
* ``organize``   -- sweep a folder tree, classify every photo locally,
  and sort copies (or moves) into coarse category folders such as
  ``animals/``, ``food/`` or ``vehicles/``, writing an ``index.json``
  manifest next to them.

Safety defaults follow HOMIE conventions: ``organize`` is a *dry run*
that copies (never moves) unless explicitly told otherwise, and model
weights are only fetched from the Hugging Face Hub when the plugin is
activated with ``allow_download: true`` -- otherwise the local cache is
used (``HF_HUB_OFFLINE`` is set), so no network calls happen without
opt-in.

Requires: ``pip install timm torch pillow``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

LOGGER = logging.getLogger("homie.plugins.mobilenet_photo_organizer")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff"}

# Coarse destination folders mapped to keywords matched against the
# predicted ImageNet label text. First match wins; unmatched photos are
# filed under "misc", low-confidence ones under "unsorted".
CATEGORY_RULES: List[Tuple[str, Tuple[str, ...]]] = [
    ("animals", ("dog", "cat", "bird", "fish", "horse", "bear", "spider",
                 "snake", "insect", "terrier", "retriever", "tabby",
                 "monkey", "elephant", "butterfly", "fox", "rabbit")),
    ("food", ("pizza", "burger", "sandwich", "coffee", "wine", "fruit",
              "mushroom", "banana", "orange", "plate", "soup",
              "ice cream", "pretzel", "bakery", "broccoli")),
    ("vehicles", ("car", "truck", "bus", "motorcycle", "bicycle", "train",
                  "boat", "ship", "airplane", "airliner", "cab", "van",
                  "scooter", "tractor", "locomotive")),
    ("nature", ("mountain", "valley", "cliff", "volcano", "lakeside",
                "seashore", "coral", "geyser", "alp", "promontory")),
    ("buildings", ("castle", "church", "palace", "bridge", "lighthouse",
                   "barn", "monastery", "dome", "library", "cinema",
                   "boathouse", "greenhouse")),
    ("people", ("suit", "jersey", "gown", "uniform", "academic", "wig",
                "bow tie", "sweatshirt")),
    ("screens_docs", ("monitor", "screen", "laptop", "notebook",
                      "desktop computer", "web site", "menu",
                      "book jacket", "envelope", "keyboard")),
]


@dataclass
class TaggedPhoto:
    """Classification result for one photo."""

    path: Path
    label: str
    confidence: float
    category: str
    top_k: List[Tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "category": self.category,
            "top_k": [[name, round(prob, 4)] for name, prob in self.top_k],
        }


class MobileNetPhotoOrganizer:
    """Batch photo tagger and organizer built on mobilenetv3_small_100.

    The model is loaded lazily on first use so activating the plugin is
    cheap; heavy imports (torch/timm/PIL) stay inside methods.
    """

    MODEL_ID = "hf_hub:timm/mobilenetv3_small_100.lamb_in1k"

    def __init__(self, device: str = "cpu", allow_download: bool = False,
                 top_k: int = 3) -> None:
        self.device = device
        self.allow_download = allow_download
        self.top_k = top_k
        self._model: Any = None
        self._transform: Any = None
        self._labels: Optional[List[str]] = None

    # -- model lifecycle -------------------------------------------------
    def load(self) -> None:
        """Load model, transforms and label map (idempotent)."""
        if self._model is not None:
            return
        if not self.allow_download:
            # Respect HOMIE's no-network-unless-opt-in policy: only use
            # already-cached weights.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import timm
        from timm.data import (ImageNetInfo, create_transform,
                               infer_imagenet_subset, resolve_data_config)

        LOGGER.info("Loading %s (device=%s)", self.MODEL_ID, self.device)
        model = timm.create_model(self.MODEL_ID, pretrained=True)
        model.eval().to(self.device)
        data_config = resolve_data_config({}, model=model)
        self._transform = create_transform(**data_config)
        subset = infer_imagenet_subset(model) or "imagenet-1k"
        info = ImageNetInfo(subset)
        self._labels = [info.index_to_description(i)
                        for i in range(model.num_classes)]
        self._model = model

    def unload(self) -> None:
        """Drop model references so memory can be reclaimed."""
        self._model = None
        self._transform = None
        self._labels = None

    # -- single-image tagging --------------------------------------------
    def tag_image(self, path: str | Path,
                  top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        """Return ``[(label, probability), ...]`` for one image."""
        self.load()
        import torch
        from PIL import Image

        k = top_k or self.top_k
        img = Image.open(path).convert("RGB")
        batch = self._transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self._model(batch).softmax(dim=-1)[0]
        values, indices = probs.topk(k)
        return [(self._labels[int(i)], float(v))
                for v, i in zip(values, indices)]

    @staticmethod
    def categorize(label: str) -> str:
        """Map a fine-grained ImageNet label to a coarse folder name."""
        text = label.lower()
        for folder, keywords in CATEGORY_RULES:
            if any(word in text for word in keywords):
                return folder
        return "misc"

    # -- batch organization ----------------------------------------------
    @staticmethod
    def _iter_photos(source: Path) -> Iterable[Path]:
        for candidate in sorted(source.rglob("*")):
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
                yield candidate

    def organize(
        self,
        source_dir: str | Path,
        dest_dir: str | Path | None = None,
        move: bool = False,
        dry_run: bool = True,
        min_confidence: float = 0.15,
    ) -> Dict[str, Any]:
        """Classify every photo under ``source_dir`` and sort by category.

        Defaults are deliberately safe: ``dry_run=True`` reports what
        *would* happen without touching files, and files are copied
        rather than moved unless ``move=True``. Photos whose top-1
        confidence falls below ``min_confidence`` go to ``unsorted/``.
        When not a dry run, an ``index.json`` manifest is written into
        ``dest_dir`` (default ``<source>/organized``).
        """
        source = Path(source_dir).expanduser().resolve()
        if not source.is_dir():
            raise NotADirectoryError(f"Not a directory: {source}")
        dest = (Path(dest_dir).expanduser().resolve()
                if dest_dir else source / "organized")

        tagged: List[TaggedPhoto] = []
        errors: List[str] = []
        for photo in self._iter_photos(source):
            if dest == photo.parent or dest in photo.parents:
                continue  # never re-ingest our own output folder
            try:
                top = self.tag_image(photo)
            except Exception as exc:  # noqa: BLE001 - one bad file must not kill the batch
                errors.append(f"{photo}: {exc}")
                continue
            label, confidence = top[0]
            category = (self.categorize(label)
                        if confidence >= min_confidence else "unsorted")
            tagged.append(TaggedPhoto(photo, label, confidence, category, top))

        counts: Dict[str, int] = {}
        for item in tagged:
            counts[item.category] = counts.get(item.category, 0) + 1
            if dry_run:
                continue
            target_dir = dest / item.category
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / item.path.name
            if target.exists():
                # Disambiguate name collisions deterministically.
                suffix = abs(hash(str(item.path))) % 10_000
                target = target_dir / f"{item.path.stem}_{suffix}{item.path.suffix}"
            if move:
                shutil.move(str(item.path), target)
            else:
                shutil.copy2(item.path, target)

        report: Dict[str, Any] = {
            "source": str(source),
            "dest": str(dest),
            "dry_run": dry_run,
            "move": move,
            "total": len(tagged),
            "categories": counts,
            "errors": errors,
            "photos": [item.to_dict() for item in tagged],
        }
        if not dry_run:
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "index.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8")
        LOGGER.info("Organized %d photos into %d categories (dry_run=%s)",
                    len(tagged), len(counts), dry_run)
        return report


_PLUGIN: Optional[MobileNetPhotoOrganizer] = None


def activate(config: Optional[Dict[str, Any]] = None) -> MobileNetPhotoOrganizer:
    """HOMIE plugin entry point; returns a ready-to-use organizer.

    Recognised config keys: ``device`` (default ``cpu``),
    ``allow_download`` (default False -- offline cache only),
    ``top_k`` (default 3) and ``preload`` (default False).
    """
    global _PLUGIN
    cfg = config or {}
    _PLUGIN = MobileNetPhotoOrganizer(
        device=str(cfg.get("device", "cpu")),
        allow_download=bool(cfg.get("allow_download", False)),
        top_k=int(cfg.get("top_k", 3)),
    )
    if cfg.get("preload", False):
        _PLUGIN.load()
    return _PLUGIN


def deactivate() -> None:
    """Release the model and clear the plugin singleton."""
    global _PLUGIN
    if _PLUGIN is not None:
        _PLUGIN.unload()
    _PLUGIN = None


__all__ = [
    "MobileNetPhotoOrganizer",
    "TaggedPhoto",
    "CATEGORY_RULES",
    "activate",
    "deactivate",
]
