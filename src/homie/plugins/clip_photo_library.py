"""HOMIE plugin: local photo-library search and zero-shot classification with CLIP.

Built on ``openai/clip-vit-base-patch32``, this plugin extends HOMIE's existing
CLIP image-search capability into a full local photo-library workflow:

* ``index_library`` -- walk a photo folder, embed every image once, and cache
  embeddings on disk so re-runs are incremental.
* ``search`` -- natural-language search over the indexed library
  ("dog on the beach at sunset"), ranked by cosine similarity.
* ``classify`` -- zero-shot classification of one image against arbitrary
  user-supplied labels, no training required.
* ``organize`` -- bucket the whole indexed library under labels using the
  cached embeddings (a report only; no files are ever moved).

Everything runs locally on CPU by default. Model weights are loaded with
``local_files_only=True`` unless the user opts in to a one-time download via
``plugins.clip_photo.allow_download: true`` in ``homie.config.yaml`` (or
``allow_download=True`` in code), matching HOMIE's no-network-by-default rule.

Requires: ``transformers``, ``torch``, ``pillow``, ``numpy``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

MODEL_ID = "openai/clip-vit-base-patch32"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_INDEX_DIR = Path.home() / ".homie" / "clip_photo_index"
DEFAULT_PROMPT_TEMPLATE = "a photo of a {}"
CLIP_LOGIT_SCALE = 100.0


class ClipPluginError(Exception):
    """Raised when the CLIP photo plugin cannot complete an operation."""


@dataclass
class PhotoMatch:
    """A single search hit from the photo library."""

    path: Path
    score: float


@dataclass
class ClassificationResult:
    """Zero-shot classification outcome for one image."""

    path: Path
    label: str
    confidence: float
    scores: Dict[str, float] = field(default_factory=dict)


class ClipPhotoLibraryPlugin:
    """Local photo-library search and zero-shot tagging via CLIP."""

    def __init__(
        self,
        cfg: Optional[Any] = None,
        *,
        index_dir: Optional[Path] = None,
        allow_download: Optional[bool] = None,
        device: Optional[str] = None,
    ) -> None:
        def _cfg(*keys: str, default: Any = None) -> Any:
            if cfg is not None and hasattr(cfg, "get"):
                return cfg.get("plugins", "clip_photo", *keys, default=default)
            return default

        self.index_dir = Path(index_dir or _cfg("index_dir", default=DEFAULT_INDEX_DIR))
        self.allow_download = bool(
            allow_download if allow_download is not None else _cfg("allow_download", default=False)
        )
        self.device = device or _cfg("device", default="cpu")

        self._model: Any = None
        self._processor: Any = None
        self._paths: List[str] = []
        self._mtimes: List[float] = []
        self._vectors: Optional[np.ndarray] = None
        self._load_index()

    # ------------------------------------------------------------------ model

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise ClipPluginError(
                "Missing dependencies. Install with: pip install transformers torch pillow"
            ) from exc

        local_only = not self.allow_download
        try:
            self._model = CLIPModel.from_pretrained(MODEL_ID, local_files_only=local_only)
            self._processor = CLIPProcessor.from_pretrained(MODEL_ID, local_files_only=local_only)
        except OSError as exc:
            raise ClipPluginError(
                f"CLIP weights for {MODEL_ID} are not cached locally. Opt in to a one-time "
                "download by setting plugins.clip_photo.allow_download: true in homie.config.yaml."
            ) from exc
        self._model.to(self.device)
        self._model.eval()
        logger.info("Loaded %s on %s", MODEL_ID, self.device)

    def _embed_images(self, paths: Sequence[Path], batch_size: int = 8) -> np.ndarray:
        self._ensure_model()
        import torch
        from PIL import Image

        chunks: List[np.ndarray] = []
        for start in range(0, len(paths), batch_size):
            batch = paths[start : start + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch]
            inputs = self._processor(images=images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                feats = self._model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            chunks.append(feats.cpu().numpy())
        return np.concatenate(chunks, axis=0)

    def _embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        self._ensure_model()
        import torch

        inputs = self._processor(
            text=list(texts), return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            feats = self._model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()

    # ------------------------------------------------------------------ index

    def _index_files(self) -> tuple[Path, Path]:
        return self.index_dir / "vectors.npy", self.index_dir / "manifest.json"

    def _load_index(self) -> None:
        vec_file, manifest_file = self._index_files()
        if not (vec_file.exists() and manifest_file.exists()):
            return
        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            self._paths = manifest["paths"]
            self._mtimes = manifest["mtimes"]
            self._vectors = np.load(vec_file)
        except Exception:  # noqa: BLE001
            logger.warning("Corrupt CLIP index at %s; starting fresh", self.index_dir)
            self._paths, self._mtimes, self._vectors = [], [], None

    def _save_index(self) -> None:
        vec_file, manifest_file = self._index_files()
        self.index_dir.mkdir(parents=True, exist_ok=True)
        np.save(vec_file, self._vectors)
        manifest_file.write_text(
            json.dumps({"paths": self._paths, "mtimes": self._mtimes}), encoding="utf-8"
        )

    def index_library(self, root: str | Path, batch_size: int = 8) -> int:
        """Index (or incrementally re-index) every image under ``root``.

        Returns the number of newly embedded images.
        """
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            raise ClipPluginError(f"Photo folder not found: {root}")

        known = {p: m for p, m in zip(self._paths, self._mtimes)}
        pending: List[Path] = []
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in IMAGE_EXTENSIONS or not path.is_file():
                continue
            if known.get(str(path)) == path.stat().st_mtime:
                continue
            pending.append(path)

        if not pending:
            logger.info("CLIP index up to date (%d photos)", len(self._paths))
            return 0

        vectors = self._embed_images(pending, batch_size=batch_size)
        for path, vec in zip(pending, vectors):
            key = str(path)
            if key in known:
                idx = self._paths.index(key)
                self._vectors[idx] = vec  # type: ignore[index]
                self._mtimes[idx] = path.stat().st_mtime
            else:
                self._paths.append(key)
                self._mtimes.append(path.stat().st_mtime)
                self._vectors = (
                    vec[None, :]
                    if self._vectors is None
                    else np.vstack([self._vectors, vec[None, :]])
                )
        self._save_index()
        logger.info("Indexed %d new photos (%d total)", len(pending), len(self._paths))
        return len(pending)

    # ------------------------------------------------------------------ query

    def search(self, query: str, top_k: int = 5) -> List[PhotoMatch]:
        """Natural-language search over the indexed photo library."""
        if self._vectors is None or not self._paths:
            raise ClipPluginError("No photos indexed yet. Call index_library() first.")
        text_vec = self._embed_texts([query])[0]
        scores = self._vectors @ text_vec
        order = np.argsort(scores)[::-1][:top_k]
        return [PhotoMatch(path=Path(self._paths[i]), score=float(scores[i])) for i in order]

    def classify(
        self,
        image_path: str | Path,
        labels: Sequence[str],
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    ) -> ClassificationResult:
        """Zero-shot classify one image against arbitrary ``labels``."""
        if not labels:
            raise ClipPluginError("classify() needs at least one label")
        image_path = Path(image_path).expanduser()
        if not image_path.is_file():
            raise ClipPluginError(f"Image not found: {image_path}")
        image_vec = self._embed_images([image_path])[0]
        text_vecs = self._embed_texts([prompt_template.format(label) for label in labels])
        probs = _softmax(CLIP_LOGIT_SCALE * (text_vecs @ image_vec))
        scores = {label: float(p) for label, p in zip(labels, probs)}
        best = max(scores, key=scores.get)
        return ClassificationResult(
            path=image_path, label=best, confidence=scores[best], scores=scores
        )

    def organize(
        self,
        labels: Sequence[str],
        threshold: float = 0.5,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    ) -> Dict[str, List[Path]]:
        """Bucket every indexed photo under its best-matching label.

        Uses only the cached image embeddings, so no re-encoding happens.
        Photos whose best confidence falls below ``threshold`` land in
        ``\"unsorted\"``. Returns a report; it never moves or renames files.
        """
        if self._vectors is None or not self._paths:
            raise ClipPluginError("No photos indexed yet. Call index_library() first.")
        if not labels:
            raise ClipPluginError("organize() needs at least one label")

        text_vecs = self._embed_texts([prompt_template.format(label) for label in labels])
        logits = CLIP_LOGIT_SCALE * (self._vectors @ text_vecs.T)
        probs = np.apply_along_axis(_softmax, 1, logits)

        buckets: Dict[str, List[Path]] = {label: [] for label in labels}
        buckets["unsorted"] = []
        for path_str, row in zip(self._paths, probs):
            best_idx = int(np.argmax(row))
            if row[best_idx] >= threshold:
                buckets[labels[best_idx]].append(Path(path_str))
            else:
                buckets["unsorted"].append(Path(path_str))
        return buckets


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.exp(logits - np.max(logits))
    return shifted / shifted.sum()


# --------------------------------------------------------------------- plugin

_plugin: Optional[ClipPhotoLibraryPlugin] = None


def activate(cfg: Optional[Any] = None, **overrides: Any) -> ClipPhotoLibraryPlugin:
    """Activate the plugin and return the shared instance.

    ``cfg`` may be a :class:`homie.config.HomieConfig`; keyword overrides
    (``index_dir``, ``allow_download``, ``device``) win over config values.
    """
    global _plugin
    if _plugin is None:
        _plugin = ClipPhotoLibraryPlugin(cfg, **overrides)
        logger.info("clip_photo_library plugin activated")
    return _plugin


def deactivate() -> None:
    """Release the shared instance and free model memory."""
    global _plugin
    if _plugin is not None:
        _plugin._model = None
        _plugin._processor = None
        _plugin = None
        logger.info("clip_photo_library plugin deactivated")


__all__ = [
    "ClipPhotoLibraryPlugin",
    "ClipPluginError",
    "PhotoMatch",
    "ClassificationResult",
    "activate",
    "deactivate",
]
