"""Local photo search and organization plugin for HOMIE.

This plugin wraps the ``openai/clip-vit-base-patch32`` model to provide
zero-shot image understanding that runs entirely on the local machine:

* ``index_directory`` - scan a folder of photos and build an embedding index.
* ``search`` - find photos matching a free-text query ("dog on a beach").
* ``classify`` - zero-shot classify a single image against custom labels.
* ``suggest_organization`` - map every indexed photo to the best-fitting
  label so HOMIE can propose album/folder groupings.

Privacy and safety notes
------------------------
* All inference happens locally via ``transformers`` + ``torch``.
* Model weights are only fetched from the Hugging Face Hub when the user
  explicitly opts in (``plugins.photo_search.allow_download: true`` in
  ``homie.config.yaml``). Otherwise loading uses ``local_files_only=True``
  and requires a pre-cached copy of the model.
* The embedding index lives under ``~/.homie/photo_index`` and contains
  only file paths and float vectors - never the images themselves.

Heavy dependencies (``torch``, ``transformers``, ``pillow``) are imported
lazily inside :meth:`PhotoSearchPlugin.activate` so importing this module
never pulls them in.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from homie.config import HomieConfig, cfg_get

logger = logging.getLogger(__name__)

PLUGIN_NAME = "photo_search"
DEFAULT_MODEL_ID = "openai/clip-vit-base-patch32"
DEFAULT_INDEX_DIR = Path.home() / ".homie" / "photo_index"
INDEX_FILE_NAME = "clip_index.json"
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"})


class PluginError(Exception):
    """Raised when the photo search plugin cannot complete an operation."""


@dataclass
class PhotoMatch:
    """A single search hit: an image path plus its cosine similarity score."""

    path: str
    score: float


class PhotoSearchPlugin:
    """Zero-shot photo search and organization backed by a local CLIP model.

    Typical usage::

        plugin = PhotoSearchPlugin(cfg)
        plugin.activate()
        plugin.index_directory(Path.home() / "Pictures")
        hits = plugin.search("sunset over the ocean", top_k=5)
        plugin.deactivate()

    Configuration (all optional, under ``plugins.photo_search`` in
    ``homie.config.yaml``)::

        plugins:
          photo_search:
            model_id: openai/clip-vit-base-patch32
            allow_download: false   # opt-in network access
            index_dir: ~/.homie/photo_index
            batch_size: 8
    """

    def __init__(self, cfg: Optional[HomieConfig] = None) -> None:
        self.cfg = cfg
        self.model_id: str = str(self._cfg("model_id", DEFAULT_MODEL_ID))
        self.allow_download: bool = bool(self._cfg("allow_download", False))
        self.index_dir: Path = Path(
            str(self._cfg("index_dir", str(DEFAULT_INDEX_DIR)))
        ).expanduser()
        self.batch_size: int = int(self._cfg("batch_size", 8))
        self.active: bool = False
        self._torch: Any = None
        self._model: Any = None
        self._processor: Any = None
        self._paths: List[str] = []
        self._embeddings: Any = None  # torch.Tensor of shape [n, dim]

    def _cfg(self, key: str, default: Any) -> Any:
        if self.cfg is None:
            return default
        return cfg_get(self.cfg, "plugins", PLUGIN_NAME, key, default=default)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def activate(self) -> None:
        """Load the CLIP model and any previously saved index."""
        if self.active:
            return
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise PluginError(
                "photo_search requires 'torch', 'transformers' and 'pillow'. "
                "Install them with: pip install torch transformers pillow"
            ) from exc

        local_only = not self.allow_download
        logger.info(
            "Loading CLIP model %s (local_files_only=%s)", self.model_id, local_only
        )
        try:
            self._model = CLIPModel.from_pretrained(
                self.model_id, local_files_only=local_only
            )
            self._processor = CLIPProcessor.from_pretrained(
                self.model_id, local_files_only=local_only
            )
        except OSError as exc:
            raise PluginError(
                f"Could not load '{self.model_id}' from the local cache. "
                "Set plugins.photo_search.allow_download: true in "
                "homie.config.yaml to opt in to a one-time download, or place "
                "the model in the Hugging Face cache manually."
            ) from exc

        self._torch = torch
        self._model.eval()
        self._load_index()
        self.active = True
        logger.info(
            "photo_search plugin active (%d photos indexed)", len(self._paths)
        )

    def deactivate(self) -> None:
        """Persist the index and release the model from memory."""
        if not self.active:
            return
        self._save_index()
        self._model = None
        self._processor = None
        self._embeddings = None
        self._torch = None
        self.active = False
        logger.info("photo_search plugin deactivated")

    def _require_active(self) -> None:
        if not self.active:
            raise PluginError("Plugin is not active. Call activate() first.")

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------
    def _embed_images(self, paths: Sequence[Path]) -> Any:
        """Return L2-normalized CLIP image embeddings for the given paths."""
        from PIL import Image

        torch = self._torch
        chunks: List[Any] = []
        batch: List[Any] = []

        def flush() -> None:
            if not batch:
                return
            inputs = self._processor(images=batch, return_tensors="pt")
            with torch.inference_mode():
                feats = self._model.get_image_features(**inputs)
            chunks.append(feats / feats.norm(dim=-1, keepdim=True))
            batch.clear()

        for path in paths:
            with Image.open(path) as img:
                batch.append(img.convert("RGB"))
            if len(batch) >= self.batch_size:
                flush()
        flush()
        return torch.cat(chunks, dim=0)

    def _embed_texts(self, texts: Sequence[str]) -> Any:
        """Return L2-normalized CLIP text embeddings for the given strings."""
        torch = self._torch
        inputs = self._processor(
            text=list(texts), return_tensors="pt", padding=True, truncation=True
        )
        with torch.inference_mode():
            feats = self._model.get_text_features(**inputs)
        return feats / feats.norm(dim=-1, keepdim=True)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def index_directory(self, directory: str | Path, recursive: bool = True) -> int:
        """Embed every image under ``directory`` and add it to the index.

        Already-indexed paths are skipped, so re-running is cheap.
        Returns the number of newly indexed photos.
        """
        self._require_active()
        root = Path(directory).expanduser()
        if not root.is_dir():
            raise PluginError(f"Not a directory: {root}")

        pattern = root.rglob("*") if recursive else root.glob("*")
        known = set(self._paths)
        new_paths = sorted(
            p
            for p in pattern
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
            and str(p.resolve()) not in known
        )
        if not new_paths:
            logger.info("No new photos found under %s", root)
            return 0

        logger.info("Indexing %d new photos under %s", len(new_paths), root)
        embedded: List[Path] = []
        chunks: List[Any] = []
        for path in new_paths:
            try:
                chunks.append(self._embed_images([path]))
                embedded.append(path)
            except Exception:  # noqa: BLE001 - skip unreadable/corrupt images
                logger.warning("Skipping unreadable image %s", path)

        if not embedded:
            return 0

        torch = self._torch
        new_embeddings = torch.cat(chunks, dim=0)
        if self._embeddings is None:
            self._embeddings = new_embeddings
        else:
            self._embeddings = torch.cat([self._embeddings, new_embeddings], dim=0)
        self._paths.extend(str(p.resolve()) for p in embedded)
        self._save_index()
        return len(embedded)

    def clear_index(self) -> None:
        """Forget all indexed photos (does not touch the image files)."""
        self._paths = []
        self._embeddings = None
        index_path = self.index_dir / INDEX_FILE_NAME
        if index_path.exists():
            index_path.unlink()
        logger.info("photo_search index cleared")

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 5) -> List[PhotoMatch]:
        """Return the ``top_k`` indexed photos best matching a text query."""
        self._require_active()
        if not query.strip():
            raise PluginError("Query must not be empty")
        if self._embeddings is None or not self._paths:
            return []

        text = self._embed_texts([query])
        scores = (self._embeddings @ text.T).squeeze(-1)
        k = min(top_k, len(self._paths))
        top = scores.topk(k)
        return [
            PhotoMatch(path=self._paths[idx], score=float(score))
            for score, idx in zip(top.values.tolist(), top.indices.tolist())
        ]

    def classify(
        self, image_path: str | Path, labels: Sequence[str]
    ) -> Dict[str, float]:
        """Zero-shot classify one image; returns ``{label: probability}``."""
        self._require_active()
        if not labels:
            raise PluginError("At least one label is required")
        path = Path(image_path).expanduser()
        if not path.is_file():
            raise PluginError(f"Image not found: {path}")

        from PIL import Image

        torch = self._torch
        prompts = [f"a photo of a {label}" for label in labels]
        with Image.open(path) as img:
            inputs = self._processor(
                text=prompts,
                images=img.convert("RGB"),
                return_tensors="pt",
                padding=True,
            )
        with torch.inference_mode():
            outputs = self._model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=-1).squeeze(0)
        return {label: float(p) for label, p in zip(labels, probs.tolist())}

    def suggest_organization(
        self, labels: Sequence[str], min_score: float = 0.2
    ) -> Dict[str, List[str]]:
        """Group every indexed photo under its best-matching label.

        Returns ``{label: [photo paths]}``; photos whose best similarity is
        below ``min_score`` land under the ``"unsorted"`` key. This only
        proposes a grouping - it never moves or renames files.
        """
        self._require_active()
        if not labels:
            raise PluginError("At least one label is required")
        groups: Dict[str, List[str]] = {label: [] for label in labels}
        groups["unsorted"] = []
        if self._embeddings is None or not self._paths:
            return groups

        prompts = [f"a photo of a {label}" for label in labels]
        text = self._embed_texts(prompts)
        scores = self._embeddings @ text.T  # [n_photos, n_labels]
        best = scores.max(dim=-1)
        for path, score, idx in zip(
            self._paths, best.values.tolist(), best.indices.tolist()
        ):
            if score < min_score:
                groups["unsorted"].append(path)
            else:
                groups[labels[idx]].append(path)
        return groups

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _save_index(self) -> None:
        if self._embeddings is None or not self._paths:
            return
        self.index_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_id": self.model_id,
            "paths": self._paths,
            "embeddings": self._embeddings.tolist(),
        }
        index_path = self.index_dir / INDEX_FILE_NAME
        index_path.write_text(json.dumps(payload), encoding="utf-8")
        logger.debug("Saved photo index (%d entries) to %s", len(self._paths), index_path)

    def _load_index(self) -> None:
        index_path = self.index_dir / INDEX_FILE_NAME
        if not index_path.exists():
            return
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Ignoring corrupt photo index at %s", index_path)
            return
        if payload.get("model_id") != self.model_id:
            logger.warning(
                "Photo index was built with %s, expected %s - rebuilding required",
                payload.get("model_id"),
                self.model_id,
            )
            return
        self._paths = list(payload.get("paths", []))
        embeddings = payload.get("embeddings") or []
        if embeddings:
            self._embeddings = self._torch.tensor(embeddings)


def register(cfg: Optional[HomieConfig] = None) -> PhotoSearchPlugin:
    """Entry point used by the HOMIE plugin loader.

    Returns an inactive plugin instance; the host calls ``activate()`` when
    the user enables photo search and ``deactivate()`` on shutdown.
    """
    return PhotoSearchPlugin(cfg)


__all__ = [
    "PhotoSearchPlugin",
    "PhotoMatch",
    "PluginError",
    "register",
]
