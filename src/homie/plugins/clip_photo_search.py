"""CLIP photo search plugin for HOMIE.

Search local photos with natural-language text, fully offline, using
OpenAI's CLIP ViT-B/32 model (https://huggingface.co/openai/clip-vit-base-patch32).

Capabilities:
  * ``index_folder(path)``       - embed every image under a folder into ChromaDB.
  * ``search(query, k)``         - return the k photos best matching a text query.
  * ``classify(image, labels)``  - zero-shot label a single photo.

Privacy / network policy:
  Inference is 100% local. The only network access ever needed is the one-off
  model weight download from the Hugging Face Hub, which is OPT-IN: activate
  the plugin with ``allow_download=True`` once (or pre-fetch the weights with
  ``hf download openai/clip-vit-base-patch32``); afterwards the plugin runs
  with ``local_files_only=True`` and never touches the network.

Dependencies (not installed by HOMIE core):
    pip install torch transformers pillow chromadb
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

MODEL_ID = "openai/clip-vit-base-patch32"
DEFAULT_DATA_DIR = Path.home() / ".homie" / "clip_photos"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
COLLECTION_NAME = "homie_photos_clip"


@dataclass
class SearchHit:
    """One photo returned by a text query."""

    path: str
    score: float

    def as_dict(self) -> Dict[str, Any]:
        return {"path": self.path, "score": round(self.score, 4)}


class ClipPhotoSearchPlugin:
    """Text-to-image photo search backed by CLIP embeddings in ChromaDB.

    Image embeddings are computed once per photo and persisted in a local
    ChromaDB collection under ``~/.homie/clip_photos``, so repeated searches
    are fast and re-indexing is idempotent.
    """

    name = "clip_photo_search"
    version = "1.0.0"

    def __init__(self, data_dir: Path | str = DEFAULT_DATA_DIR) -> None:
        self.data_dir = Path(data_dir)
        self._model = None       # transformers.CLIPModel
        self._processor = None   # transformers.CLIPProcessor
        self._collection = None  # chromadb Collection
        self._active = False

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def activate(self, allow_download: bool = False) -> None:
        """Load CLIP and open the local vector store.

        Args:
            allow_download: Permit a one-time fetch of model weights from the
                Hugging Face Hub. Defaults to False so activation never makes
                a network call unless explicitly requested.
        """
        if self._active:
            return

        import chromadb  # embedded client - no server, no network
        from transformers import CLIPModel, CLIPProcessor

        self.data_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Loading %s (local_files_only=%s)", MODEL_ID, not allow_download)
        self._model = CLIPModel.from_pretrained(
            MODEL_ID, local_files_only=not allow_download
        )
        self._processor = CLIPProcessor.from_pretrained(
            MODEL_ID, local_files_only=not allow_download
        )
        self._model.eval()

        client = chromadb.PersistentClient(path=str(self.data_dir / "chroma"))
        self._collection = client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        self._active = True
        logger.info(
            "clip_photo_search active (%d photos indexed)", self._collection.count()
        )

    def deactivate(self) -> None:
        """Release the model and detach from the vector store."""
        self._model = None
        self._processor = None
        self._collection = None
        self._active = False
        logger.info("clip_photo_search deactivated")

    # ------------------------------------------------------------------ #
    # Embedding helpers
    # ------------------------------------------------------------------ #
    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError("Plugin not active. Call activate() first.")

    def _embed_images(self, paths: Sequence[Path]) -> List[List[float]]:
        import torch
        from PIL import Image

        images = [Image.open(p).convert("RGB") for p in paths]
        inputs = self._processor(images=images, return_tensors="pt")
        with torch.no_grad():
            feats = self._model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.tolist()

    def _embed_text(self, texts: Sequence[str]) -> List[List[float]]:
        import torch

        inputs = self._processor(
            text=list(texts), return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            feats = self._model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.tolist()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def index_folder(self, folder: Path | str, batch_size: int = 16) -> int:
        """Embed every image under *folder* (recursively) into the index.

        Files are keyed by absolute path and upserted, so re-running after
        adding photos is safe and idempotent. Corrupt or unreadable images
        are skipped with a logged warning rather than aborting the run.

        Returns:
            The number of images successfully indexed.
        """
        self._require_active()
        root = Path(folder).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Not a folder: {root}")

        paths = sorted(
            p
            for p in root.rglob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
        )
        indexed = 0
        for start in range(0, len(paths), batch_size):
            batch = paths[start : start + batch_size]
            try:
                embeddings = self._embed_images(batch)
            except Exception:  # noqa: BLE001 - skip unreadable/corrupt images
                logger.exception("Failed to embed batch starting at %s", batch[0])
                continue
            self._collection.upsert(
                ids=[str(p) for p in batch],
                embeddings=embeddings,
                metadatas=[{"path": str(p), "folder": str(root)} for p in batch],
            )
            indexed += len(batch)
            logger.info("Indexed %d/%d photos", indexed, len(paths))
        return indexed

    def search(self, query: str, k: int = 5) -> List[SearchHit]:
        """Return the *k* indexed photos that best match a text *query*.

        Scores are cosine similarities in [-1, 1]; higher is better.
        """
        self._require_active()
        if not query.strip():
            raise ValueError("query must be a non-empty string")
        total = self._collection.count()
        if total == 0:
            return []

        embedding = self._embed_text([query])[0]
        res = self._collection.query(
            query_embeddings=[embedding], n_results=min(k, total)
        )
        return [
            SearchHit(path=path, score=1.0 - distance)
            for path, distance in zip(res["ids"][0], res["distances"][0])
        ]

    def classify(
        self, image_path: Path | str, labels: Sequence[str]
    ) -> Dict[str, float]:
        """Zero-shot classify one photo against arbitrary *labels*.

        Returns:
            Mapping of label -> probability (softmax over the given labels).
        """
        self._require_active()
        if not labels:
            raise ValueError("labels must be non-empty")

        import torch
        from PIL import Image

        image = Image.open(Path(image_path).expanduser()).convert("RGB")
        prompts = [f"a photo of {label}" for label in labels]
        inputs = self._processor(
            text=prompts, images=image, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]
        return {label: float(p) for label, p in zip(labels, probs)}


_PLUGIN: Optional[ClipPhotoSearchPlugin] = None


def register() -> ClipPhotoSearchPlugin:
    """Entry point used by the HOMIE plugin loader."""
    global _PLUGIN
    if _PLUGIN is None:
        _PLUGIN = ClipPhotoSearchPlugin()
    return _PLUGIN


def activate(allow_download: bool = False) -> ClipPhotoSearchPlugin:
    """Module-level activate hook: register (if needed) and load the model."""
    plugin = register()
    plugin.activate(allow_download=allow_download)
    return plugin


def deactivate() -> None:
    """Module-level deactivate hook."""
    if _PLUGIN is not None:
        _PLUGIN.deactivate()


__all__ = [
    "ClipPhotoSearchPlugin",
    "SearchHit",
    "register",
    "activate",
    "deactivate",
]
