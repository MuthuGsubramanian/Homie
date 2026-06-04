"""
Homie plugin: clip_vision
=========================

On-device image search and zero-shot tagging powered by OpenAI's
``openai/clip-vit-base-patch32`` model (Hugging Face Transformers).

This plugin lets Homie:

* Index a local directory of images into an in-memory vector store.
* Search those images by free-text query ("a screenshot of a terminal",
  "my cat sleeping", ...).
* Zero-shot tag a single image against an arbitrary set of candidate labels.

The plugin is fully local: it only touches the network when the model
weights are not yet cached *and* ``allow_download`` is explicitly enabled
in the plugin config. After the first download everything runs offline,
in keeping with Homie's local-first philosophy.

Dependencies (install once on the host running Homie):
    pip install torch transformers pillow

Typical wiring inside Homie::

    from homie.plugins import clip_vision
    plugin = clip_vision.register({
        "cache_dir": "~/.cache/homie/models",
        "index_path": "~/.cache/homie/clip_index.json",
        "allow_download": True,
        "device": "cpu",
    })
    plugin.index_directory("~/Pictures")
    hits = plugin.search("sunset over mountains", top_k=5)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

PLUGIN_NAME = "clip_vision"
DEFAULT_MODEL_ID = "openai/clip-vit-base-patch32"
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


@dataclass
class ClipPluginConfig:
    """Runtime configuration for the CLIP vision plugin."""

    model_id: str = DEFAULT_MODEL_ID
    cache_dir: Optional[Path] = None
    device: str = "cpu"
    index_path: Optional[Path] = None
    allow_download: bool = False
    batch_size: int = 16


@dataclass
class IndexedImage:
    """A single indexed image and its L2-normalized CLIP embedding."""

    path: str
    embedding: List[float] = field(default_factory=list)


class ClipVisionPlugin:
    """CLIP-based local image search and zero-shot tagging plugin.

    The model is loaded lazily on first use so that simply importing
    the plugin module is cheap. ``activate()`` performs the load and,
    when configured, restores a persisted index.
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        self.cfg: ClipPluginConfig = self._build_config(cfg or {})
        self._processor = None
        self._model = None
        self._torch = None
        self._index: List[IndexedImage] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @staticmethod
    def _build_config(raw: Dict[str, Any]) -> ClipPluginConfig:
        """Translate a plain dict into a :class:`ClipPluginConfig`."""
        cache_dir = raw.get("cache_dir")
        index_path = raw.get("index_path")
        return ClipPluginConfig(
            model_id=raw.get("model_id", DEFAULT_MODEL_ID),
            cache_dir=Path(cache_dir).expanduser() if cache_dir else None,
            device=str(raw.get("device", "cpu")),
            index_path=Path(index_path).expanduser() if index_path else None,
            allow_download=bool(raw.get("allow_download", False)),
            batch_size=int(raw.get("batch_size", 16)),
        )

    def _lazy_load_model(self) -> None:
        """Import heavy deps and load the CLIP model on first use."""
        if self._model is not None:
            return
        try:
            import torch  # type: ignore
            from transformers import CLIPModel, CLIPProcessor  # type: ignore
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "clip_vision plugin requires 'torch', 'transformers', and 'pillow'. "
                "Install with: pip install torch transformers pillow"
            ) from exc

        self._torch = torch
        load_kwargs: Dict[str, Any] = {}
        if self.cfg.cache_dir is not None:
            load_kwargs["cache_dir"] = str(self.cfg.cache_dir)
        if not self.cfg.allow_download:
            # Strictly offline: fail clearly if weights are not cached.
            load_kwargs["local_files_only"] = True

        logger.info(
            "Loading CLIP model %s on %s (download=%s)",
            self.cfg.model_id,
            self.cfg.device,
            self.cfg.allow_download,
        )
        self._processor = CLIPProcessor.from_pretrained(self.cfg.model_id, **load_kwargs)
        self._model = CLIPModel.from_pretrained(self.cfg.model_id, **load_kwargs)
        self._model.to(self.cfg.device).eval()

    def activate(self) -> None:
        """Load the model and (optionally) restore a persisted index."""
        self._lazy_load_model()
        if self.cfg.index_path and self.cfg.index_path.exists():
            try:
                self.load_index(self.cfg.index_path)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to load CLIP index from %s", self.cfg.index_path)
        logger.info(
            "clip_vision plugin activated (%d images indexed)", len(self._index)
        )

    def deactivate(self) -> None:
        """Persist the index (if configured) and release model resources."""
        if self.cfg.index_path is not None:
            try:
                self.save_index(self.cfg.index_path)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to persist CLIP index")
        self._model = None
        self._processor = None
        if self._torch is not None:
            try:
                if hasattr(self._torch, "cuda") and self._torch.cuda.is_available():
                    self._torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
        logger.info("clip_vision plugin deactivated")

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _embed_images(self, paths: Sequence[Path]):
        """Return L2-normalized CLIP image embeddings for ``paths``."""
        from PIL import Image  # type: ignore

        torch = self._torch
        assert torch is not None and self._processor is not None and self._model is not None
        images = []
        for p in paths:
            try:
                images.append(Image.open(p).convert("RGB"))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping %s: %s", p, exc)
        if not images:
            return torch.empty(0)
        inputs = self._processor(images=images, return_tensors="pt").to(self.cfg.device)
        with torch.no_grad():
            feats = self._model.get_image_features(**inputs)
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        return feats

    def _embed_text(self, texts: Sequence[str]):
        """Return L2-normalized CLIP text embeddings for ``texts``."""
        torch = self._torch
        assert torch is not None and self._processor is not None and self._model is not None
        inputs = self._processor(
            text=list(texts), return_tensors="pt", padding=True
        ).to(self.cfg.device)
        with torch.no_grad():
            feats = self._model.get_text_features(**inputs)
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        return feats

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_directory(self, directory: str | Path, recursive: bool = True) -> int:
        """Embed every supported image under ``directory`` and add to the index.

        Already-indexed paths are skipped, so this is safe to call repeatedly
        (e.g. on a scheduled automation).

        Returns the number of newly indexed images.
        """
        self._lazy_load_model()
        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Not a directory: {root}")

        iterator = root.rglob("*") if recursive else root.iterdir()
        files = [
            p for p in iterator
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
        ]
        already = {item.path for item in self._index}
        new_files = [p for p in files if str(p) not in already]

        added = 0
        for i in range(0, len(new_files), self.cfg.batch_size):
            batch = new_files[i : i + self.cfg.batch_size]
            feats = self._embed_images(batch)
            if int(getattr(feats, "shape", [0])[0]) == 0:
                continue
            for path, vec in zip(batch, feats):
                self._index.append(
                    IndexedImage(path=str(path), embedding=vec.cpu().tolist())
                )
                added += 1
        logger.info("Indexed %d new images (total %d)", added, len(self._index))
        return added

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Rank indexed images by cosine similarity to ``query``.

        Returns a list of ``(path, score)`` tuples, sorted descending.
        """
        if not self._index:
            return []
        self._lazy_load_model()
        torch = self._torch
        assert torch is not None
        text_feat = self._embed_text([query])[0]
        img_matrix = torch.tensor(
            [item.embedding for item in self._index],
            dtype=text_feat.dtype,
            device=self.cfg.device,
        )
        scores = (img_matrix @ text_feat).cpu().tolist()
        ranked = sorted(
            zip((item.path for item in self._index), scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]

    def tag(
        self,
        image_path: str | Path,
        candidate_labels: Sequence[str],
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """Zero-shot classify a single image against ``candidate_labels``.

        Returns ``(label, probability)`` tuples sorted by descending
        probability (softmax over the scaled cosine similarities).
        """
        if not candidate_labels:
            return []
        self._lazy_load_model()
        torch = self._torch
        assert torch is not None
        img_feats = self._embed_images([Path(image_path).expanduser()])
        if int(getattr(img_feats, "shape", [0])[0]) == 0:
            return []
        text_feats = self._embed_text(list(candidate_labels))
        logits = (img_feats @ text_feats.T)[0]
        probs = torch.softmax(logits * 100.0, dim=-1).cpu().tolist()
        scored = sorted(
            zip(list(candidate_labels), probs),
            key=lambda x: x[1],
            reverse=True,
        )
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_index(self, path: str | Path) -> Path:
        """Write the current index to a JSON file. Returns the resolved path."""
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_id": self.cfg.model_id,
            "items": [
                {"path": item.path, "embedding": item.embedding}
                for item in self._index
            ],
        }
        p.write_text(json.dumps(payload), encoding="utf-8")
        logger.info("Saved CLIP index (%d items) to %s", len(self._index), p)
        return p

    def load_index(self, path: str | Path) -> int:
        """Replace the in-memory index with the contents of ``path``."""
        p = Path(path).expanduser()
        if not p.exists():
            return 0
        data = json.loads(p.read_text(encoding="utf-8"))
        stored_model = data.get("model_id")
        if stored_model and stored_model != self.cfg.model_id:
            logger.warning(
                "Index was built with %s but current model is %s; embeddings may be incompatible",
                stored_model,
                self.cfg.model_id,
            )
        self._index = [
            IndexedImage(path=it["path"], embedding=list(it["embedding"]))
            for it in data.get("items", [])
        ]
        logger.info("Loaded CLIP index (%d items) from %s", len(self._index), p)
        return len(self._index)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def index_size(self) -> int:
        """Number of images currently held in the in-memory index."""
        return len(self._index)


# ----------------------------------------------------------------------
# Module-level entry points expected by the Homie plugin loader
# ----------------------------------------------------------------------

_plugin_instance: Optional[ClipVisionPlugin] = None


def register(cfg: Optional[Dict[str, Any]] = None) -> ClipVisionPlugin:
    """Create (if needed), activate, and return the singleton plugin."""
    global _plugin_instance
    if _plugin_instance is None:
        _plugin_instance = ClipVisionPlugin(cfg)
        _plugin_instance.activate()
    return _plugin_instance


def activate(cfg: Optional[Dict[str, Any]] = None) -> ClipVisionPlugin:
    """Alias for :func:`register` to match Homie's activate/deactivate pair."""
    return register(cfg)


def deactivate() -> None:
    """Tear down the singleton plugin instance, if one exists."""
    global _plugin_instance
    if _plugin_instance is not None:
        _plugin_instance.deactivate()
        _plugin_instance = None


__all__ = [
    "PLUGIN_NAME",
    "DEFAULT_MODEL_ID",
    "ClipPluginConfig",
    "ClipVisionPlugin",
    "IndexedImage",
    "register",
    "activate",
    "deactivate",
]
