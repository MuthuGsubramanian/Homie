"""Homie CLIP vision plugin.

Provides local zero-shot image classification, image-text matching, and
semantic image search using ``openai/clip-vit-large-patch14`` via the
Hugging Face ``transformers`` library.

All inference runs locally on the user's device. The model weights are
downloaded from the Hugging Face Hub on first activation and cached under
the user's local HF cache directory (``~/.cache/huggingface``). No
image data, prompts, or embeddings ever leave the machine.

Integration with Homie:
    * Registers itself via ``register(host)`` so the Homie plugin host can
      attach activate/deactivate lifecycle hooks.
    * Reads configuration from the same ``HomieConfig`` mechanism used by
      the rest of the project (``plugins.clip_vision.*`` keys).
    * Exposes a small, typed API surface (``classify_image``,
      ``score_image_text``, ``index_directory``, ``search_images``) that
      higher level Homie features (assistant tool calls, dashboard
      endpoints, RAG over local photos) can call directly.

Configuration keys (all optional, sensible defaults provided):
    plugins:
      clip_vision:
        enabled: true
        model_id: openai/clip-vit-large-patch14
        device: auto          # auto|cpu|cuda|mps
        index_path: ~/.homie/clip_index.npz
        allow_download: true  # explicit opt-in for first-run weight pull
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from homie.config import HomieConfig, cfg_get
except Exception:  # pragma: no cover - allow standalone import for tests
    HomieConfig = Any  # type: ignore[assignment,misc]

    def cfg_get(cfg: Any, *keys: str, default: Any = None) -> Any:  # type: ignore[no-redef]
        cur: Any = getattr(cfg, "raw", cfg)
        for key in keys:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur


logger = logging.getLogger("homie.plugins.clip_vision")

DEFAULT_MODEL_ID = "openai/clip-vit-large-patch14"
DEFAULT_INDEX_PATH = Path.home() / ".homie" / "clip_index.npz"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff"}


@dataclass
class ClipConfig:
    """Resolved configuration for the CLIP plugin."""

    model_id: str = DEFAULT_MODEL_ID
    device: str = "auto"
    index_path: Path = DEFAULT_INDEX_PATH
    allow_download: bool = True
    enabled: bool = True

    @classmethod
    def from_homie(cls, cfg: "HomieConfig") -> "ClipConfig":
        index_raw = cfg_get(cfg, "plugins", "clip_vision", "index_path",
                            default=str(DEFAULT_INDEX_PATH))
        return cls(
            model_id=str(cfg_get(cfg, "plugins", "clip_vision", "model_id",
                                 default=DEFAULT_MODEL_ID)),
            device=str(cfg_get(cfg, "plugins", "clip_vision", "device",
                               default="auto")),
            index_path=Path(os.path.expanduser(str(index_raw))),
            allow_download=bool(cfg_get(cfg, "plugins", "clip_vision",
                                        "allow_download", default=True)),
            enabled=bool(cfg_get(cfg, "plugins", "clip_vision", "enabled",
                                 default=True)),
        )


@dataclass
class ImageIndex:
    """In-memory image embedding index, persisted to a single .npz file."""

    paths: List[str] = field(default_factory=list)
    embeddings: Any = None  # np.ndarray (n, dim), L2-normalized

    def __len__(self) -> int:
        return len(self.paths)


class ClipVisionPlugin:
    """Local CLIP-based vision plugin for Homie.

    The model is loaded lazily on first use so that simply importing or
    registering the plugin has no heavy side effects. All methods are
    synchronous and safe to call from Homie tool-handlers running on a
    background thread.
    """

    name = "clip_vision"
    version = "1.0.0"

    def __init__(self, config: ClipConfig) -> None:
        self.config = config
        self._model = None
        self._processor = None
        self._torch = None
        self._np = None
        self._resolved_device: Optional[str] = None
        self._index: ImageIndex = ImageIndex()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def activate(self) -> None:
        """Prepare the plugin for use. Loads any persisted image index."""
        if not self.config.enabled:
            logger.info("clip_vision plugin disabled via config")
            return
        logger.info("Activating clip_vision plugin (model=%s, device=%s)",
                    self.config.model_id, self.config.device)
        self._load_index_if_present()

    def deactivate(self) -> None:
        """Release model resources and clear large in-memory state."""
        logger.info("Deactivating clip_vision plugin")
        self._model = None
        self._processor = None
        self._index = ImageIndex()
        if self._torch is not None:
            try:
                if self._resolved_device == "cuda":
                    self._torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                logger.debug("cuda cache clear failed", exc_info=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def classify_image(
        self,
        image_path: str | Path,
        labels: Sequence[str],
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Zero-shot classify ``image_path`` against ``labels``.

        Returns a list of ``(label, probability)`` tuples sorted descending,
        truncated to ``top_k`` entries.
        """
        if not labels:
            raise ValueError("labels must be a non-empty sequence")
        self._ensure_loaded()
        image = self._open_image(Path(image_path))
        prompts = [f"a photo of {label}" for label in labels]
        probs = self._image_text_probs(image, prompts)
        scored = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)
        return [(label, float(p)) for label, p in scored[: max(1, top_k)]]

    def score_image_text(
        self,
        image_path: str | Path,
        text: str,
    ) -> float:
        """Return a cosine similarity score in ``[-1, 1]`` for an image/text pair."""
        self._ensure_loaded()
        image = self._open_image(Path(image_path))
        img_emb = self._encode_images([image])
        txt_emb = self._encode_texts([text])
        sim = float((img_emb @ txt_emb.T).reshape(-1)[0])
        return sim

    def index_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
        persist: bool = True,
    ) -> int:
        """Encode every supported image in ``directory`` and add to the index.

        Returns the number of newly indexed images.
        """
        self._ensure_loaded()
        np = self._np
        root = Path(directory).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Not a directory: {root}")

        candidates = self._iter_images(root, recursive=recursive)
        already = set(self._index.paths)
        fresh_paths: List[str] = []
        fresh_images: List[Any] = []
        for p in candidates:
            spath = str(p)
            if spath in already:
                continue
            try:
                fresh_images.append(self._open_image(p))
                fresh_paths.append(spath)
            except Exception:  # noqa: BLE001
                logger.warning("Skipping unreadable image: %s", spath, exc_info=True)

        if not fresh_paths:
            return 0

        new_emb = self._encode_images(fresh_images)
        if self._index.embeddings is None or len(self._index.paths) == 0:
            merged = new_emb
        else:
            merged = np.concatenate([self._index.embeddings, new_emb], axis=0)
        self._index = ImageIndex(paths=self._index.paths + fresh_paths,
                                 embeddings=merged)

        if persist:
            self._save_index()
        logger.info("Indexed %d new images (total=%d)", len(fresh_paths), len(self._index))
        return len(fresh_paths)

    def search_images(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Semantic search the local image index using a natural-language ``query``."""
        if len(self._index) == 0:
            return []
        self._ensure_loaded()
        np = self._np
        txt_emb = self._encode_texts([query])
        sims = (self._index.embeddings @ txt_emb.T).reshape(-1)
        order = np.argsort(-sims)[: max(1, top_k)]
        return [(self._index.paths[int(i)], float(sims[int(i)])) for i in order]

    def clear_index(self) -> None:
        """Drop the in-memory index and remove the persisted file if any."""
        self._index = ImageIndex()
        try:
            if self.config.index_path.exists():
                self.config.index_path.unlink()
        except OSError:
            logger.debug("Failed to remove index file", exc_info=True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not self.config.allow_download:
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

        try:
            import numpy as np  # type: ignore
            import torch  # type: ignore
            from PIL import Image  # type: ignore  # noqa: F401
            from transformers import CLIPModel, CLIPProcessor  # type: ignore
        except ImportError as exc:  # pragma: no cover - dep guidance
            raise RuntimeError(
                "clip_vision requires `torch`, `transformers`, `pillow`, and "
                "`numpy`. Install with: pip install torch transformers pillow numpy"
            ) from exc

        self._np = np
        self._torch = torch
        self._resolved_device = self._resolve_device(torch)
        logger.info("Loading CLIP model %s on %s", self.config.model_id,
                    self._resolved_device)
        self._processor = CLIPProcessor.from_pretrained(self.config.model_id)
        self._model = CLIPModel.from_pretrained(self.config.model_id).to(
            self._resolved_device
        )
        self._model.eval()

    def _resolve_device(self, torch: Any) -> str:
        choice = (self.config.device or "auto").lower()
        if choice == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return choice

    def _open_image(self, path: Path) -> Any:
        from PIL import Image  # type: ignore

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def _encode_images(self, images: Sequence[Any]) -> Any:
        torch = self._torch
        assert self._processor is not None and self._model is not None
        inputs = self._processor(images=list(images), return_tensors="pt")
        inputs = {k: v.to(self._resolved_device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = self._model.get_image_features(**inputs)
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
        return feats.cpu().numpy()

    def _encode_texts(self, texts: Sequence[str]) -> Any:
        torch = self._torch
        assert self._processor is not None and self._model is not None
        inputs = self._processor(text=list(texts), return_tensors="pt",
                                 padding=True, truncation=True)
        inputs = {k: v.to(self._resolved_device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = self._model.get_text_features(**inputs)
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
        return feats.cpu().numpy()

    def _image_text_probs(self, image: Any, prompts: Sequence[str]) -> List[float]:
        torch = self._torch
        assert self._processor is not None and self._model is not None
        inputs = self._processor(text=list(prompts), images=image,
                                 return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self._resolved_device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits_per_image = outputs.logits_per_image  # (1, n_prompts)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy().reshape(-1)
        return [float(p) for p in probs]

    def _iter_images(self, root: Path, recursive: bool) -> Iterable[Path]:
        if recursive:
            for p in root.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    yield p
        else:
            for p in root.iterdir():
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    yield p

    def _load_index_if_present(self) -> None:
        path = self.config.index_path
        if not path.exists():
            return
        try:
            import numpy as np  # type: ignore
        except ImportError:
            logger.warning("numpy not installed; cannot load CLIP index")
            return
        try:
            with np.load(path, allow_pickle=False) as data:
                embeddings = data["embeddings"]
                paths = [str(p) for p in data["paths"].tolist()]
            self._np = np
            self._index = ImageIndex(paths=paths, embeddings=embeddings)
            logger.info("Loaded CLIP image index with %d entries", len(self._index))
        except Exception:  # noqa: BLE001
            logger.warning("Failed to load CLIP index at %s", path, exc_info=True)

    def _save_index(self) -> None:
        np = self._np
        if np is None or self._index.embeddings is None:
            return
        path = self.config.index_path
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path,
                 paths=np.array(self._index.paths, dtype=object),
                 embeddings=self._index.embeddings)


# ----------------------------------------------------------------------
# Module-level Homie plugin hooks
# ----------------------------------------------------------------------
_PLUGIN: Optional[ClipVisionPlugin] = None


def _build_plugin(cfg: Optional["HomieConfig"]) -> ClipVisionPlugin:
    if cfg is None:
        return ClipVisionPlugin(ClipConfig())
    return ClipVisionPlugin(ClipConfig.from_homie(cfg))


def activate(cfg: Optional["HomieConfig"] = None) -> ClipVisionPlugin:
    """Activate the plugin and return the live instance."""
    global _PLUGIN
    if _PLUGIN is None:
        _PLUGIN = _build_plugin(cfg)
    _PLUGIN.activate()
    return _PLUGIN


def deactivate() -> None:
    """Deactivate the plugin if it is currently active."""
    global _PLUGIN
    if _PLUGIN is not None:
        _PLUGIN.deactivate()
        _PLUGIN = None


def register(host: Any) -> Dict[str, Any]:
    """Register the plugin with a Homie plugin host.

    ``host`` is expected to expose at least ``register_plugin(metadata)``
    and provide a ``HomieConfig`` via either ``host.config`` or by being
    passed alongside as kwargs. The exact host shape is intentionally
    loose so this plugin works with any of the current Homie loaders.
    """
    cfg = getattr(host, "config", None)
    plugin = _build_plugin(cfg)
    metadata = {
        "name": plugin.name,
        "version": plugin.version,
        "description": "Local CLIP zero-shot classification and image search",
        "capabilities": [
            "classify_image",
            "score_image_text",
            "index_directory",
            "search_images",
        ],
        "activate": plugin.activate,
        "deactivate": plugin.deactivate,
        "instance": plugin,
    }
    if hasattr(host, "register_plugin"):
        host.register_plugin(metadata)
    return metadata


__all__ = [
    "ClipConfig",
    "ClipVisionPlugin",
    "ImageIndex",
    "activate",
    "deactivate",
    "register",
]
