"""Homie CLAP Audio Embeddings Plugin.

Provides local audio embedding, similarity search, and zero-shot
classification for voice memos and other audio snippets using LAION's
CLAP (Contrastive Language-Audio Pre-training) model
``laion/clap-htsat-fused``.

This plugin is local-first:
- Inference runs entirely on-device via ``transformers`` + ``torch``.
- Embeddings are persisted to a small SQLite index under
  ``~/.homie/plugins`` so they can be reused by RAG / search workflows
  without re-encoding.
- Network access happens only on the first call (model download from
  Hugging Face). Subsequent runs are fully offline. The opt-in flag
  ``allow_model_download`` must be set in plugin config to enable that
  initial fetch; otherwise the model must already be cached locally.

Public surface:
- ``register()``       returns plugin metadata + lifecycle hooks
- ``activate(ctx)``    builds the singleton and opens the index
- ``deactivate(ctx)``  releases model handles and closes the index
- ``ClapAudioPlugin``  class encapsulating the model + index
"""
from __future__ import annotations

import logging
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

PLUGIN_NAME = "clap_audio_embeddings"
PLUGIN_VERSION = "0.1.0"
MODEL_ID = "laion/clap-htsat-fused"
DEFAULT_SAMPLE_RATE = 48_000
DEFAULT_INDEX_PATH = Path.home() / ".homie" / "plugins" / "clap_audio.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS clap_audio (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL,
    tag TEXT,
    dim INTEGER NOT NULL,
    embedding BLOB NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_clap_tag ON clap_audio(tag);
"""


@dataclass
class ClapPluginConfig:
    """Runtime configuration for the CLAP plugin."""

    model_id: str = MODEL_ID
    sample_rate: int = DEFAULT_SAMPLE_RATE
    index_path: Path = DEFAULT_INDEX_PATH
    device: str = "cpu"
    allow_model_download: bool = False
    cache_dir: Optional[Path] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ClapPluginConfig":
        if not data:
            return cls()
        merged: Dict[str, Any] = {}
        for key in ("model_id", "sample_rate", "device", "allow_model_download"):
            if key in data:
                merged[key] = data[key]
        if "index_path" in data and data["index_path"]:
            merged["index_path"] = Path(data["index_path"]).expanduser()
        if "cache_dir" in data and data["cache_dir"]:
            merged["cache_dir"] = Path(data["cache_dir"]).expanduser()
        return cls(**merged)


def _encode_vector(vec: Sequence[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _decode_vector(blob: bytes, dim: int) -> List[float]:
    return list(struct.unpack(f"{dim}f", blob))


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    # Pure-Python cosine keeps the search hot path numpy-optional.
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


class ClapAudioPlugin:
    """CLAP-based audio embedding, search, and zero-shot classification.

    The model and processor are loaded lazily on first use so that simply
    activating the plugin does not pay the load cost or trigger a download.
    """

    def __init__(self, config: Optional[ClapPluginConfig] = None) -> None:
        self.config = config or ClapPluginConfig()
        self._model: Any = None
        self._processor: Any = None
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------- lifecycle
    def activate(self) -> None:
        """Open the embedding index. Model load is deferred to first use."""
        self.config.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.config.index_path))
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        logger.info(
            "[%s] activated (index=%s, model=%s, device=%s)",
            PLUGIN_NAME,
            self.config.index_path,
            self.config.model_id,
            self.config.device,
        )

    def deactivate(self) -> None:
        """Release model handles and close the index connection."""
        self._model = None
        self._processor = None
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None
        logger.info("[%s] deactivated", PLUGIN_NAME)

    # ------------------------------------------------------------ model load
    def _ensure_model(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        try:
            from transformers import ClapModel, ClapProcessor  # type: ignore
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "transformers is required for the CLAP plugin. "
                "Install with: pip install transformers torch soundfile librosa"
            ) from exc

        load_kwargs: Dict[str, Any] = {}
        if self.config.cache_dir is not None:
            load_kwargs["cache_dir"] = str(self.config.cache_dir)
        if not self.config.allow_model_download:
            # Force offline mode: model must already be cached locally.
            load_kwargs["local_files_only"] = True

        logger.info("[%s] loading model %s", PLUGIN_NAME, self.config.model_id)
        self._processor = ClapProcessor.from_pretrained(self.config.model_id, **load_kwargs)
        self._model = ClapModel.from_pretrained(self.config.model_id, **load_kwargs)
        self._model.to(self.config.device)
        self._model.eval()

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _load_audio(path: Path, sample_rate: int) -> Any:
        try:
            import soundfile as sf  # type: ignore
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "soundfile is required to read audio. Install with: pip install soundfile"
            ) from exc

        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if hasattr(data, "ndim") and data.ndim == 2:
            # Downmix multi-channel audio to mono.
            data = data.mean(axis=1)
        if sr != sample_rate:
            try:
                import librosa  # type: ignore
            except ImportError as exc:  # pragma: no cover - import guard
                raise RuntimeError(
                    f"Audio at {path} has sample rate {sr}, expected {sample_rate}. "
                    "Install librosa to enable automatic resampling."
                ) from exc
            data = librosa.resample(data, orig_sr=sr, target_sr=sample_rate)
        return data

    # ------------------------------------------------------------- embedding
    def embed_audio(self, path: Union[str, Path]) -> List[float]:
        """Return a normalized audio embedding for the given file."""
        self._ensure_model()
        import torch  # type: ignore

        audio = self._load_audio(Path(path), self.config.sample_rate)
        inputs = self._processor(
            audios=audio,
            sampling_rate=self.config.sample_rate,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        with torch.no_grad():
            features = self._model.get_audio_features(**inputs)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.squeeze(0).cpu().tolist()

    def embed_text(self, text: str) -> List[float]:
        """Return a normalized text embedding aligned with the audio space."""
        self._ensure_model()
        import torch  # type: ignore

        inputs = self._processor(text=text, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        with torch.no_grad():
            features = self._model.get_text_features(**inputs)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.squeeze(0).cpu().tolist()

    # ------------------------------------------------------------- index ops
    def index_audio(self, path: Union[str, Path], tag: Optional[str] = None) -> int:
        """Embed an audio file and persist it in the local index."""
        if self._conn is None:
            raise RuntimeError(f"{PLUGIN_NAME} is not active; call activate() first")
        resolved = Path(path).expanduser().resolve()
        vec = self.embed_audio(resolved)
        blob = _encode_vector(vec)
        cur = self._conn.execute(
            """
            INSERT INTO clap_audio (path, tag, dim, embedding)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                tag = excluded.tag,
                dim = excluded.dim,
                embedding = excluded.embedding
            """,
            (str(resolved), tag, len(vec), blob),
        )
        self._conn.commit()
        return int(cur.lastrowid or 0)

    def search_text(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Return the top-k indexed audio files most similar to a text query."""
        if self._conn is None:
            raise RuntimeError(f"{PLUGIN_NAME} is not active; call activate() first")
        query_vec = self.embed_text(query)
        results: List[Tuple[str, float]] = []
        for row in self._conn.execute("SELECT path, dim, embedding FROM clap_audio"):
            row_path, dim, blob = row
            audio_vec = _decode_vector(blob, dim)
            results.append((row_path, _cosine(query_vec, audio_vec)))
        results.sort(key=lambda item: item[1], reverse=True)
        return results[:k]

    def classify(
        self, path: Union[str, Path], labels: Sequence[str]
    ) -> List[Tuple[str, float]]:
        """Zero-shot classify an audio file against a list of textual labels."""
        if not labels:
            return []
        audio_vec = self.embed_audio(path)
        scored: List[Tuple[str, float]] = []
        for label in labels:
            text_vec = self.embed_text(label)
            scored.append((label, _cosine(audio_vec, text_vec)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored


# --------------------------------------------------------------- module hooks
_INSTANCE: Optional[ClapAudioPlugin] = None


def register() -> Dict[str, Any]:
    """Return plugin metadata used by Homie's loader."""
    return {
        "name": PLUGIN_NAME,
        "version": PLUGIN_VERSION,
        "description": (
            "Local audio embeddings, semantic search, and zero-shot "
            "classification for voice memos using LAION CLAP."
        ),
        "capabilities": ["audio.embed", "audio.search", "audio.classify"],
        "activate": activate,
        "deactivate": deactivate,
        "factory": ClapAudioPlugin,
    }


def activate(ctx: Optional[Dict[str, Any]] = None) -> ClapAudioPlugin:
    """Create and activate the singleton plugin instance.

    ``ctx`` may contain a ``"config"`` dict for ``ClapPluginConfig.from_dict``.
    """
    global _INSTANCE
    cfg_data = (ctx or {}).get("config") if ctx else None
    config = ClapPluginConfig.from_dict(cfg_data)
    _INSTANCE = ClapAudioPlugin(config)
    _INSTANCE.activate()
    return _INSTANCE


def deactivate(ctx: Optional[Dict[str, Any]] = None) -> None:
    """Deactivate and release the singleton plugin instance."""
    global _INSTANCE
    if _INSTANCE is not None:
        _INSTANCE.deactivate()
        _INSTANCE = None


__all__ = [
    "PLUGIN_NAME",
    "PLUGIN_VERSION",
    "MODEL_ID",
    "ClapPluginConfig",
    "ClapAudioPlugin",
    "register",
    "activate",
    "deactivate",
]
