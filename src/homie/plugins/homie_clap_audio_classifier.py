"""CLAP audio/environment sound classifier plugin for HOMIE.

This plugin adds *local* zero-shot audio classification to HOMIE using the
``laion/clap-htsat-fused`` CLAP model (https://huggingface.co/laion/clap-htsat-fused).
CLAP embeds audio and free-text labels into a shared space, so HOMIE can ask
"does this sound like a doorbell / dog barking / running tap?" without training a
classifier per label. This pairs naturally with HOMIE's existing voice stack
(Whisper STT / Piper TTS): Whisper handles *speech*, CLAP handles *non-speech*
environment sound and acoustic-scene awareness.

Design goals (matching HOMIE conventions):
- Single file, lazy heavy imports (torch/transformers loaded only on ``load()``).
- LOCAL-FIRST and offline by default. The model is only fetched from the network
  when the caller explicitly opts in via ``allow_download=True``; otherwise the
  plugin runs in ``HF_HUB_OFFLINE`` mode and uses the local HF cache.
- ``activate()`` / ``deactivate()`` lifecycle plus a ``register()`` helper so a
  plugin host can wire it into a tool registry.
- Type hints and docstrings throughout.

Optional runtime dependencies (installed by the host, not by this plugin):
    transformers, torch, soundfile, numpy
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "laion/clap-htsat-fused"
CLAP_SAMPLE_RATE = 48_000  # CLAP-HTSAT expects 48 kHz mono audio

# Sensible default candidate labels for home/environment sound classification.
DEFAULT_LABELS: List[str] = [
    "speech",
    "silence",
    "music",
    "dog barking",
    "doorbell",
    "knock on door",
    "running water or tap",
    "glass breaking",
    "smoke alarm or beep",
    "baby crying",
    "keyboard typing",
    "vehicle or traffic",
]


@dataclass
class ClapClassifierConfig:
    """Configuration for :class:`ClapAudioClassifier`.

    Attributes:
        model_id: HuggingFace repo id or a local path to a downloaded snapshot.
        labels: Candidate free-text labels scored against each audio clip.
        top_k: Number of highest-scoring labels returned by ``classify``.
        device: Torch device string ("cpu" or e.g. "cuda"). CPU by default to
            stay portable across desktop/mobile.
        allow_download: If False (default) the plugin runs fully offline and will
            error if the model is not already in the local cache. Set True only
            when the user has explicitly consented to a network fetch.
        cache_dir: Optional override for the HuggingFace cache location.
    """

    model_id: str = DEFAULT_MODEL_ID
    labels: List[str] = field(default_factory=lambda: list(DEFAULT_LABELS))
    top_k: int = 3
    device: str = "cpu"
    allow_download: bool = False
    cache_dir: Optional[str] = None


@dataclass
class LabelScore:
    """A single ``(label, probability)`` result."""

    label: str
    score: float

    def as_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "score": round(float(self.score), 6)}


class ClapAudioClassifier:
    """Local zero-shot audio classifier backed by CLAP.

    The model and processor are loaded lazily and guarded by a lock so the
    plugin is cheap to import and safe to call from HOMIE's scheduler threads.
    """

    def __init__(self, config: Optional[ClapClassifierConfig] = None) -> None:
        self.config = config or ClapClassifierConfig()
        self._model: Any = None
        self._processor: Any = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    @property
    def is_loaded(self) -> bool:
        """True once the model and processor are resident in memory."""
        return self._model is not None and self._processor is not None

    def load(self) -> None:
        """Load the CLAP model and processor into memory (idempotent).

        Respects ``config.allow_download``: when False, ``HF_HUB_OFFLINE`` is set
        for the duration of the load so transformers never reaches the network
        and instead resolves the model from the local cache.

        Raises:
            RuntimeError: if optional dependencies are missing, or if the model
                is unavailable locally while running offline.
        """
        if self.is_loaded:
            return
        with self._lock:
            if self.is_loaded:
                return
            try:
                import torch  # noqa: F401  (imported for side effect / device)
                from transformers import ClapModel, ClapProcessor
            except ImportError as exc:  # pragma: no cover - env dependent
                raise RuntimeError(
                    "ClapAudioClassifier requires 'transformers' and 'torch'. "
                    "Install them in the HOMIE environment to enable audio "
                    "classification."
                ) from exc

            prev_offline = os.environ.get("HF_HUB_OFFLINE")
            if not self.config.allow_download:
                os.environ["HF_HUB_OFFLINE"] = "1"
            try:
                logger.info(
                    "Loading CLAP model '%s' (offline=%s, device=%s)",
                    self.config.model_id,
                    not self.config.allow_download,
                    self.config.device,
                )
                self._processor = ClapProcessor.from_pretrained(
                    self.config.model_id, cache_dir=self.config.cache_dir
                )
                self._model = ClapModel.from_pretrained(
                    self.config.model_id, cache_dir=self.config.cache_dir
                ).to(self.config.device)
                self._model.eval()
            except Exception as exc:  # noqa: BLE001 - surface a clean error
                self._model = None
                self._processor = None
                raise RuntimeError(
                    f"Failed to load CLAP model '{self.config.model_id}'. "
                    "If running offline, pre-download it once with "
                    "allow_download=True."
                ) from exc
            finally:
                # Restore the prior offline flag so we don't leak global state.
                if not self.config.allow_download:
                    if prev_offline is None:
                        os.environ.pop("HF_HUB_OFFLINE", None)
                    else:
                        os.environ["HF_HUB_OFFLINE"] = prev_offline

    def unload(self) -> None:
        """Release model/processor references so memory can be reclaimed."""
        with self._lock:
            self._model = None
            self._processor = None
        logger.info("CLAP model unloaded")

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def classify(
        self,
        audio: Sequence[float],
        labels: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
    ) -> List[LabelScore]:
        """Classify a mono waveform against candidate text labels.

        Args:
            audio: Mono PCM samples as floats, expected at 48 kHz (CLAP's rate).
            labels: Candidate labels; defaults to ``config.labels``.
            top_k: Number of results to return; defaults to ``config.top_k``.

        Returns:
            Label/score pairs sorted by descending probability.
        """
        if not self.is_loaded:
            self.load()

        import torch

        candidate_labels = list(labels or self.config.labels)
        if not candidate_labels:
            raise ValueError("At least one candidate label is required.")
        k = top_k or self.config.top_k

        inputs = self._processor(
            text=candidate_labels,
            audios=[list(audio)],
            sampling_rate=CLAP_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        ).to(self.config.device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            # logits_per_audio: (n_audio, n_labels) similarity scores.
            probs = outputs.logits_per_audio.softmax(dim=-1)[0]

        scored = [
            LabelScore(label=lbl, score=float(probs[i]))
            for i, lbl in enumerate(candidate_labels)
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[: max(1, k)]

    def classify_file(
        self,
        path: str | Path,
        labels: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
    ) -> List[LabelScore]:
        """Classify a local audio file (wav/flac/ogg, decoded via soundfile).

        The file is loaded from disk only -- no network access. Audio is
        downmixed to mono and resampled to 48 kHz if needed.
        """
        waveform = _load_audio_mono(Path(path), target_sr=CLAP_SAMPLE_RATE)
        return self.classify(waveform, labels=labels, top_k=top_k)


def _load_audio_mono(path: Path, target_sr: int) -> List[float]:
    """Decode an audio file to a mono float list at ``target_sr`` (local only)."""
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - env dependent
        raise RuntimeError(
            "classify_file requires 'soundfile' and 'numpy'."
        ) from exc

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)  # downmix channels
    if sr != target_sr:
        # Lightweight linear resample to avoid a hard scipy/librosa dependency.
        duration = mono.shape[0] / float(sr)
        new_len = max(1, int(round(duration * target_sr)))
        old_idx = np.linspace(0.0, 1.0, num=mono.shape[0], endpoint=False)
        new_idx = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
        mono = np.interp(new_idx, old_idx, mono).astype("float32")
    return mono.tolist()


# ---------------------------------------------------------------------- #
# Plugin entry points
# ---------------------------------------------------------------------- #
_PLUGIN: Optional[ClapAudioClassifier] = None


def activate(
    config: Optional[ClapClassifierConfig] = None,
    eager: bool = False,
) -> ClapAudioClassifier:
    """Instantiate and return the singleton classifier plugin.

    Args:
        config: Optional configuration. Defaults to offline-safe settings.
        eager: If True, load the model now; otherwise it loads on first use.

    Returns:
        The active :class:`ClapAudioClassifier` instance.
    """
    global _PLUGIN
    if _PLUGIN is None:
        _PLUGIN = ClapAudioClassifier(config=config)
        logger.info("CLAP audio classifier plugin activated")
    if eager:
        _PLUGIN.load()
    return _PLUGIN


def deactivate() -> None:
    """Unload the model and drop the singleton instance."""
    global _PLUGIN
    if _PLUGIN is not None:
        _PLUGIN.unload()
        _PLUGIN = None
        logger.info("CLAP audio classifier plugin deactivated")


def register(
    registry: Any,
    config: Optional[ClapClassifierConfig] = None,
) -> ClapAudioClassifier:
    """Register the classifier as a callable tool with a HOMIE tool registry.

    The host registry is duck-typed: any object exposing ``register(name, fn,
    description=...)`` or a ``__setitem__`` mapping interface is supported. This
    keeps the plugin decoupled from a specific registry implementation.
    """
    plugin = activate(config=config)

    def _tool_classify_file(path: str, labels: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Classify a local audio file and return label/score dicts."""
        return [s.as_dict() for s in plugin.classify_file(path, labels=labels)]

    name = "clap_classify_audio"
    description = "Classify local environment/audio clips into labels using CLAP (offline)."
    register_fn: Optional[Callable] = getattr(registry, "register", None)
    if callable(register_fn):
        register_fn(name, _tool_classify_file, description=description)
    else:  # mapping-style registry fallback
        try:
            registry[name] = _tool_classify_file
        except TypeError as exc:  # noqa: BLE001
            raise TypeError(
                "registry must support .register(name, fn, description=...) or "
                "item assignment"
            ) from exc
    logger.info("Registered HOMIE tool '%s'", name)
    return plugin


__all__ = [
    "ClapClassifierConfig",
    "LabelScore",
    "ClapAudioClassifier",
    "activate",
    "deactivate",
    "register",
    "DEFAULT_MODEL_ID",
    "DEFAULT_LABELS",
]
