"""CLAP audio classification plugin for HOMIE.

Wraps the ``laion/clap-htsat-fused`` checkpoint (Contrastive Language-Audio
Pretraining) to provide local, zero-shot audio classification. Useful for:

  * Ambient sound detection (alarm, door, glass break, dog bark, ...).
  * Voice-command intent matching without an LLM round-trip.
  * Audio-event triggers that feed back into the orchestrator.

The model is downloaded from the Hugging Face Hub on first use only when the
user opts in via ``audio.clap.allow_download`` in ``homie.config.yaml`` or the
``HOMIE_ALLOW_HF_DOWNLOAD`` env var. Otherwise transformers is forced into
offline mode and only the local HF cache is used — fully local inference,
no network calls.

Heavy dependencies (``torch``, ``transformers``, ``soundfile``) are imported
lazily so that simply importing this module is cheap and side-effect free.

Plugin contract
---------------
* ``activate(cfg)``  — called by the HOMIE plugin loader on startup.
* ``deactivate()``   — releases the model from memory.
* ``register()``     — returns metadata for registry-style plugin hosts.
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

PLUGIN_NAME = "clap_audio"
PLUGIN_VERSION = "1.0.0"
DEFAULT_MODEL_ID = "laion/clap-htsat-fused"
DEFAULT_SAMPLE_RATE = 48_000  # CLAP expects 48 kHz mono
DEFAULT_LABELS: Tuple[str, ...] = (
    "speech",
    "music",
    "alarm",
    "doorbell",
    "knocking",
    "glass breaking",
    "dog barking",
    "baby crying",
    "phone ringing",
    "silence",
    "background noise",
)

AudioInput = Union[str, Path, "object"]  # str/Path or a numpy ndarray


class ClapAudioError(RuntimeError):
    """Raised for any unrecoverable error in this plugin."""


@dataclass
class ClassificationResult:
    """A single label + score returned by :meth:`ClapAudioPlugin.classify`."""

    label: str
    score: float


@dataclass
class ClapAudioConfig:
    """Configuration for the CLAP plugin.

    Values are read from ``HomieConfig`` under the ``audio.clap`` key when a
    config object is supplied; otherwise defaults are used so the plugin
    works out of the box.
    """

    model_id: str = DEFAULT_MODEL_ID
    cache_dir: Optional[str] = None
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"
    allow_download: bool = False
    default_labels: Tuple[str, ...] = field(default_factory=lambda: DEFAULT_LABELS)
    top_k: int = 3
    sample_rate: int = DEFAULT_SAMPLE_RATE

    @classmethod
    def from_homie_config(cls, cfg: Any) -> "ClapAudioConfig":
        """Build a :class:`ClapAudioConfig` from a :class:`HomieConfig`."""
        if cfg is None:
            return cls()
        try:
            from homie.config import cfg_get  # local import to avoid hard dep
        except Exception:  # noqa: BLE001
            return cls()

        labels = cfg_get(cfg, "audio", "clap", "labels", default=DEFAULT_LABELS)
        return cls(
            model_id=cfg_get(cfg, "audio", "clap", "model_id", default=DEFAULT_MODEL_ID),
            cache_dir=cfg_get(cfg, "audio", "clap", "cache_dir", default=None),
            device=cfg_get(cfg, "audio", "clap", "device", default="auto"),
            allow_download=bool(
                cfg_get(cfg, "audio", "clap", "allow_download", default=False)
            ),
            default_labels=tuple(labels) if labels else DEFAULT_LABELS,
            top_k=int(cfg_get(cfg, "audio", "clap", "top_k", default=3)),
            sample_rate=int(
                cfg_get(cfg, "audio", "clap", "sample_rate", default=DEFAULT_SAMPLE_RATE)
            ),
        )


class ClapAudioPlugin:
    """Zero-shot audio classifier backed by ``laion/clap-htsat-fused``.

    The model is loaded lazily on the first call to :meth:`classify`. Call
    :meth:`unload` (or the module-level :func:`deactivate`) to release it.
    Instances are thread-safe for read access; the underlying torch model is
    used in ``eval()`` mode with ``torch.no_grad()``.
    """

    def __init__(self, config: Optional[ClapAudioConfig] = None) -> None:
        self.config = config or ClapAudioConfig()
        self._model: Any = None
        self._processor: Any = None
        self._device: Optional[str] = None
        self._lock = threading.Lock()

    # ---- lifecycle -----------------------------------------------------

    def load(self) -> None:
        """Load model + processor. Idempotent and thread-safe."""
        with self._lock:
            if self._model is not None:
                return
            self._resolve_offline_mode()
            try:
                import torch
                from transformers import ClapModel, ClapProcessor
            except ImportError as exc:  # pragma: no cover - environment dep
                raise ClapAudioError(
                    "transformers and torch are required for the CLAP plugin. "
                    "Install with: pip install transformers torch soundfile"
                ) from exc

            cache_dir = self.config.cache_dir or os.getenv("HOMIE_HF_CACHE")
            logger.info(
                "Loading CLAP model %s (cache=%s, offline=%s)",
                self.config.model_id,
                cache_dir,
                os.environ.get("TRANSFORMERS_OFFLINE", "0"),
            )

            processor = ClapProcessor.from_pretrained(
                self.config.model_id, cache_dir=cache_dir
            )
            model = ClapModel.from_pretrained(
                self.config.model_id, cache_dir=cache_dir
            )
            device = self._pick_device(torch)
            model = model.to(device).eval()

            self._processor = processor
            self._model = model
            self._device = device
            logger.info("CLAP model ready on device=%s", device)

    def unload(self) -> None:
        """Release the model from memory."""
        with self._lock:
            self._model = None
            self._processor = None
            self._device = None
        try:
            import gc

            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def _resolve_offline_mode(self) -> None:
        """Force transformers offline unless downloads are opted in."""
        env_allow = os.getenv("HOMIE_ALLOW_HF_DOWNLOAD", "").lower() in {
            "1",
            "true",
            "yes",
        }
        if self.config.allow_download or env_allow:
            return
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    def _pick_device(self, torch_mod: Any) -> str:
        choice = (self.config.device or "auto").lower()
        if choice != "auto":
            return choice
        if torch_mod.cuda.is_available():
            return "cuda"
        mps = getattr(torch_mod.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    # ---- inference -----------------------------------------------------

    def classify(
        self,
        audio: AudioInput,
        labels: Optional[Sequence[str]] = None,
        *,
        top_k: Optional[int] = None,
    ) -> List[ClassificationResult]:
        """Run zero-shot classification on ``audio`` against ``labels``.

        Parameters
        ----------
        audio:
            Path to an audio file (any format ``soundfile`` reads) or a 1-D
            ``numpy`` array of mono samples.
        labels:
            Candidate text labels. Defaults to ``config.default_labels``.
        top_k:
            How many top-scoring labels to return. Defaults to
            ``config.top_k``.
        """
        if self._model is None:
            self.load()

        candidate_labels = list(labels) if labels else list(self.config.default_labels)
        if not candidate_labels:
            raise ClapAudioError("At least one label is required for classification")

        k = top_k if top_k is not None else self.config.top_k
        k = max(1, min(k, len(candidate_labels)))

        waveform = _load_waveform(audio, target_sr=self.config.sample_rate)

        import torch

        prompts = [f"this is the sound of {label}" for label in candidate_labels]
        inputs = self._processor(
            text=prompts,
            audios=waveform,
            return_tensors="pt",
            sampling_rate=self.config.sample_rate,
            padding=True,
        )
        inputs = {key: val.to(self._device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits_per_audio  # shape: (1, num_labels)
            probs = logits.softmax(dim=-1).squeeze(0).cpu().tolist()

        scored = sorted(
            (
                ClassificationResult(label=label, score=float(score))
                for label, score in zip(candidate_labels, probs)
            ),
            key=lambda r: r.score,
            reverse=True,
        )
        return scored[:k]

    def detect(
        self,
        audio: AudioInput,
        target_label: str,
        *,
        threshold: float = 0.5,
        contrast_labels: Optional[Sequence[str]] = None,
    ) -> Tuple[bool, float]:
        """Detect a single target sound in ``audio``.

        ``contrast_labels`` are included so the softmax has somewhere to put
        probability mass when the target sound is absent. Defaults to all
        other labels in ``config.default_labels``.

        Returns ``(detected, score)`` where ``detected`` is
        ``score >= threshold``.
        """
        if contrast_labels is None:
            contrast = [
                label
                for label in self.config.default_labels
                if label != target_label
            ]
        else:
            contrast = [label for label in contrast_labels if label != target_label]

        if not contrast:
            contrast = [label for label in DEFAULT_LABELS if label != target_label]

        full_labels = [target_label, *contrast]
        results = self.classify(audio, labels=full_labels, top_k=len(full_labels))
        score = next((r.score for r in results if r.label == target_label), 0.0)
        return score >= threshold, score

    def embed_audio(self, audio: AudioInput) -> List[float]:
        """Return the CLAP audio embedding for ``audio``.

        Useful for downstream similarity search (e.g. matching against a
        gallery of saved sound fingerprints stored in ChromaDB).
        """
        if self._model is None:
            self.load()

        waveform = _load_waveform(audio, target_sr=self.config.sample_rate)

        import torch

        inputs = self._processor(
            audios=waveform,
            return_tensors="pt",
            sampling_rate=self.config.sample_rate,
        )
        inputs = {key: val.to(self._device) for key, val in inputs.items()}
        with torch.no_grad():
            features = self._model.get_audio_features(**inputs)
        return features.squeeze(0).cpu().tolist()


# --- helpers --------------------------------------------------------------


def _load_waveform(audio: AudioInput, *, target_sr: int):
    """Load ``audio`` into a 1-D float32 mono array at ``target_sr`` Hz."""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment dep
        raise ClapAudioError("numpy is required for audio I/O") from exc

    if hasattr(audio, "dtype") and hasattr(audio, "shape"):  # numpy-like
        wave = np.asarray(audio, dtype=np.float32)
        if wave.ndim > 1:
            wave = wave.mean(axis=tuple(range(1, wave.ndim)))
        return wave

    path = Path(audio)  # type: ignore[arg-type]
    if not path.exists():
        raise ClapAudioError(f"Audio file not found: {path}")

    try:
        import soundfile as sf
    except ImportError as exc:
        raise ClapAudioError(
            "soundfile is required to read audio files (pip install soundfile)"
        ) from exc

    wave, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if sr != target_sr:
        wave = _resample(wave, sr, target_sr)
    return wave


def _resample(wave, src_sr: int, dst_sr: int):
    """Resample ``wave`` from ``src_sr`` to ``dst_sr`` Hz.

    Uses ``librosa`` when available, otherwise falls back to linear
    interpolation (lower fidelity, but zero extra dependency).
    """
    if src_sr == dst_sr:
        return wave
    try:
        import librosa

        return librosa.resample(wave, orig_sr=src_sr, target_sr=dst_sr)
    except ImportError:
        pass

    import numpy as np

    duration = wave.shape[0] / float(src_sr)
    new_len = int(round(duration * dst_sr))
    if new_len <= 1:
        return wave.astype(np.float32)
    x_old = np.linspace(0.0, 1.0, num=wave.shape[0], endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
    return np.interp(x_new, x_old, wave).astype(np.float32)


# --- module-level plugin contract ----------------------------------------

_PLUGIN: Optional[ClapAudioPlugin] = None


def activate(cfg: Any = None) -> ClapAudioPlugin:
    """Instantiate the plugin and return the singleton instance.

    Called by HOMIE's plugin loader on startup. The model itself is not
    loaded here — :meth:`ClapAudioPlugin.load` is invoked lazily on the
    first :meth:`classify` / :meth:`embed_audio` call.
    """
    global _PLUGIN
    if _PLUGIN is None:
        config = ClapAudioConfig.from_homie_config(cfg)
        _PLUGIN = ClapAudioPlugin(config=config)
        logger.info(
            "Activated %s v%s (model=%s, device=%s)",
            PLUGIN_NAME,
            PLUGIN_VERSION,
            config.model_id,
            config.device,
        )
    return _PLUGIN


def deactivate() -> None:
    """Release the loaded model (if any) and drop the singleton."""
    global _PLUGIN
    if _PLUGIN is not None:
        _PLUGIN.unload()
        _PLUGIN = None
        logger.info("Deactivated %s", PLUGIN_NAME)


def get_plugin() -> Optional[ClapAudioPlugin]:
    """Return the active plugin instance, or ``None`` if not activated."""
    return _PLUGIN


def register() -> Dict[str, Any]:
    """Return plugin metadata for registry-style plugin hosts."""
    return {
        "name": PLUGIN_NAME,
        "version": PLUGIN_VERSION,
        "description": (
            "Zero-shot audio classification with laion/clap-htsat-fused "
            "(ambient sound detection, voice-command intent matching, "
            "audio embeddings)."
        ),
        "capabilities": ("audio.classify", "audio.detect", "audio.embed"),
        "model_id": DEFAULT_MODEL_ID,
        "activate": activate,
        "deactivate": deactivate,
    }


__all__ = [
    "PLUGIN_NAME",
    "PLUGIN_VERSION",
    "DEFAULT_MODEL_ID",
    "DEFAULT_LABELS",
    "ClapAudioConfig",
    "ClapAudioPlugin",
    "ClassificationResult",
    "ClapAudioError",
    "activate",
    "deactivate",
    "get_plugin",
    "register",
]
