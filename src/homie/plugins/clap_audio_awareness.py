"""CLAP-based ambient audio awareness plugin for HOMIE.

This plugin integrates ``laion/clap-htsat-fused`` (Contrastive Language-Audio
Pretraining) to give HOMIE the ability to listen to short audio clips and
classify what it hears via natural-language labels (zero-shot).

Typical uses inside HOMIE:
    * Ambient sound monitoring (smoke alarms, doorbells, glass break,
      barking, knocking, baby crying, etc.) so the assistant can react.
    * Sound event detection feeding into automations (e.g. "if doorbell
      detected, run greet_visitor workflow").
    * Tagging recorded clips with semantic labels for later RAG retrieval
      via HOMIE's ChromaDB store.

The plugin is fully local: model weights are cached on first use and all
inference runs on-device. The only network call is the optional first-time
model download from Hugging Face, gated behind ``allow_model_download``.

Configuration (homie.config.yaml)::

    plugins:
      clap_audio_awareness:
        enabled: true
        model_id: laion/clap-htsat-fused
        device: auto            # auto|cpu|cuda
        sample_rate: 48000
        allow_model_download: false
        confidence_threshold: 0.35
        default_labels:
          - "smoke alarm beeping"
          - "doorbell ringing"
          - "glass breaking"
          - "dog barking"
          - "baby crying"
          - "knocking on a door"
          - "silence"
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from homie.config import HomieConfig


LOGGER = logging.getLogger(__name__)

PLUGIN_NAME = "clap_audio_awareness"
DEFAULT_MODEL_ID = "laion/clap-htsat-fused"
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_LABELS: Tuple[str, ...] = (
    "smoke alarm beeping",
    "doorbell ringing",
    "glass breaking",
    "dog barking",
    "baby crying",
    "knocking on a door",
    "silence",
)


@dataclass
class CLAPSettings:
    """Resolved configuration for the CLAP plugin."""

    enabled: bool = True
    model_id: str = DEFAULT_MODEL_ID
    device: str = "auto"
    sample_rate: int = DEFAULT_SAMPLE_RATE
    allow_model_download: bool = False
    default_labels: Tuple[str, ...] = DEFAULT_LABELS
    confidence_threshold: float = 0.35


@dataclass
class AudioClassification:
    """A single label/score pair."""

    label: str
    score: float


@dataclass
class CLAPDetection:
    """Result of running CLAP against a single audio clip."""

    top: AudioClassification
    scores: List[AudioClassification] = field(default_factory=list)
    triggered: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "top_label": self.top.label,
            "top_score": self.top.score,
            "triggered": self.triggered,
            "scores": [{"label": s.label, "score": s.score} for s in self.scores],
        }


def _resolve_settings(cfg: Optional["HomieConfig"]) -> CLAPSettings:
    """Build :class:`CLAPSettings` from a HomieConfig with sensible defaults."""
    if cfg is None:
        return CLAPSettings()

    from homie.config import cfg_get

    plugin_cfg = cfg_get(cfg, "plugins", PLUGIN_NAME, default={}) or {}
    labels = plugin_cfg.get("default_labels") or list(DEFAULT_LABELS)

    return CLAPSettings(
        enabled=bool(plugin_cfg.get("enabled", True)),
        model_id=str(plugin_cfg.get("model_id", DEFAULT_MODEL_ID)),
        device=str(plugin_cfg.get("device", "auto")),
        sample_rate=int(plugin_cfg.get("sample_rate", DEFAULT_SAMPLE_RATE)),
        allow_model_download=bool(plugin_cfg.get("allow_model_download", False)),
        default_labels=tuple(labels),
        confidence_threshold=float(plugin_cfg.get("confidence_threshold", 0.35)),
    )


class CLAPAudioAwarenessPlugin:
    """Ambient audio classification powered by ``laion/clap-htsat-fused``.

    The plugin is loaded lazily — model weights are only materialised the
    first time :meth:`classify` is called, keeping HOMIE startup fast.
    """

    def __init__(self, settings: Optional[CLAPSettings] = None) -> None:
        self.settings: CLAPSettings = settings or CLAPSettings()
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._active: bool = False

    @property
    def name(self) -> str:
        return PLUGIN_NAME

    @property
    def is_active(self) -> bool:
        return self._active

    def activate(self) -> None:
        """Mark the plugin as active. Does not download or load the model."""
        if self._active:
            return
        if not self.settings.enabled:
            LOGGER.info("%s disabled by config; skipping activation", PLUGIN_NAME)
            return
        LOGGER.info(
            "%s activated (model=%s, device=%s)",
            PLUGIN_NAME,
            self.settings.model_id,
            self.settings.device,
        )
        self._active = True

    def deactivate(self) -> None:
        """Release model references so memory can be reclaimed."""
        if not self._active:
            return
        self._model = None
        self._processor = None
        self._torch = None
        self._active = False
        LOGGER.info("%s deactivated", PLUGIN_NAME)

    def _resolve_device(self) -> str:
        if self.settings.device != "auto":
            return self.settings.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _ensure_loaded(self) -> None:
        """Lazy-load the CLAP model + processor on first use."""
        if self._model is not None and self._processor is not None:
            return

        try:
            import torch
            from transformers import ClapModel, ClapProcessor
        except ImportError as exc:
            raise RuntimeError(
                "CLAP plugin requires 'torch' and 'transformers'. "
                "Install with: pip install torch transformers"
            ) from exc

        local_only = not self.settings.allow_model_download
        device = self._resolve_device()
        LOGGER.info(
            "Loading CLAP model %s (device=%s, local_only=%s)",
            self.settings.model_id,
            device,
            local_only,
        )

        self._processor = ClapProcessor.from_pretrained(
            self.settings.model_id,
            local_files_only=local_only,
        )
        self._model = ClapModel.from_pretrained(
            self.settings.model_id,
            local_files_only=local_only,
        ).to(device)
        self._model.eval()
        self._torch = torch

    def classify(
        self,
        audio: Any,
        labels: Optional[Sequence[str]] = None,
    ) -> CLAPDetection:
        """Zero-shot classify an audio clip against text labels.

        Args:
            audio: 1-D ``numpy.ndarray`` of float32 samples at
                ``settings.sample_rate`` Hz, or a path to a wav/flac file.
            labels: Candidate text labels. Defaults to
                ``settings.default_labels``.

        Returns:
            :class:`CLAPDetection` with the top label and per-label scores.
        """
        if not self._active:
            raise RuntimeError(f"{PLUGIN_NAME} is not active. Call activate() first.")

        self._ensure_loaded()
        torch = self._torch
        candidate_labels: List[str] = (
            list(labels) if labels else list(self.settings.default_labels)
        )
        if not candidate_labels:
            raise ValueError("At least one candidate label is required.")

        waveform = self._load_audio(audio)

        inputs = self._processor(
            text=candidate_labels,
            audios=waveform,
            sampling_rate=self.settings.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits_per_audio  # (1, num_labels)
            probs = logits.softmax(dim=-1).squeeze(0).cpu().tolist()

        scored = [
            AudioClassification(label=lbl, score=float(p))
            for lbl, p in zip(candidate_labels, probs)
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        top = scored[0]
        return CLAPDetection(
            top=top,
            scores=scored,
            triggered=top.score >= self.settings.confidence_threshold,
        )

    def _load_audio(self, audio: Any):
        """Coerce a path or array into a 1-D float32 mono waveform."""
        if isinstance(audio, (str, Path)):
            try:
                import soundfile as sf
            except ImportError as exc:
                raise RuntimeError(
                    "Loading audio files requires 'soundfile'. "
                    "Install with: pip install soundfile"
                ) from exc
            data, sr = sf.read(str(audio), dtype="float32", always_2d=False)
            if data.ndim > 1:
                data = data.mean(axis=1)
            if sr != self.settings.sample_rate:
                data = self._resample(data, sr, self.settings.sample_rate)
            return data

        import numpy as np

        arr = np.asarray(audio, dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=tuple(range(1, arr.ndim)))
        return arr

    def _resample(self, waveform: Any, src_sr: int, dst_sr: int) -> Any:
        """Lightweight linear resample to avoid a heavy librosa/scipy dep."""
        if src_sr == dst_sr:
            return waveform
        import numpy as np

        ratio = dst_sr / src_sr
        new_len = int(round(len(waveform) * ratio))
        if new_len <= 1:
            return waveform
        old_idx = np.linspace(0.0, len(waveform) - 1.0, num=new_len, dtype="float32")
        src_idx = np.arange(len(waveform), dtype="float32")
        return np.interp(old_idx, src_idx, waveform).astype("float32")


# --- Plugin registry hooks ---------------------------------------------------

_PLUGIN_INSTANCE: Optional[CLAPAudioAwarenessPlugin] = None


def register(cfg: Optional["HomieConfig"] = None) -> CLAPAudioAwarenessPlugin:
    """Register and activate the plugin, returning the singleton instance.

    Called by HOMIE's plugin loader at startup. Subsequent calls return the
    already-registered instance.
    """
    global _PLUGIN_INSTANCE
    if _PLUGIN_INSTANCE is None:
        _PLUGIN_INSTANCE = CLAPAudioAwarenessPlugin(_resolve_settings(cfg))
    _PLUGIN_INSTANCE.activate()
    return _PLUGIN_INSTANCE


def activate(cfg: Optional["HomieConfig"] = None) -> CLAPAudioAwarenessPlugin:
    """Alias for :func:`register` to match HOMIE plugin conventions."""
    return register(cfg)


def deactivate() -> None:
    """Tear down the plugin singleton (if any)."""
    global _PLUGIN_INSTANCE
    if _PLUGIN_INSTANCE is not None:
        _PLUGIN_INSTANCE.deactivate()
        _PLUGIN_INSTANCE = None


__all__ = [
    "PLUGIN_NAME",
    "CLAPSettings",
    "AudioClassification",
    "CLAPDetection",
    "CLAPAudioAwarenessPlugin",
    "register",
    "activate",
    "deactivate",
]
