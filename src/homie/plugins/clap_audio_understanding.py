"""HOMIE audio-understanding plugin powered by ``laion/clap-htsat-fused``.

This plugin gives HOMIE the ability to *listen* to short audio clips and reason
about the sounds inside them using CLAP (Contrastive Language-Audio
Pretraining). CLAP embeds audio and natural-language labels into a shared space,
so it supports **zero-shot** sound/event detection: you hand it an audio file
plus a list of candidate descriptions ("dog barking", "glass breaking",
"running water", ...) and it ranks how well each label matches the audio. It can
also emit raw audio embeddings suitable for indexing in HOMIE's RAG store
(ChromaDB) for similarity search over a sound library.

Design notes (consistent with the rest of the HOMIE codebase):
- Single file, lazily imports heavy ML dependencies (``torch``, ``transformers``,
  ``soundfile``/``librosa``) only when inference actually runs, so importing the
  plugin stays cheap and never crashes a machine that lacks the ML stack.
- Fully **local**: inference performs no network calls. The only network
  activity is the one-time model download from the Hugging Face Hub, which is
  *opt-in* via ``allow_download`` (default ``False``). If the weights are not
  already cached locally and downloads are disabled, activation fails loudly
  instead of silently reaching out to the internet.
- Configuration is read through the existing :func:`homie.config.cfg_get`
  helper when a ``HomieConfig`` is supplied, mirroring how other modules
  (orchestrator, automations) consume config.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:  # avoid importing heavy/optional deps at module load time
    import numpy as np

logger = logging.getLogger(__name__)

PLUGIN_NAME = "clap_audio_understanding"
DEFAULT_MODEL_ID = "laion/clap-htsat-fused"

# CLAP was trained on 48 kHz mono audio; we resample to this before inference.
TARGET_SAMPLE_RATE = 48_000

# A small, generic palette of everyday sounds used when the caller does not
# supply their own candidate labels. Override per-call via ``labels=``.
DEFAULT_LABELS: Tuple[str, ...] = (
    "speech",
    "music",
    "silence",
    "dog barking",
    "cat meowing",
    "glass breaking",
    "door knocking",
    "running water",
    "car engine",
    "alarm or siren",
    "rain",
    "applause",
    "typing on a keyboard",
)


@dataclass
class ClapConfig:
    """Resolved settings for the CLAP audio-understanding plugin."""

    model_id: str = DEFAULT_MODEL_ID
    device: str = "cpu"  # "cpu", "cuda", or "mps"
    allow_download: bool = False  # opt-in network: fetch weights from HF Hub
    cache_dir: Optional[str] = None  # custom HF cache; None -> HF default
    default_labels: Tuple[str, ...] = DEFAULT_LABELS

    @classmethod
    def from_homie_config(cls, cfg: Any = None) -> "ClapConfig":
        """Build a :class:`ClapConfig` from a ``HomieConfig`` (or defaults).

        Reads the optional ``plugins.clap_audio`` section, e.g.::

            plugins:
              clap_audio:
                model_id: laion/clap-htsat-fused
                device: cpu
                allow_download: false
                cache_dir: ~/.cache/homie/clap

        Falls back to dataclass defaults for any missing key, and tolerates a
        ``None`` config so the plugin is usable standalone in tests.
        """
        if cfg is None:
            return cls()

        try:
            from homie.config import cfg_get
        except Exception:  # noqa: BLE001 - standalone / partial installs
            return cls()

        section: Dict[str, Any] = cfg_get(cfg, "plugins", "clap_audio", default={}) or {}
        labels = section.get("default_labels") or DEFAULT_LABELS
        return cls(
            model_id=section.get("model_id", DEFAULT_MODEL_ID),
            device=section.get("device", "cpu"),
            allow_download=bool(section.get("allow_download", False)),
            cache_dir=section.get("cache_dir"),
            default_labels=tuple(labels),
        )


@dataclass
class AudioPrediction:
    """Result of classifying one audio clip against candidate labels."""

    top_label: str
    scores: Dict[str, float]  # label -> probability, descending
    source: str  # path or identifier of the analysed audio

    def as_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (JSON/RAG friendly)."""
        return {
            "top_label": self.top_label,
            "scores": self.scores,
            "source": self.source,
        }


@dataclass
class ClapAudioPlugin:
    """Local zero-shot audio classifier + embedder for HOMIE.

    Lifecycle: construct, then :meth:`activate` (loads the model, honouring the
    ``allow_download`` network gate), call :meth:`classify` / :meth:`embed` as
    needed, and :meth:`deactivate` to release the model and free memory.
    """

    config: ClapConfig = field(default_factory=ClapConfig)
    _model: Any = field(default=None, init=False, repr=False)
    _processor: Any = field(default=None, init=False, repr=False)
    _active: bool = field(default=False, init=False, repr=False)

    # --- lifecycle -------------------------------------------------------

    def activate(self) -> None:
        """Load the CLAP model and processor into memory.

        Respects ``config.allow_download``: when ``False`` (default) we force
        Hugging Face into offline mode so a missing cache raises instead of
        triggering an implicit network fetch. Safe to call more than once.
        """
        if self._active:
            logger.debug("%s already active; skipping reload", PLUGIN_NAME)
            return

        try:
            import torch  # noqa: F401  (validates the ML stack is present)
            from transformers import ClapModel, ClapProcessor
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "CLAP plugin requires 'torch' and 'transformers'. Install with "
                "`pip install torch transformers soundfile`."
            ) from exc

        load_kwargs: Dict[str, Any] = {"local_files_only": not self.config.allow_download}
        if self.config.cache_dir:
            load_kwargs["cache_dir"] = str(Path(self.config.cache_dir).expanduser())

        try:
            self._model = ClapModel.from_pretrained(self.config.model_id, **load_kwargs)
            self._processor = ClapProcessor.from_pretrained(self.config.model_id, **load_kwargs)
        except OSError as exc:
            raise RuntimeError(
                f"CLAP weights for '{self.config.model_id}' are not cached locally and "
                "downloads are disabled. Set plugins.clap_audio.allow_download: true to "
                "fetch them once from the Hugging Face Hub."
            ) from exc

        self._model.to(self.config.device)
        self._model.eval()
        self._active = True
        logger.info(
            "Activated %s (model=%s device=%s)",
            PLUGIN_NAME,
            self.config.model_id,
            self.config.device,
        )

    def deactivate(self) -> None:
        """Release the model/processor and free GPU/CPU memory. Idempotent."""
        self._model = None
        self._processor = None
        self._active = False
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass
        logger.info("Deactivated %s", PLUGIN_NAME)

    @property
    def is_active(self) -> bool:
        """Whether the model is currently loaded and ready for inference."""
        return self._active

    # --- inference -------------------------------------------------------

    def classify(
        self,
        audio: "str | Path | np.ndarray",
        labels: Optional[Sequence[str]] = None,
        sample_rate: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> AudioPrediction:
        """Zero-shot classify ``audio`` against natural-language ``labels``.

        Args:
            audio: Path to an audio file, or a 1-D float waveform array.
            labels: Candidate descriptions. Defaults to ``config.default_labels``.
            sample_rate: Required only when ``audio`` is a raw array; ignored for
                file paths (read from the file header).
            top_k: If given, keep only the ``top_k`` highest-scoring labels.

        Returns:
            An :class:`AudioPrediction` with the winning label and a descending
            ``label -> probability`` map.
        """
        self._ensure_active()
        candidate = list(labels) if labels else list(self.config.default_labels)
        if not candidate:
            raise ValueError("At least one candidate label is required for classification.")

        waveform, source = self._coerce_waveform(audio, sample_rate)

        import torch

        inputs = self._processor(
            text=candidate,
            audios=waveform,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            # logits_per_audio: [n_audio, n_labels] -> softmax over labels
            probs = outputs.logits_per_audio.softmax(dim=-1).squeeze(0)

        scored = sorted(
            ((label, float(probs[i])) for i, label in enumerate(candidate)),
            key=lambda pair: pair[1],
            reverse=True,
        )
        if top_k is not None:
            scored = scored[: max(0, top_k)]

        score_map = {label: round(score, 6) for label, score in scored}
        top_label = scored[0][0] if scored else ""
        return AudioPrediction(top_label=top_label, scores=score_map, source=source)

    def embed(
        self,
        audio: "str | Path | np.ndarray",
        sample_rate: Optional[int] = None,
    ) -> List[float]:
        """Return the CLAP audio embedding for ``audio`` as a plain list.

        Useful for indexing sounds in HOMIE's ChromaDB RAG store and performing
        similarity search ("find clips that sound like this one").
        """
        self._ensure_active()
        waveform, _ = self._coerce_waveform(audio, sample_rate)

        import torch

        inputs = self._processor(
            audios=waveform,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

        with torch.no_grad():
            features = self._model.get_audio_features(**inputs)

        return features.squeeze(0).cpu().tolist()

    # --- helpers ---------------------------------------------------------

    def _ensure_active(self) -> None:
        if not self._active:
            raise RuntimeError(
                f"{PLUGIN_NAME} is not active. Call activate() before inference."
            )

    def _coerce_waveform(
        self,
        audio: "str | Path | np.ndarray",
        sample_rate: Optional[int],
    ) -> Tuple["np.ndarray", str]:
        """Normalise input into a 48 kHz mono float32 array + a source label."""
        import numpy as np

        if isinstance(audio, (str, Path)):
            path = Path(audio).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Audio file not found: {path}")
            waveform, native_sr = self._read_audio_file(path)
            source = str(path)
        else:
            waveform = np.asarray(audio, dtype=np.float32)
            if sample_rate is None:
                raise ValueError(
                    "sample_rate must be provided when passing a raw audio array."
                )
            native_sr = int(sample_rate)
            source = "<array>"

        waveform = self._to_mono(waveform)
        waveform = self._resample(waveform, native_sr, TARGET_SAMPLE_RATE)
        return waveform.astype("float32", copy=False), source

    @staticmethod
    def _read_audio_file(path: Path) -> Tuple["np.ndarray", int]:
        """Read an audio file using soundfile (preferred) or librosa."""
        try:
            import soundfile as sf

            data, native_sr = sf.read(str(path), dtype="float32", always_2d=False)
            return data, int(native_sr)
        except ImportError:
            pass

        try:
            import librosa

            data, native_sr = librosa.load(str(path), sr=None, mono=False)
            return data, int(native_sr)
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Reading audio files requires 'soundfile' or 'librosa'. Install with "
                "`pip install soundfile`."
            ) from exc

    @staticmethod
    def _to_mono(waveform: "np.ndarray") -> "np.ndarray":
        """Collapse multi-channel audio to mono by averaging channels."""
        import numpy as np

        if waveform.ndim == 1:
            return waveform
        # soundfile -> (frames, channels); librosa(mono=False) -> (channels, frames)
        axis = 1 if waveform.shape[0] >= waveform.shape[1] else 0
        return np.mean(waveform, axis=axis)

    @staticmethod
    def _resample(waveform: "np.ndarray", src_sr: int, dst_sr: int) -> "np.ndarray":
        """Resample to ``dst_sr`` (no-op if already matching)."""
        if src_sr == dst_sr:
            return waveform

        try:
            import librosa

            return librosa.resample(waveform, orig_sr=src_sr, target_sr=dst_sr)
        except ImportError:
            # Lightweight linear-interpolation fallback (lower quality).
            import numpy as np

            duration = waveform.shape[-1] / float(src_sr)
            dst_len = int(round(duration * dst_sr))
            if dst_len <= 0:
                return waveform
            src_idx = np.linspace(0.0, waveform.shape[-1] - 1, num=dst_len)
            return np.interp(src_idx, np.arange(waveform.shape[-1]), waveform).astype(
                "float32"
            )


def register(cfg: Any = None) -> ClapAudioPlugin:
    """Entry point used by HOMIE's plugin loader.

    Builds the plugin from a ``HomieConfig`` (or defaults), activates it, and
    returns the ready-to-use instance. Mirrors the activate/deactivate lifecycle
    other HOMIE components expose.
    """
    plugin = ClapAudioPlugin(config=ClapConfig.from_homie_config(cfg))
    plugin.activate()
    return plugin


__all__ = [
    "PLUGIN_NAME",
    "DEFAULT_MODEL_ID",
    "ClapConfig",
    "AudioPrediction",
    "ClapAudioPlugin",
    "register",
]
