"""Kokoro TTS voice plugin for Homie.

Deepens Homie's existing Kokoro-based speech output with richer voice features:

- **Multi-voice**: enumerate every voice shipped in the local Kokoro voice pack
  and pick a voice per utterance (e.g. ``af_heart`` for notifications,
  ``am_michael`` for status reports).
- **Voice blending**: mix two voice style vectors at an arbitrary ratio to
  create custom in-between voices, entirely offline.
- **Streaming synthesis**: sentence-level chunked generation so long answers
  start playing almost immediately instead of waiting for the full waveform.
- **Optional local playback / WAV export**: plays through ``sounddevice`` when
  installed, and can write standard 16-bit PCM WAV files with only stdlib.

Everything runs locally against the Kokoro ONNX model (~82M params, CPU-friendly).
The plugin NEVER touches the network unless ``allow_download: true`` is set in
config, in which case it fetches the model/voice files once into ``~/.homie/models``.

Configuration (all optional), read from the Homie config dict::

    tts:
      kokoro:
        model_path: ~/.homie/models/kokoro-v1.0.onnx
        voices_path: ~/.homie/models/voices-v1.0.bin
        default_voice: af_heart
        speed: 1.0
        lang: en-us
        allow_download: false   # opt-in network access

Dependencies: ``kokoro-onnx`` (pulls in onnxruntime + numpy). ``sounddevice``
is optional and only needed for ``speak()``.
"""

from __future__ import annotations

import logging
import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path.home() / ".homie" / "models" / "kokoro-v1.0.onnx"
DEFAULT_VOICES_PATH = Path.home() / ".homie" / "models" / "voices-v1.0.bin"
DEFAULT_VOICE = "af_heart"
DEFAULT_SPEED = 1.0
DEFAULT_LANG = "en-us"

# Only used when the user explicitly opts in via allow_download: true.
MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

_SENTENCE_RE = re.compile(r"(?<=[.!?;:])\s+")


class KokoroTTSError(Exception):
    """Raised when synthesis cannot proceed (missing deps, files, or voices)."""


@dataclass
class AudioChunk:
    """A synthesized slice of audio.

    ``samples`` is a 1-D float32 numpy array in [-1.0, 1.0]. Typed as ``Any``
    so importing this module never requires numpy up front.
    """

    samples: Any
    sample_rate: int
    text: str

    @property
    def duration_sec(self) -> float:
        return float(len(self.samples)) / float(self.sample_rate)


class KokoroTTSPlugin:
    """Multi-voice, streaming Kokoro TTS engine for Homie."""

    name = "kokoro_voice"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = ((config or {}).get("tts", {}) or {}).get("kokoro", {}) or {}
        self.model_path = Path(cfg.get("model_path", DEFAULT_MODEL_PATH)).expanduser()
        self.voices_path = Path(cfg.get("voices_path", DEFAULT_VOICES_PATH)).expanduser()
        self.default_voice: str = cfg.get("default_voice", DEFAULT_VOICE)
        self.speed: float = float(cfg.get("speed", DEFAULT_SPEED))
        self.lang: str = cfg.get("lang", DEFAULT_LANG)
        self.allow_download: bool = bool(cfg.get("allow_download", False))
        self._engine: Any = None
        self._voices_cache: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def activate(self) -> None:
        """Load the Kokoro ONNX engine (idempotent)."""
        if self._engine is not None:
            return
        self._ensure_model_files()
        try:
            from kokoro_onnx import Kokoro  # noqa: PLC0415 (lazy heavy import)
        except ImportError as exc:
            raise KokoroTTSError(
                "kokoro-onnx is not installed. Run: pip install kokoro-onnx"
            ) from exc
        logger.info("Loading Kokoro model from %s", self.model_path)
        self._engine = Kokoro(str(self.model_path), str(self.voices_path))
        logger.info("Kokoro TTS ready (%d voices)", len(self.list_voices()))

    def deactivate(self) -> None:
        """Release the engine and cached voice list."""
        self._engine = None
        self._voices_cache = None
        logger.info("Kokoro TTS deactivated")

    # ------------------------------------------------------------------
    # Voices
    # ------------------------------------------------------------------
    def list_voices(self) -> List[str]:
        """Return all voice names available in the local voice pack."""
        self._require_engine()
        if self._voices_cache is None:
            self._voices_cache = sorted(self._engine.get_voices())
        return list(self._voices_cache)

    def _resolve_style(
        self,
        voice: Optional[str],
        blend: Optional[Tuple[str, str, float]],
    ) -> Any:
        """Return a voice name or a blended style vector for the engine.

        ``blend`` is ``(voice_a, voice_b, ratio)`` where ``ratio`` is the
        weight of ``voice_a`` (0.0-1.0).
        """
        if blend is not None:
            import numpy as np  # noqa: PLC0415

            voice_a, voice_b, ratio = blend
            ratio = min(max(float(ratio), 0.0), 1.0)
            for name in (voice_a, voice_b):
                if name not in self.list_voices():
                    raise KokoroTTSError(f"Unknown voice '{name}'. See list_voices().")
            style_a = self._engine.get_voice_style(voice_a)
            style_b = self._engine.get_voice_style(voice_b)
            return np.add(style_a * ratio, style_b * (1.0 - ratio))
        chosen = voice or self.default_voice
        if chosen not in self.list_voices():
            raise KokoroTTSError(f"Unknown voice '{chosen}'. See list_voices().")
        return chosen

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------
    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        blend: Optional[Tuple[str, str, float]] = None,
    ) -> AudioChunk:
        """Synthesize ``text`` to a single audio chunk.

        Pass ``blend=(\"af_heart\", \"am_michael\", 0.6)`` to mix two voices.
        """
        self._require_engine()
        cleaned = text.strip()
        if not cleaned:
            raise KokoroTTSError("Cannot synthesize empty text")
        style = self._resolve_style(voice, blend)
        samples, sample_rate = self._engine.create(
            cleaned,
            voice=style,
            speed=speed if speed is not None else self.speed,
            lang=self.lang,
        )
        return AudioChunk(samples=samples, sample_rate=int(sample_rate), text=cleaned)

    def stream(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        blend: Optional[Tuple[str, str, float]] = None,
    ) -> Generator[AudioChunk, None, None]:
        """Yield audio sentence-by-sentence for low-latency playback.

        Long LLM answers start sounding after the first sentence is ready
        instead of after the whole reply is synthesized.
        """
        self._require_engine()
        for sentence in self._split_sentences(text):
            yield self.synthesize(sentence, voice=voice, speed=speed, blend=blend)

    def speak(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        blend: Optional[Tuple[str, str, float]] = None,
        streaming: bool = True,
    ) -> float:
        """Synthesize and play ``text`` through the default output device.

        Returns total audio duration in seconds. Requires the optional
        ``sounddevice`` package; raises :class:`KokoroTTSError` otherwise.
        """
        try:
            import sounddevice as sd  # noqa: PLC0415
        except ImportError as exc:
            raise KokoroTTSError(
                "sounddevice is not installed. Run: pip install sounddevice"
            ) from exc
        total = 0.0
        chunks = (
            self.stream(text, voice=voice, speed=speed, blend=blend)
            if streaming
            else iter([self.synthesize(text, voice=voice, speed=speed, blend=blend)])
        )
        for chunk in chunks:
            sd.play(chunk.samples, chunk.sample_rate, blocking=True)
            total += chunk.duration_sec
        return total

    def save_wav(
        self,
        text: str,
        out_path: str | Path,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        blend: Optional[Tuple[str, str, float]] = None,
    ) -> Path:
        """Synthesize ``text`` and write a 16-bit PCM mono WAV file."""
        import numpy as np  # noqa: PLC0415

        chunk = self.synthesize(text, voice=voice, speed=speed, blend=blend)
        path = Path(out_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        pcm = (np.clip(chunk.samples, -1.0, 1.0) * 32767.0).astype(np.int16)
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(chunk.sample_rate)
            wav.writeframes(pcm.tobytes())
        logger.info("Wrote %.1fs of audio to %s", chunk.duration_sec, path)
        return path

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _require_engine(self) -> None:
        if self._engine is None:
            raise KokoroTTSError("Plugin not activated. Call activate() first.")

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        parts = [p.strip() for p in _SENTENCE_RE.split(text.strip())]
        return [p for p in parts if p]

    def _ensure_model_files(self) -> None:
        missing = [p for p in (self.model_path, self.voices_path) if not p.exists()]
        if not missing:
            return
        if not self.allow_download:
            raise KokoroTTSError(
                "Missing Kokoro model files: "
                + ", ".join(str(p) for p in missing)
                + ". Download them manually, or set tts.kokoro.allow_download: true "
                "to let the plugin fetch them once (opt-in network access)."
            )
        from urllib.request import urlretrieve  # noqa: PLC0415 (opt-in only)

        for path, url in ((self.model_path, MODEL_URL), (self.voices_path, VOICES_URL)):
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading %s -> %s (one-time, user opted in)", url, path)
            tmp = path.with_suffix(path.suffix + ".part")
            urlretrieve(url, tmp)  # noqa: S310 (fixed https URLs, explicit opt-in)
            tmp.replace(path)


# ----------------------------------------------------------------------
# Module-level plugin protocol (Homie convention)
# ----------------------------------------------------------------------
_PLUGIN: Optional[KokoroTTSPlugin] = None


def register(config: Optional[Dict[str, Any]] = None) -> KokoroTTSPlugin:
    """Create (or return) the singleton plugin instance without loading the model."""
    global _PLUGIN
    if _PLUGIN is None:
        _PLUGIN = KokoroTTSPlugin(config)
    return _PLUGIN


def activate(config: Optional[Dict[str, Any]] = None) -> KokoroTTSPlugin:
    """Register and fully load the plugin. Entry point called by Homie."""
    plugin = register(config)
    plugin.activate()
    return plugin


def deactivate() -> None:
    """Release engine resources. Entry point called by Homie."""
    global _PLUGIN
    if _PLUGIN is not None:
        _PLUGIN.deactivate()
        _PLUGIN = None


__all__ = [
    "KokoroTTSPlugin",
    "KokoroTTSError",
    "AudioChunk",
    "register",
    "activate",
    "deactivate",
]
