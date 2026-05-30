"""Kokoro TTS plugin for Homie.

Provides local text-to-speech using the hexgrad/Kokoro-TTS model
(https://huggingface.co/spaces/hexgrad/Kokoro-TTS). Kokoro is a compact
(~82M parameter) high-quality TTS model that runs efficiently on CPU/GPU
without any network access after the initial model download.

This plugin is the local voice-output backbone for assistant replies in
Homie: text in, spoken audio out, all on-device. Network access is fully
opt-in (controlled by the ``allow_download`` setting) and is only ever
used for the one-time model fetch.

Expected config layout (homie.config.yaml)::

    tts:
      kokoro:
        model_dir: ~/.homie/models/kokoro
        voice: af_sky
        sample_rate: 24000
        speed: 1.0
        enable_playback: true
        allow_download: false   # set to true to fetch model on first run

Runtime dependencies (installed separately to keep the core lean):
    - ``kokoro`` or ``kokoro-onnx`` for the inference pipeline
    - ``sounddevice`` for live playback (optional)
    - ``soundfile`` for WAV export (optional)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from homie.config import HomieConfig, cfg_get

LOGGER = logging.getLogger(__name__)

DEFAULT_VOICE = "af_sky"
DEFAULT_LANG_CODE = "a"  # American English in Kokoro's lang code scheme
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_MODEL_DIR = Path.home() / ".homie" / "models" / "kokoro"


@dataclass
class KokoroSettings:
    """Runtime settings for the Kokoro TTS plugin."""

    model_dir: Path = DEFAULT_MODEL_DIR
    voice: str = DEFAULT_VOICE
    lang_code: str = DEFAULT_LANG_CODE
    sample_rate: int = DEFAULT_SAMPLE_RATE
    speed: float = 1.0
    enable_playback: bool = True
    allow_download: bool = False


class KokoroTTSPlugin:
    """Local Kokoro TTS engine wrapper for Homie.

    Lazily loads the Kokoro pipeline on first synthesis call. All inference
    runs locally on the user's machine; the pipeline is shared via a
    thread-safe singleton inside the plugin instance.
    """

    name: str = "kokoro_tts"
    version: str = "0.1.0"
    kind: str = "tts"

    def __init__(self, settings: KokoroSettings) -> None:
        self.settings = settings
        self._pipeline: Any = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def _ensure_pipeline(self) -> Any:
        """Lazily build the Kokoro pipeline. Thread-safe and idempotent."""
        if self._pipeline is not None:
            return self._pipeline

        with self._lock:
            if self._pipeline is not None:
                return self._pipeline

            try:
                from kokoro import KPipeline  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "kokoro package not installed. Install with "
                    "`pip install kokoro` (PyTorch backend) or "
                    "`pip install kokoro-onnx` (ONNX backend)."
                ) from exc

            if not self.settings.model_dir.exists() and not self.settings.allow_download:
                raise RuntimeError(
                    f"Kokoro model directory missing: {self.settings.model_dir}. "
                    "Set `tts.kokoro.allow_download: true` in homie.config.yaml "
                    "to fetch the model on first run."
                )

            self.settings.model_dir.mkdir(parents=True, exist_ok=True)
            self._pipeline = KPipeline(lang_code=self.settings.lang_code)
            LOGGER.info(
                "Kokoro TTS pipeline initialized (voice=%s, lang=%s, rate=%d)",
                self.settings.voice,
                self.settings.lang_code,
                self.settings.sample_rate,
            )
            return self._pipeline

    def shutdown(self) -> None:
        """Release the loaded pipeline and free any GPU/CPU buffers."""
        with self._lock:
            self._pipeline = None
            LOGGER.info("Kokoro TTS pipeline released")

    # ------------------------------------------------------------------ #
    # Core synthesis API
    # ------------------------------------------------------------------ #
    def synthesize(self, text: str) -> Optional[Any]:
        """Synthesize a single audio buffer for the given text.

        Returns the first audio chunk yielded by the Kokoro pipeline as a
        numpy array sampled at ``settings.sample_rate``. Returns ``None``
        when the input is empty.
        """
        if not text or not text.strip():
            return None
        pipeline = self._ensure_pipeline()
        for _graphemes, _phonemes, audio in pipeline(
            text,
            voice=self.settings.voice,
            speed=self.settings.speed,
        ):
            return audio
        return None

    def synthesize_stream(self, text: str) -> Iterator[Any]:
        """Yield audio chunks for long-form text without buffering them all.

        Useful when piping assistant replies straight to playback so the
        user hears the first words before the full response is rendered.
        """
        if not text or not text.strip():
            return
        pipeline = self._ensure_pipeline()
        for _graphemes, _phonemes, audio in pipeline(
            text,
            voice=self.settings.voice,
            speed=self.settings.speed,
        ):
            yield audio

    def speak(self, text: str) -> bool:
        """Synthesize and play audio through the default local output device.

        Returns ``True`` on successful playback. Returns ``False`` (logging
        the reason) when playback is disabled in config, ``sounddevice`` is
        missing, or the audio device is unavailable.
        """
        if not self.settings.enable_playback:
            LOGGER.debug("Kokoro playback disabled by config")
            return False

        try:
            import sounddevice as sd  # type: ignore
        except ImportError:
            LOGGER.warning(
                "sounddevice not installed; cannot play Kokoro audio. "
                "`pip install sounddevice` to enable local playback."
            )
            return False

        played_any = False
        try:
            for chunk in self.synthesize_stream(text):
                sd.play(chunk, samplerate=self.settings.sample_rate)
                sd.wait()
                played_any = True
        except Exception as exc:  # noqa: BLE001 - audio backends raise varied exc types
            LOGGER.exception("Kokoro playback failed: %s", exc)
            return False
        return played_any

    def save_wav(self, text: str, path: Path) -> Optional[Path]:
        """Synthesize text and write a 16-bit PCM WAV file to ``path``."""
        audio = self.synthesize(text)
        if audio is None:
            return None
        try:
            import soundfile as sf  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "soundfile required to save WAV output. Install with `pip install soundfile`."
            ) from exc
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), audio, self.settings.sample_rate, subtype="PCM_16")
        LOGGER.info("Kokoro audio saved to %s", out)
        return out


# ---------------------------------------------------------------------- #
# Plugin entry points
# ---------------------------------------------------------------------- #
_active_plugin: Optional[KokoroTTSPlugin] = None


def _settings_from_config(cfg: Optional[HomieConfig]) -> KokoroSettings:
    """Build :class:`KokoroSettings` from a Homie config (or defaults)."""
    if cfg is None:
        return KokoroSettings()

    model_dir = cfg_get(cfg, "tts", "kokoro", "model_dir", default=str(DEFAULT_MODEL_DIR))
    voice = cfg_get(cfg, "tts", "kokoro", "voice", default=DEFAULT_VOICE)
    lang_code = cfg_get(cfg, "tts", "kokoro", "lang_code", default=DEFAULT_LANG_CODE)
    sample_rate = int(cfg_get(cfg, "tts", "kokoro", "sample_rate", default=DEFAULT_SAMPLE_RATE))
    speed = float(cfg_get(cfg, "tts", "kokoro", "speed", default=1.0))
    enable_playback = bool(cfg_get(cfg, "tts", "kokoro", "enable_playback", default=True))
    allow_download = bool(cfg_get(cfg, "tts", "kokoro", "allow_download", default=False))

    return KokoroSettings(
        model_dir=Path(model_dir).expanduser(),
        voice=voice,
        lang_code=lang_code,
        sample_rate=sample_rate,
        speed=speed,
        enable_playback=enable_playback,
        allow_download=allow_download,
    )


def activate(cfg: Optional[HomieConfig] = None) -> KokoroTTSPlugin:
    """Activate the Kokoro TTS plugin and return the singleton instance."""
    global _active_plugin
    if _active_plugin is not None:
        return _active_plugin
    settings = _settings_from_config(cfg)
    _active_plugin = KokoroTTSPlugin(settings)
    LOGGER.info("Kokoro TTS plugin activated (voice=%s)", settings.voice)
    return _active_plugin


def deactivate() -> None:
    """Deactivate the Kokoro TTS plugin and release resources."""
    global _active_plugin
    if _active_plugin is not None:
        _active_plugin.shutdown()
        _active_plugin = None
        LOGGER.info("Kokoro TTS plugin deactivated")


def register(cfg: Optional[HomieConfig] = None) -> Dict[str, Any]:
    """Register the plugin with Homie's plugin manager.

    Returns a manifest dict the host can use to wire the instance into
    the assistant's voice-output pipeline.
    """
    plugin = activate(cfg)
    return {
        "name": plugin.name,
        "version": plugin.version,
        "kind": plugin.kind,
        "instance": plugin,
        "deactivate": deactivate,
    }


__all__ = [
    "KokoroSettings",
    "KokoroTTSPlugin",
    "activate",
    "deactivate",
    "register",
]
