"""Kokoro TTS voice-output plugin for HOMIE.

Gives HOMIE a local, CPU-friendly voice using the 82M-parameter Kokoro model
(https://huggingface.co/hexgrad/Kokoro-82M). All synthesis and playback happen
on-device. Downloading the model weights is opt-in: it only happens when
``tts.allow_download: true`` is set in the HOMIE config; otherwise the plugin
runs fully offline against an already-cached model.

Voice-UX iteration over the originally shipped plugin:
  * Non-blocking priority speech queue -- alerts jump ahead of chatter.
  * Barge-in: ``interrupt()`` (or ``alert()``) stops playback mid-sentence so
    HOMIE never talks over the user.
  * Sentence-level streaming -- long replies start playing after the first
    sentence instead of waiting for full synthesis.
  * Per-event voice presets (e.g. a distinct voice/speed for alerts).
  * Mute/volume controls and a WAV-file fallback for headless nodes with no
    audio device.

Config (``homie.config.yaml``)::

    tts:
      enabled: true
      voice: af_heart
      speed: 1.0
      volume: 0.9
      lang_code: a            # 'a' = American English
      allow_download: false   # opt-in network: fetch Kokoro-82M from HF
      wav_dir: ~/.homie/tts   # fallback output when no audio device
      presets:
        alert:
          voice: am_michael
          speed: 1.1

Optional dependencies: ``kokoro>=0.9``, ``sounddevice``, ``soundfile``.
The plugin degrades gracefully (logs and no-ops) when they are missing.
"""

from __future__ import annotations

import logging
import os
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from homie.config import HomieConfig, cfg_get

SAMPLE_RATE = 24_000
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _chunk_sentences(text: str) -> Iterable[str]:
    """Split text into sentences so long replies start playing immediately."""
    for part in _SENTENCE_RE.split(text):
        part = part.strip()
        if part:
            yield part


def _to_numpy(audio: Any) -> Any:
    """Convert torch tensors from Kokoro into numpy arrays for playback."""
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    return audio


@dataclass(order=True)
class SpeechRequest:
    """A queued utterance; lower ``priority`` speaks first, FIFO within a tier."""

    priority: int
    seq: int
    text: str = field(compare=False)
    preset: str = field(default="default", compare=False)


class KokoroTTSPlugin:
    """Local text-to-speech for HOMIE backed by Kokoro-82M."""

    name = "kokoro_tts"

    def __init__(self, cfg: HomieConfig) -> None:
        self.cfg = cfg
        self._pipeline: Any = None
        self._queue: "queue.PriorityQueue[SpeechRequest]" = queue.PriorityQueue()
        self._worker: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._interrupt = threading.Event()
        self._seq = 0
        self._muted = False
        self._active = False

    # ------------------------------------------------------------ lifecycle

    def activate(self) -> bool:
        """Start the speech worker. Returns True when voice output is live."""
        if not cfg_get(self.cfg, "tts", "enabled", default=True):
            logging.info("kokoro_tts: disabled via config")
            return False
        if self._active:
            return True
        self._stop.clear()
        self._worker = threading.Thread(target=self._run, name="kokoro-tts", daemon=True)
        self._worker.start()
        self._active = True
        logging.info("kokoro_tts: activated (voice=%s)", self._preset("default")["voice"])
        return True

    def deactivate(self) -> None:
        """Stop playback, drain the queue and shut the worker down."""
        if not self._active:
            return
        self._interrupt.set()
        self._stop.set()
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        if self._worker is not None:
            self._worker.join(timeout=5.0)
        self._worker = None
        self._active = False
        logging.info("kokoro_tts: deactivated")

    # ------------------------------------------------------------ public API

    def speak(self, text: str, priority: int = 5, preset: str = "default") -> bool:
        """Queue text for speech. Returns False if inactive or text is empty."""
        text = (text or "").strip()
        if not text or not self._active:
            return False
        self._seq += 1
        self._queue.put(SpeechRequest(priority=priority, seq=self._seq, text=text, preset=preset))
        return True

    def alert(self, text: str) -> bool:
        """Barge in and speak an urgent message ahead of anything queued."""
        self.interrupt()
        return self.speak(text, priority=0, preset="alert")

    def interrupt(self) -> None:
        """Stop the current utterance without draining the queue."""
        self._interrupt.set()

    def set_muted(self, muted: bool) -> None:
        """Mute (queued utterances are dropped as they surface) and stop playback."""
        self._muted = muted
        if muted:
            self.interrupt()

    # ------------------------------------------------------------ internals

    def _preset(self, name: str) -> Dict[str, Any]:
        base: Dict[str, Any] = {
            "voice": cfg_get(self.cfg, "tts", "voice", default="af_heart"),
            "speed": float(cfg_get(self.cfg, "tts", "speed", default=1.0)),
            "volume": float(cfg_get(self.cfg, "tts", "volume", default=0.9)),
        }
        presets = cfg_get(self.cfg, "tts", "presets", default={}) or {}
        override = presets.get(name) or {}
        for key in ("voice", "speed", "volume"):
            if key in override:
                base[key] = override[key]
        return base

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        if not cfg_get(self.cfg, "tts", "allow_download", default=False):
            # Stay offline unless the user explicitly opted into downloads.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        try:
            from kokoro import KPipeline
        except ImportError:
            logging.warning("kokoro_tts: 'kokoro' package not installed; voice output disabled")
            return None
        try:
            self._pipeline = KPipeline(lang_code=cfg_get(self.cfg, "tts", "lang_code", default="a"))
        except Exception:  # noqa: BLE001
            logging.exception(
                "kokoro_tts: could not load Kokoro-82M "
                "(enable tts.allow_download to fetch it once)"
            )
            return None
        return self._pipeline

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                req = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if self._muted or self._stop.is_set():
                continue
            self._interrupt.clear()
            try:
                self._speak_now(req)
            except Exception:  # noqa: BLE001
                logging.exception("kokoro_tts: synthesis failed")

    def _speak_now(self, req: SpeechRequest) -> None:
        pipeline = self._load_pipeline()
        if pipeline is None:
            return
        settings = self._preset(req.preset)
        for sentence in _chunk_sentences(req.text):
            if self._interrupt.is_set() or self._stop.is_set():
                return
            for _, _, audio in pipeline(sentence, voice=settings["voice"], speed=settings["speed"]):
                if self._interrupt.is_set() or self._stop.is_set():
                    return
                self._play(_to_numpy(audio), float(settings["volume"]))

    def _play(self, audio: Any, volume: float) -> None:
        try:
            import sounddevice as sd
        except Exception:  # noqa: BLE001
            self._write_wav(audio)
            return
        try:
            sd.play(audio * volume, SAMPLE_RATE)
            # Poll instead of sd.wait() so barge-in can cut playback short.
            deadline = time.monotonic() + (len(audio) / SAMPLE_RATE) + 0.25
            while time.monotonic() < deadline:
                if self._interrupt.is_set() or self._stop.is_set():
                    sd.stop()
                    return
                time.sleep(0.05)
            sd.wait()
        except Exception:  # noqa: BLE001
            logging.exception("kokoro_tts: playback failed, writing WAV instead")
            self._write_wav(audio)

    def _write_wav(self, audio: Any) -> None:
        try:
            import soundfile as sf
        except Exception:  # noqa: BLE001
            logging.warning("kokoro_tts: no audio backend available; dropping utterance")
            return
        wav_dir = Path(os.path.expanduser(cfg_get(self.cfg, "tts", "wav_dir", default="~/.homie/tts")))
        wav_dir.mkdir(parents=True, exist_ok=True)
        out = wav_dir / f"homie-{int(time.time() * 1000)}.wav"
        sf.write(str(out), audio, SAMPLE_RATE)
        logging.info("kokoro_tts: no audio device; wrote %s", out)


def register(cfg: Optional[HomieConfig] = None) -> KokoroTTSPlugin:
    """HOMIE plugin entry point: build and activate the Kokoro TTS plugin."""
    if cfg is None:
        from homie.config import load_config

        cfg = load_config()
    plugin = KokoroTTSPlugin(cfg)
    plugin.activate()
    return plugin


__all__ = ["KokoroTTSPlugin", "SpeechRequest", "register"]
