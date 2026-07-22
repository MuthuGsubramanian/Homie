"""Orate-inspired on-device neural TTS speech queue plugin for HOMIE.

Adapts the queueing model popularized by Orate (https://orate.to/ --
"Show HN: On-device neural text-to-speech queue for Mac") to HOMIE's
local voice stack. Instead of Orate's proprietary Mac-only engine,
synthesis is delegated to Piper (HOMIE's default TTS engine), with an
optional Kokoro-TTS backend so the two approaches can be compared
side by side on the same queue semantics.

Evaluation notes (Piper vs. Kokoro-TTS):
- Piper: fast CPU ONNX synthesis, tiny voices (~20-60 MB), lowest
  latency; best for short notification speech. Default backend.
- Kokoro-TTS (~82M params): noticeably more natural prosody at higher
  compute cost; better for long-form reading. Enable by installing the
  ``kokoro`` and ``soundfile`` packages and setting
  ``tts.queue.backend: kokoro`` in ``homie.config.yaml``.

Everything runs on-device: this module makes no network calls.
Playback uses whatever local audio player the OS provides
(PowerShell SoundPlayer on Windows, afplay on macOS, aplay/paplay/
ffplay on Linux).

Config keys (all optional, under ``tts.queue``):
    backend:      "piper" (default) or "kokoro"
    piper_path:   path to the piper binary (default: found on PATH)
    voice_model:  path to a Piper .onnx voice model (required for piper)
    voice:        Kokoro voice name (default: "af_heart")

Usage:
    from homie.plugins.tts_speech_queue import TTSQueuePlugin, Priority

    plugin = TTSQueuePlugin(cfg)
    plugin.activate()
    plugin.enqueue("Backup finished on msi.")
    plugin.enqueue("Disk almost full on lenovo!", priority=Priority.URGENT)
    ...
    plugin.deactivate()
"""

from __future__ import annotations

import itertools
import logging
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import List, Optional, Protocol

try:
    from homie.config import HomieConfig, cfg_get
except ImportError:  # standalone use outside the HOMIE tree
    HomieConfig = None  # type: ignore[assignment, misc]
    cfg_get = None  # type: ignore[assignment]


log = logging.getLogger(__name__)

SYNTH_TIMEOUT_SEC = 120


class Priority(IntEnum):
    """Queue priority. Lower value is spoken sooner."""

    URGENT = 0
    NORMAL = 5
    LOW = 9


_SEQ = itertools.count()


@dataclass(order=True)
class Utterance:
    """One queued piece of speech; ordered by (priority, arrival)."""

    priority: int
    seq: int
    text: str = field(compare=False, default="")


class SynthesisBackend(Protocol):
    """Anything that can turn text into a WAV file on disk."""

    name: str

    def synthesize(self, text: str, out_wav: Path) -> None: ...


class PiperBackend:
    """Synthesize with the Piper CLI (HOMIE's default TTS engine)."""

    name = "piper"

    def __init__(self, binary: Optional[str] = None, model: Optional[str] = None):
        self.binary = binary or shutil.which("piper")
        self.model = model
        if not self.binary:
            raise RuntimeError(
                "piper binary not found; install piper-tts or set tts.queue.piper_path"
            )
        if not self.model:
            raise RuntimeError(
                "no Piper voice model configured; set tts.queue.voice_model to a .onnx path"
            )

    def synthesize(self, text: str, out_wav: Path) -> None:
        cmd = [self.binary, "--model", str(self.model), "--output_file", str(out_wav)]
        proc = subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=SYNTH_TIMEOUT_SEC,
        )
        if proc.returncode != 0:
            detail = proc.stderr.decode("utf-8", errors="replace")[:200]
            raise RuntimeError(f"piper failed (rc={proc.returncode}): {detail}")


class KokoroBackend:
    """Synthesize with Kokoro-TTS (opt-in, heavier but more natural)."""

    name = "kokoro"
    SAMPLE_RATE = 24000

    def __init__(self, voice: str = "af_heart"):
        try:
            from kokoro import KPipeline
        except ImportError as exc:
            raise RuntimeError(
                "kokoro backend requested but not installed; pip install kokoro soundfile"
            ) from exc
        self._pipeline = KPipeline(lang_code="a")
        self._voice = voice

    def synthesize(self, text: str, out_wav: Path) -> None:
        import numpy as np
        import soundfile as sf

        chunks = [audio for _, _, audio in self._pipeline(text, voice=self._voice)]
        if not chunks:
            raise RuntimeError("kokoro produced no audio")
        sf.write(str(out_wav), np.concatenate(chunks), self.SAMPLE_RATE)


def _player_command(wav: Path) -> List[str]:
    """Build a blocking local playback command for the current OS."""
    if sys.platform == "win32":
        script = f"(New-Object Media.SoundPlayer '{wav}').PlaySync()"
        return ["powershell", "-NoProfile", "-Command", script]
    if sys.platform == "darwin":
        return ["afplay", str(wav)]
    for candidate in ("aplay", "paplay", "ffplay"):
        found = shutil.which(candidate)
        if found and candidate == "ffplay":
            return [found, "-nodisp", "-autoexit", "-loglevel", "quiet", str(wav)]
        if found:
            return [found, str(wav)]
    raise RuntimeError("no local audio player found (tried aplay/paplay/ffplay)")


class TTSQueuePlugin:
    """Priority speech queue drained by a single background worker.

    Utterances are synthesized and played strictly one at a time, so
    overlapping notifications never talk over each other -- the core
    idea borrowed from Orate.
    """

    def __init__(
        self,
        cfg: Optional["HomieConfig"] = None,
        backend: Optional[SynthesisBackend] = None,
    ):
        self.cfg = cfg
        self._queue: "queue.PriorityQueue[Utterance]" = queue.PriorityQueue()
        self._stop = threading.Event()
        self._skip = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._player_lock = threading.Lock()
        self._current_player: Optional[subprocess.Popen] = None
        self._backend = backend or self._backend_from_config()

    def _cfg(self, *keys: str, default=None):
        if self.cfg is None or cfg_get is None:
            return default
        return cfg_get(self.cfg, *keys, default=default)

    def _backend_from_config(self) -> SynthesisBackend:
        name = self._cfg("tts", "queue", "backend", default="piper")
        if name == "kokoro":
            return KokoroBackend(voice=self._cfg("tts", "queue", "voice", default="af_heart"))
        return PiperBackend(
            binary=self._cfg("tts", "queue", "piper_path"),
            model=self._cfg("tts", "queue", "voice_model"),
        )

    # -- lifecycle -----------------------------------------------------

    def activate(self) -> None:
        """Start the playback worker. Idempotent."""
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = threading.Thread(
            target=self._run, name="homie-tts-queue", daemon=True
        )
        self._worker.start()
        log.info("TTS queue activated (backend=%s)", self._backend.name)

    def deactivate(self) -> None:
        """Stop the worker, cut off current speech, and drop the queue."""
        self._stop.set()
        self.skip()
        if self._worker is not None:
            self._worker.join(timeout=5)
            self._worker = None
        self.clear()
        log.info("TTS queue deactivated")

    # -- public API ----------------------------------------------------

    def enqueue(self, text: str, priority: Priority = Priority.NORMAL) -> None:
        """Queue ``text`` for speech; urgent items jump ahead of normal ones."""
        text = (text or "").strip()
        if not text:
            return
        self._queue.put(Utterance(int(priority), next(_SEQ), text))

    def skip(self) -> None:
        """Stop the utterance currently being spoken (queue keeps going)."""
        self._skip.set()
        with self._player_lock:
            if self._current_player is not None and self._current_player.poll() is None:
                self._current_player.terminate()

    def clear(self) -> None:
        """Drop all pending utterances without speaking them."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def pending(self) -> int:
        """Approximate number of utterances waiting to be spoken."""
        return self._queue.qsize()

    # -- worker --------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                utterance = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._speak(utterance)
            except Exception:  # noqa: BLE001
                log.exception("Failed to speak utterance: %.60s", utterance.text)

    def _speak(self, utterance: Utterance) -> None:
        tmp = tempfile.NamedTemporaryFile(
            prefix="homie_tts_", suffix=".wav", delete=False
        )
        wav_path = Path(tmp.name)
        tmp.close()
        try:
            self._backend.synthesize(utterance.text, wav_path)
            if not self._stop.is_set():
                self._play(wav_path)
        finally:
            wav_path.unlink(missing_ok=True)

    def _play(self, wav: Path) -> None:
        cmd = _player_command(wav)
        with self._player_lock:
            self._skip.clear()
            self._current_player = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        proc = self._current_player
        while proc.poll() is None:
            if self._skip.is_set() or self._stop.is_set():
                proc.terminate()
                break
            time.sleep(0.1)
        with self._player_lock:
            self._current_player = None
            self._skip.clear()


def register(cfg: Optional["HomieConfig"] = None) -> TTSQueuePlugin:
    """HOMIE plugin entry point: build, activate, and return the queue."""
    plugin = TTSQueuePlugin(cfg)
    plugin.activate()
    return plugin


__all__ = [
    "Priority",
    "Utterance",
    "PiperBackend",
    "KokoroBackend",
    "TTSQueuePlugin",
    "register",
]
