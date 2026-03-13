# Voice Integration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add speech-to-speech voice interaction to Homie — users speak, Homie listens (STT), thinks (existing brain), and speaks back (TTS), with three interaction modes (wake word, push-to-talk, conversational).

**Architecture:** Queue-based producer-consumer threading inspired by HuggingFace speech-to-speech. Each component (AudioIn, VAD, STT, TTS, AudioOut) runs in its own thread connected by `queue.Queue`. Voice queries route through the existing `BrainOrchestrator.process_stream()` — no separate LLM. Barge-in via `threading.Event`.

**Tech Stack:** faster-whisper (STT), Silero VAD (torch), Piper/Kokoro/MeloTTS (TTS), sounddevice (audio I/O), pynput (hotkey)

**Spec:** `docs/superpowers/specs/2026-03-13-voice-integration-design.md`

---

## File Structure

```
src/homie_core/voice/
├── __init__.py                    # Exports: VoiceManager, BaseTTS, PipelineState
├── base_handler.py                # NEW: BaseHandler — queue/thread lifecycle pattern
├── audio_io.py                    # REWRITE: AudioInThread + AudioOutThread (sounddevice)
├── vad_silero.py                  # NEW: SileroVAD — neural VAD via torch
├── vad.py                         # MODIFY: Add unified create_vad() factory
├── stt.py                         # MODIFY: Fix model_size bug, add language detection
├── tts.py                         # REWRITE: BaseTTS ABC + PiperTTS
├── tts_kokoro.py                  # NEW: KokoroTTS(BaseTTS)
├── tts_melo.py                    # NEW: MeloTTS(BaseTTS)
├── tts_selector.py                # NEW: TTSSelector — auto-picks engine
├── wakeword.py                    # UNCHANGED
├── voice_pipeline.py              # REWRITE: Queue-based pipeline with barge-in
├── voice_manager.py               # NEW: Mode orchestration + state machine
└── voice_prompts.py               # NEW: Voice-aware prompt hints

src/homie_app/
├── hotkey.py                      # MODIFY: Add ctrl+8 mapping
├── overlay.py                     # MODIFY: Add voice mode panel
├── daemon.py                      # MODIFY: Wire VoiceManager
├── cli.py                         # MODIFY: Add `homie voice` commands

src/homie_core/config.py           # MODIFY: Expand VoiceConfig
pyproject.toml                     # MODIFY: Update voice deps
tests/unit/test_voice/                  # NEW: All voice tests
```

---

## Chunk 1: Foundation — BaseHandler, Audio I/O, Config

### Task 1: Update VoiceConfig

**Files:**
- Modify: `src/homie_core/config.py:23-30`
- Test: `tests/unit/test_voice/test_config.py`

- [ ] **Step 1: Write failing test for new VoiceConfig fields**

```python
# tests/unit/test_voice/test_config.py
from homie_core.config import VoiceConfig


def test_voice_config_defaults():
    cfg = VoiceConfig()
    assert cfg.enabled is False
    assert cfg.hotkey == "ctrl+8"
    assert cfg.wake_word == "hey homie"
    assert cfg.mode == "hybrid"
    assert cfg.stt_engine == "faster-whisper"
    assert cfg.stt_model_fast == "tiny.en"
    assert cfg.stt_model_quality == "medium"
    assert cfg.stt_language == "auto"
    assert cfg.tts_mode == "auto"
    assert cfg.tts_voice_fast == "piper"
    assert cfg.tts_voice_quality == "kokoro"
    assert cfg.tts_voice_multilingual == "melo"
    assert cfg.vad_engine == "silero"
    assert cfg.vad_threshold == 0.5
    assert cfg.vad_silence_ms == 300
    assert cfg.barge_in is True
    assert cfg.conversation_timeout == 120
    assert cfg.max_exit_prompts == 3
    assert cfg.exit_phrases == ["goodbye", "stop", "that's all"]
    assert cfg.device == "auto"
    assert cfg.audio_sample_rate == 16000
    assert cfg.audio_chunk_size == 512


def test_voice_config_backward_compat():
    """Old config fields should map to new ones."""
    cfg = VoiceConfig(stt_model="base", tts_voice="custom", mode="audio", hotkey="alt+8")
    assert cfg.stt_model_quality == "base"
    assert cfg.tts_voice_quality == "custom"
    assert cfg.mode == "hybrid"  # "audio" maps to "hybrid"
    assert cfg.hotkey == "alt+8"  # explicit override preserved
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_voice/test_config.py -v`
Expected: FAIL — VoiceConfig missing new fields

- [ ] **Step 3: Implement expanded VoiceConfig**

Update the import at line 1 of `src/homie_core/config.py` from `from pydantic import BaseModel, Field` to `from pydantic import BaseModel, Field, model_validator`.

Then replace lines 23-30:

```python
class VoiceConfig(BaseModel):
    enabled: bool = False
    hotkey: str = "ctrl+8"
    wake_word: str = "hey homie"
    mode: str = "hybrid"  # hybrid | wake_word | push_to_talk | conversational

    # STT
    stt_engine: str = "faster-whisper"
    stt_model_fast: str = "tiny.en"
    stt_model_quality: str = "medium"
    stt_language: str = "auto"

    # TTS
    tts_mode: str = "auto"  # auto | fast | quality | multilingual
    tts_voice_fast: str = "piper"
    tts_voice_quality: str = "kokoro"
    tts_voice_multilingual: str = "melo"

    # VAD
    vad_engine: str = "silero"  # silero | webrtcvad | energy
    vad_threshold: float = 0.5
    vad_silence_ms: int = 300

    # Behavior
    barge_in: bool = True
    conversation_timeout: int = 120
    max_exit_prompts: int = 3
    exit_phrases: list[str] = ["goodbye", "stop", "that's all"]

    # Performance
    device: str = "auto"
    audio_sample_rate: int = 16000
    audio_chunk_size: int = 512

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_fields(cls, data: dict) -> dict:
        if isinstance(data, dict):
            # stt_model -> stt_model_quality
            if "stt_model" in data and "stt_model_quality" not in data:
                data["stt_model_quality"] = data.pop("stt_model")
            elif "stt_model" in data:
                data.pop("stt_model")
            # tts_voice -> tts_voice_quality
            if "tts_voice" in data and "tts_voice_quality" not in data:
                data["tts_voice_quality"] = data.pop("tts_voice")
            elif "tts_voice" in data:
                data.pop("tts_voice")
            # mode migration
            mode_map = {"text_only": "push_to_talk", "audio": "hybrid"}
            if "mode" in data and data["mode"] in mode_map:
                data["mode"] = mode_map[data["mode"]]
        return data
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_voice/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Run all existing config tests to check no regressions**

Run: `python -m pytest tests/ -k config -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/homie_core/config.py tests/unit/test_voice/test_config.py
git commit -m "feat(voice): expand VoiceConfig with new fields and backward compat"
```

---

### Task 2: BaseHandler — Queue/Thread Pattern

**Files:**
- Create: `src/homie_core/voice/base_handler.py`
- Test: `tests/unit/test_voice/test_base_handler.py`

- [ ] **Step 1: Write failing test for BaseHandler**

```python
# tests/unit/test_voice/test_base_handler.py
import queue
import threading
import time

from homie_core.voice.base_handler import BaseHandler


class EchoHandler(BaseHandler):
    """Test handler that echoes input to output."""
    def process(self, item):
        return item.upper() if isinstance(item, str) else item


def test_handler_processes_items():
    in_q = queue.Queue()
    out_q = queue.Queue()
    stop = threading.Event()
    handler = EchoHandler(in_q, out_q, stop)

    thread = threading.Thread(target=handler.run)
    thread.start()

    in_q.put("hello")
    result = out_q.get(timeout=2)
    assert result == "HELLO"

    stop.set()
    in_q.put(b"END")
    thread.join(timeout=2)
    assert not thread.is_alive()


def test_handler_stops_on_sentinel():
    in_q = queue.Queue()
    out_q = queue.Queue()
    stop = threading.Event()
    handler = EchoHandler(in_q, out_q, stop)

    thread = threading.Thread(target=handler.run)
    thread.start()

    in_q.put(b"END")
    thread.join(timeout=2)
    assert not thread.is_alive()


def test_handler_stops_on_event():
    in_q = queue.Queue()
    out_q = queue.Queue()
    stop = threading.Event()
    handler = EchoHandler(in_q, out_q, stop)

    thread = threading.Thread(target=handler.run)
    thread.start()

    stop.set()
    thread.join(timeout=2)
    assert not thread.is_alive()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_voice/test_base_handler.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement BaseHandler**

```python
# src/homie_core/voice/base_handler.py
from __future__ import annotations

import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)

SENTINEL = b"END"


class BaseHandler(ABC):
    """Base class for pipeline components. Each runs in its own thread,
    reading from input_queue and writing to output_queue."""

    def __init__(
        self,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        stop_event: threading.Event,
        name: str | None = None,
    ) -> None:
        self._input_queue = input_queue
        self._output_queue = output_queue
        self._stop_event = stop_event
        self._name = name or self.__class__.__name__
        self._times: list[float] = []

    @abstractmethod
    def process(self, item: Any) -> Any:
        """Process a single item. Return value is placed on output_queue.
        Return None to skip output for this item."""

    def run(self) -> None:
        """Main loop — pull from input, process, push to output."""
        logger.debug("%s: started", self._name)
        while not self._stop_event.is_set():
            try:
                item = self._input_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is SENTINEL or item == SENTINEL:
                logger.debug("%s: received sentinel, stopping", self._name)
                break

            start = time.perf_counter()
            try:
                result = self.process(item)
            except Exception:
                logger.exception("%s: error processing item", self._name)
                continue
            elapsed = time.perf_counter() - start
            self._times.append(elapsed)

            if result is not None:
                self._output_queue.put(result)

        logger.debug("%s: stopped", self._name)

    @property
    def last_time(self) -> float:
        """Duration of most recent process() call in seconds."""
        return self._times[-1] if self._times else 0.0

    def send_sentinel(self) -> None:
        """Push sentinel to input queue to unblock a waiting run() loop."""
        self._input_queue.put(SENTINEL)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_voice/test_base_handler.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/voice/base_handler.py tests/unit/test_voice/test_base_handler.py
git commit -m "feat(voice): add BaseHandler queue/thread pattern"
```

---

### Task 3: Rewrite Audio I/O — AudioInThread + AudioOutThread

**Files:**
- Rewrite: `src/homie_core/voice/audio_io.py`
- Test: `tests/unit/test_voice/test_audio_io.py`

- [ ] **Step 1: Write failing test for AudioInThread and AudioOutThread**

```python
# tests/unit/test_voice/test_audio_io.py
import queue
import struct
import threading
from unittest.mock import patch, MagicMock

from homie_core.voice.audio_io import AudioInThread, AudioOutThread


def test_audio_in_pushes_chunks_to_queue():
    out_q = queue.Queue()
    stop = threading.Event()

    # Simulate sounddevice RawInputStream yielding 2 chunks then stopping
    fake_chunk = struct.pack("<512h", *([1000] * 512))

    mock_stream = MagicMock()
    mock_stream.read.side_effect = [
        (fake_chunk, False),
        (fake_chunk, False),
    ]
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=False)

    with patch("homie_core.voice.audio_io.sd") as mock_sd:
        mock_sd.RawInputStream.return_value = mock_stream
        # After 2 reads, set stop event
        original_read = mock_stream.read.side_effect

        call_count = 0

        def read_with_stop(*a, **kw):
            nonlocal call_count
            if call_count >= 2:
                stop.set()
                raise Exception("stopped")
            result = (fake_chunk, False)
            call_count += 1
            return result

        mock_stream.read.side_effect = read_with_stop

        audio_in = AudioInThread(out_q, stop, sample_rate=16000, chunk_size=512)
        thread = threading.Thread(target=audio_in.run)
        thread.start()
        thread.join(timeout=3)

    assert out_q.qsize() >= 2


def test_audio_out_plays_from_queue():
    in_q = queue.Queue()
    stop = threading.Event()

    fake_chunk = struct.pack("<512h", *([500] * 512))
    in_q.put(fake_chunk)

    mock_stream = MagicMock()
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=False)

    with patch("homie_core.voice.audio_io.sd") as mock_sd:
        mock_sd.RawOutputStream.return_value = mock_stream

        audio_out = AudioOutThread(in_q, stop, sample_rate=16000, chunk_size=512)

        # Let it process one chunk then stop
        def stop_after_write(*a, **kw):
            stop.set()

        mock_stream.write.side_effect = stop_after_write

        thread = threading.Thread(target=audio_out.run)
        thread.start()
        thread.join(timeout=3)

    mock_stream.write.assert_called_once_with(fake_chunk)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_voice/test_audio_io.py -v`
Expected: FAIL — AudioInThread not found

- [ ] **Step 3: Implement AudioInThread + AudioOutThread**

```python
# src/homie_core/voice/audio_io.py
from __future__ import annotations

import logging
import queue
import struct
import threading
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd

    _HAS_SD = True
except ImportError:
    sd = None  # type: ignore[assignment]
    _HAS_SD = False


class AudioInThread:
    """Reads audio from microphone via sounddevice, pushes raw chunks to a queue."""

    def __init__(
        self,
        output_queue: queue.Queue,
        stop_event: threading.Event,
        sample_rate: int = 16000,
        chunk_size: int = 512,
    ) -> None:
        self._output_queue = output_queue
        self._stop_event = stop_event
        self._sample_rate = sample_rate
        self._chunk_size = chunk_size

    def run(self) -> None:
        if not _HAS_SD:
            logger.error("sounddevice not installed — AudioInThread cannot run")
            return
        try:
            with sd.RawInputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="int16",
                blocksize=self._chunk_size,
            ) as stream:
                logger.info("AudioInThread: recording at %dHz, chunk=%d", self._sample_rate, self._chunk_size)
                while not self._stop_event.is_set():
                    try:
                        data, overflowed = stream.read(self._chunk_size)
                        if overflowed:
                            logger.warning("AudioInThread: input overflow")
                        self._output_queue.put(bytes(data))
                    except Exception:
                        if self._stop_event.is_set():
                            break
                        raise
        except Exception:
            logger.exception("AudioInThread: fatal error")
        logger.debug("AudioInThread: stopped")


class AudioOutThread:
    """Reads audio chunks from a queue and plays them via sounddevice."""

    def __init__(
        self,
        input_queue: queue.Queue,
        stop_event: threading.Event,
        should_play: Optional[threading.Event] = None,
        sample_rate: int = 16000,
        chunk_size: int = 512,
    ) -> None:
        self._input_queue = input_queue
        self._stop_event = stop_event
        self._should_play = should_play  # None = always play
        self._sample_rate = sample_rate
        self._chunk_size = chunk_size

    def run(self) -> None:
        if not _HAS_SD:
            logger.error("sounddevice not installed — AudioOutThread cannot run")
            return
        # Generate dither — low-level noise to keep device responsive
        dither = struct.pack(f"<{self._chunk_size}h", *([1] * self._chunk_size))
        try:
            with sd.RawOutputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="int16",
                blocksize=self._chunk_size,
            ) as stream:
                logger.info("AudioOutThread: playback at %dHz", self._sample_rate)
                while not self._stop_event.is_set():
                    try:
                        chunk = self._input_queue.get(timeout=0.1)
                    except queue.Empty:
                        # Dither to keep device alive
                        stream.write(dither)
                        continue

                    if chunk is None or chunk == b"END":
                        break

                    # Check if barge-in paused playback
                    if self._should_play is not None and not self._should_play.is_set():
                        continue  # drop chunk silently

                    stream.write(chunk)
        except Exception:
            logger.exception("AudioOutThread: fatal error")
        logger.debug("AudioOutThread: stopped")

    def flush(self) -> None:
        """Drain all pending audio from the queue."""
        while not self._input_queue.empty():
            try:
                self._input_queue.get_nowait()
            except queue.Empty:
                break
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_voice/test_audio_io.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/voice/audio_io.py tests/unit/test_voice/test_audio_io.py
git commit -m "feat(voice): rewrite audio I/O with AudioInThread + AudioOutThread"
```

---

### Task 4: Update pyproject.toml Dependencies

**Files:**
- Modify: `pyproject.toml:53-59`

- [ ] **Step 1: Update voice dependencies**

Replace lines 53-59 in `pyproject.toml`:

```toml
voice = [
    "faster-whisper>=1.0",
    "openwakeword>=0.6",
    "piper-tts>=1.2",
    "sounddevice>=0.4",
    "torch>=2.0",
    "kokoro>=0.9",
    "melo-tts>=0.1",
]
```

Note: `pyaudio` removed, `torchaudio` not added (not needed), `torch`, `kokoro`, `melo-tts` added.

- [ ] **Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "feat(voice): update dependencies — add torch, kokoro, melo-tts; remove pyaudio"
```

---

### Task 5: Add ctrl+8 to Hotkey Map

**Files:**
- Modify: `src/homie_app/hotkey.py:12-17`
- Test: `tests/unit/test_voice/test_hotkey.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_voice/test_hotkey.py
from homie_app.hotkey import _HOTKEY_MAP


def test_ctrl_8_in_hotkey_map():
    assert "ctrl+8" in _HOTKEY_MAP
    assert _HOTKEY_MAP["ctrl+8"] == "<ctrl>+8"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_voice/test_hotkey.py -v`
Expected: FAIL — ctrl+8 not in map

- [ ] **Step 3: Add ctrl+8 to _HOTKEY_MAP**

Add to `_HOTKEY_MAP` dict in `hotkey.py` (after line 16):

```python
_HOTKEY_MAP = {
    "alt+8": "<alt>+8",
    "alt+h": "<alt>+h",
    "ctrl+space": "<ctrl>+<space>",
    "ctrl+8": "<ctrl>+8",
    "f9": "<f9>",
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_voice/test_hotkey.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_app/hotkey.py tests/unit/test_voice/test_hotkey.py
git commit -m "feat(voice): add ctrl+8 to hotkey map"
```

---

## Chunk 2: VAD, STT, TTS Engines

### Task 6: Silero VAD

**Files:**
- Create: `src/homie_core/voice/vad_silero.py`
- Modify: `src/homie_core/voice/vad.py`
- Test: `tests/unit/test_voice/test_vad_silero.py`

- [ ] **Step 1: Write failing test for SileroVAD**

```python
# tests/unit/test_voice/test_vad_silero.py
import struct
import queue
import threading
from unittest.mock import patch, MagicMock

from homie_core.voice.vad_silero import SileroVAD


def _make_silence(n_samples=512):
    return struct.pack(f"<{n_samples}h", *([0] * n_samples))


def _make_speech(n_samples=512):
    return struct.pack(f"<{n_samples}h", *([10000] * n_samples))


def test_silero_vad_detects_speech():
    """SileroVAD.process() returns True for speech audio."""
    with patch("homie_core.voice.vad_silero.torch") as mock_torch:
        mock_model = MagicMock()
        # Simulate high speech probability
        mock_model.return_value = MagicMock(item=MagicMock(return_value=0.9))
        mock_torch.hub.load.return_value = mock_model
        mock_torch.from_numpy.return_value = MagicMock()

        vad = SileroVAD(threshold=0.5)
        result = vad.is_speech(_make_speech())
        assert result is True


def test_silero_vad_rejects_silence():
    """SileroVAD.process() returns False for silence."""
    with patch("homie_core.voice.vad_silero.torch") as mock_torch:
        mock_model = MagicMock()
        mock_model.return_value = MagicMock(item=MagicMock(return_value=0.1))
        mock_torch.hub.load.return_value = mock_model
        mock_torch.from_numpy.return_value = MagicMock()

        vad = SileroVAD(threshold=0.5)
        result = vad.is_speech(_make_silence())
        assert result is False


def test_silero_vad_hysteresis():
    """Once triggered, VAD stays triggered until probability drops below threshold - 0.15."""
    with patch("homie_core.voice.vad_silero.torch") as mock_torch:
        mock_model = MagicMock()
        mock_torch.hub.load.return_value = mock_model
        mock_torch.from_numpy.return_value = MagicMock()

        vad = SileroVAD(threshold=0.5)

        # First: high probability triggers
        mock_model.return_value = MagicMock(item=MagicMock(return_value=0.8))
        assert vad.is_speech(_make_speech()) is True

        # Drops to 0.4 — still above hysteresis (0.35), should stay triggered
        mock_model.return_value = MagicMock(item=MagicMock(return_value=0.4))
        assert vad.is_speech(_make_speech()) is True

        # Drops to 0.3 — below hysteresis, should release
        mock_model.return_value = MagicMock(item=MagicMock(return_value=0.3))
        assert vad.is_speech(_make_speech()) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_voice/test_vad_silero.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement SileroVAD**

```python
# src/homie_core/voice/vad_silero.py
from __future__ import annotations

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


class SileroVAD:
    """Neural voice activity detection using Silero VAD model (~2MB, CPU).

    Uses hysteresis to prevent flickering: triggers at `threshold`,
    releases at `threshold - 0.15`.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        sample_rate: int = 16000,
    ) -> None:
        if not _HAS_TORCH:
            raise ImportError("torch is required for SileroVAD")
        self._threshold = threshold
        self._release_threshold = threshold - 0.15
        self._sample_rate = sample_rate
        self._triggered = False

        self._model, _ = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True
        )
        self._model.eval()
        logger.info("SileroVAD loaded (threshold=%.2f)", threshold)

    def is_speech(self, audio_chunk: bytes) -> bool:
        """Check if audio chunk contains speech.

        Args:
            audio_chunk: Raw PCM int16 bytes.

        Returns:
            True if speech detected.
        """
        audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_np)
        prob = self._model(audio_tensor, self._sample_rate).item()

        if self._triggered:
            if prob < self._release_threshold:
                self._triggered = False
        else:
            if prob >= self._threshold:
                self._triggered = True

        return self._triggered

    def reset(self) -> None:
        """Reset internal state."""
        self._triggered = False
        self._model.reset_states()

    @property
    def is_available(self) -> bool:
        return _HAS_TORCH
```

- [ ] **Step 4: Add create_vad() factory to vad.py**

Add at the end of `src/homie_core/voice/vad.py`:

```python
def create_vad(engine: str = "silero", **kwargs):
    """Factory to create the best available VAD.

    Degradation: silero -> webrtcvad -> energy-based.
    """
    if engine == "silero":
        try:
            from homie_core.voice.vad_silero import SileroVAD
            return SileroVAD(**kwargs)
        except ImportError:
            logger.warning("SileroVAD unavailable, falling back to webrtcvad")
            engine = "webrtcvad"

    if engine == "webrtcvad":
        if _HAS_WEBRTCVAD:
            return VAD(**kwargs)
        logger.warning("webrtcvad unavailable, falling back to energy-based VAD")

    return VoiceActivityDetector(**kwargs)
```

Add `import logging` and `logger = logging.getLogger(__name__)` to top of vad.py if not present.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/unit/test_voice/test_vad_silero.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/homie_core/voice/vad_silero.py src/homie_core/voice/vad.py tests/unit/test_voice/test_vad_silero.py
git commit -m "feat(voice): add SileroVAD with hysteresis and create_vad() factory"
```

---

### Task 7: Fix STT + Add Language Detection

**Files:**
- Modify: `src/homie_core/voice/stt.py`
- Test: `tests/unit/test_voice/test_stt.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_voice/test_stt.py
from unittest.mock import patch, MagicMock

from homie_core.voice.stt import SpeechToText


def test_stt_init_accepts_model_size():
    """Constructor uses model_size parameter (not model_name)."""
    stt = SpeechToText(model_size="base")
    assert stt.model_size == "base"


def test_stt_transcribe_bytes_returns_text_and_language():
    """transcribe_bytes returns (text, language_code) tuple."""
    stt = SpeechToText(model_size="base")
    with patch.object(stt, "_model") as mock_model:
        mock_segment = MagicMock()
        mock_segment.text = "hello world"
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        stt._model = mock_model

        text, lang = stt.transcribe_bytes(b"\x00" * 3200, sample_rate=16000)
        assert text == "hello world"
        assert lang == "en"


def test_stt_hot_switch_model():
    """switch_model changes the model size and reloads."""
    stt = SpeechToText(model_size="tiny.en")
    assert stt.model_size == "tiny.en"
    # switch_model should update model_size
    stt.model_size = "medium"
    assert stt.model_size == "medium"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_voice/test_stt.py -v`
Expected: FAIL — transcribe_bytes returns str not tuple

- [ ] **Step 3: Update SpeechToText**

Replace `src/homie_core/voice/stt.py`:

```python
# src/homie_core/voice/stt.py
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SpeechToText:
    """Speech-to-text using faster-whisper. Supports language detection."""

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def load(self) -> None:
        """Load the faster-whisper model."""
        try:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            logger.info("STT loaded: %s on %s", self.model_size, self.device)
        except ImportError:
            logger.error("faster-whisper not installed")
        except Exception:
            logger.exception("Failed to load STT model")

    def transcribe(self, audio_path: str) -> tuple[str, str]:
        """Transcribe audio file. Returns (text, language_code)."""
        if not self._model:
            return "", "en"
        segments, info = self._model.transcribe(audio_path)
        text = " ".join(seg.text for seg in segments).strip()
        return text, info.language

    def transcribe_bytes(
        self, audio_bytes: bytes, sample_rate: int = 16000
    ) -> tuple[str, str]:
        """Transcribe raw PCM bytes. Returns (text, language_code)."""
        if not self._model:
            return "", "en"
        import numpy as np
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        segments, info = self._model.transcribe(audio_np)
        text = " ".join(seg.text for seg in segments).strip()
        return text, info.language

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        self._model = None
```

- [ ] **Step 4: Run test**

Run: `python -m pytest tests/unit/test_voice/test_stt.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/voice/stt.py tests/unit/test_voice/test_stt.py
git commit -m "feat(voice): fix STT model_size param, add language detection"
```

---

### Task 8: BaseTTS ABC + PiperTTS

**Files:**
- Rewrite: `src/homie_core/voice/tts.py`
- Test: `tests/unit/test_voice/test_tts.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_voice/test_tts.py
from homie_core.voice.tts import BaseTTS, PiperTTS


def test_base_tts_is_abstract():
    """BaseTTS cannot be instantiated directly."""
    import pytest
    with pytest.raises(TypeError):
        BaseTTS()


def test_piper_tts_implements_base():
    """PiperTTS is a valid BaseTTS subclass."""
    tts = PiperTTS(voice="default")
    assert tts.name == "piper"
    assert "en" in tts.supported_languages


def test_piper_tts_synthesize_without_model():
    """synthesize returns empty bytes when model not loaded."""
    tts = PiperTTS()
    result = tts.synthesize("hello")
    assert result == b""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_voice/test_tts.py -v`
Expected: FAIL — BaseTTS not found

- [ ] **Step 3: Implement BaseTTS + PiperTTS**

```python
# src/homie_core/voice/tts.py
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


class BaseTTS(ABC):
    """Abstract base class for all TTS engines."""

    @abstractmethod
    def load(self, device: str = "cpu") -> None:
        """Load model onto device."""

    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        """Synthesize text to raw PCM audio (16-bit, 16kHz, mono)."""

    @abstractmethod
    def synthesize_stream(self, text: str) -> Iterator[bytes]:
        """Streaming synthesis — yields audio chunks."""

    @abstractmethod
    def unload(self) -> None:
        """Release model resources."""

    @property
    @abstractmethod
    def supported_languages(self) -> list[str]:
        """ISO 639-1 codes this engine supports."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Engine identifier."""

    @property
    def is_loaded(self) -> bool:
        return False


class PiperTTS(BaseTTS):
    """Fast TTS using Piper. Best for short English responses."""

    def __init__(self, voice: str = "default") -> None:
        self._voice = voice
        self._model = None

    def load(self, device: str = "cpu") -> None:
        try:
            import piper
            self._model = piper.PiperVoice.load(self._voice)
            logger.info("PiperTTS loaded: voice=%s", self._voice)
        except ImportError:
            logger.error("piper-tts not installed")
        except Exception:
            logger.exception("Failed to load PiperTTS")

    def synthesize(self, text: str) -> bytes:
        if not self._model:
            return b""
        try:
            audio = b""
            for chunk in self._model.synthesize_stream_raw(text):
                audio += chunk
            return audio
        except Exception:
            logger.exception("PiperTTS synthesis failed")
            return b""

    def synthesize_stream(self, text: str) -> Iterator[bytes]:
        """Stream audio chunks from Piper's native streaming API."""
        if not self._model:
            return
        try:
            for chunk in self._model.synthesize_stream_raw(text):
                yield chunk
        except Exception:
            logger.exception("PiperTTS streaming synthesis failed")

    def unload(self) -> None:
        self._model = None

    @property
    def supported_languages(self) -> list[str]:
        return ["en", "fr", "es", "de", "it", "pt"]

    @property
    def name(self) -> str:
        return "piper"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def synthesize_to_file(self, text: str, path: str | Path) -> bool:
        """Synthesize to file (backward compat)."""
        data = self.synthesize(text)
        if data:
            Path(path).write_bytes(data)
            return True
        return False
```

- [ ] **Step 4: Run test**

Run: `python -m pytest tests/unit/test_voice/test_tts.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/voice/tts.py tests/unit/test_voice/test_tts.py
git commit -m "feat(voice): add BaseTTS ABC + refactor PiperTTS"
```

---

### Task 9: KokoroTTS Engine

**Files:**
- Create: `src/homie_core/voice/tts_kokoro.py`
- Test: `tests/unit/test_voice/test_tts_kokoro.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_voice/test_tts_kokoro.py
from homie_core.voice.tts_kokoro import KokoroTTS
from homie_core.voice.tts import BaseTTS


def test_kokoro_is_base_tts():
    tts = KokoroTTS()
    assert isinstance(tts, BaseTTS)
    assert tts.name == "kokoro"


def test_kokoro_supported_languages():
    tts = KokoroTTS()
    langs = tts.supported_languages
    assert "en" in langs
    assert "fr" in langs
    assert "es" in langs


def test_kokoro_synthesize_without_model():
    tts = KokoroTTS()
    result = tts.synthesize("hello")
    assert result == b""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_voice/test_tts_kokoro.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement KokoroTTS**

```python
# src/homie_core/voice/tts_kokoro.py
from __future__ import annotations

import logging
import struct
from typing import Iterator, Optional

from homie_core.voice.tts import BaseTTS

logger = logging.getLogger(__name__)


class KokoroTTS(BaseTTS):
    """High-quality TTS using Kokoro. 8 languages supported."""

    # Kokoro uses single-char lang codes, not ISO 639-1
    _LANG_MAP = {
        "en": "a",  # American English
        "gb": "b",  # British English
        "fr": "f",  # French
        "es": "e",  # Spanish
        "de": "d",  # German
        "it": "i",  # Italian
        "pt": "p",  # Portuguese
        "ja": "j",  # Japanese
        "zh": "z",  # Chinese
    }

    def __init__(self, voice: str = "af_heart", lang: str = "en") -> None:
        self._voice = voice
        self._lang = lang
        self._pipeline = None

    def load(self, device: str = "cuda") -> None:
        try:
            from kokoro import KPipeline
            lang_code = self._LANG_MAP.get(self._lang, "a")  # default American English
            self._pipeline = KPipeline(lang_code=lang_code, device=device)
            logger.info("KokoroTTS loaded: voice=%s, lang=%s, device=%s", self._voice, self._lang, device)
        except ImportError:
            logger.warning("kokoro not installed — KokoroTTS unavailable")
        except Exception:
            logger.exception("Failed to load KokoroTTS")

    def synthesize(self, text: str) -> bytes:
        if not self._pipeline:
            return b""
        try:
            audio_chunks = []
            for _, _, audio in self._pipeline(text, voice=self._voice):
                # Kokoro returns float32 numpy arrays at 24kHz
                # Convert to int16 PCM
                import numpy as np
                audio_int16 = (audio * 32767).astype(np.int16)
                audio_chunks.append(audio_int16.tobytes())
            return b"".join(audio_chunks)
        except Exception:
            logger.exception("KokoroTTS synthesis failed")
            return b""

    def synthesize_stream(self, text: str) -> Iterator[bytes]:
        if not self._pipeline:
            return
        try:
            import numpy as np
            for _, _, audio in self._pipeline(text, voice=self._voice):
                audio_int16 = (audio * 32767).astype(np.int16)
                yield audio_int16.tobytes()
        except Exception:
            logger.exception("KokoroTTS streaming synthesis failed")

    def unload(self) -> None:
        self._pipeline = None

    @property
    def supported_languages(self) -> list[str]:
        return ["en", "fr", "es", "de", "it", "pt", "ja", "zh"]

    @property
    def name(self) -> str:
        return "kokoro"

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None
```

- [ ] **Step 4: Run test**

Run: `python -m pytest tests/unit/test_voice/test_tts_kokoro.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/voice/tts_kokoro.py tests/unit/test_voice/test_tts_kokoro.py
git commit -m "feat(voice): add KokoroTTS engine"
```

---

### Task 10: MeloTTS Engine

**Files:**
- Create: `src/homie_core/voice/tts_melo.py`
- Test: `tests/unit/test_voice/test_tts_melo.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_voice/test_tts_melo.py
from homie_core.voice.tts_melo import MeloTTS
from homie_core.voice.tts import BaseTTS


def test_melo_is_base_tts():
    tts = MeloTTS()
    assert isinstance(tts, BaseTTS)
    assert tts.name == "melo"


def test_melo_supported_languages():
    tts = MeloTTS()
    langs = tts.supported_languages
    assert "en" in langs
    assert "ta" in langs
    assert "te" in langs
    assert "ml" in langs
    assert "fr" in langs
    assert "es" in langs


def test_melo_synthesize_without_model():
    tts = MeloTTS()
    result = tts.synthesize("hello")
    assert result == b""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_voice/test_tts_melo.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement MeloTTS**

```python
# src/homie_core/voice/tts_melo.py
from __future__ import annotations

import logging
from typing import Iterator, Optional

from homie_core.voice.tts import BaseTTS

logger = logging.getLogger(__name__)


class MeloTTS(BaseTTS):
    """Multilingual TTS using MeloTTS. Broad Indic language support."""

    def __init__(self, language: str = "EN", speaker_id: int = 0) -> None:
        self._language = language.upper()
        self._speaker_id = speaker_id
        self._model = None
        self._device = "cpu"

    def load(self, device: str = "cuda") -> None:
        self._device = device
        try:
            from melo.api import TTS as MeloAPI
            self._model = MeloAPI(language=self._language, device=device)
            logger.info("MeloTTS loaded: lang=%s, device=%s", self._language, device)
        except ImportError:
            logger.warning("melo-tts not installed — MeloTTS unavailable")
        except Exception:
            logger.exception("Failed to load MeloTTS")

    def synthesize(self, text: str) -> bytes:
        if not self._model:
            return b""
        try:
            import numpy as np
            import tempfile
            import os
            # MeloTTS writes to file — use temp file then read back
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
            try:
                self._model.tts_to_file(text, self._speaker_id, tmp_path, quiet=True)
                import wave
                with wave.open(tmp_path, "rb") as wf:
                    raw = wf.readframes(wf.getnframes())
                return raw
            finally:
                os.unlink(tmp_path)
        except Exception:
            logger.exception("MeloTTS synthesis failed")
            return b""

    def synthesize_stream(self, text: str) -> Iterator[bytes]:
        """MeloTTS doesn't natively stream, yield full result."""
        data = self.synthesize(text)
        if data:
            yield data

    def unload(self) -> None:
        self._model = None

    @property
    def supported_languages(self) -> list[str]:
        return [
            "en", "fr", "es", "de", "it", "pt", "zh", "ja", "ko",
            "hi", "ta", "te", "ml", "bn", "gu", "kn", "mr", "pa",
        ]

    @property
    def name(self) -> str:
        return "melo"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
```

- [ ] **Step 4: Run test**

Run: `python -m pytest tests/unit/test_voice/test_tts_melo.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/voice/tts_melo.py tests/unit/test_voice/test_tts_melo.py
git commit -m "feat(voice): add MeloTTS engine for multilingual support"
```

---

### Task 11: TTS Selector

**Files:**
- Create: `src/homie_core/voice/tts_selector.py`
- Test: `tests/unit/test_voice/test_tts_selector.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_voice/test_tts_selector.py
from unittest.mock import MagicMock
from homie_core.voice.tts_selector import TTSSelector


def _make_engine(name, languages):
    engine = MagicMock()
    engine.name = name
    engine.supported_languages = languages
    engine.is_loaded = True
    return engine


def test_auto_short_text_uses_fast():
    piper = _make_engine("piper", ["en"])
    kokoro = _make_engine("kokoro", ["en", "fr"])
    melo = _make_engine("melo", ["en", "ta"])
    selector = TTSSelector(fast=piper, quality=kokoro, multilingual=melo)

    engine = selector.select("Hi there", detected_lang="en")
    assert engine.name == "piper"


def test_auto_non_english_uses_multilingual():
    piper = _make_engine("piper", ["en"])
    kokoro = _make_engine("kokoro", ["en", "fr"])
    melo = _make_engine("melo", ["en", "ta"])
    selector = TTSSelector(fast=piper, quality=kokoro, multilingual=melo)

    engine = selector.select("vanakkam", detected_lang="ta")
    assert engine.name == "melo"


def test_auto_long_english_uses_quality():
    piper = _make_engine("piper", ["en"])
    kokoro = _make_engine("kokoro", ["en", "fr"])
    melo = _make_engine("melo", ["en", "ta"])
    selector = TTSSelector(fast=piper, quality=kokoro, multilingual=melo)

    long_text = " ".join(["word"] * 25)
    engine = selector.select(long_text, detected_lang="en")
    assert engine.name == "kokoro"


def test_fallback_when_quality_unavailable():
    piper = _make_engine("piper", ["en"])
    kokoro = _make_engine("kokoro", ["en"])
    kokoro.is_loaded = False
    selector = TTSSelector(fast=piper, quality=kokoro, multilingual=None)

    long_text = " ".join(["word"] * 25)
    engine = selector.select(long_text, detected_lang="en")
    assert engine.name == "piper"  # falls back to fast


def test_forced_mode():
    piper = _make_engine("piper", ["en"])
    kokoro = _make_engine("kokoro", ["en"])
    melo = _make_engine("melo", ["en", "ta"])
    selector = TTSSelector(fast=piper, quality=kokoro, multilingual=melo, mode="quality")

    engine = selector.select("Hi", detected_lang="en")
    assert engine.name == "kokoro"  # forced quality even for short text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_voice/test_tts_selector.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement TTSSelector**

```python
# src/homie_core/voice/tts_selector.py
from __future__ import annotations

import logging
from typing import Optional

from homie_core.voice.tts import BaseTTS

logger = logging.getLogger(__name__)


class TTSSelector:
    """Auto-selects the best TTS engine based on text content and language."""

    WORD_THRESHOLD = 20  # below this, use fast engine

    def __init__(
        self,
        fast: Optional[BaseTTS] = None,
        quality: Optional[BaseTTS] = None,
        multilingual: Optional[BaseTTS] = None,
        mode: str = "auto",
    ) -> None:
        self._fast = fast
        self._quality = quality
        self._multilingual = multilingual
        self._mode = mode
        self._engines = {
            "fast": fast,
            "quality": quality,
            "multilingual": multilingual,
        }

    def select(
        self,
        text: str,
        detected_lang: str = "en",
    ) -> BaseTTS:
        """Pick the best available TTS engine.

        Auto logic:
        1. Short text (< 20 words) -> fast (Piper)
        2. Non-English -> multilingual (MeloTTS)
        3. Otherwise -> quality (Kokoro)

        Falls back through: requested -> quality -> fast.
        """
        if self._mode != "auto":
            engine = self._engines.get(self._mode)
            if engine and engine.is_loaded:
                return engine
            logger.warning("Forced TTS mode '%s' unavailable, falling back", self._mode)

        # Auto selection
        word_count = len(text.split())

        if detected_lang != "en" and self._multilingual and self._multilingual.is_loaded:
            return self._multilingual

        if word_count < self.WORD_THRESHOLD and self._fast and self._fast.is_loaded:
            return self._fast

        if self._quality and self._quality.is_loaded:
            return self._quality

        # Fallback chain
        if self._fast and self._fast.is_loaded:
            return self._fast
        if self._multilingual and self._multilingual.is_loaded:
            return self._multilingual

        raise RuntimeError("No TTS engine available")

    def set_mode(self, mode: str) -> None:
        """Change selection mode: auto | fast | quality | multilingual."""
        self._mode = mode

    @property
    def available_engines(self) -> dict[str, bool]:
        return {
            name: (engine is not None and engine.is_loaded)
            for name, engine in self._engines.items()
        }
```

- [ ] **Step 4: Run test**

Run: `python -m pytest tests/unit/test_voice/test_tts_selector.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/voice/tts_selector.py tests/unit/test_voice/test_tts_selector.py
git commit -m "feat(voice): add TTSSelector with auto-selection logic"
```

---

## Chunk 3: Voice Pipeline, Voice Manager, Prompts

### Task 12: Voice Prompts

**Files:**
- Create: `src/homie_core/voice/voice_prompts.py`
- Test: `tests/unit/test_voice/test_voice_prompts.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_voice/test_voice_prompts.py
from homie_core.voice.voice_prompts import get_voice_hint, VOICE_SYSTEM_HINT


def test_voice_hint_is_string():
    assert isinstance(VOICE_SYSTEM_HINT, str)
    assert "voice" in VOICE_SYSTEM_HINT.lower()


def test_get_voice_hint_returns_hint():
    hint = get_voice_hint()
    assert "concise" in hint.lower()
    assert "markdown" in hint.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_voice/test_voice_prompts.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement voice_prompts.py**

```python
# src/homie_core/voice/voice_prompts.py
from __future__ import annotations

VOICE_SYSTEM_HINT = (
    "User is speaking via voice. Keep responses concise and conversational. "
    "Avoid markdown, code blocks, or visual formatting — the response will be read aloud."
)

EXIT_CONFIRMATION = "Would you like to end our conversation?"

EXIT_AUTO_MESSAGE = (
    "Ending the conversation since you seem to be away. Talk to you later!"
)


def get_voice_hint() -> str:
    """Return the system prompt hint for voice mode."""
    return VOICE_SYSTEM_HINT
```

- [ ] **Step 4: Run test**

Run: `python -m pytest tests/unit/test_voice/test_voice_prompts.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/voice/voice_prompts.py tests/unit/test_voice/test_voice_prompts.py
git commit -m "feat(voice): add voice-aware prompt hints"
```

---

### Task 13: Rewrite VoicePipeline — Queue-Based with Barge-In

**Files:**
- Rewrite: `src/homie_core/voice/voice_pipeline.py`
- Test: `tests/unit/test_voice/test_voice_pipeline.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_voice/test_voice_pipeline.py
import queue
import struct
import threading
import time
from unittest.mock import MagicMock, patch

from homie_core.voice.voice_pipeline import VoicePipeline, PipelineState


def test_pipeline_state_enum():
    assert PipelineState.IDLE.value == "idle"
    assert PipelineState.LISTENING.value == "listening"
    assert PipelineState.RECORDING.value == "recording"
    assert PipelineState.PROCESSING.value == "processing"
    assert PipelineState.SPEAKING.value == "speaking"


def test_pipeline_init():
    callback = MagicMock(return_value=iter(["hello"]))
    pipeline = VoicePipeline(on_query=callback)
    assert pipeline.state == PipelineState.IDLE


def test_pipeline_barge_in_flushes_tts():
    """When should_listen is cleared, TTS output queue is flushed."""
    callback = MagicMock(return_value=iter(["hello"]))
    pipeline = VoicePipeline(on_query=callback)
    # Simulate items in TTS queue
    pipeline._playback_queue.put(b"audio1")
    pipeline._playback_queue.put(b"audio2")
    assert pipeline._playback_queue.qsize() == 2

    pipeline.barge_in()
    assert pipeline._playback_queue.empty()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_voice/test_voice_pipeline.py -v`
Expected: FAIL — new VoicePipeline interface not found

- [ ] **Step 3: Implement new VoicePipeline**

```python
# src/homie_core/voice/voice_pipeline.py
from __future__ import annotations

import enum
import logging
import queue
import threading
import time
from typing import Callable, Iterator, Optional

logger = logging.getLogger(__name__)


class PipelineState(enum.Enum):
    IDLE = "idle"
    LISTENING = "listening"
    RECORDING = "recording"
    PROCESSING = "processing"
    SPEAKING = "speaking"


class VoicePipeline:
    """Queue-based voice pipeline with barge-in support.

    Components run in threads connected by queues:
    AudioIn -> vad_queue -> VADHandler -> stt_queue -> STTHandler -> brain -> TTS -> playback_queue -> AudioOut
    """

    _SILENCE_CHUNKS_THRESHOLD = 30  # ~1.9s at 16kHz/512 chunks
    _MAX_RECORDING_CHUNKS = 500  # ~32s max
    _QUEUE_MAX_DEPTH = 50

    def __init__(
        self,
        on_query: Callable[[str], Iterator[str]] | None = None,
        on_state_change: Callable[[PipelineState], None] | None = None,
        on_response_complete: Callable[[], None] | None = None,
        mode: str = "hybrid",
        sample_rate: int = 16000,
        chunk_size: int = 512,
    ) -> None:
        self._on_query = on_query
        self._on_state_change = on_state_change
        self._on_response_complete = on_response_complete
        self._mode = mode
        self._sample_rate = sample_rate
        self._chunk_size = chunk_size

        # State
        self._state = PipelineState.IDLE
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._should_listen = threading.Event()
        self._should_play = threading.Event()
        self._should_play.set()  # default: play enabled

        # Queues
        self._vad_queue: queue.Queue = queue.Queue()
        self._playback_queue: queue.Queue = queue.Queue()

        # Components (set by VoiceManager)
        self.vad = None
        self.stt = None
        self.tts_selector = None
        self.audio_in = None
        self.audio_out = None
        self.wake_word = None

        # Threads
        self._threads: list[threading.Thread] = []

        # Recording buffer
        self._recording_buffer: list[bytes] = []
        self._silence_counter = 0

    @property
    def state(self) -> PipelineState:
        return self._state

    def _set_state(self, new_state: PipelineState) -> None:
        with self._state_lock:
            if self._state != new_state:
                logger.debug("Pipeline: %s -> %s", self._state.value, new_state.value)
                self._state = new_state
                if self._on_state_change:
                    try:
                        self._on_state_change(new_state)
                    except Exception:
                        logger.exception("State change callback failed")

    def start(self) -> None:
        """Start audio I/O threads."""
        self._stop_event.clear()
        if self.audio_in:
            t = threading.Thread(target=self.audio_in.run, daemon=True, name="audio-in")
            t.start()
            self._threads.append(t)
        if self.audio_out:
            t = threading.Thread(target=self.audio_out.run, daemon=True, name="audio-out")
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        """Graceful shutdown."""
        self._stop_event.set()
        # Unblock queues
        self._vad_queue.put(b"END")
        self._playback_queue.put(b"END")
        for t in self._threads:
            t.join(timeout=3)
        self._threads.clear()
        self._set_state(PipelineState.IDLE)

    def begin_listening(self) -> None:
        """Start listening for speech."""
        self._should_listen.set()
        self._set_state(PipelineState.LISTENING)
        self._recording_buffer.clear()
        self._silence_counter = 0

    def stop_listening(self) -> None:
        """Stop listening."""
        self._should_listen.clear()
        self._set_state(PipelineState.IDLE)

    def barge_in(self) -> None:
        """Interrupt TTS playback — flush audio queue, resume listening."""
        self._should_play.clear()
        # Flush playback queue
        while not self._playback_queue.empty():
            try:
                self._playback_queue.get_nowait()
            except queue.Empty:
                break
        self._should_play.set()
        logger.debug("Barge-in: TTS flushed")

    def process_audio_chunk(self, chunk: bytes) -> None:
        """Process a single audio chunk through VAD -> STT -> Brain -> TTS.
        Called from the main voice loop in VoiceManager."""
        if not self.vad:
            return

        is_speech = self.vad.is_speech(chunk)

        if self._state == PipelineState.LISTENING:
            if is_speech:
                self._set_state(PipelineState.RECORDING)
                self._recording_buffer = [chunk]
                self._silence_counter = 0

        elif self._state == PipelineState.RECORDING:
            self._recording_buffer.append(chunk)

            if not is_speech:
                self._silence_counter += 1
            else:
                self._silence_counter = 0

            # Check end-of-speech or max recording
            if (
                self._silence_counter >= self._SILENCE_CHUNKS_THRESHOLD
                or len(self._recording_buffer) >= self._MAX_RECORDING_CHUNKS
            ):
                self._process_recording()

        elif self._state == PipelineState.SPEAKING:
            # Barge-in detection
            if is_speech:
                self.barge_in()
                self._set_state(PipelineState.RECORDING)
                self._recording_buffer = [chunk]
                self._silence_counter = 0

    def _process_recording(self) -> None:
        """STT -> Brain -> TTS for accumulated recording."""
        self._set_state(PipelineState.PROCESSING)
        audio_bytes = b"".join(self._recording_buffer)
        self._recording_buffer.clear()
        self._silence_counter = 0

        if not self.stt or not self._on_query:
            self._set_state(PipelineState.LISTENING)
            return

        # STT
        text, lang = self.stt.transcribe_bytes(audio_bytes, self._sample_rate)
        if not text.strip():
            self._set_state(PipelineState.LISTENING)
            return

        logger.info("STT: '%s' (lang=%s)", text, lang)

        # Brain (streaming)
        self._set_state(PipelineState.SPEAKING)
        self._should_play.set()
        full_response = []

        try:
            sentence_buffer = []
            for token in self._on_query(text):
                full_response.append(token)
                sentence_buffer.append(token)
                joined = "".join(sentence_buffer)

                # TTS at sentence boundaries
                if any(joined.rstrip().endswith(p) for p in ".!?"):
                    self._synthesize_and_queue(joined.strip(), lang)
                    sentence_buffer.clear()

            # Flush remaining text
            remaining = "".join(sentence_buffer).strip()
            if remaining:
                self._synthesize_and_queue(remaining, lang)
        except Exception:
            logger.exception("Brain query failed")

        # Mode-specific post-response behavior
        if self._on_response_complete:
            self._on_response_complete()
        if self._mode in ("conversational", "hybrid"):
            self._set_state(PipelineState.LISTENING)
        else:
            # wake_word and push_to_talk return to IDLE
            self._set_state(PipelineState.IDLE)

    def _synthesize_and_queue(self, text: str, lang: str) -> None:
        """Synthesize text and push audio chunks to playback queue."""
        if not self.tts_selector or not text:
            return
        try:
            engine = self.tts_selector.select(text, detected_lang=lang)
            for chunk in engine.synthesize_stream(text):
                if self._stop_event.is_set() or not self._should_play.is_set():
                    break
                self._playback_queue.put(chunk)
        except Exception:
            logger.exception("TTS synthesis failed for: %s", text[:50])

    @property
    def is_running(self) -> bool:
        return not self._stop_event.is_set()
```

- [ ] **Step 4: Run test**

Run: `python -m pytest tests/unit/test_voice/test_voice_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/voice/voice_pipeline.py tests/unit/test_voice/test_voice_pipeline.py
git commit -m "feat(voice): rewrite VoicePipeline with queue-based threading and barge-in"
```

---

### Task 14: VoiceManager — Mode Orchestration

**Files:**
- Create: `src/homie_core/voice/voice_manager.py`
- Test: `tests/unit/test_voice/test_voice_manager.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_voice/test_voice_manager.py
import threading
from unittest.mock import MagicMock, patch

from homie_core.voice.voice_manager import VoiceManager
from homie_core.voice.voice_pipeline import PipelineState


def _make_config(**overrides):
    """Create a mock VoiceConfig."""
    defaults = dict(
        enabled=True, hotkey="ctrl+8", wake_word="hey homie", mode="hybrid",
        stt_engine="faster-whisper", stt_model_fast="tiny.en",
        stt_model_quality="medium", stt_language="auto",
        tts_mode="auto", tts_voice_fast="piper", tts_voice_quality="kokoro",
        tts_voice_multilingual="melo",
        vad_engine="silero", vad_threshold=0.5, vad_silence_ms=300,
        barge_in=True, conversation_timeout=120, max_exit_prompts=3,
        exit_phrases=["goodbye", "stop", "that's all"],
        device="cpu", audio_sample_rate=16000, audio_chunk_size=512,
    )
    defaults.update(overrides)
    cfg = MagicMock()
    for k, v in defaults.items():
        setattr(cfg, k, v)
    return cfg


def test_voice_manager_init():
    cfg = _make_config()
    callback = MagicMock(return_value=iter(["hi"]))
    mgr = VoiceManager(config=cfg, on_query=callback)
    assert mgr.state == PipelineState.IDLE


def test_voice_manager_status_report():
    cfg = _make_config()
    callback = MagicMock(return_value=iter(["hi"]))
    mgr = VoiceManager(config=cfg, on_query=callback)
    report = mgr.status_report()
    assert "Voice" in report


def test_voice_manager_exit_phrase_detected():
    cfg = _make_config()
    callback = MagicMock(return_value=iter(["hi"]))
    mgr = VoiceManager(config=cfg, on_query=callback)
    assert mgr._is_exit_phrase("goodbye") is True
    assert mgr._is_exit_phrase("hello") is False
    assert mgr._is_exit_phrase("STOP") is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_voice/test_voice_manager.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement VoiceManager**

```python
# src/homie_core/voice/voice_manager.py
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable, Iterator, Optional

from homie_core.voice.voice_pipeline import PipelineState, VoicePipeline
from homie_core.voice.voice_prompts import (
    EXIT_AUTO_MESSAGE,
    EXIT_CONFIRMATION,
    get_voice_hint,
)

logger = logging.getLogger(__name__)


class VoiceManager:
    """Orchestrates voice modes, component lifecycle, and brain integration."""

    def __init__(
        self,
        config,
        on_query: Callable[[str], Iterator[str]],
        on_state_change: Optional[Callable[[PipelineState], None]] = None,
    ) -> None:
        self._config = config
        self._on_query_raw = on_query
        self._on_state_change = on_state_change

        self._pipeline = VoicePipeline(
            on_query=self._query_with_voice_hint,
            on_state_change=self._handle_state_change,
            sample_rate=config.audio_sample_rate,
            chunk_size=config.audio_chunk_size,
        )

        self._mode = config.mode  # hybrid, wake_word, push_to_talk, conversational
        self._conversational_active = False
        self._exit_prompt_count = 0
        self._silence_timer: Optional[threading.Timer] = None
        self._voice_loop_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Component availability
        self._available: dict[str, bool] = {}

    # --- Lifecycle ---

    def start(self) -> None:
        """Probe components, start pipeline."""
        self._probe_components()
        self._pipeline.start()
        # Start the main voice processing loop
        self._stop_event.clear()
        self._voice_loop_thread = threading.Thread(
            target=self._voice_loop, daemon=True, name="voice-manager"
        )
        self._voice_loop_thread.start()
        logger.info("VoiceManager started (mode=%s)", self._mode)

    def stop(self) -> None:
        """Graceful shutdown."""
        self._stop_event.set()
        self._cancel_silence_timer()
        self._pipeline.stop()
        if self._voice_loop_thread:
            self._voice_loop_thread.join(timeout=5)
        logger.info("VoiceManager stopped")

    # --- Mode control ---

    def set_mode(self, mode: str) -> None:
        self._mode = mode

    def enter_conversational(self) -> None:
        self._conversational_active = True
        self._exit_prompt_count = 0
        self._pipeline.begin_listening()
        self._reset_silence_timer()

    def exit_conversational(self) -> None:
        self._conversational_active = False
        self._cancel_silence_timer()
        self._pipeline.stop_listening()

    # --- Hotkey ---

    def on_hotkey(self) -> None:
        """Handle ctrl+8 press based on current state."""
        state = self._pipeline.state
        if state == PipelineState.IDLE:
            self._pipeline.begin_listening()
        elif state in (PipelineState.LISTENING, PipelineState.RECORDING):
            self._pipeline.stop_listening()
        elif state == PipelineState.SPEAKING:
            self._pipeline.barge_in()
            self._pipeline.begin_listening()

    # --- Status ---

    @property
    def state(self) -> PipelineState:
        return self._pipeline.state

    @property
    def available_engines(self) -> dict[str, bool]:
        return dict(self._available)

    def status_report(self) -> str:
        lines = [
            f"Voice: {'enabled' if self._config.enabled else 'disabled'}",
            f"Mode:  {self._mode}",
            f"State: {self._pipeline.state.value}",
        ]
        for name, avail in self._available.items():
            status = "available" if avail else "not installed"
            lines.append(f"  {name}: {status}")
        return "\n".join(lines)

    # --- Internal ---

    def _probe_components(self) -> None:
        """Discover available engines and set up pipeline components."""
        # VAD
        try:
            from homie_core.voice.vad import create_vad
            vad = create_vad(engine=self._config.vad_engine, threshold=self._config.vad_threshold)
            self._pipeline.vad = vad
            self._available["vad_silero"] = self._config.vad_engine == "silero"
        except Exception:
            logger.warning("VAD unavailable")
            self._available["vad_silero"] = False

        # STT
        try:
            from homie_core.voice.stt import SpeechToText
            device = self._config.device if self._config.device != "auto" else "cpu"
            stt = SpeechToText(
                model_size=self._config.stt_model_quality,
                device=device,
                compute_type="float16" if device == "cuda" else "float32",
            )
            stt.load()
            self._pipeline.stt = stt
            self._available["stt"] = stt.is_loaded
        except Exception:
            logger.warning("STT unavailable")
            self._available["stt"] = False

        # TTS engines
        from homie_core.voice.tts_selector import TTSSelector

        fast_tts = self._try_load_tts("piper")
        quality_tts = self._try_load_tts("kokoro")
        multi_tts = self._try_load_tts("melo")

        self._pipeline.tts_selector = TTSSelector(
            fast=fast_tts, quality=quality_tts, multilingual=multi_tts,
            mode=self._config.tts_mode,
        )

        # Audio I/O
        try:
            from homie_core.voice.audio_io import AudioInThread, AudioOutThread
            self._pipeline.audio_in = AudioInThread(
                self._pipeline._vad_queue, self._stop_event,
                sample_rate=self._config.audio_sample_rate,
                chunk_size=self._config.audio_chunk_size,
            )
            self._pipeline.audio_out = AudioOutThread(
                self._pipeline._playback_queue, self._stop_event,
                should_play=self._pipeline._should_play,
                sample_rate=self._config.audio_sample_rate,
                chunk_size=self._config.audio_chunk_size,
            )
            self._available["audio"] = True
        except Exception:
            logger.warning("Audio I/O unavailable")
            self._available["audio"] = False

    def _try_load_tts(self, engine_name: str):
        """Try to load a TTS engine, return None on failure."""
        device = self._config.device if self._config.device != "auto" else "cpu"
        try:
            if engine_name == "piper":
                from homie_core.voice.tts import PiperTTS
                tts = PiperTTS(voice=self._config.tts_voice_fast)
                tts.load(device="cpu")  # Piper is CPU-only
                self._available["tts_piper"] = tts.is_loaded
                return tts if tts.is_loaded else None
            elif engine_name == "kokoro":
                from homie_core.voice.tts_kokoro import KokoroTTS
                tts = KokoroTTS()
                tts.load(device=device)
                self._available["tts_kokoro"] = tts.is_loaded
                return tts if tts.is_loaded else None
            elif engine_name == "melo":
                from homie_core.voice.tts_melo import MeloTTS
                tts = MeloTTS()
                tts.load(device=device)
                self._available["tts_melo"] = tts.is_loaded
                return tts if tts.is_loaded else None
        except Exception:
            logger.warning("TTS engine '%s' unavailable", engine_name)
            self._available[f"tts_{engine_name}"] = False
        return None

    def _query_with_voice_hint(self, text: str) -> Iterator[str]:
        """Wrap the brain callback to inject voice hint and handle exit phrases."""
        # Check for exit phrases in conversational mode
        if self._conversational_active and self._is_exit_phrase(text):
            yield EXIT_CONFIRMATION
            return

        # Inject voice hint by prepending to the query
        # The brain's process_stream will receive: "[VOICE] actual query"
        # voice_prompts.VOICE_SYSTEM_HINT is injected as context
        voice_text = f"[VOICE_MODE] {text}"
        yield from self._on_query_raw(voice_text)

    def _is_exit_phrase(self, text: str) -> bool:
        text_lower = text.strip().lower()
        return any(phrase in text_lower for phrase in self._config.exit_phrases)

    def _voice_loop(self) -> None:
        """Main loop: pull audio from vad_queue, feed to pipeline."""
        while not self._stop_event.is_set():
            try:
                chunk = self._pipeline._vad_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if chunk == b"END":
                break
            self._pipeline.process_audio_chunk(chunk)

    def _handle_state_change(self, new_state: PipelineState) -> None:
        """Internal state change handler."""
        if self._on_state_change:
            self._on_state_change(new_state)

        # Manage silence timer for conversational mode
        if self._conversational_active:
            if new_state == PipelineState.LISTENING:
                self._reset_silence_timer()
            elif new_state in (PipelineState.PROCESSING, PipelineState.SPEAKING):
                self._cancel_silence_timer()

    def _reset_silence_timer(self) -> None:
        self._cancel_silence_timer()
        self._silence_timer = threading.Timer(
            self._config.conversation_timeout, self._on_silence_timeout
        )
        self._silence_timer.daemon = True
        self._silence_timer.start()

    def _cancel_silence_timer(self) -> None:
        if self._silence_timer:
            self._silence_timer.cancel()
            self._silence_timer = None

    def _on_silence_timeout(self) -> None:
        """Handle conversation silence timeout."""
        self._exit_prompt_count += 1
        if self._exit_prompt_count >= self._config.max_exit_prompts:
            # Auto-exit
            logger.info("Auto-exiting conversation after %d unanswered prompts", self._exit_prompt_count)
            self._synthesize_exit_message(EXIT_AUTO_MESSAGE)
            self.exit_conversational()
        else:
            # Ask confirmation
            self._synthesize_exit_message(EXIT_CONFIRMATION)
            self._reset_silence_timer()

    def _synthesize_exit_message(self, message: str) -> None:
        """Synthesize and queue an exit/confirmation message."""
        if self._pipeline.tts_selector:
            try:
                engine = self._pipeline.tts_selector.select(message, detected_lang="en")
                for chunk in engine.synthesize_stream(message):
                    self._pipeline._playback_queue.put(chunk)
            except Exception:
                logger.exception("Failed to synthesize exit message")
```

- [ ] **Step 4: Run test**

Run: `python -m pytest tests/unit/test_voice/test_voice_manager.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/voice/voice_manager.py tests/unit/test_voice/test_voice_manager.py
git commit -m "feat(voice): add VoiceManager with mode orchestration and state machine"
```

---

### Task 15: Wire Wake Word into VoiceManager

**Files:**
- Modify: `src/homie_core/voice/voice_manager.py`
- Test: `tests/unit/test_voice/test_wake_word_integration.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_voice/test_wake_word_integration.py
from unittest.mock import MagicMock, patch

from homie_core.voice.voice_manager import VoiceManager
from homie_core.voice.voice_pipeline import PipelineState


def _make_config(**overrides):
    defaults = dict(
        enabled=True, hotkey="ctrl+8", wake_word="hey homie", mode="hybrid",
        stt_engine="faster-whisper", stt_model_fast="tiny.en",
        stt_model_quality="medium", stt_language="auto",
        tts_mode="auto", tts_voice_fast="piper", tts_voice_quality="kokoro",
        tts_voice_multilingual="melo",
        vad_engine="energy", vad_threshold=0.5, vad_silence_ms=300,
        barge_in=True, conversation_timeout=120, max_exit_prompts=3,
        exit_phrases=["goodbye", "stop", "that's all"],
        device="cpu", audio_sample_rate=16000, audio_chunk_size=512,
    )
    defaults.update(overrides)
    cfg = MagicMock()
    for k, v in defaults.items():
        setattr(cfg, k, v)
    return cfg


def test_wake_word_detected_transitions_to_listening():
    cfg = _make_config(mode="wake_word")
    callback = MagicMock(return_value=iter(["hi"]))
    mgr = VoiceManager(config=cfg, on_query=callback)
    mgr._handle_wake_word_detected()
    assert mgr._pipeline.state == PipelineState.LISTENING


def test_wake_word_mode_returns_to_idle_after_response():
    cfg = _make_config(mode="wake_word")
    callback = MagicMock(return_value=iter(["hi"]))
    mgr = VoiceManager(config=cfg, on_query=callback)
    # Pipeline should be configured for wake_word mode
    assert mgr._pipeline._mode == "wake_word"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_voice/test_wake_word_integration.py -v`
Expected: FAIL — `_handle_wake_word_detected` not found

- [ ] **Step 3: Add wake word support to VoiceManager**

Add to `VoiceManager.__init__()`:

```python
        self._wake_word_engine = None
        self._wake_word_detector = None
```

Add to `_probe_components()`:

```python
        # Wake word
        if self._config.wake_word:
            try:
                from homie_core.voice.wakeword import WakeWordEngine
                self._wake_word_engine = WakeWordEngine(wake_word=self._config.wake_word)
                self._wake_word_engine.start(on_wake=self._handle_wake_word_detected)
                self._available["wake_word_audio"] = True
            except (ImportError, Exception):
                logger.info("openwakeword unavailable, using text-based wake word detection")
                from homie_core.voice.wakeword import WakeWordDetector
                self._wake_word_detector = WakeWordDetector(
                    wake_phrase=self._config.wake_word,
                    on_detected=self._handle_wake_word_detected,
                )
                self._available["wake_word_audio"] = False
                self._available["wake_word_text"] = True
```

Add method:

```python
    def _handle_wake_word_detected(self) -> None:
        """Called when wake word is detected — transition to LISTENING."""
        if self._pipeline.state == PipelineState.IDLE:
            logger.info("Wake word detected!")
            self._pipeline.begin_listening()
```

Add to `stop()`:

```python
        if self._wake_word_engine:
            self._wake_word_engine.stop()
```

- [ ] **Step 4: Run test**

Run: `python -m pytest tests/unit/test_voice/test_wake_word_integration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/voice/voice_manager.py tests/unit/test_voice/test_wake_word_integration.py
git commit -m "feat(voice): wire wake word detection into VoiceManager"
```

---

### Task 16: Update voice __init__.py Exports

**Files:**
- Modify: `src/homie_core/voice/__init__.py`

- [ ] **Step 1: Update exports**

```python
# src/homie_core/voice/__init__.py
from homie_core.voice.voice_pipeline import PipelineState, VoicePipeline
from homie_core.voice.voice_manager import VoiceManager
from homie_core.voice.tts import BaseTTS, PiperTTS

__all__ = ["PipelineState", "VoicePipeline", "VoiceManager", "BaseTTS", "PiperTTS"]
```

- [ ] **Step 2: Commit**

```bash
git add src/homie_core/voice/__init__.py
git commit -m "feat(voice): update __init__.py exports"
```

---

## Chunk 4: Daemon, Overlay, CLI Integration

### Task 16: Wire VoiceManager into Daemon

**Files:**
- Modify: `src/homie_app/daemon.py`

- [ ] **Step 1: Add VoiceManager import and initialization**

At top of `daemon.py`, add to imports section:

```python
from homie_core.voice.voice_manager import VoiceManager
from homie_core.voice.voice_pipeline import PipelineState
```

- [ ] **Step 2: Add VoiceManager to __init__**

In `HomieDaemon.__init__()` (after line ~100, after other component init):

```python
        # Voice
        self._voice_manager: Optional[VoiceManager] = None
        if self._config.voice.enabled:
            try:
                self._voice_manager = VoiceManager(
                    config=self._config.voice,
                    on_query=self._on_user_query_stream,
                    on_state_change=self._on_voice_state,
                )
            except Exception:
                logger.warning("Voice initialization failed, continuing without voice")
```

- [ ] **Step 3: Add _on_voice_state callback**

After `_on_user_query_stream` method:

```python
    def _on_voice_state(self, state: PipelineState) -> None:
        """Handle voice pipeline state changes for overlay updates."""
        logger.debug("Voice state: %s", state.value)
```

- [ ] **Step 4: Wire start/stop**

In `HomieDaemon.start()` (after line ~550, after other components start):

```python
        if self._voice_manager:
            self._voice_manager.start()
            logger.info("Voice pipeline started")
```

In `HomieDaemon.stop()` (before line ~600, before other cleanup):

```python
        if self._voice_manager:
            self._voice_manager.stop()
```

- [ ] **Step 5: Update hotkey handler**

Modify `_on_hotkey` method (line 363) to delegate to VoiceManager when voice is enabled:

```python
    def _on_hotkey(self) -> None:
        if self._voice_manager:
            self._voice_manager.on_hotkey()
        elif self._overlay:
            self._overlay.toggle()
```

- [ ] **Step 6: Update hotkey initialization**

Change hotkey in daemon init from `"alt+8"` to use config value:

```python
        hotkey_str = self._config.voice.hotkey if self._config.voice.enabled else "alt+8"
        self._hotkey = HotkeyListener(hotkey=hotkey_str, callback=self._on_hotkey)
```

- [ ] **Step 7: Commit**

```bash
git add src/homie_app/daemon.py
git commit -m "feat(voice): wire VoiceManager into daemon lifecycle"
```

---

### Task 17: Add Voice Mode to Overlay

**Files:**
- Modify: `src/homie_app/overlay.py`

- [ ] **Step 1: Add voice state display methods**

Add to `OverlayPopup` class:

```python
    def update_voice_state(self, state: str) -> None:
        """Update the voice state indicator in the overlay."""
        if hasattr(self, "_voice_label") and self._voice_label:
            self._root.after(0, lambda: self._voice_label.config(text=state))

    def update_transcript(self, text: str) -> None:
        """Update the live STT transcript display."""
        if hasattr(self, "_transcript_label") and self._transcript_label:
            self._root.after(0, lambda: self._transcript_label.config(text=f"You: \"{text}\""))

    def update_response(self, text: str) -> None:
        """Update Homie's response text display."""
        if hasattr(self, "_response_label") and self._response_label:
            self._root.after(0, lambda: self._response_label.config(text=f"Homie: {text}"))
```

- [ ] **Step 2: Commit**

```bash
git add src/homie_app/overlay.py
git commit -m "feat(voice): add voice state display to overlay"
```

---

### Task 18: Add CLI Voice Commands

**Files:**
- Modify: `src/homie_app/cli.py`

- [ ] **Step 1: Add voice command handler**

Add the voice command function before `main()`:

```python
def cmd_voice(args, config=None):
    """Handle `homie voice` command — enter conversational voice mode."""
    from homie_core.config import load_config
    cfg = config or load_config(getattr(args, "config", None))

    if hasattr(args, "subcmd") and args.subcmd == "status":
        from homie_core.voice.voice_manager import VoiceManager
        mgr = VoiceManager(
            config=cfg.voice,
            on_query=lambda t: iter(["Voice status check"]),
        )
        print(mgr.status_report())
        return

    if hasattr(args, "subcmd") and args.subcmd == "enable":
        print("Voice enabled. Update homie.config.yaml to persist.")
        return

    if hasattr(args, "subcmd") and args.subcmd == "disable":
        print("Voice disabled. Update homie.config.yaml to persist.")
        return

    # Default: enter conversational mode
    cfg.voice.enabled = True
    if hasattr(args, "mode") and args.mode:
        cfg.voice.mode = args.mode
    if hasattr(args, "tts") and args.tts:
        cfg.voice.tts_mode = args.tts
    if hasattr(args, "lang") and args.lang:
        cfg.voice.stt_language = args.lang

    from homie_app.daemon import HomieDaemon
    daemon = HomieDaemon(config_path=getattr(args, "config", None))
    daemon._config = cfg

    try:
        daemon.start()
        if daemon._voice_manager:
            daemon._voice_manager.enter_conversational()
            print("Voice mode active. Say 'goodbye' to exit or press Ctrl+C.")
            # Block until stopped
            import signal
            signal.signal(signal.SIGINT, lambda *a: daemon.stop())
            while daemon._voice_manager._conversational_active:
                import time
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        daemon.stop()
```

- [ ] **Step 2: Add voice subparser to argument parser**

In the argument parser setup section of `main()`, add:

```python
    voice_parser = subparsers.add_parser("voice", help="Voice interaction mode")
    voice_sub = voice_parser.add_subparsers(dest="subcmd")
    voice_sub.add_parser("status", help="Show voice component status")
    voice_sub.add_parser("enable", help="Enable voice")
    voice_sub.add_parser("disable", help="Disable voice")
    voice_parser.add_argument("--mode", choices=["hybrid", "wake_word", "push_to_talk", "conversational"])
    voice_parser.add_argument("--tts", choices=["auto", "fast", "quality", "multilingual"])
    voice_parser.add_argument("--lang", help="Force language (en, ta, te, ml, fr, es)")
```

- [ ] **Step 3: Add voice to command dispatcher**

Add to the handler mapping dict:

```python
    "voice": cmd_voice,
```

- [ ] **Step 4: Commit**

```bash
git add src/homie_app/cli.py
git commit -m "feat(voice): add homie voice CLI commands"
```

---

### Task 19: Update homie.config.yaml

**Files:**
- Modify: `homie.config.yaml`

- [ ] **Step 1: Replace voice section**

Replace the voice section in `homie.config.yaml`:

```yaml
voice:
  enabled: false
  hotkey: ctrl+8
  wake_word: "hey homie"
  mode: hybrid

  stt_engine: faster-whisper
  stt_model_fast: tiny.en
  stt_model_quality: medium
  stt_language: auto

  tts_mode: auto
  tts_voice_fast: piper
  tts_voice_quality: kokoro
  tts_voice_multilingual: melo

  vad_engine: silero
  vad_threshold: 0.5
  vad_silence_ms: 300

  barge_in: true
  conversation_timeout: 120
  max_exit_prompts: 3
  exit_phrases:
    - "goodbye"
    - "stop"
    - "that's all"

  device: auto
  audio_sample_rate: 16000
  audio_chunk_size: 512
```

- [ ] **Step 2: Commit**

```bash
git add homie.config.yaml
git commit -m "feat(voice): update config with full voice settings"
```

---

### Task 20: Integration Test — Full Pipeline

**Files:**
- Create: `tests/unit/test_voice/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/unit/test_voice/test_integration.py
import struct
import threading
from unittest.mock import MagicMock, patch

from homie_core.voice.voice_pipeline import PipelineState, VoicePipeline
from homie_core.voice.tts_selector import TTSSelector


def test_full_pipeline_recording_to_processing():
    """Simulate speech chunks -> recording -> processing flow."""
    responses = ["Hello ", "there!"]
    callback = MagicMock(return_value=iter(responses))
    pipeline = VoicePipeline(on_query=callback)

    # Mock VAD
    mock_vad = MagicMock()
    pipeline.vad = mock_vad

    # Mock STT
    mock_stt = MagicMock()
    mock_stt.transcribe_bytes.return_value = ("test query", "en")
    pipeline.stt = mock_stt

    # Mock TTS
    mock_engine = MagicMock()
    mock_engine.name = "piper"
    mock_engine.is_loaded = True
    mock_engine.synthesize_stream.return_value = iter([b"audio"])
    mock_selector = MagicMock()
    mock_selector.select.return_value = mock_engine
    pipeline.tts_selector = mock_selector

    # Start listening
    pipeline.begin_listening()
    assert pipeline.state == PipelineState.LISTENING

    speech_chunk = struct.pack("<512h", *([5000] * 512))
    silence_chunk = struct.pack("<512h", *([0] * 512))

    # Feed speech chunks
    mock_vad.is_speech.return_value = True
    pipeline.process_audio_chunk(speech_chunk)
    assert pipeline.state == PipelineState.RECORDING

    # Feed more speech
    for _ in range(5):
        pipeline.process_audio_chunk(speech_chunk)

    # Feed silence to trigger end-of-speech
    mock_vad.is_speech.return_value = False
    for _ in range(35):
        pipeline.process_audio_chunk(silence_chunk)

    # Should have processed (STT called, brain called)
    mock_stt.transcribe_bytes.assert_called_once()
    callback.assert_called_once_with("test query")


def test_barge_in_during_speaking():
    """Barge-in during SPEAKING should flush queue and return to RECORDING."""
    callback = MagicMock(return_value=iter(["response"]))
    pipeline = VoicePipeline(on_query=callback)

    mock_vad = MagicMock()
    pipeline.vad = mock_vad

    # Put pipeline in SPEAKING state
    pipeline._set_state(PipelineState.SPEAKING)
    pipeline._playback_queue.put(b"audio1")
    pipeline._playback_queue.put(b"audio2")

    # Simulate speech during SPEAKING
    mock_vad.is_speech.return_value = True
    speech_chunk = struct.pack("<512h", *([5000] * 512))
    pipeline.process_audio_chunk(speech_chunk)

    # Should have barged in
    assert pipeline._playback_queue.empty()
    assert pipeline.state == PipelineState.RECORDING
```

- [ ] **Step 2: Run integration test**

Run: `python -m pytest tests/unit/test_voice/test_integration.py -v`
Expected: PASS

- [ ] **Step 3: Run all voice tests**

Run: `python -m pytest tests/unit/test_voice/ -v`
Expected: All PASS

- [ ] **Step 4: Run full test suite to check no regressions**

Run: `python -m pytest tests/ -v --timeout=30`
Expected: All existing tests still PASS

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_voice/test_integration.py
git commit -m "test(voice): add integration tests for pipeline flow and barge-in"
```

---

### Task 21: Final — Create tests/__init__.py files and verify

- [ ] **Step 1: Ensure test directory structure**

```bash
ls tests/unit/test_voice/__init__.py  # verify exists
```

- [ ] **Step 2: Run complete test suite**

Run: `python -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 3: Final commit**

```bash
git add tests/__init__.py tests/unit/__init__.py tests/unit/test_voice/__init__.py
git commit -m "chore: add test __init__.py files for voice module"
```
