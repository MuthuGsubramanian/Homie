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
    sd = None
    _HAS_SD = False


class AudioInThread:
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
        if sd is None:
            logger.error("sounddevice not installed")
            return
        try:
            with sd.RawInputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="int16",
                blocksize=self._chunk_size,
            ) as stream:
                while not self._stop_event.is_set():
                    try:
                        data, overflowed = stream.read(self._chunk_size)
                        if overflowed:
                            logger.warning("AudioInThread: overflow")
                        self._output_queue.put(bytes(data))
                    except Exception:
                        if self._stop_event.is_set():
                            break
                        raise
        except Exception:
            logger.exception("AudioInThread: fatal error")


class AudioOutThread:
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
        self._should_play = should_play
        self._sample_rate = sample_rate
        self._chunk_size = chunk_size

    def run(self) -> None:
        if sd is None:
            logger.error("sounddevice not installed")
            return
        dither = struct.pack(f"<{self._chunk_size}h", *([1] * self._chunk_size))
        try:
            with sd.RawOutputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="int16",
                blocksize=self._chunk_size,
            ) as stream:
                while not self._stop_event.is_set():
                    try:
                        chunk = self._input_queue.get(timeout=0.1)
                    except queue.Empty:
                        stream.write(dither)
                        continue
                    if chunk is None or chunk == b"END":
                        break
                    if self._should_play is not None and not self._should_play.is_set():
                        continue
                    stream.write(chunk)
        except Exception:
            logger.exception("AudioOutThread: fatal error")

    def flush(self) -> None:
        while not self._input_queue.empty():
            try:
                self._input_queue.get_nowait()
            except queue.Empty:
                break
