from __future__ import annotations
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SpeechToText:
    def __init__(self, model_size: str = "large-v3", device: str = "cuda", compute_type: str = "float16") -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def load(self) -> None:
        try:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
            logger.info("STT loaded: %s on %s", self.model_size, self.device)
        except ImportError:
            logger.error("faster-whisper not installed")
        except Exception:
            logger.exception("Failed to load STT model")

    def transcribe(self, audio_path: str) -> tuple[str, str]:
        if not self._model:
            return "", "en"
        segments, info = self._model.transcribe(audio_path)
        text = " ".join(seg.text for seg in segments).strip()
        return text, info.language

    def transcribe_bytes(self, audio_bytes: bytes, sample_rate: int = 16000) -> tuple[str, str]:
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
