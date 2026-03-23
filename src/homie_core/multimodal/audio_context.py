from __future__ import annotations

import logging
import math
import struct
from typing import Sequence

logger = logging.getLogger(__name__)

# Default parameters — 16-bit mono PCM @ 16 kHz
_DEFAULT_SAMPLE_RATE = 16000
_SAMPLE_WIDTH = 2  # bytes per sample (16-bit)


class AudioContextEngine:
    """Understands audio context — classification, speech detection, noise level.

    All analysis uses basic math (energy, zero-crossing rate) on raw 16-bit
    mono PCM bytes.  No ML libraries required.
    """

    def __init__(self, sample_rate: int = _DEFAULT_SAMPLE_RATE):
        self._sample_rate = sample_rate

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify_audio(self, audio_data: bytes) -> str:
        """Classify audio into: speech, music, silence, alert, ambient.

        Heuristics:
        - Very low energy -> silence
        - High zero-crossing rate + moderate energy -> speech
        - Low zero-crossing rate + moderate/high energy -> music
        - Very high energy short burst -> alert
        - Otherwise -> ambient
        """
        samples = self._decode_pcm(audio_data)
        if not samples:
            return "silence"

        energy = self._rms_energy(samples)
        zcr = self._zero_crossing_rate(samples)

        if energy < 0.005:
            return "silence"
        if energy > 0.7 and len(samples) < self._sample_rate:
            return "alert"
        if zcr > 0.15 and energy > 0.01:
            return "speech"
        if zcr < 0.08 and energy > 0.02:
            return "music"
        return "ambient"

    def detect_speech_activity(self, audio_data: bytes) -> bool:
        """Return True if someone is speaking."""
        samples = self._decode_pcm(audio_data)
        if not samples:
            return False

        energy = self._rms_energy(samples)
        zcr = self._zero_crossing_rate(samples)
        return energy > 0.01 and zcr > 0.10

    def estimate_noise_level(self, audio_data: bytes) -> float:
        """Return noise level from 0.0 (silent) to 1.0 (very loud)."""
        samples = self._decode_pcm(audio_data)
        if not samples:
            return 0.0
        energy = self._rms_energy(samples)
        # Clamp to [0, 1]
        return min(max(energy, 0.0), 1.0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_pcm(audio_data: bytes) -> list[float]:
        """Decode raw 16-bit signed little-endian PCM to [-1.0, 1.0] floats."""
        if len(audio_data) < _SAMPLE_WIDTH:
            return []
        n_samples = len(audio_data) // _SAMPLE_WIDTH
        try:
            raw = struct.unpack(f"<{n_samples}h", audio_data[: n_samples * _SAMPLE_WIDTH])
        except struct.error:
            return []
        return [s / 32768.0 for s in raw]

    @staticmethod
    def _rms_energy(samples: Sequence[float]) -> float:
        """Root-mean-square energy of sample buffer."""
        if not samples:
            return 0.0
        return math.sqrt(sum(s * s for s in samples) / len(samples))

    @staticmethod
    def _zero_crossing_rate(samples: Sequence[float]) -> float:
        """Fraction of adjacent sample pairs that cross zero."""
        if len(samples) < 2:
            return 0.0
        crossings = sum(
            1 for i in range(1, len(samples)) if (samples[i] >= 0) != (samples[i - 1] >= 0)
        )
        return crossings / (len(samples) - 1)
