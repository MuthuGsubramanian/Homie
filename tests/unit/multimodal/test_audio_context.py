from __future__ import annotations

import math
import struct

import pytest

from homie_core.multimodal.audio_context import AudioContextEngine


def _make_pcm_silence(n_samples: int = 1600) -> bytes:
    """Generate silent PCM (all zeros)."""
    return struct.pack(f"<{n_samples}h", *([0] * n_samples))


def _make_pcm_tone(freq: float = 440.0, duration_s: float = 0.1,
                   sample_rate: int = 16000, amplitude: float = 0.5) -> bytes:
    """Generate a sine-wave tone as 16-bit PCM."""
    n_samples = int(sample_rate * duration_s)
    samples = [
        int(amplitude * 32767 * math.sin(2 * math.pi * freq * i / sample_rate))
        for i in range(n_samples)
    ]
    return struct.pack(f"<{n_samples}h", *samples)


def _make_pcm_noise(n_samples: int = 1600, amplitude: int = 20000) -> bytes:
    """Generate pseudo-noise by alternating high/low values."""
    samples = [amplitude * (1 if i % 2 == 0 else -1) for i in range(n_samples)]
    return struct.pack(f"<{n_samples}h", *samples)


class TestAudioContextEngine:
    def test_classify_silence(self):
        engine = AudioContextEngine()
        assert engine.classify_audio(_make_pcm_silence()) == "silence"

    def test_classify_empty(self):
        engine = AudioContextEngine()
        assert engine.classify_audio(b"") == "silence"

    def test_classify_speech_like(self):
        """High ZCR + moderate energy should classify as speech."""
        engine = AudioContextEngine()
        # Rapid zero-crossing noise at moderate amplitude
        data = _make_pcm_noise(n_samples=4800, amplitude=5000)
        result = engine.classify_audio(data)
        assert result == "speech"

    def test_classify_music_like(self):
        """Low ZCR + moderate energy = music."""
        engine = AudioContextEngine()
        # Low frequency tone has few zero crossings relative to sample count
        data = _make_pcm_tone(freq=50.0, duration_s=0.3, amplitude=0.3)
        result = engine.classify_audio(data)
        assert result in ("music", "ambient")  # low-freq tone is music-like

    def test_detect_speech_on_noise(self):
        engine = AudioContextEngine()
        data = _make_pcm_noise(n_samples=4800, amplitude=5000)
        assert engine.detect_speech_activity(data) is True

    def test_detect_speech_on_silence(self):
        engine = AudioContextEngine()
        assert engine.detect_speech_activity(_make_pcm_silence()) is False

    def test_noise_level_silence(self):
        engine = AudioContextEngine()
        level = engine.estimate_noise_level(_make_pcm_silence())
        assert level == 0.0

    def test_noise_level_loud(self):
        engine = AudioContextEngine()
        data = _make_pcm_tone(amplitude=0.9, duration_s=0.1)
        level = engine.estimate_noise_level(data)
        assert level > 0.3

    def test_noise_level_range(self):
        engine = AudioContextEngine()
        data = _make_pcm_tone(amplitude=0.5, duration_s=0.1)
        level = engine.estimate_noise_level(data)
        assert 0.0 <= level <= 1.0

    def test_decode_pcm_empty(self):
        assert AudioContextEngine._decode_pcm(b"") == []

    def test_decode_pcm_single_sample(self):
        data = struct.pack("<h", 16384)
        samples = AudioContextEngine._decode_pcm(data)
        assert len(samples) == 1
        assert abs(samples[0] - 0.5) < 0.01
