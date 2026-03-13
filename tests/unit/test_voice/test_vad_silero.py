import struct
from unittest.mock import patch, MagicMock
from homie_core.voice.vad_silero import SileroVAD


def _make_audio(n=512):
    return struct.pack(f"<{n}h", *([5000] * n))


def test_silero_detects_speech():
    with patch("homie_core.voice.vad_silero.torch") as mt:
        mock_model = MagicMock()
        mock_model.return_value = MagicMock(item=MagicMock(return_value=0.9))
        mt.hub.load.return_value = (mock_model, None)
        mt.from_numpy.return_value = MagicMock()
        vad = SileroVAD(threshold=0.5)
        assert vad.is_speech(_make_audio()) is True


def test_silero_rejects_silence():
    with patch("homie_core.voice.vad_silero.torch") as mt:
        mock_model = MagicMock()
        mock_model.return_value = MagicMock(item=MagicMock(return_value=0.1))
        mt.hub.load.return_value = (mock_model, None)
        mt.from_numpy.return_value = MagicMock()
        vad = SileroVAD(threshold=0.5)
        assert vad.is_speech(_make_audio()) is False


def test_silero_hysteresis():
    with patch("homie_core.voice.vad_silero.torch") as mt:
        mock_model = MagicMock()
        mt.hub.load.return_value = (mock_model, None)
        mt.from_numpy.return_value = MagicMock()
        vad = SileroVAD(threshold=0.5)
        mock_model.return_value = MagicMock(item=MagicMock(return_value=0.8))
        assert vad.is_speech(_make_audio()) is True
        mock_model.return_value = MagicMock(item=MagicMock(return_value=0.4))
        assert vad.is_speech(_make_audio()) is True  # still above release (0.35)
        mock_model.return_value = MagicMock(item=MagicMock(return_value=0.3))
        assert vad.is_speech(_make_audio()) is False
