from unittest.mock import MagicMock
from homie_core.voice.stt import SpeechToText


def test_stt_init_accepts_model_size():
    stt = SpeechToText(model_size="base")
    assert stt.model_size == "base"


def test_stt_transcribe_bytes_returns_tuple():
    stt = SpeechToText(model_size="base")
    mock_seg = MagicMock()
    mock_seg.text = "hello world"
    mock_info = MagicMock()
    mock_info.language = "en"
    stt._model = MagicMock()
    stt._model.transcribe.return_value = ([mock_seg], mock_info)
    text, lang = stt.transcribe_bytes(b"\x00" * 3200, sample_rate=16000)
    assert text == "hello world"
    assert lang == "en"


def test_stt_returns_empty_without_model():
    stt = SpeechToText()
    text, lang = stt.transcribe_bytes(b"\x00" * 100)
    assert text == ""
    assert lang == "en"


def test_stt_init():
    stt = SpeechToText(model_size="tiny")
    assert stt.model_size == "tiny"
    assert stt.is_loaded is False


def test_stt_transcribe_without_model():
    stt = SpeechToText()
    text, lang = stt.transcribe("nonexistent.wav")
    assert text == ""
    assert lang == "en"


def test_stt_unload():
    stt = SpeechToText()
    stt.unload()
    assert stt.is_loaded is False
