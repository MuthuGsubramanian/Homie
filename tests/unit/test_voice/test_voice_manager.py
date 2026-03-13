from unittest.mock import MagicMock
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

def test_voice_manager_init():
    mgr = VoiceManager(config=_make_config(), on_query=MagicMock(return_value=iter(["hi"])))
    assert mgr.state == PipelineState.IDLE

def test_status_report():
    mgr = VoiceManager(config=_make_config(), on_query=MagicMock(return_value=iter(["hi"])))
    report = mgr.status_report()
    assert "Voice" in report

def test_exit_phrase_detected():
    mgr = VoiceManager(config=_make_config(), on_query=MagicMock(return_value=iter(["hi"])))
    assert mgr._is_exit_phrase("goodbye") is True
    assert mgr._is_exit_phrase("hello") is False
    assert mgr._is_exit_phrase("STOP") is True

def test_wake_word_transitions():
    mgr = VoiceManager(config=_make_config(mode="wake_word"), on_query=MagicMock(return_value=iter(["hi"])))
    mgr._handle_wake_word_detected()
    assert mgr._pipeline.state == PipelineState.LISTENING

def test_on_hotkey_idle_to_listening():
    mgr = VoiceManager(config=_make_config(), on_query=MagicMock(return_value=iter(["hi"])))
    mgr.on_hotkey()
    assert mgr.state == PipelineState.LISTENING

def test_on_hotkey_listening_to_idle():
    mgr = VoiceManager(config=_make_config(), on_query=MagicMock(return_value=iter(["hi"])))
    mgr._pipeline.begin_listening()
    mgr.on_hotkey()
    assert mgr.state == PipelineState.IDLE
