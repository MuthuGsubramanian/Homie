# tests/unit/test_voice/test_integration.py
import struct
from unittest.mock import MagicMock
from homie_core.voice.voice_pipeline import PipelineState, VoicePipeline

def test_full_pipeline_recording_to_processing():
    responses = ["Hello ", "there!"]
    callback = MagicMock(return_value=iter(responses))
    pipeline = VoicePipeline(on_query=callback)
    mock_vad = MagicMock()
    pipeline.vad = mock_vad
    mock_stt = MagicMock()
    mock_stt.transcribe_bytes.return_value = ("test query", "en")
    pipeline.stt = mock_stt
    mock_engine = MagicMock()
    mock_engine.synthesize_stream.return_value = iter([b"audio"])
    mock_selector = MagicMock()
    mock_selector.select.return_value = mock_engine
    pipeline.tts_selector = mock_selector

    pipeline.begin_listening()
    assert pipeline.state == PipelineState.LISTENING

    speech_chunk = struct.pack("<512h", *([5000] * 512))
    silence_chunk = struct.pack("<512h", *([0] * 512))

    mock_vad.is_speech.return_value = True
    pipeline.process_audio_chunk(speech_chunk)
    assert pipeline.state == PipelineState.RECORDING

    for _ in range(5):
        pipeline.process_audio_chunk(speech_chunk)

    mock_vad.is_speech.return_value = False
    for _ in range(35):
        pipeline.process_audio_chunk(silence_chunk)

    mock_stt.transcribe_bytes.assert_called_once()
    callback.assert_called_once_with("test query")

def test_barge_in_during_speaking():
    callback = MagicMock(return_value=iter(["response"]))
    pipeline = VoicePipeline(on_query=callback)
    mock_vad = MagicMock()
    pipeline.vad = mock_vad
    pipeline._set_state(PipelineState.SPEAKING)
    pipeline._playback_queue.put(b"audio1")
    pipeline._playback_queue.put(b"audio2")
    mock_vad.is_speech.return_value = True
    pipeline.process_audio_chunk(struct.pack("<512h", *([5000] * 512)))
    assert pipeline._playback_queue.empty()
    assert pipeline.state == PipelineState.RECORDING
