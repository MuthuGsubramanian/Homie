"""Tests for the voice pipeline and related voice components.

All audio/ML dependencies are mocked so these tests run in any environment.
"""

from __future__ import annotations

import struct
import threading
import time
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

from homie_core.voice.voice_pipeline import (
    PipelineState,
    VoicePipeline,
    _SILENCE_CHUNKS_THRESHOLD,
    _MAX_RECORDING_CHUNKS,
)
from homie_core.voice.wakeword import WakeWordDetector, DEFAULT_WAKE_PHRASE
from homie_core.voice.vad import VAD


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _silent_chunk(n_samples: int = 1024) -> bytes:
    """Return a chunk of silence (all zeros) as 16-bit PCM."""
    return struct.pack(f"<{n_samples}h", *([0] * n_samples))


def _loud_chunk(n_samples: int = 1024, amplitude: int = 5000) -> bytes:
    """Return a chunk of 'speech' (constant non-zero samples)."""
    return struct.pack(f"<{n_samples}h", *([amplitude] * n_samples))


# ---------------------------------------------------------------------------
# WakeWordDetector tests
# ---------------------------------------------------------------------------

class TestWakeWordDetector(unittest.TestCase):
    def test_detects_wake_phrase(self):
        det = WakeWordDetector(wake_phrase="hey homie")
        self.assertTrue(det.check("Hey Homie, what time is it?"))

    def test_ignores_unrelated(self):
        det = WakeWordDetector(wake_phrase="hey homie")
        self.assertFalse(det.check("what is the weather today"))

    def test_callback_fires(self):
        cb = MagicMock()
        det = WakeWordDetector(wake_phrase="hey homie", on_detected=cb)
        det.check("hey homie turn on lights")
        cb.assert_called_once()

    def test_no_callback_no_error(self):
        det = WakeWordDetector(wake_phrase="hey homie", on_detected=None)
        # Should not raise even when wake phrase detected
        self.assertTrue(det.check("hey homie"))

    def test_custom_phrase(self):
        det = WakeWordDetector(wake_phrase="ok computer")
        self.assertTrue(det.check("ok computer do something"))
        self.assertFalse(det.check("hey homie"))


# ---------------------------------------------------------------------------
# VAD tests (energy-based fallback -- no webrtcvad needed)
# ---------------------------------------------------------------------------

class TestVADEnergyFallback(unittest.TestCase):
    """Test the energy-based VAD fallback (no webrtcvad dep)."""

    def test_silence_is_not_speech(self):
        vad = VAD.__new__(VAD)
        vad.sample_rate = 16000
        vad.energy_threshold = 300
        vad._vad = None  # force energy-based fallback
        self.assertFalse(vad.is_speech(_silent_chunk()))

    def test_loud_audio_is_speech(self):
        vad = VAD.__new__(VAD)
        vad.sample_rate = 16000
        vad.energy_threshold = 300
        vad._vad = None
        self.assertTrue(vad.is_speech(_loud_chunk()))

    def test_empty_chunk_is_not_speech(self):
        vad = VAD.__new__(VAD)
        vad.sample_rate = 16000
        vad.energy_threshold = 300
        vad._vad = None
        self.assertFalse(vad.is_speech(b""))


# ---------------------------------------------------------------------------
# PipelineState tests
# ---------------------------------------------------------------------------

class TestPipelineState(unittest.TestCase):
    def test_all_states_exist(self):
        expected = {"idle", "listening", "recording", "processing", "speaking"}
        actual = {s.value for s in PipelineState}
        self.assertEqual(expected, actual)


# ---------------------------------------------------------------------------
# VoicePipeline unit tests (fully mocked components)
# ---------------------------------------------------------------------------

class TestVoicePipelineInit(unittest.TestCase):
    """Test pipeline creation and graceful degradation."""

    def test_initial_state_is_idle(self):
        pipeline = VoicePipeline(on_query=lambda t: "ok")
        self.assertEqual(pipeline.state, PipelineState.IDLE)

    def test_is_not_running_initially(self):
        pipeline = VoicePipeline(on_query=lambda t: "ok")
        self.assertFalse(pipeline.is_running)

    @patch("homie_core.voice.voice_pipeline.AudioRecorder", None)
    def test_start_fails_without_audio(self):
        """Pipeline should not start if AudioRecorder is unavailable."""
        pipeline = VoicePipeline(on_query=lambda t: "ok")
        result = pipeline.start()
        self.assertFalse(result)
        self.assertFalse(pipeline.is_running)


class TestVoicePipelineStateMachine(unittest.TestCase):
    """Test state transitions with mocked components."""

    def _make_pipeline(self, on_query=None):
        """Create a pipeline with all components mocked."""
        if on_query is None:
            on_query = MagicMock(return_value="response text")

        pipeline = VoicePipeline(on_query=on_query)
        return pipeline

    def test_set_state_thread_safe(self):
        """Multiple threads can set state concurrently without error."""
        pipeline = self._make_pipeline()
        errors = []

        def toggle(n):
            try:
                for _ in range(n):
                    pipeline._set_state(PipelineState.LISTENING)
                    pipeline._set_state(PipelineState.RECORDING)
                    pipeline._set_state(PipelineState.PROCESSING)
                    pipeline._set_state(PipelineState.SPEAKING)
                    pipeline._set_state(PipelineState.IDLE)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=toggle, args=(50,)) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        self.assertEqual(errors, [])

    def test_stop_on_idle_pipeline(self):
        """Stopping a pipeline that was never started should not raise."""
        pipeline = self._make_pipeline()
        pipeline.stop()  # should be a no-op
        self.assertEqual(pipeline.state, PipelineState.IDLE)


class TestVoicePipelineRecordAndProcess(unittest.TestCase):
    """Test the record-and-process flow with mocked audio/STT/TTS."""

    def test_full_flow_wake_record_transcribe_respond(self):
        """Simulate: wake word -> record speech -> transcribe -> query -> TTS."""
        query_cb = MagicMock(return_value="The lights are now on.")
        pipeline = VoicePipeline(on_query=query_cb)

        # Mock components
        mock_recorder = MagicMock()
        # Return loud audio for a few chunks then silence to end recording
        loud = _loud_chunk()
        silent = _silent_chunk()
        read_sequence = [loud] * 5 + [silent] * (_SILENCE_CHUNKS_THRESHOLD + 1)
        mock_recorder.read_chunk = MagicMock(side_effect=read_sequence)

        mock_vad = MagicMock()
        mock_vad.is_speech = MagicMock(
            side_effect=[True] * 5 + [False] * (_SILENCE_CHUNKS_THRESHOLD + 1)
        )

        mock_stt = MagicMock()
        mock_stt.transcribe = MagicMock(return_value="turn on the lights")

        mock_tts = MagicMock()

        pipeline._recorder = mock_recorder
        pipeline._vad = mock_vad
        pipeline._stt = mock_stt
        pipeline._stt_ok = True
        pipeline._tts = mock_tts
        pipeline._tts_ok = True

        # Execute the record-and-process phase directly
        pipeline._do_record_and_process()

        # Verify flow
        mock_stt.transcribe.assert_called_once()
        query_cb.assert_called_once_with("turn on the lights")
        mock_tts.speak.assert_called_once_with("The lights are now on.")
        # Should end in LISTENING state
        self.assertEqual(pipeline.state, PipelineState.LISTENING)

    def test_empty_transcription_returns_to_listening(self):
        """If STT produces empty text, skip query and go back to listening."""
        query_cb = MagicMock()
        pipeline = VoicePipeline(on_query=query_cb)

        mock_recorder = MagicMock()
        silent = _silent_chunk()
        mock_recorder.read_chunk = MagicMock(
            side_effect=[_loud_chunk()] * 3 + [silent] * (_SILENCE_CHUNKS_THRESHOLD + 1)
        )

        mock_vad = MagicMock()
        mock_vad.is_speech = MagicMock(
            side_effect=[True] * 3 + [False] * (_SILENCE_CHUNKS_THRESHOLD + 1)
        )

        mock_stt = MagicMock()
        mock_stt.transcribe = MagicMock(return_value="   ")  # whitespace only

        pipeline._recorder = mock_recorder
        pipeline._vad = mock_vad
        pipeline._stt = mock_stt
        pipeline._stt_ok = True
        pipeline._tts_ok = False

        pipeline._do_record_and_process()

        query_cb.assert_not_called()
        self.assertEqual(pipeline.state, PipelineState.LISTENING)

    def test_stt_failure_returns_to_listening(self):
        """If STT raises, pipeline should recover and return to listening."""
        query_cb = MagicMock()
        pipeline = VoicePipeline(on_query=query_cb)

        mock_recorder = MagicMock()
        mock_recorder.read_chunk = MagicMock(
            side_effect=[_loud_chunk()] * 2 + [_silent_chunk()] * (_SILENCE_CHUNKS_THRESHOLD + 1)
        )

        mock_vad = MagicMock()
        mock_vad.is_speech = MagicMock(
            side_effect=[True] * 2 + [False] * (_SILENCE_CHUNKS_THRESHOLD + 1)
        )

        mock_stt = MagicMock()
        mock_stt.transcribe = MagicMock(side_effect=RuntimeError("model error"))

        pipeline._recorder = mock_recorder
        pipeline._vad = mock_vad
        pipeline._stt = mock_stt
        pipeline._stt_ok = True
        pipeline._tts_ok = False

        pipeline._do_record_and_process()

        query_cb.assert_not_called()
        self.assertEqual(pipeline.state, PipelineState.LISTENING)

    def test_tts_failure_does_not_crash(self):
        """If TTS fails the pipeline should still return to listening."""
        query_cb = MagicMock(return_value="some response")
        pipeline = VoicePipeline(on_query=query_cb)

        mock_recorder = MagicMock()
        mock_recorder.read_chunk = MagicMock(
            side_effect=[_loud_chunk()] * 2 + [_silent_chunk()] * (_SILENCE_CHUNKS_THRESHOLD + 1)
        )

        mock_vad = MagicMock()
        mock_vad.is_speech = MagicMock(
            side_effect=[True] * 2 + [False] * (_SILENCE_CHUNKS_THRESHOLD + 1)
        )

        mock_stt = MagicMock()
        mock_stt.transcribe = MagicMock(return_value="hello")

        mock_tts = MagicMock()
        mock_tts.speak = MagicMock(side_effect=RuntimeError("audio device busy"))

        pipeline._recorder = mock_recorder
        pipeline._vad = mock_vad
        pipeline._stt = mock_stt
        pipeline._stt_ok = True
        pipeline._tts = mock_tts
        pipeline._tts_ok = True

        pipeline._do_record_and_process()

        query_cb.assert_called_once_with("hello")
        self.assertEqual(pipeline.state, PipelineState.LISTENING)

    def test_query_callback_error_returns_fallback_message(self):
        """If on_query raises, TTS should get an error message."""
        query_cb = MagicMock(side_effect=RuntimeError("brain error"))
        pipeline = VoicePipeline(on_query=query_cb)

        mock_recorder = MagicMock()
        mock_recorder.read_chunk = MagicMock(
            side_effect=[_loud_chunk()] * 2 + [_silent_chunk()] * (_SILENCE_CHUNKS_THRESHOLD + 1)
        )

        mock_vad = MagicMock()
        mock_vad.is_speech = MagicMock(
            side_effect=[True] * 2 + [False] * (_SILENCE_CHUNKS_THRESHOLD + 1)
        )

        mock_stt = MagicMock()
        mock_stt.transcribe = MagicMock(return_value="do something")

        mock_tts = MagicMock()

        pipeline._recorder = mock_recorder
        pipeline._vad = mock_vad
        pipeline._stt = mock_stt
        pipeline._stt_ok = True
        pipeline._tts = mock_tts
        pipeline._tts_ok = True

        pipeline._do_record_and_process()

        # TTS should have been called with error fallback message
        mock_tts.speak.assert_called_once()
        spoken = mock_tts.speak.call_args[0][0]
        self.assertIn("error", spoken.lower())

    def test_recording_without_vad_uses_fixed_length(self):
        """Without VAD, recording should stop after ~80 chunks."""
        query_cb = MagicMock(return_value="ok")
        pipeline = VoicePipeline(on_query=query_cb)

        mock_recorder = MagicMock()
        mock_recorder.read_chunk = MagicMock(return_value=_loud_chunk())

        mock_stt = MagicMock()
        mock_stt.transcribe = MagicMock(return_value="hello")

        pipeline._recorder = mock_recorder
        pipeline._vad = None  # no VAD
        pipeline._stt = mock_stt
        pipeline._stt_ok = True
        pipeline._tts_ok = False

        pipeline._do_record_and_process()

        # Without VAD it should record exactly 80 chunks
        self.assertEqual(mock_recorder.read_chunk.call_count, 80)


class TestVoicePipelineStartStop(unittest.TestCase):
    """Test start/stop lifecycle with mocked AudioRecorder."""

    @patch("homie_core.voice.voice_pipeline.AudioRecorder")
    def test_start_and_stop(self, MockRecorder):
        """Pipeline thread starts and stops cleanly."""
        mock_instance = MagicMock()
        mock_instance.read_chunk = MagicMock(return_value=_silent_chunk())
        MockRecorder.return_value = mock_instance

        pipeline = VoicePipeline(on_query=lambda t: "ok")
        started = pipeline.start()
        self.assertTrue(started)
        self.assertTrue(pipeline.is_running)

        # Give the thread a moment to enter the loop
        time.sleep(0.1)

        pipeline.stop()
        self.assertFalse(pipeline.is_running)
        self.assertEqual(pipeline.state, PipelineState.IDLE)

    @patch("homie_core.voice.voice_pipeline.AudioRecorder")
    def test_double_start(self, MockRecorder):
        """Calling start() twice should be safe."""
        mock_instance = MagicMock()
        mock_instance.read_chunk = MagicMock(return_value=_silent_chunk())
        MockRecorder.return_value = mock_instance

        pipeline = VoicePipeline(on_query=lambda t: "ok")
        pipeline.start()
        result = pipeline.start()  # second start
        self.assertTrue(result)
        pipeline.stop()


if __name__ == "__main__":
    unittest.main()
