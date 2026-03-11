"""Voice pipeline -- orchestrates wake-word detection, recording,
transcription, response generation, and speech synthesis.

The pipeline runs in its own daemon thread and cycles through states::

    IDLE -> LISTENING -> RECORDING -> PROCESSING -> SPEAKING -> IDLE

All voice dependencies are optional.  When a component cannot be created
the pipeline falls back to text-only operation (or disables itself entirely
when audio I/O is unavailable).
"""

from __future__ import annotations

import enum
import logging
import threading
import time
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try importing voice components -- each may fail independently.
# ---------------------------------------------------------------------------

try:
    from homie_core.voice.audio_io import AudioRecorder, AudioUnavailableError, RATE, CHUNK
except ImportError:
    AudioRecorder = None  # type: ignore[assignment,misc]
    AudioUnavailableError = RuntimeError  # type: ignore[assignment,misc]
    RATE = 16_000
    CHUNK = 1024

try:
    from homie_core.voice.vad import VAD
except ImportError:
    VAD = None  # type: ignore[assignment,misc]

try:
    from homie_core.voice.wakeword import WakeWordDetector
except ImportError:
    WakeWordDetector = None  # type: ignore[assignment,misc]

try:
    from homie_core.voice.stt import SpeechToText
except ImportError:
    SpeechToText = None  # type: ignore[assignment,misc]

try:
    from homie_core.voice.tts import TextToSpeech
except ImportError:
    TextToSpeech = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Pipeline states
# ---------------------------------------------------------------------------

class PipelineState(enum.Enum):
    """Voice pipeline states."""

    IDLE = "idle"
    LISTENING = "listening"
    RECORDING = "recording"
    PROCESSING = "processing"
    SPEAKING = "speaking"


# ---------------------------------------------------------------------------
# Callback type
# ---------------------------------------------------------------------------

# on_query_cb(text) -> response_text
QueryCallback = Callable[[str], str]


# ---------------------------------------------------------------------------
# Voice pipeline
# ---------------------------------------------------------------------------

# How many consecutive silent chunks signal end-of-speech.
_SILENCE_CHUNKS_THRESHOLD = 30  # ~1.9 s at 16 kHz / 1024-frame chunks
# Max recording length in chunks to prevent runaway recording.
_MAX_RECORDING_CHUNKS = 500  # ~32 s


class VoicePipeline:
    """Orchestrates the full voice interaction loop.

    Parameters
    ----------
    on_query : callable
        ``(transcribed_text: str) -> response_text: str``
        Called when the user finishes speaking.  Should return the text
        response to be spoken back.
    wake_phrase : str
        Wake word / phrase to listen for (default ``"hey homie"``).
    stt_model : str
        Whisper model name (default ``"base"``).
    """

    def __init__(
        self,
        on_query: QueryCallback,
        wake_phrase: str = "hey homie",
        stt_model: str = "base",
    ) -> None:
        self._on_query = on_query
        self._wake_phrase = wake_phrase
        self._stt_model = stt_model

        self._state = PipelineState.IDLE
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Components -- initialised lazily in _init_components()
        self._recorder: Optional[object] = None
        self._vad: Optional[object] = None
        self._wakeword: Optional[object] = None
        self._stt: Optional[object] = None
        self._tts: Optional[object] = None

        # Track which components are available for graceful degradation
        self._audio_ok = False
        self._stt_ok = False
        self._tts_ok = False

    # -- public API ---------------------------------------------------------

    @property
    def state(self) -> PipelineState:
        with self._state_lock:
            return self._state

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        """Start the pipeline thread.  Returns True if started successfully."""
        if self.is_running:
            logger.warning("VoicePipeline already running")
            return True

        if not self._init_components():
            logger.error("VoicePipeline: no audio components available; not starting")
            return False

        self._stop_event.clear()
        self._set_state(PipelineState.IDLE)
        self._thread = threading.Thread(
            target=self._run_loop,
            name="voice-pipeline",
            daemon=True,
        )
        self._thread.start()
        logger.info("VoicePipeline started")
        return True

    def stop(self) -> None:
        """Signal the pipeline to stop and wait for the thread to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._cleanup_components()
        self._set_state(PipelineState.IDLE)
        logger.info("VoicePipeline stopped")

    # -- state management ---------------------------------------------------

    def _set_state(self, new_state: PipelineState) -> None:
        with self._state_lock:
            old = self._state
            self._state = new_state
        if old != new_state:
            logger.debug("Pipeline state: %s -> %s", old.value, new_state.value)

    # -- component initialisation -------------------------------------------

    def _init_components(self) -> bool:
        """Try to create each voice component.  Returns True if at least
        audio recording is available (minimum for the pipeline to run)."""

        # Audio recorder
        if AudioRecorder is not None:
            try:
                self._recorder = AudioRecorder()
                self._audio_ok = True
            except Exception as exc:
                logger.warning("AudioRecorder unavailable: %s", exc)
                self._audio_ok = False
        else:
            self._audio_ok = False

        if not self._audio_ok:
            return False

        # VAD
        if VAD is not None:
            try:
                self._vad = VAD()
            except Exception as exc:
                logger.warning("VAD unavailable: %s", exc)

        # Wake word
        if WakeWordDetector is not None:
            self._wakeword = WakeWordDetector(wake_phrase=self._wake_phrase)

        # STT
        if SpeechToText is not None:
            try:
                self._stt = SpeechToText(model_name=self._stt_model)
                self._stt_ok = True
            except Exception as exc:
                logger.warning("STT unavailable: %s", exc)
                self._stt_ok = False

        # TTS
        if TextToSpeech is not None:
            try:
                self._tts = TextToSpeech()
                self._tts_ok = True
            except Exception as exc:
                logger.warning("TTS unavailable: %s", exc)
                self._tts_ok = False

        return True

    def _cleanup_components(self) -> None:
        if self._recorder is not None and hasattr(self._recorder, "close"):
            try:
                self._recorder.close()  # type: ignore[union-attr]
            except Exception:
                pass
        self._recorder = None

    # -- main loop ----------------------------------------------------------

    def _run_loop(self) -> None:
        """Main pipeline loop running on a background thread."""
        try:
            self._recorder.open()  # type: ignore[union-attr]
        except Exception as exc:
            logger.error("Failed to open audio recorder: %s", exc)
            self._set_state(PipelineState.IDLE)
            return

        try:
            while not self._stop_event.is_set():
                try:
                    self._tick()
                except Exception as exc:
                    logger.exception("Pipeline tick error: %s", exc)
                    # Avoid tight error loops
                    time.sleep(0.5)
        finally:
            self._cleanup_components()

    def _tick(self) -> None:
        """One iteration of the pipeline loop."""
        state = self.state

        if state == PipelineState.IDLE:
            self._set_state(PipelineState.LISTENING)

        elif state == PipelineState.LISTENING:
            self._do_listen()

        elif state == PipelineState.RECORDING:
            # Recording is handled inside _do_listen -> _do_record
            pass

        elif state == PipelineState.PROCESSING:
            # Processing is handled inline after recording
            pass

        elif state == PipelineState.SPEAKING:
            # Speaking is handled inline after processing
            pass

    def _do_listen(self) -> None:
        """Listen for wake word using short audio snippets."""
        chunk = self._recorder.read_chunk()  # type: ignore[union-attr]

        # If we have a VAD, only bother with chunks that contain speech
        if self._vad is not None and not self._vad.is_speech(chunk):  # type: ignore[union-attr]
            return

        # If we have STT + wakeword detector, transcribe the chunk and check
        if self._stt_ok and self._wakeword is not None:
            try:
                snippet = self._stt.transcribe(chunk)  # type: ignore[union-attr]
                if self._wakeword.check(snippet):  # type: ignore[union-attr]
                    logger.info("Wake word detected -- starting recording")
                    self._do_record_and_process()
            except Exception as exc:
                logger.debug("Wake word check failed: %s", exc)
        else:
            # Without STT we cannot detect wake words by transcription.
            # Fall back to simple VAD-triggered recording (always record on speech).
            if self._vad is not None:
                self._do_record_and_process()

    def _do_record_and_process(self) -> None:
        """Record speech until silence, transcribe, process, and respond."""
        self._set_state(PipelineState.RECORDING)

        frames: List[bytes] = []
        silence_count = 0

        for _ in range(_MAX_RECORDING_CHUNKS):
            if self._stop_event.is_set():
                return

            chunk = self._recorder.read_chunk()  # type: ignore[union-attr]
            frames.append(chunk)

            if self._vad is not None:
                if self._vad.is_speech(chunk):  # type: ignore[union-attr]
                    silence_count = 0
                else:
                    silence_count += 1
                    if silence_count >= _SILENCE_CHUNKS_THRESHOLD:
                        break
            else:
                # Without VAD record a fixed number of chunks (~5 s)
                if len(frames) >= 80:
                    break

        pcm_audio = b"".join(frames)
        logger.info("Recorded %d bytes of audio", len(pcm_audio))

        # -- Transcribe -------------------------------------------------------
        self._set_state(PipelineState.PROCESSING)
        transcript = ""

        if self._stt_ok:
            try:
                transcript = self._stt.transcribe(pcm_audio)  # type: ignore[union-attr]
            except Exception as exc:
                logger.error("STT transcription failed: %s", exc)

        if not transcript.strip():
            logger.info("Empty transcription -- returning to listening")
            self._set_state(PipelineState.LISTENING)
            return

        logger.info("User said: %r", transcript)

        # -- Query callback (the brain) ----------------------------------------
        response_text = ""
        try:
            response_text = self._on_query(transcript)
        except Exception as exc:
            logger.error("Query callback failed: %s", exc)
            response_text = "Sorry, I encountered an error processing your request."

        # -- Speak response -----------------------------------------------------
        if response_text and self._tts_ok:
            self._set_state(PipelineState.SPEAKING)
            try:
                self._tts.speak(response_text)  # type: ignore[union-attr]
            except Exception as exc:
                logger.error("TTS failed: %s", exc)
        elif response_text:
            logger.info("TTS unavailable; response text: %s", response_text)

        self._set_state(PipelineState.LISTENING)


def is_available() -> bool:
    """Return True if the minimum deps for the voice pipeline exist."""
    return AudioRecorder is not None


__all__ = ["VoicePipeline", "PipelineState", "is_available"]
