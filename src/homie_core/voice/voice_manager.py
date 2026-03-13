from __future__ import annotations
import logging
import queue
import threading
import time
from typing import Callable, Iterator, Optional

from homie_core.voice.voice_pipeline import PipelineState, VoicePipeline
from homie_core.voice.voice_prompts import EXIT_AUTO_MESSAGE, EXIT_CONFIRMATION, get_voice_hint

logger = logging.getLogger(__name__)

class VoiceManager:
    def __init__(self, config, on_query: Callable[[str], Iterator[str]],
                 on_state_change: Optional[Callable[[PipelineState], None]] = None) -> None:
        self._config = config
        self._on_query_raw = on_query
        self._on_state_change = on_state_change
        self._pipeline = VoicePipeline(
            on_query=self._query_with_voice_hint,
            on_state_change=self._handle_state_change,
            mode=config.mode,
            sample_rate=config.audio_sample_rate,
            chunk_size=config.audio_chunk_size,
        )
        self._mode = config.mode
        self._conversational_active = False
        self._exit_prompt_count = 0
        self._silence_timer: Optional[threading.Timer] = None
        self._voice_loop_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._available: dict[str, bool] = {}
        self._wake_word_engine = None
        self._wake_word_detector = None

    def start(self) -> None:
        self._probe_components()
        self._pipeline.start()
        self._stop_event.clear()
        self._voice_loop_thread = threading.Thread(target=self._voice_loop, daemon=True, name="voice-manager")
        self._voice_loop_thread.start()
        logger.info("VoiceManager started (mode=%s)", self._mode)

    def stop(self) -> None:
        self._stop_event.set()
        self._cancel_silence_timer()
        self._pipeline.stop()
        if self._wake_word_engine:
            self._wake_word_engine.stop()
        if self._voice_loop_thread:
            self._voice_loop_thread.join(timeout=5)
        logger.info("VoiceManager stopped")

    def set_mode(self, mode: str) -> None:
        self._mode = mode
        self._pipeline._mode = mode

    def enter_conversational(self) -> None:
        self._conversational_active = True
        self._exit_prompt_count = 0
        self._pipeline.begin_listening()
        self._reset_silence_timer()

    def exit_conversational(self) -> None:
        self._conversational_active = False
        self._cancel_silence_timer()
        self._pipeline.stop_listening()

    def on_hotkey(self) -> None:
        state = self._pipeline.state
        if state == PipelineState.IDLE:
            self._pipeline.begin_listening()
        elif state in (PipelineState.LISTENING, PipelineState.RECORDING):
            self._pipeline.stop_listening()
        elif state == PipelineState.SPEAKING:
            self._pipeline.barge_in()
            self._pipeline.begin_listening()

    @property
    def state(self) -> PipelineState:
        return self._pipeline.state

    @property
    def available_engines(self) -> dict[str, bool]:
        return dict(self._available)

    def status_report(self) -> str:
        lines = [
            f"Voice: {'enabled' if self._config.enabled else 'disabled'}",
            f"Mode:  {self._mode}",
            f"State: {self._pipeline.state.value}",
        ]
        for name, avail in self._available.items():
            lines.append(f"  {name}: {'available' if avail else 'not installed'}")
        return "\n".join(lines)

    def _probe_components(self) -> None:
        # VAD
        try:
            from homie_core.voice.vad import create_vad
            vad = create_vad(engine=self._config.vad_engine, threshold=self._config.vad_threshold)
            self._pipeline.vad = vad
            self._available["vad"] = True
        except Exception:
            logger.warning("VAD unavailable")
            self._available["vad"] = False

        # STT
        try:
            from homie_core.voice.stt import SpeechToText
            device = self._config.device if self._config.device != "auto" else "cpu"
            stt = SpeechToText(
                model_size=self._config.stt_model_quality,
                device=device,
                compute_type="float16" if device == "cuda" else "float32",
            )
            stt.load()
            self._pipeline.stt = stt
            self._available["stt"] = stt.is_loaded
        except Exception:
            logger.warning("STT unavailable")
            self._available["stt"] = False

        # TTS
        from homie_core.voice.tts_selector import TTSSelector
        fast_tts = self._try_load_tts("piper")
        quality_tts = self._try_load_tts("kokoro")
        multi_tts = self._try_load_tts("melo")
        self._pipeline.tts_selector = TTSSelector(
            fast=fast_tts, quality=quality_tts, multilingual=multi_tts,
            mode=self._config.tts_mode,
        )

        # Audio I/O
        try:
            from homie_core.voice.audio_io import AudioInThread, AudioOutThread
            self._pipeline.audio_in = AudioInThread(
                self._pipeline._vad_queue, self._stop_event,
                sample_rate=self._config.audio_sample_rate,
                chunk_size=self._config.audio_chunk_size,
            )
            self._pipeline.audio_out = AudioOutThread(
                self._pipeline._playback_queue, self._stop_event,
                should_play=self._pipeline._should_play,
                sample_rate=self._config.audio_sample_rate,
                chunk_size=self._config.audio_chunk_size,
            )
            self._available["audio"] = True
        except Exception:
            logger.warning("Audio I/O unavailable")
            self._available["audio"] = False

        # Wake word
        if self._config.wake_word:
            try:
                from homie_core.voice.wakeword import WakeWordEngine
                self._wake_word_engine = WakeWordEngine(wake_word=self._config.wake_word)
                self._wake_word_engine.start(on_wake=self._handle_wake_word_detected)
                self._available["wake_word"] = True
            except (ImportError, Exception):
                try:
                    from homie_core.voice.wakeword import WakeWordDetector
                    self._wake_word_detector = WakeWordDetector(
                        wake_phrase=self._config.wake_word,
                        on_detected=self._handle_wake_word_detected,
                    )
                    self._available["wake_word"] = True
                except Exception:
                    self._available["wake_word"] = False

    def _try_load_tts(self, engine_name: str):
        device = self._config.device if self._config.device != "auto" else "cpu"
        try:
            if engine_name == "piper":
                from homie_core.voice.tts import PiperTTS
                tts = PiperTTS(voice=self._config.tts_voice_fast)
                tts.load(device="cpu")
                self._available["tts_piper"] = tts.is_loaded
                return tts if tts.is_loaded else None
            elif engine_name == "kokoro":
                from homie_core.voice.tts_kokoro import KokoroTTS
                tts = KokoroTTS()
                tts.load(device=device)
                self._available["tts_kokoro"] = tts.is_loaded
                return tts if tts.is_loaded else None
            elif engine_name == "melo":
                from homie_core.voice.tts_melo import MeloTTS
                tts = MeloTTS()
                tts.load(device=device)
                self._available["tts_melo"] = tts.is_loaded
                return tts if tts.is_loaded else None
        except Exception:
            self._available[f"tts_{engine_name}"] = False
        return None

    def _query_with_voice_hint(self, text: str) -> Iterator[str]:
        if self._conversational_active and self._is_exit_phrase(text):
            yield EXIT_CONFIRMATION
            return
        voice_text = f"[VOICE_MODE] {text}"
        yield from self._on_query_raw(voice_text)

    def _is_exit_phrase(self, text: str) -> bool:
        text_lower = text.strip().lower()
        return any(phrase in text_lower for phrase in self._config.exit_phrases)

    def _handle_wake_word_detected(self) -> None:
        if self._pipeline.state == PipelineState.IDLE:
            logger.info("Wake word detected!")
            self._pipeline.begin_listening()

    def _voice_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                chunk = self._pipeline._vad_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if chunk == b"END":
                break
            self._pipeline.process_audio_chunk(chunk)

    def _handle_state_change(self, new_state: PipelineState) -> None:
        if self._on_state_change:
            self._on_state_change(new_state)
        if self._conversational_active:
            if new_state == PipelineState.LISTENING:
                self._reset_silence_timer()
            elif new_state in (PipelineState.PROCESSING, PipelineState.SPEAKING):
                self._cancel_silence_timer()

    def _reset_silence_timer(self) -> None:
        self._cancel_silence_timer()
        self._silence_timer = threading.Timer(
            self._config.conversation_timeout, self._on_silence_timeout
        )
        self._silence_timer.daemon = True
        self._silence_timer.start()

    def _cancel_silence_timer(self) -> None:
        if self._silence_timer:
            self._silence_timer.cancel()
            self._silence_timer = None

    def _on_silence_timeout(self) -> None:
        self._exit_prompt_count += 1
        if self._exit_prompt_count >= self._config.max_exit_prompts:
            logger.info("Auto-exiting after %d unanswered prompts", self._exit_prompt_count)
            self._synthesize_exit_message(EXIT_AUTO_MESSAGE)
            self.exit_conversational()
        else:
            self._synthesize_exit_message(EXIT_CONFIRMATION)
            self._reset_silence_timer()

    def _synthesize_exit_message(self, message: str) -> None:
        if self._pipeline.tts_selector:
            try:
                engine = self._pipeline.tts_selector.select(message, detected_lang="en")
                for chunk in engine.synthesize_stream(message):
                    self._pipeline._playback_queue.put(chunk)
            except Exception:
                logger.exception("Failed to synthesize exit message")
