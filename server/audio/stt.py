import time
import logging
import numpy as np
from typing import Optional
from faster_whisper import WhisperModel

from server.config import settings

logger = logging.getLogger(__name__)

class SpeechToText:
    """
    SpeechToText wraps the faster-whisper model.
    Transcribes float32 audio arrays into text.
    """
    _model = None

    def __init__(self):
        if SpeechToText._model is None:
            logger.info(
                f"Loading Whisper model '{settings.whisper_model_size}' "
                f"on {settings.whisper_device} ({settings.whisper_compute_type})..."
            )
            try:
                SpeechToText._model = WhisperModel(
                    model_size_or_path=settings.whisper_model_size,
                    device=settings.whisper_device,
                    compute_type=settings.whisper_compute_type
                )
                logger.info("Whisper model loaded successfully.")
                
                # Warm-up run to initialize CUDA context and kernels during startup
                logger.info("Warming up Whisper model on GPU...")
                try:
                    dummy_audio = np.zeros(16000, dtype=np.float32)
                    list(SpeechToText._model.transcribe(dummy_audio, language="en")[0])
                    logger.info("Whisper warm-up complete.")
                except Exception as warmup_err:
                    logger.warning(f"Whisper warm-up failed (non-fatal): {warmup_err}")
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                raise
        
        self.model = SpeechToText._model

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Accepts a float32 audio array (16kHz), returns transcribed text.
        """
        if audio is None or len(audio) == 0:
            return ""

        # Very short audio threshold check (e.g., less than 0.1 seconds)
        # 16000 samples = 1 second
        if len(audio) < 1600:
            logger.warning("Audio chunk too short for transcription.")
            return ""

        start_time = time.time()
        
        try:
            # Define initial prompt with contextual vocabulary for home security and typical Indian English accents
            initial_prompt = "visitor, appointment, homeowner, guest, delivery, maintenance, plumber, electrician, technician, security, entry, meeting, package, sir, madam"

            # We already did VAD, so vad_filter=False
            segments, info = self.model.transcribe(
                audio,
                language=settings.whisper_language,
                beam_size=settings.whisper_beam_size,
                initial_prompt=initial_prompt,
                vad_filter=False
            )
            
            # segments is a generator; extract text from all segments
            text_parts = [segment.text for segment in segments]
            transcript = " ".join(text_parts).strip()
            
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(f"Transcription completed in {elapsed_ms:.2f}ms. Result: '{transcript}'")
            
            return transcript
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return ""
