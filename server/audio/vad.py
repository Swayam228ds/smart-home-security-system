import enum
import logging
import numpy as np
import torch
from typing import Optional, List

from server.config import (
    SAMPLE_RATE,
    CHUNK_DURATION_MS,
    VAD_THRESHOLD,
    SILENCE_DURATION_MS,
    MIN_SPEECH_DURATION_MS,
)

logger = logging.getLogger(__name__)

class VADState(enum.Enum):
    SILENCE = 1
    SPEECH = 2
    END_OF_SPEECH = 3

class VADProcessor:
    """
    VADProcessor wraps Silero VAD.
    Detects speech segments from raw PCM chunks using a state machine.
    """
    _model = None

    def __init__(self):
        if VADProcessor._model is None:
            logger.info("Loading Silero VAD model (singleton)...")
            try:
                # Load Silero VAD from torch hub
                model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    trust_repo=True
                )
                VADProcessor._model = model
                logger.info("Silero VAD model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load Silero VAD model: {e}")
                raise

        self.model = VADProcessor._model

        # Calculate chunk counts based on durations
        self.chunks_for_silence = max(1, SILENCE_DURATION_MS // CHUNK_DURATION_MS)
        self.min_speech_chunks = max(1, MIN_SPEECH_DURATION_MS // CHUNK_DURATION_MS)

        self.reset()

    def reset(self):
        """Resets the state machine for a new session."""
        self.state = VADState.SILENCE
        self.speech_buffer: List[bytes] = []
        self.silence_counter = 0
        self.speech_counter = 0
        # reset model internal states
        self.model.reset_states()

    def process_chunk(self, chunk: bytes) -> Optional[np.ndarray]:
        """
        Accepts raw PCM bytes (30ms chunks).
        Returns a complete speech segment as a float32 numpy array when end-of-speech is detected,
        otherwise returns None.
        """
        # Convert int16 PCM bytes to float32 tensor
        audio_int16 = np.frombuffer(chunk, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_float32)

        # Get speech probability
        with torch.no_grad():
            prob = self.model(audio_tensor, SAMPLE_RATE).item()

        is_speech = prob >= VAD_THRESHOLD

        if self.state == VADState.SILENCE:
            if is_speech:
                self.state = VADState.SPEECH
                self.speech_buffer.append(chunk)
                self.speech_counter = 1
                self.silence_counter = 0

        elif self.state == VADState.SPEECH:
            self.speech_buffer.append(chunk)
            self.speech_counter += 1

            if is_speech:
                self.silence_counter = 0
            else:
                self.silence_counter += 1

            if self.silence_counter >= self.chunks_for_silence:
                self.state = VADState.END_OF_SPEECH

        if self.state == VADState.END_OF_SPEECH:
            # Check if the accumulated speech is long enough
            is_valid_speech = self.speech_counter >= self.min_speech_chunks

            if is_valid_speech:
                # Remove trailing silence chunks if possible
                valid_chunks = self.speech_buffer[:-self.silence_counter]
                if not valid_chunks:
                    valid_chunks = self.speech_buffer

                full_audio_bytes = b''.join(valid_chunks)
                full_int16 = np.frombuffer(full_audio_bytes, dtype=np.int16)
                final_segment = full_int16.astype(np.float32) / 32768.0

                self.reset()
                return final_segment
            else:
                # Discard too short segments
                self.reset()

        return None
