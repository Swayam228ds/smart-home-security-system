"""
Text-to-Speech module wrapping Piper TTS.
Synthesizes text into raw 16kHz 16-bit mono PCM audio.
"""

import os
import io
import wave
import asyncio
import logging
from pathlib import Path
from typing import AsyncGenerator

from piper.voice import PiperVoice
from piper.download_voices import download_voice

from server.config import settings, PIPER_MODELS_DIR, SAMPLE_RATE

logger = logging.getLogger(__name__)


class TextToSpeech:
    """
    TextToSpeech wraps the Piper TTS model.
    Synthesizes text into raw 16kHz 16-bit mono PCM bytes.

    Uses singleton pattern — the voice model is loaded once and reused.
    """

    _voice: PiperVoice = None
    _voice_sample_rate: int = None

    def __init__(self):
        if TextToSpeech._voice is None:
            voice_name = settings.piper_voice
            data_dir = Path(PIPER_MODELS_DIR)
            model_path = data_dir / f"{voice_name}.onnx"
            config_path = data_dir / f"{voice_name}.onnx.json"

            # Download voice model if not present
            if not model_path.exists() or not config_path.exists():
                logger.info(
                    f"Piper voice model '{voice_name}' not found at {model_path}. "
                    f"Downloading..."
                )
                try:
                    data_dir.mkdir(parents=True, exist_ok=True)
                    download_voice(voice_name, data_dir)
                    logger.info(f"Piper voice model '{voice_name}' downloaded successfully.")
                except Exception as e:
                    logger.error(f"Failed to download Piper voice model: {e}")
                    raise

            # Load the voice model
            logger.info(f"Loading Piper voice from {model_path}...")
            TextToSpeech._voice = PiperVoice.load(
                model_path=str(model_path),
                config_path=str(config_path),
            )
            TextToSpeech._voice_sample_rate = TextToSpeech._voice.config.sample_rate
            logger.info(
                f"Piper voice loaded successfully "
                f"(sample_rate={TextToSpeech._voice_sample_rate}Hz)."
            )

        self.voice = TextToSpeech._voice
        self.voice_sample_rate = TextToSpeech._voice_sample_rate

    def synthesize(self, text: str) -> bytes:
        """
        Synthesizes text and returns raw 16-bit mono PCM bytes.

        The Piper model may output at a different sample rate than our
        pipeline's 16kHz. If so, we resample to match.

        Args:
            text: Text to synthesize.

        Returns:
            Raw PCM audio bytes (16-bit signed, mono).
        """
        if not text or not text.strip():
            return b""

        # Collect all audio chunks from Piper
        pcm_parts = []
        for audio_chunk in self.voice.synthesize(text):
            pcm_parts.append(audio_chunk.audio_int16_bytes)

        if not pcm_parts:
            return b""

        raw_pcm = b"".join(pcm_parts)

        # Resample if Piper's native rate differs from pipeline rate (16kHz)
        if self.voice_sample_rate != SAMPLE_RATE:
            raw_pcm = self._resample(raw_pcm, self.voice_sample_rate, SAMPLE_RATE)

        return raw_pcm

    def synthesize_wav(self, text: str) -> bytes:
        """
        Synthesizes text and returns a complete WAV file as bytes.

        Args:
            text: Text to synthesize.

        Returns:
            Complete WAV file bytes.
        """
        raw_pcm = self.synthesize(text)
        if not raw_pcm:
            return b""

        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(raw_pcm)

        return wav_io.getvalue()

    async def synthesize_to_chunks(
        self, text: str, chunk_size: int = 4096
    ) -> AsyncGenerator[bytes, None]:
        """
        Yields raw PCM chunks for streaming.
        Runs synthesis in an executor to avoid blocking the async event loop.

        Args:
            text: Text to synthesize.
            chunk_size: Size of each yielded chunk in bytes.

        Yields:
            Raw PCM audio chunks.
        """
        if not text or not text.strip():
            return

        loop = asyncio.get_event_loop()

        # Run the blocking synthesis in a thread pool
        full_pcm = await loop.run_in_executor(None, self.synthesize, text)

        # Yield in chunks
        for i in range(0, len(full_pcm), chunk_size):
            yield full_pcm[i : i + chunk_size]
            # Yield control to the event loop between chunks
            await asyncio.sleep(0)

    @staticmethod
    def _resample(pcm_data: bytes, from_rate: int, to_rate: int) -> bytes:
        """
        Simple linear resampling of 16-bit PCM audio.

        Args:
            pcm_data: Raw 16-bit PCM bytes.
            from_rate: Source sample rate.
            to_rate: Target sample rate.

        Returns:
            Resampled raw 16-bit PCM bytes.
        """
        import numpy as np

        audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)
        duration = len(audio) / from_rate
        target_samples = int(duration * to_rate)

        indices = np.linspace(0, len(audio) - 1, target_samples)
        resampled = np.interp(indices, np.arange(len(audio)), audio)

        return resampled.astype(np.int16).tobytes()
