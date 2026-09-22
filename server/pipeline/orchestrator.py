"""
Pipeline orchestrator: connects VAD → STT → ConversationManager → TTS
in a streaming async pipeline with sentence-level pipelining.
"""

import asyncio
import logging
import time
import numpy as np
from typing import AsyncGenerator, Optional

from server.audio.vad import VADProcessor
from server.audio.stt import SpeechToText
from server.audio.tts import TextToSpeech
from server.conversation.manager import ConversationManager

logger = logging.getLogger(__name__)


class AudioPipeline:
    """
    End-to-end audio pipeline orchestrator.

    Connects:
        1. VAD (speech detection from raw PCM chunks)
        2. STT (transcription of complete speech segments)
        3. ConversationManager (state machine + LLM)
        4. TTS (speech synthesis of response sentences)

    Implements sentence-level pipelining: TTS starts on sentence 1
    while the LLM is still generating sentence 2.
    """

    def __init__(self, conversation_manager: ConversationManager):
        self.conversation_manager = conversation_manager
        self.vad = VADProcessor()
        self.stt = SpeechToText()
        self.tts = TextToSpeech()

        # Barge-in control
        self._is_responding = False
        self._barge_in_requested = False

        logger.info("AudioPipeline initialized successfully.")

    def reset(self):
        """Reset pipeline state for a new session."""
        self.vad.reset()
        self._is_responding = False
        self._barge_in_requested = False
        logger.info("AudioPipeline reset for new session.")

    @property
    def is_responding(self) -> bool:
        """Whether the pipeline is currently generating/playing a response."""
        return self._is_responding

    def request_barge_in(self):
        """
        Called when speech is detected while a response is being played.
        Signals the pipeline to cancel the current response.
        """
        if self._is_responding:
            self._barge_in_requested = True
            logger.info("Barge-in requested — cancelling current response.")

    async def process_audio_chunk(self, chunk: bytes) -> Optional[str]:
        """
        Process a single audio chunk through VAD and STT.

        Args:
            chunk: Raw 16-bit PCM audio bytes (30ms @ 16kHz = 960 bytes)

        Returns:
            Transcribed text if end-of-speech detected, None otherwise.
        """
        # Check for barge-in: if we're responding and new speech comes in
        if self._is_responding:
            # Still feed the chunk to VAD to detect barge-in speech
            audio_int16 = np.frombuffer(chunk, dtype=np.int16)
            rms = np.sqrt(np.mean(audio_int16.astype(np.float32) ** 2))
            if rms > 500:  # Simple energy threshold for barge-in
                self.request_barge_in()
            return None

        # Feed chunk to VAD
        speech_segment = self.vad.process_chunk(chunk)

        if speech_segment is not None:
            # Complete speech segment detected — transcribe it
            logger.info(
                f"Speech segment detected: {len(speech_segment)} samples "
                f"({len(speech_segment) / 16000:.2f}s)"
            )

            start_time = time.time()

            # Run STT in executor to avoid blocking
            loop = asyncio.get_event_loop()
            transcript = await loop.run_in_executor(
                None, self.stt.transcribe, speech_segment
            )

            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(f"STT completed in {elapsed_ms:.1f}ms: '{transcript}'")

            if transcript.strip():
                return transcript.strip()

        return None

    async def generate_response(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Process transcribed text through the conversation manager and TTS.

        Implements sentence-level pipelining:
        - LLM generates response sentences via streaming
        - Each sentence is immediately sent to TTS
        - TTS audio chunks are yielded as they're ready

        Args:
            text: Transcribed visitor text

        Yields:
            Raw PCM audio bytes (chunks of the TTS response)
        """
        self._is_responding = True
        self._barge_in_requested = False

        pipeline_start = time.time()

        try:
            sentence_count = 0

            async for sentence in self.conversation_manager.process_input(text):
                # Check barge-in before processing each sentence
                if self._barge_in_requested:
                    logger.info("Barge-in: stopping response generation.")
                    break

                if not sentence.strip():
                    continue

                sentence_count += 1
                tts_start = time.time()

                logger.info(
                    f"TTS sentence {sentence_count}: '{sentence}'"
                )

                # Stream TTS audio chunks for this sentence
                async for audio_chunk in self.tts.synthesize_to_chunks(sentence):
                    if self._barge_in_requested:
                        logger.info("Barge-in: stopping audio streaming.")
                        break
                    yield audio_chunk

                tts_elapsed = (time.time() - tts_start) * 1000
                logger.info(
                    f"TTS sentence {sentence_count} completed in {tts_elapsed:.1f}ms"
                )

            total_elapsed = (time.time() - pipeline_start) * 1000
            logger.info(
                f"Full response pipeline completed in {total_elapsed:.1f}ms "
                f"({sentence_count} sentences)"
            )

        except Exception as e:
            logger.error(f"Error in response pipeline: {e}", exc_info=True)
            # Generate error audio
            error_text = "I'm sorry, I encountered an error. Please try again."
            async for chunk in self.tts.synthesize_to_chunks(error_text):
                yield chunk

        finally:
            self._is_responding = False
            self._barge_in_requested = False

    async def process_full_audio(self, audio_data: bytes) -> bytes:
        """
        Process a complete audio recording (for test mode / HTTP upload).

        Takes complete PCM audio, runs STT, processes through conversation,
        and returns complete TTS audio response.

        Args:
            audio_data: Complete raw 16-bit PCM audio bytes (16kHz mono)

        Returns:
            Complete raw PCM audio response bytes
        """
        pipeline_start = time.time()

        # Convert to float32 for STT
        audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        if len(audio_float32) < 1600:  # Less than 0.1s
            logger.warning("Audio too short for processing.")
            return self.tts.synthesize(
                "I couldn't hear you clearly. Could you please repeat?"
            ), ""

        # Transcribe
        stt_start = time.time()
        loop = asyncio.get_event_loop()
        transcript = await loop.run_in_executor(
            None, self.stt.transcribe, audio_float32
        )
        stt_elapsed = (time.time() - stt_start) * 1000
        logger.info(f"STT ({stt_elapsed:.1f}ms): '{transcript}'")

        if not transcript.strip():
            return self.tts.synthesize(
                "I couldn't understand what you said. Could you please repeat?"
            ), ""

        # Process through conversation manager and collect response
        full_response_text = ""
        async for sentence in self.conversation_manager.process_input(transcript):
            full_response_text += sentence + " "

        full_response_text = full_response_text.strip()
        logger.info(f"LLM response: '{full_response_text}'")

        if not full_response_text:
            full_response_text = "I'm processing your request."

        # Synthesize full response
        tts_start = time.time()
        response_audio = await loop.run_in_executor(
            None, self.tts.synthesize, full_response_text
        )
        tts_elapsed = (time.time() - tts_start) * 1000

        total_elapsed = (time.time() - pipeline_start) * 1000
        logger.info(
            f"Full pipeline completed in {total_elapsed:.1f}ms "
            f"(STT: {stt_elapsed:.1f}ms, TTS: {tts_elapsed:.1f}ms)"
        )

        return response_audio, transcript
