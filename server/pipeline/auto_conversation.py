"""
Hands-free doorbell conversation.

Started when the vision server reports that the camera captured a visitor's face.
The ESP32 microphone is ON while the assistant is listening and OFF while it is
thinking or speaking (half-duplex), so the speaker never feeds back into the mic:

    camera captures face ─► greet (mic off) ─► listen (mic on) ─► visitor stops talking
          ▲                                                              │
          └──── speak reply (mic off) ◄── think (LLM + tools) ◄── transcribe

The session ends when the assistant approves entry / takes a message, when the
visitor is silent for `auto_listen_timeout_s`, or after `auto_max_turns` turns.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

from server.config import settings
from server.conversation.states import ConversationState

logger = logging.getLogger(__name__)

VAD_CHUNK_BYTES = 512 * 2          # Silero VAD expects 512-sample frames at 16 kHz
POLL_INTERVAL_S = 0.03
ECHO_GUARD_S = 0.35                # pause after the speaker stops before opening the mic
SPEECH_ENERGY_RMS = 300            # "visitor started talking" hint for the dashboard


class AutoConversation:
    def __init__(self, conversation_manager, pipeline, receiver):
        self.cm = conversation_manager
        self.pipeline = pipeline
        self.receiver = receiver
        self.enabled: bool = settings.auto_conversation
        self.state: str = "idle"          # idle | greeting | listening | hearing | thinking | speaking | error
        self.detail: str = ""
        self.identity: Dict[str, Any] = {}
        self.turns = 0
        self.started_at: Optional[str] = None
        self.last_end_time = 0.0
        self._task: Optional[asyncio.Task] = None

    # ── Public API ───────────────────────────────────────────────────────────
    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "running": self.running,
            "state": self.state,
            "mic_on": self.state in ("listening", "hearing"),
            "detail": self.detail,
            "identity": self.identity,
            "turns": self.turns,
            "started_at": self.started_at,
            "esp32_connected": self.receiver.is_connected,
        }

    def on_vision_event(self, event: Dict[str, Any]) -> bool:
        """Start a conversation when the camera has captured a visitor. Returns True if started."""
        if not self.enabled or event.get("type") != "visit_captured" or self.running:
            return False
        identity = event.get("identity") or {}
        if settings.auto_known_only and not identity.get("known"):
            logger.info("Auto conversation: visitor not recognised, auto_known_only is set — not starting.")
            return False
        if time.time() - self.last_end_time < settings.auto_cooldown_s:
            logger.info("Auto conversation: still in cooldown after the last visitor — not starting.")
            return False
        return self.start(identity, reason=f"camera visit #{event.get('visit_id')}")

    def start(self, identity: Optional[Dict[str, Any]] = None, reason: str = "manual") -> bool:
        if self.running:
            return False
        self.identity = identity or {}
        self._task = asyncio.create_task(self._run(reason))
        return True

    async def stop(self):
        if self.running:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

    # ── Conversation loop ────────────────────────────────────────────────────
    def _set(self, state: str, detail: str = ""):
        self.state, self.detail = state, detail
        logger.info(f"Auto conversation: {state}{' — ' + detail if detail else ''}")

    async def _run(self, reason: str):
        self.turns = 0
        self.started_at = datetime.now().isoformat(timespec="seconds")
        try:
            if not await asyncio.to_thread(self.receiver._ensure_connected):
                self._set("error", "ESP32 mic/speaker not connected over USB")
                return

            await self.cm.start_session()
            self.pipeline.reset()
            self._set("greeting", f"started by {reason}")
            await self._respond("")

            while self._session_open() and self.turns < settings.auto_max_turns:
                transcript = await self._listen()
                if transcript is None:
                    self._set("speaking", "no reply from visitor — ending")
                    await self._speak("I didn't hear anything, so I'll end the conversation. Goodbye.")
                    break
                if not transcript:
                    continue
                self.turns += 1
                await self._respond(transcript)

                if self.cm.current_session and self.cm.current_session.state in (
                        ConversationState.APPROVED, ConversationState.ENDED):
                    break
        except asyncio.CancelledError:
            self._set("idle", "stopped")
            raise
        except Exception as e:
            logger.error(f"Auto conversation failed: {e}", exc_info=True)
            self._set("error", str(e))
        finally:
            await asyncio.to_thread(self._mic_off)
            if self.cm.is_session_active():
                await self.cm.end_session()
            self.last_end_time = time.time()
            if self.state != "error":
                self._set("idle", "conversation finished")

    def _session_open(self) -> bool:
        return self.cm.is_session_active()

    async def _respond(self, text: str):
        """Run the visitor's words through the LLM, then speak the reply (mic stays off)."""
        self._set("thinking", f"visitor said: {text}" if text else "preparing greeting")
        sentences = []
        async for sentence in self.cm.process_input(text):
            if sentence.strip():
                sentences.append(sentence.strip())
        reply = " ".join(sentences) or "Sorry, could you repeat that?"
        await self._speak(reply)

    async def _speak(self, text: str):
        self._set("speaking", text)
        pcm = await asyncio.to_thread(self.pipeline.tts.synthesize, text)
        ok = await asyncio.to_thread(self.receiver.play_audio, pcm)
        if not ok:
            logger.warning("ESP32 speaker playback failed.")
        await asyncio.sleep(ECHO_GUARD_S)

    def _mic_off(self):
        if self.receiver.is_recording:
            self.receiver.stop_recording()

    async def _listen(self) -> Optional[str]:
        """
        Mic on until the visitor finishes a sentence (Silero VAD end-of-speech).
        Returns the transcript, "" for unintelligible speech, or None on silence timeout.
        """
        self.pipeline.vad.reset()
        if not await asyncio.to_thread(self.receiver.start_recording):
            raise RuntimeError("could not start ESP32 microphone")
        self._set("listening", "mic on")
        started = time.time()
        heard_speech = False
        pending = b""
        try:
            while True:
                await asyncio.sleep(POLL_INTERVAL_S)
                data = self.receiver.drain()
                if data:
                    # Drop the firmware's text acknowledgement if it lands in the audio
                    data = data.replace(b"REC_OK\r\n", b"").replace(b"REC_OK\n", b"")
                    pending += data
                while len(pending) >= VAD_CHUNK_BYTES:
                    chunk, pending = pending[:VAD_CHUNK_BYTES], pending[VAD_CHUNK_BYTES:]
                    if not heard_speech:
                        rms = float(np.sqrt(np.mean(np.frombuffer(chunk, np.int16).astype(np.float32) ** 2)))
                        if rms > SPEECH_ENERGY_RMS:
                            heard_speech = True
                            self._set("hearing", "visitor is speaking")
                    segment = self.pipeline.vad.process_chunk(chunk)
                    if segment is not None:
                        await asyncio.to_thread(self._mic_off)
                        self._set("thinking", "transcribing")
                        text = await asyncio.to_thread(self.pipeline.stt.transcribe, segment)
                        return (text or "").strip()

                if not heard_speech and time.time() - started > settings.auto_listen_timeout_s:
                    return None
                if time.time() - started > settings.auto_listen_timeout_s + 30:
                    return None  # hard cap for one turn (continuous noise)
        finally:
            await asyncio.to_thread(self._mic_off)
