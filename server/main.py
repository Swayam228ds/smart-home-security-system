"""
Home Security Audio Pipeline — FastAPI Server

Endpoints:
    WebSocket /ws/audio       — Real-time audio streaming (for ESP32)
    POST     /api/test/audio  — Upload WAV file for testing
    GET      /api/sessions    — Current session info
    POST     /api/sessions    — Start a new session
    DELETE   /api/sessions    — End current session
    GET      /api/appointments       — List appointments
    POST     /api/appointments       — Add appointment
    DELETE   /api/appointments       — Remove appointment
    GET      /api/owners/status      — Owner status
    PUT      /api/owners/status      — Update owner status
    GET      /api/health             — Health check
"""

import io
import wave
import json
import struct
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from server.config import settings, SAMPLE_RATE, CHANNELS, SAMPLE_WIDTH, CHUNK_SIZE
from server.audio.vad import VADProcessor
from server.audio.stt import SpeechToText
from server.audio.tts import TextToSpeech
from server.conversation.llm import OllamaClient
from server.conversation.manager import ConversationManager
from server.security.appointments import AppointmentManager, Appointment
from server.security.owners import OwnerManager
from server.pipeline.orchestrator import AudioPipeline
from server.database.session import init_db

# ──────────────────────────────────────────────
# Logging Setup
# ──────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Global instances (initialized at startup)
# ──────────────────────────────────────────────
appointment_manager: Optional[AppointmentManager] = None
owner_manager: Optional[OwnerManager] = None
llm_client: Optional[OllamaClient] = None
conversation_manager: Optional[ConversationManager] = None
pipeline: Optional[AudioPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Loads all ML models and initializes managers at startup.
    """
    global appointment_manager, owner_manager, llm_client
    global conversation_manager, pipeline

    logger.info("=" * 60)
    logger.info("Home Security Audio Pipeline — Starting Up")
    logger.info("=" * 60)

    startup_start = time.time()

    # Initialize DB
    logger.info("Initializing Database...")
    await init_db()

    # 1. Initialize security data managers
    logger.info("Loading appointment & owner data...")
    appointment_manager = AppointmentManager()
    owner_manager = OwnerManager()

    # 2. Initialize Ollama LLM client
    logger.info("Initializing Ollama LLM client...")
    llm_client = OllamaClient()

    # 3. Initialize conversation manager
    conversation_manager = ConversationManager(
        appointment_manager=appointment_manager,
        owner_manager=owner_manager,
    )

    # 4. Initialize audio pipeline (loads VAD, STT, TTS models)
    logger.info("Loading ML models (VAD, Whisper, Piper)...")
    pipeline = AudioPipeline(conversation_manager)

    startup_elapsed = time.time() - startup_start
    logger.info(f"All systems ready in {startup_elapsed:.1f}s")
    logger.info("=" * 60)

    yield  # Application runs here

    # Shutdown
    logger.info("Shutting down...")
    if conversation_manager and conversation_manager.is_session_active():
        await conversation_manager.end_session()
    logger.info("Shutdown complete.")


# ──────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────
app = FastAPI(
    title="Home Security Audio Pipeline",
    description="Real-time audio processing for home security visitor management",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Pydantic Models for API
# ──────────────────────────────────────────────
class AppointmentCreate(BaseModel):
    visitor_name: str
    purpose: str
    date: str
    time_range: str = "all_day"
    approved: bool = True


class AppointmentDelete(BaseModel):
    visitor_name: str
    date: str


class OwnerStatusUpdate(BaseModel):
    name: str
    status: str  # 'home', 'away', 'busy', 'sleeping'


# ──────────────────────────────────────────────
# WebSocket Endpoint — Real-time Audio
# ──────────────────────────────────────────────
@app.websocket("/ws/audio")
async def websocket_audio(ws: WebSocket):
    """
    Real-time audio WebSocket endpoint.

    Protocol:
        Client → Server: Binary PCM frames (16kHz, 16-bit, mono, 30ms chunks)
        Server → Client: Binary PCM frames (TTS response audio)
        Server → Client: JSON control messages:
            {"event": "session_started", "session_id": "..."}
            {"event": "transcript", "text": "..."}
            {"event": "response_start"}
            {"event": "response_end"}
            {"event": "barge_in"}
            {"event": "session_ended"}
            {"event": "error", "message": "..."}
    """
    await ws.accept()
    logger.info("WebSocket client connected.")

    try:
        # Start a new conversation session
        session_id = await conversation_manager.start_session()
        pipeline.reset()

        await ws.send_json({
            "event": "session_started",
            "session_id": session_id,
        })

        # Generate and send initial greeting
        greeting_task = asyncio.create_task(
            _send_greeting(ws)
        )
        await greeting_task

        # Main audio processing loop
        while True:
            try:
                data = await asyncio.wait_for(ws.receive_bytes(), timeout=300)
            except asyncio.TimeoutError:
                logger.info("WebSocket timed out (5 min inactivity).")
                await ws.send_json({"event": "session_ended", "reason": "timeout"})
                break

            # Process audio chunk through pipeline
            transcript = await pipeline.process_audio_chunk(data)

            if transcript:
                logger.info(f"Visitor said: '{transcript}'")
                await ws.send_json({
                    "event": "transcript",
                    "text": transcript,
                })

                # Check if session is still active
                if not conversation_manager.is_session_active():
                    await ws.send_json({"event": "session_ended", "reason": "completed"})
                    break

                # Generate and stream response
                await ws.send_json({"event": "response_start"})

                async for audio_chunk in pipeline.generate_response(transcript):
                    await ws.send_bytes(audio_chunk)

                await ws.send_json({"event": "response_end"})

                # Check if conversation ended
                if (conversation_manager.current_session and
                    conversation_manager.current_session.state.name == "ENDED"):
                    await ws.send_json({"event": "session_ended", "reason": "completed"})
                    break

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await ws.send_json({"event": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        if conversation_manager.is_session_active():
            await conversation_manager.end_session()
        logger.info("WebSocket session ended.")


async def _send_greeting(ws: WebSocket):
    """Generate and send the initial greeting audio."""
    try:
        await ws.send_json({"event": "response_start"})

        greeting_text = ""
        async for sentence in conversation_manager.process_input(""):
            greeting_text += sentence + " "
            # Stream each sentence as audio
            async for audio_chunk in pipeline.tts.synthesize_to_chunks(sentence):
                await ws.send_bytes(audio_chunk)

        await ws.send_json({"event": "response_end"})
        logger.info(f"Greeting sent: '{greeting_text.strip()}'")

    except Exception as e:
        logger.error(f"Error sending greeting: {e}")


# ──────────────────────────────────────────────
# REST Test Endpoint — Audio Upload
# ──────────────────────────────────────────────
@app.post("/api/test/audio", summary="Upload audio file for testing")
async def test_audio_upload(file: UploadFile = File(...)):
    """
    Test endpoint: upload a WAV audio file, get audio response.

    Accepts: WAV file (16kHz mono preferred, will attempt conversion)
    Returns: WAV file with the assistant's response
    """
    if not file.filename.lower().endswith(('.wav', '.pcm', '.raw')):
        raise HTTPException(400, "Only WAV, PCM, and RAW audio files are supported.")

    content = await file.read()

    # Parse audio data
    try:
        if file.filename.lower().endswith('.wav'):
            pcm_data = _extract_pcm_from_wav(content)
        else:
            # Assume raw PCM
            pcm_data = content
    except Exception as e:
        # If it's an empty file, just use empty pcm
        pcm_data = b""

    # Start session if needed (or if they sent empty audio)
    if not conversation_manager.is_session_active() or len(pcm_data) < 100:
        if not conversation_manager.is_session_active():
            session_id = await conversation_manager.start_session()
            pipeline.reset()

        # Generate greeting first and return it immediately
        greeting_audio = b""
        async for sentence in conversation_manager.process_input(""):
            greeting_pcm = pipeline.tts.synthesize(sentence)
            greeting_audio += greeting_pcm
            
        response_wav = _pcm_to_wav(greeting_audio)
        return Response(
            content=response_wav,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=greeting.wav",
                "X-Session-Active": "True",
                "X-Session-State": "GREETING",
            },
        )

    # Process through pipeline
    response_pcm, transcript = await pipeline.process_full_audio(pcm_data)

    # Package as WAV
    response_wav = _pcm_to_wav(response_pcm)

    # Note: HTTP headers can only contain ASCII, so encode the transcript safely
    safe_transcript = transcript.encode('ascii', 'ignore').decode('ascii') if transcript else ""

    return Response(
        content=response_wav,
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=response.wav",
            "X-Session-Active": str(conversation_manager.is_session_active()),
            "X-Session-State": (
                conversation_manager.current_session.state.name
                if conversation_manager.current_session else "NONE"
            ),
            "X-Transcript": safe_transcript
        },
    )


@app.post("/api/test/text", summary="Send text input for testing")
async def test_text_input(text: str):
    """
    Test endpoint: send text directly (bypasses STT).
    Returns JSON with the assistant's text response and audio as WAV.
    """
    response_sentences = []
    response_audio = b""

    # Start session if needed
    if not conversation_manager.is_session_active():
        session_id = await conversation_manager.start_session()
        pipeline.reset()

        # If it's a new session, get the greeting FIRST and return it
        async for sentence in conversation_manager.process_input(""):
            response_sentences.append(sentence)
            sentence_pcm = pipeline.tts.synthesize(sentence)
            response_audio += sentence_pcm
            
        full_response = " ".join(response_sentences)
        
        return JSONResponse({
            "response_text": full_response,
            "session_active": conversation_manager.is_session_active(),
            "session_state": (
                conversation_manager.current_session.state.name
                if conversation_manager.current_session else "NONE"
            ),
            "visitor_name": (
                conversation_manager.current_session.visitor_name
                if conversation_manager.current_session else None
            ),
            "audio_available": len(response_audio) > 0,
        })

    # If session is already active, process the user's text normally
    async for sentence in conversation_manager.process_input(text):
        response_sentences.append(sentence)
        sentence_pcm = pipeline.tts.synthesize(sentence)
        response_audio += sentence_pcm

    full_response = " ".join(response_sentences)

    return JSONResponse({
        "response_text": full_response,
        "session_active": conversation_manager.is_session_active(),
        "session_state": (
            conversation_manager.current_session.state.name
            if conversation_manager.current_session else "NONE"
        ),
        "visitor_name": (
            conversation_manager.current_session.visitor_name
            if conversation_manager.current_session else None
        ),
        "audio_available": len(response_audio) > 0,
    })


# ──────────────────────────────────────────────
# Session Management Endpoints
# ──────────────────────────────────────────────
@app.get("/api/sessions", summary="Get current session info")
async def get_session():
    """Returns information about the current conversation session."""
    if not conversation_manager.current_session:
        return {"active": False}

    ctx = conversation_manager.current_session
    return {
        "active": conversation_manager.is_session_active(),
        "session_id": ctx.session_id,
        "state": ctx.state.name,
        "visitor_name": ctx.visitor_name,
        "visitor_purpose": ctx.visitor_purpose,
        "message_count": len(ctx.messages),
        "created_at": ctx.created_at.isoformat(),
        "last_activity": ctx.last_activity.isoformat(),
    }


@app.post("/api/sessions", summary="Start a new session")
async def start_session():
    """Starts a new conversation session (ends any existing one)."""
    session_id = await conversation_manager.start_session()
    pipeline.reset()
    return {"session_id": session_id, "state": "GREETING"}


@app.delete("/api/sessions", summary="End current session")
async def end_session():
    """Ends the current conversation session."""
    if conversation_manager.is_session_active():
        await conversation_manager.end_session()
        return {"ended": True}
    return {"ended": False, "message": "No active session"}


# ──────────────────────────────────────────────
# Appointment Endpoints
# ──────────────────────────────────────────────
@app.get("/api/appointments", summary="List all appointments")
async def list_appointments():
    """Returns all scheduled appointments."""
    appointments = appointment_manager.get_all_appointments()
    return {"appointments": [a.model_dump() for a in appointments]}


@app.post("/api/appointments", summary="Add an appointment")
async def add_appointment(data: AppointmentCreate):
    """Adds a new appointment."""
    appt = Appointment(**data.model_dump())
    appointment_manager.add_appointment(appt)
    return {"message": "Appointment added", "appointment": appt.model_dump()}


@app.delete("/api/appointments", summary="Remove an appointment")
async def remove_appointment(data: AppointmentDelete):
    """Removes an appointment by visitor name and date."""
    removed = appointment_manager.remove_appointment(data.visitor_name, data.date)
    if removed:
        return {"message": "Appointment removed"}
    raise HTTPException(404, "Appointment not found")


# ──────────────────────────────────────────────
# Owner Status Endpoints
# ──────────────────────────────────────────────
@app.get("/api/owners/status", summary="Get owner status")
async def get_owner_status():
    """Returns status of all homeowners."""
    owners = owner_manager.get_all_owners()
    return {
        "owners": [o.model_dump() for o in owners],
        "anyone_home": owner_manager.is_anyone_home(),
    }


@app.put("/api/owners/status", summary="Update owner status")
async def update_owner_status(data: OwnerStatusUpdate):
    """Updates a homeowner's status."""
    valid_statuses = {"home", "away", "busy", "sleeping"}
    if data.status not in valid_statuses:
        raise HTTPException(400, f"Status must be one of: {valid_statuses}")

    owner = owner_manager.update_status(data.name, data.status)
    if owner:
        return {"message": "Status updated", "owner": owner.model_dump()}
    raise HTTPException(404, f"Owner '{data.name}' not found")


# ──────────────────────────────────────────────
# Health Check
# ──────────────────────────────────────────────
@app.get("/api/health", summary="Health check")
async def health_check():
    """Returns server health and component status."""
    ollama_healthy = False
    if llm_client:
        ollama_healthy = await llm_client.health_check()

    return {
        "status": "healthy" if ollama_healthy else "degraded",
        "components": {
            "vad": pipeline is not None,
            "stt": pipeline is not None,
            "tts": pipeline is not None,
            "llm": ollama_healthy,
            "appointments": appointment_manager is not None,
        },
        "session_active": (
            conversation_manager.is_session_active()
            if conversation_manager else False
        ),
        "config": {
            "whisper_model": settings.whisper_model_size,
            "whisper_device": settings.whisper_device,
            "piper_voice": settings.piper_voice,
            "llm_provider": settings.llm_provider,
            "ollama_model": settings.ollama_model,
            "sample_rate": SAMPLE_RATE,
        },
    }


# ──────────────────────────────────────────────
# Audio Utility Functions
# ──────────────────────────────────────────────
def _extract_pcm_from_wav(wav_bytes: bytes) -> bytes:
    """Extract raw PCM data from a WAV file, resampling if needed."""
    wav_io = io.BytesIO(wav_bytes)

    with wave.open(wav_io, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)

    # Convert to numpy array
    if sampwidth == 2:
        audio = np.frombuffer(raw_data, dtype=np.int16)
    elif sampwidth == 4:
        audio = np.frombuffer(raw_data, dtype=np.int32)
        audio = (audio / 2**16).astype(np.int16)
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    # Convert stereo to mono
    if n_channels == 2:
        audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)
    elif n_channels > 2:
        audio = audio.reshape(-1, n_channels)[:, 0].astype(np.int16)

    # Resample to 16kHz if needed
    if framerate != SAMPLE_RATE:
        # Simple linear resampling
        duration = len(audio) / framerate
        target_samples = int(duration * SAMPLE_RATE)
        indices = np.linspace(0, len(audio) - 1, target_samples)
        audio = np.interp(indices, np.arange(len(audio)), audio.astype(np.float32))
        audio = audio.astype(np.int16)

    return audio.tobytes()


def _pcm_to_wav(pcm_data: bytes) -> bytes:
    """Package raw PCM data into a WAV file."""
    wav_io = io.BytesIO()

    with wave.open(wav_io, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_data)

    return wav_io.getvalue()


# ──────────────────────────────────────────────
# Static Files & Root Page
# ──────────────────────────────────────────────
STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    """Serve the test console HTML page."""
    index_file = STATIC_DIR / "index.html"
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=False,
    )
