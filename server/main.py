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

Combined dashboard (video + audio):
    GET      /                       — Combined security dashboard (old console at /console)
    *        /api/vision/{path}      — Proxy to the vision server API (VISION_URL)
    GET      /api/vision/video_feed  — Proxied annotated ESP32-CAM MJPEG stream
    POST     /api/events/vision      — Vision server pushes visit / recognition events here
    GET      /api/events             — Recent vision events
    GET      /api/sessions/transcript — Live conversation transcript
    GET      /api/visit_logs         — Past conversation sessions from the database
"""

import io
import wave
import json
import struct
import asyncio
import logging
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse, HTMLResponse, StreamingResponse
from sqlalchemy import select
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
from server.pipeline.auto_conversation import AutoConversation
from server.database.session import init_db, async_session
from server.database.models import VisitLog

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
auto_conv: Optional[AutoConversation] = None
periodic_timeout_task = None

async def _periodic_timeout_check():
    """Background task to periodically clean up timed-out sessions and log them to the DB."""
    while True:
        await asyncio.sleep(60)
        try:
            if conversation_manager:
                conversation_manager._check_timeout()
        except Exception as e:
            logger.error(f"Error in periodic timeout check: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Loads all ML models and initializes managers at startup.
    """
    global appointment_manager, owner_manager, llm_client
    global conversation_manager, pipeline, auto_conv

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

    from server.audio.esp32_receiver import esp32_receiver

    # 4. Initialize audio pipeline (loads VAD, STT, TTS models)
    logger.info("Loading ML models (VAD, Whisper, Piper)...")
    pipeline = AudioPipeline(conversation_manager)
    auto_conv = AutoConversation(conversation_manager, pipeline, esp32_receiver)
    logger.info(f"Hands-free conversation: {'ON' if auto_conv.enabled else 'OFF'} (starts when the camera captures a visitor)")

    # 5. Start ESP32 UDP receiver task
    logger.info("Starting ESP32 UDP Receiver on port 5005...")
    asyncio.create_task(esp32_receiver.start_listening())

    # 6. Start periodic timeout check
    global periodic_timeout_task
    periodic_timeout_task = asyncio.create_task(_periodic_timeout_check())

    startup_elapsed = time.time() - startup_start
    logger.info(f"All systems ready in {startup_elapsed:.1f}s")
    logger.info("=" * 60)

    yield  # Application runs here

    # Shutdown
    logger.info("Shutting down...")
    if periodic_timeout_task:
        periodic_timeout_task.cancel()
    if auto_conv:
        await auto_conv.stop()
    if conversation_manager and conversation_manager.is_session_active():
        await conversation_manager.end_session()
    if _vision_client is not None:
        await _vision_client.aclose()
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


class OwnerPresenceToggle(BaseModel):
    is_home: bool


class AppointmentToggleApproval(BaseModel):
    visitor_name: str
    date: str


class VisitorApprovalRequest(BaseModel):
    approved: bool
    custom_message: Optional[str] = None


class SpeakMessageRequest(BaseModel):
    message: str


class SimulateArrivalRequest(BaseModel):
    visitor_name: str
    purpose: str = "Visit"


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
async def test_audio_upload(file: UploadFile = File(...), target_speaker: str = "laptop"):
    """
    Test endpoint: upload a WAV audio file, get audio response.

    Accepts: WAV file (16kHz mono preferred, will attempt conversion)
    Returns: WAV file with the assistant's response
    """
    from server.audio.esp32_receiver import esp32_receiver

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
            
        if target_speaker == "esp32":
            esp32_receiver.play_audio(greeting_audio)

        response_wav = _pcm_to_wav(greeting_audio)
        return Response(
            content=response_wav,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=greeting.wav",
                "X-Session-Active": "True",
                "X-Session-State": "GREETING",
                "X-Played-On": target_speaker,
            },
        )

    # Process through pipeline
    response_pcm, transcript = await pipeline.process_full_audio(pcm_data)

    if target_speaker == "esp32":
        esp32_receiver.play_audio(response_pcm)

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
            "X-Transcript": safe_transcript,
            "X-Played-On": target_speaker,
        },
    )

@app.post("/api/test/esp32/start", summary="Start recording from ESP32")
async def esp32_start_recording():
    from server.audio.esp32_receiver import esp32_receiver
    if not esp32_receiver._ensure_connected():
        raise HTTPException(status_code=400, detail="ESP32 not connected. Please ensure it is plugged in and the Arduino Serial Monitor is closed.")
    esp32_receiver.start_recording()
    return {"status": "recording"}

@app.post("/api/test/esp32/stop", summary="Stop recording from ESP32 and process")
async def esp32_stop_recording(target_speaker: str = "esp32"):
    from server.audio.esp32_receiver import esp32_receiver
    pcm_data = esp32_receiver.stop_recording()
    
    if not conversation_manager.is_session_active() or len(pcm_data) < 100:
        if not conversation_manager.is_session_active():
            await conversation_manager.start_session()
            pipeline.reset()
            
        # Return greeting if empty or new session
        greeting_audio = b""
        async for sentence in conversation_manager.process_input(""):
            greeting_pcm = pipeline.tts.synthesize(sentence)
            greeting_audio += greeting_pcm
            
        played = False
        if target_speaker == "esp32" and greeting_audio:
            played = esp32_receiver.play_audio(greeting_audio)

        response_wav = _pcm_to_wav(greeting_audio)
        return Response(
            content=response_wav,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=greeting.wav",
                "X-Session-Active": "True",
                "X-Session-State": "GREETING",
                "X-Played-On": target_speaker,
                "X-Played-Success": str(played),
            },
        )

    # Process through pipeline
    response_pcm, transcript = await pipeline.process_full_audio(pcm_data)

    played = False
    if target_speaker == "esp32" and response_pcm:
        played = esp32_receiver.play_audio(response_pcm)

    response_wav = _pcm_to_wav(response_pcm)
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
            "X-Transcript": safe_transcript,
            "X-Played-On": target_speaker,
            "X-Played-Success": str(played),
        },
    )

@app.post("/api/test/esp32/greeting", summary="Trigger and play greeting through ESP32 speaker")
async def esp32_play_greeting():
    """Generates greeting and plays it directly through ESP32 MAX98357A speaker."""
    from server.audio.esp32_receiver import esp32_receiver
    if not conversation_manager.is_session_active():
        await conversation_manager.start_session()
        pipeline.reset()

    greeting_text = ""
    greeting_audio = b""
    async for sentence in conversation_manager.process_input(""):
        greeting_text += sentence + " "
        greeting_pcm = pipeline.tts.synthesize(sentence)
        greeting_audio += greeting_pcm

    greeting_text = greeting_text.strip()
    if not greeting_text:
        greeting_text = "Greeting visitor, please state your name and the purpose of your visit."
        greeting_audio = pipeline.tts.synthesize(greeting_text)

    played = esp32_receiver.play_audio(greeting_audio)
    return {
        "status": "ok",
        "response_text": greeting_text,
        "played_on_esp32": played,
        "bytes_sent": len(greeting_audio),
        "session_active": conversation_manager.is_session_active(),
    }

@app.post("/api/test/text", summary="Send text input for testing")
async def test_text_input(text: str, target_speaker: str = "laptop"):
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
        
        played_on_esp32 = False
        if target_speaker == "esp32" and response_audio:
            from server.audio.esp32_receiver import esp32_receiver
            played_on_esp32 = esp32_receiver.play_audio(response_audio)

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
            "played_on_esp32": played_on_esp32,
            "played_on": target_speaker,
        })

    # If session is already active, process the user's text normally
    async for sentence in conversation_manager.process_input(text):
        response_sentences.append(sentence)
        sentence_pcm = pipeline.tts.synthesize(sentence)
        response_audio += sentence_pcm

    full_response = " ".join(response_sentences)

    played_on_esp32 = False
    if target_speaker == "esp32" and response_audio:
        from server.audio.esp32_receiver import esp32_receiver
        played_on_esp32 = esp32_receiver.play_audio(response_audio)

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
        "played_on_esp32": played_on_esp32,
        "played_on": target_speaker,
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


@app.put("/api/appointments/toggle-approval", summary="Toggle appointment approval status")
async def toggle_appointment_approval(data: AppointmentToggleApproval):
    """Toggles whether an appointment is approved or pending."""
    appts = appointment_manager.get_all_appointments()
    for a in appts:
        if a.visitor_name.lower() == data.visitor_name.lower() and a.date == data.date:
            a.approved = not a.approved
            appointment_manager._save()
            return {"message": "Approval updated", "appointment": a.model_dump()}
    raise HTTPException(404, "Appointment not found")


@app.post("/api/owners/toggle-presence", summary="Toggle homeowner presence (home/away)")
async def toggle_owner_presence(data: OwnerPresenceToggle):
    """Toggles whether the homeowner is currently at home or away."""
    status = "home" if data.is_home else "away"
    owners = owner_manager.get_all_owners()
    target_name = owners[0].name if owners else "Owner"
    owner = owner_manager.update_status(target_name, status)
    return {
        "status": status,
        "is_home": data.is_home,
        "anyone_home": owner_manager.is_anyone_home(),
        "owner": owner.model_dump() if owner else None,
    }


# Track live visitor state for dashboard decisions
active_visitor_state: Dict[str, Any] = {
    "visitor_name": None,
    "visitor_purpose": None,
    "matched_appointment": None,
    "has_appointment": False,
    "arrival_time": None,
    "verdict": None,  # "APPROVED", "DENIED", "MESSAGE_SENT", or None
    "last_spoken_message": None,
}


@app.get("/api/visitor/live-status", summary="Get live visitor status, appointment match, and pending decision")
async def get_live_visitor_status():
    """
    Returns real-time visitor presence, appointment check, homeowner presence,
    and whether an approval prompt (if Home) or message prompt (if Away) is needed.
    """
    global active_visitor_state
    anyone_home = owner_manager.is_anyone_home()

    # 1. Active conversation session from doorbell or manual
    session_active = conversation_manager.is_session_active() if conversation_manager else False
    v_name = None
    v_purpose = None
    matched_appt = None
    session_state = "IDLE"

    if session_active and conversation_manager.current_session:
        sess = conversation_manager.current_session
        v_name = sess.visitor_name
        v_purpose = sess.visitor_purpose
        session_state = sess.state.name
        if sess.matched_appointment:
            matched_appt = sess.matched_appointment.model_dump()

    # 2. Check if vision / auto-conversation recognized someone
    if not v_name and auto_conv and auto_conv.running and auto_conv.identity:
        v_name = auto_conv.identity.get("name")
        v_purpose = "Visitor at door"

    # 3. Check if simulated or manual visitor arrival exists
    if not v_name and active_visitor_state["visitor_name"]:
        v_name = active_visitor_state["visitor_name"]
        v_purpose = active_visitor_state["visitor_purpose"]
        matched_appt = active_visitor_state.get("matched_appointment")

    # Match appointment if name is present but appointment wasn't matched yet
    if v_name and not matched_appt:
        found = appointment_manager.find_appointment(v_name, v_purpose or "")
        if found:
            matched_appt = found.model_dump()

    has_visitor = v_name is not None
    has_appointment = matched_appt is not None

    # Determine prompt type needed
    prompt_type = None
    if has_visitor and has_appointment and not active_visitor_state.get("verdict"):
        if anyone_home:
            prompt_type = "approval_needed"
        else:
            prompt_type = "message_needed"

    return {
        "has_visitor": has_visitor,
        "visitor_name": v_name,
        "visitor_purpose": v_purpose,
        "session_active": session_active,
        "session_state": session_state,
        "has_appointment": has_appointment,
        "matched_appointment": matched_appt,
        "is_anyone_home": anyone_home,
        "prompt_type": prompt_type,
        "verdict": active_visitor_state.get("verdict"),
        "last_spoken_message": active_visitor_state.get("last_spoken_message"),
    }


@app.post("/api/visitor/approve", summary="Homeowner approves or denies visitor entry")
async def approve_visitor_entry(req: VisitorApprovalRequest):
    """
    Homeowner approves or denies visitor entry from the dashboard.
    Plays confirmation/denial audio directly over the ESP32 MAX98357A door speaker.
    """
    global active_visitor_state
    from server.audio.esp32_receiver import esp32_receiver

    visitor_name = active_visitor_state.get("visitor_name") or "Visitor"
    if conversation_manager and conversation_manager.current_session and conversation_manager.current_session.visitor_name:
        visitor_name = conversation_manager.current_session.visitor_name

    if req.approved:
        active_visitor_state["verdict"] = "APPROVED"
        speech_text = (
            req.custom_message.strip()
            if req.custom_message
            else f"Welcome {visitor_name}. Entry approved by homeowner, please come in!"
        )
        if conversation_manager and conversation_manager.current_session:
            conversation_manager.current_session.state = ConversationState.APPROVED
            conversation_manager.current_session.messages.append({
                "role": "assistant",
                "content": speech_text
            })
    else:
        active_visitor_state["verdict"] = "DENIED"
        speech_text = (
            req.custom_message.strip()
            if req.custom_message
            else f"Entry is not approved at this time. Thank you, goodbye."
        )
        if conversation_manager and conversation_manager.current_session:
            conversation_manager.current_session.state = ConversationState.DENIED
            conversation_manager.current_session.messages.append({
                "role": "assistant",
                "content": speech_text
            })

    # Synthesize and broadcast through ESP32 speaker
    pcm = pipeline.tts.synthesize(speech_text)
    played = esp32_receiver.play_audio(pcm)
    active_visitor_state["last_spoken_message"] = {
        "text": speech_text,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "played_on_esp32": played,
    }

    return {
        "status": "ok",
        "verdict": active_visitor_state["verdict"],
        "spoken_message": speech_text,
        "played_on_esp32": played,
    }


@app.post("/api/visitor/speak-message", summary="Speak a custom message typed by homeowner over ESP32 door speaker")
async def speak_visitor_message(req: SpeakMessageRequest):
    """
    Homeowner types a custom message (e.g. when away) and broadcasts it
    directly to the visitor waiting at the front door through the ESP32 MAX98357A speaker.
    """
    global active_visitor_state
    from server.audio.esp32_receiver import esp32_receiver

    text = req.message.strip()
    if not text:
        raise HTTPException(400, "Message cannot be empty")

    pcm = pipeline.tts.synthesize(text)
    played = esp32_receiver.play_audio(pcm)

    active_visitor_state["verdict"] = "MESSAGE_SENT"
    active_visitor_state["last_spoken_message"] = {
        "text": text,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "played_on_esp32": played,
    }

    if conversation_manager and conversation_manager.current_session:
        conversation_manager.current_session.messages.append({
            "role": "assistant",
            "content": f"[Homeowner Broadcast to Visitor]: {text}",
        })

    return {
        "status": "ok",
        "spoken_message": text,
        "played_on_esp32": played,
    }


@app.post("/api/visitor/simulate-arrival", summary="Simulate visitor arriving at the door for dashboard testing")
async def simulate_visitor_arrival(req: SimulateArrivalRequest):
    """
    Simulates a visitor arriving at the front door with an appointment.
    Instantly triggers the homeowner prompt (Approval if Home, Message if Away).
    """
    global active_visitor_state
    appt = appointment_manager.find_appointment(req.visitor_name, req.purpose)
    if not appt:
        for a in appointment_manager.get_all_appointments():
            if (req.visitor_name.lower() in a.visitor_name.lower() or
                a.visitor_name.lower() in req.visitor_name.lower()):
                appt = a
                break

    if not conversation_manager.is_session_active():
        await conversation_manager.start_session()
        pipeline.reset()

    sess = conversation_manager.current_session
    if sess:
        sess.visitor_name = req.visitor_name
        sess.visitor_purpose = req.purpose
        if appt:
            sess.matched_appointment = appt

    active_visitor_state = {
        "visitor_name": req.visitor_name,
        "visitor_purpose": req.purpose,
        "matched_appointment": appt.model_dump() if appt else None,
        "has_appointment": appt is not None,
        "arrival_time": datetime.now().isoformat(timespec="seconds"),
        "verdict": None,
        "last_spoken_message": None,
    }
    return {
        "status": "simulated",
        "visitor": active_visitor_state,
        "is_anyone_home": owner_manager.is_anyone_home(),
    }


@app.post("/api/visitor/clear-arrival", summary="Clear current visitor from door")
async def clear_visitor_arrival():
    """Resets the visitor at the door and closes any test session."""
    global active_visitor_state
    if conversation_manager and conversation_manager.is_session_active():
        await conversation_manager.end_session()
    active_visitor_state = {
        "visitor_name": None,
        "visitor_purpose": None,
        "matched_appointment": None,
        "has_appointment": False,
        "arrival_time": None,
        "verdict": None,
        "last_spoken_message": None,
    }
    return {"status": "cleared"}


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
# Combined Dashboard: Vision Proxy & Events
# ──────────────────────────────────────────────
vision_events: deque = deque(maxlen=100)
_vision_client: Optional[httpx.AsyncClient] = None


def _vision() -> httpx.AsyncClient:
    global _vision_client
    if _vision_client is None:
        _vision_client = httpx.AsyncClient(base_url=settings.vision_url, timeout=httpx.Timeout(20.0, connect=2.0))
    return _vision_client


@app.post("/api/events/vision", summary="Receive an event from the vision server")
async def receive_vision_event(event: dict):
    event["received_at"] = datetime.now().isoformat(timespec="seconds")
    vision_events.appendleft(event)
    identity = event.get("identity") or {}
    if identity.get("known"):
        logger.info(f"Vision: visit #{event.get('visit_id')} recognized as {identity.get('name')} "
                    f"({identity.get('score'):.2f})")
    else:
        logger.info(f"Vision: {event.get('type')} (visit #{event.get('visit_id')})")
    started = auto_conv.on_vision_event(event) if auto_conv else False
    return {"ok": True, "conversation_started": started}


# ──────────────────────────────────────────────
# Hands-free conversation control
# ──────────────────────────────────────────────
class AutoToggle(BaseModel):
    enabled: bool


@app.get("/api/auto", summary="Hands-free conversation status")
async def auto_status():
    return auto_conv.status() if auto_conv else {"enabled": False, "running": False, "state": "unavailable"}


@app.post("/api/auto", summary="Enable / disable hands-free conversation")
async def auto_set(data: AutoToggle):
    auto_conv.enabled = data.enabled
    if not data.enabled:
        await auto_conv.stop()
    return auto_conv.status()


@app.post("/api/auto/start", summary="Start a hands-free conversation now")
async def auto_start():
    if auto_conv.running:
        raise HTTPException(409, "A conversation is already running")
    last = next((e for e in vision_events if e.get("type") == "visit_captured"), None)
    auto_conv.start((last or {}).get("identity"), reason="dashboard")
    return auto_conv.status()


@app.post("/api/auto/stop", summary="Stop the hands-free conversation")
async def auto_stop():
    await auto_conv.stop()
    return auto_conv.status()


@app.get("/api/events", summary="Recent vision events")
async def list_events(limit: int = 30):
    return {"events": list(vision_events)[:limit]}


@app.get("/api/vision/video_feed", include_in_schema=False)
async def vision_video_feed():
    """Relay the vision server's annotated MJPEG stream so the dashboard needs one origin."""
    client = _vision()
    try:
        req = client.build_request("GET", "/video_feed", timeout=httpx.Timeout(None, connect=2.0))
        upstream = await client.send(req, stream=True)
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Vision server unreachable at {settings.vision_url}: {e}")

    async def relay():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(relay(), media_type=upstream.headers.get(
        "content-type", "multipart/x-mixed-replace; boundary=frame"))


@app.api_route("/api/vision/{path:path}", methods=["GET", "POST", "PATCH", "DELETE"], include_in_schema=False)
async def vision_proxy(path: str, request: Request):
    """Forward /api/vision/<path> to <VISION_URL>/api/<path>."""
    try:
        upstream = await _vision().request(
            request.method,
            f"/api/{path}",
            params=dict(request.query_params),
            content=await request.body(),
            headers={"content-type": request.headers.get("content-type", "application/octet-stream")},
        )
    except httpx.HTTPError as e:
        return JSONResponse({"error": "vision_offline", "detail": str(e), "vision_url": settings.vision_url},
                            status_code=502)
    return Response(content=upstream.content, status_code=upstream.status_code,
                    media_type=upstream.headers.get("content-type"))


@app.get("/api/sessions/transcript", summary="Live conversation transcript")
async def session_transcript():
    """Visitor/assistant turns of the current session (system prompt and tool plumbing hidden)."""
    ctx = conversation_manager.current_session if conversation_manager else None
    if not ctx:
        return {"active": False, "messages": []}
    turns = []
    for m in ctx.messages:
        role, content = m.get("role"), (m.get("content") or "").strip()
        if role == "user" and content:
            turns.append({"role": "visitor", "text": content})
        elif role == "assistant" and content:
            turns.append({"role": "assistant", "text": content})
        elif role == "assistant" and m.get("tool_calls"):
            names = ", ".join(tc.get("function", {}).get("name", "?") for tc in m["tool_calls"])
            turns.append({"role": "tool", "text": f"Checking: {names}"})
    return {
        "active": conversation_manager.is_session_active(),
        "session_id": ctx.session_id,
        "state": ctx.state.name,
        "visitor_name": ctx.visitor_name,
        "visitor_purpose": ctx.visitor_purpose,
        "messages": turns,
    }


@app.get("/api/visit_logs", summary="Past conversation sessions")
async def visit_logs(limit: int = 20):
    try:
        async with async_session() as db:
            rows = (await db.execute(
                select(VisitLog).order_by(VisitLog.created_at.desc()).limit(limit)
            )).scalars().all()
    except Exception as e:
        logger.error(f"visit_logs query failed: {e}")
        return {"logs": [], "error": str(e)}
    return {"logs": [{
        "id": r.id,
        "session_id": r.session_id,
        "visitor_name": r.visitor_name,
        "visitor_purpose": r.visitor_purpose,
        "status": r.status,
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "ended_at": r.ended_at.isoformat() if r.ended_at else None,
        "turns": sum(1 for m in (r.messages or []) if m.get("role") in ("user", "assistant") and m.get("content")),
    } for r in rows]}


# ──────────────────────────────────────────────
# Static Files & Root Page
# ──────────────────────────────────────────────
STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    """Serve the combined video + audio security dashboard."""
    return HTMLResponse(content=(STATIC_DIR / "dashboard.html").read_text(encoding="utf-8"))


@app.get("/console", response_class=HTMLResponse, include_in_schema=False)
async def audio_console():
    """Serve the original audio test console."""
    index_file = STATIC_DIR / "index.html"
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


@app.get("/appointments", response_class=HTMLResponse, include_in_schema=False)
@app.get("/owner", response_class=HTMLResponse, include_in_schema=False)
@app.get("/appointments-dashboard", response_class=HTMLResponse, include_in_schema=False)
async def appointments_dashboard_page():
    """Serve the homeowner appointments and visitor access dashboard."""
    return HTMLResponse(content=(STATIC_DIR / "appointments.html").read_text(encoding="utf-8"))


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
