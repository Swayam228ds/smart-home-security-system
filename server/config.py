"""
Configuration module for the Home Security Audio Pipeline.
Uses pydantic-settings to load from .env and defines all audio/model constants.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional

# ──────────────────────────────────────────────
# Directory Paths
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CONVERSATIONS_DIR = DATA_DIR / "conversations"
PIPER_MODELS_DIR = DATA_DIR / "piper_models"

# Ensure directories exist
CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
PIPER_MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# Audio Constants
# ──────────────────────────────────────────────
SAMPLE_RATE = 16000          # 16 kHz — native for Whisper & Silero VAD
CHANNELS = 1                 # Mono
SAMPLE_WIDTH = 2             # 16-bit (2 bytes per sample)
CHUNK_DURATION_MS = 30       # 30ms per chunk (matches VAD window)
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000) * SAMPLE_WIDTH  # 960 bytes

# VAD Parameters
VAD_THRESHOLD = 0.5          # Speech probability threshold
SILENCE_DURATION_MS = 600    # Silence duration to trigger end-of-speech
MIN_SPEECH_DURATION_MS = 250 # Minimum speech segment to process

# ──────────────────────────────────────────────
# Session Constants
# ──────────────────────────────────────────────
SESSION_TIMEOUT_SECONDS = 300  # 5 minutes inactivity timeout
MAX_CONVERSATION_TURNS = 20   # Safety limit on conversation length


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    # ── Database ──
    database_url: Optional[str] = Field(default=None, description="PostgreSQL DB URL")


    # ── Whisper (STT) ──
    whisper_model_size: str = Field(
        default="small",
        description="faster-whisper model size: tiny, base, small, medium, large-v3"
    )
    whisper_device: str = Field(
        default="cuda",
        description="Device for whisper inference: cuda or cpu"
    )
    whisper_compute_type: str = Field(
        default="float16",
        description="Compute type: float16 (GPU), int8 (CPU), float32"
    )
    whisper_language: str = Field(
        default="en",
        description="Language code for whisper (e.g. en)"
    )
    whisper_beam_size: int = Field(
        default=5,
        description="Beam size for whisper decoding"
    )

    # ── Piper (TTS) ──
    piper_voice: str = Field(
        default="en_US-amy-medium",
        description="Piper TTS voice model name"
    )

    # ── Local LLM (Ollama) ──
    llm_provider: str = Field(
        default="ollama",
        description="Provider of the LLM"
    )
    ollama_host: str = Field(
        default="http://localhost:11434",
        description="Ollama API host URL"
    )
    ollama_model: str = Field(
        default="qwen3:8b",
        description="Ollama model identifier"
    )
    llm_max_tokens: int = Field(
        default=300,
        description="Max tokens per LLM response"
    )
    llm_temperature: float = Field(
        default=0.2,
        description="LLM temperature (lower = more deterministic)"
    )

    # ── Server ──
    host: str = Field(default="0.0.0.0", description="Server bind host")
    port: int = Field(default=8000, description="Server bind port")
    log_level: str = Field(default="info", description="Logging level")

    # ── Vision server (ESP32-CAM face pipeline, video/smart-home-security-system) ──
    vision_url: str = Field(
        default="http://127.0.0.1:8080",
        description="Base URL of the vision server proxied by the combined dashboard"
    )

    # ── Hands-free conversation (triggered by the camera) ──
    auto_conversation: bool = Field(
        default=True,
        description="Start a voice conversation automatically when the camera captures a visitor"
    )
    auto_known_only: bool = Field(
        default=False,
        description="Only auto-start for visitors the face recognizer knows"
    )
    auto_listen_timeout_s: float = Field(default=15.0, description="End if the visitor says nothing for this long")
    auto_max_turns: int = Field(default=12, description="Safety limit on visitor turns per conversation")
    auto_cooldown_s: float = Field(default=30.0, description="Ignore new camera visits this long after a conversation")

    # ── Data Paths ──
    appointments_file: str = Field(
        default=str(DATA_DIR / "appointments.json"),
        description="Path to appointments JSON file"
    )

    model_config = {
        "env_file": str(BASE_DIR / ".env"),
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


# Singleton settings instance
settings = Settings()
