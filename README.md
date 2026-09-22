# Home Security Audio Processing Pipeline

Real-time conversational AI pipeline for home security visitor management. Visitors speak into a microphone (connected to an ESP32), and the system transcribes their speech, processes it through a local LLM that follows a security interview protocol, and responds with synthesized speech.

## Architecture

```
Audio In (PCM) → VAD → STT → Local LLM → Backend Tools → TTS → Audio Out (PCM)
                  │      │       │            │           │
            Silero VAD  Whisper Ollama  Appt/Rules      Piper
```

**Pipeline stages:**
1. **VAD** (Silero) — Detects when the visitor starts/stops speaking
2. **STT** (faster-whisper) — Transcribes speech to text (GPU-accelerated, optimized for Indian English)
3. **LLM** (Ollama) — Dynamic context-aware security conversation capable of explicit tool calling
4. **Backend Tools** — Validates appointments, enforces homeowner rules, and processes security actions safely
5. **TTS** (Piper) — Converts response text to natural speech

**Key optimizations:**
- 100% Local Processing (No cloud dependencies)
- Indian English STT optimization (via contextual prompts and language params)
- Sentence-level pipelining (TTS starts on sentence 1 while LLM generates sentence 2)
- Barge-in support (cancels response when visitor speaks)
- All ML models loaded once at startup

## Conversation Flow & Security Decisions

The conversation is driven dynamically by the local LLM, bound by strict backend tools:

1. Greeting
2. Ask visitor name
3. Ask purpose
4. The LLM uses the `get_appointments` tool internally.
5. The backend validates the appointment against `appointments.json`.
6. If an appointment exists, the backend returns success to the LLM, and the LLM informs the visitor.
7. If no appointment exists, the LLM uses `get_homeowner_instructions` and `get_owner_status` to determine the correct action (e.g. taking a message).

*Crucially: The LLM does NOT make final security decisions. It is only allowed to propose tool calls which the backend validates.*

## Setup & Installation

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA (tested on RTX 4050 6GB)
- [Ollama](https://ollama.com/) installed and running locally

### 1. Install Dependencies

```bash
cd iot
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r server/requirements.txt
# Additional requirement for HTTPX interactions with Ollama
pip install httpx
```

### 2. Configure Environment

```bash
copy server\.env.example server\.env
```

Ensure your `.env` contains:
```env
WHISPER_MODEL_SIZE=small
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16
WHISPER_LANGUAGE=en
WHISPER_BEAM_SIZE=5

LLM_PROVIDER=ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen3:8b
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=300
```

### 3. Setup Ollama (LLM)

1. Install Ollama from [ollama.com](https://ollama.com).
2. Pull the required model:
```bash
# Wait, if qwen3 is not out, this might refer to qwen2.5:7b or similar. 
# Ensure you pull the exact model specified in OLLAMA_MODEL.
ollama pull qwen3:8b
```

### 4. VRAM Considerations (RTX 4050 6GB)
- `faster-whisper` (`small`, `float16`) uses ~1GB VRAM.
- Ollama running an 8B model natively uses ~4.5 - 5GB VRAM.
- Ensure Ollama is running a 4-bit quantized version (which is default for `ollama pull`) so it fits within the remaining 5GB without causing Out of Memory (OOM) errors. 

## Running the Server

```bash
# From the project root (iot/)
python -m uvicorn server.main:app --host 0.0.0.0 --port 8000
```

## Testing

### Manual Testing via Web UI
Navigate to `http://localhost:8000/static/index.html`. You can upload audio files or speak directly to test the Indian English STT and Ollama tool calling.

### Automated Tests
Run unit tests for the pipeline:
```bash
pip install pytest pytest-asyncio
pytest tests/test_local_pipeline.py -v
```

### Testing Whisper Independently
```python
from server.audio.stt import SpeechToText
import numpy as np

# Create fake 16kHz float32 audio
audio = np.zeros(16000, dtype=np.float32) 
stt = SpeechToText()
print(stt.transcribe(audio))
```

### Testing Ollama Independently
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "qwen3:8b",
  "prompt": "Hello",
  "stream": false
}'
```

## Managing Appointments

Edit `server/data/appointments.json` to add expected visitors:

```json
{
  "appointments": [
    {
      "visitor_name": "Rahul",
      "purpose": "plumbing repair",
      "date": "2026-09-19",
      "time_range": "10:00-12:00",
      "approved": true
    }
  ]
}
```
