"""
Integration tests for the audio processing pipeline.
Tests the server endpoints without requiring actual ML models.
"""

import io
import wave
import struct
import math
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


def _generate_test_wav(
    duration_sec: float = 2.0,
    sample_rate: int = 16000,
    frequency: float = 440.0,
) -> bytes:
    """Generate a simple sine wave WAV file for testing."""
    n_samples = int(sample_rate * duration_sec)
    samples = []
    for i in range(n_samples):
        t = i / sample_rate
        sample = int(32767 * 0.5 * math.sin(2 * math.pi * frequency * t))
        samples.append(sample)

    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f'<{len(samples)}h', *samples))

    return wav_io.getvalue()


def _generate_test_pcm(
    duration_sec: float = 2.0,
    sample_rate: int = 16000,
) -> bytes:
    """Generate raw PCM silence for testing."""
    n_samples = int(sample_rate * duration_sec)
    return np.zeros(n_samples, dtype=np.int16).tobytes()


class TestAudioUtilities:
    """Test audio format conversion utilities."""

    def test_generate_test_wav(self):
        wav_data = _generate_test_wav(duration_sec=1.0)
        assert len(wav_data) > 44  # WAV header + data

        wav_io = io.BytesIO(wav_data)
        with wave.open(wav_io, 'rb') as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000
            assert wf.getnframes() == 16000

    def test_generate_test_pcm(self):
        pcm = _generate_test_pcm(duration_sec=1.0)
        assert len(pcm) == 16000 * 2  # 16kHz * 2 bytes/sample


class TestSecurityEndpoints:
    """Test appointment and owner REST endpoints (no ML models needed)."""

    @pytest.fixture
    def mock_app(self):
        """Create a test app with mocked ML components."""
        # We mock the ML-heavy imports at the module level
        with patch('server.main.AudioPipeline') as mock_pipeline_cls, \
             patch('server.main.GeminiClient') as mock_gemini_cls, \
             patch('server.main.VADProcessor'), \
             patch('server.main.SpeechToText'), \
             patch('server.main.TextToSpeech'):

            from server.main import app
            client = TestClient(app)
            yield client

    def test_health_check(self, mock_app):
        """Test the health endpoint returns status info."""
        response = mock_app.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "components" in data
        assert "config" in data

    def test_list_appointments(self, mock_app):
        """Test listing appointments."""
        response = mock_app.get("/api/appointments")
        assert response.status_code == 200
        data = response.json()
        assert "appointments" in data

    def test_get_owner_status(self, mock_app):
        """Test getting owner status."""
        response = mock_app.get("/api/owners/status")
        assert response.status_code == 200
        data = response.json()
        assert "owners" in data
        assert "anyone_home" in data

    def test_session_not_active_initially(self, mock_app):
        """Test session endpoint when no session is active."""
        response = mock_app.get("/api/sessions")
        assert response.status_code == 200
        data = response.json()
        assert data["active"] is False


class TestSentenceAccumulator:
    """Additional sentence accumulator edge case tests."""

    def test_streaming_tokens(self):
        from server.conversation.llm import SentenceAccumulator

        acc = SentenceAccumulator()
        sentences = []

        # Simulate streaming tokens one by one
        tokens = ["Hello", ", ", "I am ", "the security ", "assistant. ",
                  "How ", "can I ", "help you? "]
        for token in tokens:
            result = acc.feed(token)
            sentences.extend(result)

        remainder = acc.flush()
        if remainder:
            sentences.append(remainder)

        assert len(sentences) >= 2
        assert any("security" in s for s in sentences)

    def test_empty_input(self):
        from server.conversation.llm import SentenceAccumulator

        acc = SentenceAccumulator()
        assert acc.feed("") == []
        assert acc.flush() is None


class TestVADProcessor:
    """Test VAD processor state machine logic (with mocked Silero model)."""

    @patch('server.audio.vad.torch')
    def test_vad_state_machine(self, mock_torch):
        """Test that VAD transitions correctly through states."""
        # Mock the Silero model
        mock_model = MagicMock()
        mock_model.return_value.item.return_value = 0.9  # High speech prob

        mock_torch.hub.load.return_value = (mock_model, None)
        mock_torch.from_numpy.side_effect = lambda x: MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()

        from server.audio.vad import VADProcessor, VADState

        vad = VADProcessor()
        assert vad.state == VADState.SILENCE

        # Feed speech chunk
        chunk = np.zeros(480, dtype=np.int16).tobytes()  # 30ms chunk
        result = vad.process_chunk(chunk)
        # Should now be in SPEECH state
        assert vad.state == VADState.SPEECH


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
