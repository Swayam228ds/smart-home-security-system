import pytest
import asyncio
from unittest.mock import MagicMock, patch

from server.conversation.manager import ConversationManager
from server.conversation.states import ConversationState
from server.security.appointments import AppointmentManager, Appointment
from server.security.owners import OwnerManager

@pytest.fixture
def manager():
    # Setup mock backend
    appts = AppointmentManager()
    appts.add_appointment(Appointment(visitor_name="Arjun", purpose="plumbing", date="today", time_range="all_day", approved=True))
    
    owners = OwnerManager()
    owners.update_status("Rahul", "home")

    # Mock OllamaClient
    with patch("server.conversation.manager.OllamaClient") as MockClient:
        m = ConversationManager(appointment_manager=appts, owner_manager=owners)
        yield m

@pytest.mark.asyncio
async def test_appointment_found_flow(manager):
    await manager.start_session()
    
    # Simulate LLM returning a tool call for get_appointments
    manager.llm_client.chat_stream.return_value = _mock_llm_stream([
        {"type": "tool_calls", "tool_calls": [
            {"function": {"name": "get_appointments", "arguments": {"visitor_name": "Arjun", "purpose": "plumbing"}}}
        ]},
        {"type": "text", "content": "Thank you, Arjun. Your appointment has been confirmed. Please wait here."}
    ])

    sentences = []
    async for sentence in manager.process_input("I'm Arjun, here for plumbing."):
        sentences.append(sentence)

    # Tool should have updated the state/matched appointment
    assert manager.current_session.matched_appointment is not None
    assert manager.current_session.matched_appointment.visitor_name == "Arjun"
    
    # LLM text response should be generated
    assert "Thank you, Arjun." in sentences[0]

@pytest.mark.asyncio
async def test_no_appointment_flow(manager):
    await manager.start_session()
    
    # Simulate LLM returning a tool call for get_appointments
    manager.llm_client.chat_stream.return_value = _mock_llm_stream([
        {"type": "tool_calls", "tool_calls": [
            {"function": {"name": "get_appointments", "arguments": {"visitor_name": "Ravi", "purpose": "sales"}}}
        ]},
        {"type": "tool_calls", "tool_calls": [
            {"function": {"name": "take_message", "arguments": {"visitor_name": "Ravi", "message": "Can I meet the owner?"}}}
        ]}
    ])

    sentences = []
    async for sentence in manager.process_input("I'm Ravi, here for sales."):
        sentences.append(sentence)

    # Appointment should not be found
    assert manager.current_session.matched_appointment is None
    # State should eventually end because take_message ends the conversation
    assert manager.current_session.state == ConversationState.ENDED

async def _mock_llm_stream(responses):
    for r in responses:
        yield r
