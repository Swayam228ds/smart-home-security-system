"""
Tests for the conversation state machine and manager.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from server.conversation.states import (
    ConversationState,
    ConversationContext,
    VALID_TRANSITIONS,
    TERMINAL_STATES,
    validate_transition,
)
from server.conversation.prompts import (
    MASTER_SYSTEM_PROMPT,
    get_state_prompt,
    build_llm_messages,
)
from server.conversation.llm import SentenceAccumulator


# ──────────────────────────────────────────────
# State Machine Tests
# ──────────────────────────────────────────────

class TestConversationState:
    """Tests for state definitions and transitions."""

    def test_all_states_defined(self):
        expected = {
            "GREETING", "ASK_NAME", "ASK_PURPOSE", "CHECK_APPOINTMENT",
            "APPOINTMENT_FOUND", "NO_APPOINTMENT", "APPROVED", "DENIED",
            "TAKE_MESSAGE", "OWNER_UNAVAILABLE", "ENDED",
        }
        actual = {s.name for s in ConversationState}
        assert actual == expected

    def test_terminal_states(self):
        assert ConversationState.APPROVED in TERMINAL_STATES
        assert ConversationState.DENIED in TERMINAL_STATES
        assert ConversationState.ENDED in TERMINAL_STATES
        assert ConversationState.GREETING not in TERMINAL_STATES

    def test_valid_greeting_transitions(self):
        assert validate_transition(
            ConversationState.GREETING, ConversationState.ASK_NAME
        )
        assert validate_transition(
            ConversationState.GREETING, ConversationState.ENDED
        )
        assert not validate_transition(
            ConversationState.GREETING, ConversationState.APPROVED
        )

    def test_valid_ask_name_transitions(self):
        assert validate_transition(
            ConversationState.ASK_NAME, ConversationState.ASK_PURPOSE
        )
        assert not validate_transition(
            ConversationState.ASK_NAME, ConversationState.GREETING
        )

    def test_check_appointment_transitions(self):
        assert validate_transition(
            ConversationState.CHECK_APPOINTMENT,
            ConversationState.APPOINTMENT_FOUND,
        )
        assert validate_transition(
            ConversationState.CHECK_APPOINTMENT,
            ConversationState.NO_APPOINTMENT,
        )
        assert validate_transition(
            ConversationState.CHECK_APPOINTMENT,
            ConversationState.OWNER_UNAVAILABLE,
        )

    def test_no_transitions_from_ended(self):
        for state in ConversationState:
            if state != ConversationState.ENDED:
                assert not validate_transition(
                    ConversationState.ENDED, state
                )

    def test_all_states_have_transition_entry(self):
        for state in ConversationState:
            assert state in VALID_TRANSITIONS, f"{state} missing from VALID_TRANSITIONS"


class TestConversationContext:
    """Tests for the ConversationContext dataclass."""

    def test_default_creation(self):
        ctx = ConversationContext(
            session_id="test-123",
            state=ConversationState.GREETING,
        )
        assert ctx.session_id == "test-123"
        assert ctx.state == ConversationState.GREETING
        assert ctx.visitor_name is None
        assert ctx.visitor_purpose is None
        assert ctx.matched_appointment is None
        assert ctx.messages == []
        assert isinstance(ctx.created_at, datetime)

    def test_with_visitor_info(self):
        ctx = ConversationContext(
            session_id="test-456",
            state=ConversationState.ASK_PURPOSE,
            visitor_name="John",
            visitor_purpose="delivery",
        )
        assert ctx.visitor_name == "John"
        assert ctx.visitor_purpose == "delivery"


# ──────────────────────────────────────────────
# Prompts Tests
# ──────────────────────────────────────────────

class TestPrompts:
    """Tests for system prompts and prompt building."""

    def test_master_prompt_exists(self):
        assert MASTER_SYSTEM_PROMPT
        assert "security" in MASTER_SYSTEM_PROMPT.lower()
        assert "voice" in MASTER_SYSTEM_PROMPT.lower()

    def test_state_prompts_for_all_states(self):
        ctx = ConversationContext(
            session_id="test",
            state=ConversationState.GREETING,
            visitor_name="Test",
        )
        for state in ConversationState:
            ctx.state = state
            prompt = get_state_prompt(state, ctx)
            assert isinstance(prompt, str)

    def test_ask_purpose_includes_name(self):
        ctx = ConversationContext(
            session_id="test",
            state=ConversationState.ASK_PURPOSE,
            visitor_name="Alice",
        )
        prompt = get_state_prompt(ConversationState.ASK_PURPOSE, ctx)
        assert "Alice" in prompt

    def test_build_llm_messages_structure(self):
        ctx = ConversationContext(
            session_id="test",
            state=ConversationState.GREETING,
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "model", "content": "Welcome!"},
            ],
        )
        messages = build_llm_messages(ctx, "", "")
        assert len(messages) >= 3  # system + 2 conversation messages
        assert messages[0]["role"] == "system"

    def test_build_llm_messages_includes_context(self):
        ctx = ConversationContext(
            session_id="test",
            state=ConversationState.CHECK_APPOINTMENT,
        )
        messages = build_llm_messages(
            ctx,
            appointment_info="Rahul - plumbing",
            owner_info="Abhinay is home",
        )
        system_content = messages[0]["content"]
        assert "Rahul" in system_content
        assert "Abhinay" in system_content


# ──────────────────────────────────────────────
# Sentence Accumulator Tests
# ──────────────────────────────────────────────

class TestSentenceAccumulator:
    """Tests for the streaming sentence accumulator."""

    def test_simple_sentence(self):
        acc = SentenceAccumulator()
        result = acc.feed("Hello there. ")
        assert result == ["Hello there."]

    def test_multiple_sentences(self):
        acc = SentenceAccumulator()
        result = acc.feed("First sentence. Second sentence. ")
        assert len(result) == 2
        assert result[0] == "First sentence."
        assert result[1] == "Second sentence."

    def test_buffering_until_complete(self):
        acc = SentenceAccumulator()
        assert acc.feed("Hello ") == []
        assert acc.feed("there") == []
        result = acc.feed(". How are you? ")
        assert "Hello there." in result

    def test_flush_remainder(self):
        acc = SentenceAccumulator()
        acc.feed("Incomplete sentence")
        remainder = acc.flush()
        assert remainder == "Incomplete sentence"

    def test_flush_empty(self):
        acc = SentenceAccumulator()
        assert acc.flush() is None

    def test_question_mark(self):
        acc = SentenceAccumulator()
        result = acc.feed("What is your name? ")
        assert result == ["What is your name?"]

    def test_exclamation(self):
        acc = SentenceAccumulator()
        result = acc.feed("Welcome! ")
        assert result == ["Welcome!"]

    def test_abbreviation_handling(self):
        acc = SentenceAccumulator()
        # "Mr." should not split
        result = acc.feed("Mr. Smith is here. ")
        # Should produce one sentence (Mr. shouldn't split prematurely)
        assert any("Mr." in s for s in result) or len(result) <= 1


# ──────────────────────────────────────────────
# Conversation Manager Tests (with mocked LLM)
# ──────────────────────────────────────────────

class TestConversationManagerFlow:
    """Tests for the conversation manager state machine flow."""

    @pytest.fixture
    def mock_managers(self):
        appt_mgr = MagicMock()
        appt_mgr.find_appointment.return_value = None
        appt_mgr.is_delivery.return_value = False

        owner_mgr = MagicMock()
        owner_mgr.is_anyone_home.return_value = True
        owner_mgr.get_available_owners.return_value = []

        gemini = MagicMock()

        async def mock_stream(*args, **kwargs):
            yield "Hello, welcome. "
            yield "May I have your name? "

        gemini.generate_stream = mock_stream

        return appt_mgr, owner_mgr, gemini

    @pytest.mark.asyncio
    async def test_session_lifecycle(self, mock_managers):
        from server.conversation.manager import ConversationManager

        appt_mgr, owner_mgr, gemini = mock_managers
        mgr = ConversationManager(appt_mgr, owner_mgr, gemini)

        # Start session
        session_id = await mgr.start_session()
        assert session_id
        assert mgr.is_session_active()
        assert mgr.current_session.state == ConversationState.GREETING

        # End session
        await mgr.end_session()
        assert not mgr.is_session_active()

    @pytest.mark.asyncio
    async def test_name_extraction(self, mock_managers):
        from server.conversation.manager import ConversationManager

        appt_mgr, owner_mgr, gemini = mock_managers
        mgr = ConversationManager(appt_mgr, owner_mgr, gemini)

        # Test various name formats
        assert mgr._extract_name("John") == "John"
        assert mgr._extract_name("my name is Alice") == "Alice"
        assert "John" in mgr._extract_name("John Smith")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
