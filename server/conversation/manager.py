import uuid
import json
import logging
from datetime import datetime
from typing import AsyncGenerator, Optional, Dict, Any

from server.conversation.states import ConversationState, ConversationContext
from server.conversation.prompts import MASTER_SYSTEM_PROMPT
from server.conversation.llm import OllamaClient, SentenceAccumulator
from server.security.appointments import AppointmentManager
from server.security.owners import OwnerManager
from server.config import settings, CONVERSATIONS_DIR, SESSION_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

class ConversationManager:
    def __init__(self, appointment_manager: AppointmentManager, owner_manager: OwnerManager):
        self.appointment_manager = appointment_manager
        self.owner_manager = owner_manager
        self.llm_client = OllamaClient()
        self.current_session: Optional[ConversationContext] = None
        self.timeout_seconds = SESSION_TIMEOUT_SECONDS

    async def start_session(self) -> str:
        if self.current_session:
            await self.end_session()

        session_id = str(uuid.uuid4())
        self.current_session = ConversationContext(
            session_id=session_id,
            state=ConversationState.GREETING,
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow()
        )

        # Initialize conversation with the system prompt
        self.current_session.messages.append({
            "role": "system",
            "content": MASTER_SYSTEM_PROMPT
        })

        return session_id

    def is_session_active(self) -> bool:
        if not self.current_session:
            return False

        if (datetime.utcnow() - self.current_session.last_activity).total_seconds() > self.timeout_seconds:
            self._check_timeout()
            return False

        return True

    def _check_timeout(self) -> None:
        if self.current_session:
            if (datetime.utcnow() - self.current_session.last_activity).total_seconds() > self.timeout_seconds:
                logger.info(f"Session {self.current_session.session_id} timed out.")
                self._log_conversation()
                self.current_session = None

    async def end_session(self) -> None:
        if self.current_session:
            self._log_conversation()
            self.current_session = None

    def _log_conversation(self) -> None:
        if not self.current_session:
            return

        # 1. Log to JSON
        CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
        log_file = str(CONVERSATIONS_DIR / f"{self.current_session.session_id}.json")

        data = {
            "session_id": self.current_session.session_id,
            "visitor_name": self.current_session.visitor_name,
            "visitor_purpose": self.current_session.visitor_purpose,
            "messages": self.current_session.messages,
            "created_at": self.current_session.created_at.isoformat(),
            "ended_at": datetime.utcnow().isoformat()
        }

        try:
            with open(log_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to log conversation to JSON: {e}")
            
        # 2. Log to SQL Database
        try:
            import asyncio
            from server.database.session import async_session
            from server.database.models import VisitLog
            
            # Capture variables locally since self.current_session will be None soon
            session_id = self.current_session.session_id
            visitor_name = self.current_session.visitor_name
            visitor_purpose = self.current_session.visitor_purpose
            status = self.current_session.state.name
            
            # Create a deep copy of the messages so it isn't mutated
            import copy
            messages = copy.deepcopy(self.current_session.messages)
            created_at = self.current_session.created_at
            
            async def save_to_db():
                try:
                    async with async_session() as session:
                        log = VisitLog(
                            session_id=session_id,
                            visitor_name=visitor_name,
                            visitor_purpose=visitor_purpose,
                            status=status,
                            messages=messages,
                            created_at=created_at,
                            ended_at=datetime.utcnow()
                        )
                        session.add(log)
                        await session.commit()
                        logger.info(f"Session {session_id} saved to DB.")
                except Exception as db_e:
                    logger.error(f"DB insert failed: {db_e}")
            
            # Run in the background
            loop = asyncio.get_event_loop()
            loop.create_task(save_to_db())
        except Exception as e:
            logger.error(f"Failed to log conversation to database: {e}")

    # ──────────────────────────────────────────────
    # Tool Execution Handlers
    # ──────────────────────────────────────────────
    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Executes the requested tool securely on the backend."""
        logger.info(f"Executing tool: {tool_name} with args {args}")

        try:
            if tool_name == "get_appointments":
                v_name = args.get("visitor_name", "")
                purpose = args.get("purpose", "")

                # Update session context for the frontend
                if v_name:
                    self.current_session.visitor_name = v_name
                if purpose:
                    self.current_session.visitor_purpose = purpose

                appt = self.appointment_manager.find_appointment(v_name, purpose)
                if appt:
                    self.current_session.matched_appointment = appt
                    return f"Appointment found for {v_name}. Purpose: {appt.purpose}. Date: {appt.date}. Approved: {appt.approved}."
                return "No appointment found for this visitor."

            elif tool_name == "get_homeowner_instructions":
                return "Delivery: leave at door. Guests with appointments: allow entry. Unknown or no appointment: take message and notify homeowner."

            elif tool_name == "get_owner_status":
                anyone_home = self.owner_manager.is_anyone_home()
                owners = self.owner_manager.get_all_owners()
                status_list = [f"{o.name} is {o.status}" for o in owners]
                return f"Anyone home: {anyone_home}. Details: " + ", ".join(status_list)

            elif tool_name == "request_entry":
                v_name = args.get("visitor_name", "")
                if v_name:
                    self.current_session.visitor_name = v_name

                if self.current_session.matched_appointment:
                    self.current_session.state = ConversationState.APPROVED
                    return "Entry approved based on confirmed appointment."
                else:
                    self.current_session.state = ConversationState.DENIED
                    return "Entry denied. No confirmed appointment. Take a message instead."

            elif tool_name == "take_message":
                v_name = args.get("visitor_name", "")
                if v_name:
                    self.current_session.visitor_name = v_name

                msg = args.get("message", "")
                logger.info(f"Message recorded for homeowner: {msg}")
                self.current_session.state = ConversationState.ENDED
                return "Message recorded successfully."

            return f"Unknown tool: {tool_name}"
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
            return f"Error executing tool: {str(e)}"

    # ──────────────────────────────────────────────
    # Main Conversation Loop
    # ──────────────────────────────────────────────
    async def process_input(self, text: str) -> AsyncGenerator[str, None]:
        if not self.current_session:
            yield "Session ended or not started."
            return

        self.current_session.last_activity = datetime.utcnow()

        if text.strip():
            self.current_session.messages.append({"role": "user", "content": text})

        # Phase 1: Tool-calling loop (non-streaming, handles thinking internally)
        max_tool_rounds = 3
        for round_num in range(max_tool_rounds):
            logger.info(f"Tool round {round_num + 1}/{max_tool_rounds}")

            result = await self.llm_client.chat_with_tools(self.current_session.messages)

            content = result.get("content", "")
            tool_calls = result.get("tool_calls", None)

            # If the model returned text AND no tool calls, it's a direct response
            if content and not tool_calls:
                logger.info(f"LLM direct response (no tools): {content[:100]}")
                self.current_session.messages.append({"role": "assistant", "content": content})

                # Yield sentences from the direct response
                accumulator = SentenceAccumulator()
                sentences = accumulator.feed(content)
                for sentence in sentences:
                    yield sentence
                remainder = accumulator.flush()
                if remainder:
                    yield remainder
                return

            # If tool calls were made, execute them
            if tool_calls:
                # Append the assistant's tool-call message
                self.current_session.messages.append({
                    "role": "assistant",
                    "content": content or "",
                    "tool_calls": tool_calls
                })

                for tc in tool_calls:
                    func = tc.get("function", {})
                    name = func.get("name", "")
                    args = func.get("arguments", {})

                    result_str = self._execute_tool(name, args)
                    logger.info(f"Tool result for {name}: {result_str[:100]}")

                    self.current_session.messages.append({
                        "role": "tool",
                        "content": result_str,
                    })

                # Continue loop to let LLM generate response based on tool results
                continue

            # No content and no tool calls — something went wrong
            logger.warning("LLM returned no content and no tool calls")
            break

        # Phase 2: Stream the final text response (no tools, faster)
        logger.info("Streaming final text response")
        accumulator = SentenceAccumulator()
        full_response = ""

        async for chunk in self.llm_client.chat_stream(self.current_session.messages):
            sentences = accumulator.feed(chunk)
            for sentence in sentences:
                full_response += sentence + " "
                yield sentence

        remainder = accumulator.flush()
        if remainder:
            full_response += remainder
            yield remainder

        if full_response.strip():
            self.current_session.messages.append({"role": "assistant", "content": full_response.strip()})
