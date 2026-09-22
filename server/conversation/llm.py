import json
import httpx
import logging
from typing import AsyncGenerator, List, Dict, Any, Optional

from server.config import settings

logger = logging.getLogger(__name__)

# Define the backend tools schema for Ollama
OLLAMA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_appointments",
            "description": "Check if a visitor has an appointment. Call this once you know the visitor's name and purpose.",
            "parameters": {
                "type": "object",
                "properties": {
                    "visitor_name": {"type": "string", "description": "Name of the visitor"},
                    "purpose": {"type": "string", "description": "Purpose of the visit"}
                },
                "required": ["visitor_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_homeowner_instructions",
            "description": "Get general instructions from the homeowner (e.g. for deliveries, guests, unknown visitors).",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_owner_status",
            "description": "Check if the homeowners are currently home, away, or busy.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "request_entry",
            "description": "Request entry for the visitor after confirming their appointment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "visitor_name": {"type": "string"},
                    "reason": {"type": "string"}
                },
                "required": ["visitor_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "take_message",
            "description": "Record a message from the visitor for the homeowner.",
            "parameters": {
                "type": "object",
                "properties": {
                    "visitor_name": {"type": "string"},
                    "message": {"type": "string"}
                },
                "required": ["visitor_name", "message"]
            }
        }
    }
]


class OllamaClient:
    """
    Client for interacting with a local Ollama instance.
    Uses non-streaming for tool-call rounds, streaming for final text response.
    """
    def __init__(self):
        self.host = settings.ollama_host.rstrip('/')
        self.model = settings.ollama_model
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        self.client = httpx.AsyncClient(timeout=120.0)

    async def chat_with_tools(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Non-streaming chat call used for tool-calling rounds.
        Returns the complete response including any tool_calls.
        """
        url = f"{self.host}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
            "tools": OLLAMA_TOOLS
        }

        try:
            logger.info(f"Ollama tool call request ({len(messages)} messages)")
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            msg = data.get("message", {})

            result = {
                "content": msg.get("content", ""),
                "tool_calls": msg.get("tool_calls", None),
            }
            logger.info(f"Ollama tool call response: content={repr(result['content'][:100])}, tool_calls={bool(result['tool_calls'])}")
            return result
        except Exception as e:
            logger.error(f"Error in Ollama chat_with_tools: {e}")
            return {"content": "I'm sorry, I encountered an error.", "tool_calls": None}

    async def chat_stream(self, messages: List[Dict[str, Any]]) -> AsyncGenerator[str, None]:
        """
        Streaming chat call used for the final text response (no tools).
        Yields text chunks only, ignoring thinking tokens.
        """
        url = f"{self.host}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
            # No tools — this is for the final text-only response
        }

        try:
            logger.info(f"Ollama stream request ({len(messages)} messages)")
            async with self.client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        msg = data.get("message", {})
                        # Only yield actual content, skip thinking tokens
                        content = msg.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to decode Ollama stream line: {line[:100]}")
        except Exception as e:
            logger.error(f"Error in Ollama chat_stream: {e}")
            yield "I'm sorry, I encountered an error."

    async def health_check(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            response = await self.client.get(f"{self.host}/api/tags")
            return response.status_code == 200
        except Exception:
            return False


class SentenceAccumulator:
    def __init__(self):
        self.buffer = ""
        self.terminators = {'.', '!', '?'}
        self.abbreviations = {"Mr.", "Mrs.", "Dr.", "Ms.", "Prof.", "Sr.", "Jr."}

    def feed(self, token: str) -> List[str]:
        sentences = []
        self.buffer += token

        while True:
            split_idx = -1
            for i, char in enumerate(self.buffer):
                if char in self.terminators:
                    if i + 1 < len(self.buffer):
                        next_char = self.buffer[i+1]
                        if next_char.isspace() or next_char == '\n':
                            words = self.buffer[:i+1].split()
                            if words and words[-1] in self.abbreviations:
                                continue
                            split_idx = i + 1
                            break

            if split_idx != -1:
                sentence = self.buffer[:split_idx].strip()
                if sentence:
                    sentences.append(sentence)
                self.buffer = self.buffer[split_idx:].lstrip()
            else:
                break

        return sentences

    def flush(self) -> Optional[str]:
        res = self.buffer.strip()
        self.buffer = ""
        return res if res else None
