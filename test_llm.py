import asyncio
import logging
from server.conversation.llm import GeminiClient
from server.conversation.states import ConversationContext, ConversationState
from server.conversation.prompts import build_llm_messages

logging.basicConfig(level=logging.INFO)

async def test_llm():
    client = GeminiClient()
    ctx = ConversationContext(session_id="test", state=ConversationState.ASK_NAME)
    ctx.messages.append({"role": "user", "content": "Hello"})
    
    msgs = build_llm_messages(ctx, "", "")
    system_prompt = msgs[0]["content"] if msgs and msgs[0]["role"] == "system" else ""
    
    print("Messages:", msgs)
    print("System Prompt:", system_prompt)
    
    full = ""
    async for chunk in client.generate_stream(msgs, system_prompt):
        print("Chunk:", repr(chunk))
        full += chunk
        
    print("\nFull output:", repr(full))

if __name__ == "__main__":
    asyncio.run(test_llm())
