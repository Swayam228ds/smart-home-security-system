import asyncio
import json
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text

DATABASE_URL = "postgresql+asyncpg://postgres:Ak1979@localhost:5432/homedb"
engine = create_async_engine(DATABASE_URL)
async_session = async_sessionmaker(engine, class_=AsyncSession)

async def main():
    async with async_session() as session:
        result = await session.execute(text('SELECT id, visitor_name, status, messages, created_at FROM visit_logs ORDER BY id DESC LIMIT 5'))
        rows = result.fetchall()
        
        for r in rows:
            msgs = r[3] if r[3] else []
            if isinstance(msgs, str):
                msgs = json.loads(msgs)
            print(f"--- ID: {r[0]} | Created: {r[4]} | Status: {r[2]} | Msgs: {len(msgs)} ---")
            for i, m in enumerate(msgs):
                role = m.get('role', 'unknown')
                content = str(m.get('content', ''))[:100]
                print(f"  {i}. {role}: {content}")
            print()

if __name__ == "__main__":
    asyncio.run(main())
