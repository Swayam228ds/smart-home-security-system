import asyncio
import json
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text

DATABASE_URL = "postgresql+asyncpg://postgres:Ak1979@localhost:5432/homedb"
engine = create_async_engine(DATABASE_URL)
async_session = async_sessionmaker(engine, class_=AsyncSession)

async def main():
    async with async_session() as session:
        result = await session.execute(text('SELECT id, visitor_name, visitor_purpose, status, created_at, messages FROM visit_logs ORDER BY id DESC LIMIT 5'))
        rows = result.fetchall()
        
        print(f"{'ID':<3} | {'Name':<15} | {'Purpose':<15} | {'Status':<10} | {'Msgs':<4} | {'Time'}")
        print('-'*85)
        for r in rows:
            msgs = r[5] if r[5] else []
            if isinstance(msgs, str):
                msgs = json.loads(msgs)
            print(f"{r[0]:<3} | {str(r[1]):<15} | {str(r[2]):<15} | {str(r[3]):<10} | {len(msgs):<4} | {r[4]}")

if __name__ == "__main__":
    asyncio.run(main())
