import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text

DATABASE_URL = "postgresql+asyncpg://postgres:Ak1979@localhost:5432/homedb"
engine = create_async_engine(DATABASE_URL)
async_session = async_sessionmaker(engine, class_=AsyncSession)

async def main():
    async with async_session() as session:
        result = await session.execute(text('SELECT count(*) FROM visit_logs'))
        print(f"Total rows: {result.fetchone()[0]}")
        
        result = await session.execute(text("SELECT id, status, created_at FROM visit_logs WHERE session_id='8b30d5c6-9c34-405e-8ebd-68168fcbdb89'"))
        rows = result.fetchall()
        print(f"Found specific session: {rows}")

if __name__ == "__main__":
    asyncio.run(main())
