import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from server.database.models import Base
from server.config import settings

logger = logging.getLogger(__name__)

# Fallback to persistent sqlite if postgres is not configured
import os
DB_URL = settings.database_url or f"sqlite+aiosqlite:///{os.path.abspath('server/data/visits.db')}"

engine = create_async_engine(DB_URL, echo=False)
async_session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

async def init_db():
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

async def get_db_session() -> AsyncSession:
    async with async_session() as session:
        yield session
