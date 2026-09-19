"""Async SQLAlchemy engine/session setup.

Supabase (and most managed Postgres providers) hand out a plain
``postgresql://`` connection string; SQLAlchemy's async engine needs the
driver named explicitly in the URL scheme, so we rewrite it to use asyncpg.
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings


def _to_async_url(database_url: str) -> str:
    """Rewrite a plain postgresql:// URL to use the asyncpg driver."""
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return database_url


_settings = get_settings()

engine = create_async_engine(_to_async_url(_settings.DATABASE_URL))
async_session_factory = async_sessionmaker(engine, expire_on_commit=False)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding a database session, closed after use."""
    async with async_session_factory() as session:
        yield session
