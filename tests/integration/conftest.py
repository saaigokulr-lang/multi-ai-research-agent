"""Shared fixtures for integration tests: an isolated SQLite database.

Tests run against an in-memory SQLite database via aiosqlite rather than
the real Supabase Postgres instance .env points at -- same schema as
app/database/models.py, different driver -- so the suite stays fast and
each test gets a clean slate. Each test gets a fresh database, and every
module that opens its own session directly (rather than through the
get_db_session dependency) -- app.api.routes' background task and
app.services.trace_recorder's node-trace writes -- gets its own
module-level async_session_factory reference patched to match, since
patching the source in app.database.session wouldn't affect names other
modules already imported from it.
"""

from collections.abc import AsyncGenerator

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.api import routes
from app.database.models import Base
from app.database.session import get_db_session
from app.main import app
from app.services import trace_recorder


@pytest.fixture(autouse=True)
async def _test_database() -> AsyncGenerator[async_sessionmaker[AsyncSession], None]:
    """Point the app at a fresh in-memory SQLite database for one test."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def _override_get_db_session() -> AsyncGenerator[AsyncSession, None]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = _override_get_db_session
    original_routes_session_factory = routes.async_session_factory
    original_trace_session_factory = trace_recorder.async_session_factory
    routes.async_session_factory = session_factory
    trace_recorder.async_session_factory = session_factory

    yield session_factory

    routes.async_session_factory = original_routes_session_factory
    trace_recorder.async_session_factory = original_trace_session_factory
    app.dependency_overrides.pop(get_db_session, None)
    await engine.dispose()


@pytest.fixture
async def db_session(_test_database: async_sessionmaker[AsyncSession]) -> AsyncGenerator[AsyncSession, None]:
    """A single session against the per-test database, for direct RunStore calls."""
    async with _test_database() as session:
        yield session
