"""Persists per-node execution traces (tokens, latency, success/failure).

Isolated here so every agent node function records traces the same way,
without each one duplicating session-management and insert logic.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.database.models import ExecutionTrace
from app.database.session import async_session_factory

logger = get_logger(__name__)


async def record_trace(
    session: AsyncSession,
    run_id: str,
    node_name: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
    status: str,
    started_at: datetime,
    completed_at: datetime,
    error: Optional[str] = None,
) -> None:
    """Insert one ExecutionTrace row for a single node run."""
    session.add(
        ExecutionTrace(
            run_id=UUID(run_id),
            node_name=node_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            status=status,
            error=error,
            started_at=started_at,
            completed_at=completed_at,
        )
    )
    await session.commit()


async def record_node_trace(
    run_id: Optional[str],
    node_name: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
    status: str,
    started_at: datetime,
    completed_at: datetime,
    error: Optional[str] = None,
) -> None:
    """Record a node's trace, opening its own short-lived session.

    A no-op if ``run_id`` is None, so node functions invoked directly (e.g.
    in unit tests, outside a real API-triggered run) don't need a database
    at all -- callers don't have to guard this themselves.
    """
    if run_id is None:
        return
    try:
        async with async_session_factory() as session:
            await record_trace(
                session,
                run_id,
                node_name,
                input_tokens,
                output_tokens,
                latency_ms,
                status,
                started_at,
                completed_at,
                error,
            )
    except Exception as exc:
        # Trace recording is observability, not core behavior -- a DB hiccup
        # here must never take down (or mask the real error from) a node.
        logger.error("Failed to record trace for run_id=%s node_name=%s: %s", run_id, node_name, exc)


def compute_latency_ms(started_at: datetime, completed_at: datetime) -> float:
    """Milliseconds between two timestamps, for populating a trace's latency_ms."""
    return (completed_at - started_at).total_seconds() * 1000


async def get_traces_for_run(session: AsyncSession, run_id: str) -> list[ExecutionTrace]:
    """Return all execution traces for a run, ordered by start time."""
    try:
        run_uuid = UUID(run_id)
    except ValueError:
        return []
    result = await session.scalars(
        select(ExecutionTrace).where(ExecutionTrace.run_id == run_uuid).order_by(ExecutionTrace.started_at)
    )
    return list(result.all())
