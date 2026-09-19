"""Integration tests for the trace_recorder service.

Uses the same in-memory SQLite pattern as test_run_store.py (see
tests/integration/conftest.py's db_session fixture) rather than a real
Postgres/Supabase connection, for fast, isolated tests. Placed under
tests/integration (not tests/unit) because that's where the shared DB
fixture lives -- matching test_run_store.py's precedent.
"""

from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from app.services import trace_recorder as trace_recorder_module
from app.services.run_store import run_store
from app.services.trace_recorder import get_traces_for_run, record_node_trace, record_trace

_QUESTION = "What are the benefits and risks of remote work?"


@pytest.mark.asyncio
async def test_record_trace_inserts_a_row(db_session) -> None:
    run_id = await run_store.create_run(db_session, _QUESTION)
    started_at = datetime.now(timezone.utc)
    completed_at = datetime.now(timezone.utc)

    await record_trace(
        db_session,
        run_id,
        node_name="planner",
        input_tokens=100,
        output_tokens=50,
        latency_ms=250.5,
        status="success",
        started_at=started_at,
        completed_at=completed_at,
    )

    traces = await get_traces_for_run(db_session, run_id)
    assert len(traces) == 1
    trace = traces[0]
    assert trace.node_name == "planner"
    assert trace.input_tokens == 100
    assert trace.output_tokens == 50
    assert trace.latency_ms == 250.5
    assert trace.status == "success"
    assert trace.error is None


@pytest.mark.asyncio
async def test_record_trace_stores_failure_and_error_message(db_session) -> None:
    run_id = await run_store.create_run(db_session, _QUESTION)
    started_at = datetime.now(timezone.utc)
    completed_at = datetime.now(timezone.utc)

    await record_trace(
        db_session,
        run_id,
        node_name="researcher",
        input_tokens=10,
        output_tokens=0,
        latency_ms=42.0,
        status="failure",
        started_at=started_at,
        completed_at=completed_at,
        error="boom",
    )

    traces = await get_traces_for_run(db_session, run_id)
    assert len(traces) == 1
    assert traces[0].status == "failure"
    assert traces[0].error == "boom"


@pytest.mark.asyncio
async def test_record_node_trace_noops_when_run_id_is_none() -> None:
    # Should return cleanly without needing a database at all.
    await record_node_trace(
        None, "planner", 0, 0, 1.0, "success", datetime.now(timezone.utc), datetime.now(timezone.utc)
    )


@pytest.mark.asyncio
async def test_record_node_trace_writes_via_its_own_session(db_session) -> None:
    run_id = await run_store.create_run(db_session, _QUESTION)
    started_at = datetime.now(timezone.utc)
    completed_at = datetime.now(timezone.utc)

    await record_node_trace(run_id, "writer", 10, 20, 5.0, "success", started_at, completed_at)

    traces = await get_traces_for_run(db_session, run_id)
    assert len(traces) == 1
    assert traces[0].node_name == "writer"


@pytest.mark.asyncio
async def test_record_node_trace_swallows_database_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """A DB hiccup while recording a trace is observability, not core
    behavior -- it must be logged, never raised, so it can't crash (or mask
    the real error from) the node function that called it."""
    monkeypatch.setattr(
        trace_recorder_module, "async_session_factory", Mock(side_effect=RuntimeError("db is down"))
    )

    await record_node_trace(
        "00000000-0000-0000-0000-000000000000",
        "planner",
        0,
        0,
        1.0,
        "success",
        datetime.now(timezone.utc),
        datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_get_traces_for_run_returns_empty_list_for_malformed_id(db_session) -> None:
    traces = await get_traces_for_run(db_session, "not-a-uuid")
    assert traces == []
