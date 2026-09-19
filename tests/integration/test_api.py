"""Integration tests for the FastAPI research endpoints.

The compiled research graph is mocked (via app.api.routes._invoke_graph) so
these tests make no real LLM or search calls. Idempotency tests mock the
Redis client (via app.api.routes.get_redis_client) with an in-memory fake
implementing just enough of GET/SET NX/EX semantics -- no real Redis
connection is needed either.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.api import routes
from app.database.models import ResearchRun
from app.main import app
from app.schemas.research import Finding, ResearchState
from app.services.trace_recorder import record_trace

client = TestClient(app)


class _FakeRedis:
    """Minimal in-memory stand-in for the bits of redis-py we use."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self._store.get(key)

    async def set(self, key: str, value: str, nx: bool = False, ex: int | None = None) -> bool:
        if nx and key in self._store:
            return False
        self._store[key] = value
        return True


class _LosingRaceRedis:
    """Forces the exact sequence a lost NX race produces: the initial GET
    misses (so the caller proceeds to create its own run), the NX SET
    always fails (someone else claimed the key first), and a follow-up GET
    returns that other request's run_id."""

    def __init__(self, winning_run_id: str) -> None:
        self._winning_run_id = winning_run_id
        self._get_calls = 0

    async def get(self, key: str) -> str | None:
        self._get_calls += 1
        return None if self._get_calls == 1 else self._winning_run_id

    async def set(self, key: str, value: str, nx: bool = False, ex: int | None = None) -> bool:
        return False


_QUESTION = "What are the benefits and risks of remote work?"

_SAMPLE_STATE = ResearchState(
    question=_QUESTION,
    sub_questions=["What are the benefits?", "What are the risks?"],
    findings=[
        Finding(
            claim="Remote work improves work-life balance.",
            source_url="https://a.example.com",
            snippet="Remote work improves work-life balance for many employees.",
            sub_question="What are the benefits?",
        )
    ],
    research_complete=True,
    research_iterations=1,
    research_incomplete=False,
    final_report="## Introduction\n\nSome report.",
)


def test_create_research_returns_202_with_pending_status() -> None:
    with patch.object(routes, "_invoke_graph", new=AsyncMock(return_value=_SAMPLE_STATE)):
        response = client.post("/research", json={"question": _QUESTION})

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "pending"
    assert body["run_id"]


def test_get_research_status_returns_404_for_unknown_id() -> None:
    response = client.get("/research/does-not-exist")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_research_status_reflects_completed_after_background_task(db_session) -> None:
    run_id = await routes.run_store.create_run(db_session, _QUESTION)

    with patch.object(routes, "_invoke_graph", new=AsyncMock(return_value=_SAMPLE_STATE)):
        await routes._execute_research_run(run_id, _QUESTION, graph=None)

    response = client.get(f"/research/{run_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["question"] == _QUESTION
    assert body["sub_questions"] == _SAMPLE_STATE.sub_questions
    assert body["research_iterations"] == 1
    assert body["research_incomplete"] is False
    assert body["final_report"] == _SAMPLE_STATE.final_report


@pytest.mark.asyncio
async def test_execute_research_run_records_failure_status_when_graph_raises(db_session) -> None:
    run_id = await routes.run_store.create_run(db_session, _QUESTION)

    with patch.object(routes, "_invoke_graph", new=AsyncMock(side_effect=RuntimeError("graph exploded"))):
        await routes._execute_research_run(run_id, _QUESTION, graph=None)

    record = await routes.run_store.get_run(db_session, run_id)
    assert record["status"] == "failed"
    assert record["error"] == "graph exploded"


def test_create_research_with_too_short_question_returns_422() -> None:
    response = client.post("/research", json={"question": "hi"})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_same_idempotency_key_returns_same_run_id_and_triggers_graph_once(db_session) -> None:
    fake_redis = _FakeRedis()
    mock_invoke_graph = AsyncMock(return_value=_SAMPLE_STATE)

    with patch.object(routes, "_invoke_graph", new=mock_invoke_graph), patch.object(
        routes, "get_redis_client", new=Mock(return_value=fake_redis)
    ):
        response1 = client.post(
            "/research", json={"question": _QUESTION, "idempotency_key": "same-key"}
        )
        response2 = client.post(
            "/research", json={"question": _QUESTION, "idempotency_key": "same-key"}
        )

    assert response1.status_code == 202
    assert response2.status_code == 202
    assert response1.json()["run_id"] == response2.json()["run_id"]
    mock_invoke_graph.assert_called_once()

    row_count = await db_session.scalar(select(func.count()).select_from(ResearchRun))
    assert row_count == 1


def test_different_idempotency_keys_create_independent_runs() -> None:
    fake_redis = _FakeRedis()
    mock_invoke_graph = AsyncMock(return_value=_SAMPLE_STATE)

    with patch.object(routes, "_invoke_graph", new=mock_invoke_graph), patch.object(
        routes, "get_redis_client", new=Mock(return_value=fake_redis)
    ):
        response1 = client.post(
            "/research", json={"question": _QUESTION, "idempotency_key": "key-one"}
        )
        response2 = client.post(
            "/research", json={"question": _QUESTION, "idempotency_key": "key-two"}
        )

    assert response1.json()["run_id"] != response2.json()["run_id"]
    assert mock_invoke_graph.call_count == 2


@pytest.mark.asyncio
async def test_losing_idempotency_race_deletes_the_orphaned_run(db_session) -> None:
    # Simulate a concurrent request that already won: it created its own
    # run and claimed the idempotency key first.
    winning_run_id = await routes.run_store.create_run(db_session, _QUESTION)

    fake_redis = _LosingRaceRedis(winning_run_id)
    mock_invoke_graph = AsyncMock(return_value=_SAMPLE_STATE)

    with patch.object(routes, "_invoke_graph", new=mock_invoke_graph), patch.object(
        routes, "get_redis_client", new=Mock(return_value=fake_redis)
    ):
        response = client.post(
            "/research", json={"question": _QUESTION, "idempotency_key": "race-key"}
        )

    assert response.status_code == 202
    assert response.json()["run_id"] == winning_run_id
    mock_invoke_graph.assert_not_called()

    row_count = await db_session.scalar(select(func.count()).select_from(ResearchRun))
    assert row_count == 1


@pytest.mark.asyncio
async def test_get_research_trace_returns_steps_and_token_totals(db_session) -> None:
    run_id = await routes.run_store.create_run(db_session, _QUESTION)
    started_at = datetime.now(timezone.utc)
    completed_at = datetime.now(timezone.utc)
    await record_trace(
        db_session,
        run_id,
        node_name="planner",
        input_tokens=100,
        output_tokens=50,
        latency_ms=250.0,
        status="success",
        started_at=started_at,
        completed_at=completed_at,
    )
    await routes.run_store.update_run(db_session, run_id, total_tokens_used=150, budget_exceeded=False)

    response = client.get(f"/research/{run_id}/trace")

    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == run_id
    assert body["total_tokens_used"] == 150
    assert body["budget_exceeded"] is False
    assert len(body["steps"]) == 1
    step = body["steps"][0]
    assert step["node_name"] == "planner"
    assert step["input_tokens"] == 100
    assert step["output_tokens"] == 50
    assert step["status"] == "success"


def test_get_research_trace_returns_404_for_unknown_id() -> None:
    response = client.get("/research/does-not-exist/trace")
    assert response.status_code == 404


def test_no_idempotency_key_skips_redis_entirely() -> None:
    mock_get_redis_client = Mock()

    with patch.object(routes, "_invoke_graph", new=AsyncMock(return_value=_SAMPLE_STATE)), patch.object(
        routes, "get_redis_client", new=mock_get_redis_client
    ):
        response = client.post("/research", json={"question": _QUESTION})

    assert response.status_code == 202
    mock_get_redis_client.assert_not_called()
