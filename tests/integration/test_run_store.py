"""Integration tests for the database-backed RunStore.

Uses an in-memory SQLite database (see conftest.py's `db_session` fixture)
rather than the real Supabase Postgres instance .env points at, so the
suite stays fast and isolated against the same schema
(app/database/models.py) -- the only thing that differs is the driver.
"""

from datetime import datetime, timezone

import pytest

from app.schemas.research import Finding
from app.services.run_store import RunStore

run_store = RunStore()

_QUESTION = "What are the benefits and risks of remote work?"


@pytest.mark.asyncio
async def test_create_run_creates_a_row_and_returns_a_valid_id(db_session) -> None:
    run_id = await run_store.create_run(db_session, _QUESTION)

    assert run_id
    record = await run_store.get_run(db_session, run_id)
    assert record is not None
    assert record["status"] == "pending"
    assert record["question"] == _QUESTION


@pytest.mark.asyncio
async def test_get_run_returns_none_for_unknown_id(db_session) -> None:
    record = await run_store.get_run(db_session, "00000000-0000-0000-0000-000000000000")
    assert record is None


@pytest.mark.asyncio
async def test_get_run_returns_none_for_malformed_id(db_session) -> None:
    record = await run_store.get_run(db_session, "not-a-uuid")
    assert record is None


@pytest.mark.asyncio
async def test_update_run_silently_noops_for_unknown_run_id(db_session) -> None:
    """A run deleted out from under a stale reference (e.g. a lost
    idempotency-key race) must not make update_run raise."""
    await run_store.update_run(
        db_session, "00000000-0000-0000-0000-000000000000", status="completed"
    )


@pytest.mark.asyncio
async def test_update_run_silently_noops_for_malformed_run_id(db_session) -> None:
    await run_store.update_run(db_session, "not-a-uuid", status="completed")


@pytest.mark.asyncio
async def test_delete_run_silently_noops_for_malformed_id(db_session) -> None:
    await run_store.delete_run(db_session, "not-a-uuid")


@pytest.mark.asyncio
async def test_update_run_updates_status_fields(db_session) -> None:
    run_id = await run_store.create_run(db_session, _QUESTION)

    await run_store.update_run(
        db_session,
        run_id,
        status="running",
        sub_questions=["What are the benefits?", "What are the risks?"],
        research_iterations=1,
        research_incomplete=False,
    )

    record = await run_store.get_run(db_session, run_id)
    assert record["status"] == "running"
    assert record["sub_questions"] == ["What are the benefits?", "What are the risks?"]
    assert record["research_iterations"] == 1
    assert record["research_incomplete"] is False


@pytest.mark.asyncio
async def test_update_run_writes_findings_and_final_report(db_session) -> None:
    run_id = await run_store.create_run(db_session, _QUESTION)
    findings = [
        Finding(
            claim="Remote work improves work-life balance.",
            source_url="https://a.example.com",
            snippet="Remote work improves work-life balance for many employees.",
            sub_question="What are the benefits?",
            retrieved_at=datetime.now(timezone.utc),
        ),
        Finding(
            claim="Remote work can weaken team cohesion.",
            source_url="https://b.example.com",
            snippet="Remote work can weaken team cohesion over time.",
            sub_question="What are the risks?",
            retrieved_at=datetime.now(timezone.utc),
        ),
    ]

    await run_store.update_run(
        db_session,
        run_id,
        status="completed",
        final_report="## Introduction\n\nSome report.",
        findings=findings,
    )

    record = await run_store.get_run(db_session, run_id)
    assert record["status"] == "completed"
    assert record["final_report"] == "## Introduction\n\nSome report."
    assert record["findings_count"] == 2


@pytest.mark.asyncio
async def test_update_run_upserts_final_report_on_repeat_calls(db_session) -> None:
    run_id = await run_store.create_run(db_session, _QUESTION)

    await run_store.update_run(db_session, run_id, status="completed", final_report="First draft.")
    await run_store.update_run(db_session, run_id, status="completed", final_report="Revised draft.")

    record = await run_store.get_run(db_session, run_id)
    assert record["final_report"] == "Revised draft."


@pytest.mark.asyncio
async def test_get_run_assembles_expected_shape_after_completion(db_session) -> None:
    run_id = await run_store.create_run(db_session, _QUESTION)
    findings = [
        Finding(
            claim="Remote work improves work-life balance.",
            source_url="https://a.example.com",
            snippet="Remote work improves work-life balance for many employees.",
            sub_question="What are the benefits?",
            retrieved_at=datetime.now(timezone.utc),
        )
    ]

    await run_store.update_run(
        db_session,
        run_id,
        status="completed",
        sub_questions=["What are the benefits?"],
        research_iterations=1,
        research_incomplete=False,
        final_report="## Introduction\n\nSome report.",
        findings=findings,
    )

    record = await run_store.get_run(db_session, run_id)

    assert record == {
        "run_id": run_id,
        "status": "completed",
        "question": _QUESTION,
        "created_at": record["created_at"],
        "sub_questions": ["What are the benefits?"],
        "research_iterations": 1,
        "research_incomplete": False,
        "final_report": "## Introduction\n\nSome report.",
        "error": None,
        "findings_count": 1,
        "total_tokens_used": 0,
        "budget_exceeded": False,
    }
