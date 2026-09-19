"""FastAPI routes for kicking off and inspecting research runs."""

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import (
    CreateResearchRequest,
    CreateResearchResponse,
    ResearchStatusResponse,
    TraceResponse,
    TraceStep,
)
from app.core.logging import get_logger
from app.database.session import async_session_factory, get_db_session
from app.schemas.research import ResearchState
from app.services.redis_client import get_redis_client
from app.services.run_store import run_store
from app.services.trace_recorder import get_traces_for_run

logger = get_logger(__name__)

router = APIRouter(prefix="/research", tags=["research"])

_IDEMPOTENCY_TTL_SECONDS = 86400  # 24h: long enough to catch accidental retries,
# short enough not to permanently block a legitimately repeated question later.


async def _invoke_graph(graph: CompiledStateGraph, question: str, run_id: str) -> ResearchState:
    """Run the compiled research graph on ``question`` and return the final state."""
    result = await graph.ainvoke(ResearchState(question=question, run_id=run_id))
    return ResearchState.model_validate(result)


async def _execute_research_run(run_id: str, question: str, graph: CompiledStateGraph) -> None:
    """Background task: run the graph and record the outcome on the run store.

    Runs outside the request's dependency-injection scope, so it opens its
    own database session directly from the session factory. Catches any
    exception so one failed run can't crash the server.
    """
    async with async_session_factory() as session:
        await run_store.update_run(session, run_id, status="running")
        try:
            state = await _invoke_graph(graph, question, run_id)
        except Exception as exc:
            logger.error("Research run %s failed: %s", run_id, exc)
            await run_store.update_run(session, run_id, status="failed", error=str(exc))
            return
        await run_store.update_run(
            session,
            run_id,
            status="completed",
            sub_questions=state.sub_questions,
            research_iterations=state.research_iterations,
            research_incomplete=state.research_incomplete,
            final_report=state.final_report,
            findings=state.findings,
            total_tokens_used=state.total_tokens_used,
            budget_exceeded=state.budget_exceeded,
        )


async def _response_for_existing_run(db: AsyncSession, run_id: str) -> CreateResearchResponse:
    """Build a response for a run that already exists (an idempotent replay)."""
    record = await run_store.get_run(db, run_id)
    return CreateResearchResponse(run_id=run_id, status=record["status"] if record else "pending")


@router.post("", response_model=CreateResearchResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_research(
    payload: CreateResearchRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db_session),
) -> CreateResearchResponse:
    """Start a new research run and return immediately without waiting for it.

    If ``idempotency_key`` is provided, a repeat request with the same key
    returns the original run instead of starting a duplicate.
    """
    idempotency_redis_key: str | None = None
    if payload.idempotency_key is not None:
        redis_client = get_redis_client()
        idempotency_redis_key = f"idempotency:{payload.idempotency_key}"

        existing_run_id = await redis_client.get(idempotency_redis_key)
        if existing_run_id is not None:
            return await _response_for_existing_run(db, existing_run_id)

    run_id = await run_store.create_run(db, payload.question)

    if idempotency_redis_key is not None:
        won_race = await redis_client.set(
            idempotency_redis_key, run_id, nx=True, ex=_IDEMPOTENCY_TTL_SECONDS
        )
        if not won_race:
            # Lost the race to a concurrent request with the same key --
            # delete the run we already created rather than leaving it
            # orphaned at status="pending" forever, and use whichever
            # run_id actually got stored instead.
            await run_store.delete_run(db, run_id)
            winning_run_id = await redis_client.get(idempotency_redis_key)
            return await _response_for_existing_run(db, winning_run_id)

    background_tasks.add_task(
        _execute_research_run, run_id, payload.question, request.app.state.research_graph
    )
    return CreateResearchResponse(run_id=run_id, status="pending")


@router.get("/{run_id}", response_model=ResearchStatusResponse)
async def get_research_status(
    run_id: str, db: AsyncSession = Depends(get_db_session)
) -> ResearchStatusResponse:
    """Return the current status (and results, if completed) of a research run."""
    record = await run_store.get_run(db, run_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    return ResearchStatusResponse(
        run_id=record["run_id"],
        status=record["status"],
        question=record["question"],
        created_at=record["created_at"],
        sub_questions=record["sub_questions"],
        research_iterations=record["research_iterations"],
        research_incomplete=record["research_incomplete"],
        final_report=record["final_report"],
        error=record["error"],
    )


@router.get("/{run_id}/trace", response_model=TraceResponse)
async def get_research_trace(
    run_id: str, db: AsyncSession = Depends(get_db_session)
) -> TraceResponse:
    """Return the step-by-step execution trace for a research run."""
    record = await run_store.get_run(db, run_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    traces = await get_traces_for_run(db, run_id)
    return TraceResponse(
        run_id=record["run_id"],
        status=record["status"],
        steps=[
            TraceStep(
                node_name=trace.node_name,
                input_tokens=trace.input_tokens,
                output_tokens=trace.output_tokens,
                latency_ms=trace.latency_ms,
                status=trace.status,
                started_at=trace.started_at,
                completed_at=trace.completed_at,
                error=trace.error,
            )
            for trace in traces
        ],
        total_tokens_used=record["total_tokens_used"],
        budget_exceeded=record["budget_exceeded"],
    )
