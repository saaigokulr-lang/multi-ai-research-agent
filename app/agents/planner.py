"""Planner agent: breaks a research question into focused sub-questions."""

from datetime import datetime, timezone

from app.core.logging import get_logger
from app.schemas.research import ResearchPlan, ResearchState
from app.services.llm_client import LLMCallError, LLMClient
from app.services.trace_recorder import compute_latency_ms, record_node_trace

logger = get_logger(__name__)

PLANNER_SYSTEM_PROMPT = """\
You are a research planning assistant. Given a research question, break it \
down into 3 to 5 focused, non-overlapping sub-questions that together cover \
the topic from distinct angles, such as: current state, benefits, risks and \
challenges, future trends, and comparisons or alternatives. Each \
sub-question should be specific enough to research independently and should \
not overlap significantly with the others."""


async def generate_research_plan(question: str, llm_client: LLMClient) -> ResearchPlan:
    """Ask the LLM to decompose ``question`` into a structured ``ResearchPlan``.

    Raises:
        LLMCallError: if the API call fails, or the response can't be parsed
            into a ``ResearchPlan``.
    """
    try:
        return await llm_client.generate_structured(
            prompt=question,
            response_model=ResearchPlan,
            system_prompt=PLANNER_SYSTEM_PROMPT,
        )
    except LLMCallError as exc:
        logger.error("Failed to generate research plan for question=%r: %s", question, exc)
        raise LLMCallError(f"Failed to generate research plan for question={question!r}: {exc}") from exc


async def planner_node(state: ResearchState) -> ResearchState:
    """Populate ``state.sub_questions`` from a freshly generated research plan."""
    llm_client = LLMClient()
    started_at = datetime.now(timezone.utc)
    try:
        plan = await generate_research_plan(state.question, llm_client)
    except Exception as exc:
        completed_at = datetime.now(timezone.utc)
        await record_node_trace(
            state.run_id,
            "planner",
            llm_client.total_input_tokens,
            llm_client.total_output_tokens,
            compute_latency_ms(started_at, completed_at),
            "failure",
            started_at,
            completed_at,
            error=str(exc),
        )
        raise

    state.sub_questions = plan.sub_questions
    state.total_tokens_used += llm_client.total_input_tokens + llm_client.total_output_tokens
    logger.info("planner_node generated %d sub-questions", len(state.sub_questions))

    completed_at = datetime.now(timezone.utc)
    await record_node_trace(
        state.run_id,
        "planner",
        llm_client.total_input_tokens,
        llm_client.total_output_tokens,
        compute_latency_ms(started_at, completed_at),
        "success",
        started_at,
        completed_at,
    )
    return state
