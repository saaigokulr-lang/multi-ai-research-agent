"""Planner agent: breaks a research question into focused sub-questions."""

from app.core.logging import get_logger
from app.schemas.research import ResearchPlan, ResearchState
from app.services.llm_client import LLMCallError, LLMClient

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
    plan = await generate_research_plan(state.question, llm_client)
    state.sub_questions = plan.sub_questions
    logger.info("planner_node generated %d sub-questions", len(state.sub_questions))
    return state
