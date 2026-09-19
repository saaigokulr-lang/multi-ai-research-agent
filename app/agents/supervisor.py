"""Supervisor agent: judges whether collected research is sufficient to write on."""

from pydantic import BaseModel

from app.core.logging import get_logger
from app.schemas.research import MAX_RESEARCH_ITERATIONS, ResearchState
from app.services.llm_client import LLMCallError, LLMClient

logger = get_logger(__name__)

SUPERVISOR_SYSTEM_PROMPT = """\
You are a research supervisor. Given a research question, its sub-questions, \
and the findings collected so far, judge whether the research is sufficient \
to write a comprehensive, well-supported report. Consider whether every \
sub-question has at least some supporting findings, and whether those \
findings seem substantive rather than thin. Return a boolean decision and \
brief reasoning."""


class SupervisorDecision(BaseModel):
    """Structured output produced by the supervisor agent."""

    research_complete: bool
    reasoning: str


def _build_evaluation_prompt(state: ResearchState) -> str:
    findings_summary = (
        "\n".join(f"- [{finding.sub_question}] {finding.claim}" for finding in state.findings)
        or "(no findings collected yet)"
    )
    sub_questions_summary = "\n".join(f"- {sub_question}" for sub_question in state.sub_questions)
    return (
        f"Research question: {state.question}\n\n"
        f"Sub-questions:\n{sub_questions_summary}\n\n"
        f"Findings so far:\n{findings_summary}"
    )


async def evaluate_research(state: ResearchState, llm_client: LLMClient) -> SupervisorDecision:
    """Ask the LLM to judge whether ``state``'s research is sufficient.

    Fails open: if the LLM call fails, defaults to ``research_complete=True``
    rather than risking an infinite research loop.
    """
    try:
        return await llm_client.generate_structured(
            prompt=_build_evaluation_prompt(state),
            response_model=SupervisorDecision,
            system_prompt=SUPERVISOR_SYSTEM_PROMPT,
        )
    except LLMCallError as exc:
        logger.error("Supervisor evaluation failed, failing open: %s", exc)
        return SupervisorDecision(
            research_complete=True,
            reasoning=f"Evaluation failed, defaulting to complete: {exc}",
        )


async def supervisor_node(state: ResearchState) -> ResearchState:
    """Evaluate research sufficiency and apply the iteration-limit safety cap."""
    llm_client = LLMClient()
    decision = await evaluate_research(state, llm_client)

    state.research_complete = decision.research_complete
    state.research_iterations += 1

    limit_forced = False
    if state.research_iterations >= MAX_RESEARCH_ITERATIONS and not decision.research_complete:
        state.research_complete = True
        state.research_incomplete = True
        limit_forced = True

    logger.info(
        "supervisor_node decision=%s iteration=%d/%d limit_forced=%s reasoning=%r",
        decision.research_complete,
        state.research_iterations,
        MAX_RESEARCH_ITERATIONS,
        limit_forced,
        decision.reasoning,
    )
    return state
