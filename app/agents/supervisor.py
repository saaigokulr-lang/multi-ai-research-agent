"""Supervisor agent: judges whether collected research is sufficient to write on."""

from datetime import datetime, timezone

from pydantic import BaseModel

from app.core.logging import get_logger
from app.schemas.research import MAX_RESEARCH_ITERATIONS, ResearchState
from app.services.llm_client import LLMCallError, LLMClient
from app.services.trace_recorder import compute_latency_ms, record_node_trace

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


# The Supervisor only judges sufficiency, not citation accuracy, so its
# prompt doesn't need source_url/full snippets -- just enough of each claim
# to gauge substance. This keeps the prompt from growing unboundedly with
# findings the way it did when full detail was included (writer.py's prompt
# still gets full finding detail, since it does need it for citations).
_CLAIM_TRUNCATE_LENGTH = 100

# Rough token estimate for logging only -- no tokenizer dependency, just the
# common ~4-chars-per-token heuristic, good enough to flag a growing prompt
# in logs before it trips a provider's real token-per-minute limit.
_CHARS_PER_TOKEN_ESTIMATE = 4


def _truncate_claim(claim: str, max_length: int = _CLAIM_TRUNCATE_LENGTH) -> str:
    if len(claim) <= max_length:
        return claim
    return claim[:max_length].rstrip() + "..."


def _build_evaluation_prompt(state: ResearchState) -> str:
    claims_by_sub_question: dict[str, list[str]] = {}
    for finding in state.findings:
        claims_by_sub_question.setdefault(finding.sub_question, []).append(finding.claim)

    if claims_by_sub_question:
        findings_summary = "\n".join(
            f"- {sub_question} ({len(claims)} finding{'s' if len(claims) != 1 else ''}):\n"
            + "\n".join(f"  - {_truncate_claim(claim)}" for claim in claims)
            for sub_question, claims in claims_by_sub_question.items()
        )
    else:
        findings_summary = "(no findings collected yet)"

    sub_questions_summary = "\n".join(f"- {sub_question}" for sub_question in state.sub_questions)
    return (
        f"Research question: {state.question}\n\n"
        f"Sub-questions:\n{sub_questions_summary}\n\n"
        f"Findings so far (grouped by sub-question, claims truncated):\n{findings_summary}"
    )


async def evaluate_research(state: ResearchState, llm_client: LLMClient) -> SupervisorDecision:
    """Ask the LLM to judge whether ``state``'s research is sufficient.

    Fails open: if the LLM call fails, defaults to ``research_complete=True``
    rather than risking an infinite research loop.
    """
    prompt = _build_evaluation_prompt(state)
    logger.info(
        "Supervisor evaluation prompt built: %d chars (~%d tokens estimated)",
        len(prompt),
        len(prompt) // _CHARS_PER_TOKEN_ESTIMATE,
    )
    try:
        return await llm_client.generate_structured(
            prompt=prompt,
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
    started_at = datetime.now(timezone.utc)
    try:
        decision = await evaluate_research(state, llm_client)

        state.research_complete = decision.research_complete
        state.research_iterations += 1

        limit_forced = False
        if state.research_iterations >= MAX_RESEARCH_ITERATIONS and not decision.research_complete:
            state.research_complete = True
            state.research_incomplete = True
            limit_forced = True

        state.total_tokens_used += llm_client.total_input_tokens + llm_client.total_output_tokens

        logger.info(
            "supervisor_node decision=%s iteration=%d/%d limit_forced=%s reasoning=%r",
            decision.research_complete,
            state.research_iterations,
            MAX_RESEARCH_ITERATIONS,
            limit_forced,
            decision.reasoning,
        )
    except Exception as exc:
        completed_at = datetime.now(timezone.utc)
        await record_node_trace(
            state.run_id,
            "supervisor",
            llm_client.total_input_tokens,
            llm_client.total_output_tokens,
            compute_latency_ms(started_at, completed_at),
            "failure",
            started_at,
            completed_at,
            error=str(exc),
        )
        raise

    completed_at = datetime.now(timezone.utc)
    await record_node_trace(
        state.run_id,
        "supervisor",
        llm_client.total_input_tokens,
        llm_client.total_output_tokens,
        compute_latency_ms(started_at, completed_at),
        "success",
        started_at,
        completed_at,
    )
    return state
