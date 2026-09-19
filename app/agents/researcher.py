"""Researcher agent: turns a sub-question into web-search-grounded findings."""

from datetime import datetime, timezone

from pydantic import BaseModel

from app.core.logging import get_logger
from app.schemas.research import Finding, ResearchState
from app.services.llm_client import LLMCallError, LLMClient
from app.tools.tavily_search import SearchToolError, search_web

logger = get_logger(__name__)

SEARCH_QUERY_SYSTEM_PROMPT = """\
You turn a research sub-question into a concise, effective web search query. \
Strip filler words and focus on the key terms a search engine would match \
well. Respond with only the search query text, nothing else."""

RESEARCHER_SYSTEM_PROMPT = """\
You are a research assistant extracting claims from web search results. You \
will be given a sub-question and a set of search results, each with a \
title, URL, and content snippet. Extract only claims that are directly \
supported by the provided content - do not use outside knowledge and do not \
fabricate information. Each claim must be paired with the exact source URL \
of the search result it came from. If none of the results support any \
claims relevant to the sub-question, return an empty list of findings."""


class ExtractedFinding(BaseModel):
    """A single claim/source pair as extracted by the LLM."""

    claim: str
    source_url: str
    snippet: str


class ExtractedFindings(BaseModel):
    """Structured output produced by the researcher agent's extraction step."""

    findings: list[ExtractedFinding]


def _build_extraction_prompt(sub_question: str, search_results: list[dict]) -> str:
    formatted_results = "\n\n".join(
        f"Result {i}:\nTitle: {result['title']}\nURL: {result['url']}\nContent: {result['content']}"
        for i, result in enumerate(search_results, start=1)
    )
    return f"Sub-question: {sub_question}\n\nSearch results:\n\n{formatted_results}"


async def generate_search_query(sub_question: str, llm_client: LLMClient) -> str:
    """Ask the LLM to turn ``sub_question`` into a concise web search query.

    Raises:
        LLMCallError: if the underlying API call fails.
    """
    try:
        response = await llm_client.generate(
            prompt=sub_question,
            system_prompt=SEARCH_QUERY_SYSTEM_PROMPT,
            temperature=0.3,
        )
    except LLMCallError as exc:
        logger.error("Failed to generate search query for sub_question=%r: %s", sub_question, exc)
        raise LLMCallError(
            f"Failed to generate search query for sub_question={sub_question!r}: {exc}"
        ) from exc
    return response.content.strip()


async def research_sub_question(sub_question: str, llm_client: LLMClient) -> list[Finding]:
    """Search the web for ``sub_question`` and extract supported findings.

    Search and extraction failures are logged and treated as "no findings"
    for this sub-question rather than raised, so one failed sub-question
    doesn't abort the whole research run.
    """
    try:
        query = await generate_search_query(sub_question, llm_client)
    except LLMCallError as exc:
        logger.error("Skipping sub_question=%r after query generation failure: %s", sub_question, exc)
        return []

    try:
        search_results = await search_web(query)
    except SearchToolError as exc:
        logger.error("Skipping sub_question=%r after search failure: %s", sub_question, exc)
        return []

    if not search_results:
        logger.warning("No search results for sub_question=%r (query=%r)", sub_question, query)
        return []

    try:
        extracted = await llm_client.generate_structured(
            prompt=_build_extraction_prompt(sub_question, search_results),
            response_model=ExtractedFindings,
            system_prompt=RESEARCHER_SYSTEM_PROMPT,
        )
    except LLMCallError as exc:
        logger.error("Skipping sub_question=%r after extraction failure: %s", sub_question, exc)
        return []

    retrieved_at = datetime.now(timezone.utc)
    return [
        Finding(
            claim=item.claim,
            source_url=item.source_url,
            snippet=item.snippet,
            sub_question=sub_question,
            retrieved_at=retrieved_at,
        )
        for item in extracted.findings
    ]


async def researcher_node(state: ResearchState) -> ResearchState:
    """Sequentially research every sub-question, accumulating findings."""
    llm_client = LLMClient()
    for sub_question in state.sub_questions:
        findings = await research_sub_question(sub_question, llm_client)
        state.findings.extend(findings)
    logger.info(
        "researcher_node collected %d findings across %d sub-questions",
        len(state.findings),
        len(state.sub_questions),
    )
    return state
