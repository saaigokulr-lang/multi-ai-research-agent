"""Writer agent: turns collected findings into a Markdown research report."""

from app.core.logging import get_logger
from app.schemas.research import ResearchState
from app.services.llm_client import LLMCallError, LLMClient

logger = get_logger(__name__)

WRITER_SYSTEM_PROMPT = """\
You are a research report writer. Given a research question, its \
sub-questions, and a set of findings, write a well-structured research \
report in Markdown format with exactly these sections, as Markdown \
headings: ## Introduction, ## Findings, ## Analysis, ## Conclusion, and \
## Sources. Only make claims that are directly supported by the provided \
findings - do not add outside knowledge or fabricate information. In the \
Findings and Analysis sections, cite sources inline where relevant (e.g. \
by URL or a bracketed number matching the Sources list). You will be given \
an authoritative list of unique source URLs separate from the findings - \
use exactly that list for the Sources section, rather than compiling it \
yourself from the findings, since the same source may appear against \
multiple findings. If you are told that the research was flagged \
incomplete, include a brief, honest note near the introduction or \
conclusion mentioning that some areas may be under-researched due to \
time/iteration limits - do not hide this from the reader."""


def _build_report_prompt(state: ResearchState) -> str:
    findings_summary = (
        "\n".join(
            f"- [{finding.sub_question}] {finding.claim} (source: {finding.source_url})"
            for finding in state.findings
        )
        or "(no findings collected)"
    )
    sub_questions_summary = "\n".join(f"- {sub_question}" for sub_question in state.sub_questions)
    unique_sources = dict.fromkeys(finding.source_url for finding in state.findings)
    sources_summary = "\n".join(f"- {url}" for url in unique_sources) or "(no sources collected)"
    return (
        f"Research question: {state.question}\n\n"
        f"Sub-questions:\n{sub_questions_summary}\n\n"
        f"Findings:\n{findings_summary}\n\n"
        f"Unique sources (use exactly this list for the Sources section):\n{sources_summary}\n\n"
        f"Research flagged incomplete: {state.research_incomplete}"
    )


async def generate_report(state: ResearchState, llm_client: LLMClient) -> str:
    """Ask the LLM to write a Markdown research report from ``state``.

    Raises:
        LLMCallError: if the underlying API call fails.
    """
    try:
        response = await llm_client.generate(
            prompt=_build_report_prompt(state),
            system_prompt=WRITER_SYSTEM_PROMPT,
        )
    except LLMCallError as exc:
        logger.error("Failed to generate report for question=%r: %s", state.question, exc)
        raise
    return response.content


async def writer_node(state: ResearchState) -> ResearchState:
    """Populate ``state.final_report`` with a freshly generated report."""
    llm_client = LLMClient()
    report = await generate_report(state, llm_client)
    state.final_report = report
    logger.info(
        "writer_node generated report of %d characters (research_incomplete=%s)",
        len(report),
        state.research_incomplete,
    )
    return state
