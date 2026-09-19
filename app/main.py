"""Manual smoke test entry point for the LLM client and practice graph.

Run with ``python -m app.main`` to verify settings, OpenRouter connectivity,
and LangGraph wiring are all working before building anything on top.
"""

import asyncio
import sys

from app.core.config import get_settings
from app.core.logging import get_logger
from app.graph.practice_graph import build_practice_graph
from app.graph.practice_state import PracticeState
from app.graph.research_graph import build_research_graph
from app.schemas.research import ResearchState
from app.services.llm_client import LLMClient

logger = get_logger(__name__)

_TEST_QUESTION = "What are the benefits and risks of remote work?"


async def run_research_query(question: str) -> None:
    """Build the research graph and run it end-to-end on ``question``."""
    research_graph = build_research_graph()
    result = await research_graph.ainvoke(ResearchState(question=question))

    final_report = result["final_report"]
    logger.info("research sub_questions=%r", result["sub_questions"])
    logger.info("research findings_count=%d", len(result["findings"]))
    logger.info(
        "research_iterations=%d research_incomplete=%s",
        result["research_iterations"],
        result["research_incomplete"],
    )
    logger.info("final_report length=%d", len(final_report))
    print("\n" + "=" * 80)
    print(final_report)
    print("=" * 80 + "\n")


async def main() -> None:
    """Load settings, ask the user for a research question, then run the full pipeline."""
    get_settings()

    question = " ".join(sys.argv[1:]) or input("Enter your research question: ").strip() or _TEST_QUESTION

    client = LLMClient()

    response = await client.generate(prompt="Say hello in one sentence.")

    logger.info("content=%r", response.content)
    logger.info("model=%s", response.model)
    logger.info("input_tokens=%d", response.input_tokens)
    logger.info("output_tokens=%d", response.output_tokens)
    logger.info("latency_ms=%.2f", response.latency_ms)

    practice_graph = build_practice_graph()
    result = practice_graph.invoke(PracticeState(name="World"))
    logger.info("practice graph message=%r", result["message"])

    await run_research_query(question)


if __name__ == "__main__":
    asyncio.run(main())
