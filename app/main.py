"""Manual smoke test entry point for the LLM client and practice graph.

Run with ``python -m app.main`` to verify settings, OpenRouter connectivity,
and LangGraph wiring are all working before building anything on top.
"""

import asyncio

from app.core.config import get_settings
from app.core.logging import get_logger
from app.graph.practice_graph import build_practice_graph
from app.graph.practice_state import PracticeState
from app.services.llm_client import LLMClient

logger = get_logger(__name__)


async def main() -> None:
    """Load settings, call the LLM once, and run the practice graph."""
    get_settings()
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


if __name__ == "__main__":
    asyncio.run(main())
