"""Unit tests for the end-to-end research graph.

Mocks at the LLM/search boundary (LLMClient.generate / generate_structured,
search_web) rather than mocking whole node functions, so these tests
actually exercise the real graph wiring and routing logic.
"""

from unittest.mock import AsyncMock, patch

import pytest

from app.agents.researcher import SEARCH_QUERY_SYSTEM_PROMPT, ExtractedFinding, ExtractedFindings
from app.agents.supervisor import SupervisorDecision
from app.agents.writer import WRITER_SYSTEM_PROMPT
from app.graph.research_graph import build_research_graph
from app.schemas.research import MAX_RESEARCH_ITERATIONS, ResearchPlan, ResearchState
from app.services.llm_client import LLMClient, LLMResponse

_SUB_QUESTIONS = ["What is the current state?", "What are the risks?"]

_SEARCH_RESULTS = [{"title": "Result", "url": "https://a.example.com", "content": "Some content."}]


@pytest.fixture(autouse=True)
def _settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/test")
    monkeypatch.setenv("REDIS_URL", "redis://localhost")
    from app.core.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _make_fake_generate_structured(supervisor_decisions: list[SupervisorDecision]):
    """Route generate_structured to a canned response based on response_model."""
    decisions_iter = iter(supervisor_decisions)

    async def fake_generate_structured(
        self, prompt, response_model, system_prompt=None, model=None, temperature=0.7
    ):
        if response_model is ResearchPlan:
            return ResearchPlan(sub_questions=_SUB_QUESTIONS)
        if response_model is ExtractedFindings:
            return ExtractedFindings(
                findings=[
                    ExtractedFinding(
                        claim="A supported claim.",
                        source_url="https://a.example.com",
                        snippet="Some content.",
                    )
                ]
            )
        if response_model is SupervisorDecision:
            return next(decisions_iter)
        raise AssertionError(f"Unexpected response_model: {response_model}")

    return fake_generate_structured


async def _fake_generate(self, prompt, system_prompt=None, model=None, temperature=0.7):
    """Route generate() to a canned response based on system_prompt."""
    if system_prompt == SEARCH_QUERY_SYSTEM_PROMPT:
        content = "mock search query"
    elif system_prompt == WRITER_SYSTEM_PROMPT:
        content = "## Introduction\n\nReport body.\n\n## Sources\n\n- https://a.example.com"
    else:
        raise AssertionError(f"Unexpected system_prompt: {system_prompt!r}")
    return LLMResponse(content=content, model="test-model", input_tokens=1, output_tokens=1, latency_ms=1.0)


@pytest.mark.asyncio
async def test_happy_path_completes_without_looping() -> None:
    supervisor_decisions = [SupervisorDecision(research_complete=True, reasoning="Sufficient.")]

    with patch.object(LLMClient, "generate", new=_fake_generate), patch.object(
        LLMClient, "generate_structured", new=_make_fake_generate_structured(supervisor_decisions)
    ), patch("app.agents.researcher.search_web", new=AsyncMock(return_value=_SEARCH_RESULTS)) as mock_search_web:
        graph = build_research_graph()
        result = await graph.ainvoke(ResearchState(question="How is AI changing healthcare?"))

    assert result["sub_questions"] == _SUB_QUESTIONS
    assert result["research_iterations"] == 1
    assert result["research_complete"] is True
    assert result["research_incomplete"] is False
    assert len(result["findings"]) == len(_SUB_QUESTIONS)
    assert result["final_report"] != ""
    assert mock_search_web.call_count == len(_SUB_QUESTIONS)


@pytest.mark.asyncio
async def test_loop_then_stop_at_iteration_limit() -> None:
    supervisor_decisions = [
        SupervisorDecision(research_complete=False, reasoning="Not yet."),
        SupervisorDecision(research_complete=False, reasoning="Still not enough."),
    ]

    with patch.object(LLMClient, "generate", new=_fake_generate), patch.object(
        LLMClient, "generate_structured", new=_make_fake_generate_structured(supervisor_decisions)
    ), patch("app.agents.researcher.search_web", new=AsyncMock(return_value=_SEARCH_RESULTS)) as mock_search_web:
        graph = build_research_graph()
        result = await graph.ainvoke(ResearchState(question="How is AI changing healthcare?"))

    assert result["research_iterations"] == MAX_RESEARCH_ITERATIONS
    assert result["research_complete"] is True
    assert result["research_incomplete"] is True
    assert result["final_report"] != ""
    # Researcher ran once per supervisor pass (2 passes x 2 sub-questions each).
    assert mock_search_web.call_count == MAX_RESEARCH_ITERATIONS * len(_SUB_QUESTIONS)
