"""Unit tests for the planner agent. LLM calls are mocked; no network I/O."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.agents.planner import generate_research_plan, planner_node
from app.schemas.research import ResearchPlan, ResearchState
from app.services.llm_client import LLMCallError, LLMClient

_SUB_QUESTIONS = [
    "What is the current state of the technology?",
    "What are the key benefits?",
    "What are the main risks and challenges?",
    "What future trends are expected?",
]


def _make_parsed_completion(parsed: ResearchPlan | None) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))])


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


@pytest.mark.asyncio
async def test_generate_research_plan_returns_valid_plan() -> None:
    client = LLMClient()
    expected = ResearchPlan(sub_questions=_SUB_QUESTIONS)
    with patch.object(
        client._client.chat.completions,
        "parse",
        new=AsyncMock(return_value=_make_parsed_completion(expected)),
    ):
        plan = await generate_research_plan("How is AI changing healthcare?", client)

    assert isinstance(plan, ResearchPlan)
    assert plan.sub_questions == _SUB_QUESTIONS


@pytest.mark.asyncio
async def test_planner_node_updates_sub_questions_only() -> None:
    client = LLMClient()
    expected = ResearchPlan(sub_questions=_SUB_QUESTIONS)
    state = ResearchState(question="How is AI changing healthcare?")

    with patch("app.agents.planner.LLMClient", return_value=client), patch.object(
        client._client.chat.completions,
        "parse",
        new=AsyncMock(return_value=_make_parsed_completion(expected)),
    ):
        result = await planner_node(state)

    assert result.sub_questions == _SUB_QUESTIONS
    assert result.question == "How is AI changing healthcare?"
    assert result.findings == []
    assert result.research_complete is False
    assert result.final_report == ""


@pytest.mark.asyncio
async def test_generate_research_plan_raises_clear_error_on_unparseable_response() -> None:
    client = LLMClient()
    with patch.object(
        client._client.chat.completions,
        "parse",
        new=AsyncMock(return_value=_make_parsed_completion(None)),
    ):
        with pytest.raises(LLMCallError):
            await generate_research_plan("How is AI changing healthcare?", client)
