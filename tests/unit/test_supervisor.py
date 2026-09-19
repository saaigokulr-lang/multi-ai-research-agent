"""Unit tests for the supervisor agent. LLM calls are mocked; no network I/O."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.agents.supervisor import SupervisorDecision, evaluate_research, supervisor_node
from app.schemas.research import Finding, ResearchState
from app.services.llm_client import LLMCallError, LLMClient


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


def _make_parsed_completion(parsed: SupervisorDecision | None) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))])


def _make_state(sub_questions: list[str] | None = None, findings: list[Finding] | None = None) -> ResearchState:
    return ResearchState(
        question="How is AI changing healthcare?",
        sub_questions=sub_questions or ["What is the current state?"],
        findings=findings or [],
    )


@pytest.mark.asyncio
async def test_evaluate_research_returns_valid_decision() -> None:
    client = LLMClient()
    expected = SupervisorDecision(research_complete=True, reasoning="All sub-questions are well supported.")
    with patch.object(
        client._client.chat.completions,
        "parse",
        new=AsyncMock(return_value=_make_parsed_completion(expected)),
    ):
        decision = await evaluate_research(_make_state(), client)

    assert isinstance(decision, SupervisorDecision)
    assert decision.research_complete is True
    assert decision.reasoning == "All sub-questions are well supported."


@pytest.mark.asyncio
async def test_evaluate_research_fails_open_on_llm_call_error() -> None:
    client = LLMClient()
    with patch.object(
        client._client.chat.completions,
        "parse",
        new=AsyncMock(side_effect=LLMCallError("boom")),
    ):
        decision = await evaluate_research(_make_state(), client)

    assert decision.research_complete is True
    assert "failed" in decision.reasoning.lower()


@pytest.mark.asyncio
async def test_supervisor_node_increments_iterations_by_one() -> None:
    client = LLMClient()
    decision = SupervisorDecision(research_complete=False, reasoning="Needs more research.")
    state = _make_state()
    assert state.research_iterations == 0

    with patch("app.agents.supervisor.LLMClient", return_value=client), patch(
        "app.agents.supervisor.evaluate_research", new=AsyncMock(return_value=decision)
    ):
        result = await supervisor_node(state)

    assert result.research_iterations == 1


@pytest.mark.asyncio
async def test_supervisor_node_forces_completion_at_iteration_limit() -> None:
    client = LLMClient()
    decision = SupervisorDecision(research_complete=False, reasoning="Still insufficient.")
    state = _make_state()
    state.research_iterations = 1  # next increment reaches MAX_RESEARCH_ITERATIONS (2)

    with patch("app.agents.supervisor.LLMClient", return_value=client), patch(
        "app.agents.supervisor.evaluate_research", new=AsyncMock(return_value=decision)
    ):
        result = await supervisor_node(state)

    assert result.research_iterations == 2
    assert result.research_complete is True
    assert result.research_incomplete is True


@pytest.mark.asyncio
async def test_supervisor_node_does_not_flag_incomplete_when_llm_completes_on_its_own() -> None:
    client = LLMClient()
    decision = SupervisorDecision(research_complete=True, reasoning="Sufficient coverage.")
    state = _make_state()
    state.research_iterations = 1

    with patch("app.agents.supervisor.LLMClient", return_value=client), patch(
        "app.agents.supervisor.evaluate_research", new=AsyncMock(return_value=decision)
    ):
        result = await supervisor_node(state)

    assert result.research_iterations == 2
    assert result.research_complete is True
    assert result.research_incomplete is False
