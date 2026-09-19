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
async def test_planner_node_accumulates_tokens_into_state() -> None:
    client = LLMClient()
    expected = ResearchPlan(sub_questions=_SUB_QUESTIONS)
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(parsed=expected))],
        usage=SimpleNamespace(prompt_tokens=120, completion_tokens=45),
    )
    state = ResearchState(question="How is AI changing healthcare?")

    with patch("app.agents.planner.LLMClient", return_value=client), patch.object(
        client._client.chat.completions,
        "parse",
        new=AsyncMock(return_value=completion),
    ):
        result = await planner_node(state)

    assert result.total_tokens_used == 120 + 45


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


@pytest.mark.asyncio
async def test_planner_node_reraises_and_records_failure_trace_on_error() -> None:
    """planner_node's except branch: a failed plan generation must still
    propagate (rather than being swallowed) after recording a trace. Uses
    the default run_id=None so record_node_trace's own DB write is a no-op,
    isolating this test to planner_node's error-handling behavior."""
    client = LLMClient()
    state = ResearchState(question="How is AI changing healthcare?")

    with patch("app.agents.planner.LLMClient", return_value=client), patch.object(
        client._client.chat.completions,
        "parse",
        new=AsyncMock(return_value=_make_parsed_completion(None)),
    ):
        with pytest.raises(LLMCallError):
            await planner_node(state)
