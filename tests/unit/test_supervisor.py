"""Unit tests for the supervisor agent. LLM calls are mocked; no network I/O."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.agents.supervisor import SupervisorDecision, evaluate_research, supervisor_node
from app.schemas.research import Finding, ResearchState
from app.services.llm_client import LLMCallError, LLMClient


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
async def test_evaluate_research_prompt_is_compact_and_grouped_by_sub_question() -> None:
    """The Supervisor's prompt must stay compact as findings accumulate --
    this is the fix for a real failure where full claim/source_url/snippet
    detail per finding grew the prompt past a provider's token-per-minute
    limit. Full detail is still fine in writer.py's prompt, which needs it
    for citations; the Supervisor only needs enough to judge sufficiency."""
    client = LLMClient()
    long_claim_1 = "A" * 150
    long_claim_2 = "B" * 150
    state = _make_state(
        sub_questions=["What is the current state?", "What are the risks?"],
        findings=[
            Finding(
                claim=long_claim_1,
                source_url="https://example.com/unique-source-one",
                snippet="Full detailed snippet text that should never reach the supervisor prompt.",
                sub_question="What is the current state?",
            ),
            Finding(
                claim="A second, shorter supporting claim.",
                source_url="https://example.com/unique-source-two",
                snippet="Another full snippet.",
                sub_question="What is the current state?",
            ),
            Finding(
                claim=long_claim_2,
                source_url="https://example.com/unique-source-three",
                snippet="Yet another snippet.",
                sub_question="What are the risks?",
            ),
        ],
    )
    expected = SupervisorDecision(research_complete=True, reasoning="Sufficient.")
    mock_parse = AsyncMock(return_value=_make_parsed_completion(expected))

    with patch.object(client._client.chat.completions, "parse", new=mock_parse):
        await evaluate_research(state, client)

    _, kwargs = mock_parse.call_args
    user_message = next(m["content"] for m in kwargs["messages"] if m["role"] == "user")

    # Sub-question groupings and per-sub-question finding counts are present.
    assert "What is the current state?" in user_message
    assert "What are the risks?" in user_message
    assert "2 findings" in user_message
    assert "1 finding)" in user_message

    # Claims are truncated, not sent in full.
    assert long_claim_1 not in user_message
    assert long_claim_2 not in user_message
    assert long_claim_1[:80] in user_message

    # Source URLs and full snippets never reach the supervisor's prompt --
    # that detail is the Writer's concern, not the Supervisor's.
    assert "https://example.com/unique-source-one" not in user_message
    assert "https://example.com/unique-source-two" not in user_message
    assert "https://example.com/unique-source-three" not in user_message
    assert "Full detailed snippet text" not in user_message


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


@pytest.mark.asyncio
async def test_supervisor_node_reraises_and_records_failure_trace_on_error() -> None:
    """supervisor_node's own except branch only sees exceptions
    evaluate_research doesn't already fail open on (it only catches
    LLMCallError) -- an unexpected error must still propagate after
    recording a trace. Uses the default run_id=None so record_node_trace's
    own DB write is a no-op."""
    client = LLMClient()
    state = _make_state()

    with patch("app.agents.supervisor.LLMClient", return_value=client), patch(
        "app.agents.supervisor.evaluate_research", new=AsyncMock(side_effect=RuntimeError("boom"))
    ):
        with pytest.raises(RuntimeError):
            await supervisor_node(state)
