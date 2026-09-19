"""Integration tests for full-graph execution tracing and the token budget cutoff.

Mocks LLMClient.generate/generate_structured directly (same approach as
test_research_graph.py) but has the fakes also bump total_input_tokens/
total_output_tokens the way the real implementation does, so token
accumulation and the budget cutoff actually exercise real node logic.
"""

from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest
from sqlalchemy import select

from app.agents.researcher import SEARCH_QUERY_SYSTEM_PROMPT, ExtractedFinding, ExtractedFindings
from app.agents.supervisor import SupervisorDecision
from app.agents.writer import WRITER_SYSTEM_PROMPT
from app.database.models import ExecutionTrace
from app.graph.research_graph import build_research_graph
from app.schemas.research import ResearchPlan, ResearchState
from app.services.llm_client import LLMClient, LLMResponse
from app.services.run_store import run_store

_QUESTION = "How is AI changing healthcare?"
_SUB_QUESTIONS = ["What is the current state?", "What are the risks?"]
_SEARCH_RESULTS = [{"title": "Result", "url": "https://a.example.com", "content": "Some content."}]

# Every simulated LLM call costs exactly 100 tokens (50 in + 50 out), so the
# token math in the budget test below is easy to reason about.
_INPUT_TOKENS_PER_CALL = 50
_OUTPUT_TOKENS_PER_CALL = 50


def _make_fake_generate_structured(supervisor_decisions: list[SupervisorDecision]):
    decisions_iter = iter(supervisor_decisions)

    async def fake_generate_structured(
        self, prompt, response_model, system_prompt=None, model=None, temperature=0.7
    ):
        self.total_input_tokens += _INPUT_TOKENS_PER_CALL
        self.total_output_tokens += _OUTPUT_TOKENS_PER_CALL
        if response_model is ResearchPlan:
            return ResearchPlan(sub_questions=_SUB_QUESTIONS)
        if response_model is ExtractedFindings:
            return ExtractedFindings(
                findings=[
                    ExtractedFinding(
                        claim="A supported claim.", source_url="https://a.example.com", snippet="Some content."
                    )
                ]
            )
        if response_model is SupervisorDecision:
            return next(decisions_iter)
        raise AssertionError(f"Unexpected response_model: {response_model}")

    return fake_generate_structured


async def _fake_generate(self, prompt, system_prompt=None, model=None, temperature=0.7):
    self.total_input_tokens += _INPUT_TOKENS_PER_CALL
    self.total_output_tokens += _OUTPUT_TOKENS_PER_CALL
    if system_prompt == SEARCH_QUERY_SYSTEM_PROMPT:
        content = "mock search query"
    elif system_prompt == WRITER_SYSTEM_PROMPT:
        content = "## Introduction\n\nReport body.\n\n## Sources\n\n- https://a.example.com"
    else:
        raise AssertionError(f"Unexpected system_prompt: {system_prompt!r}")
    return LLMResponse(
        content=content,
        model="test-model",
        input_tokens=_INPUT_TOKENS_PER_CALL,
        output_tokens=_OUTPUT_TOKENS_PER_CALL,
        latency_ms=1.0,
    )


async def _traces_for(db_session, run_id: str) -> list[ExecutionTrace]:
    result = await db_session.scalars(
        select(ExecutionTrace).where(ExecutionTrace.run_id == UUID(run_id)).order_by(ExecutionTrace.started_at)
    )
    return list(result.all())


@pytest.mark.asyncio
async def test_full_graph_run_produces_expected_execution_traces(db_session) -> None:
    run_id = await run_store.create_run(db_session, _QUESTION)
    supervisor_decisions = [SupervisorDecision(research_complete=True, reasoning="Sufficient.")]

    with patch.object(LLMClient, "generate", new=_fake_generate), patch.object(
        LLMClient, "generate_structured", new=_make_fake_generate_structured(supervisor_decisions)
    ), patch("app.agents.researcher.search_web", new=AsyncMock(return_value=_SEARCH_RESULTS)):
        graph = build_research_graph()
        result = await graph.ainvoke(ResearchState(question=_QUESTION, run_id=run_id))

    assert result["final_report"] != ""

    traces = await _traces_for(db_session, run_id)
    assert [t.node_name for t in traces] == ["planner", "researcher", "supervisor", "writer"]
    assert all(t.status == "success" for t in traces)
    assert all((t.input_tokens + t.output_tokens) > 0 for t in traces)


@pytest.mark.asyncio
async def test_loop_produces_one_trace_per_node_per_iteration(db_session) -> None:
    run_id = await run_store.create_run(db_session, _QUESTION)
    supervisor_decisions = [
        SupervisorDecision(research_complete=False, reasoning="Not yet."),
        SupervisorDecision(research_complete=False, reasoning="Still not enough."),
    ]

    with patch.object(LLMClient, "generate", new=_fake_generate), patch.object(
        LLMClient, "generate_structured", new=_make_fake_generate_structured(supervisor_decisions)
    ), patch("app.agents.researcher.search_web", new=AsyncMock(return_value=_SEARCH_RESULTS)):
        graph = build_research_graph()
        result = await graph.ainvoke(ResearchState(question=_QUESTION, run_id=run_id))

    assert result["research_iterations"] == 2

    traces = await _traces_for(db_session, run_id)
    # 1 planner + (researcher, supervisor) x 2 loop passes + 1 writer.
    assert [t.node_name for t in traces] == [
        "planner",
        "researcher",
        "supervisor",
        "researcher",
        "supervisor",
        "writer",
    ]


@pytest.mark.asyncio
async def test_budget_exceeded_still_completes_with_report(db_session, monkeypatch) -> None:
    run_id = await run_store.create_run(db_session, _QUESTION)
    # planner costs 100 tokens; each researcher sub-question costs another
    # 200 (search-query call + extraction call). A budget of 250 lets the
    # first sub-question through but cuts off before the second, so this
    # exercises stopping partway through the loop -- not just never starting.
    monkeypatch.setattr("app.agents.researcher.MAX_TOKEN_BUDGET", 250)
    monkeypatch.setattr("app.agents.writer.MAX_TOKEN_BUDGET", 250)

    supervisor_decisions = [SupervisorDecision(research_complete=True, reasoning="Sufficient.")]

    with patch.object(LLMClient, "generate", new=_fake_generate), patch.object(
        LLMClient, "generate_structured", new=_make_fake_generate_structured(supervisor_decisions)
    ), patch("app.agents.researcher.search_web", new=AsyncMock(return_value=_SEARCH_RESULTS)):
        graph = build_research_graph()
        result = await graph.ainvoke(ResearchState(question=_QUESTION, run_id=run_id))

    assert result["budget_exceeded"] is True
    assert len(result["findings"]) == 1  # only the first sub-question got processed
    assert result["final_report"] != ""
