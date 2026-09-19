"""One true end-to-end sanity test: hits the real POST /research endpoint
(unlike test_api.py, which mocks app.api.routes._invoke_graph out entirely)
so the real graph, the real _invoke_graph/_execute_research_run wiring, and
the real database layer all run together. Only the network boundary --
LLMClient.generate/generate_structured and search_web -- is mocked, using
the same fake pattern as test_tracing.py. This is meant to catch
integration issues between components that fully-mocked unit tests miss,
even with the network boundary itself mocked out.
"""

from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.agents.researcher import SEARCH_QUERY_SYSTEM_PROMPT, ExtractedFinding, ExtractedFindings
from app.agents.supervisor import SupervisorDecision
from app.agents.writer import WRITER_SYSTEM_PROMPT
from app.database.models import ExecutionTrace, Report, ResearchRun
from app.main import app
from app.schemas.research import ResearchPlan
from app.services.llm_client import LLMClient, LLMResponse

client = TestClient(app)

_QUESTION = "How is AI changing healthcare?"
_SUB_QUESTIONS = ["What is the current state?", "What are the risks?"]
_SEARCH_RESULTS = [{"title": "Result", "url": "https://a.example.com", "content": "Some content."}]


async def _fake_generate_structured(
    self, prompt, response_model, system_prompt=None, model=None, temperature=0.7
):
    self.total_input_tokens += 50
    self.total_output_tokens += 50
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
        return SupervisorDecision(research_complete=True, reasoning="Sufficient.")
    raise AssertionError(f"Unexpected response_model: {response_model}")


async def _fake_generate(self, prompt, system_prompt=None, model=None, temperature=0.7):
    self.total_input_tokens += 50
    self.total_output_tokens += 50
    if system_prompt == SEARCH_QUERY_SYSTEM_PROMPT:
        content = "mock search query"
    elif system_prompt == WRITER_SYSTEM_PROMPT:
        content = "## Introduction\n\nReport body.\n\n## Sources\n\n- https://a.example.com"
    else:
        raise AssertionError(f"Unexpected system_prompt: {system_prompt!r}")
    return LLMResponse(content=content, model="test-model", input_tokens=50, output_tokens=50, latency_ms=1.0)


@pytest.mark.asyncio
async def test_full_research_run_via_real_api_completes_with_traces_and_persisted_results(db_session) -> None:
    """Posts a real request through the FastAPI app (real _invoke_graph, real
    graph, real database session plumbing) and checks every layer landed
    consistently: HTTP status endpoint, execution_traces rows, and the
    underlying ResearchRun/Report rows -- not just the API's own view of them."""
    with patch.object(LLMClient, "generate", new=_fake_generate), patch.object(
        LLMClient, "generate_structured", new=_fake_generate_structured
    ), patch("app.agents.researcher.search_web", new=AsyncMock(return_value=_SEARCH_RESULTS)):
        create_response = client.post("/research", json={"question": _QUESTION})
        assert create_response.status_code == 202
        run_id = create_response.json()["run_id"]

        # TestClient runs BackgroundTasks synchronously before returning the
        # response above, so the run is already finished by this point.
        status_response = client.get(f"/research/{run_id}")
        trace_response = client.get(f"/research/{run_id}/trace")

    assert status_response.status_code == 200
    status_body = status_response.json()
    assert status_body["status"] == "completed"
    assert status_body["sub_questions"] == _SUB_QUESTIONS
    assert status_body["final_report"] != ""

    assert trace_response.status_code == 200
    trace_body = trace_response.json()
    assert [step["node_name"] for step in trace_body["steps"]] == [
        "planner",
        "researcher",
        "supervisor",
        "writer",
    ]
    assert all(step["status"] == "success" for step in trace_body["steps"])
    assert trace_body["total_tokens_used"] > 0
    assert trace_body["budget_exceeded"] is False

    # Cross-check the API's view against the underlying rows directly, so
    # this test would still catch a bug where the API and DB disagree.
    run_uuid = UUID(run_id)
    run_row = await db_session.get(ResearchRun, run_uuid)
    assert run_row.status == "completed"
    assert run_row.total_tokens_used == trace_body["total_tokens_used"]
    assert run_row.budget_exceeded is False

    report_row = await db_session.scalar(select(Report).where(Report.run_id == run_uuid))
    assert report_row is not None
    assert report_row.content == status_body["final_report"]

    trace_row_count = await db_session.scalar(
        select(func.count()).select_from(ExecutionTrace).where(ExecutionTrace.run_id == run_uuid)
    )
    assert trace_row_count == 4
