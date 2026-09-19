"""Unit tests for the researcher agent. LLM calls and search_web are mocked;
no network I/O."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.agents.researcher import ExtractedFinding, ExtractedFindings, research_sub_question, researcher_node
from app.schemas.research import Finding, ResearchState
from app.services.llm_client import LLMClient
from app.tools.tavily_search import SearchToolError

_SAMPLE_SEARCH_RESULTS = [
    {"title": "Result One", "url": "https://a.example.com", "content": "Content A"},
    {"title": "Result Two", "url": "https://b.example.com", "content": "Content B"},
]


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


def _make_parsed_completion(parsed: ExtractedFindings | None) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))])


@pytest.mark.asyncio
async def test_research_sub_question_fills_sub_question_and_retrieved_at() -> None:
    client = LLMClient()
    extracted = ExtractedFindings(
        findings=[
            ExtractedFinding(claim="Claim A", source_url="https://a.example.com", snippet="Content A"),
        ]
    )

    with patch(
        "app.agents.researcher.generate_search_query", new=AsyncMock(return_value="test query")
    ), patch(
        "app.agents.researcher.search_web", new=AsyncMock(return_value=_SAMPLE_SEARCH_RESULTS)
    ), patch.object(
        client._client.chat.completions,
        "parse",
        new=AsyncMock(return_value=_make_parsed_completion(extracted)),
    ):
        findings = await research_sub_question("What is X?", client)

    assert len(findings) == 1
    finding = findings[0]
    assert isinstance(finding, Finding)
    assert finding.claim == "Claim A"
    assert finding.source_url == "https://a.example.com"
    assert finding.sub_question == "What is X?"
    assert isinstance(finding.retrieved_at, datetime)


@pytest.mark.asyncio
async def test_research_sub_question_returns_empty_list_and_skips_llm_when_no_results() -> None:
    client = LLMClient()
    mock_generate_structured = AsyncMock()

    with patch(
        "app.agents.researcher.generate_search_query", new=AsyncMock(return_value="test query")
    ), patch(
        "app.agents.researcher.search_web", new=AsyncMock(return_value=[])
    ), patch.object(
        client, "generate_structured", new=mock_generate_structured
    ):
        findings = await research_sub_question("What is X?", client)

    assert findings == []
    mock_generate_structured.assert_not_called()


@pytest.mark.asyncio
async def test_research_sub_question_returns_empty_list_on_search_tool_error() -> None:
    client = LLMClient()

    with patch(
        "app.agents.researcher.generate_search_query", new=AsyncMock(return_value="test query")
    ), patch(
        "app.agents.researcher.search_web", new=AsyncMock(side_effect=SearchToolError("boom"))
    ):
        findings = await research_sub_question("What is X?", client)

    assert findings == []


@pytest.mark.asyncio
async def test_researcher_node_processes_sub_questions_sequentially_and_accumulates_findings() -> None:
    state = ResearchState(question="Big question", sub_questions=["Sub Q1", "Sub Q2"])

    finding_1 = Finding(claim="Claim 1", source_url="https://a.example.com", snippet="A", sub_question="Sub Q1")
    finding_2 = Finding(claim="Claim 2", source_url="https://b.example.com", snippet="B", sub_question="Sub Q2")
    mock_research_sub_question = AsyncMock(side_effect=[[finding_1], [finding_2]])

    with patch("app.agents.researcher.LLMClient", return_value=LLMClient()), patch(
        "app.agents.researcher.research_sub_question", new=mock_research_sub_question
    ):
        result = await researcher_node(state)

    assert result.findings == [finding_1, finding_2]
    assert mock_research_sub_question.call_count == 2
    called_sub_questions = [call.args[0] for call in mock_research_sub_question.call_args_list]
    assert called_sub_questions == ["Sub Q1", "Sub Q2"]
