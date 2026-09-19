"""Unit tests for the writer agent. LLM calls are mocked; no network I/O."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.agents.writer import generate_report, writer_node
from app.schemas.research import Finding, ResearchState
from app.services.llm_client import LLMCallError, LLMClient

_SAMPLE_REPORT = "## Introduction\n\nSome intro.\n\n## Sources\n\n- https://a.example.com"


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


def _make_completion(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        model="openai/gpt-4o-mini",
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


def _make_state(research_incomplete: bool = False) -> ResearchState:
    return ResearchState(
        question="How is AI changing healthcare?",
        sub_questions=["What is the current state?"],
        findings=[
            Finding(
                claim="AI improves diagnostic accuracy.",
                source_url="https://a.example.com",
                snippet="AI improves diagnostic accuracy in radiology.",
                sub_question="What is the current state?",
            )
        ],
        research_incomplete=research_incomplete,
    )


@pytest.mark.asyncio
async def test_generate_report_returns_llm_content_as_is() -> None:
    client = LLMClient()
    with patch.object(
        client._client.chat.completions,
        "create",
        new=AsyncMock(return_value=_make_completion(_SAMPLE_REPORT)),
    ):
        report = await generate_report(_make_state(), client)

    assert report == _SAMPLE_REPORT


@pytest.mark.asyncio
async def test_generate_report_includes_research_incomplete_context() -> None:
    client = LLMClient()
    mock_create = AsyncMock(return_value=_make_completion(_SAMPLE_REPORT))
    with patch.object(client._client.chat.completions, "create", new=mock_create):
        await generate_report(_make_state(research_incomplete=True), client)

    _, kwargs = mock_create.call_args
    user_message = next(m["content"] for m in kwargs["messages"] if m["role"] == "user")
    assert "Research flagged incomplete: True" in user_message


@pytest.mark.asyncio
async def test_generate_report_false_research_incomplete_reflected_in_prompt() -> None:
    client = LLMClient()
    mock_create = AsyncMock(return_value=_make_completion(_SAMPLE_REPORT))
    with patch.object(client._client.chat.completions, "create", new=mock_create):
        await generate_report(_make_state(research_incomplete=False), client)

    _, kwargs = mock_create.call_args
    user_message = next(m["content"] for m in kwargs["messages"] if m["role"] == "user")
    assert "Research flagged incomplete: False" in user_message


@pytest.mark.asyncio
async def test_writer_node_sets_final_report_and_leaves_other_state_untouched() -> None:
    client = LLMClient()
    state = _make_state()

    with patch("app.agents.writer.LLMClient", return_value=client), patch.object(
        client._client.chat.completions,
        "create",
        new=AsyncMock(return_value=_make_completion(_SAMPLE_REPORT)),
    ):
        result = await writer_node(state)

    assert result.final_report == _SAMPLE_REPORT
    assert result.question == "How is AI changing healthcare?"
    assert result.sub_questions == ["What is the current state?"]
    assert len(result.findings) == 1
    assert result.research_incomplete is False


@pytest.mark.asyncio
async def test_generate_report_deduplicates_sources_in_prompt() -> None:
    client = LLMClient()
    state = ResearchState(
        question="How is AI changing healthcare?",
        sub_questions=["What is the current state?", "What are the risks?"],
        findings=[
            Finding(
                claim="AI improves diagnostic accuracy.",
                source_url="https://a.example.com",
                snippet="AI improves diagnostic accuracy in radiology.",
                sub_question="What is the current state?",
            ),
            Finding(
                claim="AI raises data privacy concerns.",
                source_url="https://a.example.com",
                snippet="AI raises data privacy concerns in hospitals.",
                sub_question="What are the risks?",
            ),
            Finding(
                claim="AI adoption varies by region.",
                source_url="https://b.example.com",
                snippet="AI adoption varies by region.",
                sub_question="What are the risks?",
            ),
        ],
    )
    mock_create = AsyncMock(return_value=_make_completion(_SAMPLE_REPORT))
    with patch.object(client._client.chat.completions, "create", new=mock_create):
        await generate_report(state, client)

    _, kwargs = mock_create.call_args
    user_message = next(m["content"] for m in kwargs["messages"] if m["role"] == "user")
    sources_section = user_message.split("Unique sources (use exactly this list for the Sources section):\n")[
        1
    ].split("\n\n")[0]

    assert sources_section.count("https://a.example.com") == 1
    assert sources_section.count("https://b.example.com") == 1


@pytest.mark.asyncio
async def test_generate_report_propagates_llm_call_error() -> None:
    client = LLMClient()
    with patch.object(
        client._client.chat.completions,
        "create",
        new=AsyncMock(side_effect=LLMCallError("boom")),
    ):
        with pytest.raises(LLMCallError):
            await generate_report(_make_state(), client)
