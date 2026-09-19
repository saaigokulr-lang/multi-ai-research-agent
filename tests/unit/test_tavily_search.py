"""Unit tests for the Tavily search tool. The SDK call is mocked; no network I/O."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.tools.tavily_search import SearchToolError, search_web


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


def _mock_tavily_client(search_return: dict | None = None, side_effect: Exception | None = None) -> MagicMock:
    mock_client = MagicMock()
    if side_effect is not None:
        mock_client.search = AsyncMock(side_effect=side_effect)
    else:
        mock_client.search = AsyncMock(return_value=search_return)
    return mock_client


@pytest.mark.asyncio
async def test_search_web_returns_expected_list_of_dicts() -> None:
    raw_response = {
        "results": [
            {"title": "Result One", "url": "https://a.example.com", "content": "Content A", "score": 0.9},
            {"title": "Result Two", "url": "https://b.example.com", "content": "Content B", "score": 0.8},
        ]
    }
    mock_client = _mock_tavily_client(search_return=raw_response)

    with patch("app.tools.tavily_search.AsyncTavilyClient", return_value=mock_client):
        results = await search_web("test query")

    assert results == [
        {"title": "Result One", "url": "https://a.example.com", "content": "Content A"},
        {"title": "Result Two", "url": "https://b.example.com", "content": "Content B"},
    ]


@pytest.mark.asyncio
async def test_search_web_empty_results_returns_empty_list() -> None:
    mock_client = _mock_tavily_client(search_return={"results": []})

    with patch("app.tools.tavily_search.AsyncTavilyClient", return_value=mock_client):
        results = await search_web("no results query")

    assert results == []


@pytest.mark.asyncio
async def test_sdk_exception_is_wrapped_as_search_tool_error() -> None:
    mock_client = _mock_tavily_client(side_effect=RuntimeError("boom"))

    with patch("app.tools.tavily_search.AsyncTavilyClient", return_value=mock_client):
        with pytest.raises(SearchToolError):
            await search_web("failing query")


@pytest.mark.asyncio
async def test_duplicate_urls_are_deduplicated() -> None:
    raw_response = {
        "results": [
            {"title": "First", "url": "https://dup.example.com", "content": "First content"},
            {"title": "Duplicate", "url": "https://dup.example.com", "content": "Duplicate content"},
            {"title": "Unique", "url": "https://unique.example.com", "content": "Unique content"},
        ]
    }
    mock_client = _mock_tavily_client(search_return=raw_response)

    with patch("app.tools.tavily_search.AsyncTavilyClient", return_value=mock_client):
        results = await search_web("dedup query")

    assert len(results) == 2
    assert [r["url"] for r in results] == ["https://dup.example.com", "https://unique.example.com"]
