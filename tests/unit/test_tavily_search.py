"""Unit tests for the Tavily search tool. The SDK call is mocked; no network I/O."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.tools.tavily_search import SearchToolError, search_web


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """search_web retries with real backoff delays in production code
    (1s, 2s, ...) -- neutralize the sleep so tests that trigger retries
    don't actually wait."""
    monkeypatch.setattr("app.core.retry.asyncio.sleep", AsyncMock())


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

    # Tavily's exceptions have no common base, so search_web retries on the
    # general Exception -- all 3 attempts should have run before giving up.
    assert mock_client.search.call_count == 3


@pytest.mark.asyncio
async def test_search_web_retries_and_succeeds_on_third_attempt() -> None:
    raw_response = {"results": [{"title": "Result", "url": "https://a.example.com", "content": "Content"}]}
    mock_client = MagicMock()
    mock_client.search = AsyncMock(side_effect=[RuntimeError("boom"), RuntimeError("boom"), raw_response])

    with patch("app.tools.tavily_search.AsyncTavilyClient", return_value=mock_client):
        results = await search_web("test query")

    assert results == [{"title": "Result", "url": "https://a.example.com", "content": "Content"}]
    assert mock_client.search.call_count == 3


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
