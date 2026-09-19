"""Web search tool backed by Tavily.

We isolate the Tavily SDK behind ``search_web`` so callers depend on our own
``SearchToolError`` type instead of the SDK's assorted exception classes
(which share no common base), keeping the rest of the app free to swap
search providers later without touching call sites.
"""

from tavily import AsyncTavilyClient

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class SearchToolError(Exception):
    """Raised when the underlying Tavily search call fails.

    Callers only need to handle this one exception type instead of every
    possible error the Tavily SDK might throw (auth, bad request, timeout...).
    """


async def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Search the web for ``query`` via Tavily, returning normalized results.

    Each result dict has at minimum "title", "url", and "content" keys.
    Results are deduplicated by URL. Returns an empty list if Tavily finds
    no results; this is not treated as an error.

    Raises:
        SearchToolError: if the underlying Tavily API call fails.
    """
    settings = get_settings()
    client = AsyncTavilyClient(api_key=settings.TAVILY_API_KEY)

    try:
        response = await client.search(query=query, max_results=max_results)
    except Exception as exc:
        logger.error("Tavily search failed for query=%r: %s", query, exc)
        raise SearchToolError(f"Tavily search failed for query={query!r}: {exc}") from exc

    raw_results = response.get("results", [])

    deduped: dict[str, dict] = {}
    for result in raw_results:
        url = result.get("url", "")
        if url in deduped:
            continue
        deduped[url] = {
            "title": result.get("title", ""),
            "url": url,
            "content": result.get("content", ""),
        }

    return list(deduped.values())
