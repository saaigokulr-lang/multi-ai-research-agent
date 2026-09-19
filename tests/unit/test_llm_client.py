"""Unit tests for LLMClient. The OpenAI SDK call is mocked; no network I/O."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from openai import APIConnectionError, RateLimitError

from app.services.llm_client import FALLBACK_MODEL, LLMCallError, LLMClient, LLMResponse


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retries on APIConnectionError use real backoff delays in production
    code (1s, 2s, ...) -- neutralize the sleep so tests that trigger retries
    don't actually wait."""
    monkeypatch.setattr("app.core.retry.asyncio.sleep", AsyncMock())


def _make_completion(model: str = "openai/gpt-4o-mini", content: str = "Hello!") -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


def _make_rate_limit_error() -> RateLimitError:
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(status_code=429, request=request)
    return RateLimitError("Rate limit exceeded", response=response, body=None)


@pytest.mark.asyncio
async def test_generate_returns_populated_llm_response() -> None:
    client = LLMClient()
    with patch.object(
        client._client.chat.completions,
        "create",
        new=AsyncMock(return_value=_make_completion()),
    ):
        result = await client.generate(prompt="Say hello.")

    assert isinstance(result, LLMResponse)
    assert result.content == "Hello!"
    assert result.model == "openai/gpt-4o-mini"
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert result.latency_ms >= 0


@pytest.mark.asyncio
async def test_sdk_exception_is_wrapped_as_llm_call_error() -> None:
    client = LLMClient()
    sdk_error = APIConnectionError(request=SimpleNamespace())
    mock_create = AsyncMock(side_effect=sdk_error)
    with patch.object(client._client.chat.completions, "create", new=mock_create):
        with pytest.raises(LLMCallError):
            await client.generate(prompt="Say hello.")

    # APIConnectionError is retryable: all 3 attempts should have run before
    # LLMCallError was ultimately raised.
    assert mock_create.call_count == 3


@pytest.mark.asyncio
async def test_retries_primary_call_and_succeeds_on_third_attempt() -> None:
    client = LLMClient()
    sdk_error = APIConnectionError(request=SimpleNamespace())
    mock_create = AsyncMock(side_effect=[sdk_error, sdk_error, _make_completion(content="Recovered")])

    with patch.object(client._client.chat.completions, "create", new=mock_create):
        result = await client.generate(prompt="Say hello.")

    assert result.content == "Recovered"
    assert mock_create.call_count == 3


@pytest.mark.asyncio
async def test_default_model_used_when_none_passed() -> None:
    client = LLMClient()
    mock_create = AsyncMock(return_value=_make_completion())
    with patch.object(client._client.chat.completions, "create", new=mock_create):
        await client.generate(prompt="Say hello.")

    _, kwargs = mock_create.call_args
    assert kwargs["model"] == client._default_model


def _client_with_groq_configured(monkeypatch: pytest.MonkeyPatch) -> LLMClient:
    from app.core.config import get_settings

    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    get_settings.cache_clear()
    client = LLMClient()
    assert client._fallback_client is not None
    return client


@pytest.mark.asyncio
async def test_rate_limit_falls_back_to_groq_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client_with_groq_configured(monkeypatch)
    mock_fallback_create = AsyncMock(return_value=_make_completion(model=FALLBACK_MODEL, content="From Groq"))

    with patch.object(
        client._client.chat.completions, "create", new=AsyncMock(side_effect=_make_rate_limit_error())
    ), patch.object(client._fallback_client.chat.completions, "create", new=mock_fallback_create):
        result = await client.generate(prompt="Say hello.")

    assert result.content == "From Groq"
    _, kwargs = mock_fallback_create.call_args
    assert kwargs["model"] == FALLBACK_MODEL


@pytest.mark.asyncio
async def test_rate_limit_without_fallback_configured_raises_llm_call_error() -> None:
    client = LLMClient()
    assert client._fallback_client is None

    with patch.object(
        client._client.chat.completions, "create", new=AsyncMock(side_effect=_make_rate_limit_error())
    ):
        with pytest.raises(LLMCallError):
            await client.generate(prompt="Say hello.")


@pytest.mark.asyncio
async def test_non_rate_limit_error_does_not_trigger_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client_with_groq_configured(monkeypatch)
    sdk_error = APIConnectionError(request=SimpleNamespace())
    mock_fallback_create = AsyncMock(return_value=_make_completion(model=FALLBACK_MODEL))

    with patch.object(
        client._client.chat.completions, "create", new=AsyncMock(side_effect=sdk_error)
    ), patch.object(client._fallback_client.chat.completions, "create", new=mock_fallback_create):
        with pytest.raises(LLMCallError):
            await client.generate(prompt="Say hello.")

    mock_fallback_create.assert_not_called()


@pytest.mark.asyncio
async def test_both_primary_and_fallback_failures_raise_llm_call_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client_with_groq_configured(monkeypatch)
    fallback_error = APIConnectionError(request=SimpleNamespace())

    with patch.object(
        client._client.chat.completions, "create", new=AsyncMock(side_effect=_make_rate_limit_error())
    ), patch.object(
        client._fallback_client.chat.completions, "create", new=AsyncMock(side_effect=fallback_error)
    ):
        with pytest.raises(LLMCallError):
            await client.generate(prompt="Say hello.")
