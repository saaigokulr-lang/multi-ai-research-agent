"""Unit tests for LLMClient. The OpenAI SDK call is mocked; no network I/O."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from openai import APIConnectionError

from app.services.llm_client import LLMCallError, LLMClient, LLMResponse


def _make_completion(model: str = "openai/gpt-4o-mini", content: str = "Hello!") -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


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
    with patch.object(
        client._client.chat.completions,
        "create",
        new=AsyncMock(side_effect=sdk_error),
    ):
        with pytest.raises(LLMCallError):
            await client.generate(prompt="Say hello.")


@pytest.mark.asyncio
async def test_default_model_used_when_none_passed() -> None:
    client = LLMClient()
    mock_create = AsyncMock(return_value=_make_completion())
    with patch.object(client._client.chat.completions, "create", new=mock_create):
        await client.generate(prompt="Say hello.")

    _, kwargs = mock_create.call_args
    assert kwargs["model"] == client._default_model
