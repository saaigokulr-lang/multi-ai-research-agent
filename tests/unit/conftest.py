"""Shared fixtures for unit tests.

Every unit test file needs the same env vars stubbed (Settings() requires
them) plus GROQ_API_KEY neutralized -- previously duplicated verbatim across
seven files; centralized here since it's identical everywhere it's used.
"""

import pytest


@pytest.fixture(autouse=True)
def _settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/test")
    monkeypatch.setenv("REDIS_URL", "redis://localhost")
    # Settings reads .env directly, and the real .env has a real GROQ_API_KEY
    # configured -- an empty string here outranks it (still falsy) so
    # LLMClient() only constructs one real AsyncOpenAI client, not two.
    monkeypatch.setenv("GROQ_API_KEY", "")
    from app.core.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
