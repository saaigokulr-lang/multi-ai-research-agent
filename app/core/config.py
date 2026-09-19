"""Application configuration.

Settings are loaded once via ``get_settings()`` and cached for the process
lifetime. We fail fast at startup if required environment variables are
missing, rather than letting a misconfiguration surface later as a confusing
runtime error deep inside an API call.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, validated application settings sourced from the environment.

    Using pydantic-settings (rather than raw ``os.environ`` reads) means
    missing or malformed values are caught immediately at construction time,
    with a single clear error listing everything that's wrong.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    OPENROUTER_API_KEY: str
    TAVILY_API_KEY: str
    DATABASE_URL: str
    REDIS_URL: str
    DEFAULT_MODEL: str = Field(default="openai/gpt-4o-mini")
    LOG_LEVEL: str = Field(default="INFO")


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide ``Settings`` instance, parsing it only once.

    ``lru_cache`` gives us a de facto singleton without module-level mutable
    state: every caller gets the same validated instance, and env parsing
    (plus its potential failure) happens exactly once, on first use.
    """
    return Settings()
