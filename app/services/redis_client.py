"""Redis client factory.

Isolated the same way ``LLMClient`` isolates the OpenAI SDK, so the rest of
the app depends on one reusable client rather than each caller constructing
its own.
"""

from functools import lru_cache

from redis.asyncio import Redis

from app.core.config import get_settings


@lru_cache
def get_redis_client() -> Redis:
    """Return the process-wide Redis client, constructed once.

    Built directly from REDIS_URL (rather than parsed host/port) so the
    driver handles scheme-specific details -- like Upstash's "rediss://"
    TLS scheme -- on its own.
    """
    settings = get_settings()
    return Redis.from_url(settings.REDIS_URL, decode_responses=True)
