"""Unit tests for the Redis client factory. No real Redis connection is
made -- constructing a redis-py client from a URL doesn't connect eagerly."""

from redis.asyncio import Redis

from app.services.redis_client import get_redis_client


def test_get_redis_client_returns_redis_instance_built_from_settings() -> None:
    get_redis_client.cache_clear()
    try:
        client = get_redis_client()
        assert isinstance(client, Redis)
        assert get_redis_client() is client  # lru_cache: same instance every call
    finally:
        get_redis_client.cache_clear()
