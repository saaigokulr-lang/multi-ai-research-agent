"""Reusable async retry helper with exponential backoff.

A single place to retry a flaky external call, rather than each service
(LLM client, search tool, ...) hand-rolling its own retry loop.
"""

import asyncio
import functools
from typing import Awaitable, Callable, ParamSpec, TypeVar

from app.core.logging import get_logger

logger = get_logger(__name__)

_P = ParamSpec("_P")
_T = TypeVar("_T")


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    retryable_exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[_P, Awaitable[_T]]], Callable[_P, Awaitable[_T]]]:
    """Decorate an async function to retry it on ``retryable_exceptions``.

    Uses exponential backoff: delay = base_delay * (2 ** attempt), so with
    the default ``base_delay`` of 1.0, attempts wait ~1s, then ~2s, then
    ~4s before giving up. An exception not in ``retryable_exceptions``
    propagates immediately -- it is never retried. Once ``max_attempts``
    have failed, the last exception is re-raised as-is, never swallowed.
    """

    def decorator(func: Callable[_P, Awaitable[_T]]) -> Callable[_P, Awaitable[_T]]:
        @functools.wraps(func)
        async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            last_exc: BaseException
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as exc:
                    last_exc = exc
                    if attempt == max_attempts - 1:
                        break
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        "%s failed on attempt %d/%d (%s: %s); retrying in %.2fs",
                        func.__name__,
                        attempt + 1,
                        max_attempts,
                        type(exc).__name__,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)
            raise last_exc

        return wrapper

    return decorator
