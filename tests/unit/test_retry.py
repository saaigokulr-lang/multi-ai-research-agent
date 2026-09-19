"""Unit tests for the retry helper.

Uses a tiny base_delay (0.01s) throughout so the suite stays fast; only
attempt counts and the final raised/returned value are asserted, never
actual timing.
"""

import pytest

from app.core.retry import with_retry


class _RetryableError(Exception):
    pass


class _OtherError(Exception):
    pass


@pytest.mark.asyncio
async def test_retries_exact_number_of_times_then_raises_last_exception() -> None:
    call_count = 0

    @with_retry(max_attempts=3, base_delay=0.01, retryable_exceptions=(_RetryableError,))
    async def always_fails() -> None:
        nonlocal call_count
        call_count += 1
        raise _RetryableError(f"boom {call_count}")

    with pytest.raises(_RetryableError, match="boom 3"):
        await always_fails()

    assert call_count == 3


@pytest.mark.asyncio
async def test_succeeds_on_a_later_attempt_without_exhausting_retries() -> None:
    call_count = 0

    @with_retry(max_attempts=3, base_delay=0.01, retryable_exceptions=(_RetryableError,))
    async def fails_twice_then_succeeds() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise _RetryableError("boom")
        return "success"

    result = await fails_twice_then_succeeds()

    assert result == "success"
    assert call_count == 3


@pytest.mark.asyncio
async def test_non_matching_exception_is_not_retried() -> None:
    call_count = 0

    @with_retry(max_attempts=3, base_delay=0.01, retryable_exceptions=(_RetryableError,))
    async def raises_other_error() -> None:
        nonlocal call_count
        call_count += 1
        raise _OtherError("nope")

    with pytest.raises(_OtherError):
        await raises_other_error()

    assert call_count == 1
