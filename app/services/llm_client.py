"""LLM client for OpenRouter's OpenAI-compatible chat completions endpoint.

We isolate the OpenAI SDK behind ``LLMClient`` so callers depend on our own
``LLMResponse``/``LLMCallError`` types instead of the SDK's, keeping the rest
of the app free to swap providers later without touching call sites.
"""

import time
from typing import Any, Optional, TypeVar

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, OpenAIError, RateLimitError
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.retry import with_retry

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Bounds how long a single request to either provider can hang.
_LLM_TIMEOUT_SECONDS = 30.0

# Network/timeout hiccups are worth retrying; auth errors, bad requests, and
# rate limits are not (rate limits go through the Groq fallback path instead).
_RETRYABLE_LLM_EXCEPTIONS = (APIConnectionError, APITimeoutError)

# Models differ between providers, so a rate-limited OpenRouter model can't
# just be replayed as-is against Groq -- this is the equivalent model used
# for every fallback call, regardless of what was originally requested.
# Chosen for explicit "structured_outputs" support, since generate_structured()
# (planner/researcher/supervisor) depends on it.
FALLBACK_MODEL = "openai/gpt-oss-120b"

logger = get_logger(__name__)

_ResponseModelT = TypeVar("_ResponseModelT", bound=BaseModel)


class LLMResponse(BaseModel):
    """Normalized result of a single LLM generation call."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class LLMCallError(Exception):
    """Raised when the underlying LLM API call fails.

    Callers only need to handle this one exception type instead of every
    possible error the OpenAI SDK might throw (network, auth, rate limit...).
    """


class LLMClient:
    """Thin async wrapper around OpenRouter's chat completions API.

    Falls back to Groq for a single retry when OpenRouter specifically
    returns a rate-limit error, if ``GROQ_API_KEY`` is configured.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._default_model = settings.DEFAULT_MODEL
        self._client = AsyncOpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url=_OPENROUTER_BASE_URL,
            timeout=_LLM_TIMEOUT_SECONDS,
        )
        self._fallback_client = (
            AsyncOpenAI(
                api_key=settings.GROQ_API_KEY,
                base_url=_GROQ_BASE_URL,
                timeout=_LLM_TIMEOUT_SECONDS,
            )
            if settings.GROQ_API_KEY
            else None
        )
        # Cumulative usage across every call this instance has made. Callers
        # that create a fresh LLMClient per unit of work (every agent node
        # does) can read these once they're done to get that work's total
        # token cost -- generate_structured() returns just the parsed model,
        # not a token-carrying wrapper, so this is the one place both call
        # shapes' usage is available uniformly.
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    @staticmethod
    def _build_messages(
        prompt: str, system_prompt: Optional[str]
    ) -> list[ChatCompletionMessageParam]:
        messages: list[ChatCompletionMessageParam] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _accumulate_usage(self, completion: Any) -> None:
        usage = getattr(completion, "usage", None)
        if usage is not None:
            self.total_input_tokens += usage.prompt_tokens or 0
            self.total_output_tokens += usage.completion_tokens or 0

    async def _create_completion(
        self, method_name: str, resolved_model: str, call_kwargs: dict[str, Any], error_label: str
    ) -> Any:
        """Call ``method_name`` on the primary client, retrying against Groq
        (once, with ``FALLBACK_MODEL``) if OpenRouter returns a rate-limit
        error and a fallback client is configured.

        Network/timeout errors are retried (with backoff) against whichever
        provider raised them; retries wrap around the existing fallback and
        error-wrapping logic below, they don't replace it.

        Raises:
            LLMCallError: if the primary call fails for any other reason, or
                if the Groq fallback isn't configured or also fails.
        """

        @with_retry(max_attempts=3, base_delay=1.0, retryable_exceptions=_RETRYABLE_LLM_EXCEPTIONS)
        async def _call_primary() -> Any:
            method = getattr(self._client.chat.completions, method_name)
            return await method(model=resolved_model, **call_kwargs)

        try:
            completion = await _call_primary()
            self._accumulate_usage(completion)
            return completion
        except RateLimitError as exc:
            if self._fallback_client is None:
                logger.error("%s failed for model %s: %s", error_label, resolved_model, exc)
                raise LLMCallError(f"{error_label} failed for model {resolved_model}: {exc}") from exc

            logger.warning(
                "OpenRouter rate-limited for model %s; falling back to Groq model %s",
                resolved_model,
                FALLBACK_MODEL,
            )

            @with_retry(max_attempts=2, base_delay=1.0, retryable_exceptions=_RETRYABLE_LLM_EXCEPTIONS)
            async def _call_fallback() -> Any:
                fallback_method = getattr(self._fallback_client.chat.completions, method_name)
                return await fallback_method(model=FALLBACK_MODEL, **call_kwargs)

            try:
                completion = await _call_fallback()
            except OpenAIError as fallback_exc:
                logger.error("Groq fallback also failed for model %s: %s", FALLBACK_MODEL, fallback_exc)
                raise LLMCallError(
                    f"{error_label} failed for model {resolved_model} (rate limited), and Groq "
                    f"fallback failed for model {FALLBACK_MODEL}: {fallback_exc}"
                ) from fallback_exc

            logger.info("Request served by fallback provider Groq (model=%s)", FALLBACK_MODEL)
            self._accumulate_usage(completion)
            return completion
        except OpenAIError as exc:
            logger.error("%s failed for model %s: %s", error_label, resolved_model, exc)
            raise LLMCallError(f"{error_label} failed for model {resolved_model}: {exc}") from exc

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Generate a single completion, returning a normalized ``LLMResponse``.

        Raises:
            LLMCallError: if the underlying API call fails for any reason.
        """
        resolved_model = model or self._default_model
        messages = self._build_messages(prompt, system_prompt)

        start = time.perf_counter()
        completion = await self._create_completion(
            "create",
            resolved_model,
            {"messages": messages, "temperature": temperature},
            "LLM call",
        )
        latency_ms = (time.perf_counter() - start) * 1000

        usage = completion.usage
        return LLMResponse(
            content=completion.choices[0].message.content or "",
            model=completion.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency_ms,
        )

    async def generate_structured(
        self,
        prompt: str,
        response_model: type[_ResponseModelT],
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
    ) -> _ResponseModelT:
        """Generate a completion parsed directly into ``response_model``.

        Uses the SDK's ``chat.completions.parse`` structured-output path so
        callers get a validated Pydantic instance instead of hand-rolled JSON
        parsing at each call site.

        Raises:
            LLMCallError: if the API call fails, or the response can't be
                parsed into ``response_model``.
        """
        resolved_model = model or self._default_model
        messages = self._build_messages(prompt, system_prompt)

        completion = await self._create_completion(
            "parse",
            resolved_model,
            {"messages": messages, "response_format": response_model, "temperature": temperature},
            "Structured LLM call",
        )

        parsed = completion.choices[0].message.parsed
        if parsed is None:
            logger.error("Structured LLM response did not parse into %s", response_model.__name__)
            raise LLMCallError(
                f"Structured LLM response could not be parsed into {response_model.__name__}"
            )
        return parsed
