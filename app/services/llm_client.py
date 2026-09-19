"""LLM client for OpenRouter's OpenAI-compatible chat completions endpoint.

We isolate the OpenAI SDK behind ``LLMClient`` so callers depend on our own
``LLMResponse``/``LLMCallError`` types instead of the SDK's, keeping the rest
of the app free to swap providers later without touching call sites.
"""

import time
from typing import Optional, TypeVar

from openai import AsyncOpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.logging import get_logger

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

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
    """Thin async wrapper around OpenRouter's chat completions API."""

    def __init__(self) -> None:
        settings = get_settings()
        self._default_model = settings.DEFAULT_MODEL
        self._client = AsyncOpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url=_OPENROUTER_BASE_URL,
        )

    @staticmethod
    def _build_messages(
        prompt: str, system_prompt: Optional[str]
    ) -> list[ChatCompletionMessageParam]:
        messages: list[ChatCompletionMessageParam] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

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
        try:
            completion = await self._client.chat.completions.create(
                model=resolved_model,
                messages=messages,
                temperature=temperature,
            )
        except OpenAIError as exc:
            logger.error("LLM call failed for model %s: %s", resolved_model, exc)
            raise LLMCallError(f"LLM call failed for model {resolved_model}: {exc}") from exc
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

        try:
            completion = await self._client.chat.completions.parse(
                model=resolved_model,
                messages=messages,
                response_format=response_model,
                temperature=temperature,
            )
        except OpenAIError as exc:
            logger.error("Structured LLM call failed for model %s: %s", resolved_model, exc)
            raise LLMCallError(f"Structured LLM call failed for model {resolved_model}: {exc}") from exc

        parsed = completion.choices[0].message.parsed
        if parsed is None:
            logger.error("Structured LLM response did not parse into %s", response_model.__name__)
            raise LLMCallError(
                f"Structured LLM response could not be parsed into {response_model.__name__}"
            )
        return parsed
