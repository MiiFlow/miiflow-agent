"""xAI Grok provider implementation."""

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

import openai
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from ..core.client import ChatResponse, ModelClient
from ..core.exceptions import (
    AuthenticationError,
    ModelError,
    ProviderError,
    RateLimitError,
    TimeoutError as MiiflowTimeoutError,
    is_retryable_error,
)
from ..core.message import Message, MessageRole
from ..core.metrics import TokenCount
from ..core.stream_normalizer import OpenAIStreamNormalizer
from ..core.streaming import StreamChunk
from ..models.xai import reasoning_effort_levels
from ..core.schema_normalizer import SchemaMode, normalize_json_schema
from .openai_client import OpenAIClient
from .sdk_client_cache import get_or_create_sdk_client


logger = logging.getLogger(__name__)


class XAIClient(ModelClient):
    """xAI provider client."""

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(model=model, api_key=api_key, **kwargs)
        self.effort = kwargs.get("effort")
        self.effort_overrides_call = kwargs.get("effort_overrides_call", False)
        self.provider_name = "xai"
        base_url = kwargs.get("base_url") or "https://api.x.ai/v1"
        self.client = get_or_create_sdk_client(
            f"xai:{base_url}",
            api_key,
            lambda: openai.AsyncOpenAI(api_key=api_key, base_url=base_url),
        )

    def _request_params(
        self, messages, temperature, max_tokens, tools, json_schema, kwargs
    ):
        """Build the same supported wire request for chat and streaming."""
        params = {"model": self.model, "messages": messages}
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if tools:
            params["tools"] = tools
            params["tool_choice"] = kwargs.get("tool_choice", "auto")
        if json_schema is not None:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response_schema",
                    "strict": True,
                    "schema": normalize_json_schema(
                        json_schema, mode=SchemaMode.STRICT
                    ),
                },
            }
        if kwargs.get("reasoning_effort") is not None:
            params["reasoning_effort"] = kwargs["reasoning_effort"]
        for key in (
            "top_p",
            "seed",
            "parallel_tool_calls",
            "response_format",
            "extra_headers",
        ):
            if key in kwargs and not (
                key == "response_format" and json_schema is not None
            ):
                params[key] = kwargs[key]
        return params

    def _with_configured_effort(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Follow the shared effort contract: unsupported defaults are dropped.

        Explicit call values remain configuration errors, outside API exception
        handling. A supported pinned default can override a per-call value.
        """
        levels = reasoning_effort_levels(self.model)
        if self.effort is not None and (
            kwargs.get("reasoning_effort") is None or self.effort_overrides_call
        ):
            if self.effort in levels:
                kwargs = {**kwargs, "reasoning_effort": self.effort}
            else:
                logger.debug(
                    "Dropping effort=%r for %s: model accepts %s",
                    self.effort,
                    self.model,
                    levels,
                )
        effort = kwargs.get("reasoning_effort")
        if effort is not None and levels and effort not in levels:
            raise ValueError(
                f"Unsupported reasoning effort {effort!r} for {self.model}: {levels}"
            )
        return kwargs

    def convert_schema_to_provider_format(
        self, schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert universal schema to xAI format (OpenAI compatible)."""
        return OpenAIClient.convert_schema_to_openai_format(schema)

    def convert_message_to_provider_format(self, message: Message) -> Dict[str, Any]:
        """Convert Message to xAI format (OpenAI compatible)."""
        return OpenAIClient.convert_message_to_openai_format(message)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception(is_retryable_error),
        reraise=True,
    )
    async def achat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ChatResponse:
        """Send chat completion request to xAI Grok."""
        kwargs = self._with_configured_effort(kwargs)
        try:
            # Offload sync DocumentBlock I/O / PDF extraction off the event loop.
            openai_messages = await asyncio.to_thread(
                lambda: [
                    self.convert_message_to_provider_format(msg) for msg in messages
                ]
            )

            request_params = self._request_params(
                openai_messages, temperature, max_tokens, tools, json_schema, kwargs
            )

            response = await asyncio.wait_for(
                self.client.chat.completions.create(**request_params),
                timeout=self.timeout,
            )

            choice = response.choices[0]
            content = choice.message.content or ""

            response_message = Message(
                role=MessageRole.ASSISTANT,
                content=content,
                tool_calls=choice.message.tool_calls,
            )

            usage = TokenCount.from_openai_usage(getattr(response, "usage", None))

            return ChatResponse(
                message=response_message,
                usage=usage,
                model=self.model,
                provider=self.provider_name,
                finish_reason=choice.finish_reason,
                metadata={"response_id": response.id},
            )

        except openai.AuthenticationError as e:
            raise AuthenticationError(str(e), self.provider_name, original_error=e)
        except openai.RateLimitError as e:
            raise RateLimitError(str(e), self.provider_name, original_error=e)
        except openai.BadRequestError as e:
            raise ModelError(str(e), self.model, original_error=e)
        except asyncio.TimeoutError as e:
            raise MiiflowTimeoutError(
                "Request timed out", self.timeout, original_error=e
            )
        except Exception as e:
            raise ProviderError(
                f"xAI Grok API error: {str(e)}", self.provider_name, original_error=e
            )

    async def astream_chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Send streaming chat completion request to xAI Grok."""
        kwargs = self._with_configured_effort(kwargs)
        try:
            # Offload sync DocumentBlock I/O / PDF extraction off the event loop.
            openai_messages = await asyncio.to_thread(
                lambda: [
                    self.convert_message_to_provider_format(msg) for msg in messages
                ]
            )

            request_params = self._request_params(
                openai_messages, temperature, max_tokens, tools, json_schema, kwargs
            )
            request_params.update(stream=True, stream_options={"include_usage": True})

            stream = await asyncio.wait_for(
                self.client.chat.completions.create(**request_params),
                timeout=self.timeout,
            )

            # Each stream owns its accumulator, including concurrent calls.
            normalizer = OpenAIStreamNormalizer(OpenAIClient._tool_name_mapping)
            last_usage = None
            try:
                async for chunk in stream:
                    normalized_chunk = normalizer.normalize_chunk(chunk)
                    # Usage is a cumulative snapshot, not a per-chunk delta.
                    # Emit only the last snapshot so LLMClient can sum safely,
                    # even if a gateway includes usage on content chunks too.
                    if normalized_chunk.usage is not None:
                        last_usage = normalized_chunk.usage
                        normalized_chunk.usage = None
                    if (
                        normalized_chunk.delta
                        or normalized_chunk.tool_calls
                        or normalized_chunk.finish_reason
                    ):
                        yield normalized_chunk
                if last_usage is not None:
                    yield StreamChunk(
                        content=normalized_chunk.content, delta="", usage=last_usage
                    )
            finally:
                await stream.close()

        except openai.AuthenticationError as e:
            raise AuthenticationError(str(e), self.provider_name, original_error=e)
        except openai.RateLimitError as e:
            raise RateLimitError(str(e), self.provider_name, original_error=e)
        except openai.BadRequestError as e:
            raise ModelError(str(e), self.model, original_error=e)
        except asyncio.TimeoutError as e:
            raise MiiflowTimeoutError(
                "Streaming request timed out", self.timeout, original_error=e
            )
        except Exception as e:
            raise ProviderError(
                f"xAI Grok streaming error: {str(e)}",
                self.provider_name,
                original_error=e,
            )
