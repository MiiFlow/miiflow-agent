"""xAI Grok provider implementation."""

import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional

import openai
from tenacity import retry, stop_after_attempt, wait_exponential

from ..core.client import ChatResponse, ModelClient
from ..core.exceptions import (
    AuthenticationError,
    ModelError,
    ProviderError,
    RateLimitError,
    TimeoutError as MiiflowTimeoutError,
)
from ..core.message import Message, MessageRole
from ..core.metrics import TokenCount, UsageData
from ..core.streaming import StreamChunk
from .openai_client import OpenAIClient, OpenAIStreaming


class XAIClient(OpenAIStreaming, ModelClient):
    """xAI provider client."""

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(
            model=model,
            api_key=api_key,
            **kwargs
        )
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1"
        )
        self.provider_name = "xai"

        # Initialize streaming state
        self._accumulated_content = ""
        self._accumulated_tool_calls = {}
    
    def convert_schema_to_provider_format(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert universal schema to xAI format (OpenAI compatible)."""
        return OpenAIClient.convert_schema_to_openai_format(schema)

    def convert_message_to_provider_format(self, message: Message) -> Dict[str, Any]:
        """Convert Message to xAI format (OpenAI compatible)."""
        return OpenAIClient.convert_message_to_openai_format(message)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def achat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ChatResponse:
        """Send chat completion request to xAI Grok."""
        try:
            openai_messages = [self.convert_message_to_provider_format(msg) for msg in messages]
            
            request_params = {
                "model": self.model,
                "messages": openai_messages,
                "temperature": temperature,
            }
            
            if max_tokens:
                request_params["max_tokens"] = max_tokens
            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = "auto"

            # Add JSON schema support (OpenAI-compatible)
            if json_schema:
                request_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response_schema",
                        "strict": True,
                        "schema": json_schema
                    }
                }

            response = await asyncio.wait_for(
                self.client.chat.completions.create(**request_params),
                timeout=self.timeout
            )
            
            choice = response.choices[0]
            content = choice.message.content or ""
            
            response_message = Message(
                role=MessageRole.ASSISTANT,
                content=content,
                tool_calls=choice.message.tool_calls
            )
            
            usage = TokenCount(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens
            )
            
            return ChatResponse(
                message=response_message,
                usage=usage,
                model=self.model,
                provider=self.provider_name,
                finish_reason=choice.finish_reason,
                metadata={"response_id": response.id}
            )
            
        except openai.AuthenticationError as e:
            raise AuthenticationError(str(e), self.provider_name, original_error=e)
        except openai.RateLimitError as e:
            retry_after = getattr(e.response.headers, 'retry-after', None)
            raise RateLimitError(str(e), self.provider_name, retry_after=retry_after, original_error=e)
        except openai.BadRequestError as e:
            raise ModelError(str(e), self.model, original_error=e)
        except asyncio.TimeoutError as e:
            raise MiiflowTimeoutError("Request timed out", self.timeout, original_error=e)
        except Exception as e:
            raise ProviderError(f"xAI Grok API error: {str(e)}", self.provider_name, original_error=e)
    
    async def astream_chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        """Send streaming chat completion request to xAI Grok."""
        try:
            openai_messages = [self.convert_message_to_provider_format(msg) for msg in messages]

            request_params = {
                "model": self.model,
                "messages": openai_messages,
                "temperature": temperature,
                "stream": True,
            }

            if max_tokens:
                request_params["max_tokens"] = max_tokens
            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = "auto"

            # Add JSON schema support (OpenAI-compatible)
            if json_schema:
                request_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response_schema",
                        "strict": True,
                        "schema": json_schema
                    }
                }

            stream = await asyncio.wait_for(
                self.client.chat.completions.create(**request_params),
                timeout=self.timeout
            )

            # Reset stream state for new streaming session
            self._reset_stream_state()

            async for chunk in stream:
                if not chunk.choices:
                    continue

                normalized_chunk = self._normalize_stream_chunk(chunk)

                # Only yield if there's content or metadata
                if (
                    normalized_chunk.delta
                    or normalized_chunk.tool_calls
                    or normalized_chunk.finish_reason
                ):
                    yield normalized_chunk
            
        except openai.AuthenticationError as e:
            raise AuthenticationError(str(e), self.provider_name, original_error=e)
        except openai.RateLimitError as e:
            retry_after = getattr(e.response.headers, 'retry-after', None)
            raise RateLimitError(str(e), self.provider_name, retry_after=retry_after, original_error=e)
        except openai.BadRequestError as e:
            raise ModelError(str(e), self.model, original_error=e)
        except asyncio.TimeoutError as e:
            raise MiiflowTimeoutError("Streaming request timed out", self.timeout, original_error=e)
        except Exception as e:
            raise ProviderError(f"xAI Grok streaming error: {str(e)}", self.provider_name, original_error=e)
