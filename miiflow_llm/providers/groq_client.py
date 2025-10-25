"""Groq provider implementation."""

import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional

import groq
from groq import AsyncGroq, Groq
from tenacity import retry, stop_after_attempt, wait_exponential

from ..core.client import ModelClient, ChatResponse, StreamChunk
from ..core.message import Message, MessageRole
from ..core.metrics import TokenCount, UsageData
from ..core.exceptions import (
    AuthenticationError,
    RateLimitError,
    ModelError,
    TimeoutError as MiiflowTimeoutError,
    ProviderError,
)
from .stream_normalizer import get_stream_normalizer


class GroqClient(ModelClient):
    """Groq provider client."""
    
    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(model=model, api_key=api_key, **kwargs)
        self.client = AsyncGroq(api_key=api_key)
        self.provider_name = "groq"
        self.stream_normalizer = get_stream_normalizer("groq")
    
    def convert_schema_to_provider_format(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert universal schema to Groq format (OpenAI compatible)."""
        return {
            "type": "function",
            "function": schema
        }
    
    def convert_message_to_provider_format(self, message: Message) -> Dict[str, Any]:
        """Convert Message to Groq format (reuse OpenAI logic since compatible)."""
        from .openai_client import OpenAIClient
        return OpenAIClient.convert_message_to_provider_format(OpenAIClient("", ""), message)
    
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
        """Send chat completion request to Groq."""
        try:
            groq_messages = [self.convert_message_to_provider_format(msg) for msg in messages]

            request_params = {
                "model": self.model,
                "messages": groq_messages,
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
            
        except groq.AuthenticationError as e:
            raise AuthenticationError(str(e), self.provider_name, original_error=e)
        except groq.RateLimitError as e:
            retry_after = getattr(e.response.headers, 'retry-after', None)
            raise RateLimitError(str(e), self.provider_name, retry_after=retry_after, original_error=e)
        except groq.BadRequestError as e:
            raise ModelError(str(e), self.model, original_error=e)
        except groq.APIConnectionError as e:
            raise ProviderError(f"Groq connection error: {str(e)}", self.provider_name, original_error=e)
        except asyncio.TimeoutError as e:
            raise MiiflowTimeoutError("Request timed out", self.timeout, original_error=e)
        except groq.APIStatusError as e:
            # Handle other HTTP status codes
            if e.status_code >= 500:
                raise ProviderError(f"Groq server error ({e.status_code}): {str(e)}", self.provider_name, original_error=e)
            else:
                raise ModelError(f"Groq API error ({e.status_code}): {str(e)}", self.model, original_error=e)
        except Exception as e:
            raise ProviderError(f"Groq API error: {str(e)}", self.provider_name, original_error=e)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def astream_chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        """Send streaming chat completion request to Groq."""
        try:
            groq_messages = [self.convert_message_to_provider_format(msg) for msg in messages]

            request_params = {
                "model": self.model,
                "messages": groq_messages,
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
                        "schema": json_schema
                    }
                }
            
            stream = await asyncio.wait_for(
                self.client.chat.completions.create(**request_params),
                timeout=self.timeout
            )
            
            accumulated_content = ""
            
            async for chunk in stream:
                if not chunk.choices:
                    continue
                
                # Use stream normalizer to convert Groq format to unified format
                normalized_chunk = self.stream_normalizer.normalize(chunk)
                
                # Accumulate content
                if normalized_chunk.delta:
                    accumulated_content += normalized_chunk.delta
                
                # Update accumulated content in the chunk
                normalized_chunk.content = accumulated_content
                
                yield normalized_chunk
            
        except groq.AuthenticationError as e:
            raise AuthenticationError(str(e), self.provider_name, original_error=e)
        except groq.RateLimitError as e:
            retry_after = getattr(e.response.headers, 'retry-after', None)
            raise RateLimitError(str(e), self.provider_name, retry_after=retry_after, original_error=e)
        except groq.BadRequestError as e:
            raise ModelError(str(e), self.model, original_error=e)
        except groq.APIConnectionError as e:
            raise ProviderError(f"Groq streaming connection error: {str(e)}", self.provider_name, original_error=e)
        except asyncio.TimeoutError as e:
            raise MiiflowTimeoutError("Streaming request timed out", self.timeout, original_error=e)
        except groq.APIStatusError as e:
            # Handle other HTTP status codes
            if e.status_code >= 500:
                raise ProviderError(f"Groq streaming server error ({e.status_code}): {str(e)}", self.provider_name, original_error=e)
            else:
                raise ModelError(f"Groq streaming API error ({e.status_code}): {str(e)}", self.model, original_error=e)
        except Exception as e:
            raise ProviderError(f"Groq streaming error: {str(e)}", self.provider_name, original_error=e)
