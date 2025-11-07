"""OpenRouter client implementation."""

import json
from typing import Any, AsyncIterator, Dict, List, Optional

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from ..core.client import ModelClient
from ..core.exceptions import AuthenticationError, ModelError, ProviderError
from ..core.message import Message, MessageRole
from ..core.metrics import TokenCount
from .openai_client import OpenAIClient, OpenAIStreaming


class OpenRouterClient(OpenAIStreaming, ModelClient):
    """OpenRouter client implementation using OpenAI-compatible API."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        app_name: Optional[str] = None,
        app_url: Optional[str] = None,
        **kwargs
    ):
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai is required for OpenRouter. Install with: pip install openai"
            )

        if not api_key:
            raise AuthenticationError("OpenRouter API key is required", provider="openrouter")

        headers = {}
        if app_name:
            headers["HTTP-Referer"] = app_url or "https://localhost"
            headers["X-Title"] = app_name

        super().__init__(
            model=model,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs
        )

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=timeout,
            max_retries=max_retries,
            default_headers=headers
        )
        self.provider_name = "openrouter"

        # Initialize streaming state
        self._accumulated_content = ""
        self._accumulated_tool_calls = {}
    
    def convert_schema_to_provider_format(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert universal schema to OpenRouter format (OpenAI compatible)."""
        return OpenAIClient.convert_schema_to_openai_format(schema)

    def convert_message_to_provider_format(self, message: Message) -> Dict[str, Any]:
        """Convert Message to OpenRouter format (OpenAI compatible)."""
        return OpenAIClient.convert_message_to_openai_format(message)

    def _convert_messages_to_openai_format(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert messages to OpenAI format."""
        openai_messages = []

        for message in messages:
            openai_message = self.convert_message_to_provider_format(message)
            openai_messages.append(openai_message)

        return openai_messages
    
    async def achat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Send chat completion request to OpenRouter."""
        try:
            openai_messages = self._convert_messages_to_openai_format(messages)

            request_params = {
                "model": self.model,
                "messages": openai_messages,
                "temperature": temperature,
                "stream": False,
                **kwargs
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
            
            response = await self.client.chat.completions.create(**request_params)
            
            content = response.choices[0].message.content or ""
            
            usage = TokenCount()
            if response.usage:
                usage = TokenCount(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens
                )
            
            response_message = Message(
                role=MessageRole.ASSISTANT,
                content=content,
                tool_calls=response.choices[0].message.tool_calls
            )
            
            from ..core.client import ChatResponse
            return ChatResponse(
                message=response_message,
                usage=usage,
                model=self.model,
                provider=self.provider_name,
                finish_reason=response.choices[0].finish_reason
            )
            
        except Exception as e:
            raise ProviderError(f"OpenRouter API error: {e}", provider="openrouter")
    
    async def astream_chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AsyncIterator:
        """Send streaming chat completion request to OpenRouter."""
        try:
            openai_messages = self._convert_messages_to_openai_format(messages)

            request_params = {
                "model": self.model,
                "messages": openai_messages,
                "temperature": temperature,
                "stream": True,
                **kwargs
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
            
            response_stream = await self.client.chat.completions.create(**request_params)

            # Reset stream state for new streaming session
            self._reset_stream_state()

            async for chunk in response_stream:
                normalized_chunk = self._normalize_stream_chunk(chunk)

                # Only yield if there's content or metadata
                if (
                    normalized_chunk.delta
                    or normalized_chunk.tool_calls
                    or normalized_chunk.finish_reason
                ):
                    yield normalized_chunk
            
        except Exception as e:
            raise ProviderError(f"OpenRouter streaming error: {e}", provider="openrouter")


# Popular OpenRouter models (gateway to many providers)
OPENROUTER_MODELS = {
    # OpenAI models via OpenRouter
    "openai/gpt-4o": "openai/gpt-4o",
    "openai/gpt-4o-mini": "openai/gpt-4o-mini", 
    "openai/gpt-4-turbo": "openai/gpt-4-turbo",
    "openai/o1-preview": "openai/o1-preview",
    "openai/o1-mini": "openai/o1-mini",
    
    # Anthropic models via OpenRouter
    "anthropic/claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-haiku": "anthropic/claude-3-haiku",
    "anthropic/claude-3-opus": "anthropic/claude-3-opus",
    
    # Google models via OpenRouter
    "google/gemini-pro-1.5": "google/gemini-pro-1.5",
    "google/gemini-flash-1.5": "google/gemini-flash-1.5",
    
    # Meta Llama models
    "meta-llama/llama-3.1-405b-instruct": "meta-llama/llama-3.1-405b-instruct",
    "meta-llama/llama-3.1-70b-instruct": "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3.1-8b-instruct": "meta-llama/llama-3.1-8b-instruct",
    
    # Mistral models
    "mistralai/mixtral-8x7b-instruct": "mistralai/mixtral-8x7b-instruct",
    "mistralai/mistral-7b-instruct": "mistralai/mistral-7b-instruct",
    
    # Other popular models
    "qwen/qwen-2.5-72b-instruct": "qwen/qwen-2.5-72b-instruct",
    "microsoft/wizardlm-2-8x22b": "microsoft/wizardlm-2-8x22b",
    "nvidia/llama-3.1-nemotron-70b-instruct": "nvidia/llama-3.1-nemotron-70b-instruct",
    
    # Free models (good for testing)
    "meta-llama/llama-3.2-3b-instruct:free": "meta-llama/llama-3.2-3b-instruct:free",
    "qwen/qwen-2-7b-instruct:free": "qwen/qwen-2-7b-instruct:free",
}
