"""Mistral client implementation."""

import json
from typing import Any, AsyncIterator, Dict, List, Optional

from mistralai import Mistral

from ..core.client import ModelClient
from ..core.exceptions import AuthenticationError, ModelError, ProviderError
from ..core.message import ImageBlock, Message, MessageRole, TextBlock
from ..core.metrics import TokenCount
from ..core.streaming import StreamChunk


class MistralClient(ModelClient):
    """Mistral client implementation."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        **kwargs
    ):
        if not api_key:
            raise AuthenticationError("Mistral API key is required", provider="mistral")

        super().__init__(
            model=model,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs
        )

        self.client = Mistral(api_key=api_key)
        self.provider_name = "mistral"

        # Streaming state
        self._accumulated_content = ""

    def _reset_stream_state(self):
        """Reset streaming state for a new streaming session."""
        self._accumulated_content = ""

    def _normalize_stream_chunk(self, chunk: Any) -> StreamChunk:
        """Normalize Mistral streaming format to unified StreamChunk."""
        content = ""
        delta = ""
        finish_reason = None
        tool_calls = None
        usage = None

        try:
            if hasattr(chunk, 'choices') and chunk.choices:
                choice = chunk.choices[0]

                if hasattr(choice, 'delta') and choice.delta:
                    if hasattr(choice.delta, 'content'):
                        delta = choice.delta.content or ""
                        content = delta
                    if hasattr(choice.delta, 'tool_calls'):
                        tool_calls = choice.delta.tool_calls

                if hasattr(choice, 'finish_reason'):
                    finish_reason = choice.finish_reason

            if hasattr(chunk, 'usage') and chunk.usage:
                usage = TokenCount(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens
                )

        except AttributeError:
            content = str(chunk) if chunk else ""
            delta = content

        return StreamChunk(
            content=content,
            delta=delta,
            finish_reason=finish_reason,
            usage=usage,
            tool_calls=tool_calls
        )
    
    def convert_schema_to_provider_format(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert universal schema to Mistral format (OpenAI compatible)."""
        return {
            "type": "function",
            "function": schema
        }
    
    def _convert_messages_to_mistral_format(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert messages to Mistral format (OpenAI-compatible multimodal)."""
        mistral_messages = []

        for message in messages:
            role = message.role.value

            mistral_message = {"role": role}

            if isinstance(message.content, str):
                mistral_message["content"] = message.content
            elif isinstance(message.content, list):
                # Build content array for multimodal (OpenAI-compatible format)
                content_list = []
                for block in message.content:
                    if isinstance(block, TextBlock):
                        content_list.append({"type": "text", "text": block.text})
                    elif isinstance(block, ImageBlock):
                        # Mistral uses OpenAI-compatible image format
                        content_list.append({
                            "type": "image_url",
                            "image_url": {"url": block.image_url}
                        })

                mistral_message["content"] = content_list
            else:
                mistral_message["content"] = str(message.content)

            if message.name:
                mistral_message["name"] = message.name
            if message.tool_calls:
                mistral_message["tool_calls"] = message.tool_calls
            if message.tool_call_id:
                mistral_message["tool_call_id"] = message.tool_call_id

            mistral_messages.append(mistral_message)

        return mistral_messages
    
    async def achat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Send chat completion request to Mistral."""
        try:
            mistral_messages = self._convert_messages_to_mistral_format(messages)

            # Prepare request parameters
            request_params = {
                "model": self.model,
                "messages": mistral_messages,
                "temperature": temperature,
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
                        "strict": True,
                        "schema": json_schema
                    }
                }
            
            # Make API call
            response = await self.client.chat.complete_async(**request_params)
            
            # Extract response content
            content = response.choices[0].message.content or ""
            
            # Extract token usage
            usage = TokenCount()
            if response.usage:
                usage = TokenCount(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens
                )
            
            # Create response message
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
            raise ProviderError(f"Mistral API error: {e}", provider="mistral")
    
    async def astream_chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AsyncIterator:
        """Send streaming chat completion request to Mistral."""
        try:
            mistral_messages = self._convert_messages_to_mistral_format(messages)

            # Prepare request parameters
            request_params = {
                "model": self.model,
                "messages": mistral_messages,
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
                        "strict": True,
                        "schema": json_schema
                    }
                }
            
            # Stream response
            response_stream = await self.client.chat.stream_async(**request_params)

            # Reset stream state for new streaming session
            self._reset_stream_state()

            async for chunk in response_stream:
                # Normalize Mistral format to unified format
                normalized_chunk = self._normalize_stream_chunk(chunk)

                # Accumulate content
                if normalized_chunk.delta:
                    self._accumulated_content += normalized_chunk.delta

                # Update accumulated content in the chunk
                normalized_chunk.content = self._accumulated_content

                yield normalized_chunk
            
        except Exception as e:
            raise ProviderError(f"Mistral streaming error: {e}", provider="mistral")


MISTRAL_MODELS = {
    "mistral-large-latest": "mistral-large-latest",
    "mistral-medium-latest": "mistral-medium-latest", 
    "mistral-small-latest": "mistral-small-latest",
    "mistral-large-2402": "mistral-large-2402",
    "mistral-medium-2312": "mistral-medium-2312",
    "mistral-small-2312": "mistral-small-2312",
    "mistral-tiny-2312": "mistral-tiny-2312",
    "mixtral-8x7b-instruct-v0.1": "mixtral-8x7b-instruct-v0.1",
    "mixtral-8x22b-instruct-v0.1": "mixtral-8x22b-instruct-v0.1",
    "mistral-7b-instruct-v0.1": "mistral-7b-instruct-v0.1",
    "mistral-7b-instruct-v0.2": "mistral-7b-instruct-v0.2",
    "mistral-7b-instruct-v0.3": "mistral-7b-instruct-v0.3",
    "codestral-latest": "codestral-latest",
    "codestral-2405": "codestral-2405",
    "mistral-embed": "mistral-embed",
}
