"""TogetherAI client implementation."""

import json
from typing import Any, AsyncIterator, Dict, List, Optional

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from ..core.client import ModelClient
from ..core.message import Message, MessageRole
from ..core.metrics import TokenCount
from ..core.exceptions import ProviderError, AuthenticationError, ModelError
from .stream_normalizer import get_stream_normalizer


class TogetherClient(ModelClient):
    """TogetherAI client implementation using OpenAI-compatible API."""
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        **kwargs
    ):
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai is required for TogetherAI. Install with: pip install openai"
            )
        
        super().__init__(model, api_key, timeout, max_retries, **kwargs)
        
        if not api_key:
            raise AuthenticationError("TogetherAI API key is required", provider="together")
        
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.together.xyz/v1",
            timeout=timeout,
            max_retries=max_retries
        )
        
        self.provider_name = "together"
        self.stream_normalizer = get_stream_normalizer("together")
    
    def _convert_messages_to_openai_format(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert messages to OpenAI format."""
        openai_messages = []
        
        for message in messages:
            openai_message = message.to_openai_format()
            openai_messages.append(openai_message)
        
        return openai_messages
    
    async def achat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        """Send chat completion request to TogetherAI."""
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
            raise ProviderError(f"TogetherAI API error: {e}", provider="together")
    
    async def astream_chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncIterator:
        """Send streaming chat completion request to TogetherAI."""
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
            
            response_stream = await self.client.chat.completions.create(**request_params)
            
            accumulated_content = ""
            
            async for chunk in response_stream:
                normalized_chunk = self.stream_normalizer.normalize(chunk)
                
                if normalized_chunk.delta:
                    accumulated_content += normalized_chunk.delta
                
                normalized_chunk.content = accumulated_content
                
                yield normalized_chunk
            
        except Exception as e:
            raise ProviderError(f"TogetherAI streaming error: {e}", provider="together")


# Popular TogetherAI models
TOGETHER_MODELS = {
    # Meta Llama models
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    
    # Qwen models  
    "Qwen/Qwen2.5-7B-Instruct-Turbo": "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "Qwen/Qwen2.5-72B-Instruct-Turbo": "Qwen/Qwen2.5-72B-Instruct-Turbo",
    
    # Mistral models on Together
    "mistralai/Mistral-7B-Instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mixtral-8x22B-Instruct-v0.1": "mistralai/Mixtral-8x22B-Instruct-v0.1",
    
    # Other popular models
    "microsoft/DialoGPT-medium": "microsoft/DialoGPT-medium",
    "togethercomputer/RedPajama-INCITE-7B-Chat": "togethercomputer/RedPajama-INCITE-7B-Chat",
}
