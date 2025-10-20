"""Ollama client implementation for local models."""

import json
from typing import Any, AsyncIterator, Dict, List, Optional

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from ..core.client import ModelClient
from ..core.message import Message, MessageRole, TextBlock, ImageBlock
from ..core.metrics import TokenCount
from ..core.exceptions import ProviderError, AuthenticationError, ModelError
from .stream_normalizer import get_stream_normalizer


class OllamaClient(ModelClient):
    """Ollama client implementation for local models."""
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 120.0,  # Longer timeout for local inference
        max_retries: int = 3,
        base_url: str = "http://localhost:11434",
        **kwargs
    ):
        if not AIOHTTP_AVAILABLE:
            raise ImportError(
                "aiohttp is required for Ollama. Install with: pip install aiohttp"
            )
        
        super().__init__(model, api_key, timeout, max_retries, **kwargs)
        
        self.base_url = base_url.rstrip('/')
        self.provider_name = "ollama"
        self.stream_normalizer = get_stream_normalizer("ollama")
        
        self.api_key = api_key
    
    def convert_schema_to_provider_format(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert universal schema to Ollama format (OpenAI compatible)."""
        return {
            "type": "function",
            "function": schema
        }
    
    def _convert_messages_to_ollama_format(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert messages to Ollama format."""
        ollama_messages = []
        
        for message in messages:
            ollama_message = {
                "role": message.role.value,
                "content": ""
            }
            
            if isinstance(message.content, str):
                ollama_message["content"] = message.content
            elif isinstance(message.content, list):
                content_parts = []
                images = []
                
                for block in message.content:
                    if isinstance(block, TextBlock):
                        content_parts.append(block.text)
                    elif isinstance(block, ImageBlock):
                        if block.image_url.startswith("data:"):
                            images.append(block.image_url.split(",")[1])
                        else:
                            content_parts.append(f"[Image: {block.image_url}]")
                
                ollama_message["content"] = " ".join(content_parts)
                if images:
                    ollama_message["images"] = images
            else:
                ollama_message["content"] = str(message.content)
            
            if message.tool_calls:
                ollama_message["tool_calls"] = message.tool_calls
            
            ollama_messages.append(ollama_message)
        
        return ollama_messages
    
    async def achat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        """Send chat completion request to Ollama."""
        try:
            ollama_messages = self._convert_messages_to_ollama_format(messages)
            
            payload = {
                "model": self.model,
                "messages": ollama_messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                }
            }
            
            if max_tokens:
                payload["options"]["num_predict"] = max_tokens
            
            payload["options"].update(kwargs)
            
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ProviderError(f"Ollama API error {response.status}: {error_text}", provider="ollama")
                    
                    result = await response.json()
            
            message_data = result.get("message", {})
            content = message_data.get("content", "")
            tool_calls = message_data.get("tool_calls", None)

            usage = TokenCount(
                input_tokens=sum(len(msg.get("content", "").split()) for msg in ollama_messages),
                output_tokens=len(content.split()),
                total_tokens=sum(len(msg.get("content", "").split()) for msg in ollama_messages) + len(content.split())
            )

            response_message = Message(
                role=MessageRole.ASSISTANT,
                content=content,
                tool_calls=tool_calls if tool_calls else None
            )
            
            from ..core.client import ChatResponse
            return ChatResponse(
                message=response_message,
                usage=usage,
                model=self.model,
                provider=self.provider_name,
                finish_reason="stop"
            )
            
        except Exception as e:
            if isinstance(e, ProviderError):
                raise
            raise ProviderError(f"Ollama API error: {e}", provider="ollama")
    
    async def astream_chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncIterator:
        """Send streaming chat completion request to Ollama."""
        try:
            ollama_messages = self._convert_messages_to_ollama_format(messages)
            
            payload = {
                "model": self.model,
                "messages": ollama_messages,
                "stream": True,
                "options": {
                    "temperature": temperature,
                }
            }
            
            if max_tokens:
                payload["options"]["num_predict"] = max_tokens
            
            payload["options"].update(kwargs)
            
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ProviderError(f"Ollama API error {response.status}: {error_text}", provider="ollama")
                    
                    accumulated_content = ""
                    
                    async for line in response.content:
                        if line:
                            try:
                                chunk_data = json.loads(line.decode('utf-8'))
                                
                                normalized_chunk = self.stream_normalizer.normalize(chunk_data)
                                
                                if normalized_chunk.delta:
                                    accumulated_content += normalized_chunk.delta
                                
                                normalized_chunk.content = accumulated_content
                                
                                yield normalized_chunk
                                
                                if normalized_chunk.finish_reason:
                                    break
                                        
                            except json.JSONDecodeError:
                                continue
            
        except Exception as e:
            if isinstance(e, ProviderError):
                raise
            raise ProviderError(f"Ollama streaming error: {e}", provider="ollama")


# Popular Ollama models (these need to be pulled locally first)
OLLAMA_MODELS = {
    # Llama models
    "llama3.1": "llama3.1",
    "llama3.1:8b": "llama3.1:8b",
    "llama3.1:70b": "llama3.1:70b",
    "llama3.1:405b": "llama3.1:405b",
    "llama3.2": "llama3.2",
    "llama3.2:3b": "llama3.2:3b",
    
    # Mistral models
    "mistral": "mistral",
    "mistral:7b": "mistral:7b",
    "mixtral": "mixtral",
    "mixtral:8x7b": "mixtral:8x7b",
    "mixtral:8x22b": "mixtral:8x22b",
    
    # Code models
    "codellama": "codellama",
    "codellama:7b": "codellama:7b",
    "codellama:13b": "codellama:13b",
    "codellama:34b": "codellama:34b",
    
    # Other popular models
    "phi3": "phi3",
    "phi3:mini": "phi3:mini",
    "phi3:medium": "phi3:medium",
    "gemma2": "gemma2",
    "gemma2:9b": "gemma2:9b",
    "gemma2:27b": "gemma2:27b",
    "qwen2.5": "qwen2.5",
    "qwen2.5:7b": "qwen2.5:7b",
    "qwen2.5:14b": "qwen2.5:14b",
    
    # Vision models
    "llava": "llava",
    "llava:7b": "llava:7b",
    "llava:13b": "llava:13b",
    "bakllava": "bakllava",
    
    # Embedding models
    "nomic-embed-text": "nomic-embed-text",
    "mxbai-embed-large": "mxbai-embed-large",
}
