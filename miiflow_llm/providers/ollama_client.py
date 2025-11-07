"""Ollama client implementation for local models."""

import json
from typing import Any, AsyncIterator, Dict, List, Optional

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from ..core.client import ModelClient
from ..core.exceptions import AuthenticationError, ModelError, ProviderError
from ..core.message import ImageBlock, Message, MessageRole, TextBlock
from ..core.metrics import TokenCount
from ..core.streaming import StreamChunk


class OllamaClient(ModelClient):
    """Ollama client implementation for local models."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        base_url: str = "http://localhost:11434",
        **kwargs
    ):
        if not AIOHTTP_AVAILABLE:
            raise ImportError(
                "aiohttp is required for Ollama. Install with: pip install aiohttp"
            )

        super().__init__(
            model=model,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs
        )
        self.base_url = base_url.rstrip('/')
        self.provider_name = "ollama"
        self.api_key = api_key

        # Streaming state
        self._accumulated_content = ""

    def _reset_stream_state(self):
        """Reset streaming state for a new streaming session."""
        self._accumulated_content = ""

    def _normalize_stream_chunk(self, chunk: Any) -> StreamChunk:
        """Normalize Ollama streaming format to unified StreamChunk."""
        content = ""
        delta = ""
        finish_reason = None

        try:
            if isinstance(chunk, dict):
                if "message" in chunk:
                    delta = chunk["message"].get("content", "")
                    content = delta
                if chunk.get("done", False):
                    finish_reason = "stop"
            elif hasattr(chunk, 'message'):
                delta = chunk.message.get("content", "")
                content = delta
                if hasattr(chunk, 'done') and chunk.done:
                    finish_reason = "stop"
            else:
                content = str(chunk) if chunk else ""
                delta = content

        except (AttributeError, TypeError):
            content = str(chunk) if chunk else ""
            delta = content

        return StreamChunk(
            content=content,
            delta=delta,
            finish_reason=finish_reason,
            usage=None,
            tool_calls=None
        )

    async def _image_url_to_base64(self, image_url: str) -> str:
        """Convert image URL to base64 data for Ollama."""
        if image_url.startswith("data:"):
            # Already base64 data URI - extract the base64 part
            return image_url.split(",")[1]

        # External URL - download and convert to base64
        try:
            import base64
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url, timeout=aiohttp.ClientTimeout(total=self.timeout)) as response:
                    if response.status != 200:
                        raise ProviderError(f"Failed to download image from {image_url}: {response.status}", provider="ollama")
                    image_bytes = await response.read()
                    return base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            raise ProviderError(f"Error converting image URL to base64: {e}", provider="ollama")
    
    def convert_schema_to_provider_format(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert universal schema to Ollama format (OpenAI compatible)."""
        return {
            "type": "function",
            "function": schema
        }
    
    async def _convert_messages_to_ollama_format(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert messages to Ollama format with async image download."""
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
                        # Convert image URL to base64 (handles both data URIs and external URLs)
                        try:
                            base64_data = await self._image_url_to_base64(block.image_url)
                            images.append(base64_data)
                        except Exception as e:
                            # If conversion fails, add as text placeholder
                            content_parts.append(f"[Image failed to load: {block.image_url}]")

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
        json_schema: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Send chat completion request to Ollama."""
        try:
            ollama_messages = await self._convert_messages_to_ollama_format(messages)

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

            # Add JSON schema support
            if json_schema:
                # Pass schema object to format parameter
                payload["format"] = json_schema
            
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
            
            content = result.get("message", {}).get("content", "")
            
            usage = TokenCount(
                input_tokens=sum(len(msg.get("content", "").split()) for msg in ollama_messages),
                output_tokens=len(content.split()),
                total_tokens=sum(len(msg.get("content", "").split()) for msg in ollama_messages) + len(content.split())
            )
            
            response_message = Message(
                role=MessageRole.ASSISTANT,
                content=content
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
        json_schema: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AsyncIterator:
        """Send streaming chat completion request to Ollama."""
        try:
            ollama_messages = await self._convert_messages_to_ollama_format(messages)

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

            # Add JSON schema support
            if json_schema:
                # Pass schema object to format parameter
                payload["format"] = json_schema

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

                    # Reset stream state for new streaming session
                    self._reset_stream_state()

                    async for line in response.content:
                        if line:
                            try:
                                chunk_data = json.loads(line.decode('utf-8'))

                                normalized_chunk = self._normalize_stream_chunk(chunk_data)

                                if normalized_chunk.delta:
                                    self._accumulated_content += normalized_chunk.delta

                                normalized_chunk.content = self._accumulated_content

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
