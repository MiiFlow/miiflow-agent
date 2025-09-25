"""Anthropic provider implementation."""

import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional

import anthropic
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


class AnthropicClient(ModelClient):
    """Anthropic provider client."""
    
    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(model=model, api_key=api_key, **kwargs)
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.provider_name = "anthropic"
        self.stream_normalizer = get_stream_normalizer("anthropic")
    
    def convert_schema_to_provider_format(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert universal schema to Anthropic format."""
        return {
            "name": schema["name"],
            "description": schema["description"],
            "input_schema": schema["parameters"]
        }
    
    def convert_message_to_provider_format(self, message: Message) -> Dict[str, Any]:
        """Convert Message to Anthropic format."""
        from ..core.message import TextBlock, ImageBlock, DocumentBlock
        
        anthropic_message = {"role": message.role.value}
        
        if isinstance(message.content, str):
            anthropic_message["content"] = message.content
        else:
            content_list = []
            for block in message.content:
                if isinstance(block, TextBlock):
                    content_list.append({"type": "text", "text": block.text})
                elif isinstance(block, ImageBlock):
                    if block.image_url.startswith("data:"):
                        base64_content, media_type = self._extract_base64_from_data_uri(block.image_url)
                        content_list.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_content
                            }
                        })
                    else:
                        content_list.append({
                            "type": "image",
                            "source": {
                                "type": "url",
                                "media_type": "image/jpeg",  # Default for URLs
                                "data": block.image_url
                            }
                        })
                elif isinstance(block, DocumentBlock):
                    if block.document_url.startswith("data:"):
                        base64_content, media_type = self._extract_base64_from_data_uri(block.document_url)
                        content_list.append({
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_content
                            }
                        })
                    else:
                        content_list.append({
                            "type": "document", 
                            "source": {
                                "type": "url",
                                "media_type": f"application/{block.document_type}",
                                "data": block.document_url
                            }
                        })
            anthropic_message["content"] = content_list
            
        return anthropic_message
    
    @staticmethod
    def _extract_base64_from_data_uri(data_uri: str) -> tuple[str, str]:
        """Universal base64 extractor for all multimedia content types."""
        if not data_uri.startswith("data:"):
            return data_uri, "application/octet-stream"
        
        try:
            if "," not in data_uri:
                return data_uri, "application/octet-stream"
                
            header, base64_content = data_uri.split(",", 1)
            
            # Extract media type from header: data:media_type;base64
            media_type = "application/octet-stream"  # default fallback
            if ":" in header and ";" in header:
                media_type = header.split(":")[1].split(";")[0]
            
            return base64_content, media_type
            
        except Exception:
            return data_uri, "application/octet-stream"
    
    def _prepare_messages(self, messages: List[Message]) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """Prepare messages for Anthropic format (system separate)."""
        system_content = None
        anthropic_messages = []
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_content = msg.content if isinstance(msg.content, str) else str(msg.content)
            else:
                anthropic_messages.append(self.convert_message_to_provider_format(msg))
        
        return system_content, anthropic_messages
    
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
        **kwargs
    ) -> ChatResponse:
        """Send chat completion request to Anthropic."""
        try:
            system_content, anthropic_messages = self._prepare_messages(messages)
            
            request_params = {
                "model": self.model,
                "messages": anthropic_messages,
                "temperature": temperature,
                "max_tokens": max_tokens or 1024,
            }
            
            if system_content:
                request_params["system"] = system_content
            if tools:
                request_params["tools"] = tools
            
            response = await asyncio.wait_for(
                self.client.messages.create(**request_params),
                timeout=self.timeout
            )
            
            content = ""
            if response.content:
                for block in response.content:
                    if hasattr(block, 'text'):
                        content += block.text
            
            response_message = Message(
                role=MessageRole.ASSISTANT,
                content=content,
                tool_calls=getattr(response, 'tool_calls', None)
            )
            
            usage = TokenCount(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens
            )
            
            return ChatResponse(
                message=response_message,
                usage=usage,
                model=self.model,
                provider=self.provider_name,
                finish_reason=response.stop_reason,
                metadata={"response_id": response.id}
            )
            
        except anthropic.AuthenticationError as e:
            raise AuthenticationError(str(e), self.provider_name, original_error=e)
        except anthropic.RateLimitError as e:
            raise RateLimitError(str(e), self.provider_name, original_error=e)
        except anthropic.BadRequestError as e:
            raise ModelError(str(e), self.model, original_error=e)
        except asyncio.TimeoutError as e:
            raise MiiflowTimeoutError("Request timed out", self.timeout, original_error=e)
        except Exception as e:
            raise ProviderError(f"Anthropic API error: {str(e)}", self.provider_name, original_error=e)
    
    async def astream_chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        """Send streaming chat completion request to Anthropic."""
        try:
            system_content, anthropic_messages = self._prepare_messages(messages)
            
            request_params = {
                "model": self.model,
                "messages": anthropic_messages,
                "temperature": temperature,
                "max_tokens": max_tokens or 1024,
                "stream": True,
            }
            
            if system_content:
                request_params["system"] = system_content
            if tools:
                request_params["tools"] = tools
            
            stream = await asyncio.wait_for(
                self.client.messages.create(**request_params),
                timeout=self.timeout
            )
            
            accumulated_content = ""
            final_usage = None
            
            async for event in stream:
                if event.type == "content_block_start":
                    continue
                elif event.type == "content_block_delta":
                    if hasattr(event.delta, 'text'):
                        content_delta = event.delta.text
                        accumulated_content += content_delta
                        
                        yield StreamChunk(
                            content=accumulated_content,
                            delta=content_delta,
                            finish_reason=None,
                            usage=None,
                            tool_calls=None
                        )
                elif event.type == "content_block_stop":
                    continue
                elif event.type == "message_delta":
                    if hasattr(event.delta, 'stop_reason'):
                        yield StreamChunk(
                            content=accumulated_content,
                            delta="",
                            finish_reason=event.delta.stop_reason,
                            usage=None,
                            tool_calls=None
                        )
                elif event.type == "message_stop":
                    break
            
        except anthropic.AuthenticationError as e:
            raise AuthenticationError(str(e), self.provider_name, original_error=e)
        except anthropic.RateLimitError as e:
            raise RateLimitError(str(e), self.provider_name, original_error=e)
        except anthropic.BadRequestError as e:
            raise ModelError(str(e), self.model, original_error=e)
        except asyncio.TimeoutError as e:
            raise MiiflowTimeoutError("Streaming request timed out", self.timeout, original_error=e)
        except Exception as e:
            raise ProviderError(f"Anthropic streaming error: {str(e)}", self.provider_name, original_error=e)
