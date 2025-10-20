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
        # Track sanitized -> original name mappings for tool calls
        self._tool_name_mapping: Dict[str, str] = {}
    
    def convert_schema_to_provider_format(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert universal schema to Anthropic format.

        Note: Anthropic requires tool names to match ^[a-zA-Z0-9_-]{1,128}$
        We sanitize names by replacing invalid characters with underscores.
        """
        import re

        # Sanitize name: replace spaces and invalid chars with underscores
        original_name = schema["name"]
        sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '_', original_name)

        # Remove consecutive underscores and trim
        sanitized_name = re.sub(r'_+', '_', sanitized_name).strip('_')

        # Truncate to 128 chars if needed
        sanitized_name = sanitized_name[:128]

        # Store mapping if name was changed (for reversing tool calls)
        if sanitized_name != original_name:
            self._tool_name_mapping[sanitized_name] = original_name

        return {
            "name": sanitized_name,
            "description": schema["description"],
            "input_schema": schema["parameters"]
        }
    
    def convert_message_to_provider_format(self, message: Message) -> Dict[str, Any]:
        """Convert Message to Anthropic format."""
        from ..core.message import TextBlock, ImageBlock, DocumentBlock

        anthropic_message = {"role": message.role.value}

        # Handle tool result messages (for sending tool outputs back)
        if message.tool_call_id and message.role == MessageRole.USER:
            # This is a tool result message
            anthropic_message["content"] = [
                {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": message.content if isinstance(message.content, str) else str(message.content)
                }
            ]
            return anthropic_message

        # Handle assistant messages with tool calls
        if message.tool_calls and message.role == MessageRole.ASSISTANT:
            content_list = []

            # Add text content if present
            if message.content:
                content_list.append({"type": "text", "text": message.content})

            # Add tool use blocks
            for tool_call in message.tool_calls:
                import json
                content_list.append({
                    "type": "tool_use",
                    "id": tool_call.get("id", ""),
                    "name": tool_call.get("function", {}).get("name", ""),
                    "input": tool_call.get("function", {}).get("arguments", {})
                })

            anthropic_message["content"] = content_list
            return anthropic_message

        # Handle regular messages
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

                # Debug: Log tools being sent to Anthropic
                import json
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"Anthropic tools parameter:\n{json.dumps(tools, indent=2, default=str)}")

            response = await asyncio.wait_for(
                self.client.messages.create(**request_params),
                timeout=self.timeout
            )

            # Extract content and tool calls from response
            content = ""
            tool_calls = []

            if response.content:
                for block in response.content:
                    if hasattr(block, 'text'):
                        content += block.text
                    elif hasattr(block, 'type') and block.type == 'tool_use':
                        # Convert Anthropic tool_use to OpenAI-compatible format
                        # Restore original tool name if it was sanitized
                        tool_name = block.name
                        original_name = self._tool_name_mapping.get(tool_name, tool_name)

                        logger.debug(f"Tool call extracted: {tool_name} -> {original_name}")
                        logger.debug(f"Tool call input: {block.input}")
                        logger.debug(f"Tool call input type: {type(block.input)}")

                        tool_calls.append({
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": original_name,
                                "arguments": block.input
                            }
                        })

            if tool_calls:
                logger.debug(f"Returning {len(tool_calls)} tool calls to orchestrator")

            response_message = Message(
                role=MessageRole.ASSISTANT,
                content=content,
                tool_calls=tool_calls if tool_calls else None
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
        import logging
        logger = logging.getLogger(__name__)

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
            tool_calls = []
            current_tool_use = None
            accumulated_tool_json = ""  # Accumulate partial JSON for tool arguments
            final_usage = None

            async for event in stream:
                if event.type == "content_block_start":
                    # Check if this is a tool use block
                    if hasattr(event, 'content_block') and hasattr(event.content_block, 'type'):
                        if event.content_block.type == 'tool_use':
                            # Restore original tool name if it was sanitized
                            tool_name = event.content_block.name
                            original_name = self._tool_name_mapping.get(tool_name, tool_name)

                            logger.debug(f"Streaming tool call started: {tool_name} -> {original_name}")

                            current_tool_use = {
                                "id": event.content_block.id,
                                "type": "function",
                                "function": {
                                    "name": original_name,
                                    "arguments": {}
                                }
                            }
                            accumulated_tool_json = ""  # Reset for new tool
                    continue

                elif event.type == "content_block_delta":
                    if hasattr(event.delta, 'text'):
                        # Text content
                        content_delta = event.delta.text
                        accumulated_content += content_delta

                        yield StreamChunk(
                            content=accumulated_content,
                            delta=content_delta,
                            finish_reason=None,
                            usage=None,
                            tool_calls=None
                        )
                    elif hasattr(event.delta, 'partial_json'):
                        # Tool use input (streaming JSON)
                        if current_tool_use:
                            # Accumulate partial JSON chunks
                            accumulated_tool_json += event.delta.partial_json

                            # Try to parse accumulated JSON
                            import json
                            try:
                                current_tool_use["function"]["arguments"] = json.loads(accumulated_tool_json)
                                logger.debug(f"Parsed tool arguments: {current_tool_use['function']['arguments']}")
                            except json.JSONDecodeError:
                                # Still accumulating, not complete yet
                                pass

                elif event.type == "content_block_stop":
                    # Finalize tool use if present
                    if current_tool_use:
                        # Final parse attempt with accumulated JSON
                        if accumulated_tool_json:
                            import json
                            try:
                                current_tool_use["function"]["arguments"] = json.loads(accumulated_tool_json)
                                logger.debug(f"Final tool arguments: {current_tool_use['function']['arguments']}")
                            except json.JSONDecodeError as e:
                                logger.error(f"Failed to parse tool arguments: {e}. JSON: {accumulated_tool_json}")
                                current_tool_use["function"]["arguments"] = {}

                        tool_calls.append(current_tool_use)
                        logger.debug(f"Tool call finalized: {current_tool_use['function']['name']}")
                        current_tool_use = None
                        accumulated_tool_json = ""
                    continue

                elif event.type == "message_delta":
                    if hasattr(event.delta, 'stop_reason'):
                        yield StreamChunk(
                            content=accumulated_content,
                            delta="",
                            finish_reason=event.delta.stop_reason,
                            usage=None,
                            tool_calls=tool_calls if tool_calls else None
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
