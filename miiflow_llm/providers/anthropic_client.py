"""Anthropic provider implementation."""

import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from ..core.client import ChatResponse, ModelClient, StreamChunk
from ..core.exceptions import AuthenticationError, ModelError, ProviderError, RateLimitError
from ..core.exceptions import TimeoutError as MiiflowTimeoutError
from ..core.message import Message, MessageRole
from ..core.metrics import TokenCount, UsageData
from ..utils.image import data_uri_to_base64_and_mimetype
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
        sanitized_name = re.sub(r"[^a-zA-Z0-9_-]", "_", original_name)

        # Remove consecutive underscores and trim
        sanitized_name = re.sub(r"_+", "_", sanitized_name).strip("_")

        # Truncate to 128 chars if needed
        sanitized_name = sanitized_name[:128]

        # Store mapping if name was changed (for reversing tool calls)
        if sanitized_name != original_name:
            self._tool_name_mapping[sanitized_name] = original_name

        return {
            "name": sanitized_name,
            "description": schema["description"],
            "input_schema": schema["parameters"],
        }

    def convert_message_to_provider_format(self, message: Message) -> Dict[str, Any]:
        """Convert Message to Anthropic format."""
        from ..core.message import DocumentBlock, ImageBlock, TextBlock

        # Handle tool result messages (for sending tool outputs back)
        # Anthropic expects "user" role for tool results, not "tool"
        if message.tool_call_id and message.role in (MessageRole.USER, MessageRole.TOOL):
            # This is a tool result message - Anthropic requires "user" role
            anthropic_message = {"role": "user"}

            # Ensure tool result content is not empty or whitespace-only
            tool_content = (
                message.content if isinstance(message.content, str) else str(message.content)
            )
            if not tool_content or not tool_content.strip():
                tool_content = "[empty result]"

            anthropic_message["content"] = [
                {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": tool_content,
                }
            ]
            return anthropic_message

        anthropic_message = {"role": message.role.value}

        # Handle assistant messages with tool calls
        if message.tool_calls and message.role == MessageRole.ASSISTANT:
            content_list = []

            # Add text content if present and non-whitespace
            if message.content and message.content.strip():
                content_list.append({"type": "text", "text": message.content})

            # Add tool use blocks
            for tool_call in message.tool_calls:
                import json

                content_list.append(
                    {
                        "type": "tool_use",
                        "id": tool_call.get("id", ""),
                        "name": tool_call.get("function", {}).get("name", ""),
                        "input": tool_call.get("function", {}).get("arguments", {}),
                    }
                )

            anthropic_message["content"] = content_list
            return anthropic_message

        # Handle regular messages
        if isinstance(message.content, str):
            # Anthropic requires non-empty, non-whitespace content
            # Ensure we always have content, or use a placeholder
            content = message.content.strip() if message.content else ""

            if not content:
                # Empty content - use a minimal placeholder that's not whitespace
                # Anthropic rejects whitespace-only content
                anthropic_message["content"] = [{"type": "text", "text": "[no content]"}]
            else:
                anthropic_message["content"] = content
        else:
            content_list = []
            for block in message.content:
                if isinstance(block, TextBlock):
                    # Only add text blocks with non-whitespace content
                    if block.text and block.text.strip():
                        content_list.append({"type": "text", "text": block.text})
                elif isinstance(block, ImageBlock):
                    if block.image_url.startswith("data:"):
                        base64_content, media_type = data_uri_to_base64_and_mimetype(
                            block.image_url
                        )
                        content_list.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": base64_content,
                                },
                            }
                        )
                    else:
                        content_list.append(
                            {"type": "image", "source": {"type": "url", "url": block.image_url}}
                        )
                elif isinstance(block, DocumentBlock):
                    if block.document_url.startswith("data:"):
                        base64_content, media_type = data_uri_to_base64_and_mimetype(
                            block.document_url
                        )
                        content_list.append(
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": base64_content,
                                },
                            }
                        )
                    else:
                        content_list.append(
                            {
                                "type": "document",
                                "source": {
                                    "type": "url",
                                    "media_type": f"application/{block.document_type}",
                                    "data": block.document_url,
                                },
                            }
                        )

            # Ensure content_list is not empty (after filtering whitespace-only blocks)
            if not content_list:
                content_list = [{"type": "text", "text": "[no content]"}]

            anthropic_message["content"] = content_list

        return anthropic_message

    def _prepare_messages(
        self, messages: List[Message]
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
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
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), reraise=True
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
        """Send chat completion request to Anthropic."""
        try:
            system_content, anthropic_messages = self._prepare_messages(messages)

            # Handle JSON schema using Anthropic's native tool-based structured output
            json_tool_name = None
            if json_schema:
                json_tool_name = "json_tool"
                json_tool = {
                    "name": json_tool_name,
                    "description": "Respond with structured JSON matching the specified schema",
                    "input_schema": json_schema,
                }

                if tools:
                    tools = list(tools) + [json_tool]
                else:
                    tools = [json_tool]

                kwargs["tool_choice"] = {"type": "tool", "name": json_tool_name}

            request_params = {
                "model": self.model,
                "messages": anthropic_messages,
                "temperature": temperature,
                "max_tokens": max_tokens or 1024,
                **kwargs,
            }

            if system_content:
                request_params["system"] = system_content
            if tools:
                request_params["tools"] = tools

                # Debug: Log tools being sent to Anthropic
                import json
                import logging

                logger = logging.getLogger(__name__)
                logger.debug(
                    f"Anthropic tools parameter:\n{json.dumps(tools, indent=2, default=str)}"
                )

            response = await asyncio.wait_for(
                self.client.messages.create(**request_params), timeout=self.timeout
            )

            # Extract content and tool calls from response
            content = ""
            tool_calls = []

            if response.content:
                for block in response.content:
                    if hasattr(block, "text"):
                        content += block.text
                    elif hasattr(block, "type") and block.type == "tool_use":
                        if json_tool_name and block.name == json_tool_name:
                            # Extract JSON from tool response
                            content = json.dumps(block.input)
                        else:
                            # Convert Anthropic tool_use to OpenAI-compatible format
                            # Restore original tool name if it was sanitized
                            tool_name = block.name
                            original_name = self._tool_name_mapping.get(tool_name, tool_name)

                            tool_calls.append(
                                {
                                    "id": block.id,
                                    "type": "function",
                                    "function": {"name": original_name, "arguments": block.input},
                                }
                            )

            if tool_calls:
                logger.debug(f"Returning {len(tool_calls)} tool calls to orchestrator")

            response_message = Message(
                role=MessageRole.ASSISTANT,
                content=content,
                tool_calls=tool_calls if tool_calls else None,
            )

            usage = TokenCount(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            )

            return ChatResponse(
                message=response_message,
                usage=usage,
                model=self.model,
                provider=self.provider_name,
                finish_reason=response.stop_reason,
                metadata={"response_id": response.id},
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
            raise ProviderError(
                f"Anthropic API error: {str(e)}", self.provider_name, original_error=e
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
        """Send streaming chat completion request to Anthropic."""
        import logging

        logger = logging.getLogger(__name__)

        try:
            system_content, anthropic_messages = self._prepare_messages(messages)

            # Debug: Log the messages being sent
            import json

            logger.debug(f"Streaming request to Anthropic with {len(anthropic_messages)} messages:")
            for idx, msg in enumerate(anthropic_messages):
                logger.debug(
                    f"  Message {idx}: role={msg.get('role')}, content_type={type(msg.get('content'))}, content_length={len(str(msg.get('content')))}"
                )
                logger.debug(
                    f"    Content preview: {json.dumps(msg.get('content'), default=str)[:200]}"
                )

            json_tool_name = None
            if json_schema:
                json_tool_name = "json_tool"
                json_tool = {
                    "name": json_tool_name,
                    "description": "Respond with structured JSON matching the specified schema",
                    "input_schema": json_schema,
                }

                if tools:
                    tools = list(tools) + [json_tool]
                else:
                    tools = [json_tool]

                kwargs["tool_choice"] = {"type": "tool", "name": json_tool_name}

            request_params = {
                "model": self.model,
                "messages": anthropic_messages,
                "temperature": temperature,
                "max_tokens": max_tokens or 1024,
                "stream": True,
                **kwargs,
            }

            if system_content:
                request_params["system"] = system_content
            if tools:
                request_params["tools"] = tools

            stream = await asyncio.wait_for(
                self.client.messages.create(**request_params), timeout=self.timeout
            )

            # Reset and configure the normalizer for this streaming session
            self.stream_normalizer.reset()
            self.stream_normalizer.set_tool_name_mapping(self._tool_name_mapping)

            async for event in stream:
                # Use the normalizer to convert Anthropic events to StreamChunk
                normalized_chunk = self.stream_normalizer.normalize(event)

                # Only yield if there's actual content or metadata to send
                if (
                    normalized_chunk.delta
                    or normalized_chunk.tool_calls
                    or normalized_chunk.finish_reason
                ):
                    yield normalized_chunk

                # Stop on message_stop event
                if hasattr(event, "type") and event.type == "message_stop":
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
            raise ProviderError(
                f"Anthropic streaming error: {str(e)}", self.provider_name, original_error=e
            )
