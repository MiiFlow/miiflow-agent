"""OpenAI provider implementation."""

from __future__ import annotations

import asyncio
import copy
import json
import re
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional

import openai
from tenacity import retry, stop_after_attempt, wait_exponential

from ..core.client import ChatResponse, ModelClient
from ..core.exceptions import AuthenticationError, ModelError, ProviderError, RateLimitError
from ..core.exceptions import TimeoutError as MiiflowTimeoutError
from ..core.message import DocumentBlock, ImageBlock, Message, MessageRole, TextBlock
from ..core.metrics import TokenCount, UsageData
from ..core.schema_normalizer import SchemaMode, normalize_json_schema
from ..core.stream_normalizer import OpenAIStreamNormalizer
from ..core.streaming import StreamChunk
from ..models.openai import get_token_param_name, supports_native_mcp, supports_temperature

if TYPE_CHECKING:
    from ..core.tools.mcp import NativeMCPServerConfig


def _sanitize_tool_name(name: str) -> str:
    """Sanitize tool name to match OpenAI's pattern: ^[a-zA-Z0-9_-]+$"""
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized[:64]  # OpenAI has a 64 char limit for function names


class OpenAIClient(ModelClient):
    """OpenAI provider client."""

    # Class-level mapping shared across instances for tool name resolution
    # Maps sanitized names back to original names
    _tool_name_mapping: Dict[str, str] = {}

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(model=model, api_key=api_key, **kwargs)
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.provider_name = "openai"

        # Stream normalizer for unified streaming handling
        # Note: Pass class-level mapping for tool name restoration
        self._stream_normalizer = OpenAIStreamNormalizer(OpenAIClient._tool_name_mapping)

    def _supports_native_mcp(self) -> bool:
        """Check if current model supports native MCP via Responses API."""
        return supports_native_mcp(self.model)

    def convert_schema_to_provider_format(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert universal schema to OpenAI format with name sanitization."""
        original_name = schema["name"]
        sanitized_name = _sanitize_tool_name(original_name)

        # Track mapping for restoring original names from tool call responses
        if sanitized_name != original_name:
            OpenAIClient._tool_name_mapping[sanitized_name] = original_name

        sanitized_schema = {**schema, "name": sanitized_name}
        return {"type": "function", "function": sanitized_schema}

    @staticmethod
    def convert_schema_to_openai_format(schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert universal schema to OpenAI format with name sanitization.

        Note: For proper name mapping support, use convert_schema_to_provider_format instead.
        """
        original_name = schema["name"]
        sanitized_name = _sanitize_tool_name(original_name)

        # Track mapping for restoring original names from tool call responses
        if sanitized_name != original_name:
            OpenAIClient._tool_name_mapping[sanitized_name] = original_name

        sanitized_schema = {**schema, "name": sanitized_name}
        return {"type": "function", "function": sanitized_schema}

    def convert_message_to_provider_format(self, message: Message) -> Dict[str, Any]:
        return OpenAIClient.convert_message_to_openai_format(message)

    @staticmethod
    def convert_message_to_openai_format(message: Message) -> Dict[str, Any]:
        """Convert universal Message to OpenAI format (static for reuse by compatible providers)."""
        openai_message = {"role": message.role.value}

        if isinstance(message.content, str):
            openai_message["content"] = message.content
        else:
            content_list = []
            for block in message.content:
                if isinstance(block, TextBlock):
                    content_list.append({"type": "text", "text": block.text})
                elif isinstance(block, ImageBlock):
                    content_list.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": block.image_url, "detail": block.detail},
                        }
                    )
                elif isinstance(block, DocumentBlock):
                    try:
                        from ..utils.pdf_extractor import extract_pdf_text_simple

                        pdf_text = extract_pdf_text_simple(block.document_url)

                        filename_info = f" [{block.filename}]" if block.filename else ""
                        pdf_content = f"[PDF Document{filename_info}]\n\n{pdf_text}"

                        content_list.append({"type": "text", "text": pdf_content})
                    except Exception as e:
                        filename_info = f" {block.filename}" if block.filename else ""
                        error_content = f"[Error processing PDF{filename_info}: {str(e)}]"
                        content_list.append({"type": "text", "text": error_content})

            openai_message["content"] = content_list

        if message.name:
            openai_message["name"] = message.name
        if message.tool_call_id:
            openai_message["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            # Sanitize tool names in tool_calls for OpenAI compatibility
            sanitized_tool_calls = []
            for tc in message.tool_calls:
                sanitized_tc = copy.deepcopy(tc) if isinstance(tc, dict) else tc
                if isinstance(sanitized_tc, dict) and "function" in sanitized_tc:
                    original_name = sanitized_tc["function"].get("name", "")
                    sanitized_tc["function"]["name"] = _sanitize_tool_name(original_name)
                sanitized_tool_calls.append(sanitized_tc)
            openai_message["tool_calls"] = sanitized_tool_calls

        return openai_message

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
        mcp_servers: Optional[List["NativeMCPServerConfig"]] = None,
        **kwargs,
    ) -> ChatResponse:
        """Send chat completion request to OpenAI.

        Args:
            messages: List of conversation messages
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            tools: Tool schemas for function calling
            json_schema: JSON schema for structured output
            mcp_servers: Optional list of MCP server configs for native MCP support.
                        When provided, uses Responses API instead of Chat Completions.
            **kwargs: Additional parameters passed to the API

        Returns:
            ChatResponse with assistant message and metadata
        """
        # Auto-detect: use Responses API if MCP servers are provided
        use_native_mcp = mcp_servers and len(mcp_servers) > 0 and self._supports_native_mcp()

        if use_native_mcp:
            return await self._achat_responses_api(
                messages, temperature, max_tokens, tools, mcp_servers, **kwargs
            )

        # Use standard Chat Completions API
        try:
            openai_messages = [self.convert_message_to_provider_format(msg) for msg in messages]

            request_params = {
                "model": self.model,
                "messages": openai_messages,
            }

            if supports_temperature(self.model):
                request_params["temperature"] = temperature

            if max_tokens is not None:
                request_params[get_token_param_name(self.model)] = max_tokens
            else:
                # Provide sensible default when not specified
                request_params[get_token_param_name(self.model)] = 16384
            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = "auto"

            if json_schema:
                normalized_schema = normalize_json_schema(json_schema, SchemaMode.STRICT)
                request_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response_schema",
                        "schema": normalized_schema,
                    },
                }

            response = await asyncio.wait_for(
                self.client.chat.completions.create(**request_params), timeout=self.timeout
            )

            choice = response.choices[0]
            content = choice.message.content or ""

            response_message = Message(
                role=MessageRole.ASSISTANT, content=content, tool_calls=choice.message.tool_calls
            )

            usage = TokenCount(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

            return ChatResponse(
                message=response_message,
                usage=usage,
                model=self.model,
                provider=self.provider_name,
                finish_reason=choice.finish_reason,
                metadata={"response_id": response.id},
            )

        except openai.AuthenticationError as e:
            raise AuthenticationError(str(e), self.provider_name, original_error=e)
        except openai.RateLimitError as e:
            retry_after = getattr(e.response.headers, "retry-after", None)
            raise RateLimitError(
                str(e), self.provider_name, retry_after=retry_after, original_error=e
            )
        except openai.BadRequestError as e:
            raise ModelError(str(e), self.model, original_error=e)
        except asyncio.TimeoutError as e:
            raise MiiflowTimeoutError("Request timed out", self.timeout, original_error=e)
        except Exception as e:
            raise ProviderError(f"OpenAI API error: {str(e)}", self.provider_name, original_error=e)

    async def _achat_responses_api(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        mcp_servers: Optional[List["NativeMCPServerConfig"]] = None,
        **kwargs,
    ) -> ChatResponse:
        """Chat using OpenAI Responses API (supports native MCP).

        The Responses API is a different API surface from Chat Completions,
        designed for agentic workflows with native MCP server support.

        Args:
            messages: List of conversation messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            tools: Regular function tools
            mcp_servers: MCP server configurations for native MCP

        Returns:
            ChatResponse with assistant message and metadata
        """
        try:
            # Build tools array with MCP servers
            response_tools = []

            # Add regular function tools
            if tools:
                response_tools.extend(tools)

            # Add MCP servers as native MCP tools
            if mcp_servers:
                for server in mcp_servers:
                    response_tools.append(server.to_openai_format())

            # Convert messages to Responses API input format
            # Responses API uses a different input structure
            input_items = self._convert_messages_to_responses_input(messages)

            request_params = {
                "model": self.model,
                "input": input_items,
            }

            if response_tools:
                request_params["tools"] = response_tools

            if max_tokens:
                request_params["max_output_tokens"] = max_tokens

            # Use Responses API endpoint
            response = await asyncio.wait_for(
                self.client.responses.create(**request_params), timeout=self.timeout
            )

            return self._parse_responses_api_response(response)

        except openai.AuthenticationError as e:
            raise AuthenticationError(str(e), self.provider_name, original_error=e)
        except openai.RateLimitError as e:
            retry_after = getattr(e.response.headers, "retry-after", None)
            raise RateLimitError(
                str(e), self.provider_name, retry_after=retry_after, original_error=e
            )
        except openai.BadRequestError as e:
            raise ModelError(str(e), self.model, original_error=e)
        except asyncio.TimeoutError as e:
            raise MiiflowTimeoutError("Request timed out", self.timeout, original_error=e)
        except Exception as e:
            raise ProviderError(
                f"OpenAI Responses API error: {str(e)}", self.provider_name, original_error=e
            )

    def _convert_messages_to_responses_input(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert messages to Responses API input format.

        The Responses API uses a different input structure than Chat Completions.
        It expects an array of input items rather than messages.
        """
        input_items = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                # System messages become system items
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                input_items.append(
                    {
                        "type": "message",
                        "role": "system",
                        "content": content,
                    }
                )
            elif msg.role == MessageRole.USER:
                # User messages
                if isinstance(msg.content, str):
                    input_items.append(
                        {
                            "type": "message",
                            "role": "user",
                            "content": msg.content,
                        }
                    )
                else:
                    # Handle multimodal content
                    content_parts = []
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            content_parts.append({"type": "input_text", "text": block.text})
                        elif isinstance(block, ImageBlock):
                            content_parts.append(
                                {
                                    "type": "input_image",
                                    "image_url": block.image_url,
                                }
                            )
                    input_items.append(
                        {
                            "type": "message",
                            "role": "user",
                            "content": content_parts,
                        }
                    )
            elif msg.role == MessageRole.ASSISTANT:
                # Assistant messages
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                input_items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": content,
                    }
                )
            elif msg.role == MessageRole.TOOL:
                # Tool results
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": msg.tool_call_id or "",
                        "output": msg.content if isinstance(msg.content, str) else str(msg.content),
                    }
                )

        return input_items

    def _parse_responses_api_response(self, response: Any) -> ChatResponse:
        """Parse Responses API output format.

        The Responses API returns output items instead of choices/messages.
        """
        content = ""
        tool_calls = []

        # Extract content and tool calls from output items
        for item in getattr(response, "output", []):
            item_type = getattr(item, "type", None)

            if item_type == "message":
                # Text message content
                for part in getattr(item, "content", []):
                    part_type = getattr(part, "type", None)
                    if part_type == "output_text":
                        content += getattr(part, "text", "")
                    elif part_type == "text":
                        content += getattr(part, "text", "")

            elif item_type == "function_call":
                # Regular function call
                tool_calls.append(
                    {
                        "id": getattr(item, "call_id", ""),
                        "type": "function",
                        "function": {
                            "name": getattr(item, "name", ""),
                            "arguments": getattr(item, "arguments", "{}"),
                        },
                    }
                )

            elif item_type == "mcp_call":
                # Native MCP call result
                tool_calls.append(
                    {
                        "id": getattr(item, "id", ""),
                        "type": "mcp_function",
                        "function": {
                            "name": getattr(item, "name", ""),
                            "arguments": getattr(item, "arguments", "{}"),
                        },
                        "server_label": getattr(item, "server_label", None),
                    }
                )

        # Extract usage
        usage_data = getattr(response, "usage", None)
        usage = TokenCount(
            prompt_tokens=getattr(usage_data, "input_tokens", 0) if usage_data else 0,
            completion_tokens=getattr(usage_data, "output_tokens", 0) if usage_data else 0,
            total_tokens=getattr(usage_data, "total_tokens", 0) if usage_data else 0,
        )

        response_message = Message(
            role=MessageRole.ASSISTANT,
            content=content,
            tool_calls=tool_calls if tool_calls else None,
        )

        return ChatResponse(
            message=response_message,
            usage=usage,
            model=self.model,
            provider=self.provider_name,
            finish_reason=getattr(response, "status", "stop"),
            metadata={"response_id": getattr(response, "id", "")},
        )

    async def astream_chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        mcp_servers: Optional[List["NativeMCPServerConfig"]] = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Stream chat completion from OpenAI.

        Args:
            messages: List of conversation messages
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            tools: Tool schemas for function calling
            json_schema: JSON schema for structured output
            mcp_servers: Optional list of MCP server configs for native MCP support.
                        When provided, uses Responses API streaming instead of Chat Completions.
            **kwargs: Additional parameters passed to the API

        Yields:
            StreamChunk with delta content and metadata
        """
        # Auto-detect: use Responses API if MCP servers are provided
        use_native_mcp = mcp_servers and len(mcp_servers) > 0 and self._supports_native_mcp()

        if use_native_mcp:
            async for chunk in self._astream_chat_responses_api(
                messages, temperature, max_tokens, tools, mcp_servers, **kwargs
            ):
                yield chunk
            return

        # Use standard Chat Completions API streaming
        try:
            openai_messages = [self.convert_message_to_provider_format(msg) for msg in messages]

            request_params = {
                "model": self.model,
                "messages": openai_messages,
                "stream": True,
            }

            if supports_temperature(self.model):
                request_params["temperature"] = temperature

            if max_tokens is not None:
                request_params[get_token_param_name(self.model)] = max_tokens
            else:
                # Provide sensible default when not specified
                request_params[get_token_param_name(self.model)] = 16384
            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = "auto"

            if json_schema:
                normalized_schema = normalize_json_schema(json_schema, SchemaMode.STRICT)
                request_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response_schema",
                        "schema": normalized_schema,
                    },
                }

            stream = await asyncio.wait_for(
                self.client.chat.completions.create(**request_params), timeout=self.timeout
            )

            # Reset stream state for new streaming session
            self._stream_normalizer.reset_state()

            async for chunk in stream:
                if not chunk.choices:
                    continue

                normalized_chunk = self._stream_normalizer.normalize_chunk(chunk)

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
            retry_after = getattr(e.response.headers, "retry-after", None)
            raise RateLimitError(
                str(e), self.provider_name, retry_after=retry_after, original_error=e
            )
        except openai.BadRequestError as e:
            raise ModelError(str(e), self.model, original_error=e)
        except asyncio.TimeoutError as e:
            raise MiiflowTimeoutError("Streaming request timed out", self.timeout, original_error=e)
        except Exception as e:
            raise ProviderError(
                f"OpenAI streaming error: {str(e)}", self.provider_name, original_error=e
            )

    async def _astream_chat_responses_api(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        mcp_servers: Optional[List["NativeMCPServerConfig"]] = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Stream chat using OpenAI Responses API (supports native MCP).

        The Responses API supports streaming with a different event format.

        Args:
            messages: List of conversation messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            tools: Regular function tools
            mcp_servers: MCP server configurations for native MCP

        Yields:
            StreamChunk with delta content and metadata
        """
        try:
            # Build tools array with MCP servers
            response_tools = []

            # Add regular function tools
            if tools:
                response_tools.extend(tools)

            # Add MCP servers as native MCP tools
            if mcp_servers:
                for server in mcp_servers:
                    response_tools.append(server.to_openai_format())

            # Convert messages to Responses API input format
            input_items = self._convert_messages_to_responses_input(messages)

            request_params = {
                "model": self.model,
                "input": input_items,
                "stream": True,
            }

            if response_tools:
                request_params["tools"] = response_tools

            if max_tokens:
                request_params["max_output_tokens"] = max_tokens

            # Use Responses API streaming
            stream = await asyncio.wait_for(
                self.client.responses.create(**request_params), timeout=self.timeout
            )

            accumulated_content = ""

            async for event in stream:
                event_type = getattr(event, "type", None)

                if event_type == "response.output_text.delta":
                    # Text content delta
                    delta = getattr(event, "delta", "")
                    accumulated_content += delta
                    yield StreamChunk(
                        content=accumulated_content,
                        delta=delta,
                        finish_reason=None,
                    )

                elif event_type == "response.function_call_arguments.done":
                    # Function call complete
                    yield StreamChunk(
                        content=accumulated_content,
                        delta="",
                        finish_reason=None,
                        tool_calls=[
                            {
                                "id": getattr(event, "call_id", ""),
                                "type": "function",
                                "function": {
                                    "name": getattr(event, "name", ""),
                                    "arguments": getattr(event, "arguments", "{}"),
                                },
                            }
                        ],
                    )

                elif event_type == "response.mcp_call.done":
                    # MCP call complete
                    yield StreamChunk(
                        content=accumulated_content,
                        delta="",
                        finish_reason=None,
                        tool_calls=[
                            {
                                "id": getattr(event, "id", ""),
                                "type": "mcp_function",
                                "function": {
                                    "name": getattr(event, "name", ""),
                                    "arguments": getattr(event, "arguments", "{}"),
                                },
                                "server_label": getattr(event, "server_label", None),
                            }
                        ],
                    )

                elif event_type == "response.done":
                    # Response complete
                    response_data = getattr(event, "response", None)
                    usage = None
                    if response_data:
                        usage_data = getattr(response_data, "usage", None)
                        if usage_data:
                            usage = TokenCount(
                                prompt_tokens=getattr(usage_data, "input_tokens", 0),
                                completion_tokens=getattr(usage_data, "output_tokens", 0),
                                total_tokens=getattr(usage_data, "total_tokens", 0),
                            )

                    yield StreamChunk(
                        content=accumulated_content,
                        delta="",
                        finish_reason="stop",
                        usage=usage,
                    )

        except openai.AuthenticationError as e:
            raise AuthenticationError(str(e), self.provider_name, original_error=e)
        except openai.RateLimitError as e:
            retry_after = getattr(e.response.headers, "retry-after", None)
            raise RateLimitError(
                str(e), self.provider_name, retry_after=retry_after, original_error=e
            )
        except openai.BadRequestError as e:
            raise ModelError(str(e), self.model, original_error=e)
        except asyncio.TimeoutError as e:
            raise MiiflowTimeoutError("Streaming request timed out", self.timeout, original_error=e)
        except Exception as e:
            raise ProviderError(
                f"OpenAI Responses API streaming error: {str(e)}",
                self.provider_name,
                original_error=e,
            )
