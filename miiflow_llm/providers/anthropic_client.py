"""Anthropic provider implementation."""

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from ..core.client import ChatResponse, ModelClient
from ..core.exceptions import AuthenticationError, ModelError, ProviderError, RateLimitError
from ..core.exceptions import TimeoutError as MiiflowTimeoutError
from ..core.message import Message, MessageRole
from ..core.metrics import TokenCount, UsageData
from ..core.schema_normalizer import SchemaMode, normalize_json_schema
from ..core.stream_normalizer import AnthropicStreamNormalizer
from ..core.streaming import StreamChunk
from ..models.anthropic import supports_native_mcp, supports_structured_outputs
from ..utils.image import data_uri_to_base64_and_mimetype

if TYPE_CHECKING:
    from ..core.tools.mcp import NativeMCPServerConfig

logger = logging.getLogger(__name__)


class AnthropicClient(ModelClient):
    """Anthropic provider client."""

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(model=model, api_key=api_key, **kwargs)
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.provider_name = "anthropic"
        self._tool_name_mapping: Dict[str, str] = {}

        # Stream normalizer for unified streaming handling
        # Note: Pass instance-level mapping for tool name restoration
        self._stream_normalizer = AnthropicStreamNormalizer(self._tool_name_mapping)

    def _supports_structured_outputs(self) -> bool:
        """Check if the current model supports native structured outputs."""
        return supports_structured_outputs(self.model)

    def convert_schema_to_provider_format(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert universal schema to Anthropic format with loosened constraints.

        Supports strict mode for models that support structured outputs:
        - If model supports structured outputs and schema has 'strict': True,
          use native strict mode without loosening schema
        - Otherwise, loosen schema for better compatibility
        """
        import re

        original_name = schema["name"]
        sanitized_name = re.sub(r"[^a-zA-Z0-9_-]", "_", original_name)
        sanitized_name = re.sub(r"_+", "_", sanitized_name).strip("_")[:128]

        if sanitized_name != original_name:
            self._tool_name_mapping[sanitized_name] = original_name

        # Check if strict mode is requested and supported
        # Check both top-level 'strict' and metadata['strict']
        strict_flag = schema.get("strict", False) or schema.get("metadata", {}).get("strict", False)
        use_strict = strict_flag and self._supports_structured_outputs()

        tool_definition = {
            "name": sanitized_name,
            "description": schema["description"],
            "input_schema": (
                normalize_json_schema(schema["parameters"], SchemaMode.NATIVE_STRICT)
                if use_strict
                else normalize_json_schema(schema["parameters"], SchemaMode.LOOSE)
            ),
        }

        # Add strict flag if using strict mode
        if use_strict:
            tool_definition["strict"] = True

        return tool_definition

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
                        # Fixed: url should be a string, not a dict
                        content_list.append(
                            {
                                "type": "document",
                                "source": {
                                    "type": "url",
                                    "url": block.document_url,
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

    def _supports_native_mcp(self) -> bool:
        """Check if the current model supports native MCP via beta API."""
        return supports_native_mcp(self.model)

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
        """Send chat completion request to Anthropic.

        Supports:
        1. Native MCP (for all Claude models via beta API):
           Uses mcp_servers parameter for server-side tool execution
        2. Native structured outputs (for supported models like Claude Sonnet 4.5):
           Uses output_format parameter with guaranteed schema compliance
        3. Tool-based workaround (for older models):
           Uses a synthetic tool to force JSON output
        """
        try:

            system_content, anthropic_messages = self._prepare_messages(messages)

            # Check for native MCP
            use_native_mcp = mcp_servers and len(mcp_servers) > 0 and self._supports_native_mcp()

            # Handle JSON schema
            json_tool_name = None
            use_native_structured_output = json_schema and self._supports_structured_outputs()

            if json_schema:
                if use_native_structured_output:
                    # Use native structured output API (beta feature)
                    # Prepare schema by ensuring additionalProperties is set
                    prepared_schema = normalize_json_schema(json_schema, SchemaMode.NATIVE_STRICT)

                    request_params = {
                        "model": self.model,
                        "messages": anthropic_messages,
                        "temperature": temperature,
                        "betas": ["structured-outputs-2025-11-13"],
                        "output_format": {
                            "type": "json_schema",
                            "schema": prepared_schema,
                        },
                        **kwargs,
                    }

                    # Anthropic requires max_tokens, use 4096 as default if not specified
                    if max_tokens is not None:
                        request_params["max_tokens"] = max_tokens
                    else:
                        request_params["max_tokens"] = 4096

                    logger.debug(f"Using native structured output API for model {self.model}")
                else:
                    # Fall back to tool-based approach for older models
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
                        **kwargs,
                    }

                    # Anthropic requires max_tokens, use 4096 as default if not specified
                    if max_tokens is not None:
                        request_params["max_tokens"] = max_tokens
                    else:
                        request_params["max_tokens"] = 4096

                    logger.debug(f"Using tool-based JSON output for model {self.model}")
            else:
                # Regular request without JSON schema
                request_params = {
                    "model": self.model,
                    "messages": anthropic_messages,
                    "temperature": temperature,
                    **kwargs,
                }

                # Anthropic requires max_tokens, use 4096 as default if not specified
                if max_tokens is not None:
                    request_params["max_tokens"] = max_tokens
                else:
                    request_params["max_tokens"] = 4096

            if system_content:
                request_params["system"] = system_content
            if tools:
                request_params["tools"] = tools
                logger.debug(
                    f"Anthropic tools parameter:\n{json.dumps(tools, indent=2, default=str)}"
                )

            # Handle native MCP servers
            if use_native_mcp:
                # Convert NativeMCPServerConfig to Anthropic format
                anthropic_mcp_servers = [
                    server.to_anthropic_format() for server in mcp_servers
                ]
                request_params["mcp_servers"] = anthropic_mcp_servers

                # Add MCP beta header (combine with existing betas if any)
                betas = request_params.get("betas", [])
                if isinstance(betas, list):
                    betas = list(betas)  # Make a copy
                else:
                    betas = []
                if "mcp-client-2025-04-04" not in betas:
                    betas.append("mcp-client-2025-04-04")
                request_params["betas"] = betas

                logger.debug(
                    f"Using native MCP with {len(mcp_servers)} servers: "
                    f"{[s.name for s in mcp_servers]}"
                )

            # Determine which client to use
            # Use beta client for structured outputs or native MCP
            use_beta_client = use_native_structured_output or use_native_mcp
            if use_beta_client:
                response = await asyncio.wait_for(
                    self.client.beta.messages.create(**request_params), timeout=self.timeout
                )
            else:
                response = await asyncio.wait_for(
                    self.client.messages.create(**request_params), timeout=self.timeout
                )

            # Extract content and tool calls from response
            content = ""
            tool_calls = []
            mcp_tool_results = []  # Track MCP tool results for metadata

            if response.content:
                for block in response.content:
                    if hasattr(block, "text"):
                        content += block.text
                    elif hasattr(block, "type") and block.type == "tool_use":
                        if json_tool_name and block.name == json_tool_name:
                            # Extract JSON from tool response (fallback mode)
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
                    elif hasattr(block, "type") and block.type == "mcp_tool_use":
                        # Native MCP tool call (server-side execution)
                        # These are handled by Anthropic's API directly
                        tool_calls.append(
                            {
                                "id": getattr(block, "id", ""),
                                "type": "mcp_function",
                                "function": {
                                    "name": getattr(block, "name", ""),
                                    "arguments": getattr(block, "input", {}),
                                },
                                "server_name": getattr(block, "server_name", None),
                            }
                        )
                        logger.debug(f"MCP tool use: {getattr(block, 'name', 'unknown')}")
                    elif hasattr(block, "type") and block.type == "mcp_tool_result":
                        # Native MCP tool result (already executed by API)
                        is_error = getattr(block, "is_error", False)
                        result_content = getattr(block, "content", "")

                        # Extract text content from result
                        if isinstance(result_content, list):
                            # Content is a list of blocks
                            for item in result_content:
                                if hasattr(item, "text"):
                                    content += item.text
                                elif isinstance(item, dict) and "text" in item:
                                    content += item["text"]
                        elif isinstance(result_content, str):
                            content += result_content

                        mcp_tool_results.append({
                            "tool_use_id": getattr(block, "tool_use_id", ""),
                            "is_error": is_error,
                            "content": result_content,
                        })

                        if is_error:
                            logger.warning(f"MCP tool error: {result_content}")
                        else:
                            logger.debug(f"MCP tool result received")

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

            # Build metadata
            metadata = {"response_id": response.id}
            if mcp_tool_results:
                metadata["mcp_tool_results"] = mcp_tool_results
            if use_native_mcp:
                metadata["native_mcp"] = True

            return ChatResponse(
                message=response_message,
                usage=usage,
                model=self.model,
                provider=self.provider_name,
                finish_reason=response.stop_reason,
                metadata=metadata,
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
        mcp_servers: Optional[List["NativeMCPServerConfig"]] = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Send streaming chat completion request to Anthropic.

        Supports:
        1. Native MCP (for all Claude models via beta API):
           Uses mcp_servers parameter for server-side tool execution
        2. Native structured outputs (for supported models like Claude Sonnet 4.5):
           Uses output_format parameter with guaranteed schema compliance (streaming supported!)
        3. Tool-based workaround (for older models):
           Uses a synthetic tool to force JSON output
        """
        import json
        import logging

        logger = logging.getLogger(__name__)

        try:
            system_content, anthropic_messages = self._prepare_messages(messages)

            logger.debug(f"Streaming request to Anthropic with {len(anthropic_messages)} messages:")
            for idx, msg in enumerate(anthropic_messages):
                logger.debug(
                    f"  Message {idx}: role={msg.get('role')}, content_type={type(msg.get('content'))}, content_length={len(str(msg.get('content')))}"
                )
                logger.debug(
                    f"    Content preview: {json.dumps(msg.get('content'), default=str)[:200]}"
                )

            # Check for native MCP
            use_native_mcp = mcp_servers and len(mcp_servers) > 0 and self._supports_native_mcp()

            # Handle JSON schema
            json_tool_name = None
            use_native_structured_output = json_schema and self._supports_structured_outputs()

            if json_schema:
                if use_native_structured_output:
                    # Use native structured output API (beta feature) - streaming supported!
                    # Prepare schema by ensuring additionalProperties is set
                    prepared_schema = normalize_json_schema(json_schema, SchemaMode.NATIVE_STRICT)

                    request_params = {
                        "model": self.model,
                        "messages": anthropic_messages,
                        "temperature": temperature,
                        "stream": True,
                        "betas": ["structured-outputs-2025-11-13"],
                        "output_format": {
                            "type": "json_schema",
                            "schema": prepared_schema,
                        },
                        **kwargs,
                    }

                    # Anthropic requires max_tokens, use 4096 as default if not specified
                    if max_tokens is not None:
                        request_params["max_tokens"] = max_tokens
                    else:
                        request_params["max_tokens"] = 4096

                    logger.debug(
                        f"Using native structured output API with streaming for model {self.model}"
                    )
                else:
                    # Fall back to tool-based approach for older models
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
                        "stream": True,
                        **kwargs,
                    }

                    # Anthropic requires max_tokens, use 4096 as default if not specified
                    if max_tokens is not None:
                        request_params["max_tokens"] = max_tokens
                    else:
                        request_params["max_tokens"] = 4096

                    logger.debug(
                        f"Using tool-based JSON output with streaming for model {self.model}"
                    )
            else:
                # Regular streaming without JSON schema
                request_params = {
                    "model": self.model,
                    "messages": anthropic_messages,
                    "temperature": temperature,
                    "stream": True,
                    **kwargs,
                }

                # Anthropic requires max_tokens, use 4096 as default if not specified
                if max_tokens is not None:
                    request_params["max_tokens"] = max_tokens
                else:
                    request_params["max_tokens"] = 4096

            if system_content:
                request_params["system"] = system_content
            if tools:
                request_params["tools"] = tools

            # Handle native MCP servers
            if use_native_mcp:
                # Convert NativeMCPServerConfig to Anthropic format
                anthropic_mcp_servers = [
                    server.to_anthropic_format() for server in mcp_servers
                ]
                request_params["mcp_servers"] = anthropic_mcp_servers

                # Add MCP beta header (combine with existing betas if any)
                betas = request_params.get("betas", [])
                if isinstance(betas, list):
                    betas = list(betas)  # Make a copy
                else:
                    betas = []
                if "mcp-client-2025-04-04" not in betas:
                    betas.append("mcp-client-2025-04-04")
                request_params["betas"] = betas

                logger.debug(
                    f"Streaming with native MCP: {len(mcp_servers)} servers"
                )

            # Determine which client to use
            # Use beta client for structured outputs or native MCP
            use_beta_client = use_native_structured_output or use_native_mcp
            if use_beta_client:
                stream = await asyncio.wait_for(
                    self.client.beta.messages.create(**request_params), timeout=self.timeout
                )
            else:
                stream = await asyncio.wait_for(
                    self.client.messages.create(**request_params), timeout=self.timeout
                )

            # Reset stream state for new streaming session
            self._stream_normalizer.reset_state()

            async for event in stream:
                # Normalize Anthropic events to StreamChunk
                normalized_chunk = self._stream_normalizer.normalize_chunk(event)

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
