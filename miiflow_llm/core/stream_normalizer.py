"""Stream normalization protocols and base implementations.

This module provides a unified approach to normalizing streaming responses
from different LLM providers into a consistent StreamChunk format.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from .metrics import TokenCount
from .streaming import StreamChunk


@dataclass
class StreamState:
    """Encapsulates streaming state to avoid scattered instance variables.

    This provides a clean container for all streaming state, making it easy
    to reset between streaming sessions and reducing instance variable clutter.
    """

    accumulated_content: str = ""
    accumulated_tool_calls: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    current_tool_use: Optional[Dict[str, Any]] = None
    accumulated_tool_json: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    # MCP-specific state
    current_mcp_tool_use: Optional[Dict[str, Any]] = None
    accumulated_mcp_tool_json: str = ""
    mcp_tool_results: List[Dict[str, Any]] = field(default_factory=list)


class BaseStreamNormalizer(ABC):
    """Abstract base class for stream chunk normalization.

    Each provider client should create a normalizer instance and use it
    to normalize streaming chunks to a unified StreamChunk format.

    Usage:
        normalizer = AnthropicStreamNormalizer(tool_name_mapping)
        normalizer.reset_state()

        async for chunk in provider_stream:
            normalized = normalizer.normalize_chunk(chunk)
            yield normalized
    """

    def __init__(self, tool_name_mapping: Optional[Dict[str, str]] = None):
        """Initialize the normalizer.

        Args:
            tool_name_mapping: Maps sanitized tool names back to original names.
                              Used to restore original tool names that were sanitized
                              for provider compatibility.
        """
        self._state = StreamState()
        self._tool_name_mapping = tool_name_mapping or {}

    def reset_state(self) -> None:
        """Reset streaming state for a new streaming session."""
        self._state = StreamState()

    @abstractmethod
    def normalize_chunk(self, chunk: Any) -> StreamChunk:
        """Normalize a provider-specific chunk to unified StreamChunk format.

        Args:
            chunk: Provider-specific streaming chunk (varies by provider)

        Returns:
            Normalized StreamChunk with consistent structure
        """
        pass

    def _build_chunk(
        self,
        delta: str = "",
        finish_reason: Optional[str] = None,
        usage: Optional[TokenCount] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> StreamChunk:
        """Build a StreamChunk with accumulated content.

        Helper method that handles content accumulation and creates
        the final StreamChunk object.
        """
        if delta:
            self._state.accumulated_content += delta

        return StreamChunk(
            content=self._state.accumulated_content,
            delta=delta,
            finish_reason=finish_reason,
            usage=usage,
            tool_calls=tool_calls,
        )

    def _restore_tool_name(self, sanitized_name: str) -> str:
        """Restore original tool name from sanitized version."""
        return self._tool_name_mapping.get(sanitized_name, sanitized_name)


class OpenAIStreamNormalizer(BaseStreamNormalizer):
    """Stream normalizer for OpenAI-compatible APIs.

    Works with OpenAI, Groq, XAI, OpenRouter, and other OpenAI-compatible providers.
    Handles the choices/delta streaming format with incremental tool call updates.
    """

    def normalize_chunk(self, chunk: Any) -> StreamChunk:
        """Normalize OpenAI streaming format to unified StreamChunk."""
        delta = ""
        finish_reason = None
        tool_calls = None
        usage = None

        try:
            if hasattr(chunk, "choices") and chunk.choices:
                choice = chunk.choices[0]

                if hasattr(choice, "delta") and choice.delta:
                    # Handle text content
                    if hasattr(choice.delta, "content") and choice.delta.content:
                        delta = choice.delta.content

                    # Handle tool call deltas
                    if hasattr(choice.delta, "tool_calls") and choice.delta.tool_calls:
                        tool_calls = self._accumulate_tool_calls(choice.delta.tool_calls)

                if hasattr(choice, "finish_reason"):
                    finish_reason = choice.finish_reason

            usage = self._extract_usage(chunk)

        except AttributeError:
            delta = str(chunk) if chunk else ""

        return self._build_chunk(delta, finish_reason, usage, tool_calls)

    def _accumulate_tool_calls(self, tool_call_deltas: List[Any]) -> List[Dict[str, Any]]:
        """Accumulate tool call deltas into complete tool calls."""
        normalized_tool_calls = []

        for tool_call_delta in tool_call_deltas:
            idx = getattr(tool_call_delta, "index", 0)

            # Initialize accumulator for this index
            if idx not in self._state.accumulated_tool_calls:
                self._state.accumulated_tool_calls[idx] = {
                    "id": None,
                    "type": "function",
                    "function": {"name": None, "arguments": ""},
                }

            # Update ID if present
            if hasattr(tool_call_delta, "id") and tool_call_delta.id:
                self._state.accumulated_tool_calls[idx]["id"] = tool_call_delta.id

            # Update function name and arguments
            if hasattr(tool_call_delta, "function") and tool_call_delta.function:
                if hasattr(tool_call_delta.function, "name") and tool_call_delta.function.name:
                    sanitized_name = tool_call_delta.function.name
                    original_name = self._restore_tool_name(sanitized_name)
                    self._state.accumulated_tool_calls[idx]["function"]["name"] = original_name

                if (
                    hasattr(tool_call_delta.function, "arguments")
                    and tool_call_delta.function.arguments
                ):
                    self._state.accumulated_tool_calls[idx]["function"][
                        "arguments"
                    ] += tool_call_delta.function.arguments

            # Emit the current state as a dict
            normalized_tool_calls.append(self._state.accumulated_tool_calls[idx].copy())

        return normalized_tool_calls

    def _extract_usage(self, chunk: Any) -> Optional[TokenCount]:
        """Extract usage information from chunk."""
        if hasattr(chunk, "usage") and chunk.usage:
            return TokenCount(
                prompt_tokens=chunk.usage.prompt_tokens,
                completion_tokens=chunk.usage.completion_tokens,
                total_tokens=chunk.usage.total_tokens,
            )
        return None


class AnthropicStreamNormalizer(BaseStreamNormalizer):
    """Stream normalizer for Anthropic API.

    Handles the event-based streaming format with content_block_start,
    content_block_delta, and content_block_stop events.
    """

    def normalize_chunk(self, chunk: Any) -> StreamChunk:
        """Normalize Anthropic streaming format to unified StreamChunk.

        Handles both regular tool_use blocks and MCP-specific blocks:
        - mcp_tool_use: Native MCP tool invocation (server-side)
        - mcp_tool_result: Result from MCP tool execution
        """
        delta = ""
        finish_reason = None
        usage = None
        tool_calls = None

        try:
            if hasattr(chunk, "type"):
                if chunk.type == "content_block_start":
                    tool_calls = self._handle_content_block_start(chunk)

                elif chunk.type == "content_block_delta":
                    delta = self._handle_content_block_delta(chunk)

                elif chunk.type == "content_block_stop":
                    tool_calls = self._handle_content_block_stop()

                elif chunk.type == "message_delta":
                    if hasattr(chunk.delta, "stop_reason"):
                        finish_reason = chunk.delta.stop_reason

                elif chunk.type == "message_stop":
                    finish_reason = "stop"

            usage = self._extract_usage(chunk)

        except AttributeError:
            delta = str(chunk) if chunk else ""

        return self._build_chunk(delta, finish_reason, usage, tool_calls)

    def _handle_content_block_start(self, chunk: Any) -> Optional[List[Dict[str, Any]]]:
        """Handle content_block_start event.

        Handles both regular tool_use and MCP-specific blocks:
        - tool_use: Regular tool invocation
        - mcp_tool_use: Native MCP tool invocation
        - mcp_tool_result: Result from MCP tool (content extracted as text)
        """
        if hasattr(chunk, "content_block") and hasattr(chunk.content_block, "type"):
            block_type = chunk.content_block.type

            if block_type == "tool_use":
                tool_name = chunk.content_block.name
                original_name = self._restore_tool_name(tool_name)

                self._state.current_tool_use = {
                    "id": chunk.content_block.id,
                    "type": "function",
                    "function": {"name": original_name, "arguments": {}},
                }
                self._state.accumulated_tool_json = ""

                # Yield tool call immediately
                return [self._state.current_tool_use]

            elif block_type == "mcp_tool_use":
                # Native MCP tool invocation (server-side execution)
                tool_name = getattr(chunk.content_block, "name", "")
                server_name = getattr(chunk.content_block, "server_name", None)

                self._state.current_mcp_tool_use = {
                    "id": getattr(chunk.content_block, "id", ""),
                    "type": "mcp_function",
                    "function": {"name": tool_name, "arguments": {}},
                    "server_name": server_name,
                }
                self._state.accumulated_mcp_tool_json = ""

                # Yield MCP tool call immediately
                return [self._state.current_mcp_tool_use]

            elif block_type == "mcp_tool_result":
                # MCP tool result - store for metadata
                is_error = getattr(chunk.content_block, "is_error", False)
                self._state.mcp_tool_results.append({
                    "tool_use_id": getattr(chunk.content_block, "tool_use_id", ""),
                    "is_error": is_error,
                    "content": "",  # Will be accumulated in delta
                })

        return None

    def _handle_content_block_delta(self, chunk: Any) -> str:
        """Handle content_block_delta event.

        Handles deltas for text, tool_use JSON, MCP tool_use JSON, and MCP tool results.
        """
        if hasattr(chunk.delta, "text"):
            return chunk.delta.text

        if hasattr(chunk.delta, "partial_json"):
            # Handle regular tool_use JSON accumulation
            if self._state.current_tool_use:
                self._state.accumulated_tool_json += chunk.delta.partial_json

                # Try to parse accumulated JSON
                try:
                    self._state.current_tool_use["function"]["arguments"] = json.loads(
                        self._state.accumulated_tool_json
                    )
                except json.JSONDecodeError:
                    # Still accumulating
                    pass

            # Handle MCP tool_use JSON accumulation
            elif self._state.current_mcp_tool_use:
                self._state.accumulated_mcp_tool_json += chunk.delta.partial_json

                # Try to parse accumulated JSON
                try:
                    self._state.current_mcp_tool_use["function"]["arguments"] = json.loads(
                        self._state.accumulated_mcp_tool_json
                    )
                except json.JSONDecodeError:
                    # Still accumulating
                    pass

        # Handle MCP tool result content
        if self._state.mcp_tool_results and hasattr(chunk.delta, "type"):
            if chunk.delta.type == "text" and hasattr(chunk.delta, "text"):
                # Accumulate text content from MCP result
                if self._state.mcp_tool_results:
                    self._state.mcp_tool_results[-1]["content"] += chunk.delta.text
                return chunk.delta.text

        return ""

    def _handle_content_block_stop(self) -> Optional[List[Dict[str, Any]]]:
        """Handle content_block_stop event.

        Handles both regular tool_use and MCP tool_use blocks.
        """
        # Handle regular tool_use block completion
        if self._state.current_tool_use:
            if self._state.accumulated_tool_json:
                try:
                    self._state.current_tool_use["function"]["arguments"] = json.loads(
                        self._state.accumulated_tool_json
                    )
                except json.JSONDecodeError:
                    self._state.current_tool_use["function"]["arguments"] = {}

            self._state.tool_calls.append(self._state.current_tool_use)
            result = [self._state.current_tool_use]

            self._state.current_tool_use = None
            self._state.accumulated_tool_json = ""

            return result

        # Handle MCP tool_use block completion
        if self._state.current_mcp_tool_use:
            if self._state.accumulated_mcp_tool_json:
                try:
                    self._state.current_mcp_tool_use["function"]["arguments"] = json.loads(
                        self._state.accumulated_mcp_tool_json
                    )
                except json.JSONDecodeError:
                    self._state.current_mcp_tool_use["function"]["arguments"] = {}

            self._state.tool_calls.append(self._state.current_mcp_tool_use)
            result = [self._state.current_mcp_tool_use]

            self._state.current_mcp_tool_use = None
            self._state.accumulated_mcp_tool_json = ""

            return result

        return None

    def _extract_usage(self, chunk: Any) -> Optional[TokenCount]:
        """Extract usage information from chunk."""
        if hasattr(chunk, "usage") and chunk.usage is not None:
            # Handle None values from Bedrock which may have usage object but None fields
            input_tokens = getattr(chunk.usage, "input_tokens", None) or 0
            output_tokens = getattr(chunk.usage, "output_tokens", None) or 0
            return TokenCount(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            )
        return None


class GeminiStreamNormalizer(BaseStreamNormalizer):
    """Stream normalizer for Google Gemini API.

    Handles the candidates/parts streaming format.
    """

    def normalize_chunk(self, chunk: Any) -> StreamChunk:
        """Normalize Gemini streaming format to unified StreamChunk."""
        delta = ""
        finish_reason = None
        usage = None

        try:
            if hasattr(chunk, "candidates") and chunk.candidates:
                candidate = chunk.candidates[0]

                if hasattr(candidate, "content") and candidate.content.parts:
                    delta = candidate.content.parts[0].text

                if hasattr(candidate, "finish_reason") and candidate.finish_reason:
                    finish_reason = self._get_finish_reason_name(candidate.finish_reason)

            usage = self._extract_usage(chunk)

        except AttributeError:
            if hasattr(chunk, "text"):
                delta = chunk.text
            else:
                delta = str(chunk) if chunk else ""

        return self._build_chunk(delta, finish_reason, usage, None)

    def _get_finish_reason_name(self, finish_reason: Any) -> Optional[str]:
        """Safely extract finish_reason name.

        Gemini 2.5 models may return new undocumented finish_reason enum values
        (like 12) that come as raw integers instead of enum objects.
        """
        if finish_reason is None:
            return None
        if hasattr(finish_reason, "name"):
            return finish_reason.name
        if isinstance(finish_reason, int):
            return f"UNKNOWN_{finish_reason}"
        return str(finish_reason)

    def _extract_usage(self, chunk: Any) -> Optional[TokenCount]:
        """Extract usage information from chunk."""
        if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
            return TokenCount(
                prompt_tokens=getattr(chunk.usage_metadata, "prompt_token_count", 0) or 0,
                completion_tokens=getattr(chunk.usage_metadata, "candidates_token_count", 0) or 0,
                total_tokens=getattr(chunk.usage_metadata, "total_token_count", 0) or 0,
            )
        return None


class OllamaStreamNormalizer(BaseStreamNormalizer):
    """Stream normalizer for Ollama API.

    Handles the JSON-lines streaming format used by local Ollama models.
    """

    def normalize_chunk(self, chunk: Any) -> StreamChunk:
        """Normalize Ollama streaming format to unified StreamChunk."""
        delta = ""
        finish_reason = None

        try:
            if isinstance(chunk, dict):
                if "message" in chunk:
                    delta = chunk["message"].get("content", "")
                if chunk.get("done", False):
                    finish_reason = "stop"
            elif hasattr(chunk, "message"):
                delta = chunk.message.get("content", "")
                if hasattr(chunk, "done") and chunk.done:
                    finish_reason = "stop"
            else:
                delta = str(chunk) if chunk else ""

        except (AttributeError, TypeError):
            delta = str(chunk) if chunk else ""

        return self._build_chunk(delta, finish_reason, None, None)


class MistralStreamNormalizer(BaseStreamNormalizer):
    """Stream normalizer for Mistral API.

    Handles the OpenAI-like streaming format with some differences.
    """

    def normalize_chunk(self, chunk: Any) -> StreamChunk:
        """Normalize Mistral streaming format to unified StreamChunk."""
        delta = ""
        finish_reason = None
        tool_calls = None
        usage = None

        try:
            if hasattr(chunk, "choices") and chunk.choices:
                choice = chunk.choices[0]

                if hasattr(choice, "delta") and choice.delta:
                    delta = getattr(choice.delta, "content", "") or ""
                    tool_calls = getattr(choice.delta, "tool_calls", None)

                finish_reason = getattr(choice, "finish_reason", None)

            usage = self._extract_usage(chunk)

        except AttributeError:
            delta = str(chunk) if chunk else ""

        return self._build_chunk(delta, finish_reason, usage, tool_calls)

    def _extract_usage(self, chunk: Any) -> Optional[TokenCount]:
        """Extract usage information from chunk."""
        if hasattr(chunk, "usage") and chunk.usage:
            return TokenCount(
                prompt_tokens=chunk.usage.prompt_tokens,
                completion_tokens=chunk.usage.completion_tokens,
                total_tokens=chunk.usage.total_tokens,
            )
        return None
