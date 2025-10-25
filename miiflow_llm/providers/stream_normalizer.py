"""Provider-specific streaming normalization."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from ..core.client import StreamChunk
from ..core.metrics import TokenCount


class BaseStreamNormalizer(ABC):
    """Abstract base class for provider stream normalizers."""
    
    @abstractmethod
    def normalize(self, chunk: Any) -> StreamChunk:
        """Convert provider-specific chunk to unified StreamChunk."""
        pass


class OpenAIStreamNormalizer(BaseStreamNormalizer):
    """OpenAI streaming format normalizer with stateful tool call handling."""

    def __init__(self):
        """Initialize normalizer with state for tool call accumulation."""
        self.accumulated_content = ""
        self.accumulated_tool_calls = {}  # index -> {id, type, function: {name, arguments}}

    def reset(self):
        """Reset normalizer state for a new streaming session."""
        self.accumulated_content = ""
        self.accumulated_tool_calls = {}

    def normalize(self, chunk: Any) -> StreamChunk:
        """Handle OpenAI's streaming format and normalize tool calls to dicts."""
        delta = ""
        finish_reason = None
        tool_calls = None
        usage = None

        try:
            if hasattr(chunk, 'choices') and chunk.choices:
                choice = chunk.choices[0]

                if hasattr(choice, 'delta') and choice.delta:
                    # Handle text content
                    if hasattr(choice.delta, 'content') and choice.delta.content:
                        delta = choice.delta.content
                        self.accumulated_content += delta

                    # Handle tool call deltas - convert to standard dict format
                    if hasattr(choice.delta, 'tool_calls') and choice.delta.tool_calls:
                        normalized_tool_calls = []

                        for tool_call_delta in choice.delta.tool_calls:
                            idx = tool_call_delta.index if hasattr(tool_call_delta, 'index') else 0

                            # Initialize accumulator for this index
                            if idx not in self.accumulated_tool_calls:
                                self.accumulated_tool_calls[idx] = {
                                    'id': None,
                                    'type': 'function',
                                    'function': {'name': None, 'arguments': ''}
                                }

                            # Update ID if present
                            if hasattr(tool_call_delta, 'id') and tool_call_delta.id:
                                self.accumulated_tool_calls[idx]['id'] = tool_call_delta.id

                            # Update function name and arguments
                            if hasattr(tool_call_delta, 'function') and tool_call_delta.function:
                                if hasattr(tool_call_delta.function, 'name') and tool_call_delta.function.name:
                                    self.accumulated_tool_calls[idx]['function']['name'] = tool_call_delta.function.name

                                if hasattr(tool_call_delta.function, 'arguments') and tool_call_delta.function.arguments:
                                    self.accumulated_tool_calls[idx]['function']['arguments'] += tool_call_delta.function.arguments

                            # Emit the current state as a dict
                            normalized_tool_calls.append(self.accumulated_tool_calls[idx].copy())

                        tool_calls = normalized_tool_calls

                if hasattr(choice, 'finish_reason'):
                    finish_reason = choice.finish_reason

            if hasattr(chunk, 'usage') and chunk.usage:
                usage = TokenCount(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens
                )

        except AttributeError:
            delta = str(chunk) if chunk else ""
            self.accumulated_content += delta

        return StreamChunk(
            content=self.accumulated_content,
            delta=delta,
            finish_reason=finish_reason,
            usage=usage,
            tool_calls=tool_calls
        )


class AnthropicStreamNormalizer(BaseStreamNormalizer):
    """Anthropic streaming format normalizer with stateful tool call handling."""

    def __init__(self):
        """Initialize normalizer with state for tool call accumulation."""
        self.accumulated_content = ""
        self.current_tool_use = None
        self.accumulated_tool_json = ""
        self.tool_calls = []
        self.tool_name_mapping = {}  # For sanitized -> original name mapping

    def reset(self):
        """Reset normalizer state for a new streaming session."""
        self.accumulated_content = ""
        self.current_tool_use = None
        self.accumulated_tool_json = ""
        self.tool_calls = []

    def set_tool_name_mapping(self, mapping: dict):
        """Set the tool name mapping from sanitized to original names."""
        self.tool_name_mapping = mapping

    def normalize(self, chunk: Any) -> StreamChunk:
        """Handle Anthropic's streaming format with tool calls."""
        delta = ""
        finish_reason = None
        usage = None
        tool_calls = None

        try:
            # Anthropic event types
            if hasattr(chunk, 'type'):
                if chunk.type == "content_block_start":
                    # Check if this is a tool use block
                    if hasattr(chunk, 'content_block') and hasattr(chunk.content_block, 'type'):
                        if chunk.content_block.type == 'tool_use':
                            # Restore original tool name if it was sanitized
                            tool_name = chunk.content_block.name
                            original_name = self.tool_name_mapping.get(tool_name, tool_name)

                            self.current_tool_use = {
                                "id": chunk.content_block.id,
                                "type": "function",
                                "function": {
                                    "name": original_name,
                                    "arguments": {}
                                }
                            }
                            self.accumulated_tool_json = ""

                            # Yield tool call immediately
                            tool_calls = [self.current_tool_use]

                elif chunk.type == "content_block_delta":
                    if hasattr(chunk.delta, 'text'):
                        # Text content
                        delta = chunk.delta.text
                        self.accumulated_content += delta

                    elif hasattr(chunk.delta, 'partial_json'):
                        # Tool use input (streaming JSON)
                        if self.current_tool_use:
                            self.accumulated_tool_json += chunk.delta.partial_json

                            # Try to parse accumulated JSON
                            import json
                            try:
                                self.current_tool_use["function"]["arguments"] = json.loads(self.accumulated_tool_json)
                            except json.JSONDecodeError:
                                # Still accumulating
                                pass

                elif chunk.type == "content_block_stop":
                    # Finalize tool use if present
                    if self.current_tool_use:
                        if self.accumulated_tool_json:
                            import json
                            try:
                                self.current_tool_use["function"]["arguments"] = json.loads(self.accumulated_tool_json)
                            except json.JSONDecodeError:
                                self.current_tool_use["function"]["arguments"] = {}

                        self.tool_calls.append(self.current_tool_use)
                        # Yield complete tool call
                        tool_calls = [self.current_tool_use]

                        self.current_tool_use = None
                        self.accumulated_tool_json = ""

                elif chunk.type == "message_delta":
                    if hasattr(chunk.delta, 'stop_reason'):
                        finish_reason = chunk.delta.stop_reason

                elif chunk.type == "message_stop":
                    finish_reason = "stop"

            if hasattr(chunk, 'usage'):
                usage = TokenCount(
                    prompt_tokens=getattr(chunk.usage, 'input_tokens', 0),
                    completion_tokens=getattr(chunk.usage, 'output_tokens', 0),
                    total_tokens=getattr(chunk.usage, 'input_tokens', 0) + getattr(chunk.usage, 'output_tokens', 0)
                )

        except AttributeError:
            delta = str(chunk) if chunk else ""
            self.accumulated_content += delta

        return StreamChunk(
            content=self.accumulated_content,
            delta=delta,
            finish_reason=finish_reason,
            usage=usage,
            tool_calls=tool_calls
        )


class GroqStreamNormalizer(BaseStreamNormalizer):
    """Groq streaming format normalizer."""
    
    def normalize(self, chunk: Any) -> StreamChunk:
        """Handle Groq's streaming format (OpenAI-compatible)."""
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


class GeminiStreamNormalizer(BaseStreamNormalizer):
    """Google Gemini streaming format normalizer."""
    
    def normalize(self, chunk: Any) -> StreamChunk:
        """Handle Google Gemini's streaming format."""
        content = ""
        delta = ""
        finish_reason = None
        usage = None
        
        try:
            if hasattr(chunk, 'candidates') and chunk.candidates:
                candidate = chunk.candidates[0]
                
                if hasattr(candidate, 'content') and candidate.content.parts:
                    delta = candidate.content.parts[0].text
                    content = delta
                
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                    finish_reason = candidate.finish_reason.name
            
            if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                usage = TokenCount(
                    prompt_tokens=getattr(chunk.usage_metadata, 'prompt_token_count', 0) or 0,
                    completion_tokens=getattr(chunk.usage_metadata, 'candidates_token_count', 0) or 0,
                    total_tokens=getattr(chunk.usage_metadata, 'total_token_count', 0) or 0
                )
                
        except AttributeError:
            if hasattr(chunk, 'text'):
                content = chunk.text
                delta = content
            else:
                content = str(chunk) if chunk else ""
                delta = content
        
        return StreamChunk(
            content=content,
            delta=delta,
            finish_reason=finish_reason,
            usage=usage,
            tool_calls=None
        )


class MistralStreamNormalizer(BaseStreamNormalizer):
    """Mistral streaming format normalizer."""

    def normalize(self, chunk: Any) -> StreamChunk:
        """Handle Mistral's streaming format (OpenAI-compatible)."""
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
                    # Extract tool calls from streaming delta
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


class OllamaStreamNormalizer(BaseStreamNormalizer):
    """Ollama streaming format normalizer."""
    
    def normalize(self, chunk: Any) -> StreamChunk:
        """Handle Ollama's streaming format."""
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
            usage=None,  # Ollama doesn't provide detailed usage in streams
            tool_calls=None
        )


# OpenAI-compatible providers can reuse the same normalizer
class TogetherStreamNormalizer(OpenAIStreamNormalizer):
    """TogetherAI uses OpenAI-compatible format."""
    pass


class OpenRouterStreamNormalizer(OpenAIStreamNormalizer):
    """OpenRouter uses OpenAI-compatible format."""
    pass


class XAIStreamNormalizer(OpenAIStreamNormalizer):
    """XAI uses OpenAI-compatible format."""
    pass


# Registry for easy provider-specific normalizer lookup
STREAM_NORMALIZERS = {
    "openai": OpenAIStreamNormalizer,
    "anthropic": AnthropicStreamNormalizer,
    "groq": GroqStreamNormalizer,
    "gemini": GeminiStreamNormalizer,
    "mistral": MistralStreamNormalizer,
    "ollama": OllamaStreamNormalizer,
    "together": TogetherStreamNormalizer,
    "openrouter": OpenRouterStreamNormalizer,
    "xai": XAIStreamNormalizer,
}


def get_stream_normalizer(provider: str) -> BaseStreamNormalizer:
    """Get appropriate stream normalizer for provider."""
    normalizer_class = STREAM_NORMALIZERS.get(provider.lower())
    if not normalizer_class:
        # Fallback to OpenAI-compatible format
        normalizer_class = OpenAIStreamNormalizer
    
    return normalizer_class()