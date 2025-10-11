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
    """OpenAI streaming format normalizer."""
    
    def normalize(self, chunk: Any) -> StreamChunk:
        """Handle OpenAI's streaming format (GPT-4, GPT-5, GPT-4o)."""
        content = ""
        delta = ""
        finish_reason = None
        tool_calls = None
        usage = None
        
        try:
            if hasattr(chunk, 'choices') and chunk.choices:
                choice = chunk.choices[0]
                
                if hasattr(choice, 'delta') and choice.delta:
                    if hasattr(choice.delta, 'content') and choice.delta.content:
                        delta = choice.delta.content
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


class AnthropicStreamNormalizer(BaseStreamNormalizer):
    """Anthropic streaming format normalizer."""
    
    def normalize(self, chunk: Any) -> StreamChunk:
        """Handle Anthropic's streaming format."""
        content = ""
        delta = ""
        finish_reason = None
        usage = None
        
        try:
            # Anthropic event types
            if hasattr(chunk, 'type'):
                if chunk.type == "content_block_delta":
                    if hasattr(chunk.delta, 'text'):
                        delta = chunk.delta.text
                        content = delta
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
            content = str(chunk) if chunk else ""
            delta = content
        
        return StreamChunk(
            content=content,
            delta=delta,
            finish_reason=finish_reason,
            usage=usage,
            tool_calls=None
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
        usage = None
        
        try:
            if hasattr(chunk, 'choices') and chunk.choices:
                choice = chunk.choices[0]
                
                if hasattr(choice, 'delta') and choice.delta:
                    if hasattr(choice.delta, 'content'):
                        delta = choice.delta.content or ""
                        content = delta
                
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
            tool_calls=None
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