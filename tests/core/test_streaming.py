"""Fixed tests for streaming normalization and structured output."""

import pytest
from unittest.mock import MagicMock, AsyncMock
import json

from miiflow_llm.core.streaming import (
    IncrementalParser,
    UnifiedStreamingClient,
    EnhancedStreamChunk
)
from miiflow_llm.providers.openai_client import OpenAIClient
from miiflow_llm.providers.anthropic_client import AnthropicClient
from miiflow_llm.core import Message, TokenCount, MessageRole
from miiflow_llm.core.streaming import StreamChunk


class TestStreamNormalizers:
    """Test suite for stream normalizers (now inline in clients)."""

    @pytest.fixture
    def openai_client(self):
        """Create OpenAI client instance."""
        return OpenAIClient(model="gpt-4", api_key="test_key")

    @pytest.fixture
    def anthropic_client(self):
        """Create Anthropic client instance."""
        return AnthropicClient(model="claude-3-5-sonnet-20241022", api_key="test_key")

    def test_openai_chunk_normalization(self, openai_client):
        """Test OpenAI chunk normalization."""
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta = MagicMock()
        chunk.choices[0].delta.content = "Hello world"
        chunk.choices[0].finish_reason = None
        chunk.usage = None

        result = openai_client._normalize_stream_chunk(chunk)

        assert isinstance(result, StreamChunk)
        assert result.content == "Hello world"
        assert result.delta == "Hello world"
        assert result.finish_reason is None

    def test_openai_final_chunk(self, openai_client):
        """Test OpenAI final chunk with usage."""
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta = MagicMock()
        chunk.choices[0].delta.content = ""
        chunk.choices[0].finish_reason = "stop"
        chunk.usage = MagicMock()
        chunk.usage.prompt_tokens = 10
        chunk.usage.completion_tokens = 20
        chunk.usage.total_tokens = 30

        result = openai_client._normalize_stream_chunk(chunk)

        assert result.finish_reason == "stop"
        assert result.usage is not None
        assert result.usage.total_tokens == 30

    def test_anthropic_chunk_normalization(self, anthropic_client):
        """Test Anthropic chunk normalization."""
        chunk = MagicMock()
        chunk.type = "content_block_delta"
        chunk.delta = MagicMock()
        chunk.delta.text = "Anthropic response"

        result = anthropic_client._normalize_stream_chunk(chunk)

        assert isinstance(result, StreamChunk)
        assert result.content == "Anthropic response"
        assert result.delta == "Anthropic response"

    def test_anthropic_stop_chunk(self, anthropic_client):
        """Test Anthropic stop chunk."""
        chunk = MagicMock()
        chunk.type = "message_stop"

        result = anthropic_client._normalize_stream_chunk(chunk)

        assert result.finish_reason == "stop"

    def test_groq_chunk_normalization(self, openai_client):
        """Test Groq chunk normalization (uses OpenAI mixin)."""
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta = MagicMock()
        chunk.choices[0].delta.content = "Groq response"
        chunk.choices[0].finish_reason = None
        chunk.usage = None

        result = openai_client._normalize_stream_chunk(chunk)

        assert isinstance(result, StreamChunk)
        assert result.content == "Groq response"
        assert result.delta == "Groq response"


class TestIncrementalParser:
    """Test suite for IncrementalParser."""
    
    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return IncrementalParser()
    
    def test_complete_json_parsing(self, parser):
        """Test parsing complete JSON object."""
        json_text = '{"name": "test", "value": 123}'
        
        result = parser.try_parse_partial(json_text)
        
        assert result is not None
        assert result["name"] == "test"
        assert result["value"] == 123
    
    def test_incremental_json_parsing(self, parser):
        """Test incremental JSON parsing."""
        # First partial content
        result1 = parser.try_parse_partial('{"name": "test"')
        # Parser may attempt to complete partial JSON, so check if it's reasonable
        if result1 is not None:
            assert "name" in result1
        
        # Complete the JSON
        result2 = parser.try_parse_partial(', "value": 123}')
        assert result2 is not None
        assert "name" in result2
        assert "value" in result2
    
    def test_multiple_json_objects(self, parser):
        """Test parsing multiple JSON objects."""
        text = '{"first": 1} {"second": 2}'
        
        result = parser.try_parse_partial(text)
        
        # Should extract the complete objects
        complete_objects = parser._extract_complete_json_objects(text)
        assert len(complete_objects) == 2
        assert complete_objects[0]["first"] == 1
        assert complete_objects[1]["second"] == 2
    
    def test_finalize_parse_strategies(self, parser):
        """Test different finalization strategies."""
        # Strategy 1: Direct JSON
        result1 = parser.finalize_parse('{"valid": "json"}')
        assert result1["valid"] == "json"
        
        # Strategy 2: Extract from mixed text
        result2 = parser.finalize_parse('Some text {"embedded": "json"} more text')
        assert result2["embedded"] == "json"
    
    def test_malformed_json_handling(self, parser):
        """Test handling of malformed JSON."""
        # Should not crash on malformed JSON
        result = parser.try_parse_partial('{"malformed": json}')
        assert result is None or isinstance(result, dict)


class TestUnifiedStreamingClient:
    """Test suite for UnifiedStreamingClient."""
    
    @pytest.fixture
    def mock_client(self):
        """Create mock client."""
        client = MagicMock()
        client.provider_name = "openai"
        return client
    
    @pytest.fixture
    def unified_client(self, mock_client):
        """Create unified streaming client."""
        return UnifiedStreamingClient(mock_client)
    
    @pytest.fixture
    def sample_messages(self):
        """Create sample messages."""
        return [
            Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
            Message(role=MessageRole.USER, content="Hello!")
        ]
    
    @pytest.mark.asyncio
    async def test_stream_with_schema(self, unified_client, mock_client, sample_messages):
        """Test streaming with schema parsing."""
        # Mock streaming chunks
        async def mock_stream(messages, **kwargs):
            chunk = StreamChunk(
                content='{"test": "value"}',
                delta='{"test": "value"}',
                finish_reason="stop",
                usage=None,
                tool_calls=None
            )
            yield chunk
        
        mock_client.astream_chat = mock_stream
        
        chunks = []
        async for chunk in unified_client.stream_with_schema(sample_messages):
            chunks.append(chunk)
        
        assert len(chunks) >= 1
        # Should have proper metadata
        assert chunks[0].metadata["provider"] == "openai"
    
    @pytest.mark.asyncio
    async def test_error_handling_in_stream(self, unified_client, mock_client, sample_messages):
        """Test error handling during streaming."""
        async def mock_failing_stream(messages, **kwargs):
            # Async generator that fails after first iteration
            if False:  # Never executes, but makes it an async generator
                yield
            raise RuntimeError("Stream failed")
        
        mock_client.astream_chat = mock_failing_stream
        
        with pytest.raises(RuntimeError):
            chunks = []
            async for chunk in unified_client.stream_with_schema(sample_messages):
                chunks.append(chunk)


if __name__ == "__main__":
    pytest.main([__file__])