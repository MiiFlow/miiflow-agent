"""Tests for streaming normalization and structured output."""

import pytest
from unittest.mock import MagicMock, AsyncMock
import json

from miiflow_llm.core.streaming import (
    ProviderStreamNormalizer,
    IncrementalParser,
    UnifiedStreamingClient,
    StreamContent,
    EnhancedStreamChunk
)
from miiflow_llm.core import Message, TokenCount


class TestProviderStreamNormalizer:
    """Test suite for ProviderStreamNormalizer."""
    
    @pytest.fixture
    def normalizer(self):
        """Create normalizer instance."""
        return ProviderStreamNormalizer()
    
    def test_openai_chunk_normalization(self, normalizer):
        """Test OpenAI chunk normalization."""
        # Mock OpenAI chunk (GPT-5 style)
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "Hello world"
        chunk.choices[0].finish_reason = None
        
        result = normalizer.normalize_chunk(chunk, "openai")
        
        assert isinstance(result, StreamContent)
        assert result.content == "Hello world"
        assert result.is_delta is True
        assert result.is_complete is False
        assert result.metadata["provider"] == "openai"
    
    def test_openai_final_chunk(self, normalizer):
        """Test OpenAI final chunk with usage."""
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = None
        chunk.choices[0].finish_reason = "stop"
        chunk.usage.prompt_tokens = 10
        chunk.usage.completion_tokens = 20
        chunk.usage.total_tokens = 30
        
        result = normalizer.normalize_chunk(chunk, "openai")
        
        assert result.content == ""
        assert result.is_complete is True
        assert result.finish_reason == "stop"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 20
    
    def test_anthropic_chunk_normalization(self, normalizer):
        """Test Anthropic chunk normalization."""
        # Mock Anthropic content block delta
        chunk = MagicMock()
        chunk.type = "content_block_delta"
        
        # Configure mock to NOT have a delta attribute directly  
        del chunk.delta  # Remove the auto-created delta
        chunk.content_block_delta.delta.text = "Hello from Claude"
        
        result = normalizer.normalize_chunk(chunk, "anthropic")
        
        assert result.content == "Hello from Claude"
        assert result.is_delta is True
        assert result.metadata["provider"] == "anthropic"
    
    def test_anthropic_stop_chunk(self, normalizer):
        """Test Anthropic stop chunk."""
        chunk = MagicMock()
        chunk.type = "message_stop"
        
        result = normalizer.normalize_chunk(chunk, "anthropic")
        
        assert result.is_complete is True
        assert result.finish_reason == "stop"
    
    def test_groq_chunk_normalization(self, normalizer):
        """Test Groq chunk normalization."""
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "Groq response"
        chunk.choices[0].finish_reason = None
        
        result = normalizer.normalize_chunk(chunk, "groq")
        
        assert result.content == "Groq response"
        assert result.metadata["provider"] == "groq"
    
    def test_groq_string_fallback(self, normalizer):
        """Test Groq string fallback."""
        # Test with a chunk that will fall back to generic normalization
        chunk = MagicMock()
        # Don't set up the expected attributes, so it falls back to generic
        
        result = normalizer.normalize_chunk(chunk, "groq")
        
        # Should use generic fallback and convert mock to string
        assert result.is_delta is True
        # Content will be string representation of mock
    
    def test_gemini_chunk_normalization(self, normalizer):
        """Test Gemini chunk normalization."""
        chunk = MagicMock()
        
        # Use proper mock configuration
        chunk.candidates = [MagicMock()]
        chunk.candidates[0].content.parts = [MagicMock()]
        chunk.candidates[0].content.parts[0].text = "Gemini says hello"
        chunk.candidates[0].finish_reason = None
        
        result = normalizer.normalize_chunk(chunk, "gemini")
        
        assert result.content == "Gemini says hello"
        assert result.metadata["provider"] == "gemini"
    
    def test_gemini_with_usage(self, normalizer):
        """Test Gemini chunk with usage metadata."""
        chunk = MagicMock()
        
        # Use proper mock configuration
        chunk.candidates = [MagicMock()]
        chunk.candidates[0].content.parts = [MagicMock()]
        chunk.candidates[0].content.parts[0].text = "Final response"
        chunk.candidates[0].finish_reason.name = "STOP"
        
        chunk.usage_metadata.prompt_token_count = 15
        chunk.usage_metadata.candidates_token_count = 25
        chunk.usage_metadata.total_token_count = 40
        
        result = normalizer.normalize_chunk(chunk, "gemini")
        
        assert result.finish_reason == "STOP"
        assert result.usage.prompt_tokens == 15
        assert result.usage.completion_tokens == 25
        assert result.usage.total_tokens == 40
    
    def test_ollama_chunk_normalization(self, normalizer):
        """Test Ollama chunk normalization."""
        # Ollama uses dict format
        chunk = {
            "message": {"content": "Local LLM response"},
            "done": False
        }
        
        result = normalizer.normalize_chunk(chunk, "ollama")
        
        assert result.content == "Local LLM response"
        assert result.is_complete is False
        assert result.metadata["provider"] == "ollama"
    
    def test_ollama_final_chunk(self, normalizer):
        """Test Ollama final chunk."""
        chunk = {
            "message": {"content": ""},
            "done": True
        }
        
        result = normalizer.normalize_chunk(chunk, "ollama")
        
        assert result.is_complete is True
        assert result.finish_reason == "stop"
    
    def test_generic_fallback(self, normalizer):
        """Test generic fallback normalization."""
        chunk = MagicMock()
        chunk.content = "Fallback content"
        
        result = normalizer.normalize_chunk(chunk, "unknown_provider")
        
        assert result.content == "Fallback content"
        assert result.metadata["provider"] == "generic"
    
    def test_provider_routing(self, normalizer):
        """Test that providers are routed correctly."""
        providers = ["openai", "anthropic", "groq", "together", "xai", 
                    "gemini", "openrouter", "mistral", "ollama"]
        
        chunk = MagicMock()
        chunk.content = "test"
        
        for provider in providers:
            result = normalizer.normalize_chunk(chunk, provider)
            assert isinstance(result, StreamContent)


class TestIncrementalParser:
    """Test suite for IncrementalParser."""
    
    def test_complete_json_parsing(self):
        """Test parsing complete JSON objects."""
        parser = IncrementalParser()
        
        # Add complete JSON
        result = parser.try_parse_partial('{"name": "test", "value": 42}')
        
        assert result is not None
        assert result["name"] == "test"
        assert result["value"] == 42
    
    def test_incremental_json_parsing(self):
        """Test incremental JSON parsing."""
        parser = IncrementalParser()
        
        # Partial JSON
        result1 = parser.try_parse_partial('{"name": "test"')
        assert result1 is None or "name" in result1
        
        # Complete it
        result2 = parser.try_parse_partial(', "value": 42}')
        assert result2 is not None
        assert result2["value"] == 42
    
    def test_multiple_json_objects(self):
        """Test extracting multiple JSON objects."""
        parser = IncrementalParser()
        
        text = '{"first": 1} some text {"second": 2}'
        result = parser.try_parse_partial(text)
        
        assert result is not None
        # Should get the most recent complete object
        assert result["second"] == 2
    
    def test_finalize_parse_strategies(self):
        """Test different finalization strategies."""
        parser = IncrementalParser()
        
        # Strategy 1: Direct JSON
        result1 = parser.finalize_parse('{"direct": true}')
        assert result1["direct"] is True
        
        # Strategy 2: Extract from text  
        parser2 = IncrementalParser()
        result2 = parser2.finalize_parse('Here is the JSON: {"extracted": true} and more text')
        assert result2["extracted"] is True
        
        # Strategy 3: Regex fallback
        parser3 = IncrementalParser()
        result3 = parser3.finalize_parse('"name": "value", "count": 5')
        assert result3["name"] == "value"
        assert result3["count"] == 5
    
    def test_malformed_json_handling(self):
        """Test handling of malformed JSON."""
        parser = IncrementalParser()
        
        # Should not crash on malformed JSON
        result = parser.try_parse_partial('{"broken": "json"')
        # May return None or partial result, but shouldn't crash
        
        # Finalize should try fallback strategies
        final = parser.finalize_parse('{"broken": "json", "incomplete"')
        # Should extract what it can or return None gracefully


class TestUnifiedStreamingClient:
    """Test suite for UnifiedStreamingClient."""
    
    @pytest.fixture
    def mock_client(self):
        """Create mock client."""
        client = MagicMock()
        client.provider_name = "test"
        return client
    
    @pytest.fixture
    def unified_client(self, mock_client):
        """Create unified streaming client."""
        return UnifiedStreamingClient(mock_client)
    
    @pytest.mark.asyncio
    async def test_stream_with_schema(self, unified_client, mock_client, sample_messages):
        """Test streaming with schema parsing."""
        # Mock streaming chunks that form JSON
        mock_chunks = [
            '{"status": "thinking"}',
            '{"status": "generating", "progress": 50}',
            '{"status": "complete", "result": "Final answer"}'
        ]
        
        async def mock_stream(messages, **kwargs):
            for chunk_text in mock_chunks:
                # Create mock StreamChunk
                chunk = MagicMock()
                chunk.content = chunk_text
                chunk.delta = chunk_text
                chunk.finish_reason = None
                chunk.usage = None
                yield chunk
            
            # Final chunk - properly structured to trigger completion
            final_chunk = MagicMock() 
            final_chunk.content = ""
            final_chunk.delta = ""
            final_chunk.finish_reason = "stop"
            final_chunk.usage = TokenCount(prompt_tokens=10, completion_tokens=20, total_tokens=30)
            yield final_chunk
        
        mock_client.stream_chat = mock_stream
        mock_client.provider_name = "test"  # Set provider name for normalization
        
        enhanced_chunks = []
        async for chunk in unified_client.stream_with_schema(sample_messages):
            enhanced_chunks.append(chunk)
        
        # Should get enhanced chunks with partial parsing
        assert len(enhanced_chunks) > 0
        
        # Check that we got EnhancedStreamChunk objects
        for chunk in enhanced_chunks:
            assert isinstance(chunk, EnhancedStreamChunk)
        
        # Should get enhanced chunks
        assert len(enhanced_chunks) > 0
        
        # Look for completion in any chunk
        completed_chunks = [c for c in enhanced_chunks if c.is_complete]
        assert len(completed_chunks) > 0
    
    @pytest.mark.asyncio
    async def test_error_handling_in_stream(self, unified_client, mock_client, sample_messages):
        """Test error handling in unified streaming."""
        async def error_stream(messages, **kwargs):
            first_chunk = MagicMock()
            first_chunk.finish_reason = None  # Make sure it doesn't complete
            first_chunk.content = "partial"
            yield first_chunk
            raise Exception("Stream error")
        
        mock_client.stream_chat = error_stream
        mock_client.provider_name = "test"
        
        with pytest.raises(Exception):
            async for chunk in unified_client.stream_with_schema(sample_messages):
                pass
    
    @pytest.mark.asyncio
    async def test_normalization_integration(self, unified_client, mock_client, sample_messages):
        """Test that normalization works with unified client."""
        # Create chunk that needs normalization
        raw_chunk = MagicMock()
        raw_chunk.choices = [MagicMock()]
        raw_chunk.choices[0].delta.content = "Test content"
        raw_chunk.choices[0].finish_reason = None
        
        async def mock_stream(messages, **kwargs):
            yield raw_chunk
        
        mock_client.stream_chat = mock_stream
        mock_client.provider_name = "openai"
        
        chunks = []
        async for chunk in unified_client.stream_with_schema(sample_messages):
            chunks.append(chunk)
        
        # Should normalize OpenAI chunk correctly
        assert len(chunks) > 0
        first_chunk = chunks[0]
        assert first_chunk.delta == "Test content"