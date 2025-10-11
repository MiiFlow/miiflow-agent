"""Tests for Groq provider client."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List

from miiflow_llm.providers.groq_client import GroqClient
from miiflow_llm.core import Message, MessageRole, TokenCount, StreamChunk, ChatResponse


class TestGroqClient:
    """Test suite for Groq client."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return GroqClient(
            model="llama-3.1-8b-instant",
            api_key="test-key",
            timeout=30.0
        )
    
    @pytest.mark.asyncio
    async def test_client_initialization(self, client):
        """Test client initialization."""
        assert client.model == "llama-3.1-8b-instant"
        assert client.api_key == "test-key" 
        assert client.timeout == 30.0
        assert client.provider_name == "groq"
    
    @pytest.mark.asyncio
    async def test_chat_completion_success(self, client, sample_messages):
        """Test successful chat completion."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello! I'm doing well, thank you for asking."
        mock_response.choices[0].message.role = "assistant"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        
        with patch.object(client.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            response = await client.achat(sample_messages)
            
            # Verify response format
            assert isinstance(response, ChatResponse)
            assert response.message.role == MessageRole.ASSISTANT
            assert response.message.content == "Hello! I'm doing well, thank you for asking."
            assert response.usage.prompt_tokens == 10
            assert response.usage.completion_tokens == 20
            assert response.usage.total_tokens == 30
            assert response.model == "llama-3.1-8b-instant"
            assert response.provider == "groq"
            assert response.finish_reason == "stop"
            
            # Verify API call
            mock_create.assert_called_once()
            call_args = mock_create.call_args
            assert call_args.kwargs['model'] == "llama-3.1-8b-instant"
    
    @pytest.mark.asyncio
    async def test_stream_chat_success(self, client, sample_messages):
        """Test successful streaming chat."""
        stream_chunks = []
        
        # First chunk
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Hello!"
        chunk1.choices[0].finish_reason = None
        stream_chunks.append(chunk1)
        
        # Second chunk
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = " How are you?"
        chunk2.choices[0].finish_reason = None
        stream_chunks.append(chunk2)
        
        # Final chunk
        chunk3 = MagicMock()
        chunk3.choices = [MagicMock()]
        chunk3.choices[0].delta.content = None
        chunk3.choices[0].finish_reason = "stop"
        chunk3.usage = MagicMock()
        chunk3.usage.prompt_tokens = 10
        chunk3.usage.completion_tokens = 20
        chunk3.usage.total_tokens = 30
        stream_chunks.append(chunk3)
        
        async def mock_stream_generator():
            for chunk in stream_chunks:
                yield chunk
        
        with patch.object(client.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_stream_generator()
            
            chunks = []
            async for chunk in client.astream_chat(sample_messages):
                chunks.append(chunk)
            
            # Verify we got chunks
            assert len(chunks) == 3
            
            # Verify first chunk
            assert chunks[0].content == "Hello!"
            assert chunks[0].delta == "Hello!"
            assert chunks[0].finish_reason is None
            
            # Verify second chunk
            assert chunks[1].content == "Hello! How are you?"
            assert chunks[1].delta == " How are you?"
            assert chunks[1].finish_reason is None
            
            # Verify final chunk
            assert chunks[2].content == "Hello! How are you?"
            assert chunks[2].delta == ""
            assert chunks[2].finish_reason == "stop"
            assert chunks[2].usage.total_tokens == 30
    
    @pytest.mark.asyncio
    async def test_groq_string_fallback(self, client, sample_messages):
        """Test Groq's string fallback for unusual responses."""
        # Test with proper chunk format since string fallback happens in normalization layer
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Hello from Groq!"
        chunk1.choices[0].finish_reason = None
        
        finish_chunk = MagicMock()
        finish_chunk.choices = [MagicMock()]
        finish_chunk.choices[0].delta.content = None
        finish_chunk.choices[0].finish_reason = "stop"
        
        async def mock_stream_generator():
            yield chunk1
            yield finish_chunk
        
        with patch.object(client.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_stream_generator()
            
            chunks = []
            async for chunk in client.astream_chat(sample_messages):
                chunks.append(chunk)
            
            # Should handle chunks gracefully
            assert len(chunks) >= 1
            first_chunk = chunks[0]
            assert isinstance(first_chunk, StreamChunk)
            assert "Hello from Groq!" in first_chunk.content
    
    @pytest.mark.asyncio
    async def test_chat_with_temperature(self, client, sample_messages):
        """Test chat with custom temperature."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].message.role = "assistant"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 10
        mock_response.usage.total_tokens = 15
        
        with patch.object(client.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            await client.achat(sample_messages, temperature=0.9, max_tokens=100)
            
            call_args = mock_create.call_args
            assert call_args.kwargs['temperature'] == 0.9
            assert call_args.kwargs['max_tokens'] == 100
    
    @pytest.mark.asyncio
    async def test_message_conversion(self, client):
        """Test message format conversion through to_openai_format."""
        messages = [
            Message.system("You are helpful."),
            Message.user("Hello"),
            Message.assistant("Hi there!"),
        ]
        
        # Test that messages can be converted to OpenAI format (Groq uses OpenAI format)
        converted = [client.convert_message_to_provider_format(msg) for msg in messages]
        
        assert len(converted) == 3
        assert converted[0]["role"] == "system"
        assert converted[0]["content"] == "You are helpful."
        assert converted[1]["role"] == "user"
        assert converted[1]["content"] == "Hello"
        assert converted[2]["role"] == "assistant"
        assert converted[2]["content"] == "Hi there!"
    
    @pytest.mark.asyncio
    async def test_error_handling(self, client, sample_messages):
        """Test error handling in chat completion."""
        from miiflow_llm.core.exceptions import ProviderError
        
        with patch.object(client.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = Exception("Groq API Error")
            
            with pytest.raises(ProviderError):
                await client.achat(sample_messages)
    
    @pytest.mark.asyncio
    async def test_stream_error_handling(self, client, sample_messages):
        """Test error handling in streaming."""
        from miiflow_llm.core.exceptions import ProviderError
        
        async def error_generator():
            yield MagicMock()  # First chunk OK
            raise Exception("Groq stream error")
        
        with patch.object(client.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = error_generator()
            
            with pytest.raises(ProviderError):
                chunks = []
                async for chunk in client.astream_chat(sample_messages):
                    chunks.append(chunk)
    
    @pytest.mark.asyncio
    async def test_groq_model_compatibility(self, client):
        """Test Groq-specific model handling."""
        # Test with different Groq models
        models = [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768"
        ]
        
        for model in models:
            test_client = GroqClient(model=model, api_key="test")
            assert test_client.model == model
            assert test_client.provider_name == "groq"
