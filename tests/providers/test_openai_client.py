"""Tests for OpenAI provider client."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List

from miiflow_llm.providers.openai_client import OpenAIClient
from miiflow_llm.core import Message, MessageRole, TokenCount, StreamChunk, ChatResponse


class TestOpenAIClient:
    """Test suite for OpenAI client."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return OpenAIClient(
            model="gpt-4o-mini",
            api_key="test-key",
            timeout=30.0
        )
    
    @pytest.mark.asyncio
    async def test_client_initialization(self, client):
        """Test client initialization."""
        assert client.model == "gpt-4o-mini"
        assert client.api_key == "test-key"
        assert client.timeout == 30.0
        assert client.provider_name == "openai"
    
    @pytest.mark.asyncio
    async def test_chat_completion_success(self, client, sample_messages, mock_openai_response):
        """Test successful chat completion."""
        with patch.object(client.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_openai_response
            
            response = await client.chat(sample_messages)
            
            # Verify response format
            assert isinstance(response, ChatResponse)
            assert response.message.role == MessageRole.ASSISTANT
            assert response.message.content == "Hello! I'm doing well, thank you for asking."
            assert response.usage.prompt_tokens == 10
            assert response.usage.completion_tokens == 20
            assert response.usage.total_tokens == 30
            assert response.model == "gpt-4o-mini"
            assert response.provider == "openai"
            assert response.finish_reason == "stop"
            
            # Verify API call
            mock_create.assert_called_once()
            call_args = mock_create.call_args
            assert call_args.kwargs['model'] == "gpt-4o-mini"
            assert len(call_args.kwargs['messages']) == 2
    
    @pytest.mark.asyncio
    async def test_stream_chat_success(self, client, sample_messages, mock_openai_stream_chunks):
        """Test successful streaming chat."""
        async def mock_stream_generator():
            for chunk in mock_openai_stream_chunks:
                yield chunk
        
        with patch.object(client.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_stream_generator()
            
            chunks = []
            async for chunk in client.stream_chat(sample_messages):
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
            
            # Verify API call
            mock_create.assert_called_once()
            call_args = mock_create.call_args
            assert call_args.kwargs['stream'] is True
    
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
            
            await client.chat(sample_messages, temperature=0.9, max_tokens=100)
            
            call_args = mock_create.call_args
            assert call_args.kwargs['temperature'] == 0.9
            assert call_args.kwargs['max_tokens'] == 100
    
    @pytest.mark.asyncio
    async def test_chat_with_tools(self, client, sample_messages):
        """Test chat with tool calls."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information"
                }
            }
        ]
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_response.choices[0].message.role = "assistant"
        mock_response.choices[0].message.tool_calls = [{"id": "call_123", "function": {"name": "get_weather"}}]
        mock_response.choices[0].finish_reason = "tool_calls"
        mock_response.usage.prompt_tokens = 15
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 20
        
        with patch.object(client.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            response = await client.chat(sample_messages, tools=tools)
            
            assert response.finish_reason == "tool_calls"
            assert response.message.tool_calls is not None
            assert len(response.message.tool_calls) == 1
            
            call_args = mock_create.call_args
            assert call_args.kwargs['tools'] == tools
    
    @pytest.mark.asyncio
    async def test_message_conversion(self, client):
        """Test message format conversion through to_openai_format."""
        messages = [
            Message.system("You are helpful."),
            Message.user("Hello"),
            Message.assistant("Hi there!"),
        ]
        
        # Test that messages can be converted to OpenAI format
        converted = [msg.to_openai_format() for msg in messages]
        
        assert len(converted) == 3
        assert converted[0]["role"] == "system"
        assert converted[0]["content"] == "You are helpful."
        assert converted[1]["role"] == "user"
        assert converted[1]["content"] == "Hello"
        assert converted[2]["role"] == "assistant"
        assert converted[2]["content"] == "Hi there!"
    
    @pytest.mark.asyncio
    async def test_multimodal_message_conversion(self, client):
        """Test multimodal message conversion."""
        from miiflow_llm.core.message import TextBlock, ImageBlock
        
        multimodal_message = Message.user([
            TextBlock(text="What's in this image?"),
            ImageBlock(image_url="data:image/jpeg;base64,/9j/4AAQSkZJRg...", detail="high")
        ])
        
        # Test multimodal message conversion through to_openai_format
        converted = multimodal_message.to_openai_format()
        
        assert converted["role"] == "user"
        # Basic validation that multimodal conversion works
        assert "content" in converted
    
    @pytest.mark.asyncio
    async def test_error_handling(self, client, sample_messages):
        """Test error handling in chat completion."""
        from miiflow_llm.core.exceptions import ProviderError
        
        with patch.object(client.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = Exception("API Error")
            
            with pytest.raises(ProviderError):
                await client.chat(sample_messages)
    
    @pytest.mark.asyncio
    async def test_stream_error_handling(self, client, sample_messages):
        """Test error handling in streaming."""
        from miiflow_llm.core.exceptions import ProviderError
        
        async def error_generator():
            yield MagicMock()  # First chunk OK
            raise Exception("Stream error")
        
        with patch.object(client.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = error_generator()
            
            with pytest.raises(ProviderError):
                chunks = []
                async for chunk in client.stream_chat(sample_messages):
                    chunks.append(chunk)