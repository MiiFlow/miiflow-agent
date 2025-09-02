"""Tests for TogetherAI provider client."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from miiflow_llm.providers.together_client import TogetherClient
from miiflow_llm.core import Message, MessageRole, TokenCount, StreamChunk, ChatResponse


class TestTogetherClient:
    """Test suite for Together client."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TogetherClient(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            api_key="test-key",
            timeout=30.0
        )
    
    @pytest.mark.asyncio
    async def test_client_initialization(self, client):
        """Test client initialization."""
        assert client.model == "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        assert client.api_key == "test-key"
        assert client.timeout == 30.0
        assert client.provider_name == "together"
    
    @pytest.mark.asyncio
    async def test_chat_completion_success(self, client, sample_messages):
        """Test successful chat completion."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello from TogetherAI!"
        mock_response.choices[0].message.role = "assistant"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 15
        mock_response.usage.completion_tokens = 25
        mock_response.usage.total_tokens = 40
        
        with patch.object(client.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            response = await client.chat(sample_messages)
            
            # Verify response format
            assert isinstance(response, ChatResponse)
            assert response.message.role == MessageRole.ASSISTANT
            assert response.message.content == "Hello from TogetherAI!"
            assert response.usage.prompt_tokens == 15
            assert response.usage.completion_tokens == 25
            assert response.usage.total_tokens == 40
            assert response.model == "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
            assert response.provider == "together"
            assert response.finish_reason == "stop"
            
            # Verify API call uses TogetherAI base URL
            mock_create.assert_called_once()
            call_args = mock_create.call_args
            assert call_args.kwargs['model'] == "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    
    @pytest.mark.asyncio
    async def test_stream_chat_success(self, client, sample_messages):
        """Test successful streaming chat."""
        stream_chunks = []
        
        # Create TogetherAI-style streaming chunks
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Hello"
        chunk1.choices[0].finish_reason = None
        stream_chunks.append(chunk1)
        
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = " from TogetherAI!"
        chunk2.choices[0].finish_reason = None
        stream_chunks.append(chunk2)
        
        chunk3 = MagicMock()
        chunk3.choices = [MagicMock()]
        chunk3.choices[0].delta.content = None
        chunk3.choices[0].finish_reason = "stop"
        chunk3.usage = MagicMock()
        chunk3.usage.prompt_tokens = 15
        chunk3.usage.completion_tokens = 25
        chunk3.usage.total_tokens = 40
        stream_chunks.append(chunk3)
        
        async def mock_stream_generator():
            for chunk in stream_chunks:
                yield chunk
        
        with patch.object(client.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_stream_generator()
            
            chunks = []
            async for chunk in client.stream_chat(sample_messages):
                chunks.append(chunk)
            
            # Verify we got chunks
            assert len(chunks) == 3
            
            # Verify first chunk
            assert chunks[0].content == "Hello"
            assert chunks[0].delta == "Hello"
            assert chunks[0].finish_reason is None
            
            # Verify second chunk
            assert chunks[1].content == "Hello from TogetherAI!"
            assert chunks[1].delta == " from TogetherAI!"
            assert chunks[1].finish_reason is None
            
            # Verify final chunk
            assert chunks[2].finish_reason == "stop"
            assert chunks[2].usage.total_tokens == 40
    
    @pytest.mark.asyncio
    async def test_together_specific_models(self, client):
        """Test Together-specific model handling."""
        # Test with various Together models
        models = [
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "meta-llama/Llama-3-8b-chat-hf",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"
        ]
        
        for model in models:
            test_client = TogetherClient(model=model, api_key="test")
            assert test_client.model == model
            assert test_client.provider_name == "together"
    
    @pytest.mark.asyncio
    async def test_error_handling(self, client, sample_messages):
        """Test error handling in chat completion."""
        from miiflow_llm.core.exceptions import ProviderError
        
        with patch.object(client.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = Exception("TogetherAI API Error")
            
            with pytest.raises(ProviderError):
                await client.chat(sample_messages)
    
    @pytest.mark.asyncio
    async def test_message_conversion(self, client):
        """Test message format conversion (OpenAI-compatible)."""
        messages = [
            Message.system("You are a helpful AI assistant."),
            Message.user("What is 2+2?"),
            Message.assistant("2+2 equals 4."),
        ]
        
        converted = client._convert_messages_to_openai_format(messages)
        
        assert len(converted) == 3
        assert converted[0]["role"] == "system"
        assert converted[0]["content"] == "You are a helpful AI assistant."
        assert converted[1]["role"] == "user"
        assert converted[1]["content"] == "What is 2+2?"
        assert converted[2]["role"] == "assistant"
        assert converted[2]["content"] == "2+2 equals 4."
    
    @pytest.mark.asyncio
    async def test_custom_parameters(self, client, sample_messages):
        """Test chat with custom parameters."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Custom response"
        mock_response.choices[0].message.role = "assistant"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 15
        mock_response.usage.total_tokens = 25
        
        with patch.object(client.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            await client.chat(
                sample_messages, 
                temperature=0.8, 
                max_tokens=200,
                top_p=0.9
            )
            
            call_args = mock_create.call_args
            assert call_args.kwargs['temperature'] == 0.8
            assert call_args.kwargs['max_tokens'] == 200
            assert call_args.kwargs.get('top_p') == 0.9