"""Tests for main LLMClient interface."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from miiflow_llm.core import LLMClient, Message, MessageRole, TokenCount, StreamChunk, ChatResponse
from miiflow_llm.core.exceptions import MiiflowLLMError


class TestLLMClient:
    """Test suite for main LLMClient."""
    
    @pytest.mark.asyncio
    async def test_create_openai_client(self):
        """Test creating OpenAI client via factory."""
        with patch('miiflow_llm.utils.env.get_api_key') as mock_get_key:
            mock_get_key.return_value = "test-key"
            
            client = LLMClient.create("openai", "gpt-4o-mini")
            
            assert client.client.model == "gpt-4o-mini"
            assert client.client.provider_name == "openai"
    
    @pytest.mark.asyncio
    async def test_create_anthropic_client(self):
        """Test creating Anthropic client via factory."""
        with patch('miiflow_llm.utils.env.get_api_key') as mock_get_key:
            mock_get_key.return_value = "test-key"
            
            client = LLMClient.create("anthropic", "claude-3-haiku-20240307")
            
            assert client.client.model == "claude-3-haiku-20240307"
            assert client.client.provider_name == "anthropic"
    
    @pytest.mark.asyncio
    async def test_create_with_explicit_api_key(self):
        """Test creating client with explicit API key."""
        client = LLMClient.create("openai", "gpt-4o-mini", api_key="explicit-key")
        
        assert client.client.api_key == "explicit-key"
    
    @pytest.mark.asyncio
    async def test_create_missing_api_key(self):
        """Test error when API key is missing."""
        with patch('miiflow_llm.utils.env.get_api_key') as mock_get_key:
            mock_get_key.return_value = None
            
            with pytest.raises(ValueError, match="No API key found"):
                LLMClient.create("openai", "gpt-4o-mini")
    
    @pytest.mark.asyncio
    async def test_create_ollama_no_key_required(self):
        """Test Ollama creation doesn't require API key."""
        with patch('miiflow_llm.utils.env.get_api_key') as mock_get_key:
            mock_get_key.return_value = None
            
            # Should not raise error for Ollama
            client = LLMClient.create("ollama", "llama3.1:8b")
            assert client.client.provider_name == "ollama"
    
    @pytest.mark.asyncio
    async def test_chat_message_normalization(self):
        """Test chat with message normalization."""
        mock_client = MagicMock()
        mock_response = ChatResponse(
            message=Message.assistant("Test response"),
            usage=TokenCount(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            model="test-model",
            provider="test"
        )
        mock_client.achat = AsyncMock(return_value=mock_response)
        
        llm_client = LLMClient(mock_client)
        
        # Test with dict messages
        dict_messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"}
        ]
        
        response = await llm_client.achat(dict_messages)
        
        # Should convert to Message objects
        call_args = mock_client.achat.call_args[0][0]
        assert all(isinstance(msg, Message) for msg in call_args)
        assert call_args[0].role == MessageRole.SYSTEM
        assert call_args[1].role == MessageRole.USER
        
        assert response == mock_response
    
    @pytest.mark.asyncio
    async def test_stream_chat_with_metrics(self):
        """Test streaming chat with metrics collection."""
        mock_client = MagicMock()
        
        async def mock_stream(messages, **kwargs):
            yield StreamChunk(content="Hello", delta="Hello")
            yield StreamChunk(
                content="Hello world", 
                delta=" world",
                finish_reason="stop",
                usage=TokenCount(prompt_tokens=5, completion_tokens=10, total_tokens=15)
            )
        
        mock_client.astream_chat = mock_stream
        mock_client.provider_name = "test"
        mock_client.model = "test-model"
        
        llm_client = LLMClient(mock_client)
        
        chunks = []
        async for chunk in llm_client.astream_chat([Message.user("Hello")]):
            chunks.append(chunk)
        
        assert len(chunks) == 2
        assert chunks[0].content == "Hello"
        assert chunks[1].finish_reason == "stop"
        
        # Check metrics were recorded
        metrics = llm_client.get_metrics()
        assert metrics is not None
    
    @pytest.mark.asyncio
    async def test_stream_with_schema_integration(self):
        """Test stream_with_schema method."""
        mock_client = MagicMock()
        mock_client.provider_name = "test"
        
        async def mock_stream(messages, **kwargs):
            yield StreamChunk(content='{"status": "complete"}', delta='{"status": "complete"}')
        
        mock_client.astream_chat = mock_stream
        
        llm_client = LLMClient(mock_client)
        
        enhanced_chunks = []
        async for chunk in llm_client.stream_with_schema([Message.user("Generate JSON")]):
            enhanced_chunks.append(chunk)
        
        assert len(enhanced_chunks) > 0
        # Should get EnhancedStreamChunk objects
        from miiflow_llm.core.streaming import EnhancedStreamChunk
        for chunk in enhanced_chunks:
            assert isinstance(chunk, EnhancedStreamChunk)
    
    @pytest.mark.asyncio
    async def test_error_handling_and_metrics(self):
        """Test error handling records failed metrics."""
        mock_client = MagicMock()
        mock_client.achat = AsyncMock(side_effect=Exception("API Error"))
        mock_client.provider_name = "test"
        mock_client.model = "test-model"
        
        llm_client = LLMClient(mock_client)
        
        with pytest.raises(Exception):
            await llm_client.achat([Message.user("Hello")])
        
        # Should still record metrics for failed request
        metrics = llm_client.get_metrics()
        assert metrics is not None
    
    @pytest.mark.asyncio
    async def test_supported_providers(self):
        """Test all supported providers can be created."""
        providers = [
            ("openai", "gpt-4o-mini"),
            ("anthropic", "claude-3-haiku-20240307"),
            ("groq", "llama-3.1-8b-instant"),
            ("together", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"),
            ("gemini", "gemini-1.5-flash"),
            ("xai", "grok-beta"),
            ("openrouter", "meta-llama/llama-3.2-3b-instruct:free"),
            ("ollama", "llama3.1:8b"),
        ]
        
        for provider, model in providers:
            with patch('miiflow_llm.utils.env.get_api_key') as mock_get_key:
                if provider == "ollama":
                    mock_get_key.return_value = None  # Ollama doesn't need key
                else:
                    mock_get_key.return_value = "test-key"
                
                client = LLMClient.create(provider, model)
                assert client.client.provider_name == provider
                assert client.client.model == model
    
    @pytest.mark.asyncio
    async def test_mistral_provider_creation(self):
        """Test Mistral provider creation."""
        with patch('miiflow_llm.utils.env.get_api_key') as mock_get_key:
            mock_get_key.return_value = "test-key"
            
            # Mock the actual Mistral client
            with patch('miiflow_llm.providers.mistral_client.Mistral') as mock_mistral:
                mock_mistral.return_value = MagicMock()
                
                client = LLMClient.create("mistral", "mistral-small-latest")
                assert client.client.provider_name == "mistral"
                assert client.client.model == "mistral-small-latest"
