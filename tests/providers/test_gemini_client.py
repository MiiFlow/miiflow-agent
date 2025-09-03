"""Tests for Gemini provider client."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from miiflow_llm.providers.gemini_client import GeminiClient
from miiflow_llm.core import Message, MessageRole, TokenCount, StreamChunk, ChatResponse


class TestGeminiClient:
    """Test suite for Gemini client."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return GeminiClient(
            model="gemini-1.5-flash",
            api_key="test-key",
            timeout=30.0
        )
    
    @pytest.mark.asyncio
    async def test_client_initialization(self, client):
        """Test client initialization."""
        assert client.model == "gemini-1.5-flash"
        assert client.api_key == "test-key"
        assert client.timeout == 30.0
        assert client.provider_name == "gemini"
    
    @pytest.mark.asyncio
    async def test_chat_completion_success(self, client, sample_messages):
        """Test successful chat completion."""
        # Mock Gemini response format
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock()]
        mock_response.candidates[0].content.parts[0].text = "Hello from Gemini!"
        mock_response.candidates[0].finish_reason.name = "STOP"
        mock_response.usage_metadata.prompt_token_count = 12
        mock_response.usage_metadata.candidates_token_count = 18
        mock_response.usage_metadata.total_token_count = 30
        
        with patch.object(client.client, 'generate_content') as mock_generate:
            mock_generate.return_value = mock_response
            
            response = await client.achat(sample_messages)
            
            # Verify response format
            assert isinstance(response, ChatResponse)
            assert response.message.role == MessageRole.ASSISTANT
            assert response.message.content == "Hello from Gemini!"
            assert response.usage.prompt_tokens == 12
            assert response.usage.completion_tokens == 18
            assert response.usage.total_tokens == 30
            assert response.model == "gemini-1.5-flash"
            assert response.provider == "gemini"
            assert response.finish_reason == "STOP"
    
    @pytest.mark.asyncio
    async def test_stream_chat_success(self, client, sample_messages):
        """Test successful streaming chat."""
        # Mock Gemini streaming chunks
        stream_chunks = []
        
        chunk1 = MagicMock()
        chunk1.candidates = [MagicMock()]
        chunk1.candidates[0].content.parts = [MagicMock()]
        chunk1.candidates[0].content.parts[0].text = "Hello"
        chunk1.candidates[0].finish_reason = None
        stream_chunks.append(chunk1)
        
        chunk2 = MagicMock()
        chunk2.candidates = [MagicMock()]
        chunk2.candidates[0].content.parts = [MagicMock()]
        chunk2.candidates[0].content.parts[0].text = " from Gemini!"
        chunk2.candidates[0].finish_reason = None
        stream_chunks.append(chunk2)
        
        chunk3 = MagicMock()
        chunk3.candidates = [MagicMock()]
        chunk3.candidates[0].content.parts = [MagicMock()]
        chunk3.candidates[0].content.parts[0].text = ""
        chunk3.candidates[0].finish_reason.name = "STOP"
        chunk3.usage_metadata.prompt_token_count = 12
        chunk3.usage_metadata.candidates_token_count = 18
        chunk3.usage_metadata.total_token_count = 30
        stream_chunks.append(chunk3)
        
        # Return a regular iterable (not async generator) for the mock
        with patch.object(client.client, 'generate_content') as mock_generate:
            mock_generate.return_value = stream_chunks
            
            chunks = []
            async for chunk in client.astream_chat(sample_messages):
                chunks.append(chunk)
            
            # Verify we got chunks
            assert len(chunks) >= 2
            
            # Find content chunks
            content_chunks = [c for c in chunks if c.delta]
            assert len(content_chunks) >= 2
            
            # Verify content accumulation
            full_content = "".join(c.delta for c in content_chunks)
            assert "Hello from Gemini!" in full_content
            
            # Verify final chunk has finish_reason
            final_chunk = chunks[-1]
            assert final_chunk.finish_reason == "STOP"
    
    @pytest.mark.asyncio
    async def test_system_message_handling(self, client):
        """Test system message handling in Gemini format."""
        messages = [
            Message.system("You are a helpful assistant specializing in math."),
            Message.user("What is 5 + 7?")
        ]
        
        # Test that messages are converted properly
        converted = client._convert_messages_to_gemini_format(messages)
        
        # Gemini handles system messages differently
        # They might be prepended to user messages or handled specially
        assert len(converted) >= 1
        # Basic validation that conversion works
        assert isinstance(converted, list)
    
    @pytest.mark.asyncio
    async def test_multimodal_message_conversion(self, client):
        """Test multimodal message conversion for Gemini."""
        from miiflow_llm.core.message import TextBlock, ImageBlock
        
        multimodal_message = Message.user([
            TextBlock(text="Describe this image"),
            ImageBlock(image_url="data:image/jpeg;base64,/9j/4AAQ...", detail="high")
        ])
        
        # Test that multimodal conversion works
        converted = client._convert_messages_to_gemini_format([multimodal_message])
        
        # Basic validation - should not crash
        assert isinstance(converted, list)
        assert len(converted) >= 1
    
    @pytest.mark.asyncio
    async def test_gemini_specific_parameters(self, client, sample_messages):
        """Test Gemini-specific parameters."""
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock()]
        mock_response.candidates[0].content.parts[0].text = "Test response"
        mock_response.candidates[0].finish_reason.name = "STOP"
        mock_response.usage_metadata.prompt_token_count = 5
        mock_response.usage_metadata.candidates_token_count = 10
        mock_response.usage_metadata.total_token_count = 15
        
        with patch.object(client.client, 'generate_content') as mock_generate:
            mock_generate.return_value = mock_response
            
            await client.achat(
                sample_messages,
                temperature=0.8,
                max_tokens=150,
                top_p=0.95
            )
            
            # Verify that method was called
            mock_generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, client, sample_messages):
        """Test error handling in chat completion."""
        from miiflow_llm.core.exceptions import ProviderError
        
        with patch.object(client.client, 'generate_content') as mock_generate:
            mock_generate.side_effect = Exception("Gemini API Error")
            
            with pytest.raises(ProviderError):
                await client.achat(sample_messages)
    
    @pytest.mark.asyncio
    async def test_stream_error_handling(self, client, sample_messages):
        """Test error handling in streaming."""
        from miiflow_llm.core.exceptions import ProviderError
        
        async def error_generator():
            yield MagicMock()  # First chunk OK
            raise Exception("Gemini stream error")
        
        with patch.object(client.client, 'generate_content') as mock_generate:
            mock_generate.return_value = error_generator()
            
            with pytest.raises(ProviderError):
                chunks = []
                async for chunk in client.astream_chat(sample_messages):
                    chunks.append(chunk)
    
    @pytest.mark.asyncio
    async def test_gemini_models(self, client):
        """Test Gemini model variations."""
        models = [
            "gemini-1.5-flash",
            "gemini-1.5-pro", 
            "gemini-1.5-flash-8b",
            "gemini-1.0-pro"
        ]
        
        for model in models:
            test_client = GeminiClient(model=model, api_key="test")
            assert test_client.model == model
            assert test_client.provider_name == "gemini"