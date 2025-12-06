"""Tests for Gemini provider client."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from miiflow_llm.core import Message, MessageRole, TokenCount, StreamChunk, ChatResponse


# Patch genai at module level to prevent any real API calls
@pytest.fixture(autouse=True)
def mock_genai():
    """Mock the genai module for all tests."""
    with patch('miiflow_llm.providers.gemini_client.genai') as mock:
        mock_model = MagicMock()
        mock.GenerativeModel.return_value = mock_model
        mock.configure = MagicMock()
        yield mock


class TestGeminiClient:
    """Test suite for Gemini client."""

    @pytest.fixture
    def client(self, mock_genai):
        """Create test client with mocked SDK."""
        from miiflow_llm.providers.gemini_client import GeminiClient

        client = GeminiClient(
            model="gemini-1.5-flash",
            api_key="test-key",
            timeout=30.0
        )
        return client

    @pytest.fixture
    def mock_response(self):
        """Create a standard mock response."""
        mock_part = MagicMock()
        mock_part.text = "Hello from Gemini!"
        mock_part.function_call = None

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]
        mock_candidate.finish_reason.name = "STOP"

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata.prompt_token_count = 12
        mock_response.usage_metadata.candidates_token_count = 18
        mock_response.usage_metadata.total_token_count = 30
        return mock_response

    @pytest.mark.asyncio
    async def test_client_initialization(self, client):
        """Test client initialization."""
        assert client.model == "gemini-1.5-flash"
        assert client.api_key == "test-key"
        assert client.timeout == 30.0
        assert client.provider_name == "gemini"

    @pytest.mark.asyncio
    async def test_chat_completion_success(self, client, sample_messages, mock_response):
        """Test successful chat completion."""
        # Patch asyncio.to_thread to return the mock response directly
        async def mock_to_thread(func, *args, **kwargs):
            return mock_response

        with patch('asyncio.to_thread', side_effect=mock_to_thread):
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
        chunk1.candidates[0].content.parts[0].function_call = None
        chunk1.candidates[0].finish_reason = None
        stream_chunks.append(chunk1)

        chunk2 = MagicMock()
        chunk2.candidates = [MagicMock()]
        chunk2.candidates[0].content.parts = [MagicMock()]
        chunk2.candidates[0].content.parts[0].text = " from Gemini!"
        chunk2.candidates[0].content.parts[0].function_call = None
        chunk2.candidates[0].finish_reason = None
        stream_chunks.append(chunk2)

        chunk3 = MagicMock()
        chunk3.candidates = [MagicMock()]
        chunk3.candidates[0].content.parts = [MagicMock()]
        chunk3.candidates[0].content.parts[0].text = ""
        chunk3.candidates[0].content.parts[0].function_call = None
        chunk3.candidates[0].finish_reason.name = "STOP"
        chunk3.usage_metadata.prompt_token_count = 12
        chunk3.usage_metadata.candidates_token_count = 18
        chunk3.usage_metadata.total_token_count = 30
        stream_chunks.append(chunk3)

        # Patch asyncio.to_thread to return the stream chunks
        async def mock_to_thread(func, *args, **kwargs):
            return stream_chunks

        with patch('asyncio.to_thread', side_effect=mock_to_thread):
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
        converted = await client._convert_messages_to_gemini_format(messages)

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
        converted = await client._convert_messages_to_gemini_format([multimodal_message])

        # Basic validation - should not crash
        assert isinstance(converted, list)
        assert len(converted) >= 1

    @pytest.mark.asyncio
    async def test_gemini_specific_parameters(self, client, sample_messages):
        """Test Gemini-specific parameters."""
        mock_part = MagicMock()
        mock_part.text = "Test response"
        mock_part.function_call = None

        mock_resp = MagicMock()
        mock_resp.candidates = [MagicMock()]
        mock_resp.candidates[0].content.parts = [mock_part]
        mock_resp.candidates[0].finish_reason.name = "STOP"
        mock_resp.usage_metadata.prompt_token_count = 5
        mock_resp.usage_metadata.candidates_token_count = 10
        mock_resp.usage_metadata.total_token_count = 15

        call_args = []

        async def mock_to_thread(func, *args, **kwargs):
            call_args.append((func, args, kwargs))
            return mock_resp

        with patch('asyncio.to_thread', side_effect=mock_to_thread):
            await client.achat(
                sample_messages,
                temperature=0.8,
                max_tokens=150,
                top_p=0.95
            )

            # Verify that the call was made
            assert len(call_args) == 1

    @pytest.mark.asyncio
    async def test_error_handling(self, client, sample_messages):
        """Test error handling in chat completion."""
        from miiflow_llm.core.exceptions import ProviderError

        async def mock_to_thread_error(func, *args, **kwargs):
            raise Exception("Gemini API Error")

        with patch('asyncio.to_thread', side_effect=mock_to_thread_error):
            with pytest.raises(ProviderError):
                await client.achat(sample_messages)

    @pytest.mark.asyncio
    async def test_stream_error_handling(self, client, sample_messages):
        """Test error handling in streaming."""
        from miiflow_llm.core.exceptions import ProviderError

        async def mock_to_thread_error(func, *args, **kwargs):
            raise Exception("Gemini stream error")

        with patch('asyncio.to_thread', side_effect=mock_to_thread_error):
            with pytest.raises(ProviderError):
                chunks = []
                async for chunk in client.astream_chat(sample_messages):
                    chunks.append(chunk)

    @pytest.mark.asyncio
    async def test_gemini_models(self, mock_genai):
        """Test Gemini model variations."""
        from miiflow_llm.providers.gemini_client import GeminiClient

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
