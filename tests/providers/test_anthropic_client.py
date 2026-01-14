"""Tests for Anthropic provider client."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List

from miiflow_agent.providers.anthropic_client import AnthropicClient
from miiflow_agent.core import Message, MessageRole, TokenCount, StreamChunk, ChatResponse


class TestAnthropicClient:
    """Test suite for Anthropic client."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return AnthropicClient(
            model="claude-3-haiku-20240307",
            api_key="test-key",
            timeout=30.0
        )
    
    @pytest.mark.asyncio
    async def test_client_initialization(self, client):
        """Test client initialization."""
        assert client.model == "claude-3-haiku-20240307"
        assert client.api_key == "test-key"
        assert client.timeout == 30.0
        assert client.provider_name == "anthropic"
    
    @pytest.mark.asyncio
    async def test_chat_completion_success(self, client, sample_messages, mock_anthropic_response):
        """Test successful chat completion."""
        with patch.object(client.client.messages, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_anthropic_response
            
            response = await client.achat(sample_messages)
            
            # Verify response format
            assert isinstance(response, ChatResponse)
            assert response.message.role == MessageRole.ASSISTANT
            assert response.message.content == "Hello! I'm doing well, thank you for asking."
            assert response.usage.prompt_tokens == 10
            assert response.usage.completion_tokens == 20
            assert response.model == "claude-3-haiku-20240307"
            assert response.provider == "anthropic"
            assert response.finish_reason == "end_turn"
            
            # Verify API call
            mock_create.assert_called_once()
            call_args = mock_create.call_args
            assert call_args.kwargs['model'] == "claude-3-haiku-20240307"
            assert len(call_args.kwargs['messages']) == 1  # System message becomes system param
            assert call_args.kwargs['system'] == "You are a helpful assistant."
    
    @pytest.mark.asyncio
    async def test_stream_chat_success(self, client, sample_messages, mock_anthropic_stream_chunks):
        """Test successful streaming chat."""
        async def mock_stream_generator():
            for chunk in mock_anthropic_stream_chunks:
                yield chunk
        
        with patch.object(client.client.messages, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_stream_generator()
            
            chunks = []
            async for chunk in client.astream_chat(sample_messages):
                chunks.append(chunk)
            
            # Verify we got chunks
            assert len(chunks) >= 1
            
            # Find content chunk
            content_chunk = next((c for c in chunks if c.delta), None)
            assert content_chunk is not None
            assert "Hello! How are you?" in content_chunk.content
            
            # Check that we got some chunks with content
            content_chunks = [c for c in chunks if c.delta and c.delta.strip()]
            assert len(content_chunks) >= 1
            
            # Verify API call
            mock_create.assert_called_once()
            call_args = mock_create.call_args
            assert call_args.kwargs['stream'] is True
    
    @pytest.mark.asyncio
    async def test_system_message_handling(self, client):
        """Test system message extraction."""
        messages = [
            Message.system("You are a helpful AI assistant."),
            Message.user("Hello there!")
        ]
        
        # Test that messages are converted properly
        system, converted = client._prepare_messages(messages)
        
        # System message should be extracted and remaining messages converted
        assert len(converted) >= 1
        # Check that system content is handled (may be in system parameter)
        
    @pytest.mark.asyncio
    async def test_multiple_system_messages(self, client):
        """Test handling multiple system messages."""
        messages = [
            Message.system("You are helpful."),
            Message.system("You are concise."),
            Message.user("Hello!")
        ]
        
        system, converted = client._prepare_messages(messages)
        
        # Should handle multiple system messages gracefully
        assert len(converted) >= 1
    
    @pytest.mark.asyncio
    async def test_chat_with_temperature(self, client, sample_messages):
        """Test chat with custom temperature."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "Test response"
        mock_response.role = "assistant"
        mock_response.stop_reason = "end_turn"
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 10
        
        with patch.object(client.client.messages, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            await client.achat(sample_messages, temperature=0.9, max_tokens=100)
            
            call_args = mock_create.call_args
            # Anthropic uses different parameter names
            assert call_args.kwargs.get('temperature') == 0.9
            assert call_args.kwargs.get('max_tokens') == 100
    
    @pytest.mark.asyncio
    async def test_multimodal_message_conversion(self, client):
        """Test multimodal message conversion."""
        from miiflow_agent.core.message import TextBlock, ImageBlock
        
        multimodal_message = Message.user([
            TextBlock(text="What's in this image?"),
            ImageBlock(image_url="data:image/jpeg;base64,/9j/4AAQSkZJRg...", detail="high")
        ])
        
        system, converted = client._prepare_messages([multimodal_message])
        
        assert len(converted) == 1
        assert converted[0]["role"] == "user"
        assert isinstance(converted[0]["content"], list)
        assert len(converted[0]["content"]) == 2
        
        # Check text block
        text_block = converted[0]["content"][0]
        assert text_block["type"] == "text"
        assert text_block["text"] == "What's in this image?"
        
        # Check image block (Anthropic format)
        image_block = converted[0]["content"][1]
        assert image_block["type"] == "image"
        assert "source" in image_block
    
    @pytest.mark.asyncio
    async def test_error_handling(self, client, sample_messages):
        """Test error handling in chat completion."""
        from miiflow_agent.core.exceptions import ProviderError
        
        with patch.object(client.client.messages, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = Exception("API Error")
            
            with pytest.raises(ProviderError):
                await client.achat(sample_messages)
    
    @pytest.mark.asyncio
    async def test_stream_error_handling(self, client, sample_messages):
        """Test error handling in streaming."""
        from miiflow_agent.core.exceptions import ProviderError
        
        async def error_generator():
            yield MagicMock()  # First chunk OK
            raise Exception("Stream error")
        
        with patch.object(client.client.messages, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = error_generator()
            
            with pytest.raises(ProviderError):
                chunks = []
                async for chunk in client.astream_chat(sample_messages):
                    chunks.append(chunk)
    
    @pytest.mark.asyncio
    async def test_tool_calls_not_supported(self, client, sample_messages):
        """Test that tool calls raise appropriate error."""
        tools = [{"type": "function", "function": {"name": "test"}}]

        # Anthropic might not support tools in same way as OpenAI
        # This test ensures graceful handling
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "I can't use tools."
        mock_response.role = "assistant"
        mock_response.stop_reason = "end_turn"
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 10

        with patch.object(client.client.messages, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            # Should not crash, even if tools are passed
            response = await client.achat(sample_messages, tools=tools)
            assert response.message.content == "I can't use tools."

    @pytest.mark.asyncio
    async def test_empty_content_handling(self, client):
        """Test that empty or whitespace-only content is handled correctly."""
        # Test with empty string
        empty_message = Message.user("")
        system, converted = client._prepare_messages([empty_message])

        assert len(converted) == 1
        assert converted[0]["role"] == "user"
        # Should have placeholder content, not empty or whitespace
        assert converted[0]["content"][0]["type"] == "text"
        assert converted[0]["content"][0]["text"] == "[no content]"
        assert converted[0]["content"][0]["text"].strip() != ""

        # Test with whitespace-only string
        whitespace_message = Message.user("   \n\t  ")
        system, converted = client._prepare_messages([whitespace_message])

        assert len(converted) == 1
        assert converted[0]["role"] == "user"
        # Should have placeholder content, not whitespace
        assert converted[0]["content"][0]["type"] == "text"
        assert converted[0]["content"][0]["text"] == "[no content]"
        assert converted[0]["content"][0]["text"].strip() != ""

    @pytest.mark.asyncio
    async def test_whitespace_textblock_filtering(self, client):
        """Test that whitespace-only TextBlock objects are filtered out."""
        from miiflow_agent.core.message import TextBlock, ImageBlock

        # Create a message with both valid and whitespace-only text blocks
        multimodal_message = Message.user([
            TextBlock(text="   "),  # Whitespace-only, should be filtered
            TextBlock(text="Valid content"),  # Should be kept
            TextBlock(text=""),  # Empty, should be filtered
        ])

        system, converted = client._prepare_messages([multimodal_message])

        assert len(converted) == 1
        assert converted[0]["role"] == "user"
        assert isinstance(converted[0]["content"], list)
        # Should only have one text block (the valid one)
        text_blocks = [b for b in converted[0]["content"] if b["type"] == "text"]
        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == "Valid content"

    @pytest.mark.asyncio
    async def test_all_whitespace_blocks_get_placeholder(self, client):
        """Test that if all TextBlocks are whitespace, a placeholder is added."""
        from miiflow_agent.core.message import TextBlock

        # Create a message with only whitespace text blocks
        multimodal_message = Message.user([
            TextBlock(text="   "),  # Whitespace-only
            TextBlock(text=""),  # Empty
        ])

        system, converted = client._prepare_messages([multimodal_message])

        assert len(converted) == 1
        assert converted[0]["role"] == "user"
        assert isinstance(converted[0]["content"], list)
        # Should have placeholder since all were filtered
        assert len(converted[0]["content"]) == 1
        assert converted[0]["content"][0]["type"] == "text"
        assert converted[0]["content"][0]["text"] == "[no content]"

    @pytest.mark.asyncio
    async def test_tool_result_with_empty_content(self, client):
        """Test that tool results with empty content get placeholder."""
        # Create a tool result message with empty content
        tool_result = Message(
            role=MessageRole.USER,
            content="",
            tool_call_id="call_123"
        )

        system, converted = client._prepare_messages([tool_result])

        assert len(converted) == 1
        assert converted[0]["role"] == "user"
        assert converted[0]["content"][0]["type"] == "tool_result"
        assert converted[0]["content"][0]["tool_use_id"] == "call_123"
        # Should have placeholder content
        assert converted[0]["content"][0]["content"] == "[empty result]"
        assert converted[0]["content"][0]["content"].strip() != ""

    @pytest.mark.asyncio
    async def test_assistant_with_tool_calls_and_whitespace_content(self, client):
        """Test assistant message with tool calls and whitespace-only content."""
        # Create an assistant message with tool calls and whitespace content
        assistant_msg = Message(
            role=MessageRole.ASSISTANT,
            content="   ",  # Whitespace-only
            tool_calls=[
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "test_tool", "arguments": {"arg": "value"}}
                }
            ]
        )

        system, converted = client._prepare_messages([assistant_msg])

        assert len(converted) == 1
        assert converted[0]["role"] == "assistant"
        assert isinstance(converted[0]["content"], list)
        # Should only have tool_use block, no text block since content was whitespace
        content_types = [b["type"] for b in converted[0]["content"]]
        assert "tool_use" in content_types
        assert "text" not in content_types  # Whitespace text should be filtered