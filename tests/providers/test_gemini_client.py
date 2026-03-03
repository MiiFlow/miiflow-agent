"""Tests for Gemini provider client."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from miiflow_agent.core import Message, MessageRole, TokenCount, StreamChunk, ChatResponse


# Patch GEMINI_AVAILABLE to True so GeminiClient can be instantiated
# even when google-generativeai is not installed
@pytest.fixture(autouse=True)
def mock_gemini_available():
    """Mock GEMINI_AVAILABLE for all tests."""
    with patch("miiflow_agent.providers.gemini_client.GEMINI_AVAILABLE", True):
        yield


def _make_rest_response(
    text="",
    function_calls=None,
    finish_reason="STOP",
    prompt_tokens=12,
    completion_tokens=18,
    total_tokens=30,
):
    """Build a Gemini REST API response dict.

    function_calls: list of dicts. Each dict may contain a special
    "thoughtSignature" key which will be placed as a sibling of
    "functionCall" in the part (matching the real API structure).
    """
    parts = []
    if text:
        parts.append({"text": text})
    for fc in function_calls or []:
        fc = dict(fc)  # shallow copy
        part: dict = {}
        # Extract thoughtSignature — it's a sibling of functionCall, not nested inside
        thought_sig = fc.pop("thoughtSignature", None)
        part["functionCall"] = fc
        if thought_sig is not None:
            part["thoughtSignature"] = thought_sig
        parts.append(part)
    return {
        "candidates": [
            {
                "content": {"role": "model", "parts": parts},
                "finishReason": finish_reason,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": prompt_tokens,
            "candidatesTokenCount": completion_tokens,
            "totalTokenCount": total_tokens,
        },
    }


def _mock_httpx_response(data, status_code=200):
    """Create a mock httpx.Response for non-streaming calls."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    resp.text = json.dumps(data)
    return resp


class TestGeminiClient:
    """Test suite for Gemini client."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from miiflow_agent.providers.gemini_client import GeminiClient

        client = GeminiClient(model="gemini-2.5-flash", api_key="test-key", timeout=30.0)
        return client

    @pytest.mark.asyncio
    async def test_client_initialization(self, client):
        """Test client initialization."""
        assert client.model == "gemini-2.5-flash"
        assert client.api_key == "test-key"
        assert client.timeout == 30.0
        assert client.provider_name == "gemini"

    @pytest.mark.asyncio
    async def test_chat_completion_success(self, client, sample_messages):
        """Test successful chat completion via REST API."""
        rest_data = _make_rest_response(text="Hello from Gemini!")

        mock_resp = _mock_httpx_response(rest_data)

        async def mock_post(url, json=None):
            return mock_resp

        mock_http = AsyncMock()
        mock_http.post = mock_post
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        with patch("miiflow_agent.providers.gemini_client.httpx.AsyncClient", return_value=mock_http):
            response = await client.achat(sample_messages)

        assert isinstance(response, ChatResponse)
        assert response.message.role == MessageRole.ASSISTANT
        assert response.message.content == "Hello from Gemini!"
        assert response.usage.prompt_tokens == 12
        assert response.usage.completion_tokens == 18
        assert response.usage.total_tokens == 30
        assert response.model == "gemini-2.5-flash"
        assert response.provider == "gemini"
        assert response.finish_reason == "STOP"

    @pytest.mark.asyncio
    async def test_chat_with_tool_calls(self, client, sample_messages):
        """Test chat completion with function calls."""
        rest_data = _make_rest_response(
            function_calls=[
                {"name": "get_weather", "args": {"city": "Tokyo"}}
            ]
        )

        mock_resp = _mock_httpx_response(rest_data)

        async def mock_post(url, json=None):
            return mock_resp

        mock_http = AsyncMock()
        mock_http.post = mock_post
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        tools = [
            {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            }
        ]

        with patch("miiflow_agent.providers.gemini_client.httpx.AsyncClient", return_value=mock_http):
            response = await client.achat(sample_messages, tools=tools)

        assert response.message.tool_calls is not None
        assert len(response.message.tool_calls) == 1
        tc = response.message.tool_calls[0]
        assert tc["function"]["name"] == "get_weather"
        assert tc["function"]["arguments"] == {"city": "Tokyo"}

    @pytest.mark.asyncio
    async def test_thought_signature_round_trip(self, client):
        """Test that thought_signature is captured from response and preserved in next request."""
        # Step 1: API returns a functionCall with thoughtSignature
        rest_data = _make_rest_response(
            function_calls=[
                {
                    "name": "get_weather",
                    "args": {"city": "Tokyo"},
                    "thoughtSignature": "abc123sig",
                }
            ]
        )

        mock_resp = _mock_httpx_response(rest_data)

        async def mock_post(url, json=None):
            return mock_resp

        mock_http = AsyncMock()
        mock_http.post = mock_post
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        tools = [
            {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            }
        ]

        with patch("miiflow_agent.providers.gemini_client.httpx.AsyncClient", return_value=mock_http):
            response = await client.achat(
                [Message.user("What is the weather in Tokyo?")], tools=tools
            )

        # Verify thought_signature is captured in function_call_metadata
        tc = response.message.tool_calls[0]
        assert tc["function_call_metadata"]["thought_signature"] == "abc123sig"
        assert tc["function_call_metadata"]["gemini_function_name"] == "get_weather"

        # Step 2: Build multi-turn conversation with the tool call result
        # The assistant message carries the tool_calls with metadata
        assistant_msg = response.message
        tool_result_msg = Message(
            role=MessageRole.TOOL,
            content="Sunny, 25C",
            name="get_weather",
        )

        messages = [
            Message.user("What is the weather in Tokyo?"),
            assistant_msg,
            tool_result_msg,
        ]

        # Convert to Gemini format and verify thought_signature is preserved
        gemini_msgs = await client._convert_messages_to_gemini_format(messages)

        # Find the model message with function_call
        model_msg = next(m for m in gemini_msgs if m["role"] == "model")
        fc_part = next(p for p in model_msg["parts"] if "function_call" in p)

        assert fc_part["function_call"]["thought_signature"] == "abc123sig"

        # Step 3: Verify the REST format conversion preserves it as camelCase
        from miiflow_agent.providers.gemini_client import _convert_to_rest_format

        rest_msgs = _convert_to_rest_format(gemini_msgs)
        rest_model_msg = next(m for m in rest_msgs if m["role"] == "model")
        rest_fc_part = next(p for p in rest_model_msg["parts"] if "functionCall" in p)

        # thoughtSignature is a sibling of functionCall, not nested inside
        assert rest_fc_part["thoughtSignature"] == "abc123sig"

    @pytest.mark.asyncio
    async def test_stream_chat_success(self, client, sample_messages):
        """Test successful streaming chat via SSE."""
        # Build SSE data lines
        chunk1_data = _make_rest_response(text="Hello", finish_reason=None)
        chunk2_data = _make_rest_response(text=" from Gemini!", finish_reason="STOP")

        sse_lines = [
            f"data: {json.dumps(chunk1_data)}",
            "",
            f"data: {json.dumps(chunk2_data)}",
            "",
        ]

        # Create async line iterator
        async def aiter_lines():
            for line in sse_lines:
                yield line

        mock_stream_resp = MagicMock()
        mock_stream_resp.status_code = 200
        mock_stream_resp.aiter_lines = aiter_lines
        mock_stream_resp.__aenter__ = AsyncMock(return_value=mock_stream_resp)
        mock_stream_resp.__aexit__ = AsyncMock(return_value=False)

        mock_http = AsyncMock()
        mock_http.stream = MagicMock(return_value=mock_stream_resp)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        with patch("miiflow_agent.providers.gemini_client.httpx.AsyncClient", return_value=mock_http):
            chunks = []
            async for chunk in client.astream_chat(sample_messages):
                chunks.append(chunk)

        assert len(chunks) >= 2
        content_chunks = [c for c in chunks if c.delta]
        full_content = "".join(c.delta for c in content_chunks)
        assert "Hello" in full_content
        assert "from Gemini!" in full_content

        # Verify final chunk has finish_reason
        final_chunk = chunks[-1]
        assert final_chunk.finish_reason == "STOP"

    @pytest.mark.asyncio
    async def test_stream_with_thought_signature(self, client):
        """Test that streaming captures thought_signature from functionCall."""
        chunk_data = _make_rest_response(
            function_calls=[
                {
                    "name": "search",
                    "args": {"query": "test"},
                    "thoughtSignature": "stream_sig_456",
                }
            ],
            finish_reason="STOP",
        )

        sse_lines = [f"data: {json.dumps(chunk_data)}", ""]

        async def aiter_lines():
            for line in sse_lines:
                yield line

        mock_stream_resp = MagicMock()
        mock_stream_resp.status_code = 200
        mock_stream_resp.aiter_lines = aiter_lines
        mock_stream_resp.__aenter__ = AsyncMock(return_value=mock_stream_resp)
        mock_stream_resp.__aexit__ = AsyncMock(return_value=False)

        mock_http = AsyncMock()
        mock_http.stream = MagicMock(return_value=mock_stream_resp)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        tools = [
            {
                "name": "search",
                "description": "Search",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            }
        ]

        with patch("miiflow_agent.providers.gemini_client.httpx.AsyncClient", return_value=mock_http):
            chunks = []
            async for chunk in client.astream_chat(
                [Message.user("search for test")], tools=tools
            ):
                chunks.append(chunk)

        # Find chunk with tool calls
        tc_chunks = [c for c in chunks if c.tool_calls]
        assert len(tc_chunks) >= 1

        tc = tc_chunks[0].tool_calls[0]
        assert tc["function_call_metadata"]["thought_signature"] == "stream_sig_456"

    @pytest.mark.asyncio
    async def test_system_message_handling(self, client):
        """Test system message handling in Gemini format."""
        messages = [
            Message.system("You are a helpful assistant specializing in math."),
            Message.user("What is 5 + 7?"),
        ]

        converted = await client._convert_messages_to_gemini_format(messages)
        assert len(converted) == 1  # Only user message, system is extracted separately
        assert isinstance(converted, list)

        # Verify system instruction extraction
        sys_instr = client._extract_system_instruction(messages)
        assert sys_instr == "You are a helpful assistant specializing in math."

    @pytest.mark.asyncio
    async def test_error_handling(self, client, sample_messages):
        """Test error handling in chat completion."""
        from miiflow_agent.core.exceptions import ProviderError

        mock_resp = _mock_httpx_response({}, status_code=400)
        mock_resp.text = "Bad request"

        async def mock_post(url, json=None):
            return mock_resp

        mock_http = AsyncMock()
        mock_http.post = mock_post
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        with patch("miiflow_agent.providers.gemini_client.httpx.AsyncClient", return_value=mock_http):
            with pytest.raises(ProviderError):
                await client.achat(sample_messages)

    @pytest.mark.asyncio
    async def test_stream_error_handling(self, client, sample_messages):
        """Test error handling in streaming."""
        from miiflow_agent.core.exceptions import ProviderError

        mock_stream_resp = MagicMock()
        mock_stream_resp.status_code = 500
        mock_stream_resp.aread = AsyncMock(return_value=b"Internal Server Error")
        mock_stream_resp.__aenter__ = AsyncMock(return_value=mock_stream_resp)
        mock_stream_resp.__aexit__ = AsyncMock(return_value=False)

        mock_http = AsyncMock()
        mock_http.stream = MagicMock(return_value=mock_stream_resp)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        with patch("miiflow_agent.providers.gemini_client.httpx.AsyncClient", return_value=mock_http):
            with pytest.raises(ProviderError):
                async for _ in client.astream_chat(sample_messages):
                    pass

    @pytest.mark.asyncio
    async def test_gemini_models(self):
        """Test Gemini model variations."""
        from miiflow_agent.providers.gemini_client import GeminiClient

        models = [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
            "gemini-1.5-pro",
        ]

        for model in models:
            test_client = GeminiClient(model=model, api_key="test")
            assert test_client.model == model
            assert test_client.provider_name == "gemini"

    @pytest.mark.asyncio
    async def test_tool_name_sanitization_round_trip(self, client, sample_messages):
        """Test that sanitized tool names are mapped back to originals."""
        rest_data = _make_rest_response(
            function_calls=[{"name": "my_special_tool", "args": {"x": 1}}]
        )

        mock_resp = _mock_httpx_response(rest_data)

        async def mock_post(url, json=None):
            return mock_resp

        mock_http = AsyncMock()
        mock_http.post = mock_post
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        tools = [
            {
                "name": "my/special/tool",  # Has chars that need sanitizing
                "description": "A tool",
                "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}},
            }
        ]

        with patch("miiflow_agent.providers.gemini_client.httpx.AsyncClient", return_value=mock_http):
            response = await client.achat(sample_messages, tools=tools)

        tc = response.message.tool_calls[0]
        # The original name should be restored
        assert tc["function"]["name"] == "my/special/tool"
        # The gemini name should be in metadata
        assert tc["function_call_metadata"]["gemini_function_name"] == "my_special_tool"

    @pytest.mark.asyncio
    async def test_convert_to_rest_format(self):
        """Test the _convert_to_rest_format helper."""
        from miiflow_agent.providers.gemini_client import _convert_to_rest_format

        messages = [
            {
                "role": "model",
                "parts": [
                    {
                        "function_call": {
                            "name": "search",
                            "args": {"q": "test"},
                            "thought_signature": "sig123",
                        }
                    }
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "function_response": {
                            "name": "search",
                            "response": {"result": "found it"},
                        }
                    }
                ],
            },
        ]

        rest = _convert_to_rest_format(messages)

        # Verify camelCase conversion
        assert "functionCall" in rest[0]["parts"][0]
        # thoughtSignature is a sibling of functionCall, not nested inside it
        assert rest[0]["parts"][0]["thoughtSignature"] == "sig123"
        assert "thoughtSignature" not in rest[0]["parts"][0]["functionCall"]

        assert "functionResponse" in rest[1]["parts"][0]
        assert rest[1]["parts"][0]["functionResponse"]["name"] == "search"

    @pytest.mark.asyncio
    async def test_parse_rest_response_with_thought_signature(self):
        """Test _parse_rest_response extracts thoughtSignature from part sibling."""
        from miiflow_agent.providers.gemini_client import _parse_rest_response

        # thoughtSignature is a sibling of functionCall in the part, not nested inside
        data = _make_rest_response(
            function_calls=[
                {
                    "name": "tool_a",
                    "args": {"param": "val"},
                    "thoughtSignature": "mysig",
                }
            ]
        )

        # Verify the test data structure is correct (sibling, not nested)
        part = data["candidates"][0]["content"]["parts"][0]
        assert "thoughtSignature" in part
        assert "thoughtSignature" not in part["functionCall"]

        content, tool_calls, usage, finish_reason = _parse_rest_response(data, {})

        assert content == ""
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "tool_a"
        assert tool_calls[0]["function_call_metadata"]["thought_signature"] == "mysig"
        assert finish_reason == "STOP"

    @pytest.mark.asyncio
    async def test_rest_url_building(self, client):
        """Test REST API URL construction."""
        url = client._build_rest_url(streaming=False)
        assert "generateContent" in url
        assert "key=test-key" in url
        assert "gemini-2.5-flash" in url

        url_stream = client._build_rest_url(streaming=True)
        assert "streamGenerateContent" in url_stream
        assert "alt=sse" in url_stream

    @pytest.mark.asyncio
    async def test_rest_url_strips_models_prefix(self):
        """Test that models/ prefix in model name is stripped to avoid double prefix."""
        from miiflow_agent.providers.gemini_client import GeminiClient

        client = GeminiClient(model="models/gemini-2.5-flash", api_key="test-key")
        url = client._build_rest_url(streaming=False)
        # Should have /models/gemini-2.5-flash NOT /models/models/gemini-2.5-flash
        assert "/models/gemini-2.5-flash:" in url
        assert "/models/models/" not in url
