"""Tests for the OpenRouter model-family whitelist."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from miiflow_agent.core.exceptions import ModelError, RateLimitError
from miiflow_agent.core.message import (
    DocumentBlock,
    ImageBlock,
    Message,
    TextBlock,
    VideoBlock,
)
from miiflow_agent.models.openrouter import is_openrouter_model_allowed
from miiflow_agent.providers.openrouter_client import OpenRouterClient


@pytest.mark.parametrize(
    "model",
    [
        "deepseek/deepseek-r1",
        "deepseek-ai/deepseek-v3",
        "deepseek/deepseek-r1.1:free",
        "z-ai/glm-5.3",
        "thudm/glm-4-32b",
        "x-ai/grok-4.6",
        "~x-ai/grok-latest",
        "~z-ai/glm-latest",
        "z-ai/glm-4.5-air:free",
    ],
)
def test_allowed_openrouter_model_families(model):
    assert is_openrouter_model_allowed(model)
    assert OpenRouterClient(model=model, api_key="test-key").model == model


@pytest.mark.parametrize(
    "model",
    [
        "openai/gpt-5",
        "anthropic/claude-sonnet-4",
        "aion-labs/aion-3.0-mini",
        "deepseek/not-deepseek",
        "deepseek/deepseekish-r1",
        "x-ai/not-grok",
        "x-ai/grok-4/other-provider",
        "~x-ai/grok-4.6",
        "x-ai/grok-4.6:online",
        "x-ai/grok-4.6:nitro",
        "x-ai/grok-4.6:free:online",
        "x-ai/grok-4.6:",
        "x-ai/grok-4.6-",
        "x-ai/grok-4.6\n",
        "X-AI/GROK-4.6",
        "",
    ],
)
def test_other_openrouter_models_are_rejected(model):
    assert not is_openrouter_model_allowed(model)
    with pytest.raises(ModelError, match="allowed families are DeepSeek, GLM, and Grok"):
        OpenRouterClient(model=model, api_key="test-key")


@pytest.mark.parametrize("model", [None, 42, {}, []])
def test_non_string_openrouter_models_are_rejected(model):
    assert not is_openrouter_model_allowed(model)
    with pytest.raises(ModelError):
        OpenRouterClient(model=model, api_key="test-key")


def test_multimodal_messages_use_openrouter_wire_format():
    client = OpenRouterClient(model="x-ai/grok-4.6", api_key="test-key")
    message = Message.user(
        [
            TextBlock(text="Review these inputs"),
            ImageBlock(image_url="https://example.com/chart.png", detail="high"),
            DocumentBlock(
                document_url="data:application/pdf;base64,JVBERi0x",
                filename="report.pdf",
            ),
            VideoBlock(video_url="https://example.com/demo.mp4"),
        ]
    )

    assert client._convert_message_to_dict(message)["content"] == [
        {"type": "text", "text": "Review these inputs"},
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/chart.png", "detail": "high"},
        },
        {
            "type": "file",
            "file": {
                "filename": "report.pdf",
                "file_data": "data:application/pdf;base64,JVBERi0x",
            },
        },
        {
            "type": "video_url",
            "video_url": {"url": "https://example.com/demo.mp4"},
        },
    ]


def test_tool_call_arguments_are_json_encoded():
    client = OpenRouterClient(model="deepseek/deepseek-r1", api_key="test-key")
    message = Message.assistant(
        "",
        tool_calls=[
            {
                "id": "call-1",
                "function": {"name": "lookup", "arguments": {"query": "test"}},
            }
        ],
    )

    converted = client._convert_message_to_dict(message)

    assert converted["tool_calls"][0]["function"]["arguments"] == json.dumps({"query": "test"})


@pytest.mark.asyncio
async def test_chat_preserves_cache_and_reasoning_usage():
    client = OpenRouterClient(model="deepseek/deepseek-r1", api_key="test-key")
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "id": "generation-1",
        "choices": [
            {
                "message": {"content": "done"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 25,
            "total_tokens": 125,
            "prompt_tokens_details": {"cached_tokens": 80},
            "completion_tokens_details": {"reasoning_tokens": 10},
        },
    }
    http_client = MagicMock()
    http_client.post = AsyncMock(return_value=response)
    http_client.__aenter__ = AsyncMock(return_value=http_client)
    http_client.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "miiflow_agent.providers.openrouter_client.httpx.AsyncClient",
        return_value=http_client,
    ):
        result = await client.achat([Message.user("hello")])

    assert result.usage.prompt_tokens == 100
    assert result.usage.cache_read_tokens == 80
    assert result.usage.reasoning_tokens == 10


@pytest.mark.asyncio
async def test_streaming_accumulates_content_and_emits_final_usage():
    client = OpenRouterClient(model="deepseek/deepseek-r1", api_key="test-key")
    sse_lines = [
        'data: {"choices":[{"delta":{"content":"Hel"},"finish_reason":null}]}',
        'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}',
        (
            'data: {"choices":[],"usage":{"prompt_tokens":20,'
            '"completion_tokens":5,"total_tokens":25,'
            '"prompt_tokens_details":{"cached_tokens":12},'
            '"completion_tokens_details":{"reasoning_tokens":3}}}'
        ),
        "data: [DONE]",
    ]

    async def aiter_lines():
        for line in sse_lines:
            yield line

    stream_response = MagicMock()
    stream_response.status_code = 200
    stream_response.aiter_lines = aiter_lines
    stream_response.__aenter__ = AsyncMock(return_value=stream_response)
    stream_response.__aexit__ = AsyncMock(return_value=False)
    http_client = MagicMock()
    http_client.stream.return_value = stream_response
    http_client.__aenter__ = AsyncMock(return_value=http_client)
    http_client.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "miiflow_agent.providers.openrouter_client.httpx.AsyncClient",
        return_value=http_client,
    ):
        chunks = [chunk async for chunk in client.astream_chat([Message.user("hello")])]

    assert [chunk.content for chunk in chunks] == ["Hel", "Hello", "Hello"]
    assert chunks[-1].usage.total_tokens == 25
    assert chunks[-1].usage.cache_read_tokens == 12
    assert chunks[-1].usage.reasoning_tokens == 3


@pytest.mark.asyncio
async def test_streaming_error_event_is_not_silently_dropped():
    client = OpenRouterClient(model="deepseek/deepseek-r1", api_key="test-key")

    async def aiter_lines():
        yield 'data: {"error":{"code":429,"message":"rate limited"}}'

    stream_response = MagicMock()
    stream_response.status_code = 200
    stream_response.aiter_lines = aiter_lines
    stream_response.__aenter__ = AsyncMock(return_value=stream_response)
    stream_response.__aexit__ = AsyncMock(return_value=False)
    http_client = MagicMock()
    http_client.stream.return_value = stream_response
    http_client.__aenter__ = AsyncMock(return_value=http_client)
    http_client.__aexit__ = AsyncMock(return_value=False)

    with (
        patch(
            "miiflow_agent.providers.openrouter_client.httpx.AsyncClient",
            return_value=http_client,
        ),
        pytest.raises(RateLimitError, match="rate limited") as error,
    ):
        _ = [chunk async for chunk in client.astream_chat([Message.user("hello")])]

    assert error.value.error_type.value == "rate_limited"


def _sse_client(sse_lines, model="deepseek/deepseek-r1"):
    """A client whose HTTP stream replays `sse_lines`."""
    client = OpenRouterClient(model=model, api_key="test-key")

    async def aiter_lines():
        for line in sse_lines:
            yield line

    stream_response = MagicMock()
    stream_response.status_code = 200
    stream_response.aiter_lines = aiter_lines
    stream_response.__aenter__ = AsyncMock(return_value=stream_response)
    stream_response.__aexit__ = AsyncMock(return_value=False)
    http_client = MagicMock()
    http_client.stream.return_value = stream_response
    http_client.__aenter__ = AsyncMock(return_value=http_client)
    http_client.__aexit__ = AsyncMock(return_value=False)
    return client, http_client


@pytest.mark.asyncio
async def test_streaming_surfaces_reasoning_deltas():
    """Every model this client admits is a reasoning model, so the thinking
    phase is the start of most turns rather than an edge case.

    Reading only `delta.content` left those chunks empty, the yield guard
    skipped them, and the stream sat silent for the whole phase while the
    reasoning text was dropped. OpenRouter documents the field as `reasoning`
    on each message, with `reasoning_content` as an alias.
    """
    client, http_client = _sse_client(
        [
            'data: {"choices":[{"delta":{"reasoning":"Let me "},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"reasoning_content":"think."},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"content":"Answer"},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ]
    )

    with patch(
        "miiflow_agent.providers.openrouter_client.httpx.AsyncClient",
        return_value=http_client,
    ):
        chunks = [c async for c in client.astream_chat([Message.user("hi")])]

    assert [c.thinking_delta for c in chunks] == ["Let me ", "think.", None]
    # Reasoning is not answer text: it must not leak into the content the
    # caller renders or accumulates.
    assert [c.content for c in chunks] == ["", "", "Answer"]


@pytest.mark.asyncio
async def test_a_reasoning_only_chunk_is_not_swallowed():
    # The regression in one line: a chunk with reasoning and nothing else must
    # still reach the consumer.
    client, http_client = _sse_client(
        ['data: {"choices":[{"delta":{"reasoning":"only"}}]}', "data: [DONE]"]
    )

    with patch(
        "miiflow_agent.providers.openrouter_client.httpx.AsyncClient",
        return_value=http_client,
    ):
        chunks = [c async for c in client.astream_chat([Message.user("hi")])]

    assert len(chunks) == 1
    assert chunks[0].thinking_delta == "only"
