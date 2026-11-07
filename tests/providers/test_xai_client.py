"""Direct xAI wire contracts, exercised without network credentials."""

from unittest.mock import AsyncMock, patch

import openai
import httpx
import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from miiflow_agent import LLMClient
from miiflow_agent.core import Message, MessageRole
from miiflow_agent.core.exceptions import AuthenticationError
from miiflow_agent.providers.xai_client import XAIClient


@pytest.fixture
def client():
    with patch("miiflow_agent.providers.xai_client.openai.AsyncOpenAI") as sdk:
        sdk.return_value.chat.completions.create = AsyncMock()
        yield XAIClient("grok-4.7", api_key="test-key")


def completion():
    return ChatCompletion.model_validate(
        {
            "id": "chat-1",
            "object": "chat.completion",
            "created": 0,
            "model": "grok-4.7",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Done"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 30,
                "total_tokens": 130,
                "prompt_tokens_details": {"cached_tokens": 80},
                "completion_tokens_details": {"reasoning_tokens": 20},
            },
        }
    )


class Stream:
    def __init__(self, chunks):
        self.chunks = chunks
        self.close = AsyncMock()

    async def __aiter__(self):
        for chunk in self.chunks:
            yield ChatCompletionChunk.model_validate(
                {
                    "id": "chat-1",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "grok-4.7",
                    **chunk,
                }
            )


@pytest.mark.asyncio
async def test_factory_environment_key(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    with (
        patch("miiflow_agent.utils.env.load_env_file"),
        patch("miiflow_agent.providers.xai_client.openai.AsyncOpenAI") as sdk,
    ):
        wrapper = LLMClient.create("xai", "grok-4.7")
    assert wrapper.client.provider_name == "xai"
    sdk.assert_called_once_with(api_key="xai-test", base_url="https://api.x.ai/v1")


@pytest.mark.asyncio
async def test_chat_request_and_usage(client):
    client.client.chat.completions.create.return_value = completion()
    response = await client.achat(
        [Message(role=MessageRole.USER, content="Hello")],
        max_tokens=1200,
        reasoning_effort="low",
        top_p=0.8,
        frequency_penalty=1,
        tools=[
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": {"type": "object"}},
            }
        ],
        tool_choice="required",
        json_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
    )
    params = client.client.chat.completions.create.call_args.kwargs
    assert params["reasoning_effort"] == "low"
    assert params["max_tokens"] == 1200
    assert params["top_p"] == 0.8
    assert params["tool_choice"] == "required"
    assert "frequency_penalty" not in params
    assert (
        params["response_format"]["json_schema"]["schema"]["additionalProperties"]
        is False
    )
    assert response.provider == "xai"
    assert response.usage.cache_read_tokens == 80
    assert response.usage.reasoning_tokens == 20
    assert response.usage.total_tokens == 130


@pytest.mark.asyncio
async def test_stream_tools_and_usage_only_chunk(client):
    stream = Stream(
        [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": '{"q":',
                                    },
                                }
                            ]
                        },
                    }
                ]
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '"ok"}'}}
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
            {"choices": [], "usage": completion().usage.model_dump()},
        ]
    )
    client.client.chat.completions.create.return_value = stream
    chunks = [c async for c in client.astream_chat([], reasoning_effort="medium")]
    assert chunks[-2].tool_calls[0]["function"]["arguments"] == '{"q":"ok"}'
    assert chunks[-1].usage.cache_read_tokens == 80
    assert chunks[-1].usage.reasoning_tokens == 20
    params = client.client.chat.completions.create.call_args.kwargs
    assert params["stream_options"] == {"include_usage": True}
    assert params["reasoning_effort"] == "medium"
    stream.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_stream_closes_on_early_exit(client):
    stream = Stream([{"choices": [{"index": 0, "delta": {"content": "hello"}}]}])
    client.client.chat.completions.create.return_value = stream
    generator = client.astream_chat([])
    await anext(generator)
    await generator.aclose()
    stream.close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_authentication_errors_are_provider_errors(client, streaming):
    client.client.chat.completions.create.side_effect = openai.AuthenticationError(
        "bad key",
        response=httpx.Response(
            401, request=httpx.Request("POST", "https://api.x.ai/v1/chat/completions")
        ),
        body=None,
    )
    with pytest.raises(AuthenticationError):
        if streaming:
            await anext(client.astream_chat([]))
        else:
            await client.achat([])


def test_effort_default_and_override(client):
    client.effort = "low"
    assert client._with_configured_effort({})["reasoning_effort"] == "low"
    assert (
        client._with_configured_effort({"reasoning_effort": "high"})["reasoning_effort"]
        == "high"
    )
    client.effort_overrides_call = True
    assert (
        client._with_configured_effort({"reasoning_effort": "high"})["reasoning_effort"]
        == "low"
    )
    client.effort = "none"
    assert client._with_configured_effort({}) == {}
    assert (
        client._with_configured_effort({"reasoning_effort": "high"})["reasoning_effort"]
        == "high"
    )


@pytest.mark.asyncio
async def test_sdk_cache_isolates_credentials_and_endpoints():
    with patch("miiflow_agent.providers.xai_client.openai.AsyncOpenAI") as sdk:
        sdk.side_effect = lambda **kwargs: object()
        first = XAIClient("grok-4.7", api_key="first")
        same = XAIClient("grok-4.3", api_key="first")
        other_key = XAIClient("grok-4.7", api_key="second")
        regional = XAIClient(
            "grok-4.7", api_key="first", base_url="https://us.api.x.ai/v1"
        )
        assert first.client is same.client
        assert first.client is not other_key.client
        assert first.client is not regional.client
        assert sdk.call_count == 3


@pytest.mark.asyncio
async def test_interleaved_streams_do_not_share_tool_accumulators(client):
    def tool_stream(call_id, name):
        return Stream(
            [
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": call_id,
                                        "type": "function",
                                        "function": {
                                            "name": name,
                                            "arguments": '{"q":',
                                        },
                                    }
                                ]
                            },
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {"index": 0, "function": {"arguments": "1}"}}
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
            ]
        )

    client.client.chat.completions.create.side_effect = [
        tool_stream("a", "alpha"),
        tool_stream("b", "beta"),
    ]
    first, second = client.astream_chat([]), client.astream_chat([])
    try:
        await anext(first)
        await anext(second)
        a, b = await anext(first), await anext(second)
        assert a.tool_calls[0]["id"] == "a"
        assert b.tool_calls[0]["id"] == "b"
        assert a.tool_calls[0]["function"] == {"name": "alpha", "arguments": '{"q":1}'}
        assert b.tool_calls[0]["function"] == {"name": "beta", "arguments": '{"q":1}'}
    finally:
        await first.aclose()
        await second.aclose()


@pytest.mark.asyncio
async def test_image_and_tool_result_round_trip(client):
    from miiflow_agent.core.message import ImageBlock, TextBlock

    reply = completion()
    reply.choices[0].message.tool_calls = [
        openai.types.chat.ChatCompletionMessageFunctionToolCall(
            id="call-1", type="function", function={"name": "lookup", "arguments": "{}"}
        )
    ]
    client.client.chat.completions.create.return_value = reply
    response = await client.achat(
        [
            Message(
                role=MessageRole.USER,
                content=[
                    TextBlock(text="Describe"),
                    ImageBlock(image_url="https://example.com/image.png"),
                ],
            ),
        ]
    )
    assert (
        client.client.chat.completions.create.call_args.kwargs["messages"][0][
            "content"
        ][1]["type"]
        == "image_url"
    )
    await client.achat(
        [
            response.message,
            Message(role=MessageRole.TOOL, content="Found", tool_call_id="call-1"),
        ]
    )
    messages = client.client.chat.completions.create.call_args.kwargs["messages"]
    assert messages[0]["tool_calls"][0].function.name == "lookup"
    assert messages[1]["tool_call_id"] == "call-1"


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_unsupported_configured_effort_drops_but_explicit_effort_errors(
    client, streaming
):
    client.effort = "max"
    if streaming:
        client.client.chat.completions.create.return_value = Stream([])
        assert [c async for c in client.astream_chat([])] == []
    else:
        client.client.chat.completions.create.return_value = completion()
        await client.achat([])
    assert (
        "reasoning_effort" not in client.client.chat.completions.create.call_args.kwargs
    )
    with pytest.raises(ValueError, match="Unsupported reasoning effort"):
        if streaming:
            await anext(client.astream_chat([], reasoning_effort="none"))
        else:
            await client.achat([], reasoning_effort="none")
    assert client.client.chat.completions.create.await_count == 1


@pytest.mark.asyncio
async def test_cumulative_usage_emitted_only_once(client):
    usage = completion().usage.model_dump()
    stream = Stream(
        [
            {"choices": [{"index": 0, "delta": {"content": "Hello"}}], "usage": usage},
            {
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": usage,
            },
            {"choices": [], "usage": usage},
        ]
    )
    client.client.chat.completions.create.return_value = stream
    chunks = [c async for c in client.astream_chat([])]
    assert len([c for c in chunks if c.usage]) == 1
    assert sum(c.usage.total_tokens for c in chunks if c.usage) == 130
    assert chunks[-1].usage.cache_read_tokens == 80
    assert chunks[-1].usage.reasoning_tokens == 20
    stream.close.assert_awaited_once()
