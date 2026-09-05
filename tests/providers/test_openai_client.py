"""Tests for OpenAI provider client."""

import httpx
import openai
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List

from miiflow_agent.providers.openai_client import OpenAIClient
from miiflow_agent.core import (
    Message,
    MessageRole,
    TokenCount,
    StreamChunk,
    ChatResponse,
)
from miiflow_agent.core.exceptions import ModelError
from miiflow_agent.core.message import DocumentBlock, TextBlock, VideoBlock


def _responses_response(text="ok", *, tool_calls=None):
    output = [
        SimpleNamespace(
            type="message",
            content=[SimpleNamespace(type="output_text", text=text)],
        )
    ]
    for tool_call in tool_calls or []:
        output.append(
            SimpleNamespace(
                type="function_call",
                call_id=tool_call["id"],
                name=tool_call["name"],
                arguments=tool_call.get("arguments", "{}"),
            )
        )
    return SimpleNamespace(
        id="resp_test",
        status="completed",
        output=output,
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            input_tokens_details=SimpleNamespace(cached_tokens=2, cache_write_tokens=3),
            output_tokens_details=SimpleNamespace(reasoning_tokens=4),
        ),
    )


def _function_call_events(item_id, call_id, name, arguments, *, index=0):
    """The real event sequence for one streamed function call: the name and
    call_id ride on the output-item events, the arguments-done event has
    neither (its `name` is None on the wire)."""
    started = SimpleNamespace(
        type="function_call", id=item_id, call_id=call_id, name=name, arguments=""
    )
    done = SimpleNamespace(
        type="function_call",
        id=item_id,
        call_id=call_id,
        name=name,
        arguments=arguments,
        status="completed",
    )
    return [
        SimpleNamespace(type="response.output_item.added", output_index=index, item=started),
        SimpleNamespace(
            type="response.function_call_arguments.delta",
            item_id=item_id,
            output_index=index,
            delta=arguments,
        ),
        SimpleNamespace(
            type="response.function_call_arguments.done",
            item_id=item_id,
            output_index=index,
            name=None,
            arguments=arguments,
        ),
        SimpleNamespace(type="response.output_item.done", output_index=index, item=done),
    ]


def _responses_stream(events, *, terminal=None):
    async def _gen():
        for event in events:
            yield event
        yield terminal or SimpleNamespace(type="response.completed", response=_responses_response())

    return _gen()


def _final_tool_calls(chunks):
    """Last emission per call id — the finalized item that replaces the
    announce-time placeholder, as the orchestrator merges them."""
    final = {}
    for chunk in chunks:
        for call in chunk.tool_calls or []:
            final[call["id"]] = call
    return list(final.values())


class TestOpenAIClient:
    """Test suite for OpenAI client."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return OpenAIClient(model="gpt-4.1-nano", api_key="test-key", timeout=30.0)

    @pytest.mark.asyncio
    async def test_client_initialization(self, client):
        """Test client initialization."""
        assert client.model == "gpt-4.1-nano"
        assert client.api_key == "test-key"
        assert client.timeout == 30.0
        assert client.provider_name == "openai"

    @pytest.mark.asyncio
    async def test_chat_completion_success(self, client, sample_messages, mock_openai_response):
        """Test successful chat completion."""
        with patch.object(
            client.client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_openai_response

            response = await client.achat(sample_messages)

            # Verify response format
            assert isinstance(response, ChatResponse)
            assert response.message.role == MessageRole.ASSISTANT
            assert response.message.content == "Hello! I'm doing well, thank you for asking."
            assert response.usage.prompt_tokens == 10
            assert response.usage.completion_tokens == 20
            assert response.usage.total_tokens == 30
            assert response.model == "gpt-4.1-nano"
            assert response.provider == "openai"
            assert response.finish_reason == "stop"

            # Verify API call
            mock_create.assert_called_once()
            call_args = mock_create.call_args
            assert call_args.kwargs["model"] == "gpt-4.1-nano"
            assert len(call_args.kwargs["messages"]) == 2

    @pytest.mark.asyncio
    async def test_stream_chat_success(self, client, sample_messages, mock_openai_stream_chunks):
        """Test successful streaming chat."""

        async def mock_stream_generator():
            for chunk in mock_openai_stream_chunks:
                yield chunk

        with patch.object(
            client.client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
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

            # Verify API call
            mock_create.assert_called_once()
            call_args = mock_create.call_args
            assert call_args.kwargs["stream"] is True

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

        with patch.object(
            client.client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_response

            await client.achat(sample_messages, temperature=0.9, max_tokens=100)

            call_args = mock_create.call_args
            assert call_args.kwargs["temperature"] == 0.9
            assert call_args.kwargs["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_chat_with_tools(self, client, sample_messages):
        """Test chat with tool calls."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information",
                },
            }
        ]

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_response.choices[0].message.role = "assistant"
        mock_response.choices[0].message.tool_calls = [
            {"id": "call_123", "function": {"name": "get_weather"}}
        ]
        mock_response.choices[0].finish_reason = "tool_calls"
        mock_response.usage.prompt_tokens = 15
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 20

        with patch.object(
            client.client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_response

            response = await client.achat(sample_messages, tools=tools)

            assert response.finish_reason == "tool_calls"
            assert response.message.tool_calls is not None
            assert len(response.message.tool_calls) == 1

            call_args = mock_create.call_args
            assert call_args.kwargs["tools"] == tools

    @pytest.mark.asyncio
    async def test_message_conversion(self, client):
        """Test message format conversion through to_openai_format."""
        messages = [
            Message.system("You are helpful."),
            Message.user("Hello"),
            Message.assistant("Hi there!"),
        ]

        # Test that messages can be converted to OpenAI format using provider
        converted = [client.convert_message_to_provider_format(msg) for msg in messages]

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
        from miiflow_agent.core.message import TextBlock, ImageBlock

        multimodal_message = Message.user(
            [
                TextBlock(text="What's in this image?"),
                ImageBlock(image_url="data:image/jpeg;base64,/9j/4AAQSkZJRg...", detail="high"),
            ]
        )

        # Test multimodal message conversion through provider
        converted = client.convert_message_to_provider_format(multimodal_message)

        assert converted["role"] == "user"
        # Basic validation that multimodal conversion works
        assert "content" in converted

    @pytest.mark.asyncio
    async def test_error_handling(self, client, sample_messages):
        """Test error handling in chat completion."""
        from miiflow_agent.core.exceptions import ProviderError

        with patch.object(
            client.client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
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

        with patch.object(
            client.client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = error_generator()

            with pytest.raises(ProviderError):
                chunks = []
                async for chunk in client.astream_chat(sample_messages):
                    chunks.append(chunk)


class TestReasoningEffort:
    """Endpoint-aware ``reasoning_effort`` forwarding and compatibility fallback.

    GPT-5.6 tool turns use Responses so effort remains active. The narrow Chat
    fallback remains for callers that bypass endpoint routing.
    """

    _TOOLS = [
        {
            "type": "function",
            "function": {"name": "get_weather", "description": "", "parameters": {}},
        }
    ]

    def _client(self, model="gpt-5.6-terra"):
        return OpenAIClient(model=model, api_key="test-key", timeout=30.0)

    @pytest.mark.asyncio
    async def test_reasoning_effort_forwarded_without_tools(
        self, sample_messages, mock_openai_response
    ):
        client = self._client()
        with patch.object(
            client.client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_openai_response
            await client.achat(sample_messages, reasoning_effort="high")

        assert mock_create.call_args.kwargs.get("reasoning_effort") == "high"

    @pytest.mark.asyncio
    async def test_reasoning_effort_with_tools_uses_responses(self, sample_messages):
        client = self._client()
        with patch.object(client.client.responses, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = _responses_response()
            await client.achat(sample_messages, tools=self._TOOLS, reasoning_effort="high")

        request = mock_create.call_args.kwargs
        assert request["reasoning"] == {"effort": "high"}
        assert request["tools"] == [
            {
                "type": "function",
                "name": "get_weather",
                "description": "",
                "parameters": {},
            }
        ]

    @pytest.mark.asyncio
    async def test_reasoning_effort_dropped_for_unsupported_model(
        self, sample_messages, mock_openai_response
    ):
        client = self._client(model="gpt-4.1-nano")
        with patch.object(
            client.client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_openai_response
            await client.achat(sample_messages, reasoning_effort="high")

        assert "reasoning_effort" not in mock_create.call_args.kwargs

    @pytest.mark.asyncio
    async def test_reasoning_effort_with_tools_streaming_uses_responses(self, sample_messages):
        client = self._client()

        async def _gen():
            yield SimpleNamespace(type="response.completed", response=_responses_response())

        with patch.object(client.client.responses, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = _gen()
            async for _ in client.astream_chat(
                sample_messages, tools=self._TOOLS, reasoning_effort="high"
            ):
                pass

        assert mock_create.call_args.kwargs["reasoning"] == {"effort": "high"}

    @pytest.mark.asyncio
    async def test_exact_tools_reasoning_400_retries_once_without_effort(
        self, mock_openai_response
    ):
        client = self._client()
        error = openai.BadRequestError(
            message=(
                "Function tools with reasoning_effort are not supported for "
                "gpt-5.6-terra in /v1/chat/completions."
            ),
            response=httpx.Response(
                400,
                request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
            ),
            body=None,
        )
        request_params = {
            "model": "gpt-5.6-terra",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": self._TOOLS,
            "reasoning_effort": "high",
        }

        with patch.object(
            client.client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.side_effect = [error, mock_openai_response]
            response = await client._create_chat_completion(request_params)

        assert response is mock_openai_response
        assert mock_create.await_count == 2
        assert mock_create.await_args_list[0].kwargs["reasoning_effort"] == "high"
        assert "reasoning_effort" not in mock_create.await_args_list[1].kwargs

    @pytest.mark.asyncio
    async def test_unrelated_400_is_not_retried(self):
        client = self._client()
        error = openai.BadRequestError(
            message="Invalid tool schema",
            response=httpx.Response(
                400,
                request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
            ),
            body=None,
        )

        with patch.object(
            client.client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.side_effect = error
            with pytest.raises(openai.BadRequestError):
                await client._create_chat_completion(
                    {
                        "model": "gpt-5.6-terra",
                        "messages": [{"role": "user", "content": "hello"}],
                        "reasoning_effort": "high",
                    }
                )

        mock_create.assert_awaited_once()


class TestOpenAIRequestMapping:
    _TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look something up",
                "parameters": {"type": "object", "properties": {}},
                "strict": True,
            },
        }
    ]

    @pytest.mark.asyncio
    async def test_chat_forwards_supported_parameters_and_fine_tune_override(
        self, sample_messages, mock_openai_response
    ):
        client = OpenAIClient(model="gpt-4.1", api_key="test-key")
        with patch.object(
            client.client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_openai_response
            await client.achat(
                sample_messages,
                tools=self._TOOLS,
                tool_choice="none",
                frequency_penalty=1.2,
                presence_penalty=-0.4,
                fine_tuned_model="ft:gpt-4.1:miiflow:test",
                seed=42,
            )

        request = mock_create.call_args.kwargs
        assert request["model"] == "ft:gpt-4.1:miiflow:test"
        assert request["tool_choice"] == "none"
        assert request["frequency_penalty"] == 1.2
        assert request["presence_penalty"] == -0.4
        assert request["seed"] == 42

    @pytest.mark.asyncio
    async def test_chat_forwards_verbosity(self, sample_messages, mock_openai_response):
        client = OpenAIClient(model="gpt-5.5", api_key="test-key")
        with patch.object(
            client.client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_openai_response
            await client.achat(sample_messages, verbosity="low")

        assert mock_create.call_args.kwargs["verbosity"] == "low"

    @pytest.mark.asyncio
    async def test_responses_forwards_reasoning_text_schema_and_tool_choice(self, sample_messages):
        client = OpenAIClient(model="gpt-5.6-terra", api_key="test-key")
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }
        with patch.object(client.client.responses, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = _responses_response()
            await client.achat(
                sample_messages,
                tools=self._TOOLS,
                json_schema=schema,
                reasoning_effort="max",
                reasoning_mode="pro",
                verbosity="high",
                tool_choice={"type": "function", "function": {"name": "lookup"}},
                prompt_cache_key="stable-prefix",
                max_tokens=321,
            )

        request = mock_create.call_args.kwargs
        assert request["model"] == "gpt-5.6-terra"
        assert request["reasoning"] == {"effort": "max", "mode": "pro"}
        assert request["text"]["verbosity"] == "high"
        assert request["text"]["format"] == {
            "type": "json_schema",
            "name": "response_schema",
            "schema": {**schema, "additionalProperties": False},
            "strict": True,
        }
        assert request["tool_choice"] == {"type": "function", "name": "lookup"}
        assert request["prompt_cache_key"] == "stable-prefix"
        assert request["max_output_tokens"] == 321
        assert request["tools"][0]["strict"] is True

    @pytest.mark.asyncio
    async def test_current_endpoint_options_are_forwarded(self, sample_messages):
        client = OpenAIClient(model="gpt-5.6-terra", api_key="test-key")
        with patch.object(client.client.responses, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = _responses_response()
            await client.achat(
                sample_messages,
                context_management=[{"type": "compact", "compact_threshold": 1000}],
                moderation={"model": "omni-moderation-latest"},
                prompt_cache_options={"mode": "explicit", "ttl": "30m"},
                prompt_cache_retention="24h",
            )

        request = mock_create.call_args.kwargs
        assert request["context_management"][0]["type"] == "compact"
        assert request["moderation"]["model"] == "omni-moderation-latest"
        assert request["prompt_cache_options"]["mode"] == "explicit"
        assert request["prompt_cache_retention"] == "24h"

    @pytest.mark.asyncio
    async def test_legacy_sol_pro_alias_maps_to_sol_pro_mode(self, sample_messages):
        client = OpenAIClient(model="gpt-5.6-sol-pro", api_key="test-key")
        with patch.object(client.client.responses, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = _responses_response()
            await client.achat(sample_messages)

        request = mock_create.call_args.kwargs
        assert request["model"] == "gpt-5.6-sol"
        assert request["reasoning"] == {"mode": "pro"}

    @pytest.mark.asyncio
    async def test_gpt54_pro_always_uses_responses(self, sample_messages):
        client = OpenAIClient(model="gpt-5.4-pro", api_key="test-key")
        with patch.object(client.client.responses, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = _responses_response()
            await client.achat(sample_messages, reasoning_effort="xhigh")

        assert mock_create.call_args.kwargs["reasoning"] == {"effort": "xhigh"}

    @pytest.mark.asyncio
    async def test_nonstreaming_pro_model_adapts_to_stream_interface(self, sample_messages):
        client = OpenAIClient(model="gpt-5.5-pro", api_key="test-key")
        with patch.object(client.client.responses, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = _responses_response("complete answer")
            chunks = [chunk async for chunk in client.astream_chat(sample_messages)]

        assert mock_create.call_args.kwargs.get("stream") is None
        assert len(chunks) == 1
        assert chunks[0].content == "complete answer"
        assert chunks[0].usage.cache_write_tokens == 3

    @pytest.mark.asyncio
    async def test_background_response_is_polled_and_adapted_to_stream(self, sample_messages):
        client = OpenAIClient(model="gpt-5.6-sol", api_key="test-key")
        queued = SimpleNamespace(id="resp_bg", status="queued")
        completed = _responses_response("background answer")
        with (
            patch.object(
                client.client.responses,
                "create",
                new_callable=AsyncMock,
                return_value=queued,
            ) as mock_create,
            patch.object(
                client.client.responses,
                "retrieve",
                new_callable=AsyncMock,
                return_value=completed,
            ) as mock_retrieve,
            patch(
                "miiflow_agent.providers.openai_client.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            chunks = [
                chunk
                async for chunk in client.astream_chat(
                    sample_messages, use_responses_api=True, background=True
                )
            ]

        assert mock_create.call_args.kwargs["background"] is True
        mock_retrieve.assert_awaited_once_with("resp_bg")
        assert [chunk.content for chunk in chunks] == ["background answer"]

    @pytest.mark.asyncio
    async def test_responses_stream_restores_sanitized_tool_name(self, sample_messages):
        client = OpenAIClient(model="gpt-5.6-terra", api_key="test-key")
        tools = [
            client.convert_schema_to_provider_format(
                {"name": "lookup.item", "description": "", "parameters": {}}
            )
        ]

        with patch.object(
            client.client.responses,
            "create",
            new_callable=AsyncMock,
            return_value=_responses_stream(
                _function_call_events("fc_1", "call_3", "lookup_item", '{"q":"x"}')
            ),
        ):
            chunks = [chunk async for chunk in client.astream_chat(sample_messages, tools=tools)]

        final = _final_tool_calls(chunks)
        assert final == [
            {
                "id": "call_3",
                "type": "function",
                "index": 0,
                "function": {"name": "lookup.item", "arguments": '{"q":"x"}'},
            }
        ]

    @pytest.mark.asyncio
    async def test_responses_stream_names_tool_call_from_output_item(self, sample_messages):
        """The name and call_id are NOT on `function_call_arguments.done`.

        Live Responses API (gpt-5.6-terra, 2026-09-01): that event carries
        only item_id + arguments, `name` is None. Reading the name there
        produced a nameless call the orchestrator rejected as malformed —
        three in a row halted the turn. The name must come from the
        output-item events.
        """
        client = OpenAIClient(model="gpt-5.6-terra", api_key="test-key")
        tools = [
            client.convert_schema_to_provider_format(
                {"name": "generate_image", "description": "", "parameters": {}}
            )
        ]

        with patch.object(
            client.client.responses,
            "create",
            new_callable=AsyncMock,
            return_value=_responses_stream(
                _function_call_events("fc_1", "call_1", "generate_image", '{"prompt":"a dog"}')
            ),
        ):
            chunks = [chunk async for chunk in client.astream_chat(sample_messages, tools=tools)]

        # The call is announced as soon as its item starts, already named …
        first = next(c for c in chunks if c.tool_calls)
        assert first.tool_calls[0]["function"]["name"] == "generate_image"
        assert first.tool_calls[0]["id"] == "call_1"
        # … and finalized with the complete arguments.
        final = _final_tool_calls(chunks)
        assert len(final) == 1
        assert final[0]["id"] == "call_1"
        assert final[0]["function"] == {
            "name": "generate_image",
            "arguments": '{"prompt":"a dog"}',
        }

    @pytest.mark.asyncio
    async def test_responses_stream_keeps_parallel_calls_distinct(self, sample_messages):
        client = OpenAIClient(model="gpt-5.6-terra", api_key="test-key")
        events = _function_call_events("fc_1", "call_1", "a", '{"x":1}', index=0)
        events += _function_call_events("fc_2", "call_2", "b", '{"y":2}', index=1)

        with patch.object(
            client.client.responses,
            "create",
            new_callable=AsyncMock,
            return_value=_responses_stream(events),
        ):
            chunks = [
                chunk async for chunk in client.astream_chat(sample_messages, tools=self._TOOLS)
            ]

        final = _final_tool_calls(chunks)
        assert [(c["index"], c["id"], c["function"]["name"]) for c in final] == [
            (0, "call_1", "a"),
            (1, "call_2", "b"),
        ]

    @pytest.mark.asyncio
    async def test_responses_stream_captures_usage_on_completed(self, sample_messages):
        """`response.completed` is the terminal event — not `response.done`,
        which never fires. Matching the wrong name meant streamed Responses
        calls reported no usage and billed as zero tokens."""
        client = OpenAIClient(model="gpt-5.6-terra", api_key="test-key")

        with patch.object(
            client.client.responses,
            "create",
            new_callable=AsyncMock,
            return_value=_responses_stream(
                [SimpleNamespace(type="response.output_text.delta", delta="hi")]
            ),
        ):
            chunks = [
                chunk
                async for chunk in client.astream_chat(sample_messages, use_responses_api=True)
            ]

        last = chunks[-1]
        assert last.finish_reason == "stop"
        assert last.usage is not None
        assert last.usage.prompt_tokens == 10
        assert last.usage.completion_tokens == 20

    @pytest.mark.asyncio
    async def test_responses_stream_incomplete_max_tokens_is_length(self, sample_messages):
        client = OpenAIClient(model="gpt-5.6-terra", api_key="test-key")
        response = _responses_response()
        response.incomplete_details = SimpleNamespace(reason="max_output_tokens")

        with patch.object(
            client.client.responses,
            "create",
            new_callable=AsyncMock,
            return_value=_responses_stream(
                [], terminal=SimpleNamespace(type="response.incomplete", response=response)
            ),
        ):
            chunks = [
                chunk
                async for chunk in client.astream_chat(sample_messages, use_responses_api=True)
            ]

        assert chunks[-1].finish_reason == "length"

    @pytest.mark.asyncio
    async def test_responses_stream_yields_native_mcp_call_with_result(self, sample_messages):
        client = OpenAIClient(model="gpt-5.6-terra", api_key="test-key")
        item = SimpleNamespace(
            type="mcp_call",
            id="mcp_1",
            name="list_prs",
            arguments='{"repo":"x"}',
            server_label="github",
            output="3 open PRs",
            error=None,
        )

        with patch.object(
            client.client.responses,
            "create",
            new_callable=AsyncMock,
            return_value=_responses_stream(
                [SimpleNamespace(type="response.output_item.done", output_index=0, item=item)]
            ),
        ):
            chunks = [
                chunk
                async for chunk in client.astream_chat(sample_messages, use_responses_api=True)
            ]

        mcp = next(c for c in chunks if c.tool_calls)
        assert mcp.tool_calls[0]["type"] == "mcp_function"
        assert mcp.tool_calls[0]["function"]["name"] == "list_prs"
        assert mcp.mcp_tool_results == [
            {"tool_use_id": "mcp_1", "is_error": False, "content": "3 open PRs"}
        ]

    @pytest.mark.asyncio
    async def test_structured_outputs_fail_fast_for_gpt54_pro(self, sample_messages):
        client = OpenAIClient(model="gpt-5.4-pro", api_key="test-key")
        with pytest.raises(ModelError, match="Structured outputs are not supported"):
            await client.achat(sample_messages, json_schema={"type": "object"})

    def test_responses_history_replays_assistant_function_calls(self):
        client = OpenAIClient(model="gpt-5.6-terra", api_key="test-key")
        messages = [
            Message.assistant(
                "",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"q":"x"}'},
                    }
                ],
            ),
            Message.tool("result", tool_call_id="call_1"),
        ]

        assert client._convert_messages_to_responses_input(messages) == [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": '{"q":"x"}',
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "result"},
        ]

    def test_responses_preserves_files_video_references_and_dict_arguments(self):
        client = OpenAIClient(model="gpt-5.6-terra", api_key="test-key")
        messages = [
            Message.user(
                [
                    TextBlock(text="Review these"),
                    DocumentBlock(
                        document_url="https://example.com/brief.pdf",
                        filename="brief.pdf",
                    ),
                    DocumentBlock(
                        document_url="data:application/pdf;base64,JVBERi0x",
                        filename="inline.pdf",
                    ),
                    VideoBlock(video_url="https://example.com/demo.mp4"),
                ]
            ),
            Message.assistant(
                "",
                tool_calls=[
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": {"q": "x"}},
                    }
                ],
            ),
            Message.tool(
                [
                    TextBlock(text="tool text"),
                    DocumentBlock(document_url="https://example.com/tool.pdf"),
                ],
                tool_call_id="call_2",
            ),
        ]

        converted = client._convert_messages_to_responses_input(messages)
        # No `filename` beside `file_url`: the Responses API 400s on the pair
        # ("mutually exclusive parameters ... 'file_id' or 'filename'"), which
        # is what killed every turn carrying a URL-backed PDF. The name is not
        # lost — it becomes the label ahead of the file.
        assert converted[0]["content"][1] == {
            "type": "input_text",
            "text": "[Document: brief.pdf]",
        }
        assert converted[0]["content"][2] == {
            "type": "input_file",
            "file_url": "https://example.com/brief.pdf",
        }
        # Inline bytes keep `filename` — that is the one source it pairs with.
        assert converted[0]["content"][3] == {
            "type": "input_file",
            "file_data": "data:application/pdf;base64,JVBERi0x",
            "filename": "inline.pdf",
        }
        assert "https://example.com/demo.mp4" in converted[0]["content"][4]["text"]
        assert converted[1]["arguments"] == '{"q": "x"}'
        # An unnamed URL document gets no label and no `filename`.
        assert converted[2]["output"] == [
            {"type": "input_text", "text": "tool text"},
            {"type": "input_file", "file_url": "https://example.com/tool.pdf"},
        ]


class TestNativeMCPServerIdentity:
    """The shared `mcp_function` shape carries `server_name`, not OpenAI's label.

    Every consumer reads `server_name`: `recording` stamps the event and the
    persisted timeline item from it, and `enhanced_response_generator` DROPS a
    native-MCP timeline item that lacks one rather than replay it as a local
    `tool_use`. Emitting `server_label` here meant every OpenAI-originated
    connector call was recorded with `server_name=None` and then silently
    dropped from turn-boundary replay.
    """

    @pytest.mark.asyncio
    async def test_stream_maps_label_back_to_the_configured_server_name(
        self, sample_messages
    ):
        from miiflow_agent.core.tools.mcp.mcp_connection import NativeMCPServerConfig

        # A display name OpenAI cannot accept verbatim — the sanitizer rewrites
        # the space, which is exactly why the label cannot be reversed by
        # string surgery.
        server = NativeMCPServerConfig(name="Kopperfield Admin", url="https://x/mcp")
        assert server.to_openai_format()["server_label"] == "Kopperfield_Admin"

        client = OpenAIClient(model="gpt-5.6-terra", api_key="test-key")
        item = SimpleNamespace(
            type="mcp_call",
            id="mcp_1",
            name="get_load_calc",
            arguments='{"id":"1"}',
            server_label="Kopperfield_Admin",
            output="{}",
            error=None,
        )

        with patch.object(
            client.client.responses,
            "create",
            new_callable=AsyncMock,
            return_value=_responses_stream(
                [SimpleNamespace(type="response.output_item.done", output_index=0, item=item)]
            ),
        ):
            chunks = [
                chunk
                async for chunk in client.astream_chat(
                    sample_messages, use_responses_api=True, mcp_servers=[server]
                )
            ]

        call = next(c for c in chunks if c.tool_calls).tool_calls[0]
        assert call["server_name"] == "Kopperfield Admin"
        assert "server_label" not in call

    @pytest.mark.asyncio
    async def test_unknown_label_falls_back_to_the_label_itself(self, sample_messages):
        """A name is still better than None: None is what gets the call dropped."""
        client = OpenAIClient(model="gpt-5.6-terra", api_key="test-key")
        item = SimpleNamespace(
            type="mcp_call",
            id="mcp_1",
            name="list_prs",
            arguments="{}",
            server_label="github",
            output="ok",
            error=None,
        )

        with patch.object(
            client.client.responses,
            "create",
            new_callable=AsyncMock,
            return_value=_responses_stream(
                [SimpleNamespace(type="response.output_item.done", output_index=0, item=item)]
            ),
        ):
            chunks = [
                chunk
                async for chunk in client.astream_chat(sample_messages, use_responses_api=True)
            ]

        assert next(c for c in chunks if c.tool_calls).tool_calls[0]["server_name"] == "github"


class TestNonStreamingNativeMCP:
    """`_parse_responses_api_response` must handle BOTH halves of an mcp_call.

    Verified live against the Responses API (gpt-5.6-terra + the public
    DeepWiki connector) while checking the `server_name` change: the parser
    took no `mcp_servers` argument at all, so reading one raised
    `NameError: name 'mcp_servers' is not defined` and the whole non-streaming
    request failed. It also kept only the call half, so every provider-executed
    call on this path was recorded as "No result returned by the provider" and
    nothing could replay the pair on the next turn.
    """

    def _response(self):
        return SimpleNamespace(
            id="resp_1",
            status="completed",
            usage=None,
            output=[
                SimpleNamespace(
                    type="mcp_call",
                    id="mcp_1",
                    name="read_wiki_structure",
                    arguments='{"repoName":"x/y"}',
                    server_label="Deep_Wiki",
                    output="# Pages",
                    error=None,
                ),
                SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text", text="It covers x.")],
                ),
            ],
        )

    @pytest.mark.asyncio
    async def test_non_streaming_mcp_call_keeps_its_result_and_server_name(
        self, sample_messages
    ):
        from miiflow_agent.core.tools.mcp.mcp_connection import NativeMCPServerConfig

        server = NativeMCPServerConfig(name="Deep Wiki", url="https://x/mcp")
        client = OpenAIClient(model="gpt-5.6-terra", api_key="test-key")

        with patch.object(
            client.client.responses,
            "create",
            new_callable=AsyncMock,
            return_value=self._response(),
        ):
            response = await client.achat(sample_messages, mcp_servers=[server])

        call = response.message.tool_calls[0]
        assert call["type"] == "mcp_function"
        assert call["server_name"] == "Deep Wiki"
        assert response.metadata["native_mcp"] is True
        assert response.metadata["mcp_tool_results"] == [
            {"tool_use_id": "mcp_1", "is_error": False, "content": "# Pages"}
        ]

    @pytest.mark.asyncio
    async def test_a_failed_non_streaming_mcp_call_carries_its_error(
        self, sample_messages
    ):
        client = OpenAIClient(model="gpt-5.6-terra", api_key="test-key")
        response_obj = self._response()
        response_obj.output[0].error = "upstream 502"
        response_obj.output[0].output = None

        with patch.object(
            client.client.responses,
            "create",
            new_callable=AsyncMock,
            return_value=response_obj,
        ):
            response = await client.achat(sample_messages, use_responses_api=True)

        assert response.metadata["mcp_tool_results"] == [
            {"tool_use_id": "mcp_1", "is_error": True, "content": "upstream 502"}
        ]

    @pytest.mark.asyncio
    async def test_a_turn_with_no_mcp_call_carries_no_mcp_metadata(
        self, sample_messages
    ):
        """Guard against stamping `native_mcp` on every Responses turn."""
        client = OpenAIClient(model="gpt-5.6-terra", api_key="test-key")
        plain = SimpleNamespace(
            id="resp_2",
            status="completed",
            usage=None,
            output=[
                SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text", text="hi")],
                )
            ],
        )

        with patch.object(
            client.client.responses,
            "create",
            new_callable=AsyncMock,
            return_value=plain,
        ):
            response = await client.achat(sample_messages, use_responses_api=True)

        assert "mcp_tool_results" not in response.metadata
        assert "native_mcp" not in response.metadata
