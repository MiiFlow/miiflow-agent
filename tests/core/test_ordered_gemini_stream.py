from miiflow_agent.core.stream_normalizer import GeminiStreamNormalizer


def test_mixed_parts_preserve_order_thought_channel_signatures_and_usage():
    normalizer = GeminiStreamNormalizer()
    chunks = list(
        normalizer.normalize_ordered_chunks(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "Reasoning", "thought": True},
                                {"text": "Checking"},
                                {
                                    "functionCall": {
                                        "name": "lookup",
                                        "args": {"id": 1},
                                    },
                                    "thoughtSignature": "opaque",
                                },
                                {"text": "Result"},
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 4,
                    "candidatesTokenCount": 5,
                    "totalTokenCount": 9,
                },
            }
        )
    )
    assert len(chunks) == 4
    assert chunks[0].thinking_delta == "Reasoning"
    assert chunks[0].delta == ""
    assert chunks[1].delta == "Checking"
    assert (
        chunks[2].tool_calls[0]["function_call_metadata"]["thought_signature"]
        == "opaque"
    )
    assert chunks[3].delta == "Result"
    assert chunks[3].content == "CheckingResult"
    assert all(c.usage is None and c.finish_reason is None for c in chunks[:-1])
    assert chunks[-1].usage is not None
    assert chunks[-1].finish_reason == "STOP"


def test_parallel_same_name_calls_remain_distinct():
    parts = [{"functionCall": {"name": "lookup", "args": {"id": i}}} for i in range(2)]
    chunks = list(
        GeminiStreamNormalizer().normalize_ordered_chunks(
            {"candidates": [{"content": {"parts": parts}}]}
        )
    )
    assert chunks[0].tool_calls[0]["id"] != chunks[1].tool_calls[0]["id"]


def test_signed_text_and_function_parts_round_trip_without_merging():
    from miiflow_agent.core.message import Message, MessageRole
    from miiflow_agent.providers.gemini_client import (
        GeminiClient,
        _convert_to_rest_format,
    )

    parts = [
        {
            "text": "Reasoning",
            "thought": True,
            "thoughtSignature": "reasoning-signature",
        },
        {"text": "Checking", "thoughtSignature": "text-signature"},
        {
            "functionCall": {"name": "lookup", "args": {}},
            "thoughtSignature": "call-signature",
        },
        {"text": "", "thoughtSignature": "empty-signature"},
    ]
    client = GeminiClient(api_key="test-key", model="gemini-2.5-flash")
    message = Message(
        role=MessageRole.ASSISTANT,
        content="Checking",
        metadata={"gemini_content_parts": parts},
    )
    import asyncio

    converted = asyncio.run(client._convert_messages_to_gemini_format([message]))
    assert _convert_to_rest_format(converted)[0]["parts"] == parts


def test_call_ids_stay_unique_across_requests_in_one_turn():
    """Each ReAct step is a new request; ids restarting per request made step 2's
    first call share step 1's id, so consumers pairing by id merged the two."""
    from miiflow_agent.providers.gemini_client import _parse_rest_response

    response = {"candidates": [{"content": {"parts": [{"functionCall": {"name": "tool_call", "args": {}}}]}}]}
    normalizer = GeminiStreamNormalizer()
    streamed = []
    for _ in range(2):
        normalizer.reset_state()
        streamed += [c.tool_calls[0]["id"] for c in normalizer.normalize_ordered_chunks(response)]
    parsed = [_parse_rest_response(response, {})[1][0]["id"] for _ in range(2)]
    assert len(set(streamed + parsed)) == 4


def test_wire_call_id_is_kept():
    from miiflow_agent.providers.gemini_client import _parse_rest_response

    response = {"candidates": [{"content": {"parts": [{"functionCall": {"id": "w1", "name": "lookup"}}]}}]}
    (chunk,) = GeminiStreamNormalizer().normalize_ordered_chunks(response)
    assert chunk.tool_calls[0]["id"] == "w1"
    assert _parse_rest_response(response, {})[1][0]["id"] == "w1"
