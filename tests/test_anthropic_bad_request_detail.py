"""A rejected tool schema names the tool, not just Anthropic's array index."""

from miiflow_agent.providers.anthropic_client import AnthropicClient


def _params():
    return {"tools": [{"name": f"tool_{i}", "input_schema": {}} for i in range(20)]}


def test_tool_index_in_the_message_resolves_to_the_tool_name():
    err = ValueError(
        "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', "
        "'message': 'tools.17.custom.input_schema: JSON schema is invalid.'}}"
    )
    detail = AnthropicClient._bad_request_detail(err, _params())
    assert detail.endswith("[tool 17 = tool_17]")


def test_a_message_without_an_index_is_returned_verbatim():
    err = ValueError("Error code: 400 - prompt is too long")
    assert AnthropicClient._bad_request_detail(err, _params()) == str(err)


def test_an_index_past_the_array_is_left_alone():
    err = ValueError("tools.99.custom.input_schema: JSON schema is invalid.")
    assert AnthropicClient._bad_request_detail(err, {"tools": []}) == str(err)
