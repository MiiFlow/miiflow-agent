"""Provider-executed MCP calls must record the arguments they actually sent.

OpenAI's Responses API returns `mcp_call.arguments` as a JSON string, so the
old dict-only check recorded `input: {}` for every native-MCP call on that
path — an empty, blameless entry in the timeline for a call the remote server
rejected on its arguments.
"""

from miiflow_agent.core.react.recording import _call_arguments


def test_dict_arguments_pass_through():
    assert _call_arguments({"limit": 10}) == {"limit": 10}


def test_json_string_arguments_are_parsed():
    assert _call_arguments('{"__description": "Get session context"}') == {
        "__description": "Get session context"
    }


def test_empty_and_missing_arguments_are_empty_dicts():
    assert _call_arguments(None) == {}
    assert _call_arguments("") == {}
    assert _call_arguments("   ") == {}
    assert _call_arguments("{}") == {}


def test_unparseable_or_non_object_arguments_never_raise():
    assert _call_arguments("{not json") == {}
    assert _call_arguments("[1, 2]") == {}
    assert _call_arguments(42) == {}


def test_openai_declares_mcp_call_arguments_as_a_string():
    """The reason the dict-only check dropped EVERY call, not some.

    `McpCall.arguments` is typed `str` in the SDK, so on OpenAI's native-MCP
    path the old `isinstance(..., dict)` was never once true — every
    provider-executed call recorded an empty input. Pinned against the vendor
    type so a future SDK that starts sending objects is a test failure rather
    than a silent behaviour change.
    """
    from openai.types.responses.response_output_item import McpCall

    assert McpCall.__annotations__["arguments"] is str


def test_anthropic_shape_still_passes_through_as_a_dict():
    """The other live branch: `anthropic_client` yields `block.input`, already
    parsed. Both providers feed this one function."""
    assert _call_arguments({"query": "q"}) == {"query": "q"}
