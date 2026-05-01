"""Tests for strict-by-default tool registration and the per-request cap.

Strict mode (`additionalProperties: false` + Anthropic's structured-outputs
flag) prevents the model from passing the wrong tool's parameter shape to
the wrong tool — the exact failure mode CRITICAL RULE 7 in the Adlyse
prompt warns about. We default it on so every new tool gets the
protection without an explicit opt-in. Tools that need the looser shape
(e.g. bare `items={"type":"object"}` arrays) opt out with strict=False.
"""

import pytest

from miiflow_agent.core.tools.decorators import tool


def test_tool_default_is_strict():
    @tool(name="t_default")
    def f(ctx, x: str) -> str:
        return x

    assert f._tool_schema.metadata.get("strict") is True


def test_explicit_strict_false_survives():
    @tool(name="t_loose", strict=False)
    def f(ctx, x: str) -> str:
        return x

    # strict=False must NOT set the metadata flag (so downstream provider
    # adapters fall through to the LOOSE schema mode).
    assert "strict" not in f._tool_schema.metadata


def test_explicit_strict_true_still_works():
    @tool(name="t_strict", strict=True)
    def f(ctx, x: str) -> str:
        return x

    assert f._tool_schema.metadata.get("strict") is True


# ---- _apply_strict_tool_cap ----


def _make_tool(name: str, strict: bool):
    base = {
        "name": name,
        "description": f"tool {name}",
        "input_schema": {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "additionalProperties": False if strict else True,
        },
    }
    if strict:
        base["strict"] = True
    return base


def test_apply_strict_tool_cap_no_op_when_under_limit():
    from miiflow_agent.providers.anthropic_client import AnthropicClient

    tools = [_make_tool(f"t{i}", strict=True) for i in range(20)]
    request_params = {"tools": tools}
    AnthropicClient._apply_strict_tool_cap(request_params)
    assert all(t.get("strict") is True for t in request_params["tools"])


def test_apply_strict_tool_cap_demotes_excess_strict_tail():
    from miiflow_agent.providers.anthropic_client import AnthropicClient

    # 25 strict tools — 5 over the cap. Tail 5 should lose strict.
    tools = [_make_tool(f"t{i}", strict=True) for i in range(25)]
    request_params = {"tools": tools}
    AnthropicClient._apply_strict_tool_cap(request_params)

    new_tools = request_params["tools"]
    strict_count = sum(1 for t in new_tools if t.get("strict"))
    assert strict_count == 20
    # First 20 stay strict
    for t in new_tools[:20]:
        assert t.get("strict") is True
        assert t["input_schema"]["additionalProperties"] is False
    # Last 5 are demoted: no strict flag, additionalProperties relaxed
    for t in new_tools[20:]:
        assert "strict" not in t
        assert t["input_schema"]["additionalProperties"] is True


def test_apply_strict_tool_cap_preserves_non_strict_tools():
    """Non-strict tools shouldn't count toward the cap or be touched."""
    from miiflow_agent.providers.anthropic_client import AnthropicClient

    # 15 strict + 10 non-strict = 25 tools, but only 15 are strict — under cap.
    tools = [_make_tool(f"strict{i}", strict=True) for i in range(15)] + [
        _make_tool(f"loose{i}", strict=False) for i in range(10)
    ]
    request_params = {"tools": tools}
    AnthropicClient._apply_strict_tool_cap(request_params)

    new_tools = request_params["tools"]
    # No demotion — all originally strict tools still strict
    for t in new_tools[:15]:
        assert t.get("strict") is True
    # Non-strict tools untouched
    for t in new_tools[15:]:
        assert "strict" not in t


def test_apply_strict_tool_cap_handles_no_tools():
    from miiflow_agent.providers.anthropic_client import AnthropicClient

    AnthropicClient._apply_strict_tool_cap({})  # no "tools" key
    AnthropicClient._apply_strict_tool_cap({"tools": []})
    AnthropicClient._apply_strict_tool_cap({"tools": None})
    # Should not raise.


def _make_tool_with_optional_params(name: str, optional_count: int, strict: bool = True):
    """Strict tool whose schema contains `optional_count` optional properties."""
    properties = {f"opt{i}": {"type": "string"} for i in range(optional_count)}
    base = {
        "name": name,
        "description": f"tool {name}",
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": [],  # all optional
            "additionalProperties": False if strict else True,
        },
    }
    if strict:
        base["strict"] = True
    return base


def test_apply_strict_tool_cap_enforces_optional_param_cap():
    """Anthropic caps total optional params across strict schemas at 24.

    10 strict tools with 8 optional params each = 80 total. Tool count is
    under the 20-tool cap, but optional params far exceed 24. We expect
    tail demotion until the surviving strict tools fit under 24 optional
    params total.
    """
    from miiflow_agent.providers.anthropic_client import AnthropicClient

    tools = [_make_tool_with_optional_params(f"t{i}", optional_count=8) for i in range(10)]
    request_params = {"tools": tools}
    AnthropicClient._apply_strict_tool_cap(request_params)

    new_tools = request_params["tools"]
    surviving_strict = [t for t in new_tools if t.get("strict")]
    # 8 + 8 + 8 = 24 fits exactly; a fourth tool would push to 32.
    assert len(surviving_strict) == 3
    # Demoted tools (tail) lose strict flag and have additionalProperties relaxed.
    for t in new_tools[3:]:
        assert "strict" not in t
        assert t["input_schema"]["additionalProperties"] is True


def test_apply_strict_tool_cap_counts_nested_optional_params():
    """Nested object properties also count toward Anthropic's optional-params cap."""
    from miiflow_agent.providers.anthropic_client import AnthropicClient

    # Each tool has 1 top-level optional property with 30 nested optional
    # children. A single such tool already busts the 24 cap.
    nested = {
        "type": "object",
        "properties": {f"n{i}": {"type": "string"} for i in range(30)},
        "required": [],
    }
    tool = {
        "name": "nested_strict",
        "description": "nested",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"payload": nested},
            "required": [],
            "additionalProperties": False,
        },
    }
    request_params = {"tools": [tool]}
    AnthropicClient._apply_strict_tool_cap(request_params)

    # Only one strict tool — phase 1 leaves it; phase 2 demotes it because
    # nested optional param count (1 + 30 = 31) exceeds 24.
    assert "strict" not in request_params["tools"][0]
    assert request_params["tools"][0]["input_schema"]["additionalProperties"] is True


def test_apply_strict_tool_cap_no_op_when_optional_params_under_cap():
    from miiflow_agent.providers.anthropic_client import AnthropicClient

    # 5 strict tools, 4 optional params each = 20 total. Under both caps.
    tools = [_make_tool_with_optional_params(f"t{i}", optional_count=4) for i in range(5)]
    request_params = {"tools": tools}
    AnthropicClient._apply_strict_tool_cap(request_params)

    assert all(t.get("strict") is True for t in request_params["tools"])


def test_count_optional_properties_handles_required_array():
    """Properties listed in `required` don't count as optional."""
    from miiflow_agent.providers.anthropic_client import AnthropicClient

    schema = {
        "type": "object",
        "properties": {
            "a": {"type": "string"},
            "b": {"type": "string"},
            "c": {"type": "string"},
        },
        "required": ["a", "b"],
    }
    assert AnthropicClient._count_optional_properties(schema) == 1


def test_count_optional_properties_handles_array_items():
    from miiflow_agent.providers.anthropic_client import AnthropicClient

    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "string"},
                        "y": {"type": "string"},
                    },
                    "required": ["x"],
                },
            }
        },
        "required": ["items"],
    }
    # `items` is required at top, but `y` inside array items is optional.
    assert AnthropicClient._count_optional_properties(schema) == 1
