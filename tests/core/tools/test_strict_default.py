"""Tests for strict-mode tool registration (opt-in) and the per-request cap.

Strict mode (`additionalProperties: false` + Anthropic's structured-outputs
flag) prevents the model from passing the wrong tool's parameter shape to
the wrong tool. But Anthropic compiles every strict tool into one
constrained-decoding grammar and caps the combined strict set per request
(≤20 tools / ≤24 optional / ≤16 union params, plus a residual "grammar too
large"). Defaulting strict ON made every pre-loaded tool set overflow that
grammar and 400, so strict is now OPT-IN: only the handful of read/query
tools with confusable siblings set strict=True. Everything else (incl. the
union-heavy `*_mutate` schemas) stays loose.
"""

import pytest

from miiflow_agent.core.tools.decorators import tool


def test_tool_default_is_non_strict():
    @tool(name="t_default")
    def f(ctx, x: str) -> str:
        return x

    # strict is opt-in: the default must NOT set the metadata flag, so
    # provider adapters fall through to the LOOSE schema mode.
    assert "strict" not in f._tool_schema.metadata


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


# ---- union-type param cap (Phase 3) ----


def _make_union_tool(name: str, n_union: int, strict: bool = True):
    """A strict tool whose schema has `n_union` union-typed (nullable) params."""
    props = {
        f"u{i}": {"type": ["string", "null"], "description": "union"} for i in range(n_union)
    }
    base = {
        "name": name,
        "description": f"tool {name}",
        "input_schema": {
            "type": "object",
            "properties": props,
            "required": [],
            "additionalProperties": False if strict else True,
        },
    }
    if strict:
        base["strict"] = True
    return base


def test_count_union_type_params_counts_nullable_and_anyof():
    from miiflow_agent.providers.anthropic_client import AnthropicClient

    schema = {
        "type": "object",
        "properties": {
            "a": {"type": ["string", "null"]},  # union (nullable)
            "b": {"anyOf": [{"type": "string"}, {"type": "integer"}]},  # union
            "c": {"type": "string"},  # not a union
            "nested": {
                "type": "object",
                "properties": {"d": {"type": ["number", "null"]}},  # union, nested
            },
        },
    }
    assert AnthropicClient._count_union_type_params(schema) == 3


def test_apply_strict_tool_cap_demotes_for_union_param_cap():
    """Under the 20-tool and 24-optional caps but over the 16-union cap → demote tail."""
    from miiflow_agent.providers.anthropic_client import AnthropicClient

    # 3 strict tools, 6 union params each = 18 union params (>16), but only 3
    # tools and the union params are required (0 optional) — so phases 1 and 2
    # are no-ops and only the union cap triggers.
    tools = [_make_union_tool(f"t{i}", n_union=6) for i in range(3)]
    request_params = {"tools": tools}
    AnthropicClient._apply_strict_tool_cap(request_params)
    strict_after = [t for t in request_params["tools"] if t.get("strict")]
    union_total = sum(
        AnthropicClient._count_union_type_params(t["input_schema"]) for t in strict_after
    )
    assert union_total <= 16
    assert len(strict_after) < 3  # at least one demoted


# ---- schema-too-complex reactive fallback (grammar-compilation 400) ----


def test_is_schema_too_complex_error_matches_message():
    from miiflow_agent.providers.anthropic_client import AnthropicClient

    class _Err(Exception):
        message = (
            "Error code: 400 - {'error': {'message': 'Schema is too complex for "
            "compilation. Try reducing the number of tools or simplifying tool schemas.'}}"
        )

    assert AnthropicClient._is_schema_too_complex_error(_Err()) is True
    assert AnthropicClient._is_schema_too_complex_error(Exception("rate limit")) is False


def test_is_schema_too_complex_error_matches_grammar_too_large_message():
    """Anthropic's alternate wording for the same grammar-compilation 400."""
    from miiflow_agent.providers.anthropic_client import AnthropicClient

    class _Err(Exception):
        message = (
            "Error code: 400 - {'type': 'error', 'error': {'type': "
            "'invalid_request_error', 'message': 'The compiled grammar is too "
            "large, which would cause performance issues. Simplify your tool "
            "schemas or reduce the number of strict tools.'}}"
        )

    assert AnthropicClient._is_schema_too_complex_error(_Err()) is True


def test_demote_all_strict_tools_demotes_every_strict_tool():
    from miiflow_agent.providers.anthropic_client import AnthropicClient

    a = _make_tool("a", strict=True)
    b = _make_tool("b", strict=False)
    c = _make_tool("c", strict=True)
    request_params = {"tools": [a, b, c]}
    changed = AnthropicClient._demote_all_strict_tools(request_params)
    assert changed is True
    assert all("strict" not in t for t in request_params["tools"])
    assert all(t["input_schema"]["additionalProperties"] is True for t in request_params["tools"])
    # originals not mutated
    assert a.get("strict") is True and c.get("strict") is True


def test_demote_all_strict_tools_no_op_when_none_strict():
    from miiflow_agent.providers.anthropic_client import AnthropicClient

    request_params = {"tools": [_make_tool("a", strict=False)]}
    assert AnthropicClient._demote_all_strict_tools(request_params) is False
    assert AnthropicClient._demote_all_strict_tools({}) is False
    assert AnthropicClient._demote_all_strict_tools({"tools": []}) is False
