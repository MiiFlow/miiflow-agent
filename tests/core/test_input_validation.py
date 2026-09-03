"""Tests for constraint enforcement on tool inputs.

Validation was presence-only: type, enum, minimum/maximum, pattern, and item
types were emitted to providers but never enforced client-side, so bad values
reached tool bodies and failed wherever they happened to crash.
"""

import pytest

from miiflow_agent.core.tools import tool
from miiflow_agent.core.tools.exceptions import ToolExecutionError
from miiflow_agent.core.tools.function.function_tool import FunctionTool
from miiflow_agent.core.tools.input_validation import validate_inputs_against
from miiflow_agent.core.tools.schemas import ParameterSchema
from miiflow_agent.core.tools.types import ParameterType


def _params(**schemas):
    return schemas


def _p(name, type_, **kwargs):
    kwargs.setdefault("description", name)
    return ParameterSchema(name=name, type=type_, **kwargs)


class TestValidateInputsAgainst:
    def test_valid_inputs_pass_through(self):
        params = _params(
            q=_p("q", ParameterType.STRING),
            limit=_p("limit", ParameterType.INTEGER, required=False),
        )
        validated, errors = validate_inputs_against(params, {"q": "ads", "limit": 5}, enforce=True)
        assert errors == []
        assert validated == {"q": "ads", "limit": 5}

    def test_missing_required_reported(self):
        params = _params(q=_p("q", ParameterType.STRING))
        _, errors = validate_inputs_against(params, {}, enforce=True)
        assert errors == ["Missing required parameter: q"]

    def test_type_mismatch_reported_with_types(self):
        params = _params(limit=_p("limit", ParameterType.INTEGER))
        _, errors = validate_inputs_against(params, {"limit": "abc"}, enforce=True)
        assert len(errors) == 1
        assert "must be integer" in errors[0] and "'abc'" in errors[0]

    def test_numeric_string_coerces(self):
        params = _params(
            limit=_p("limit", ParameterType.INTEGER),
            ratio=_p("ratio", ParameterType.NUMBER),
            flag=_p("flag", ParameterType.BOOLEAN),
        )
        validated, errors = validate_inputs_against(params, {"limit": "5", "ratio": "0.5", "flag": "true"}, enforce=True)
        assert errors == []
        assert validated == {"limit": 5, "ratio": 0.5, "flag": True}

    def test_bool_is_not_an_integer(self):
        params = _params(limit=_p("limit", ParameterType.INTEGER))
        _, errors = validate_inputs_against(params, {"limit": True}, enforce=True)
        assert len(errors) == 1

    def test_enum_enforced(self):
        params = _params(
            platform=_p(
                "platform", ParameterType.STRING, enum=["meta", "google"]
            )
        )
        _, errors = validate_inputs_against(params, {"platform": "tiktok"}, enforce=True)
        assert len(errors) == 1
        assert "'meta'" in errors[0] and "'tiktok'" in errors[0]

    def test_range_enforced(self):
        params = _params(
            limit=_p("limit", ParameterType.INTEGER, minimum=1, maximum=100)
        )
        _, low = validate_inputs_against(params, {"limit": 0}, enforce=True)
        _, high = validate_inputs_against(params, {"limit": 200}, enforce=True)
        _, ok = validate_inputs_against(params, {"limit": 50}, enforce=True)
        assert ">= 1" in low[0]
        assert "<= 100" in high[0]
        assert ok == []

    def test_pattern_enforced_unanchored(self):
        params = _params(
            account=_p("account", ParameterType.STRING, pattern=r"act_\d+")
        )
        _, bad = validate_inputs_against(params, {"account": "12345"}, enforce=True)
        _, ok = validate_inputs_against(params, {"account": "act_12345"}, enforce=True)
        assert len(bad) == 1
        assert ok == []

    def test_array_item_types_enforced(self):
        params = _params(
            ids=_p("ids", ParameterType.ARRAY, items={"type": "integer"})
        )
        _, bad = validate_inputs_against(params, {"ids": [1, "x", 3]}, enforce=True)
        _, ok = validate_inputs_against(params, {"ids": [1, 2, 3]}, enforce=True)
        assert len(bad) == 1 and "element 1" in bad[0]
        assert ok == []

    def test_all_problems_reported_at_once(self):
        params = _params(
            q=_p("q", ParameterType.STRING),
            limit=_p("limit", ParameterType.INTEGER, minimum=1),
        )
        _, errors = validate_inputs_against(params, {"limit": 0}, enforce=True)
        assert len(errors) == 2  # missing q AND limit below minimum

    def test_explicit_none_always_passes(self):
        """Whether the body accepts None is the body's contract — base
        passed it through even for required params."""
        for required in (True, False):
            params = _params(
                limit=_p("limit", ParameterType.INTEGER, required=required)
            )
            validated, errors = validate_inputs_against(
                params, {"limit": None}, enforce=True
            )
            assert errors == []
            assert validated == {"limit": None}

    def test_unknown_params_are_dropped(self):
        """Base behavior restored: models hallucinate stray kwargs, and
        passing them to fn(**validated) turns a working call into a
        TypeError."""
        params = _params(q=_p("q", ParameterType.STRING))
        for enforce in (True, False):
            validated, errors = validate_inputs_against(
                params, {"q": "x", "extra": 1}, enforce=enforce
            )
            assert errors == []
            assert "extra" not in validated

    def test_lax_mode_logs_and_passes_values_untouched(self):
        """Default (no MIIFLOW_STRICT_TOOL_VALIDATION): constraint and type
        violations pass through uncoerced — pre-enforcement fidelity."""
        params = _params(
            limit=_p("limit", ParameterType.INTEGER, minimum=1, maximum=100),
            platform=_p("platform", ParameterType.STRING, enum=["meta"]),
        )
        validated, errors = validate_inputs_against(
            params, {"limit": "not-a-number", "platform": "tiktok"}, enforce=False
        )
        assert errors == []
        assert validated == {"limit": "not-a-number", "platform": "tiktok"}

    def test_json_string_coerces_to_array_or_object(self):
        """Models hand structured params over as JSON text — `render_table`
        emitted rows='[{...}]' and the string reached the browser, where zod
        rejected it. Scalars have always coerced from their string form; so
        must arrays and objects."""
        params = _params(
            rows=_p("rows", ParameterType.ARRAY),
            config=_p("config", ParameterType.OBJECT),
        )
        for enforce in (True, False):
            validated, errors = validate_inputs_against(
                params,
                {"rows": '[{"a": 1}]', "config": '{"sortable": true}'},
                enforce=enforce,
            )
            assert errors == []
            assert validated["rows"] == [{"a": 1}]
            assert validated["config"] == {"sortable": True}

    def test_json_string_of_the_wrong_shape_is_not_coerced(self):
        """Narrow rule, same as the scalar branches: it must parse AND parse
        to the declared type. An object is not an array."""
        params = _params(rows=_p("rows", ParameterType.ARRAY))
        _, errors = validate_inputs_against(
            params, {"rows": '{"a": 1}'}, enforce=True
        )
        assert len(errors) == 1
        _, errors = validate_inputs_against(
            params, {"rows": "not json at all"}, enforce=True
        )
        assert len(errors) == 1

    def test_lax_mode_applies_successful_coercions(self):
        """A coercion that succeeds is a repair, not a rejection. Withholding
        it until strict mode only moved the failure downstream to whatever
        could not name it."""
        params = _params(
            limit=_p("limit", ParameterType.INTEGER),
            rows=_p("rows", ParameterType.ARRAY),
        )
        validated, errors = validate_inputs_against(
            params, {"limit": "10", "rows": "[1, 2]"}, enforce=False
        )
        assert errors == []
        assert validated == {"limit": 10, "rows": [1, 2]}

    def test_lax_mode_still_enforces_presence(self):
        params = _params(q=_p("q", ParameterType.STRING))
        _, errors = validate_inputs_against(params, {}, enforce=False)
        assert errors == ["Missing required parameter: q"]


class TestFunctionToolValidation:
    async def test_hallucinated_param_is_dropped_not_fatal(self):
        @tool(name="lookup0")
        async def lookup0(q: str):
            """Lookup."""
            return {"q": q}

        ft = FunctionTool(lookup0)
        result = await ft.acall(q="x", extra_param="hallucinated")
        assert result.success is True
        assert result.output == {"q": "x"}

    async def test_constraint_violation_becomes_failed_result(self, monkeypatch):
        monkeypatch.setenv("MIIFLOW_STRICT_TOOL_VALIDATION", "1")
        @tool(
            name="lookup",
            parameters={
                "limit": ParameterSchema(
                    name="limit",
                    type=ParameterType.INTEGER,
                    description="rows",
                    minimum=1,
                    maximum=100,
                ),
            },
        )
        async def lookup(limit: int):
            return {"rows": limit}

        ft = FunctionTool(lookup)
        result = await ft.acall(limit=500)
        assert result.success is False
        assert "<= 100" in result.error

    async def test_coerced_value_reaches_the_tool(self, monkeypatch):
        monkeypatch.setenv("MIIFLOW_STRICT_TOOL_VALIDATION", "1")
        @tool(
            name="lookup2",
            parameters={
                "limit": ParameterSchema(
                    name="limit", type=ParameterType.INTEGER, description="rows"
                ),
            },
        )
        async def lookup2(limit: int):
            assert isinstance(limit, int)
            return {"rows": limit}

        ft = FunctionTool(lookup2)
        result = await ft.acall(limit="7")
        assert result.success is True
        assert result.output == {"rows": 7}
