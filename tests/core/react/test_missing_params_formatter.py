"""Tests for the actionable missing-required-params error formatter.

The orchestrator used to feed the LLM a generic string ("Tool X requires
parameters Y. This may indicate incomplete streaming or malformed LLM
response.") that gave the model nothing to correct against. The new
formatter mirrors claude-code's <tool_use_error> pattern: name the gap,
dump the schema with REQUIRED markers, and instruct the model to retry.
"""

from miiflow_agent.core.react.orchestrator import _format_missing_params_error


def _render_chart_schema():
    return {
        "name": "render_chart",
        "description": "Render a chart.",
        "parameters": {
            "type": "object",
            "properties": {
                "chart_type": {
                    "type": "string",
                    "enum": ["line", "bar", "pie", "area", "scatter"],
                    "description": "The type of chart to render",
                },
                "series": {
                    "type": "array",
                    "description": (
                        "Array of data series. Each has 'name' and 'data'."
                    ),
                },
                "title": {"type": "string", "description": "Chart title"},
                "legend": {"type": "boolean", "description": "Show legend"},
            },
            "required": ["chart_type", "series"],
        },
    }


def test_includes_tool_use_error_wrapper():
    msg = _format_missing_params_error(
        tool_name="render_chart",
        missing_params=["series"],
        provided_params=["chart_type", "title", "legend"],
        tool_schema=_render_chart_schema(),
    )
    assert msg.startswith("<tool_use_error>")
    assert msg.endswith("</tool_use_error>")
    assert "render_chart" in msg
    assert "InputValidationError" in msg


def test_names_missing_and_provided():
    msg = _format_missing_params_error(
        tool_name="render_chart",
        missing_params=["series"],
        provided_params=["chart_type", "title", "legend"],
        tool_schema=_render_chart_schema(),
    )
    assert "['series']" in msg
    assert "['chart_type', 'title', 'legend']" in msg


def test_schema_marks_required_vs_optional():
    msg = _format_missing_params_error(
        tool_name="render_chart",
        missing_params=["series"],
        provided_params=["chart_type", "title", "legend"],
        tool_schema=_render_chart_schema(),
    )
    # The required field should be marked REQUIRED in the schema dump
    assert "series (REQUIRED" in msg
    assert "chart_type (REQUIRED" in msg
    # Optional fields must NOT carry the REQUIRED marker
    assert "title (optional" in msg
    assert "legend (optional" in msg


def test_missing_field_listed_first_in_schema():
    """The model's eye lands on the first item — put the gap there."""
    msg = _format_missing_params_error(
        tool_name="render_chart",
        missing_params=["series"],
        provided_params=["chart_type", "title", "legend"],
        tool_schema=_render_chart_schema(),
    )
    series_pos = msg.find("series (REQUIRED")
    chart_type_pos = msg.find("chart_type (REQUIRED")
    title_pos = msg.find("title (optional")
    assert 0 < series_pos < chart_type_pos < title_pos


def test_includes_enum_values():
    msg = _format_missing_params_error(
        tool_name="render_chart",
        missing_params=["chart_type"],
        provided_params=[],
        tool_schema=_render_chart_schema(),
    )
    assert "one of: line, bar, pie, area, scatter" in msg


def test_has_explicit_retry_instruction():
    msg = _format_missing_params_error(
        tool_name="render_chart",
        missing_params=["series"],
        provided_params=["chart_type", "title", "legend"],
        tool_schema=_render_chart_schema(),
    )
    assert "Retry" in msg
    assert "REQUIRED" in msg


def test_handles_tool_with_no_required_field():
    """Sanity: schema with no `required` list shouldn't crash."""
    schema = {
        "name": "noop",
        "parameters": {
            "type": "object",
            "properties": {"x": {"type": "string", "description": "anything"}},
        },
    }
    msg = _format_missing_params_error(
        tool_name="noop",
        missing_params=["x"],  # caller asserted x is required even if schema is loose
        provided_params=[],
        tool_schema=schema,
    )
    assert "noop" in msg
    # x has no required marker in schema, so formatter should mark it optional
    assert "x (optional" in msg
