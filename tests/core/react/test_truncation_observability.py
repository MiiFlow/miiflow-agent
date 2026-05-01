"""Tests for the truncation observability surface.

When a tool_use truncation reappears in production we want one log line + one
queryable event + one step.metadata blob with everything needed to diagnose,
without re-running the session. These tests pin those signals.
"""

import pytest

from miiflow_agent.core.react.enums import ReActEventType
from miiflow_agent.core.react.events.bus import EventFactory
from miiflow_agent.core.react.orchestrator import _preparse_tool_args_string


# ---------- _preparse_tool_args_string ----------


def test_preparse_valid_json_string_becomes_dict():
    tc = {"function": {"name": "f", "arguments": '{"a": 1, "b": "x"}'}}
    _preparse_tool_args_string(tc, step_number=1, tool_name="f")
    assert tc["function"]["arguments"] == {"a": 1, "b": "x"}
    assert "_truncation_error" not in tc


def test_preparse_malformed_json_attaches_truncation_error():
    """OpenAI-style: model emitted a string of args that doesn't parse.
    Without this marker, the truncation handler can't tell this case
    apart from a runtime error and routes to schema validation noise.
    """
    bad = '{"customer_id": "123", "query": "SELECT * FROM camp'  # truncated
    tc = {"function": {"name": "google_ads_query", "arguments": bad}}
    _preparse_tool_args_string(tc, step_number=4, tool_name="google_ads_query")
    assert tc["function"]["arguments"] == {}
    assert "_truncation_error" in tc
    err = tc["_truncation_error"]
    assert err["kind"] == "json_parse_failed"
    assert err["accumulated_length"] == len(bad)
    assert err["raw_prefix"].startswith('{"customer_id":')


def test_preparse_dict_args_unchanged():
    """Anthropic path — args are already a dict, leave them alone."""
    tc = {"function": {"name": "f", "arguments": {"already": "dict"}}}
    _preparse_tool_args_string(tc, step_number=1, tool_name="f")
    assert tc["function"]["arguments"] == {"already": "dict"}
    assert "_truncation_error" not in tc


def test_preparse_empty_string_unchanged():
    """Empty string is an LLM mistake (called a tool with no args), not
    a parse failure — fall through to schema validation."""
    tc = {"function": {"name": "f", "arguments": ""}}
    _preparse_tool_args_string(tc, step_number=1, tool_name="f")
    assert tc["function"]["arguments"] == ""
    assert "_truncation_error" not in tc


def test_preparse_none_unchanged():
    tc = {"function": {"name": "f", "arguments": None}}
    _preparse_tool_args_string(tc, step_number=1, tool_name="f")
    assert tc["function"]["arguments"] is None
    assert "_truncation_error" not in tc


# ---------- EventFactory.llm_truncated ----------


def test_llm_truncated_event_carries_full_postmortem_payload():
    event = EventFactory.llm_truncated(
        step_number=4,
        finish_reason="length",
        output_tokens=32768,
        max_tokens=32768,
        input_tokens=12500,
        tool_names=["render_table"],
        accumulated_json_length=14882,
        raw_prefix='{"columns":[{"key":"name"...',
    )
    assert event.event_type == ReActEventType.LLM_TRUNCATED
    assert event.step_number == 4
    assert event.data["finish_reason"] == "length"
    assert event.data["output_tokens"] == 32768
    assert event.data["max_tokens"] == 32768
    assert event.data["input_tokens"] == 12500
    assert event.data["tool_names"] == ["render_table"]
    assert event.data["accumulated_json_length"] == 14882
    assert event.data["raw_prefix"].startswith('{"columns":')


def test_llm_truncated_event_handles_missing_optional_fields():
    """Even when finish_reason fires without a tool in flight, the event
    should serialize cleanly (no missing-key surprises for consumers)."""
    event = EventFactory.llm_truncated(
        step_number=2,
        finish_reason="length",
        output_tokens=8192,
        max_tokens=8192,
    )
    assert event.event_type == ReActEventType.LLM_TRUNCATED
    assert event.data["tool_names"] == []
    assert event.data["accumulated_json_length"] is None
    assert event.data["raw_prefix"] is None
    assert event.data["input_tokens"] is None
