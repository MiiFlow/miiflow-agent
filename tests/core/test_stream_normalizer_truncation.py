"""Tests for tool_use streaming truncation handling.

Production saw a flood of "missing required parameters" errors caused by Claude
hitting max_tokens mid-tool_use. The stream normalizer used to silently swallow
the resulting truncated JSON as `{}`, hiding the truncation from the orchestrator.
These tests pin the new behavior: truncated JSON must surface as
`_truncation_error` on the tool call, not be swallowed.
"""

from types import SimpleNamespace

import pytest

from miiflow_agent.core.stream_normalizer import AnthropicStreamNormalizer


def _start_tool_use(name: str = "render_chart", tool_id: str = "tu_1", index: int = 0):
    """Build a minimal Anthropic content_block_start for a tool_use block."""
    return SimpleNamespace(
        type="content_block_start",
        index=index,
        content_block=SimpleNamespace(type="tool_use", name=name, id=tool_id),
        usage=None,
    )


def _delta(partial_json: str, index: int = 0):
    """Build a content_block_delta with input_json_delta semantics."""
    return SimpleNamespace(
        type="content_block_delta",
        index=index,
        delta=SimpleNamespace(partial_json=partial_json),
        usage=None,
    )


def _stop():
    return SimpleNamespace(type="content_block_stop", usage=None)


@pytest.fixture
def normalizer():
    return AnthropicStreamNormalizer()


def test_truncated_json_surfaces_as_truncation_error(normalizer):
    """When Claude truncates mid-tool_use, _truncation_error must be set."""
    normalizer.normalize_chunk(_start_tool_use())
    # Simulate Claude hitting max_tokens partway through emitting `series`.
    # The accumulated buffer is unparseable JSON.
    normalizer.normalize_chunk(_delta('{"chart_type":"line","title":"Q4",'))
    normalizer.normalize_chunk(_delta('"series":[{"name":"Spend","data":[{"x":"Jan","y":1'))
    final = normalizer.normalize_chunk(_stop())

    assert final.tool_calls is not None and len(final.tool_calls) == 1
    tool_call = final.tool_calls[0]
    assert tool_call["function"]["arguments"] == {}, (
        "truncated JSON must not be silently parsed as {} without a marker"
    )
    assert "_truncation_error" in tool_call, (
        "truncated tool_use blocks must carry _truncation_error so the "
        "orchestrator can route them to the truncation handler"
    )
    err = tool_call["_truncation_error"]
    assert err["kind"] == "json_parse_failed"
    assert err["accumulated_length"] > 0
    assert err["raw_prefix"].startswith("{")


def test_complete_json_has_no_truncation_error(normalizer):
    """Sanity check: well-formed JSON should round-trip without a marker."""
    normalizer.normalize_chunk(_start_tool_use(name="render_chart"))
    normalizer.normalize_chunk(_delta('{"chart_type":"bar",'))
    normalizer.normalize_chunk(_delta('"series":[{"name":"S","data":[{"x":"a","y":1}]}]}'))
    final = normalizer.normalize_chunk(_stop())

    tool_call = final.tool_calls[0]
    assert "_truncation_error" not in tool_call
    assert tool_call["function"]["arguments"] == {
        "chart_type": "bar",
        "series": [{"name": "S", "data": [{"x": "a", "y": 1}]}],
    }


def test_no_mid_delta_parsing_preserves_final_args(normalizer):
    """Args must reflect the final content_block_stop parse, not an
    intermediate state. Mid-delta parsing used to produce a partial dict
    that survived into the final args when later deltas made the buffer
    unparseable — that was the source of the
    `Provided parameters: ['chart_type','title','legend']` pattern in prod.
    """
    normalizer.normalize_chunk(_start_tool_use(name="render_table"))
    # First delta is, on its own, valid JSON for an object missing the
    # required `rows`/`columns`. Pre-fix this would have populated args.
    normalizer.normalize_chunk(_delta('{"title":"X","sortable":true,"paginated":false}'))
    final_intermediate_chunk = normalizer.normalize_chunk(
        _delta(' EXTRA_GARBAGE_THAT_BREAKS_JSON')
    )
    # The intermediate stream chunk should not have set args from a
    # coincidentally-parseable prefix.
    intermediate_args = (
        normalizer._state.current_tool_use["function"]["arguments"]
        if normalizer._state.current_tool_use
        else None
    )
    assert intermediate_args == {}, (
        "args must stay empty until content_block_stop — no mid-delta parsing"
    )

    final = normalizer.normalize_chunk(_stop())
    tool_call = final.tool_calls[0]
    # The buffer is unparseable at stop, so we expect the truncation marker.
    assert "_truncation_error" in tool_call
    assert tool_call["function"]["arguments"] == {}
