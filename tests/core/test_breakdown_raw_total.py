"""The policy decision and the calibration estimate share ONE token walk.

``should_compress`` sizes the request; ``record_estimate`` then needs the
uncorrected total for the calibrator. It used to re-walk every message and
tool schema to get it. ``TokenBreakdown.raw_total`` carries the number the
breakdown was derived from, so the second walk is gone — and it must equal
what ``raw_total()`` computes independently, or calibration drifts.
"""

import pytest

from miiflow_agent.core.context import RequestShape, get_counter
from miiflow_agent.core.context.shape import TokenBreakdown
from miiflow_agent.core.message import Message, MessageRole

pytestmark = pytest.mark.unit


def _shape() -> RequestShape:
    tools = [
        {
            "name": f"tool_{i}",
            "description": "Query an account and return rows. " * 6,
            "input_schema": {
                "type": "object",
                "properties": {"q": {"type": "string", "description": "x" * 80}},
            },
        }
        for i in range(5)
    ]
    messages = [
        Message(role=MessageRole.SYSTEM, content="You are terse. " * 40),
        Message(role=MessageRole.USER, content="How did spend trend? " * 20),
        Message(role=MessageRole.ASSISTANT, content="Up 12%. " * 30),
    ]
    return RequestShape(messages=messages, tools=tools, provider="anthropic", model="claude-sonnet-5")


def test_breakdown_carries_the_uncorrected_total():
    counter = get_counter("anthropic", "claude-sonnet-5")
    shape = _shape()
    breakdown = counter.breakdown(shape)
    assert breakdown.raw_total > 0
    assert breakdown.raw_total == counter.raw_total(shape)


def test_hand_built_breakdown_reports_not_recorded():
    assert TokenBreakdown(system=10, tools=20, messages=30).raw_total == 0
