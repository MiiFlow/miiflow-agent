"""Phase 2 (R1): a sub-agent's clarification surfaces to the parent instead of dropping.

These pin the mechanism that fixes the dropped-CLARIFICATION_NEEDED bug:
  * ``forward_subagent_events`` captures + forwards + returns a child interrupt when
    ``surface_interrupts=True`` (and is byte-for-byte legacy when False).
  * ``dispatch_subagent`` surfaces the captured interrupt on the result.
  * ``is_clarification_result`` detects the marker regardless of tool name, so the
    dispatch observation built by ``child_clarification_observation`` pauses the parent.
"""

import pytest

from miiflow_agent.core.react.dispatch import dispatch_subagent, forward_subagent_events
from miiflow_agent.core.react.enums import ReActEventType
from miiflow_agent.core.react.react_events import ReActEvent
from miiflow_agent.core.subagent import SubAgentResult
from miiflow_agent.core.tools.clarification import (
    child_clarification_observation,
    child_tool_approval_observation,
    is_clarification_result,
    is_tool_approval_result,
)
from miiflow_agent.core.tools.schemas import ToolResult


class _Bus:
    def __init__(self):
        self.published = []

    async def publish(self, event):
        self.published.append(event)


async def _events(*evs):
    for e in evs:
        yield e


def _clarification_event():
    return ReActEvent(
        event_type=ReActEventType.CLARIFICATION_NEEDED,
        step_number=2,
        data={"questions": [{"key": "goal", "question": "leads or sales?"}], "context": "needed"},
    )


def _approval_event():
    return ReActEvent(
        event_type=ReActEventType.TOOL_APPROVAL_NEEDED,
        step_number=2,
        data={
            "tool_name": "google_ads_mutate",
            "tool_inputs": {"operation": "create_search_campaign"},
            "tool_call_id": "child_tool_call",
            "interrupt_id": "int_tool_child_tool_call",
            "reason": "needs approval",
        },
    )


@pytest.mark.asyncio
async def test_forward_captures_and_forwards_child_clarification():
    bus = _Bus()
    captured = await forward_subagent_events(
        _events(_clarification_event()),
        parent_event_bus=bus,
        parent_step_number=5,
        subagent_id="sub_x",
        own_path=["sub_x"],
        surface_interrupts=True,
    )
    assert captured is not None
    assert captured["kind"] == "clarification"
    assert captured["subagent_path"] == ["sub_x"]
    # The interrupt was re-published up the parent bus (path-annotated) for live UI.
    assert any(e.event_type == ReActEventType.CLARIFICATION_NEEDED for e in bus.published)
    fwd = next(e for e in bus.published if e.event_type == ReActEventType.CLARIFICATION_NEEDED)
    assert fwd.data["subagent_id"] == "sub_x"


@pytest.mark.asyncio
async def test_forward_captures_and_forwards_child_tool_approval():
    bus = _Bus()
    captured = await forward_subagent_events(
        _events(_approval_event()),
        parent_event_bus=bus,
        parent_step_number=5,
        subagent_id="sub_x",
        own_path=["sub_x"],
        surface_interrupts=True,
    )
    assert captured is not None
    assert captured["kind"] == "tool_approval"
    assert captured["subagent_path"] == ["sub_x"]
    fwd = next(e for e in bus.published if e.event_type == ReActEventType.TOOL_APPROVAL_NEEDED)
    assert fwd.data["tool_call_id"] == "child_tool_call"
    assert fwd.data["subagent_id"] == "sub_x"


@pytest.mark.asyncio
async def test_forward_legacy_drops_when_flag_off():
    bus = _Bus()
    captured = await forward_subagent_events(
        _events(_clarification_event()),
        parent_event_bus=bus,
        parent_step_number=5,
        subagent_id="sub_x",
        own_path=["sub_x"],
        surface_interrupts=False,
    )
    assert captured is None
    # Unchanged legacy behaviour: the clarification is not forwarded.
    assert not any(
        e.event_type == ReActEventType.CLARIFICATION_NEEDED for e in bus.published
    )


class _PausingSubAgent:
    handle = "investigator"
    name = "Investigator"

    async def stream(self, handoff):
        yield _clarification_event()

    def final_result(self):
        return SubAgentResult(answer="", status="completed", duration_ms=1)


class _ApprovalPausingSubAgent:
    handle = "investigator"
    name = "Investigator"

    async def stream(self, handoff):
        yield _approval_event()

    def final_result(self):
        return SubAgentResult(
            answer="",
            status="completed",
            duration_ms=1,
            child_run_id="thread_child",
        )


@pytest.mark.asyncio
async def test_dispatch_subagent_surfaces_pending_child_interrupt():
    from miiflow_agent.core.react.dispatch import DispatchCounter

    bus = _Bus()
    result = await dispatch_subagent(
        _PausingSubAgent(),
        handoff=type("H", (), {"dispatch_chain": [], "depth": 1})(),
        parent_event_bus=bus,
        parent_step_number=0,
        parent_assistant_id="parent",
        child_id="child",
        counter=DispatchCounter(),
        surface_interrupts=True,
    )
    pci = result.metadata.get("pending_child_interrupt")
    assert pci is not None and pci["kind"] == "clarification"


@pytest.mark.asyncio
async def test_dispatch_subagent_surfaces_pending_child_tool_approval():
    from miiflow_agent.core.react.dispatch import DispatchCounter

    bus = _Bus()
    result = await dispatch_subagent(
        _ApprovalPausingSubAgent(),
        handoff=type("H", (), {"dispatch_chain": [], "depth": 1})(),
        parent_event_bus=bus,
        parent_step_number=0,
        parent_assistant_id="parent",
        child_id="child",
        counter=DispatchCounter(),
        surface_interrupts=True,
    )
    pci = result.metadata.get("pending_child_interrupt")
    assert pci is not None and pci["kind"] == "tool_approval"
    assert pci["data"]["interrupt_id"] == "int_tool_child_tool_call"


def test_marker_observation_detected_regardless_of_tool_name():
    obs = child_clarification_observation(
        questions=[{"key": "goal", "question": "leads or sales?"}],
        context="needed",
        dispatch_meta={"handle": "investigator", "status": "awaiting_clarification"},
    )
    # The parent's pause detection must fire on the dispatch_assistant result, whose
    # tool name is NOT ask_user_clarification — detection is by marker, not name.
    res = ToolResult(name="dispatch_assistant", input={}, success=True, output=obs)
    assert is_clarification_result(res) is True


def test_tool_approval_marker_observation_detected_regardless_of_tool_name():
    obs = child_tool_approval_observation(
        tool_name="google_ads_mutate",
        tool_inputs={"operation": "create_search_campaign"},
        tool_call_id="child_tool_call",
        interrupt_id="int_tool_child_tool_call",
        reason="needs approval",
        dispatch_meta={"subagent_path": ["sub_x"]},
    )
    res = ToolResult(name="dispatch_assistant", input={}, success=True, output=obs)
    assert is_tool_approval_result(res) is True
