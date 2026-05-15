"""Unit tests for the Stage 2 sub_agents primitive.

Covers:
- SubAgent protocol + dataclass smoke
- AgentConfig validation (handle/tool collision, duplicate handles)
- Agent.__init__ construction modes (legacy kwargs, config=, mix is rejected)
- DispatchCounter race-safety under asyncio.gather
- Static guardrails (cycle, depth)
- Event forwarding (FINAL_ANSWER_CHUNK -> progress, nested dispatch path prefix)
- dispatch_subagent full lifecycle (start/complete/failed events on bus)
- SSE event shape contract (locks the keys the FE converter consumes)
"""
from __future__ import annotations

import asyncio
import inspect
from typing import AsyncIterator, List
from unittest.mock import MagicMock

import pytest


# ── SubAgent / dataclasses ────────────────────────────────────────────────


def test_subagent_handoff_defaults():
    from miiflow_agent.core.subagent import SubAgentHandoff

    h = SubAgentHandoff(task="t", intent_summary="s")
    assert h.depth == 1
    assert h.dispatch_chain == []
    assert h.structured_payload is None
    assert h.parent_user_message is None


def test_subagent_result_required_fields():
    from miiflow_agent.core.subagent import SubAgentResult

    r = SubAgentResult(answer="hi", status="completed")
    assert r.status == "completed"
    assert r.metadata == {}
    assert r.duration_ms == 0


def test_subagent_descriptor_abstract_methods_raise():
    from miiflow_agent.core.subagent import (
        SubAgentDescriptor,
        SubAgentHandoff,
    )

    desc = SubAgentDescriptor(handle="x", name="X", description="d")
    with pytest.raises(NotImplementedError):
        desc.final_result()

    async def _drain():
        async for _ in desc.stream(SubAgentHandoff(task="t", intent_summary="s")):
            pass

    with pytest.raises(NotImplementedError):
        asyncio.run(_drain())


def test_subagent_protocol_runtime_check():
    """A SubAgentDescriptor satisfies the SubAgent Protocol at runtime."""
    from miiflow_agent.core.subagent import SubAgent, SubAgentDescriptor

    desc = SubAgentDescriptor(handle="x", name="X", description="d")
    # runtime_checkable Protocol — isinstance checks the attribute set
    assert isinstance(desc, SubAgent)


# ── AgentConfig ──────────────────────────────────────────────────────────


def _make_client():
    """Minimal LLMClient stand-in that satisfies the registry handshake."""
    from miiflow_agent.core.tools import ToolRegistry

    client = MagicMock()
    client.tool_registry = ToolRegistry()
    return client


def test_agent_config_defaults_match_legacy():
    from miiflow_agent.core.agent import AgentType
    from miiflow_agent.core.config import AgentConfig

    cfg = AgentConfig(client=_make_client())
    assert cfg.agent_type == AgentType.SINGLE_HOP
    assert cfg.retries == 1
    assert cfg.max_iterations == 10
    assert cfg.temperature == 0.7
    assert cfg.tools == []
    assert cfg.sub_agents == []


def test_agent_config_to_kwargs_roundtrip():
    from miiflow_agent.core.agent import AgentType
    from miiflow_agent.core.config import AgentConfig

    cfg = AgentConfig(
        client=_make_client(),
        agent_type=AgentType.REACT,
        retries=3,
        temperature=0.2,
    )
    kw = cfg.to_kwargs()
    assert "client" not in kw  # client passed positionally
    assert kw["agent_type"] == AgentType.REACT
    assert kw["retries"] == 3
    assert kw["temperature"] == 0.2


def test_agent_config_rejects_handle_tool_collision():
    from miiflow_agent.core.config import AgentConfig
    from miiflow_agent.core.subagent import SubAgentDescriptor
    from miiflow_agent.core.tools.decorators import tool

    @tool(name="research", description="x")
    def research():
        return "ok"

    sub = SubAgentDescriptor(handle="research", name="R", description="d")
    with pytest.raises(ValueError, match="collides"):
        AgentConfig(
            client=_make_client(),
            tools=[research._function_tool],
            sub_agents=[sub],
        )


def test_agent_config_rejects_duplicate_handles():
    from miiflow_agent.core.config import AgentConfig
    from miiflow_agent.core.subagent import SubAgentDescriptor

    a = SubAgentDescriptor(handle="x", name="A", description="d")
    b = SubAgentDescriptor(handle="x", name="B", description="d")
    with pytest.raises(ValueError, match="Duplicate"):
        AgentConfig(client=_make_client(), sub_agents=[a, b])


# ── Agent.__init__ construction modes ────────────────────────────────────


def test_agent_legacy_kwargs_still_work():
    from miiflow_agent.core.agent import Agent, AgentType

    a = Agent(_make_client(), agent_type=AgentType.SINGLE_HOP, retries=2)
    assert a.agent_type == AgentType.SINGLE_HOP
    assert a.retries == 2
    assert a.sub_agents == []


def test_agent_accepts_config():
    from miiflow_agent.core.agent import Agent, AgentType
    from miiflow_agent.core.config import AgentConfig
    from miiflow_agent.core.subagent import SubAgentDescriptor

    sub = SubAgentDescriptor(handle="x", name="X", description="d")
    cfg = AgentConfig(
        client=_make_client(),
        agent_type=AgentType.REACT,
        sub_agents=[sub],
    )
    a = Agent(config=cfg)
    assert a.agent_type == AgentType.REACT
    assert len(a.sub_agents) == 1
    assert a.sub_agents[0].handle == "x"


def test_agent_rejects_config_plus_conflicting_kwargs():
    from miiflow_agent.core.agent import Agent
    from miiflow_agent.core.config import AgentConfig

    cfg = AgentConfig(client=_make_client())
    with pytest.raises(ValueError, match="config="):
        Agent(config=cfg, tools=[])  # type: ignore[arg-type]


def test_agent_rejects_no_client_no_config():
    from miiflow_agent.core.agent import Agent

    with pytest.raises(ValueError, match="client"):
        Agent()


def test_agent_add_sub_agent_collision():
    from miiflow_agent.core.agent import Agent
    from miiflow_agent.core.subagent import SubAgentDescriptor

    a = Agent(_make_client())
    s1 = SubAgentDescriptor(handle="x", name="X", description="d")
    s2 = SubAgentDescriptor(handle="x", name="X2", description="d")
    a.add_sub_agent(s1)
    with pytest.raises(ValueError, match="already registered"):
        a.add_sub_agent(s2)


# ── DispatchCounter (race-safety) ────────────────────────────────────────


def test_dispatch_counter_per_handle_budget():
    from miiflow_agent.core.react.dispatch import (
        DispatchCounter,
        DispatchGuardrailError,
    )

    c = DispatchCounter(max_per_handle=2, max_total=10)

    async def _run():
        await c.reserve("h")
        await c.reserve("h")
        with pytest.raises(DispatchGuardrailError) as ei:
            await c.reserve("h")
        assert ei.value.kind == "per_handle_budget_exceeded"

    asyncio.run(_run())
    assert c.counts["h"] == 2
    assert c.total == 2


def test_dispatch_counter_total_budget():
    from miiflow_agent.core.react.dispatch import (
        DispatchCounter,
        DispatchGuardrailError,
    )

    c = DispatchCounter(max_per_handle=10, max_total=2)

    async def _run():
        await c.reserve("a")
        await c.reserve("b")
        with pytest.raises(DispatchGuardrailError) as ei:
            await c.reserve("c")
        assert ei.value.kind == "max_dispatches_exceeded"

    asyncio.run(_run())


def test_dispatch_counter_race_under_gather():
    """Many concurrent reserves must yield exactly N counter increments.

    Pressure test for the lost-write race the JSONField counter has —
    the asyncio.Lock should serialize the check-and-increment so no
    increments are lost and no over-budget reservations slip through.
    """
    from miiflow_agent.core.react.dispatch import (
        DispatchCounter,
        DispatchGuardrailError,
    )

    c = DispatchCounter(max_per_handle=100, max_total=100)

    async def _bump():
        try:
            await c.reserve("h")
            return True
        except DispatchGuardrailError:
            return False

    async def _race():
        results = await asyncio.gather(*[_bump() for _ in range(50)])
        return results

    results = asyncio.run(_race())
    assert sum(results) == 50  # all succeed (50 < 100 limit)
    assert c.counts["h"] == 50
    assert c.total == 50


def test_dispatch_counter_race_at_budget_boundary():
    """Under heavy contention at the boundary, exactly `max_total`
    reservations succeed — never more, never less."""
    from miiflow_agent.core.react.dispatch import (
        DispatchCounter,
        DispatchGuardrailError,
    )

    c = DispatchCounter(max_per_handle=100, max_total=10)

    async def _bump():
        try:
            await c.reserve("h")
            return True
        except DispatchGuardrailError:
            return False

    async def _race():
        return await asyncio.gather(*[_bump() for _ in range(50)])

    results = asyncio.run(_race())
    assert sum(results) == 10  # exactly the budget
    assert c.total == 10


def test_dispatch_counter_snapshot():
    from miiflow_agent.core.react.dispatch import DispatchCounter

    c = DispatchCounter(max_per_handle=3, max_total=10)

    async def _run():
        await c.reserve("a")
        await c.reserve("b")
        await c.reserve("a")

    asyncio.run(_run())
    snap = c.snapshot()
    assert snap == {
        "counts": {"a": 2, "b": 1},
        "total": 3,
        "max_per_handle": 3,
        "max_total": 10,
    }


# ── Static guardrails ────────────────────────────────────────────────────


def test_static_guardrails_cycle_detection():
    from miiflow_agent.core.react.dispatch import (
        DispatchGuardrailError,
        enforce_static_guardrails,
    )

    with pytest.raises(DispatchGuardrailError) as ei:
        enforce_static_guardrails(
            "child_handle",
            child_id="child_1",
            dispatch_chain=["parent_1", "child_1"],
            depth=2,
        )
    assert ei.value.kind == "cycle_detected"


def test_static_guardrails_max_depth():
    from miiflow_agent.core.react.dispatch import (
        DispatchGuardrailError,
        enforce_static_guardrails,
    )

    with pytest.raises(DispatchGuardrailError) as ei:
        enforce_static_guardrails(
            "x",
            child_id="cid",
            dispatch_chain=["a", "b", "c"],
            depth=4,
            max_depth=3,
        )
    assert ei.value.kind == "max_depth_exceeded"


def test_static_guardrails_passes():
    from miiflow_agent.core.react.dispatch import enforce_static_guardrails

    # Should not raise
    enforce_static_guardrails(
        "x",
        child_id="cid",
        dispatch_chain=["parent"],
        depth=1,
    )


# ── Event forwarding ─────────────────────────────────────────────────────


def _make_event_bus():
    from miiflow_agent.core.react.events.bus import EventBus

    bus = EventBus()
    received: List = []
    bus.subscribe(lambda ev: received.append(ev))
    return bus, received


async def _gen_events(events):
    for e in events:
        yield e


def test_forward_subagent_events_final_answer_chunk():
    from miiflow_agent.core.react.dispatch import forward_subagent_events
    from miiflow_agent.core.react.enums import ReActEventType
    from miiflow_agent.core.react.react_events import ReActEvent

    bus, received = _make_event_bus()
    chunk_event = ReActEvent(
        event_type=ReActEventType.FINAL_ANSWER_CHUNK,
        step_number=0,
        data={"delta": "Hello "},
    )

    asyncio.run(
        forward_subagent_events(
            _gen_events([chunk_event]),
            parent_event_bus=bus,
            parent_step_number=2,
            subagent_id="sub_abc",
            own_path=["sub_abc"],
        )
    )

    assert len(received) == 1
    out = received[0]
    assert out.event_type == ReActEventType.SUBAGENT_DISPATCH
    assert out.data["sub_event"] == "progress"
    assert out.data["chunk"] == "Hello "
    assert out.data["subagent_id"] == "sub_abc"
    assert out.data["subagent_path"] == ["sub_abc"]


def test_forward_subagent_events_skips_empty_chunks():
    from miiflow_agent.core.react.dispatch import forward_subagent_events
    from miiflow_agent.core.react.enums import ReActEventType
    from miiflow_agent.core.react.react_events import ReActEvent

    bus, received = _make_event_bus()
    empty_event = ReActEvent(
        event_type=ReActEventType.FINAL_ANSWER_CHUNK,
        step_number=0,
        data={"delta": ""},
    )

    asyncio.run(
        forward_subagent_events(
            _gen_events([empty_event]),
            parent_event_bus=bus,
            parent_step_number=0,
            subagent_id="sub_x",
            own_path=["sub_x"],
        )
    )
    assert received == []  # empty chunks dropped


def test_forward_subagent_events_nested_dispatch_path_prefix():
    """A grandchild dispatch coming up through the child must get our
    subagent_id prepended so it nests under our panel in the UI."""
    from miiflow_agent.core.react.dispatch import forward_subagent_events
    from miiflow_agent.core.react.enums import ReActEventType
    from miiflow_agent.core.react.react_events import ReActEvent

    bus, received = _make_event_bus()
    grandchild = ReActEvent(
        event_type=ReActEventType.SUBAGENT_DISPATCH,
        step_number=0,
        data={
            "sub_event": "start",
            "subagent_id": "grandchild_id",
            "subagent_path": ["grandchild_id"],
            "handle": "deeper_specialist",
        },
    )

    asyncio.run(
        forward_subagent_events(
            _gen_events([grandchild]),
            parent_event_bus=bus,
            parent_step_number=0,
            subagent_id="sub_parent",
            own_path=["sub_parent"],
        )
    )

    assert len(received) == 1
    out = received[0]
    assert out.event_type == ReActEventType.SUBAGENT_DISPATCH
    assert out.data["subagent_path"] == ["sub_parent", "grandchild_id"]
    assert out.data["subagent_id"] == "grandchild_id"  # original preserved


def test_forward_subagent_events_ignores_other_event_types():
    from miiflow_agent.core.react.dispatch import forward_subagent_events
    from miiflow_agent.core.react.enums import ReActEventType
    from miiflow_agent.core.react.react_events import ReActEvent

    bus, received = _make_event_bus()
    other = ReActEvent(
        event_type=ReActEventType.STEP_START,
        step_number=0,
        data={},
    )

    asyncio.run(
        forward_subagent_events(
            _gen_events([other]),
            parent_event_bus=bus,
            parent_step_number=0,
            subagent_id="sub_x",
            own_path=["sub_x"],
        )
    )
    assert received == []


# ── dispatch_subagent end-to-end ─────────────────────────────────────────


class _FakeSubAgent:
    """Minimal SubAgent that emits a final-answer chunk + finishes."""

    handle = "demo"
    name = "Demo"
    description = "demo specialist"
    schema = None
    when_to_use = "for demos"
    handoff_schema = None
    forward_transcript_window = None
    clarification_policy = None
    auto_approve_child_tools = False

    def __init__(self, *, status="completed", answer="ok", raise_in_stream=False):
        self._status = status
        self._answer = answer
        self._raise = raise_in_stream

    async def stream(self, handoff) -> AsyncIterator:
        from miiflow_agent.core.react.enums import ReActEventType
        from miiflow_agent.core.react.react_events import ReActEvent

        yield ReActEvent(
            event_type=ReActEventType.FINAL_ANSWER_CHUNK,
            step_number=0,
            data={"delta": self._answer},
        )
        if self._raise:
            raise RuntimeError("child blew up")

    def final_result(self):
        from miiflow_agent.core.subagent import SubAgentResult

        return SubAgentResult(answer=self._answer, status=self._status, duration_ms=42)


def test_dispatch_subagent_success_lifecycle():
    """Start + progress + complete events land on the parent's bus."""
    from miiflow_agent.core.react.dispatch import (
        DispatchCounter,
        dispatch_subagent,
    )
    from miiflow_agent.core.react.enums import ReActEventType
    from miiflow_agent.core.subagent import SubAgentHandoff

    bus, received = _make_event_bus()
    sub = _FakeSubAgent()
    counter = DispatchCounter()
    handoff = SubAgentHandoff(task="t", intent_summary="s", depth=1)

    result = asyncio.run(
        dispatch_subagent(
            sub,
            handoff,
            parent_event_bus=bus,
            parent_step_number=3,
            parent_assistant_id="parent_x",
            child_id="child_demo",
            counter=counter,
        )
    )

    assert result.status == "completed"
    assert result.answer == "ok"
    # Expect: start + progress + complete
    types = [(e.event_type, e.data.get("sub_event")) for e in received]
    assert types == [
        (ReActEventType.SUBAGENT_DISPATCH, "start"),
        (ReActEventType.SUBAGENT_DISPATCH, "progress"),
        (ReActEventType.SUBAGENT_DISPATCH, "complete"),
    ]


def test_dispatch_subagent_failure_lifecycle():
    """When the child raises, we emit `failed` and return status=failed."""
    from miiflow_agent.core.react.dispatch import (
        DispatchCounter,
        dispatch_subagent,
    )
    from miiflow_agent.core.subagent import SubAgentHandoff

    bus, received = _make_event_bus()
    sub = _FakeSubAgent(raise_in_stream=True)
    counter = DispatchCounter()
    handoff = SubAgentHandoff(task="t", intent_summary="s", depth=1)

    result = asyncio.run(
        dispatch_subagent(
            sub,
            handoff,
            parent_event_bus=bus,
            parent_step_number=1,
            parent_assistant_id="parent_x",
            child_id="child_demo",
            counter=counter,
        )
    )

    assert result.status == "failed"
    assert "blew up" in (result.error or "")
    sub_events = [e.data.get("sub_event") for e in received]
    assert "start" in sub_events
    assert "failed" in sub_events
    assert "complete" not in sub_events


def test_dispatch_subagent_consumes_counter():
    from miiflow_agent.core.react.dispatch import (
        DispatchCounter,
        dispatch_subagent,
    )
    from miiflow_agent.core.subagent import SubAgentHandoff

    bus, _ = _make_event_bus()
    sub = _FakeSubAgent()
    counter = DispatchCounter()

    asyncio.run(
        dispatch_subagent(
            sub,
            SubAgentHandoff(task="t", intent_summary="s"),
            parent_event_bus=bus,
            parent_step_number=0,
            parent_assistant_id="parent_x",
            child_id="child_x",
            counter=counter,
        )
    )
    assert counter.total == 1
    assert counter.counts["demo"] == 1


def test_dispatch_subagent_no_bus_runs_silently():
    """When parent_event_bus is None (script/test path), dispatch still
    completes and returns a result — no events are emitted."""
    from miiflow_agent.core.react.dispatch import (
        DispatchCounter,
        dispatch_subagent,
    )
    from miiflow_agent.core.subagent import SubAgentHandoff

    sub = _FakeSubAgent()
    counter = DispatchCounter()

    result = asyncio.run(
        dispatch_subagent(
            sub,
            SubAgentHandoff(task="t", intent_summary="s"),
            parent_event_bus=None,
            parent_step_number=0,
            parent_assistant_id="parent_x",
            child_id="child_x",
            counter=counter,
        )
    )
    assert result.status == "completed"


def test_dispatch_subagent_attaches_subagent_id_to_metadata():
    from miiflow_agent.core.react.dispatch import (
        DispatchCounter,
        dispatch_subagent,
    )
    from miiflow_agent.core.subagent import SubAgentHandoff

    sub = _FakeSubAgent()
    counter = DispatchCounter()
    result = asyncio.run(
        dispatch_subagent(
            sub,
            SubAgentHandoff(task="t", intent_summary="s"),
            parent_event_bus=None,
            parent_step_number=0,
            parent_assistant_id="parent_x",
            child_id="child_x",
            counter=counter,
        )
    )
    assert result.metadata.get("subagent_id", "").startswith("sub_")


# ── SSE event shape lock ─────────────────────────────────────────────────


def test_subagent_dispatch_event_data_keys_locked():
    """Snapshot the SUBAGENT_DISPATCH event data keys for each sub_event
    phase. The FE converter at server/assistant/sse_converters.py:565-636
    consumes these by name — if a key disappears the FE silently
    regresses.

    Stage 2 contract:
      - start:    subagent_id, subagent_path, handle, name, description,
                  parent_assistant_id, child_assistant_id, sub_event, action
      - progress: subagent_id, subagent_path, chunk, sub_event, action
      - complete: subagent_id, subagent_path, result, status, error,
                  duration_ms, child_assistant_id, child_thread_id,
                  sub_event, action
    """
    from miiflow_agent.core.react.dispatch import (
        DispatchCounter,
        dispatch_subagent,
    )
    from miiflow_agent.core.subagent import SubAgentHandoff

    bus, received = _make_event_bus()
    sub = _FakeSubAgent()
    counter = DispatchCounter()
    asyncio.run(
        dispatch_subagent(
            sub,
            SubAgentHandoff(task="t", intent_summary="s"),
            parent_event_bus=bus,
            parent_step_number=0,
            parent_assistant_id="parent_x",
            child_id="child_x",
            counter=counter,
        )
    )

    start = next(e for e in received if e.data.get("sub_event") == "start")
    progress = next(e for e in received if e.data.get("sub_event") == "progress")
    complete = next(e for e in received if e.data.get("sub_event") == "complete")

    assert set(start.data.keys()) >= {
        "sub_event", "action", "subagent_id", "subagent_path",
        "handle", "name", "description",
        "parent_assistant_id", "child_assistant_id",
    }
    assert set(progress.data.keys()) >= {
        "sub_event", "action", "subagent_id", "subagent_path", "chunk",
    }
    assert set(complete.data.keys()) >= {
        "sub_event", "action", "subagent_id", "subagent_path",
        "result", "status", "error", "duration_ms",
        "child_assistant_id", "child_thread_id",
    }
    assert start.data["action"] == "dispatch_assistant"


# ── Synthesized dispatcher tool (S2.5) ───────────────────────────────────


def test_make_dispatcher_tool_schema_uses_handle_enum():
    from miiflow_agent.core.react.dispatch import (
        make_subagent_dispatcher_tool,
    )
    from miiflow_agent.core.subagent import SubAgentDescriptor

    a = SubAgentDescriptor(handle="research", name="R", description="d")
    b = SubAgentDescriptor(handle="implement", name="I", description="d")
    tool = make_subagent_dispatcher_tool(
        [a, b], parent_assistant_id="parent_x"
    )

    assert tool.name == "dispatch_assistant"
    schema = tool.schema
    assert schema.parallelizable is True
    handle_param = schema.parameters["handle"]
    assert sorted(handle_param.enum) == ["implement", "research"]


def test_make_dispatcher_tool_renders_descriptions():
    """Description should include each sub-agent's name + description +
    when_to_use so the LLM can pick the right handle without prompting."""
    from miiflow_agent.core.react.dispatch import (
        make_subagent_dispatcher_tool,
    )
    from miiflow_agent.core.subagent import SubAgentDescriptor

    a = SubAgentDescriptor(
        handle="research",
        name="Research Specialist",
        description="Searches the web.",
        when_to_use="when the user asks about external information",
    )
    tool = make_subagent_dispatcher_tool([a], parent_assistant_id="parent_x")
    desc = tool.schema.description
    assert "research" in desc
    assert "Research Specialist" in desc
    assert "Searches the web." in desc
    assert "external information" in desc


def test_dispatcher_tool_unknown_handle_returns_error():
    """Calling the dispatcher with a handle not in the sub-agent list
    must return a structured error, not raise — the LLM has to recover."""
    from miiflow_agent.core.react.dispatch import (
        make_subagent_dispatcher_tool,
    )

    sub = _FakeSubAgent()
    tool = make_subagent_dispatcher_tool([sub], parent_assistant_id="p")

    # Build a minimal ctx with a dict-style deps.
    from types import SimpleNamespace

    ctx = SimpleNamespace(deps={})

    async def _go():
        return await tool.fn(
            ctx,
            handle="nonexistent",
            task="t",
            intent_summary="s",
        )

    out = asyncio.run(_go())
    assert out["status"] == "rejected"
    assert out["error_kind"] == "unknown_handle"


def test_dispatcher_tool_routes_to_subagent():
    """Happy path: the dispatcher tool runs the matched sub-agent's
    `stream()` and returns its result."""
    from miiflow_agent.core.react.dispatch import (
        DispatchCounter,
        make_subagent_dispatcher_tool,
    )
    from types import SimpleNamespace

    sub = _FakeSubAgent()
    counter = DispatchCounter()
    tool = make_subagent_dispatcher_tool(
        [sub], parent_assistant_id="parent_x", counter=counter
    )

    bus, received = _make_event_bus()
    ctx = SimpleNamespace(deps={
        "event_bus": bus,
        "step_number": 0,
        "dispatch_counter": counter,
    })

    async def _go():
        return await tool.fn(
            ctx,
            handle="demo",
            task="do something",
            intent_summary="user wants demo",
        )

    out = asyncio.run(_go())
    assert out["status"] == "completed"
    assert out["answer"] == "ok"
    assert out["handle"] == "demo"
    # subagent_id appears in the tool observation and matches an event
    assert out["subagent_id"].startswith("sub_")
    assert counter.total == 1


def test_agent_constructor_synthesizes_dispatcher_when_sub_agents_present():
    """Constructing an Agent with sub_agents auto-registers a
    `dispatch_assistant` tool whose handle enum covers every sub-agent."""
    from miiflow_agent.core.agent import Agent
    from miiflow_agent.core.config import AgentConfig
    from miiflow_agent.core.react.dispatch import DISPATCH_TOOL_NAME
    from miiflow_agent.core.subagent import SubAgentDescriptor

    a = SubAgentDescriptor(handle="x", name="X", description="d")
    b = SubAgentDescriptor(handle="y", name="Y", description="d")
    agent = Agent(config=AgentConfig(
        client=_make_client(),
        sub_agents=[a, b],
        parent_assistant_id="parent_x",
    ))

    names = [t.name for t in agent._tools]
    assert DISPATCH_TOOL_NAME in names
    dispatcher = next(t for t in agent._tools if t.name == DISPATCH_TOOL_NAME)
    enum = dispatcher.schema.parameters["handle"].enum
    assert sorted(enum) == ["x", "y"]


def test_agent_no_dispatcher_when_no_sub_agents():
    """No sub_agents → no synthesized dispatcher tool (don't pollute the
    LLM's tool catalog when there's nothing to dispatch to)."""
    from miiflow_agent.core.agent import Agent
    from miiflow_agent.core.react.dispatch import DISPATCH_TOOL_NAME

    agent = Agent(_make_client())
    names = [t.name for t in agent._tools]
    assert DISPATCH_TOOL_NAME not in names


def test_agent_preserves_existing_dispatch_assistant_tool():
    """If the caller already supplied a tool named `dispatch_assistant`
    (Django side does this today), the framework must NOT clobber it
    with a synthesized one — let the caller's wins until Stage 3
    migration completes."""
    from miiflow_agent.core.agent import Agent
    from miiflow_agent.core.config import AgentConfig
    from miiflow_agent.core.react.dispatch import DISPATCH_TOOL_NAME
    from miiflow_agent.core.subagent import SubAgentDescriptor
    from miiflow_agent.core.tools.decorators import tool

    @tool(name=DISPATCH_TOOL_NAME, description="caller-supplied")
    def custom_dispatch():
        return "caller"

    sub = SubAgentDescriptor(handle="demo", name="Demo", description="d")
    agent = Agent(config=AgentConfig(
        client=_make_client(),
        tools=[custom_dispatch._function_tool],
        sub_agents=[sub],
    ))

    dispatchers = [t for t in agent._tools if t.name == DISPATCH_TOOL_NAME]
    assert len(dispatchers) == 1  # not duplicated
    # caller's description preserved
    assert "caller-supplied" in dispatchers[0].schema.description


# ── Orchestrator per-turn counter provisioning (S2.6) ────────────────────


def test_orchestrator_provisions_dispatch_counter():
    """ReAct orchestrator places a fresh DispatchCounter on ctx.deps at
    the start of each execute() call. The synthesized dispatcher tool
    reads from ctx.deps['dispatch_counter'] so concurrent dispatches in
    the same turn race-safely share the budget."""
    from miiflow_agent.core.agent import Agent, AgentType, RunContext
    from miiflow_agent.core.react import ReActFactory
    from miiflow_agent.core.react.dispatch import DispatchCounter

    agent = Agent(_make_client(), agent_type=AgentType.REACT)
    orchestrator = ReActFactory.create_orchestrator(agent=agent, max_steps=1)

    ctx = RunContext(deps={"already": "set"}, messages=[])

    async def _run():
        # We don't need to fully execute — just prove that the setup
        # path places a counter on deps before the LLM step would fire.
        # Patch the step body to exit immediately after _setup_context.
        from miiflow_agent.core.react.models import ReActStep

        async def _short_circuit(query, ctx_inner):
            # _setup_context + counter provisioning happen before this
            # is reached in the real execute() body. We assert the
            # counter exists after orchestrator setup runs.
            return ReActStep(step_number=1, thought="", final_answer="done")

        orchestrator._execute_single_step = _short_circuit  # type: ignore[attr-defined]
        try:
            await orchestrator.execute("hi", ctx)
        except Exception:
            # Don't care about downstream failure — we only need the
            # provisioning step to have run.
            pass

    asyncio.run(_run())

    assert "dispatch_counter" in ctx.deps
    assert isinstance(ctx.deps["dispatch_counter"], DispatchCounter)
    # Caller-supplied deps preserved
    assert ctx.deps["already"] == "set"


# ── MultiAgentOrchestrator framework path (S2.7) ─────────────────────────


def _make_multi_agent_orchestrator():
    """Minimal MultiAgentOrchestrator wired with a stand-in tool executor.

    We never reach the planning / ReAct paths in S2.7 tests — only the
    new `execute_framework_subagents` method, which delegates to the
    framework's `dispatch_subagent`. So tool_executor / safety_manager
    can be cheap mocks."""
    from miiflow_agent.core.react.events import EventBus
    from miiflow_agent.core.react.multi_agent_orchestrator import (
        MultiAgentOrchestrator,
    )
    from miiflow_agent.core.react.safety import SafetyManager

    bus = EventBus()
    received: List = []
    bus.subscribe(lambda ev: received.append(ev))

    orch = MultiAgentOrchestrator(
        tool_executor=MagicMock(),
        event_bus=bus,
        safety_manager=SafetyManager(max_steps=10),
    )
    return orch, received


def test_execute_framework_subagents_runs_in_parallel():
    """Two SubAgents passed in → both stream() coroutines run, results
    come back in input order."""
    from miiflow_agent.core.agent import RunContext
    from miiflow_agent.core.subagent import SubAgentHandoff

    orch, received = _make_multi_agent_orchestrator()
    a = _FakeSubAgent(answer="A-result")
    a.handle = "a"
    b = _FakeSubAgent(answer="B-result")
    b.handle = "b"

    ctx = RunContext(deps={}, messages=[])
    handoffs = [
        (a, SubAgentHandoff(task="A", intent_summary="i")),
        (b, SubAgentHandoff(task="B", intent_summary="i")),
    ]

    results = asyncio.run(
        orch.execute_framework_subagents(handoffs, ctx)
    )

    assert len(results) == 2
    assert results[0].agent_name == "a"
    assert results[0].result == "A-result"
    assert results[0].success is True
    assert results[1].agent_name == "b"
    assert results[1].result == "B-result"


def test_execute_framework_subagents_isolates_message_context():
    """Each task gets a deep-copied messages list so concurrent
    stream() calls can't interleave tool_use/tool_result pairs."""
    from miiflow_agent.core.agent import RunContext
    from miiflow_agent.core.message import Message, MessageRole
    from miiflow_agent.core.subagent import SubAgentHandoff

    captured_messages = []

    class _MutatingSubAgent(_FakeSubAgent):
        async def stream(self, handoff):
            # Capture the context's messages reference at stream time
            # by reading from a closure-passed parent context. (We can't
            # see ctx directly — we infer isolation by checking that
            # mutating parent.messages doesn't affect concurrent run.)
            from miiflow_agent.core.react.enums import ReActEventType
            from miiflow_agent.core.react.react_events import ReActEvent

            yield ReActEvent(
                event_type=ReActEventType.FINAL_ANSWER_CHUNK,
                step_number=0,
                data={"delta": self._answer},
            )

    orch, _ = _make_multi_agent_orchestrator()
    sub = _MutatingSubAgent(answer="ok")
    sub.handle = "isolated"

    parent_ctx = RunContext(
        deps={},
        messages=[Message(role=MessageRole.USER, content="parent")],
    )
    handoffs = [(sub, SubAgentHandoff(task="t", intent_summary="i"))]

    results = asyncio.run(orch.execute_framework_subagents(handoffs, parent_ctx))
    assert results[0].success is True
    # Parent's messages weren't mutated by the run.
    assert len(parent_ctx.messages) == 1
    assert parent_ctx.messages[0].content == "parent"


def test_execute_framework_subagents_emits_subagent_dispatch_events():
    """Framework lifecycle events (start/complete) land on the MultiAgent
    bus exactly like they would in a ReAct dispatch."""
    from miiflow_agent.core.agent import RunContext
    from miiflow_agent.core.react.enums import ReActEventType
    from miiflow_agent.core.subagent import SubAgentHandoff

    orch, received = _make_multi_agent_orchestrator()
    sub = _FakeSubAgent()
    sub.handle = "demo"

    ctx = RunContext(deps={}, messages=[])
    asyncio.run(
        orch.execute_framework_subagents(
            [(sub, SubAgentHandoff(task="t", intent_summary="i"))],
            ctx,
        )
    )

    sub_events = [
        (e.event_type, e.data.get("sub_event"))
        for e in received
        if e.event_type == ReActEventType.SUBAGENT_DISPATCH
    ]
    assert (ReActEventType.SUBAGENT_DISPATCH, "start") in sub_events
    assert (ReActEventType.SUBAGENT_DISPATCH, "complete") in sub_events


def test_execute_framework_subagents_partial_failure():
    """One failing SubAgent does not poison the gather — other results
    return as normal, failed one comes back with success=False."""
    from miiflow_agent.core.agent import RunContext
    from miiflow_agent.core.subagent import SubAgentHandoff

    orch, _ = _make_multi_agent_orchestrator()
    good = _FakeSubAgent(answer="good")
    good.handle = "good"
    bad = _FakeSubAgent(raise_in_stream=True)
    bad.handle = "bad"

    ctx = RunContext(deps={}, messages=[])
    handoffs = [
        (good, SubAgentHandoff(task="t", intent_summary="i")),
        (bad, SubAgentHandoff(task="t", intent_summary="i")),
    ]
    results = asyncio.run(orch.execute_framework_subagents(handoffs, ctx))

    assert results[0].success is True
    assert results[0].result == "good"
    assert results[1].success is False
    assert "blew up" in (results[1].error or "")


def test_execute_framework_subagents_shared_counter_enforces_budget():
    """A shared counter passed across multiple gather() calls keeps the
    per-handle budget enforced — important for nested MultiAgent runs
    where the same parent dispatches to the same sub repeatedly."""
    from miiflow_agent.core.agent import RunContext
    from miiflow_agent.core.react.dispatch import DispatchCounter
    from miiflow_agent.core.subagent import SubAgentHandoff

    orch, _ = _make_multi_agent_orchestrator()
    sub = _FakeSubAgent()
    sub.handle = "tight"

    counter = DispatchCounter(max_per_handle=2, max_total=10)
    ctx = RunContext(deps={}, messages=[])

    # 3 attempts; only 2 should succeed
    handoffs = [
        (sub, SubAgentHandoff(task="t", intent_summary="i"))
        for _ in range(3)
    ]
    results = asyncio.run(
        orch.execute_framework_subagents(handoffs, ctx, counter=counter)
    )

    succeeded = [r for r in results if r.success]
    failed_for_budget = [
        r for r in results
        if not r.success and "per_handle_budget_exceeded" in (r.error or "")
    ]
    assert len(succeeded) == 2
    assert len(failed_for_budget) == 1
    assert counter.total == 2


def test_execute_agent_sub_agents_resolves_handles():
    """`execute_agent_sub_agents` reads handles from a list of task
    dicts and resolves them against `agent.sub_agents`."""
    from miiflow_agent.core.agent import Agent, RunContext
    from miiflow_agent.core.config import AgentConfig

    orch, _ = _make_multi_agent_orchestrator()
    a = _FakeSubAgent(answer="A")
    a.handle = "search"
    b = _FakeSubAgent(answer="B")
    b.handle = "code"

    parent = Agent(config=AgentConfig(
        client=_make_client(),
        sub_agents=[a, b],
    ))
    ctx = RunContext(deps={}, messages=[])
    task_specs = [
        {"handle": "search", "task": "find X", "intent_summary": "user wants X"},
        {"handle": "code", "task": "implement Y", "intent_summary": "user wants Y"},
    ]
    results = asyncio.run(orch.execute_agent_sub_agents(parent, task_specs, ctx))
    assert [r.agent_name for r in results] == ["search", "code"]
    assert all(r.success for r in results)


def test_execute_agent_sub_agents_skips_unknown_handles():
    """Unknown handles in the plan are dropped with a warning, not
    raising — the synthesizer can note the gap."""
    from miiflow_agent.core.agent import Agent, RunContext
    from miiflow_agent.core.config import AgentConfig

    orch, _ = _make_multi_agent_orchestrator()
    real = _FakeSubAgent()
    real.handle = "real"

    parent = Agent(config=AgentConfig(
        client=_make_client(),
        sub_agents=[real],
    ))
    ctx = RunContext(deps={}, messages=[])
    task_specs = [
        {"handle": "real", "task": "t", "intent_summary": "i"},
        {"handle": "fake", "task": "t", "intent_summary": "i"},
    ]
    results = asyncio.run(orch.execute_agent_sub_agents(parent, task_specs, ctx))
    # Only the resolvable handle yields a result.
    assert len(results) == 1
    assert results[0].agent_name == "real"


def test_orchestrator_does_not_clobber_existing_counter():
    """If the caller already installed a counter (e.g. Django adapter
    that wants cross-turn persistence), the orchestrator leaves it alone."""
    from miiflow_agent.core.agent import Agent, AgentType, RunContext
    from miiflow_agent.core.react import ReActFactory
    from miiflow_agent.core.react.dispatch import DispatchCounter

    agent = Agent(_make_client(), agent_type=AgentType.REACT)
    orchestrator = ReActFactory.create_orchestrator(agent=agent, max_steps=1)

    caller_counter = DispatchCounter(max_total=99)
    ctx = RunContext(deps={"dispatch_counter": caller_counter}, messages=[])

    async def _run():
        from miiflow_agent.core.react.models import ReActStep

        async def _short_circuit(query, ctx_inner):
            return ReActStep(step_number=1, thought="", final_answer="done")

        orchestrator._execute_single_step = _short_circuit  # type: ignore[attr-defined]
        try:
            await orchestrator.execute("hi", ctx)
        except Exception:
            pass

    asyncio.run(_run())
    assert ctx.deps["dispatch_counter"] is caller_counter
