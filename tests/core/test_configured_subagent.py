"""Tests for ConfiguredSubAgent — the DynamicSubAgentConfig → SubAgent adapter."""
from __future__ import annotations

import asyncio
from typing import AsyncIterator, List

import pytest

from miiflow_agent.core.react.configured_subagent import (
    ConfiguredSubAgent,
    _compose_query,
    configured_subagents_from_registry,
    make_registry_dispatcher_tool,
)
from miiflow_agent.core.react.enums import ReActEventType
from miiflow_agent.core.react.react_events import ReActEvent
from miiflow_agent.core.react.subagent_registry import (
    DynamicSubAgentConfig,
    SubAgentRegistry,
)
from miiflow_agent.core.subagent import SubAgent, SubAgentHandoff


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeChildAgent:
    """Replaces a real Agent. Records calls and yields canned events.

    Avoids LLM dependencies — ConfiguredSubAgent's job is to translate
    between the SubAgent Protocol and Agent.stream's interface, not to
    test what Agent.stream actually does (that's covered elsewhere).
    """

    def __init__(self, events: List[ReActEvent]):
        self._events = events
        self.last_query: str = ""
        self.last_context = None
        self.last_max_steps: int = 0
        self.last_max_time: float = 0.0

    async def stream(
        self, *, query: str, context, max_steps: int, max_time_seconds: float
    ) -> AsyncIterator[ReActEvent]:
        self.last_query = query
        self.last_context = context
        self.last_max_steps = max_steps
        self.last_max_time = max_time_seconds
        for event in self._events:
            yield event


def _answer_chunk(delta: str, step: int = 0) -> ReActEvent:
    return ReActEvent(
        event_type=ReActEventType.FINAL_ANSWER_CHUNK,
        step_number=step,
        data={"delta": delta},
    )


def _config(**overrides) -> DynamicSubAgentConfig:
    """Minimal valid DynamicSubAgentConfig with overrides."""
    base = dict(
        name="tester",
        description="A test specialist.",
        system_prompt="You are a tester.",
        tools=["file_read"],
        max_steps=7,
        timeout_seconds=42.0,
    )
    base.update(overrides)
    return DynamicSubAgentConfig(**base)


# ---------------------------------------------------------------------------
# Protocol satisfaction & static fields
# ---------------------------------------------------------------------------


def test_configured_subagent_satisfies_subagent_protocol():
    sub = ConfiguredSubAgent(_config(), _FakeChildAgent([]))
    assert isinstance(sub, SubAgent)


def test_protocol_fields_derive_from_config():
    config = _config(name="explorer", description="Find files.")
    sub = ConfiguredSubAgent(config, _FakeChildAgent([]))
    assert sub.handle == "explorer"
    assert sub.name == "explorer"
    assert sub.description == "Find files."
    assert "explorer" in sub.when_to_use
    assert "Find files." in sub.when_to_use
    assert sub.schema is None  # use default dispatch_assistant schema


def test_when_to_use_override_wins():
    sub = ConfiguredSubAgent(
        _config(), _FakeChildAgent([]), when_to_use="Custom hint."
    )
    assert sub.when_to_use == "Custom hint."


# ---------------------------------------------------------------------------
# stream() — query composition, event passthrough, answer accumulation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_passes_query_with_intent_and_user_message():
    fake = _FakeChildAgent([_answer_chunk("done")])
    sub = ConfiguredSubAgent(_config(), fake)
    handoff = SubAgentHandoff(
        task="Find files matching X.",
        intent_summary="User wants to refactor module Y.",
        parent_user_message="please refactor Y for me",
    )

    async for _ in sub.stream(handoff):
        pass

    assert "Find files matching X." in fake.last_query
    assert "[Parent intent: User wants to refactor module Y.]" in fake.last_query
    assert "please refactor Y for me" in fake.last_query


@pytest.mark.asyncio
async def test_stream_yields_every_child_event():
    events = [
        ReActEvent(event_type=ReActEventType.STEP_START, step_number=1, data={}),
        _answer_chunk("hello "),
        _answer_chunk("world"),
    ]
    sub = ConfiguredSubAgent(_config(), _FakeChildAgent(events))

    received = []
    async for event in sub.stream(SubAgentHandoff(task="t", intent_summary="")):
        received.append(event.event_type)

    assert received == [
        ReActEventType.STEP_START,
        ReActEventType.FINAL_ANSWER_CHUNK,
        ReActEventType.FINAL_ANSWER_CHUNK,
    ]


@pytest.mark.asyncio
async def test_final_result_accumulates_answer_chunks():
    events = [_answer_chunk("part 1 "), _answer_chunk("part 2")]
    sub = ConfiguredSubAgent(_config(), _FakeChildAgent(events))

    async for _ in sub.stream(SubAgentHandoff(task="t", intent_summary="")):
        pass

    result = sub.final_result()
    assert result.status == "completed"
    assert result.answer == "part 1 part 2"
    assert result.duration_ms >= 0


@pytest.mark.asyncio
async def test_final_result_when_no_answer_chunks_is_failed():
    """No FINAL_ANSWER_CHUNK in the stream — treated as a failed run.

    Otherwise an empty-string answer would be returned as success, and
    the parent LLM has no way to tell that something went wrong inside
    the child.
    """
    events = [ReActEvent(event_type=ReActEventType.STEP_START, step_number=1, data={})]
    sub = ConfiguredSubAgent(_config(), _FakeChildAgent(events))

    async for _ in sub.stream(SubAgentHandoff(task="t", intent_summary="")):
        pass

    result = sub.final_result()
    assert result.status == "failed"
    assert "no final answer" in (result.error or "")


def test_final_result_before_stream_consumed():
    sub = ConfiguredSubAgent(_config(), _FakeChildAgent([]))
    result = sub.final_result()
    assert result.status == "failed"
    assert "never consumed" in (result.error or "")


@pytest.mark.asyncio
async def test_stop_condition_with_failure_populates_subagent_result():
    """A STOP_CONDITION event carrying a structured failure dict must
    propagate to SubAgentResult.failure so the dispatch envelope can
    surface the cause to the parent agent.

    Otherwise a child halted by RepeatedToolErrorCondition only leaves
    the canned "I ran into repeated issues" fallback answer in the
    parent's tool observation and the parent has no actionable signal.
    """
    failure_payload = {
        "stop_reason": "repeated_tool_error",
        "description": "stopped after 3 consecutive failures of google_ads_query",
        "last_tool": "google_ads_query",
        "last_tool_error": "Segment 'segments.date' is referenced in WHERE but missing from SELECT.",
        "last_tool_input": {"customer_id": "4447141884", "query": "SELECT ..."},
        "attempts_seen": 4,
    }
    events = [
        ReActEvent(
            event_type=ReActEventType.STOP_CONDITION,
            step_number=5,
            data={
                "stop_reason": "repeated_tool_error",
                "description": failure_payload["description"],
                "failure": failure_payload,
            },
        ),
        _answer_chunk("I ran into repeated issues..."),
    ]
    sub = ConfiguredSubAgent(_config(), _FakeChildAgent(events))

    async for _ in sub.stream(SubAgentHandoff(task="t", intent_summary="")):
        pass

    result = sub.final_result()
    assert result.failure == failure_payload
    # Status stays "completed" because the orchestrator still produced
    # the canned fallback answer — but error now carries a real summary
    # so legacy log paths see something meaningful too.
    assert result.status == "completed"
    assert result.error is not None
    assert "google_ads_query" in result.error
    assert "repeated_tool_error" in result.error


@pytest.mark.asyncio
async def test_stop_condition_without_failure_payload_is_ignored():
    """Older orchestrators emit STOP_CONDITION events without ``failure``.
    Those must not regress the existing path — no failure metadata, plain
    answer, plain error semantics.
    """
    events = [
        ReActEvent(
            event_type=ReActEventType.STOP_CONDITION,
            step_number=1,
            data={"stop_reason": "max_steps", "description": "hit step cap"},
        ),
        _answer_chunk("done"),
    ]
    sub = ConfiguredSubAgent(_config(), _FakeChildAgent(events))

    async for _ in sub.stream(SubAgentHandoff(task="t", intent_summary="")):
        pass

    result = sub.final_result()
    assert result.failure is None
    assert result.status == "completed"
    assert result.error is None


# ---------------------------------------------------------------------------
# stream() — max_steps and timeout pass through; deps factory threads
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_config_steps_and_timeout_thread_to_child():
    fake = _FakeChildAgent([_answer_chunk("ok")])
    sub = ConfiguredSubAgent(
        _config(max_steps=11, timeout_seconds=123.0), fake
    )

    async for _ in sub.stream(SubAgentHandoff(task="t", intent_summary="")):
        pass

    assert fake.last_max_steps == 11
    assert fake.last_max_time == 123.0


@pytest.mark.asyncio
async def test_default_deps_factory_surfaces_handoff_payload():
    fake = _FakeChildAgent([_answer_chunk("ok")])
    sub = ConfiguredSubAgent(_config(), fake)
    payload = {"campaign_ids": [1, 2, 3]}

    async for _ in sub.stream(
        SubAgentHandoff(task="t", intent_summary="", structured_payload=payload)
    ):
        pass

    assert fake.last_context is not None
    assert fake.last_context.deps == {"handoff_payload": payload}


@pytest.mark.asyncio
async def test_custom_deps_factory_overrides_default():
    fake = _FakeChildAgent([_answer_chunk("ok")])
    sub = ConfiguredSubAgent(
        _config(),
        fake,
        deps_factory=lambda h: {"custom": "deps", "task_seen": h.task},
    )

    async for _ in sub.stream(SubAgentHandoff(task="the-task", intent_summary="")):
        pass

    assert fake.last_context.deps == {"custom": "deps", "task_seen": "the-task"}


# ---------------------------------------------------------------------------
# Query composition edge cases
# ---------------------------------------------------------------------------


def test_compose_query_task_only():
    handoff = SubAgentHandoff(task="just the task", intent_summary="")
    assert _compose_query(handoff) == "just the task"


def test_compose_query_intent_without_user_message():
    handoff = SubAgentHandoff(task="t", intent_summary="why")
    composed = _compose_query(handoff)
    assert composed.startswith("t")
    assert "[Parent intent: why]" in composed
    assert "user message" not in composed


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def test_configured_subagents_from_registry_preserves_priority_order():
    registry = SubAgentRegistry()
    registry.register(_config(name="low", priority=1))
    registry.register(_config(name="high", priority=10))
    registry.register(_config(name="mid", priority=5))

    sub_agents = configured_subagents_from_registry(
        registry, child_agent_factory=lambda c: _FakeChildAgent([])
    )

    assert [s.handle for s in sub_agents] == ["high", "mid", "low"]


def test_make_registry_dispatcher_tool_returns_function_tool():
    from miiflow_agent.core.tools.function.function_tool import FunctionTool

    registry = SubAgentRegistry()
    registry.register(_config(name="alpha"))
    registry.register(_config(name="beta"))

    tool = make_registry_dispatcher_tool(
        registry,
        child_agent_factory=lambda c: _FakeChildAgent([]),
        parent_assistant_id="parent_x",
    )

    assert isinstance(tool, FunctionTool)
    assert tool.name == "dispatch_assistant"
    # The handle enum on the schema should list both registered configs.
    handle_param = tool.schema.parameters["handle"]
    assert sorted(handle_param.enum) == ["alpha", "beta"]


# ---------------------------------------------------------------------------
# Integration: dispatch_subagent + ConfiguredSubAgent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_subagent_with_configured_subagent_completes():
    from miiflow_agent.core.react.dispatch import (
        DispatchCounter,
        dispatch_subagent,
    )

    fake = _FakeChildAgent([_answer_chunk("the answer")])
    sub = ConfiguredSubAgent(_config(name="explorer"), fake)

    result = await dispatch_subagent(
        sub,
        SubAgentHandoff(task="find X", intent_summary="why X"),
        parent_event_bus=None,
        parent_step_number=0,
        parent_assistant_id="parent",
        child_id="explorer",
        counter=DispatchCounter(),
    )

    assert result.status == "completed"
    assert result.answer == "the answer"
    # ConfiguredSubAgent forwards intent_summary into the composed query.
    assert "[Parent intent: why X]" in fake.last_query


# ---------------------------------------------------------------------------
# Default-template removal: deprecation warning on register_defaults=True
# ---------------------------------------------------------------------------


def test_register_defaults_true_emits_deprecation_warning():
    with pytest.warns(DeprecationWarning, match="register_defaults=True"):
        registry = SubAgentRegistry(register_defaults=True)
    # Registry still ships empty regardless of the flag.
    assert len(registry) == 0


def test_register_defaults_false_is_silent_and_empty():
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")  # any warning is a test failure
        registry = SubAgentRegistry()
    assert len(registry) == 0
