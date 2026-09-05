"""A provider-executed MCP turn must publish its tool events BEFORE the answer.

Production thread thread_07UFBr5sVrTFxro3UTR7DDiN (2026-09-05): the user asked
for a briefing, Anthropic ran four connector tools server-side, and the whole
answer streamed to the browser before a single tool frame existed. Anthropic's
block order in that turn was mcp_tool_use -> mcp_tool_result -> ... -> text, but
the orchestrator collected the MCP blocks and drained them only after the stream
closed, so on the wire every ACTION_PLANNED/OBSERVATION landed AFTER the last
answer delta (execution_timeline timestamps 1788583812.31-813.08 vs the final
delta at 1788583811.97).

Two things broke as a result:
  1. the reasoning panel materialized above an already-finished answer — the
     frontend renders the panel above the bubble, so with `chunks[]` still empty
     there was no panel at all while the answer typed out;
  2. `streaming_service` treats ACTION_PLANNED as proof that everything streamed
     so far was preamble and zeroes its accumulators, so the late event threw the
     real answer away and it survived only because FINAL_ANSWER then re-emitted
     it whole — a visible double.

These tests pin the wire order, and the parsed arguments that only exist on the
FINALIZED mcp_tool_use block.
"""

from types import SimpleNamespace

import pytest

from miiflow_agent import RunContext
from miiflow_agent.core.message import Message
from miiflow_agent.core.react.enums import ReActEventType
from miiflow_agent.core.react.events import EventBus
from miiflow_agent.core.react.orchestrator import ExecutionState, ReActOrchestrator
from miiflow_agent.core.react.recording import OutcomeRecording
from miiflow_agent.core.react.safety import SafetyManager

CALL_ID = "mcptoolu_01MAeD5F26Uw2Dr2vBxRG2ya"


def _chunk(**kwargs):
    base = dict(
        delta="",
        thinking_delta=None,
        tool_calls=None,
        mcp_tool_results=None,
        usage=None,
        cost=0,
        finish_reason=None,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


class _CapturingRecording(OutcomeRecording):
    """Keeps the real behaviour, remembers what reached the durable seam."""

    def __init__(self, orch):
        super().__init__(orch)
        self.observations = []

    async def record_tool_observation(self, context, state, **kwargs):
        self.observations.append(kwargs)
        return await super().record_tool_observation(context, state, **kwargs)


class _AnthropicShapedExecutor:
    """The live Anthropic wire shape for a connector turn.

    The mcp_tool_use is yielded TWICE for one call: a placeholder at
    `content_block_start` (name known, arguments still generating) and the same
    dict again at `content_block_stop`, by then filled in by the normalizer.
    That second sighting is the only one carrying arguments.
    """

    agent = SimpleNamespace(temperature=0, max_tokens=1024)

    def __init__(self, with_result: bool = True):
        self._with_result = with_result

    def _build_native_tool_schemas(self):
        return []

    async def stream_with_tools(self, messages, prebuilt_tools=None):
        placeholder = {
            "id": CALL_ID,
            "type": "mcp_function",
            "function": {"name": "list_load_calcs", "arguments": {}},
            "server_name": "Kopperfield Admin",
        }
        yield _chunk(tool_calls=[placeholder])
        # content_block_stop: the normalizer parses the accumulated JSON into
        # the SAME dict object and yields it again.
        placeholder["function"]["arguments"] = {"limit": 3}
        yield _chunk(tool_calls=[placeholder])
        if self._with_result:
            yield _chunk(
                mcp_tool_results=[
                    {
                        "tool_use_id": CALL_ID,
                        "is_error": False,
                        "content": '{"load_calculations": []}',
                    }
                ]
            )
        yield _chunk(delta="Here are your 3 most recent load calculations.")
        yield _chunk(finish_reason="end_turn")

    def has_tool(self, name):
        return False

    def list_tools(self):
        return []

    def get_tool_schema(self, name):
        return {"parameters": {"required": []}}

    async def execute_tool(self, name, inputs, context=None):
        raise AssertionError(f"local dispatch attempted for {name}")


async def _run_step(with_result: bool = True):
    orch = ReActOrchestrator.__new__(ReActOrchestrator)
    orch.tool_executor = _AnthropicShapedExecutor(with_result=with_result)
    orch.event_bus = EventBus()
    orch.context_compressor = None
    orch.safety_manager = SafetyManager(max_steps=5)
    recording = _CapturingRecording(orch)
    orch._recording = recording

    context = RunContext(deps={}, messages=[Message.user("brief me on my load calcs")])
    state = ExecutionState()
    state.current_step = 1

    step = await ReActOrchestrator._execute_reasoning_step_native(orch, context, state)
    return orch, recording, step


def _types(orch):
    return [e.event_type for e in orch.event_bus.event_buffer]


@pytest.mark.asyncio
async def test_tool_events_precede_the_first_answer_delta():
    """The regression itself: this order was inverted on the wire."""
    orch, _, _ = await _run_step()
    order = _types(orch)

    planned = order.index(ReActEventType.ACTION_PLANNED)
    observed = order.index(ReActEventType.OBSERVATION)
    first_answer = order.index(ReActEventType.FINAL_ANSWER_CHUNK)

    assert planned < observed < first_answer, order


@pytest.mark.asyncio
async def test_call_is_announced_the_moment_its_block_starts():
    """ACTION_STREAMING is the early chip; local tool calls already get one."""
    orch, _, _ = await _run_step()
    order = _types(orch)

    assert ReActEventType.ACTION_STREAMING in order
    assert order.index(ReActEventType.ACTION_STREAMING) < order.index(
        ReActEventType.ACTION_PLANNED
    )


@pytest.mark.asyncio
async def test_planned_event_carries_the_parsed_arguments():
    """Published on the FINALIZED block, not the content_block_start placeholder.

    The normalizer never parses MCP arguments mid-block, so announcing on the
    first sighting would record an argument-less, blameless call.
    """
    orch, _, _ = await _run_step()
    planned = [
        e
        for e in orch.event_bus.event_buffer
        if e.event_type == ReActEventType.ACTION_PLANNED
    ]
    assert len(planned) == 1
    assert planned[0].data["action_input"] == {"limit": 3}
    assert planned[0].data["executor"] == "native_mcp"
    assert planned[0].data["server_name"] == "Kopperfield Admin"


@pytest.mark.asyncio
async def test_pair_is_published_exactly_once():
    """The post-stream sweep must not re-emit what the stream already recorded."""
    orch, recording, _ = await _run_step()
    order = _types(orch)

    assert order.count(ReActEventType.ACTION_PLANNED) == 1
    assert order.count(ReActEventType.OBSERVATION) == 1
    assert len(recording.observations) == 1


@pytest.mark.asyncio
async def test_the_text_after_the_result_is_still_the_answer():
    _, _, step = await _run_step()
    assert step.answer == "Here are your 3 most recent load calculations."
    assert step.action is None


@pytest.mark.asyncio
async def test_provider_execution_time_is_measured():
    """We do not run these calls, so the block clock is the only duration we get.

    Every native-MCP observation in production carried a NULL
    execution_time_ms (129/129 over seven days) because the call was only ever
    seen after the fact.
    """
    _, recording, _ = await _run_step()
    assert recording.observations[0]["execution_time_ms"] is not None
    assert recording.observations[0]["execution_time_ms"] >= 0


@pytest.mark.asyncio
async def test_a_call_the_provider_never_answered_is_swept_as_a_failure():
    """A silent gap in the trail is worse than a visible incomplete call."""
    orch, recording, _ = await _run_step(with_result=False)
    order = _types(orch)

    assert order.count(ReActEventType.ACTION_PLANNED) == 1
    assert order.count(ReActEventType.OBSERVATION) == 1
    assert recording.observations[0]["success"] is False
    assert (
        recording.observations[0]["observation"]
        == "No result returned by the provider for this call."
    )
