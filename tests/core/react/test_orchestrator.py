"""Comprehensive tests for ReAct orchestrator execution flow."""

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from typing import List, Optional

from miiflow_agent import (
    LLMClient,
    Agent,
    AgentType,
    RunContext,
    Message,
    tool,
    ToolRegistry,
)
from miiflow_agent.core.react.orchestrator import (
    ExecutionState,
    ReActOrchestrator,
    _extract_failure_metadata,
)
from miiflow_agent.core.react.factory import ReActFactory
from miiflow_agent.core.react.enums import ReActEventType, StopReason
from miiflow_agent.core.react.models import ReActStep, ReActResult, ToolInvocation
from miiflow_agent.core.react.events import EventBus
from miiflow_agent.core.react.safety import RepeatedToolErrorCondition, SafetyManager
from miiflow_agent.core.react.tool_executor import AgentToolExecutor
from miiflow_agent.core.message import MessageRole
from miiflow_agent.core.tools.schemas import ToolResult


@dataclass
class MockDeps:
    """Mock dependencies for testing."""

    user_id: str = "test_user"


class TestExecutionState:
    """Test ExecutionState internal class."""

    def test_initial_state(self):
        """Test initial execution state."""
        state = ExecutionState()
        assert state.current_step == 0
        assert state.steps == []
        assert state.final_answer is None
        assert state.is_running is True

    def test_step_increment(self):
        """Test step counter increment."""
        state = ExecutionState()
        state.current_step += 1
        assert state.current_step == 1

    def test_final_answer_stops_execution(self):
        """Test that setting final answer signals completion."""
        state = ExecutionState()
        state.final_answer = "The answer is 42"
        # Note: is_running is controlled by the orchestrator loop, not the state
        assert state.final_answer == "The answer is 42"


class TestReActOrchestratorSetup:
    """Test ReAct orchestrator setup and initialization."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        mock_client = MagicMock()
        mock_client.provider_name = "openai"
        agent = MagicMock(spec=Agent)
        agent.client = mock_client
        agent.tool_registry = MagicMock()
        agent.tool_registry.list_tools.return_value = ["add", "multiply"]
        agent._tools = []
        return agent

    @pytest.fixture
    def orchestrator(self, mock_agent):
        """Create orchestrator with mock components."""
        return ReActFactory.create_orchestrator(
            agent=mock_agent,
            max_steps=10,
            max_budget=None,
            max_time_seconds=None,
        )

    def test_orchestrator_creation(self, orchestrator):
        """Test orchestrator is created with correct components."""
        assert orchestrator.tool_executor is not None
        assert orchestrator.event_bus is not None
        assert orchestrator.safety_manager is not None

    def test_context_setup_with_empty_query(self, orchestrator):
        """Test that empty query raises error when no user message exists."""
        context = RunContext(deps=None, messages=[])

        with pytest.raises(ValueError, match="Query cannot be empty"):
            orchestrator._setup_context("", context)

    def test_context_setup_with_existing_user_message(self, orchestrator):
        """Test that empty query is allowed if user message already exists."""
        context = RunContext(
            deps=None, messages=[Message(role=MessageRole.USER, content="Hello")]
        )
        # Should not raise
        orchestrator._setup_context("", context)

    def test_context_setup_adds_system_prompt(self, orchestrator):
        """Test that system prompt is added to context."""
        context = RunContext(deps=None, messages=[])
        orchestrator._setup_context("What is 2+2?", context)

        # Should have system prompt and user message
        assert len(context.messages) >= 2
        system_msgs = [m for m in context.messages if m.role == MessageRole.SYSTEM]
        assert len(system_msgs) >= 1


class TestReActTurnClassification:
    @pytest.mark.asyncio
    async def test_text_before_tool_call_is_not_final_answer_chunk(self):
        """A provider text preamble followed by tool_use is thinking, not answer."""

        class FakeExecutor:
            agent = SimpleNamespace(temperature=0, max_tokens=1024)

            def _build_native_tool_schemas(self):
                return [{"name": "lookup"}]

            async def stream_with_tools(self, messages):
                yield SimpleNamespace(
                    delta="I'll check that. ",
                    thinking_delta=None,
                    tool_calls=None,
                    usage=None,
                    cost=0,
                    finish_reason=None,
                )
                yield SimpleNamespace(
                    delta="",
                    thinking_delta=None,
                    tool_calls=[
                        {
                            "id": "tc_1",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": {"query": "x"},
                            },
                        }
                    ],
                    usage=None,
                    cost=0,
                    finish_reason="tool_calls",
                )

            def get_tool_schema(self, name):
                return {"parameters": {"required": []}}

            def has_tool(self, name):
                return name == "lookup"

            async def execute_tool(self, name, inputs, context=None):
                return ToolResult(name=name, input=inputs, output="ok", success=True)

        orch = ReActOrchestrator.__new__(ReActOrchestrator)
        orch.tool_executor = FakeExecutor()
        orch.event_bus = EventBus()
        orch.context_compressor = None
        orch.safety_manager = SafetyManager(max_steps=5)

        context = RunContext(deps={}, messages=[Message.user("do it")])
        state = ExecutionState()
        state.current_step = 1

        await ReActOrchestrator._execute_reasoning_step_native(orch, context, state)

        event_types = [e.event_type for e in orch.event_bus.event_buffer]
        assert ReActEventType.FINAL_ANSWER_CHUNK not in event_types
        assert ReActEventType.THINKING_CHUNK in event_types
        assert ReActEventType.ACTION_PLANNED in event_types

    @pytest.mark.asyncio
    async def test_truncated_answer_text_is_non_final_thinking_chunk(self):
        """A length-truncated answer is surfaced, but never as committed final text."""

        class FakeExecutor:
            agent = SimpleNamespace(temperature=0, max_tokens=8)

            def _build_native_tool_schemas(self):
                return []

            async def stream_with_tools(self, messages):
                yield SimpleNamespace(
                    delta="This answer was cut off mid",
                    thinking_delta=None,
                    tool_calls=None,
                    usage=None,
                    cost=0,
                    finish_reason="length",
                )

        orch = ReActOrchestrator.__new__(ReActOrchestrator)
        orch.tool_executor = FakeExecutor()
        orch.event_bus = EventBus()
        orch.context_compressor = None
        orch.safety_manager = SafetyManager(max_steps=5)

        context = RunContext(deps={}, messages=[Message.user("answer")])
        state = ExecutionState()
        state.current_step = 1

        step = await ReActOrchestrator._execute_reasoning_step_native(
            orch, context, state
        )

        event_types = [e.event_type for e in orch.event_bus.event_buffer]
        assert ReActEventType.THINKING_CHUNK in event_types
        assert ReActEventType.FINAL_ANSWER_CHUNK not in event_types
        assert step.answer is None


class TestReActOrchestratorSafetyConditions:
    """Test safety conditions and stop mechanisms."""

    def test_safety_manager_max_steps(self):
        """Test SafetyManager enforces max steps."""
        manager = SafetyManager(max_steps=5)

        steps = []
        current_step = 4

        # Should not stop at step 4
        condition = manager.should_stop(steps, current_step)
        assert condition is None

        current_step = 5
        condition = manager.should_stop(steps, current_step)
        assert condition is not None
        assert condition.get_stop_reason() == StopReason.MAX_STEPS

    def test_safety_manager_max_budget(self):
        """Test SafetyManager enforces budget limit."""
        manager = SafetyManager(max_steps=100, max_budget=1.0)

        # Steps with cost below budget
        steps = [ReActStep(step_number=1, thought="T", cost=0.5)]
        current_step = 1

        condition = manager.should_stop(steps, current_step)
        assert condition is None

        # Steps with cost above budget
        steps = [
            ReActStep(step_number=1, thought="T", cost=0.5),
            ReActStep(step_number=2, thought="T", cost=0.6),
        ]
        current_step = 2

        condition = manager.should_stop(steps, current_step)
        assert condition is not None
        assert condition.get_stop_reason() == StopReason.MAX_BUDGET


def _step_with_invocation(
    step_number: int,
    *,
    name: str,
    observation: str,
    error: Optional[str] = None,
) -> ReActStep:
    """Build a ReActStep carrying one ToolInvocation. Centralised so the
    error-condition tests below stay focused on the matching logic."""
    inv = ToolInvocation(
        tool_call_id=f"toolu_{step_number}",
        name=name,
        observation=observation,
        error=error,
    )
    return ReActStep(
        step_number=step_number,
        thought="",
        action=name,
        tool_invocations=[inv],
    )


class TestRepeatedToolErrorCondition:
    """RepeatedToolErrorCondition catches loops where the model retries
    the same tool with varying inputs and the tool keeps rejecting with
    the same error code. ErrorThresholdCondition can't see these because
    soft tool errors (observations containing ``{"error": "..."}``) are
    deliberately not classified as step-level failures."""

    def test_fires_on_three_same_code_errors_in_a_row(self):
        cond = RepeatedToolErrorCondition(max_repeats=3)
        steps = [
            _step_with_invocation(
                i,
                name="recommend_action",
                observation=(
                    "{'error': 'investigation_report failed validation', "
                    "'code': 'investigation_report_invalid'}"
                ),
            )
            for i in range(1, 4)
        ]
        assert cond.should_stop(steps, current_step=3) is True

    def test_does_not_fire_on_two_errors(self):
        cond = RepeatedToolErrorCondition(max_repeats=3)
        steps = [
            _step_with_invocation(
                i,
                name="recommend_action",
                observation="{'error': 'x', 'code': 'investigation_report_invalid'}",
            )
            for i in range(1, 3)
        ]
        assert cond.should_stop(steps, current_step=2) is False

    def test_does_not_fire_when_codes_differ(self):
        """Different error codes mean the model is making progress
        (each retry uncovers a new issue); not a loop."""
        cond = RepeatedToolErrorCondition(max_repeats=3)
        steps = [
            _step_with_invocation(
                1,
                name="recommend_action",
                observation="{'error': 'a', 'code': 'investigation_report_invalid'}",
            ),
            _step_with_invocation(
                2,
                name="recommend_action",
                observation="{'error': 'b', 'code': 'enrich_action_type_mismatch'}",
            ),
            _step_with_invocation(
                3,
                name="recommend_action",
                observation="{'error': 'c', 'code': 'investigation_report_invalid'}",
            ),
        ]
        assert cond.should_stop(steps, current_step=3) is False

    def test_does_not_fire_when_tools_differ(self):
        """Same error code on different tools is also progress, not a loop."""
        cond = RepeatedToolErrorCondition(max_repeats=3)
        steps = [
            _step_with_invocation(
                1,
                name="recommend_action",
                observation="{'error': 'x', 'code': 'foo'}",
            ),
            _step_with_invocation(
                2,
                name="set_investigation",
                observation="{'error': 'x', 'code': 'foo'}",
            ),
            _step_with_invocation(
                3,
                name="recommend_action",
                observation="{'error': 'x', 'code': 'foo'}",
            ),
        ]
        assert cond.should_stop(steps, current_step=3) is False

    def test_success_in_streak_resets(self):
        """A successful invocation between failures breaks the streak —
        the model recovered and is now retrying for a different reason."""
        cond = RepeatedToolErrorCondition(max_repeats=3)
        steps = [
            _step_with_invocation(
                1,
                name="recommend_action",
                observation="{'error': 'x', 'code': 'foo'}",
            ),
            _step_with_invocation(
                2,
                name="recommend_action",
                observation="{'success': True}",
            ),
            _step_with_invocation(
                3,
                name="recommend_action",
                observation="{'error': 'x', 'code': 'foo'}",
            ),
            _step_with_invocation(
                4,
                name="recommend_action",
                observation="{'error': 'x', 'code': 'foo'}",
            ),
        ]
        assert cond.should_stop(steps, current_step=4) is False

    def test_framework_error_also_counted(self):
        """Framework-level errors (transport, malformed tool_use) count
        toward the streak even though they don't carry a ``code``."""
        cond = RepeatedToolErrorCondition(max_repeats=3)
        steps = [
            _step_with_invocation(
                i,
                name="google_ads_query",
                observation="",
                error="Tool 'google_ads_query' failed: API error: ...",
            )
            for i in range(1, 4)
        ]
        assert cond.should_stop(steps, current_step=3) is True

    def test_handles_observation_without_code(self):
        """Observations carrying ``{"error": "..."}`` but no ``code``
        field still count — fall back to the observation prefix as the
        signature. Same prefix N times → loop."""
        cond = RepeatedToolErrorCondition(max_repeats=3)
        steps = [
            _step_with_invocation(
                i,
                name="some_tool",
                observation='{"error": "exact same message every time"}',
            )
            for i in range(1, 4)
        ]
        assert cond.should_stop(steps, current_step=3) is True

    def test_ignores_observations_with_error_none(self):
        """Regression: dispatch_assistant returns
        ``{'handle': ..., 'status': 'completed', 'error': None}`` on
        every successful sub-agent dispatch. The pre-fix substring
        check matched any observation containing the literal
        ``'error'`` — so 3+ parallel dispatches to the same sub-agent
        handle registered as the same (tool, prefix) error key and
        tripped the condition. The root agent stopped before it could
        process the investigator results and call recommend_action.

        After the fix: ``'error': None`` (and any falsy/non-string
        value) does NOT count as an error invocation."""
        cond = RepeatedToolErrorCondition(max_repeats=3)
        steps = [
            _step_with_invocation(
                i,
                name="dispatch_assistant",
                observation=(
                    "{'handle': 'opportunity_investigator', "
                    "'child_assistant_id': 'assistant_X', "
                    "'subagent_id': 'sub_aaa', "
                    "'status': 'completed', "
                    "'answer': '...long JSON report...', "
                    "'error': None}"
                ),
            )
            for i in range(1, 5)  # 4 successful parallel dispatches
        ]
        assert (
            cond.should_stop(steps, current_step=4) is False
        ), "successful dispatches with 'error': None must not be classified as errors"

    def test_ignores_observations_with_error_null_json(self):
        """Same regression in JSON-quoted form: ``\"error\": null``
        on a successful tool return must not register as an error."""
        cond = RepeatedToolErrorCondition(max_repeats=3)
        steps = [
            _step_with_invocation(
                i,
                name="dispatch_assistant",
                observation='{"handle": "x", "status": "completed", "error": null}',
            )
            for i in range(1, 5)
        ]
        assert cond.should_stop(steps, current_step=4) is False

    def test_wired_into_safety_manager_by_default(self):
        """SafetyManager opts the condition in by default so all agents
        get the protection without per-config wiring."""
        manager = SafetyManager(max_steps=100)
        types = [type(c).__name__ for c in manager.conditions]
        assert "RepeatedToolErrorCondition" in types

    def test_safety_manager_returns_condition_when_triggered(self):
        """End-to-end via SafetyManager. Inputs must differ so the
        plain RepeatedActionsCondition (which matches on action +
        action_input) doesn't fire first — that's the whole point of
        the new condition: catch loops where the model varies input."""
        manager = SafetyManager(max_steps=100, max_repeated_tool_errors=3)
        steps = []
        for i in range(1, 4):
            step = _step_with_invocation(
                i,
                name="recommend_action",
                observation="{'error': 'x', 'code': 'foo'}",
            )
            step.action_input = {"attempt": i}  # vary input each retry
            step.tool_invocations[0].inputs = {"attempt": i}
            steps.append(step)
        result = manager.should_stop(steps, current_step=3)
        assert result is not None
        assert result.get_stop_reason() == StopReason.REPEATED_TOOL_ERROR


class TestReActResult:
    """Test ReActResult data structure."""

    def test_result_statistics(self):
        """Test result calculates correct statistics."""
        steps = [
            ReActStep(
                step_number=1,
                thought="Think",
                action="tool1",
                execution_time=0.5,
                tokens_used=100,
            ),
            ReActStep(
                step_number=2, thought="Think more", execution_time=0.3, tokens_used=50
            ),
            ReActStep(
                step_number=3,
                thought="Final",
                answer="Done",
                execution_time=0.2,
                tokens_used=30,
            ),
        ]

        result = ReActResult(
            steps=steps,
            final_answer="Done",
            stop_reason=StopReason.ANSWER_COMPLETE,
        )

        assert result.steps_count == 3
        assert result.action_steps_count == 1
        assert result.error_steps_count == 0
        assert result.total_tokens == 180
        assert result.total_execution_time == 1.0

    def test_result_success_rate(self):
        """Test success rate calculation."""
        steps = [
            ReActStep(step_number=1, thought="OK"),
            ReActStep(step_number=2, thought="Error", error="Failed"),
            ReActStep(step_number=3, thought="OK", answer="Done"),
        ]

        result = ReActResult(
            steps=steps,
            final_answer="Done",
            stop_reason=StopReason.ANSWER_COMPLETE,
        )

        # 2 out of 3 steps succeeded
        assert result.success_rate == pytest.approx(2 / 3)

    def test_result_tools_used(self):
        """Test tools used extraction."""
        steps = [
            ReActStep(step_number=1, thought="T", action="search"),
            ReActStep(step_number=2, thought="T", action="calculate"),
            ReActStep(step_number=3, thought="T", action="search"),  # duplicate
            ReActStep(step_number=4, thought="T", answer="Done"),
        ]

        result = ReActResult(
            steps=steps,
            final_answer="Done",
            stop_reason=StopReason.ANSWER_COMPLETE,
        )

        tools = result.tools_used
        assert "search" in tools
        assert "calculate" in tools
        assert len(tools) == 2  # unique tools


class TestEventBus:
    """Test EventBus functionality."""

    @pytest.mark.asyncio
    async def test_event_subscription(self):
        """Test subscribing to events."""
        bus = EventBus()
        events_received = []

        def handler(event):
            events_received.append(event)

        bus.subscribe(handler)

        # Create and publish an event
        event = MagicMock()
        event.event_type = ReActEventType.STEP_START
        await bus.publish(event)  # publish is async

        assert len(events_received) == 1

    def test_event_filtering(self):
        """Test getting events by filter."""
        bus = EventBus()

        # Add some events to the internal buffer
        event1 = MagicMock()
        event1.event_type = ReActEventType.STEP_START
        event1.step_number = 1

        event2 = MagicMock()
        event2.event_type = ReActEventType.THOUGHT
        event2.step_number = 1

        event3 = MagicMock()
        event3.event_type = ReActEventType.STEP_START
        event3.step_number = 2

        bus.event_buffer = [event1, event2, event3]

        # Filter by event type
        step_starts = bus.get_events(event_type=ReActEventType.STEP_START)
        assert len(step_starts) == 2


class TestNativeToolCallingMode:
    """Test ReAct orchestrator with native tool calling."""

    def _create_mock_agent_with_native_tools(self, responses: List[dict]):
        """Helper to create a mock agent for native tool calling mode.

        Args:
            responses: List of dicts with 'content', 'tool_calls', and optional 'finish_reason'
        """
        from miiflow_agent.core.client import StreamChunk

        mock_model_client = MagicMock()
        mock_model_client.provider_name = "openai"
        mock_model_client.convert_schema_to_provider_format = MagicMock(
            side_effect=lambda x: x
        )

        response_list = list(responses)
        response_index = [0]

        async def create_stream_generator(response_data):
            """Create an async generator for a response with optional tool calls."""
            content = response_data.get("content", "")
            tool_calls = response_data.get("tool_calls", None)

            # Stream content in chunks
            chunk_size = 50
            for i in range(0, len(content), chunk_size):
                chunk_text = content[i : i + chunk_size]
                chunk = StreamChunk(
                    content=content[: i + chunk_size],
                    delta=chunk_text,
                    finish_reason=None,
                    usage=None,
                    tool_calls=None,
                )
                yield chunk

            # Final chunk with tool calls and finish reason
            final_chunk = StreamChunk(
                content=content,
                delta="",
                finish_reason=response_data.get("finish_reason", "stop"),
                usage=MagicMock(
                    prompt_tokens=10, completion_tokens=20, total_tokens=30
                ),
                tool_calls=tool_calls,
            )
            yield final_chunk

        async def mock_astream_chat(*args, **kwargs):
            if response_index[0] < len(response_list):
                response_data = response_list[response_index[0]]
                response_index[0] += 1
                async for chunk in create_stream_generator(response_data):
                    yield chunk

        mock_model_client.astream_chat = mock_astream_chat

        mock_llm_client = MagicMock()
        mock_llm_client.client = mock_model_client
        mock_llm_client._client = mock_model_client
        mock_llm_client.astream_chat = mock_astream_chat
        mock_llm_client.tool_registry = ToolRegistry()

        agent = MagicMock()
        agent.client = mock_llm_client
        agent.tool_registry = mock_llm_client.tool_registry
        agent.temperature = 0.7
        agent.max_tokens = None
        agent._tools = []

        return agent

    @pytest.mark.asyncio
    async def test_native_mode_direct_answer(self):
        """Test native mode with direct answer (no tool calls)."""
        response = {
            "content": "The answer is 42.",
            "tool_calls": None,
            "finish_reason": "stop",
        }

        mock_agent = self._create_mock_agent_with_native_tools([response])

        orchestrator = ReActFactory.create_orchestrator(
            agent=mock_agent,
            max_steps=5,
        )

        context = RunContext(deps=None, messages=[])
        result = await orchestrator.execute("What is the meaning of life?", context)

        assert isinstance(result, ReActResult)
        assert result.final_answer == "The answer is 42."
        assert result.stop_reason == StopReason.ANSWER_COMPLETE

    @pytest.mark.asyncio
    async def test_native_mode_with_tool_call(self):
        """Test native mode with tool call in response."""
        # First response: tool call only (no preamble, per prompt invariant)
        first_response = {
            "content": "",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": '{"expression": "2+2"}',
                    },
                }
            ],
            "finish_reason": "tool_calls",
        }

        # Second response: final answer (plain text, no XML)
        second_response = {
            "content": "2 + 2 equals 4.",
            "tool_calls": None,
            "finish_reason": "stop",
        }

        mock_agent = self._create_mock_agent_with_native_tools(
            [first_response, second_response]
        )

        @tool("calculator", "Calculate an expression")
        def calculator(expression: str) -> str:
            return str(eval(expression))

        mock_agent.tool_registry.register(calculator)

        orchestrator = ReActFactory.create_orchestrator(
            agent=mock_agent,
            max_steps=5,
        )

        context = RunContext(deps=None, messages=[])
        result = await orchestrator.execute("What is 2 + 2?", context)

        assert isinstance(result, ReActResult)
        assert "4" in result.final_answer
        assert result.stop_reason == StopReason.ANSWER_COMPLETE

    @pytest.mark.asyncio
    async def test_preamble_with_tool_call_logs_observability_event(self, caplog):
        """When the model emits text alongside a tool call in the same turn,
        the orchestrator should still execute the tool and reach the real
        final answer. The preamble may briefly appear in the answer stream
        as a UI artifact (the price of live streaming UX), and the
        tool-call preamble is logged at INFO for observability.
        """
        preamble = "Let me pull data from both platforms simultaneously."
        real_answer = "Based on the data, campaign ROAS is 3.5x for the month."

        first_response = {
            "content": preamble,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "fetch_metrics",
                        "arguments": '{"platform": "meta"}',
                    },
                }
            ],
            "finish_reason": "tool_calls",
        }
        second_response = {
            "content": real_answer,
            "tool_calls": None,
            "finish_reason": "stop",
        }

        mock_agent = self._create_mock_agent_with_native_tools(
            [first_response, second_response]
        )

        @tool("fetch_metrics", "Fetch campaign metrics")
        def fetch_metrics(platform: str) -> str:
            return "roas=3.5"

        mock_agent.tool_registry.register(fetch_metrics)

        orchestrator = ReActFactory.create_orchestrator(
            agent=mock_agent,
            max_steps=5,
        )

        import logging

        with caplog.at_level(logging.INFO):
            context = RunContext(deps=None, messages=[])
            result = await orchestrator.execute("What is my ROAS?", context)

        assert isinstance(result, ReActResult)
        assert result.final_answer == real_answer
        # The tool-call preamble must have been logged for observability.
        assert any("Tool-call preamble" in rec.message for rec in caplog.records)


class TestToolFuzzyMatching:
    """Test tool name fuzzy matching for LLM hallucinations."""

    @pytest.fixture
    def orchestrator_with_tools(self):
        """Create orchestrator with sample tools."""
        mock_agent = MagicMock()
        mock_agent.client = MagicMock()
        mock_agent.client.provider_name = "openai"
        mock_agent.tool_registry = ToolRegistry()
        mock_agent._tools = []

        @tool("Addition", "Add two numbers")
        def addition(a: int, b: int) -> int:
            return a + b

        @tool("Multiplication", "Multiply two numbers")
        def multiplication(a: int, b: int) -> int:
            return a * b

        @tool("search_database", "Search the database")
        def search_database(query: str) -> str:
            return f"Results for: {query}"

        mock_agent.tool_registry.register(addition)
        mock_agent.tool_registry.register(multiplication)
        mock_agent.tool_registry.register(search_database)

        return ReActFactory.create_orchestrator(
            agent=mock_agent,
            max_steps=10,
        )

    def test_find_similar_tool_prefix_match(self, orchestrator_with_tools):
        """Test finding tool by prefix match."""
        # "Add" should match "Addition"
        result = orchestrator_with_tools._find_similar_tool("Add")
        assert result == "Addition"

    def test_find_similar_tool_case_insensitive(self, orchestrator_with_tools):
        """Test case-insensitive matching."""
        result = orchestrator_with_tools._find_similar_tool("addition")
        assert result == "Addition"

        result = orchestrator_with_tools._find_similar_tool("MULTIPLICATION")
        assert result == "Multiplication"

    def test_find_similar_tool_substring(self, orchestrator_with_tools):
        """Test substring matching."""
        result = orchestrator_with_tools._find_similar_tool("search")
        assert result == "search_database"

    def test_find_similar_tool_no_match(self, orchestrator_with_tools):
        """Test no match returns None."""
        result = orchestrator_with_tools._find_similar_tool("completely_different")
        assert result is None

    def test_find_similar_tool_empty_input(self, orchestrator_with_tools):
        """Test empty input returns None."""
        assert orchestrator_with_tools._find_similar_tool("") is None
        assert orchestrator_with_tools._find_similar_tool(None) is None

    def test_is_similar_enough_basic(self, orchestrator_with_tools):
        """Test basic similarity check."""
        # Same string
        assert orchestrator_with_tools._is_similar_enough("hello", "hello") is True

        # One character difference
        assert orchestrator_with_tools._is_similar_enough("hello", "hallo") is True

        # Two character difference
        assert orchestrator_with_tools._is_similar_enough("hello", "hxllx") is True

    def test_is_similar_enough_length_threshold(self, orchestrator_with_tools):
        """Test length difference threshold."""
        # More than 2 characters length difference
        assert orchestrator_with_tools._is_similar_enough("hi", "hello") is False
        assert orchestrator_with_tools._is_similar_enough("a", "abcd") is False


class TestResultBuilding:
    """Test result building methods."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator with mock components."""
        mock_agent = MagicMock()
        mock_agent.client = MagicMock()
        mock_agent.client.provider_name = "openai"
        mock_agent.tool_registry = MagicMock()
        mock_agent.tool_registry.list_tools.return_value = []
        mock_agent._tools = []

        return ReActFactory.create_orchestrator(
            agent=mock_agent,
            max_steps=10,
        )

    @pytest.mark.asyncio
    async def test_build_result_with_final_answer(self, orchestrator):
        """Test building result when final answer is present."""
        state = ExecutionState()
        state.steps = [
            ReActStep(
                step_number=1,
                thought="Thinking",
                answer="The answer",
                cost=0.01,
                tokens_used=100,
            )
        ]
        state.final_answer = "The answer"

        result = await orchestrator._build_result(state)

        assert result.final_answer == "The answer"
        assert result.stop_reason == StopReason.ANSWER_COMPLETE
        assert result.steps_count == 1
        assert result.total_tokens == 100

    @pytest.mark.asyncio
    async def test_build_result_generates_fallback(self, orchestrator):
        """Test building result generates fallback when no answer."""
        state = ExecutionState()
        state.steps = [
            ReActStep(step_number=1, thought="Thinking", observation="Some observation")
        ]
        state.final_answer = None

        result = await orchestrator._build_result(state)

        assert result.stop_reason == StopReason.FORCED_STOP
        assert "Some observation" not in result.final_answer
        assert "complete answer" in result.final_answer

    def test_build_error_result(self, orchestrator):
        """Test building error result."""
        state = ExecutionState()
        state.steps = [ReActStep(step_number=1, thought="Started")]
        error = ValueError("Something went wrong")

        result = orchestrator._build_error_result(state, error)

        assert result.stop_reason == StopReason.FORCED_STOP
        assert "Something went wrong" in result.final_answer
        assert result.steps_count == 1

    def test_generate_fallback_answer_does_not_leak_observation(self, orchestrator):
        """Fallback answer must not expose transient tool observations."""
        steps = [ReActStep(step_number=1, thought="T", observation="The data shows X")]

        fallback = orchestrator._generate_fallback_answer(steps)

        assert "The data shows X" not in fallback
        assert "complete answer" in fallback

    def test_generate_fallback_answer_does_not_leak_thought(self, orchestrator):
        """Fallback answer must not expose internal reasoning as final text."""
        steps = [ReActStep(step_number=1, thought="My analysis is complete")]

        fallback = orchestrator._generate_fallback_answer(steps)

        assert "My analysis is complete" not in fallback
        assert "complete answer" in fallback

    def test_generate_fallback_answer_empty_steps(self, orchestrator):
        """Test fallback answer for empty steps."""
        fallback = orchestrator._generate_fallback_answer([])

        assert "No reasoning steps" in fallback

    def test_generate_fallback_answer_hides_tool_error_leak(self, orchestrator):
        """Regression: consecutive tool-error steps must NOT surface the raw
        observation to the user. Previously the loop halted on
        ErrorThresholdCondition with the last step's observation being a raw
        tool-execution error string, which leaked parameter names and schema
        details into the final answer."""
        raw_error = (
            "Tool execution failed: Tool 'meta_ads_insights' received unknown "
            "parameter(s): ['customer_id', 'query']. Valid parameters are: "
            "['ad_account_id', 'breakdowns', 'date_from', 'date_to', 'level']"
        )
        steps = [
            ReActStep(step_number=i, thought="", observation=raw_error, error=raw_error)
            for i in range(1, 4)
        ]

        fallback = orchestrator._generate_fallback_answer(steps)

        assert raw_error not in fallback
        assert "customer_id" not in fallback
        assert "ad_account_id" not in fallback
        assert "repeated issues" in fallback.lower() or "try" in fallback.lower()


class TestExtractFailureMetadata:
    """``_extract_failure_metadata`` produces the structured payload the
    dispatch envelope carries up to the parent agent when a safety
    condition halts the child loop."""

    def test_pulls_last_failing_invocation_from_observation(self):
        """A tool whose observation carries `Tool execution failed:` is
        counted as a failure even when the step itself isn't marked
        ``is_error_step`` (matches RepeatedToolErrorCondition's lookback)."""
        observation = (
            "Tool execution failed: Tool 'google_ads_query' failed: Segment "
            "'segments.date' is referenced in WHERE but missing from SELECT."
        )
        steps = [
            ReActStep(
                step_number=i,
                thought="",
                tool_invocations=[
                    ToolInvocation(
                        name="google_ads_query",
                        inputs={
                            "customer_id": "1",
                            "query": "SELECT campaign.id FROM campaign WHERE segments.date BETWEEN ...",
                        },
                        observation=observation,
                    )
                ],
            )
            for i in range(1, 5)
        ]

        payload = _extract_failure_metadata(
            steps,
            stop_reason="repeated_tool_error",
            description="stopped after 3 consecutive identical failures",
        )

        assert payload["stop_reason"] == "repeated_tool_error"
        assert payload["last_tool"] == "google_ads_query"
        assert "segments.date" in payload["last_tool_error"]
        assert payload["last_tool_input"]["customer_id"] == "1"
        assert payload["attempts_seen"] == 4

    def test_truncates_long_inputs_and_errors(self):
        """A failing query body can be thousands of chars — the failure
        payload must stay bounded so the parent's tool observation
        doesn't blow up."""
        long_query = (
            "SELECT campaign.id, " + ("metrics.cost_micros, " * 200) + "FROM campaign"
        )
        long_error = "Tool execution failed: " + ("x" * 5000)
        steps = [
            ReActStep(
                step_number=1,
                thought="",
                tool_invocations=[
                    ToolInvocation(
                        name="google_ads_query",
                        inputs={"customer_id": "1", "query": long_query},
                        observation=long_error,
                    )
                ],
            )
        ]

        payload = _extract_failure_metadata(
            steps,
            stop_reason="repeated_tool_error",
            description="halted",
            truncation=200,
            input_truncation=100,
        )

        # Error obs is hard-capped; input value gets a truncation marker.
        assert len(payload["last_tool_error"]) == 200
        assert "truncated" in payload["last_tool_input"]["query"]

    def test_no_failing_invocation_returns_minimal_payload(self):
        """When the safety condition fires but no failing invocation is
        visible in the last 8 steps, we still return the stop_reason and
        description — but no last_tool/last_tool_error fields."""
        steps = [
            ReActStep(step_number=1, thought="thinking", observation="ok"),
        ]

        payload = _extract_failure_metadata(
            steps,
            stop_reason="max_steps",
            description="step cap hit",
        )

        assert payload == {
            "stop_reason": "max_steps",
            "description": "step cap hit",
        }


class TestSanitizeErrorMessage:
    """Test error message sanitization."""

    def test_removes_stack_traces(self):
        """Test that stack traces are removed."""
        error_msg = """Traceback (most recent call last):
  File "/path/to/file.py", line 123, in function
    do_something()
  File "/path/to/other.py", line 456, in other_function
    raise ValueError("The actual error")
ValueError: The actual error"""

        result = _sanitize_error_message(error_msg)

        assert "Traceback" not in result
        assert "File " not in result
        assert "line " not in result
        # Should preserve the actual error message
        assert "actual error" in result.lower() or "ValueError" in result

    def test_handles_empty_input(self):
        """Test handling of empty input."""
        assert _sanitize_error_message("") == "Unknown error occurred"
        assert _sanitize_error_message(None) == "Unknown error occurred"

    def test_truncates_long_messages(self):
        """Test that very long messages are truncated."""
        long_msg = "A" * 1000
        result = _sanitize_error_message(long_msg)

        assert len(result) <= 503  # 500 + "..."
        assert result.endswith("...")

    def test_preserves_simple_error(self):
        """Test that simple error messages are preserved."""
        simple_error = "Connection failed: timeout after 30s"
        result = _sanitize_error_message(simple_error)

        assert result == simple_error


# Import the sanitize function for testing
from miiflow_agent.core.react.orchestrator import _sanitize_error_message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
