"""Comprehensive tests for ReAct orchestrator execution flow."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from typing import List, Optional

from miiflow_llm import LLMClient, Agent, AgentType, RunContext, Message, tool, ToolRegistry
from miiflow_llm.core.react.orchestrator import ReActOrchestrator, ExecutionState
from miiflow_llm.core.react.factory import ReActFactory
from miiflow_llm.core.react.data import (
    ReActStep,
    ReActResult,
    ReActEventType,
    StopReason,
)
from miiflow_llm.core.react.events import EventBus
from miiflow_llm.core.react.safety import SafetyManager
from miiflow_llm.core.react.tool_executor import AgentToolExecutor
from miiflow_llm.core.react.parsing.xml_parser import XMLReActParser
from miiflow_llm.core.message import MessageRole


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
            use_native_tools=False,
        )

    def test_orchestrator_creation(self, orchestrator):
        """Test orchestrator is created with correct components."""
        assert orchestrator.tool_executor is not None
        assert orchestrator.event_bus is not None
        assert orchestrator.safety_manager is not None
        assert orchestrator.parser is not None
        assert orchestrator.use_native_tools is False

    def test_orchestrator_native_mode(self, mock_agent):
        """Test orchestrator in native tool calling mode."""
        orchestrator = ReActFactory.create_orchestrator(
            agent=mock_agent,
            use_native_tools=True,
        )
        assert orchestrator.use_native_tools is True

    def test_context_setup_with_empty_query(self, orchestrator):
        """Test that empty query raises error when no user message exists."""
        context = RunContext(deps=None, messages=[])

        with pytest.raises(ValueError, match="Query cannot be empty"):
            orchestrator._setup_context("", context)

    def test_context_setup_with_existing_user_message(self, orchestrator):
        """Test that empty query is allowed if user message already exists."""
        context = RunContext(
            deps=None,
            messages=[Message(role=MessageRole.USER, content="Hello")]
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


class TestReActOrchestratorExecution:
    """Test ReAct orchestrator execution flow."""

    def _create_mock_agent_with_streaming(self, responses: List[str]):
        """Helper to create a properly mocked Agent with streaming LLMClient.

        The ReAct orchestrator uses streaming via tool_executor.stream_without_tools(),
        which internally calls self._client.astream_chat(). So we need to mock
        astream_chat to return async generators that yield StreamChunk objects.
        """
        from miiflow_llm.core.client import StreamChunk

        # Create mock model client (the underlying provider client)
        mock_model_client = MagicMock()
        mock_model_client.provider_name = "openai"
        mock_model_client.convert_schema_to_provider_format = MagicMock(side_effect=lambda x: x)

        # Create async generator factory for streaming responses
        async def create_stream_generator(response_text):
            """Create an async generator that yields chunks for a response."""
            # Yield the response in chunks (simulate streaming)
            chunk_size = 50  # characters per chunk
            for i in range(0, len(response_text), chunk_size):
                chunk_text = response_text[i:i + chunk_size]
                chunk = StreamChunk(
                    content=response_text[:i + chunk_size],
                    delta=chunk_text,
                    finish_reason=None,
                    usage=None,
                    tool_calls=None,
                )
                yield chunk

            # Final chunk with finish_reason
            final_chunk = StreamChunk(
                content=response_text,
                delta="",
                finish_reason="stop",
                usage=MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30),
                tool_calls=None,
            )
            yield final_chunk

        # Create response generators - copy each time to avoid exhaustion
        response_list = list(responses)
        response_index = [0]  # Use list to allow mutation in closure

        async def mock_astream_chat(*args, **kwargs):
            """Mock async stream chat that returns next response generator."""
            if response_index[0] < len(response_list):
                response_text = response_list[response_index[0]]
                response_index[0] += 1
                async for chunk in create_stream_generator(response_text):
                    yield chunk

        mock_model_client.astream_chat = mock_astream_chat

        # Create mock LLMClient
        mock_llm_client = MagicMock()
        mock_llm_client.client = mock_model_client
        mock_llm_client._client = mock_model_client
        mock_llm_client.astream_chat = mock_astream_chat
        mock_llm_client.tool_registry = ToolRegistry()

        # Create the agent
        agent = MagicMock()
        agent.client = mock_llm_client
        agent.tool_registry = mock_llm_client.tool_registry
        agent.temperature = 0.7
        agent.max_tokens = None
        agent._tools = []

        return agent

    @pytest.mark.asyncio
    async def test_single_step_direct_answer(self):
        """Test agent provides answer without using tools."""
        response_text = """<thinking>Simple question, direct answer.</thinking>

<answer>The answer is 42.</answer>"""

        mock_agent = self._create_mock_agent_with_streaming([response_text])

        # Create orchestrator
        orchestrator = ReActFactory.create_orchestrator(
            agent=mock_agent,
            max_steps=5,
            use_native_tools=False,
        )

        context = RunContext(deps=None, messages=[])
        result = await orchestrator.execute("What is the meaning of life?", context)

        assert isinstance(result, ReActResult)
        assert result.final_answer == "The answer is 42."
        assert result.stop_reason == StopReason.ANSWER_COMPLETE
        assert result.steps_count >= 1

    @pytest.mark.asyncio
    async def test_multi_step_tool_usage(self):
        """Test thought → action → observation → answer flow."""
        # First response: tool call (using correct XML format)
        first_response = """<thinking>I need to calculate.</thinking>

<tool_call name="add">{"a": 2, "b": 2}</tool_call>"""

        # Second response: final answer after observation
        second_response = """<thinking>Got the result, now I can answer.</thinking>

<answer>2 + 2 equals 4.</answer>"""

        mock_agent = self._create_mock_agent_with_streaming([first_response, second_response])

        @tool("add", "Add two numbers")
        def add(a: int, b: int) -> int:
            return a + b

        mock_agent.tool_registry.register(add)

        orchestrator = ReActFactory.create_orchestrator(
            agent=mock_agent,
            max_steps=5,
            use_native_tools=False,
        )

        context = RunContext(deps=None, messages=[])
        result = await orchestrator.execute("What is 2 + 2?", context)

        assert isinstance(result, ReActResult)
        assert "4" in result.final_answer
        assert result.stop_reason == StopReason.ANSWER_COMPLETE
        # Should have at least 2 steps: tool call + answer
        assert result.steps_count >= 1

    @pytest.mark.asyncio
    async def test_max_iterations_stop(self):
        """Test stop condition: max steps reached or forced stop due to continuous tool calls."""
        # Always return a tool call (never answer) - using correct XML format
        tool_response = """<thinking>Let me try again.</thinking>

<tool_call name="search">{"query": "test"}</tool_call>"""

        # Provide enough responses for many iterations
        mock_agent = self._create_mock_agent_with_streaming([tool_response] * 10)

        @tool("search", "Search for something")
        def search(query: str) -> str:
            return f"Results for: {query}"

        mock_agent.tool_registry.register(search)

        # Set max_steps to 3
        orchestrator = ReActFactory.create_orchestrator(
            agent=mock_agent,
            max_steps=3,
            use_native_tools=False,
        )

        context = RunContext(deps=None, messages=[])
        result = await orchestrator.execute("Keep searching", context)

        # Should stop due to either max steps or forced stop
        # (the orchestrator may force stop when no answer is provided)
        assert result.stop_reason in [StopReason.MAX_STEPS, StopReason.FORCED_STOP]
        assert result.steps_count <= 3
        # The tool was being called, so there should be action steps
        assert result.action_steps_count >= 1

    @pytest.mark.asyncio
    async def test_tool_execution_error_recovery(self):
        """Test graceful handling of tool failures."""
        # First response: tool call (using correct XML format)
        tool_call_response = """<thinking>Let me use the tool.</thinking>

<tool_call name="failing_tool">{}</tool_call>"""

        # Second response after error: provide answer
        answer_response = """<thinking>The tool failed, I'll answer based on what I know.</thinking>

<answer>I encountered an error but here's my best answer.</answer>"""

        mock_agent = self._create_mock_agent_with_streaming([tool_call_response, answer_response])

        @tool("failing_tool", "A tool that always fails")
        def failing_tool() -> str:
            raise RuntimeError("Tool execution failed!")

        mock_agent.tool_registry.register(failing_tool)

        orchestrator = ReActFactory.create_orchestrator(
            agent=mock_agent,
            max_steps=5,
            use_native_tools=False,
        )

        context = RunContext(deps=None, messages=[])
        result = await orchestrator.execute("Use the tool", context)

        # Should complete despite error
        assert result.final_answer is not None
        assert result.stop_reason == StopReason.ANSWER_COMPLETE

    @pytest.mark.asyncio
    async def test_context_injection_in_tools(self):
        """Test RunContext passed to tools correctly."""
        # Response that calls the context-aware tool (using correct XML format)
        tool_call_response = """<thinking>Let me get the user info.</thinking>

<tool_call name="get_user_info">{}</tool_call>"""

        answer_response = """<thinking>Got the user info.</thinking>

<answer>Your user ID is test_user_123.</answer>"""

        mock_agent = self._create_mock_agent_with_streaming([tool_call_response, answer_response])

        context_captured = []

        @tool("get_user_info", "Get current user info")
        def get_user_info(ctx: RunContext[MockDeps]) -> str:
            context_captured.append(ctx)
            return f"User ID: {ctx.deps.user_id}"

        mock_agent.tool_registry.register(get_user_info)

        orchestrator = ReActFactory.create_orchestrator(
            agent=mock_agent,
            max_steps=5,
            use_native_tools=False,
        )

        deps = MockDeps(user_id="test_user_123")
        context = RunContext(deps=deps, messages=[])
        result = await orchestrator.execute("Who am I?", context)

        # Verify context was passed to tool
        assert len(context_captured) >= 1
        assert context_captured[0].deps.user_id == "test_user_123"


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


class TestReActResult:
    """Test ReActResult data structure."""

    def test_result_statistics(self):
        """Test result calculates correct statistics."""
        steps = [
            ReActStep(step_number=1, thought="Think", action="tool1", execution_time=0.5, tokens_used=100),
            ReActStep(step_number=2, thought="Think more", execution_time=0.3, tokens_used=50),
            ReActStep(step_number=3, thought="Final", answer="Done", execution_time=0.2, tokens_used=30),
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
        assert result.success_rate == pytest.approx(2/3)

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


class TestXMLParser:
    """Additional parser tests for edge cases."""

    def test_parse_with_markdown_code_blocks(self):
        """Test parsing response with markdown formatting."""
        parser = XMLReActParser()

        response = """<thinking>
Let me think about this problem.
</thinking>

```xml
<tool_call>
<tool_name>search</tool_name>
<tool_input>{"query": "test"}</tool_input>
</tool_call>
```"""

        result = parser.parse_complete(response)
        assert result is not None
        # Parser should handle markdown-wrapped XML

    def test_parse_malformed_json_in_tool_input(self):
        """Test handling of malformed JSON in tool input."""
        parser = XMLReActParser()

        response = """<thinking>Using tool</thinking>

<tool_call>
<tool_name>search</tool_name>
<tool_input>{invalid json}</tool_input>
</tool_call>"""

        # Should not crash, but may return None or partial result
        result = parser.parse_complete(response)
        # Parser should gracefully handle this


class TestNativeToolCallingMode:
    """Test ReAct orchestrator with native tool calling (use_native_tools=True)."""

    def _create_mock_agent_with_native_tools(self, responses: List[dict]):
        """Helper to create a mock agent for native tool calling mode.

        Args:
            responses: List of dicts with 'content', 'tool_calls', and optional 'finish_reason'
        """
        from miiflow_llm.core.client import StreamChunk

        mock_model_client = MagicMock()
        mock_model_client.provider_name = "openai"
        mock_model_client.convert_schema_to_provider_format = MagicMock(side_effect=lambda x: x)

        response_list = list(responses)
        response_index = [0]

        async def create_stream_generator(response_data):
            """Create an async generator for a response with optional tool calls."""
            content = response_data.get("content", "")
            tool_calls = response_data.get("tool_calls", None)

            # Stream content in chunks
            chunk_size = 50
            for i in range(0, len(content), chunk_size):
                chunk_text = content[i:i + chunk_size]
                chunk = StreamChunk(
                    content=content[:i + chunk_size],
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
                usage=MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30),
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
            "content": "<thinking>Simple question.</thinking>\n\n<answer>The answer is 42.</answer>",
            "tool_calls": None,
            "finish_reason": "stop"
        }

        mock_agent = self._create_mock_agent_with_native_tools([response])

        orchestrator = ReActFactory.create_orchestrator(
            agent=mock_agent,
            max_steps=5,
            use_native_tools=True,
        )

        context = RunContext(deps=None, messages=[])
        result = await orchestrator.execute("What is the meaning of life?", context)

        assert isinstance(result, ReActResult)
        assert result.final_answer == "The answer is 42."
        assert result.stop_reason == StopReason.ANSWER_COMPLETE

    @pytest.mark.asyncio
    async def test_native_mode_with_tool_call(self):
        """Test native mode with tool call in response."""
        # First response: tool call
        first_response = {
            "content": "<thinking>I need to calculate.</thinking>",
            "tool_calls": [{
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "calculator",
                    "arguments": '{"expression": "2+2"}'
                }
            }],
            "finish_reason": "tool_calls"
        }

        # Second response: final answer
        second_response = {
            "content": "<thinking>Got the result.</thinking>\n\n<answer>2 + 2 equals 4.</answer>",
            "tool_calls": None,
            "finish_reason": "stop"
        }

        mock_agent = self._create_mock_agent_with_native_tools([first_response, second_response])

        @tool("calculator", "Calculate an expression")
        def calculator(expression: str) -> str:
            return str(eval(expression))

        mock_agent.tool_registry.register(calculator)

        orchestrator = ReActFactory.create_orchestrator(
            agent=mock_agent,
            max_steps=5,
            use_native_tools=True,
        )

        context = RunContext(deps=None, messages=[])
        result = await orchestrator.execute("What is 2 + 2?", context)

        assert isinstance(result, ReActResult)
        assert "4" in result.final_answer
        assert result.stop_reason == StopReason.ANSWER_COMPLETE


class TestResponseClassification:
    """Test response classification methods."""

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

    def test_heuristic_detects_final_answer_phrases(self, orchestrator):
        """Test heuristic detects common final answer indicators."""
        final_answer_texts = [
            "The answer is 42.",
            "In conclusion, the data shows positive growth.",
            "To summarize, you have 5 items.",
            "Based on the analysis, the best option is A.",
            "Therefore, the answer is yes.",
        ]

        for text in final_answer_texts:
            assert orchestrator._heuristic_is_final_answer(text) is True, f"Failed for: {text}"

    def test_heuristic_detects_thinking_phrases(self, orchestrator):
        """Test heuristic detects phrases indicating more work needed."""
        thinking_texts = [
            "I need to check the database first.",
            "Let me analyze the data.",
            "I should verify this information.",
            "First, I will search for the answer.",
            "I'll look into this further.",
        ]

        for text in thinking_texts:
            assert orchestrator._heuristic_is_final_answer(text) is False, f"Failed for: {text}"

    def test_heuristic_detects_declarative_statements(self, orchestrator):
        """Test heuristic detects declarative statements with data."""
        declarative_texts = [
            "You have 131 accounts in your database.",
            "There are 5 items in the list.",
            "The total is 100 units.",
        ]

        for text in declarative_texts:
            assert orchestrator._heuristic_is_final_answer(text) is True, f"Failed for: {text}"

    def test_heuristic_handles_empty_input(self, orchestrator):
        """Test heuristic handles empty or None input."""
        assert orchestrator._heuristic_is_final_answer("") is False
        assert orchestrator._heuristic_is_final_answer(None) is False
        assert orchestrator._heuristic_is_final_answer("   ") is False

    def test_heuristic_long_text_is_answer(self, orchestrator):
        """Test heuristic treats long text as final answer."""
        long_text = "A" * 250  # 250 characters
        assert orchestrator._heuristic_is_final_answer(long_text) is True


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

    def test_build_result_with_final_answer(self, orchestrator):
        """Test building result when final answer is present."""
        state = ExecutionState()
        state.steps = [
            ReActStep(step_number=1, thought="Thinking", answer="The answer", cost=0.01, tokens_used=100)
        ]
        state.final_answer = "The answer"

        result = orchestrator._build_result(state)

        assert result.final_answer == "The answer"
        assert result.stop_reason == StopReason.ANSWER_COMPLETE
        assert result.steps_count == 1
        assert result.total_tokens == 100

    def test_build_result_generates_fallback(self, orchestrator):
        """Test building result generates fallback when no answer."""
        state = ExecutionState()
        state.steps = [
            ReActStep(step_number=1, thought="Thinking", observation="Some observation")
        ]
        state.final_answer = None

        result = orchestrator._build_result(state)

        assert result.stop_reason == StopReason.FORCED_STOP
        assert "Some observation" in result.final_answer

    def test_build_error_result(self, orchestrator):
        """Test building error result."""
        state = ExecutionState()
        state.steps = [ReActStep(step_number=1, thought="Started")]
        error = ValueError("Something went wrong")

        result = orchestrator._build_error_result(state, error)

        assert result.stop_reason == StopReason.FORCED_STOP
        assert "Something went wrong" in result.final_answer
        assert result.steps_count == 1

    def test_generate_fallback_answer_with_observation(self, orchestrator):
        """Test fallback answer uses observation."""
        steps = [ReActStep(step_number=1, thought="T", observation="The data shows X")]

        fallback = orchestrator._generate_fallback_answer(steps)

        assert "The data shows X" in fallback

    def test_generate_fallback_answer_with_thought_only(self, orchestrator):
        """Test fallback answer uses thought when no observation."""
        steps = [ReActStep(step_number=1, thought="My analysis is complete")]

        fallback = orchestrator._generate_fallback_answer(steps)

        assert "My analysis is complete" in fallback

    def test_generate_fallback_answer_empty_steps(self, orchestrator):
        """Test fallback answer for empty steps."""
        fallback = orchestrator._generate_fallback_answer([])

        assert "No reasoning steps" in fallback


class TestErrorHandlingPaths:
    """Test error handling in orchestrator."""

    def _create_mock_agent_with_streaming(self, responses: List[str]):
        """Helper to create agent with streaming mock."""
        from miiflow_llm.core.client import StreamChunk

        mock_model_client = MagicMock()
        mock_model_client.provider_name = "openai"
        mock_model_client.convert_schema_to_provider_format = MagicMock(side_effect=lambda x: x)

        response_list = list(responses)
        response_index = [0]

        async def create_stream_generator(response_text):
            chunk_size = 50
            for i in range(0, len(response_text), chunk_size):
                chunk_text = response_text[i:i + chunk_size]
                chunk = StreamChunk(
                    content=response_text[:i + chunk_size],
                    delta=chunk_text,
                    finish_reason=None,
                    usage=None,
                    tool_calls=None,
                )
                yield chunk

            final_chunk = StreamChunk(
                content=response_text,
                delta="",
                finish_reason="stop",
                usage=MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30),
                tool_calls=None,
            )
            yield final_chunk

        async def mock_astream_chat(*args, **kwargs):
            if response_index[0] < len(response_list):
                response_text = response_list[response_index[0]]
                response_index[0] += 1
                async for chunk in create_stream_generator(response_text):
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
    async def test_tool_not_found_with_fuzzy_correction(self):
        """Test that tool not found triggers fuzzy matching."""
        # LLM requests "Add" but actual tool is "Addition"
        response = """<thinking>Let me add the numbers.</thinking>

<tool_call name="Add">{"a": 2, "b": 3}</tool_call>"""

        answer_response = """<thinking>Got the result.</thinking>

<answer>2 + 3 equals 5.</answer>"""

        mock_agent = self._create_mock_agent_with_streaming([response, answer_response])

        @tool("Addition", "Add two numbers")
        def addition(a: int, b: int) -> int:
            return a + b

        mock_agent.tool_registry.register(addition)

        orchestrator = ReActFactory.create_orchestrator(
            agent=mock_agent,
            max_steps=5,
            use_native_tools=False,
        )

        context = RunContext(deps=None, messages=[])
        result = await orchestrator.execute("What is 2 + 3?", context)

        # Should still complete because of fuzzy matching
        assert result.final_answer is not None

    @pytest.mark.asyncio
    async def test_handles_malformed_tool_input(self):
        """Test graceful handling of malformed tool input."""
        # Tool call with invalid JSON
        response = """<thinking>Using tool.</thinking>

<tool_call name="search">not valid json</tool_call>"""

        answer_response = """<thinking>I'll provide an answer anyway.</thinking>

<answer>Here's what I found.</answer>"""

        mock_agent = self._create_mock_agent_with_streaming([response, answer_response])

        @tool("search", "Search for something")
        def search(query: str) -> str:
            return f"Results: {query}"

        mock_agent.tool_registry.register(search)

        orchestrator = ReActFactory.create_orchestrator(
            agent=mock_agent,
            max_steps=5,
            use_native_tools=False,
        )

        context = RunContext(deps=None, messages=[])
        result = await orchestrator.execute("Search for something", context)

        # Should not crash, should complete
        assert result is not None


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


class TestEventStreamingIntegration:
    """Test event streaming during orchestrator execution."""

    def _create_mock_agent_with_streaming(self, responses: List[str]):
        """Helper to create agent with streaming mock."""
        from miiflow_llm.core.client import StreamChunk

        mock_model_client = MagicMock()
        mock_model_client.provider_name = "openai"
        mock_model_client.convert_schema_to_provider_format = MagicMock(side_effect=lambda x: x)

        response_list = list(responses)
        response_index = [0]

        async def create_stream_generator(response_text):
            chunk_size = 50
            for i in range(0, len(response_text), chunk_size):
                chunk_text = response_text[i:i + chunk_size]
                chunk = StreamChunk(
                    content=response_text[:i + chunk_size],
                    delta=chunk_text,
                    finish_reason=None,
                    usage=None,
                    tool_calls=None,
                )
                yield chunk

            final_chunk = StreamChunk(
                content=response_text,
                delta="",
                finish_reason="stop",
                usage=MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30),
                tool_calls=None,
            )
            yield final_chunk

        async def mock_astream_chat(*args, **kwargs):
            if response_index[0] < len(response_list):
                response_text = response_list[response_index[0]]
                response_index[0] += 1
                async for chunk in create_stream_generator(response_text):
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
    async def test_events_published_during_execution(self):
        """Test that events are published during execution."""
        response = """<thinking>Let me think about this.</thinking>

<answer>The answer is 42.</answer>"""

        mock_agent = self._create_mock_agent_with_streaming([response])

        orchestrator = ReActFactory.create_orchestrator(
            agent=mock_agent,
            max_steps=5,
            use_native_tools=False,
        )

        # Collect events
        events_received = []

        def event_handler(event):
            events_received.append(event)

        orchestrator.event_bus.subscribe(event_handler)

        context = RunContext(deps=None, messages=[])
        await orchestrator.execute("What is the answer?", context)

        # Should have received events
        assert len(events_received) > 0

        # Should have step_start event
        event_types = [e.event_type for e in events_received]
        assert ReActEventType.STEP_START in event_types


# Import the sanitize function for testing
from miiflow_llm.core.react.orchestrator import _sanitize_error_message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
