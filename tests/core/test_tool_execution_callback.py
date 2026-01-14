"""Tests for tool execution callback functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from miiflow_agent.core.callbacks import (
    CallbackEvent,
    CallbackEventType,
    CallbackRegistry,
    get_global_registry,
)
from miiflow_agent.core.react.tool_executor import AgentToolExecutor
from miiflow_agent.core.tools import ToolRegistry, ToolResult


class TestToolExecutionCallback:
    """Test that tool execution emits TOOL_EXECUTED callbacks."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent with tool registry."""
        mock_client = MagicMock()
        mock_client.tool_registry = ToolRegistry()

        agent = MagicMock()
        agent.client = mock_client
        agent.tool_registry = mock_client.tool_registry

        return agent

    @pytest.fixture
    def captured_events(self):
        """Container for captured callback events."""
        return []

    @pytest.fixture
    def callback_registry(self, captured_events):
        """Get a clean callback registry for testing."""
        registry = get_global_registry()

        async def capture_event(event: CallbackEvent):
            captured_events.append(event)

        registry.register(CallbackEventType.TOOL_EXECUTED, capture_event)
        yield registry
        registry.unregister(CallbackEventType.TOOL_EXECUTED, capture_event)

    @pytest.mark.asyncio
    async def test_tool_execution_emits_callback(
        self, mock_agent, callback_registry, captured_events
    ):
        """Test that executing a tool emits a TOOL_EXECUTED callback event."""
        # Create tool executor
        executor = AgentToolExecutor(mock_agent)

        # Mock the tool registry's execute_safe method
        expected_result = ToolResult(
            name="test_tool", input={"arg1": "value1"}, output="test output", success=True
        )
        mock_agent.tool_registry.execute_safe = AsyncMock(return_value=expected_result)

        # Execute a tool
        result = await executor.execute_tool("test_tool", {"arg1": "value1"})

        # Verify callback was emitted
        assert len(captured_events) == 1
        event = captured_events[0]

        assert event.event_type == CallbackEventType.TOOL_EXECUTED
        assert event.tool_name == "test_tool"
        assert event.tool_inputs == {"arg1": "value1"}
        assert event.tool_output == "test output"
        assert event.success is True
        assert event.tool_execution_time_ms is not None
        assert event.tool_execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_tool_execution_callback_with_context_injection(
        self, mock_agent, callback_registry, captured_events
    ):
        """Test callback when tool uses context injection."""
        executor = AgentToolExecutor(mock_agent)

        expected_result = ToolResult(
            name="context_tool", input={"arg": "value"}, output="context output", success=True
        )
        mock_agent.tool_registry.execute_safe_with_context = AsyncMock(
            return_value=expected_result
        )

        mock_context = MagicMock()
        result = await executor.execute_tool(
            "context_tool", {"arg": "value"}, context=mock_context
        )

        # Verify callback was emitted
        assert len(captured_events) == 1
        event = captured_events[0]

        assert event.event_type == CallbackEventType.TOOL_EXECUTED
        assert event.tool_name == "context_tool"

    @pytest.mark.asyncio
    async def test_tool_execution_callback_on_failure(
        self, mock_agent, callback_registry, captured_events
    ):
        """Test callback is emitted even when tool execution fails."""
        executor = AgentToolExecutor(mock_agent)

        # Mock failed tool execution
        failed_result = ToolResult(
            name="failing_tool", input={}, output="error message", success=False
        )
        mock_agent.tool_registry.execute_safe = AsyncMock(return_value=failed_result)

        result = await executor.execute_tool("failing_tool", {})

        # Verify callback was emitted with success=False
        assert len(captured_events) == 1
        event = captured_events[0]

        assert event.event_type == CallbackEventType.TOOL_EXECUTED
        assert event.tool_name == "failing_tool"
        assert event.success is False

    @pytest.mark.asyncio
    async def test_multiple_tool_executions_emit_multiple_callbacks(
        self, mock_agent, callback_registry, captured_events
    ):
        """Test that multiple tool executions emit separate callbacks."""
        executor = AgentToolExecutor(mock_agent)

        # Return different ToolResults for each tool
        async def mock_execute(tool_name, **kwargs):
            return ToolResult(name=tool_name, input=kwargs, output="result", success=True)

        mock_agent.tool_registry.execute_safe = mock_execute

        # Execute multiple tools
        await executor.execute_tool("tool1", {"a": 1})
        await executor.execute_tool("tool2", {"b": 2})
        await executor.execute_tool("tool3", {"c": 3})

        # Verify all callbacks were emitted
        assert len(captured_events) == 3
        assert captured_events[0].tool_name == "tool1"
        assert captured_events[1].tool_name == "tool2"
        assert captured_events[2].tool_name == "tool3"


class TestCallbackEventType:
    """Test the new TOOL_EXECUTED callback event type."""

    def test_tool_executed_event_type_exists(self):
        """Verify TOOL_EXECUTED is a valid callback event type."""
        assert hasattr(CallbackEventType, "TOOL_EXECUTED")
        assert CallbackEventType.TOOL_EXECUTED.value == "tool_executed"

    def test_callback_event_tool_fields(self):
        """Test that CallbackEvent has tool execution fields."""
        event = CallbackEvent(
            event_type=CallbackEventType.TOOL_EXECUTED,
            tool_name="test_tool",
            tool_inputs={"key": "value"},
            tool_output="result",
            tool_execution_time_ms=100.5,
            success=True,
        )

        assert event.tool_name == "test_tool"
        assert event.tool_inputs == {"key": "value"}
        assert event.tool_output == "result"
        assert event.tool_execution_time_ms == 100.5
        assert event.success is True
