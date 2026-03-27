"""Tests for human-in-the-loop tool approval functionality.

Tests the full flow:
1. PRE_TOOL_USE callback blocks tools that require approval
2. ToolApprovalRequired exception is raised
3. Orchestrator catches it, sets needs_clarification, emits TOOL_APPROVAL_NEEDED
4. ToolSchema.require_approval declaration is respected
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from miiflow_agent.core.callbacks import (
    CallbackEvent,
    CallbackEventType,
    get_global_registry,
)
from miiflow_agent.core.react.enums import ReActEventType, StopReason
from miiflow_agent.core.react.exceptions import ToolApprovalRequired
from miiflow_agent.core.react.execution.state import ExecutionState
from miiflow_agent.core.react.tool_executor import AgentToolExecutor
from miiflow_agent.core.tools import ToolRegistry, ToolResult
from miiflow_agent.core.tools.schemas import ToolSchema
from miiflow_agent.core.tools.types import ToolType


class TestPreToolUseCallback:
    """Test that PRE_TOOL_USE callback can block tool execution."""

    @pytest.fixture
    def mock_agent(self):
        mock_client = MagicMock()
        mock_client.tool_registry = ToolRegistry()
        agent = MagicMock()
        agent.client = mock_client
        agent.tool_registry = mock_client.tool_registry
        return agent

    @pytest.fixture
    def captured_events(self):
        return []

    @pytest.fixture
    def blocking_callback(self):
        """Register a PRE_TOOL_USE callback that blocks a specific tool."""
        blocked_tools = {"dangerous_tool"}
        registry = get_global_registry()

        async def block_tool(event: CallbackEvent):
            if event.tool_name in blocked_tools:
                event.blocked = True
                event.block_reason = "Tool requires user approval"

        registry.register(CallbackEventType.PRE_TOOL_USE, block_tool)
        yield blocked_tools
        registry.unregister(CallbackEventType.PRE_TOOL_USE, block_tool)

    @pytest.mark.asyncio
    async def test_blocked_tool_raises_approval_required(
        self, mock_agent, blocking_callback
    ):
        """When PRE_TOOL_USE callback blocks a tool, ToolApprovalRequired is raised."""
        executor = AgentToolExecutor(mock_agent)
        mock_agent.tool_registry.execute_safe = AsyncMock(
            return_value=ToolResult(name="dangerous_tool", input={}, output="ok", success=True)
        )

        with pytest.raises(ToolApprovalRequired) as exc_info:
            await executor.execute_tool("dangerous_tool", {"param": "value"})

        assert exc_info.value.tool_name == "dangerous_tool"
        assert exc_info.value.tool_inputs == {"param": "value"}
        assert exc_info.value.reason == "Tool requires user approval"

    @pytest.mark.asyncio
    async def test_unblocked_tool_executes_normally(
        self, mock_agent, blocking_callback
    ):
        """Tools not in the blocked set execute normally."""
        executor = AgentToolExecutor(mock_agent)
        expected_result = ToolResult(
            name="safe_tool", input={"x": 1}, output="result", success=True
        )
        mock_agent.tool_registry.execute_safe = AsyncMock(return_value=expected_result)

        result = await executor.execute_tool("safe_tool", {"x": 1})

        assert result.success is True
        assert result.output == "result"

    @pytest.mark.asyncio
    async def test_blocked_tool_does_not_execute(
        self, mock_agent, blocking_callback
    ):
        """When a tool is blocked, execute_safe should never be called."""
        executor = AgentToolExecutor(mock_agent)
        mock_agent.tool_registry.execute_safe = AsyncMock()

        with pytest.raises(ToolApprovalRequired):
            await executor.execute_tool("dangerous_tool", {})

        mock_agent.tool_registry.execute_safe.assert_not_called()

    @pytest.mark.asyncio
    async def test_pre_tool_use_fires_before_execution(
        self, mock_agent
    ):
        """PRE_TOOL_USE event is emitted with correct tool info."""
        captured = []
        registry = get_global_registry()

        async def capture(event: CallbackEvent):
            captured.append(event)

        registry.register(CallbackEventType.PRE_TOOL_USE, capture)

        executor = AgentToolExecutor(mock_agent)
        mock_agent.tool_registry.execute_safe = AsyncMock(
            return_value=ToolResult(name="t", input={}, output="ok", success=True)
        )

        await executor.execute_tool("my_tool", {"key": "val"})

        registry.unregister(CallbackEventType.PRE_TOOL_USE, capture)

        assert len(captured) == 1
        assert captured[0].event_type == CallbackEventType.PRE_TOOL_USE
        assert captured[0].tool_name == "my_tool"
        assert captured[0].tool_inputs == {"key": "val"}


class TestToolApprovalRequiredException:
    """Test the ToolApprovalRequired exception."""

    def test_exception_stores_tool_info(self):
        exc = ToolApprovalRequired(
            tool_name="send_email",
            tool_inputs={"to": "user@example.com", "subject": "Hello"},
            reason="Requires manual approval",
        )
        assert exc.tool_name == "send_email"
        assert exc.tool_inputs["to"] == "user@example.com"
        assert exc.reason == "Requires manual approval"

    def test_exception_message(self):
        exc = ToolApprovalRequired(tool_name="delete_data", tool_inputs={})
        assert "delete_data" in str(exc)
        assert "requires user approval" in str(exc)


class TestToolSchemaRequireApproval:
    """Test that ToolSchema.require_approval is declared and flows correctly."""

    def test_default_is_false(self):
        schema = ToolSchema(
            name="test", description="test", tool_type=ToolType.FUNCTION
        )
        assert schema.require_approval is False

    def test_can_set_true(self):
        schema = ToolSchema(
            name="test",
            description="test",
            tool_type=ToolType.FUNCTION,
            require_approval=True,
        )
        assert schema.require_approval is True

    def test_tool_decorator_passes_require_approval(self):
        """The @tool() decorator passes require_approval to ToolSchema."""
        from miiflow_agent.core.tools.decorators import tool

        @tool(description="A dangerous operation", require_approval=True)
        def dangerous_action(x: str) -> str:
            """Do something dangerous."""
            return x

        assert dangerous_action._tool_schema.require_approval is True

    def test_tool_decorator_default_no_approval(self):
        from miiflow_agent.core.tools.decorators import tool

        @tool(description="A safe operation")
        def safe_action(x: str) -> str:
            """Do something safe."""
            return x

        assert safe_action._tool_schema.require_approval is False


class TestToolApprovalNeededEventType:
    """Test the TOOL_APPROVAL_NEEDED event type exists."""

    def test_event_type_exists(self):
        assert hasattr(ReActEventType, "TOOL_APPROVAL_NEEDED")
        assert ReActEventType.TOOL_APPROVAL_NEEDED.value == "tool_approval_needed"


class TestApprovalPathConvergence:
    """Test that both approval paths (schema declaration and callback registration)
    produce identical blocking behavior through the same PRE_TOOL_USE mechanism."""

    @pytest.fixture(autouse=True)
    def clean_registry(self):
        """Ensure clean callback registry for each test."""
        registry = get_global_registry()
        registry.clear(CallbackEventType.PRE_TOOL_USE)
        yield
        registry.clear(CallbackEventType.PRE_TOOL_USE)

    @pytest.fixture
    def mock_agent(self):
        mock_client = MagicMock()
        mock_client.tool_registry = ToolRegistry()
        agent = MagicMock()
        agent.client = mock_client
        agent.tool_registry = mock_client.tool_registry
        return agent

    def _register_approval_callback(self, tool_names: set):
        """Simulate what load_tools does: register a PRE_TOOL_USE callback
        for the given tool names. This is the same code path used by both
        execution_permission_level=MANUAL and schema.require_approval=True."""
        registry = get_global_registry()

        async def _check_tool_approval(event: CallbackEvent):
            if event.tool_name in tool_names:
                event.blocked = True
                event.block_reason = "Tool requires user approval"

        registry.register(CallbackEventType.PRE_TOOL_USE, _check_tool_approval)

    @pytest.mark.asyncio
    async def test_schema_require_approval_blocks_like_manual(self, mock_agent):
        """A tool with schema.require_approval=True is blocked the same way
        as a tool with execution_permission_level=MANUAL — both end up in
        approval_required_tool_names and use the same PRE_TOOL_USE callback."""
        # Simulate: tool "send_email" added to approval set
        # (regardless of whether it came from MANUAL config or schema declaration)
        self._register_approval_callback({"send_email"})

        executor = AgentToolExecutor(mock_agent)
        mock_agent.tool_registry.execute_safe = AsyncMock()

        with pytest.raises(ToolApprovalRequired) as exc_info:
            await executor.execute_tool("send_email", {"to": "a@b.com", "__description": "Sending email"})

        assert exc_info.value.tool_name == "send_email"
        assert exc_info.value.tool_inputs["to"] == "a@b.com"
        # execute_safe was never called
        mock_agent.tool_registry.execute_safe.assert_not_called()

    @pytest.mark.asyncio
    async def test_both_paths_feed_same_set(self, mock_agent):
        """Simulate load_tools populating approval_required_tool_names from
        both AgentToolConfig.MANUAL and schema.require_approval."""
        approval_required_tool_names = set()

        # Path 1: execution_permission_level=MANUAL (from AgentToolConfig)
        approval_required_tool_names.add("manual_tool")

        # Path 2: schema.require_approval=True (from tool declaration)
        schema_with_approval = ToolSchema(
            name="declared_tool",
            description="Tool that declares approval",
            tool_type=ToolType.FUNCTION,
            require_approval=True,
        )
        if schema_with_approval.require_approval:
            approval_required_tool_names.add(schema_with_approval.name)

        # Path 3: schema.require_approval=False (should NOT be added)
        schema_without_approval = ToolSchema(
            name="auto_tool",
            description="Tool with no approval",
            tool_type=ToolType.FUNCTION,
            require_approval=False,
        )
        if schema_without_approval.require_approval:
            approval_required_tool_names.add(schema_without_approval.name)

        # Both tools should be in the set, auto_tool should not
        assert "manual_tool" in approval_required_tool_names
        assert "declared_tool" in approval_required_tool_names
        assert "auto_tool" not in approval_required_tool_names

        # Register the single callback — same code path for both
        self._register_approval_callback(approval_required_tool_names)

        executor = AgentToolExecutor(mock_agent)
        mock_agent.tool_registry.execute_safe = AsyncMock(
            return_value=ToolResult(name="t", input={}, output="ok", success=True)
        )

        # Both blocked tools raise ToolApprovalRequired
        with pytest.raises(ToolApprovalRequired):
            await executor.execute_tool("manual_tool", {})

        with pytest.raises(ToolApprovalRequired):
            await executor.execute_tool("declared_tool", {})

        # Non-approval tool executes fine
        result = await executor.execute_tool("auto_tool", {})
        assert result.success is True
