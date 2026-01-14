"""Comprehensive tests for Plan & Execute orchestrator."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from typing import List

from miiflow_agent import RunContext, ToolRegistry, tool
from miiflow_agent.core.react.plan_execute_orchestrator import PlanAndExecuteOrchestrator
from miiflow_agent.core.react.enums import PlanExecuteEventType, StopReason
from miiflow_agent.core.react.models import Plan, SubTask, PlanExecuteResult, ReActResult
from miiflow_agent.core.react.react_events import PlanExecuteEvent
from miiflow_agent.core.react.events import EventBus
from miiflow_agent.core.react.safety import SafetyManager
from miiflow_agent.core.react.tool_executor import AgentToolExecutor
from miiflow_agent.core.react.orchestrator import ReActOrchestrator


class TestPlanDataStructures:
    """Test Plan and SubTask data structures."""

    def test_subtask_creation(self):
        """Test SubTask creation and default values."""
        subtask = SubTask(
            id=1,
            description="Search for information",
            required_tools=["search"],
            dependencies=[],
            success_criteria="Found relevant results",
        )

        assert subtask.id == 1
        assert subtask.description == "Search for information"
        assert subtask.required_tools == ["search"]
        assert subtask.dependencies == []
        assert subtask.status == "pending"
        assert subtask.result is None
        assert subtask.error is None

    def test_subtask_to_dict(self):
        """Test SubTask serialization."""
        subtask = SubTask(
            id=1,
            description="Test task",
            required_tools=["tool1"],
            dependencies=[],
            success_criteria="Done",
        )
        subtask.status = "completed"
        subtask.result = "Success"

        data = subtask.to_dict()

        assert data["id"] == 1
        assert data["description"] == "Test task"
        assert data["status"] == "completed"
        assert data["result"] == "Success"

    def test_plan_creation(self):
        """Test Plan creation with subtasks."""
        subtasks = [
            SubTask(id=1, description="Step 1", required_tools=[], dependencies=[]),
            SubTask(id=2, description="Step 2", required_tools=[], dependencies=[1]),
        ]

        plan = Plan(
            goal="Complete the task",
            reasoning="Breaking down into steps",
            subtasks=subtasks,
        )

        assert plan.goal == "Complete the task"
        assert len(plan.subtasks) == 2
        assert plan.subtasks[1].dependencies == [1]

    def test_plan_statistics(self):
        """Test Plan computed statistics."""
        subtasks = [
            SubTask(id=1, description="Step 1", required_tools=[], dependencies=[]),
            SubTask(id=2, description="Step 2", required_tools=[], dependencies=[]),
            SubTask(id=3, description="Step 3", required_tools=[], dependencies=[]),
        ]
        subtasks[0].status = "completed"
        subtasks[1].status = "completed"
        subtasks[2].status = "failed"

        plan = Plan(goal="Test", reasoning="Test", subtasks=subtasks)

        assert plan.completed_subtasks == 2
        assert plan.failed_subtasks == 1
        assert plan.total_subtasks == 3
        assert plan.progress_percentage == pytest.approx(2 / 3 * 100)


class TestPlanExecuteResult:
    """Test PlanExecuteResult data structure."""

    def test_result_creation(self):
        """Test PlanExecuteResult creation."""
        plan = Plan(goal="Test", reasoning="Test", subtasks=[])
        result = PlanExecuteResult(
            plan=plan,
            final_answer="The answer is 42",
            stop_reason=StopReason.ANSWER_COMPLETE,
            replans=0,
            total_cost=0.001,
            total_execution_time=1.5,
            total_tokens=100,
        )

        assert result.final_answer == "The answer is 42"
        assert result.stop_reason == StopReason.ANSWER_COMPLETE
        assert result.replans == 0
        assert result.total_cost == 0.001
        assert result.total_tokens == 100

    def test_result_to_dict(self):
        """Test PlanExecuteResult serialization."""
        plan = Plan(goal="Test", reasoning="Test", subtasks=[])
        result = PlanExecuteResult(
            plan=plan,
            final_answer="Done",
            stop_reason=StopReason.ANSWER_COMPLETE,
        )

        data = result.to_dict()

        assert "plan" in data
        assert data["final_answer"] == "Done"
        assert data["stop_reason"] == "answer_complete"


class TestPlanExecuteEventTypes:
    """Test Plan & Execute event types."""

    def test_event_types_exist(self):
        """Test all expected event types exist."""
        expected_types = [
            "PLANNING_START",
            "PLANNING_COMPLETE",
            "SUBTASK_START",
            "SUBTASK_COMPLETE",
            "SUBTASK_FAILED",
            "REPLANNING_START",
            "REPLANNING_COMPLETE",
            "FINAL_ANSWER",
        ]

        for type_name in expected_types:
            assert hasattr(PlanExecuteEventType, type_name)

    def test_event_creation(self):
        """Test PlanExecuteEvent creation."""
        event = PlanExecuteEvent(
            event_type=PlanExecuteEventType.PLANNING_START,
            data={"goal": "Test goal"},
        )

        assert event.event_type == PlanExecuteEventType.PLANNING_START
        assert event.data["goal"] == "Test goal"


class TestPlanAndExecuteOrchestrator:
    """Test PlanAndExecuteOrchestrator functionality."""

    def _create_mock_tool_executor(self):
        """Create a mock tool executor."""
        mock_executor = MagicMock(spec=AgentToolExecutor)
        mock_executor.build_tools_description = MagicMock(return_value="- search: Search tool\n- calculate: Calculator")

        # Mock the underlying client
        mock_client = MagicMock()
        mock_model_client = MagicMock()
        mock_client.client = mock_model_client
        mock_executor._client = mock_client

        return mock_executor

    def _create_mock_subtask_orchestrator(self, result_answer="Subtask result", raise_exception=None):
        """Create a mock ReAct orchestrator for subtask execution."""
        mock_subtask_orchestrator = MagicMock(spec=ReActOrchestrator)

        # Create a mock event_bus with subscribe/unsubscribe methods
        mock_event_bus = MagicMock()
        mock_event_bus.subscribe = MagicMock()
        mock_event_bus.unsubscribe = MagicMock()
        mock_subtask_orchestrator.event_bus = mock_event_bus

        if raise_exception:
            mock_subtask_orchestrator.execute = AsyncMock(side_effect=raise_exception)
        else:
            # Configure mock to return successful results
            mock_result = ReActResult(
                final_answer=result_answer,
                stop_reason=StopReason.ANSWER_COMPLETE,
                steps=[],
            )
            mock_subtask_orchestrator.execute = AsyncMock(return_value=mock_result)
        return mock_subtask_orchestrator

    def _create_orchestrator(self, tool_executor=None, subtask_orchestrator=None):
        """Create orchestrator with mock components."""
        if tool_executor is None:
            tool_executor = self._create_mock_tool_executor()

        if subtask_orchestrator is None:
            subtask_orchestrator = self._create_mock_subtask_orchestrator()

        event_bus = EventBus()
        safety_manager = SafetyManager(max_steps=10)

        orchestrator = PlanAndExecuteOrchestrator(
            tool_executor=tool_executor,
            event_bus=event_bus,
            safety_manager=safety_manager,
            subtask_orchestrator=subtask_orchestrator,
            max_replans=2,
        )

        return orchestrator

    def test_orchestrator_creation(self):
        """Test orchestrator initialization."""
        orchestrator = self._create_orchestrator()

        assert orchestrator.max_replans == 2
        assert orchestrator.event_bus is not None
        assert orchestrator.safety_manager is not None

    def test_dependencies_met_no_deps(self):
        """Test dependency check with no dependencies."""
        orchestrator = self._create_orchestrator()

        subtask = SubTask(
            id=1,
            description="Task without deps",
            required_tools=[],
            dependencies=[],
        )

        assert orchestrator._dependencies_met(subtask, set()) is True

    def test_dependencies_met_with_deps(self):
        """Test dependency check with dependencies."""
        orchestrator = self._create_orchestrator()

        subtask = SubTask(
            id=2,
            description="Task with deps",
            required_tools=[],
            dependencies=[1],
        )

        # No deps completed
        assert orchestrator._dependencies_met(subtask, set()) is False

        # Deps completed
        assert orchestrator._dependencies_met(subtask, {1}) is True

    def test_format_plan_status(self):
        """Test plan status formatting."""
        orchestrator = self._create_orchestrator()

        subtasks = [
            SubTask(id=1, description="First task", required_tools=[], dependencies=[]),
            SubTask(id=2, description="Second task", required_tools=[], dependencies=[1]),
        ]
        subtasks[0].status = "completed"
        subtasks[0].result = "Done"
        subtasks[1].status = "failed"
        subtasks[1].error = "Network error"

        plan = Plan(goal="Test", reasoning="Test", subtasks=subtasks)

        status = orchestrator._format_plan_status(plan)

        assert "✓" in status  # completed
        assert "✗" in status  # failed
        assert "First task" in status
        assert "Second task" in status
        assert "Network error" in status

    @pytest.mark.asyncio
    async def test_execute_with_existing_plan(self):
        """Test execution with pre-generated plan."""
        mock_executor = self._create_mock_tool_executor()

        # Mock stream_without_tools for synthesis
        async def mock_stream(*args, **kwargs):
            from miiflow_agent.core.client import StreamChunk
            yield StreamChunk(content="Final answer", delta="Final answer", finish_reason="stop", usage=None, tool_calls=None)

        mock_executor.stream_without_tools = mock_stream

        # Create subtask orchestrator that returns success
        mock_subtask_orchestrator = self._create_mock_subtask_orchestrator(result_answer="Tool output")

        orchestrator = self._create_orchestrator(mock_executor, mock_subtask_orchestrator)

        # Create existing plan
        plan = Plan(
            goal="Test goal",
            reasoning="Pre-planned",
            subtasks=[
                SubTask(
                    id=1,
                    description="Execute search",
                    required_tools=["search"],
                    dependencies=[],
                    success_criteria="Found results",
                ),
            ],
        )

        context = RunContext(deps=None, messages=[])
        result = await orchestrator.execute("Test goal", context, existing_plan=plan)

        assert isinstance(result, PlanExecuteResult)
        assert result.final_answer is not None
        # Should use the existing plan, not generate new one
        assert result.plan.reasoning == "Pre-planned"

    @pytest.mark.asyncio
    async def test_execute_empty_plan_direct_response(self):
        """Test direct response when LLM returns empty plan."""
        mock_executor = self._create_mock_tool_executor()

        # Mock stream_without_tools for direct response
        async def mock_stream(*args, **kwargs):
            from miiflow_agent.core.client import StreamChunk
            yield StreamChunk(content="Hello!", delta="Hello!", finish_reason="stop", usage=None, tool_calls=None)

        mock_executor.stream_without_tools = mock_stream

        orchestrator = self._create_orchestrator(mock_executor)

        # Create empty plan (simple query)
        empty_plan = Plan(goal="Hello", reasoning="Simple greeting", subtasks=[])

        context = RunContext(deps=None, messages=[])
        result = await orchestrator.execute("Hello", context, existing_plan=empty_plan)

        assert isinstance(result, PlanExecuteResult)
        assert result.final_answer == "Hello!"
        assert result.stop_reason == StopReason.ANSWER_COMPLETE
        assert result.replans == 0

    @pytest.mark.asyncio
    async def test_subtask_execution_with_tool(self):
        """Test subtask execution via ReAct orchestrator."""
        mock_executor = self._create_mock_tool_executor()

        # Create subtask orchestrator that returns specific result
        mock_subtask_orchestrator = self._create_mock_subtask_orchestrator(result_answer="Search results")

        orchestrator = self._create_orchestrator(mock_executor, mock_subtask_orchestrator)

        subtask = SubTask(
            id=1,
            description="Search for Python docs",
            required_tools=["search"],
            dependencies=[],
        )

        context = RunContext(deps=None, messages=[])
        success = await orchestrator._execute_subtask(subtask, context, total_subtasks=1)

        assert success is True
        assert subtask.status == "completed"
        assert subtask.result == "Search results"
        mock_subtask_orchestrator.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_subtask_execution_without_tool(self):
        """Test subtask execution without tools (LLM reasoning via ReAct)."""
        mock_executor = self._create_mock_tool_executor()

        # Create subtask orchestrator that returns reasoning result
        mock_subtask_orchestrator = self._create_mock_subtask_orchestrator(result_answer="Reasoning result")

        orchestrator = self._create_orchestrator(mock_executor, mock_subtask_orchestrator)

        subtask = SubTask(
            id=1,
            description="Analyze the data",
            required_tools=[],  # No tools required
            dependencies=[],
        )

        context = RunContext(deps=None, messages=[])
        success = await orchestrator._execute_subtask(subtask, context, total_subtasks=1)

        assert success is True
        assert subtask.status == "completed"
        assert subtask.result == "Reasoning result"

    @pytest.mark.asyncio
    async def test_subtask_execution_tool_failure_marks_status(self):
        """Test that orchestrator exception marks subtask status as failed.

        Note: When the ReAct orchestrator raises an exception (e.g., from a tool failure),
        the subtask is marked as failed and execution returns False.
        """
        mock_executor = self._create_mock_tool_executor()

        # Create subtask orchestrator that raises an exception (simulating tool failure)
        mock_subtask_orchestrator = self._create_mock_subtask_orchestrator(
            raise_exception=RuntimeError("Tool failed")
        )

        orchestrator = self._create_orchestrator(mock_executor, mock_subtask_orchestrator)

        subtask = SubTask(
            id=1,
            description="Failing task",
            required_tools=["failing_tool"],
            dependencies=[],
        )

        context = RunContext(deps=None, messages=[])
        success = await orchestrator._execute_subtask(subtask, context, total_subtasks=1)

        # Exception in orchestrator causes failure
        assert success is False
        assert subtask.status == "failed"
        assert "Tool failed" in subtask.error

    @pytest.mark.asyncio
    async def test_subtask_execution_exception_returns_false(self):
        """Test that exceptions during execution return False."""
        mock_executor = self._create_mock_tool_executor()

        # Create subtask orchestrator that raises exception
        mock_subtask_orchestrator = self._create_mock_subtask_orchestrator(
            raise_exception=RuntimeError("Network error")
        )

        orchestrator = self._create_orchestrator(mock_executor, mock_subtask_orchestrator)

        subtask = SubTask(
            id=1,
            description="Failing task",
            required_tools=["failing_tool"],
            dependencies=[],
        )

        context = RunContext(deps=None, messages=[])
        success = await orchestrator._execute_subtask(subtask, context, total_subtasks=1)

        # Exceptions cause the execution to return False
        assert success is False
        assert subtask.status == "failed"
        assert "Network error" in subtask.error

    @pytest.mark.asyncio
    async def test_plan_execution_respects_dependencies(self):
        """Test that plan execution respects subtask dependencies."""
        mock_executor = self._create_mock_tool_executor()

        # Create subtask orchestrator that returns success
        mock_subtask_orchestrator = self._create_mock_subtask_orchestrator(result_answer="Result")

        # Mock stream for synthesis
        async def mock_stream(*args, **kwargs):
            from miiflow_agent.core.client import StreamChunk
            yield StreamChunk(content="Done", delta="Done", finish_reason="stop", usage=None, tool_calls=None)

        mock_executor.stream_without_tools = mock_stream

        orchestrator = self._create_orchestrator(mock_executor, mock_subtask_orchestrator)

        # Create plan with dependencies
        plan = Plan(
            goal="Test",
            reasoning="Test",
            subtasks=[
                SubTask(id=1, description="First", required_tools=["tool1"], dependencies=[]),
                SubTask(id=2, description="Second", required_tools=["tool2"], dependencies=[1]),
                SubTask(id=3, description="Third", required_tools=["tool3"], dependencies=[1, 2]),
            ],
        )

        context = RunContext(deps=None, messages=[])
        success = await orchestrator._execute_plan(plan, context)

        assert success is True
        # All subtasks should be completed
        for subtask in plan.subtasks:
            assert subtask.status == "completed"


class TestPlanExecuteEventBusIntegration:
    """Test event bus integration for Plan & Execute."""

    @pytest.mark.asyncio
    async def test_planning_events_emitted(self):
        """Test that planning events are emitted."""
        events_received = []

        def handler(event):
            events_received.append(event)

        event_bus = EventBus()
        event_bus.subscribe(handler)

        mock_executor = MagicMock(spec=AgentToolExecutor)
        mock_executor.build_tools_description = MagicMock(return_value="tools")

        # Mock stream_without_tools
        async def mock_stream(*args, **kwargs):
            from miiflow_agent.core.client import StreamChunk
            yield StreamChunk(content="Answer", delta="Answer", finish_reason="stop", usage=None, tool_calls=None)

        mock_executor.stream_without_tools = mock_stream

        orchestrator = PlanAndExecuteOrchestrator(
            tool_executor=mock_executor,
            event_bus=event_bus,
            safety_manager=SafetyManager(max_steps=10),
        )

        # Execute with empty plan (triggers direct response path)
        empty_plan = Plan(goal="Hello", reasoning="Simple", subtasks=[])
        context = RunContext(deps=None, messages=[])

        await orchestrator.execute("Hello", context, existing_plan=empty_plan)

        # Check events were emitted
        event_types = [e.event_type for e in events_received]
        assert PlanExecuteEventType.PLANNING_START in event_types
        assert PlanExecuteEventType.PLANNING_COMPLETE in event_types
        assert PlanExecuteEventType.FINAL_ANSWER in event_types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
