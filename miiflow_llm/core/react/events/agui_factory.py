"""AG-UI Event Factory - creates AG-UI protocol events.

This module provides a factory for creating AG-UI (Agent-User Interaction Protocol)
events, enabling native AG-UI support in miiflow-llm.

Requires the 'agui' extra: pip install miiflow-llm[agui]
"""

from typing import Optional, Dict, Any, TYPE_CHECKING
import uuid
import json
import logging

logger = logging.getLogger(__name__)

# Lazy import AG-UI types to make it optional
try:
    from ag_ui.core import (
        EventType,
        BaseEvent,
        RunStartedEvent,
        RunFinishedEvent,
        RunErrorEvent,
        StepStartedEvent,
        StepFinishedEvent,
        TextMessageStartEvent,
        TextMessageContentEvent,
        TextMessageEndEvent,
        ToolCallStartEvent,
        ToolCallArgsEvent,
        ToolCallEndEvent,
        ToolCallResultEvent,
        CustomEvent,
    )
    AGUI_AVAILABLE = True
except ImportError:
    AGUI_AVAILABLE = False
    # Define placeholder types for type checking
    if TYPE_CHECKING:
        from ag_ui.core import (
            EventType,
            BaseEvent,
            RunStartedEvent,
            RunFinishedEvent,
            RunErrorEvent,
            StepStartedEvent,
            StepFinishedEvent,
            TextMessageStartEvent,
            TextMessageContentEvent,
            TextMessageEndEvent,
            ToolCallStartEvent,
            ToolCallArgsEvent,
            ToolCallEndEvent,
            ToolCallResultEvent,
            CustomEvent,
        )


def check_agui_available():
    """Check if AG-UI protocol is available."""
    if not AGUI_AVAILABLE:
        raise ImportError(
            "AG-UI protocol not installed. Install with: pip install miiflow-llm[agui]"
        )


class AGUIEventFactory:
    """Factory for creating AG-UI protocol events.

    This factory creates standardized AG-UI events for agent-to-frontend communication,
    enabling interoperability with the AG-UI ecosystem (LangGraph, CrewAI, CopilotKit, etc.).

    Usage:
        factory = AGUIEventFactory(thread_id="thread_123", message_id="msg_456")

        # Lifecycle events
        yield factory.run_started()

        # Message events
        yield factory.text_message_start()
        yield factory.text_message_content("Hello, ")
        yield factory.text_message_content("world!")
        yield factory.text_message_end()

        # Tool events
        yield factory.tool_call_start("search", "Search the web")
        yield factory.tool_call_args("search", '{"query": "weather"}')
        yield factory.tool_call_end("search")
        yield factory.tool_call_result("search", "Sunny, 72°F")

        yield factory.run_finished()
    """

    def __init__(self, thread_id: str, message_id: str):
        """Initialize the AG-UI event factory.

        Args:
            thread_id: The thread/conversation ID
            message_id: The current message ID being generated
        """
        check_agui_available()

        self.thread_id = thread_id
        self.run_id = str(uuid.uuid4())
        self.message_id = message_id
        self._tool_call_ids: Dict[str, str] = {}  # action -> tool_call_id
        self._message_started = False

    # ============ Lifecycle Events ============

    def run_started(self) -> "RunStartedEvent":
        """Create a RunStartedEvent to signal the start of agent execution.

        Returns:
            RunStartedEvent with thread_id and run_id
        """
        return RunStartedEvent(
            type=EventType.RUN_STARTED,
            thread_id=self.thread_id,
            run_id=self.run_id
        )

    def run_finished(self, result: Optional[Any] = None) -> "RunFinishedEvent":
        """Create a RunFinishedEvent to signal successful completion.

        Args:
            result: Optional result data from the agent run

        Returns:
            RunFinishedEvent with thread_id and run_id
        """
        return RunFinishedEvent(
            type=EventType.RUN_FINISHED,
            thread_id=self.thread_id,
            run_id=self.run_id,
            result=result
        )

    def run_error(self, message: str, code: Optional[str] = None) -> "RunErrorEvent":
        """Create a RunErrorEvent to signal a failure.

        Args:
            message: Error message describing what went wrong
            code: Optional error code for programmatic handling

        Returns:
            RunErrorEvent with error details
        """
        return RunErrorEvent(
            type=EventType.RUN_ERROR,
            message=message,
            code=code
        )

    # ============ Step Events ============

    def step_started(self, step_name: str) -> "StepStartedEvent":
        """Create a StepStartedEvent for a discrete phase of work.

        Use this for planning phases, subtasks, or other named steps.

        Args:
            step_name: Name of the step (e.g., "planning", "subtask_1")

        Returns:
            StepStartedEvent with step name
        """
        return StepStartedEvent(
            type=EventType.STEP_STARTED,
            step_name=step_name
        )

    def step_finished(self, step_name: str) -> "StepFinishedEvent":
        """Create a StepFinishedEvent to complete a step.

        Args:
            step_name: Name of the step that finished

        Returns:
            StepFinishedEvent with step name
        """
        return StepFinishedEvent(
            type=EventType.STEP_FINISHED,
            step_name=step_name
        )

    # ============ Message Events ============

    def text_message_start(self, role: str = "assistant") -> "TextMessageStartEvent":
        """Create a TextMessageStartEvent to begin a message.

        Args:
            role: The role of the message sender (default: "assistant")

        Returns:
            TextMessageStartEvent with message_id and role
        """
        self._message_started = True
        return TextMessageStartEvent(
            type=EventType.TEXT_MESSAGE_START,
            message_id=self.message_id,
            role=role
        )

    def text_message_content(self, delta: str) -> "TextMessageContentEvent":
        """Create a TextMessageContentEvent with streaming text.

        Args:
            delta: The incremental text chunk to stream

        Returns:
            TextMessageContentEvent with the delta
        """
        return TextMessageContentEvent(
            type=EventType.TEXT_MESSAGE_CONTENT,
            message_id=self.message_id,
            delta=delta
        )

    def text_message_end(self) -> "TextMessageEndEvent":
        """Create a TextMessageEndEvent to complete a message.

        Returns:
            TextMessageEndEvent with message_id
        """
        self._message_started = False
        return TextMessageEndEvent(
            type=EventType.TEXT_MESSAGE_END,
            message_id=self.message_id
        )

    @property
    def message_started(self) -> bool:
        """Check if a message has been started but not ended."""
        return self._message_started

    # ============ Tool Call Events ============

    def tool_call_start(
        self,
        action: str,
        tool_description: Optional[str] = None
    ) -> "ToolCallStartEvent":
        """Create a ToolCallStartEvent to begin a tool invocation.

        Args:
            action: The name of the tool being called
            tool_description: Optional description of what the tool does

        Returns:
            ToolCallStartEvent with tool_call_id and tool_call_name
        """
        tool_call_id = str(uuid.uuid4())
        self._tool_call_ids[action] = tool_call_id
        return ToolCallStartEvent(
            type=EventType.TOOL_CALL_START,
            tool_call_id=tool_call_id,
            tool_call_name=action,
            parent_message_id=self.message_id
        )

    def tool_call_args(self, action: str, args_delta: str) -> "ToolCallArgsEvent":
        """Create a ToolCallArgsEvent to stream tool arguments.

        Args:
            action: The name of the tool
            args_delta: Incremental JSON argument string

        Returns:
            ToolCallArgsEvent with the delta
        """
        tool_call_id = self._tool_call_ids.get(action, str(uuid.uuid4()))
        return ToolCallArgsEvent(
            type=EventType.TOOL_CALL_ARGS,
            tool_call_id=tool_call_id,
            delta=args_delta
        )

    def tool_call_args_from_dict(self, action: str, args: dict) -> "ToolCallArgsEvent":
        """Create a ToolCallArgsEvent from a dictionary of arguments.

        Args:
            action: The name of the tool
            args: Dictionary of tool arguments

        Returns:
            ToolCallArgsEvent with JSON-serialized args
        """
        return self.tool_call_args(action, json.dumps(args))

    def tool_call_end(self, action: str) -> "ToolCallEndEvent":
        """Create a ToolCallEndEvent to complete tool specification.

        Args:
            action: The name of the tool

        Returns:
            ToolCallEndEvent with tool_call_id
        """
        tool_call_id = self._tool_call_ids.get(action, str(uuid.uuid4()))
        return ToolCallEndEvent(
            type=EventType.TOOL_CALL_END,
            tool_call_id=tool_call_id
        )

    def tool_call_result(
        self,
        action: str,
        result: str,
        success: bool = True
    ) -> "ToolCallResultEvent":
        """Create a ToolCallResultEvent with tool execution output.

        Args:
            action: The name of the tool
            result: The result content from tool execution
            success: Whether the tool execution was successful

        Returns:
            ToolCallResultEvent with the result
        """
        tool_call_id = self._tool_call_ids.pop(action, str(uuid.uuid4()))
        return ToolCallResultEvent(
            type=EventType.TOOL_CALL_RESULT,
            tool_call_id=tool_call_id,
            content=result
        )

    # ============ Custom Events ============

    def custom(self, name: str, value: Any) -> "CustomEvent":
        """Create a CustomEvent for application-specific data.

        Args:
            name: The name of the custom event
            value: The value/data for the event

        Returns:
            CustomEvent with name and value
        """
        return CustomEvent(
            type=EventType.CUSTOM,
            name=name,
            value=value
        )

    def thinking_chunk(self, delta: str, content: str = "") -> "CustomEvent":
        """Create a custom event for streaming thinking/reasoning.

        This is used to stream the agent's internal reasoning process.

        Args:
            delta: The incremental thinking text
            content: The accumulated thinking content so far

        Returns:
            CustomEvent with name="thinking"
        """
        return self.custom("thinking", {"delta": delta, "content": content})

    def progress(
        self,
        completed: int,
        total: int,
        percentage: float,
        wave_number: Optional[int] = None
    ) -> "CustomEvent":
        """Create a custom event for progress updates.

        Args:
            completed: Number of completed items
            total: Total number of items
            percentage: Completion percentage (0-100)
            wave_number: Optional wave/batch number for parallel execution

        Returns:
            CustomEvent with name="progress"
        """
        data = {
            "completed": completed,
            "total": total,
            "percentage": percentage
        }
        if wave_number is not None:
            data["wave_number"] = wave_number
        return self.custom("progress", data)

    def suggested_actions(self, actions: list) -> "CustomEvent":
        """Create a custom event for suggested follow-up actions.

        Args:
            actions: List of suggested action strings or objects

        Returns:
            CustomEvent with name="suggested_actions"
        """
        return self.custom("suggested_actions", actions)

    def planning_status(self, status: str, plan_data: Optional[dict] = None) -> "CustomEvent":
        """Create a custom event for planning phase status.

        Args:
            status: Planning status ("started", "complete", "replanning")
            plan_data: Optional plan data including subtasks

        Returns:
            CustomEvent with name="planning"
        """
        data = {"status": status}
        if plan_data:
            data["plan"] = plan_data
        return self.custom("planning", data)
