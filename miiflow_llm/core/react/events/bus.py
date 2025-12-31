"""Simplified event bus for ReAct events - eliminates duplicate emit methods.

Supports both legacy ReAct events and AG-UI protocol events.
"""

import asyncio
import logging
from typing import List, Optional, Callable, Any, Union, Literal, TYPE_CHECKING

from ..enums import ReActEventType, PlanExecuteEventType
from ..models import ReActStep
from ..react_events import ReActEvent, PlanExecuteEvent

logger = logging.getLogger(__name__)

# Type alias for event format
EventFormat = Literal["react", "agui"]

# Optional AG-UI imports
try:
    from .agui_factory import AGUIEventFactory, AGUI_AVAILABLE
    if TYPE_CHECKING:
        from ag_ui.core import BaseEvent as AGUIBaseEvent
except ImportError:
    AGUIEventFactory = None
    AGUI_AVAILABLE = False
    if TYPE_CHECKING:
        AGUIBaseEvent = Any


class EventBus:
    """Event bus supporting both ReAct and AG-UI event formats.

    This event bus can operate in two modes:
    - "react": Emits legacy ReActEvent objects (default)
    - "agui": Emits AG-UI protocol events for ecosystem interoperability

    Usage:
        # Legacy mode (default)
        bus = EventBus()
        await bus.publish(EventFactory.thinking_chunk(1, "Hello", "Hello"))

        # AG-UI mode
        bus = EventBus(
            event_format="agui",
            thread_id="thread_123",
            message_id="msg_456"
        )
        await bus.publish(bus.agui_factory.text_message_content("Hello"))
    """

    def __init__(
        self,
        buffer_size: int = 100,
        event_format: EventFormat = "react",
        thread_id: Optional[str] = None,
        message_id: Optional[str] = None,
    ):
        """Initialize the event bus.

        Args:
            buffer_size: Maximum number of events to buffer
            event_format: "react" for legacy events, "agui" for AG-UI protocol
            thread_id: Thread ID (required for agui format)
            message_id: Message ID (required for agui format)
        """
        self.buffer_size = buffer_size
        self.event_format = event_format
        self.event_buffer: List[Any] = []
        self.subscribers: List[Callable[[Any], None]] = []

        # AG-UI factory (only created if using agui format)
        self._agui_factory: Optional["AGUIEventFactory"] = None
        if event_format == "agui":
            if not AGUI_AVAILABLE:
                raise ImportError(
                    "AG-UI protocol not installed. Install with: pip install miiflow-llm[agui]"
                )
            if not thread_id or not message_id:
                raise ValueError(
                    "thread_id and message_id are required for AG-UI event format"
                )
            self._agui_factory = AGUIEventFactory(thread_id, message_id)

    @property
    def agui_factory(self) -> "AGUIEventFactory":
        """Get the AG-UI event factory.

        Raises:
            ValueError: If AG-UI format is not enabled

        Returns:
            The AGUIEventFactory instance
        """
        if not self._agui_factory:
            raise ValueError(
                "AG-UI factory not initialized. "
                "Set event_format='agui' with thread_id and message_id."
            )
        return self._agui_factory

    @property
    def is_agui_mode(self) -> bool:
        """Check if the event bus is in AG-UI mode."""
        return self.event_format == "agui"

    def subscribe(self, callback: Callable[[Any], None]):
        """Subscribe to events with a callback.

        Args:
            callback: Function to call when an event is published
        """
        self.subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[Any], None]):
        """Remove a subscriber.

        Args:
            callback: The callback to remove
        """
        if callback in self.subscribers:
            self.subscribers.remove(callback)

    async def publish(self, event: Union[ReActEvent, PlanExecuteEvent, "AGUIBaseEvent", Any]):
        """Publish an event to all subscribers.

        This method works with ReActEvent, PlanExecuteEvent, and AG-UI BaseEvent types.
        When in AG-UI mode, legacy events are automatically converted.

        Args:
            event: The event to publish
        """
        # Auto-convert legacy events to AG-UI if in AG-UI mode
        if self.is_agui_mode:
            if isinstance(event, ReActEvent):
                agui_event = self._convert_react_to_agui(event)
                if agui_event is None:
                    return  # Skip events that don't have AG-UI equivalents
                event = agui_event
            elif isinstance(event, PlanExecuteEvent):
                agui_event = self._convert_plan_execute_to_agui(event)
                if agui_event is None:
                    return  # Skip events that don't have AG-UI equivalents
                event = agui_event

        # Add to buffer
        self.event_buffer.append(event)
        if len(self.event_buffer) > self.buffer_size:
            self.event_buffer.pop(0)

        # Notify all subscribers
        for callback in self.subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Error in event subscriber: {e}")

    def get_events(self, **filters) -> List[Any]:
        """Get filtered events from buffer using simple kwargs.

        Args:
            **filters: Attribute filters to apply

        Returns:
            List of matching events
        """
        if not filters:
            return self.event_buffer.copy()

        result = []
        for event in self.event_buffer:
            if all(
                getattr(event, key, None) == value
                or (isinstance(value, list) and getattr(event, key, None) in value)
                for key, value in filters.items()
            ):
                result.append(event)

        return result

    def clear_buffer(self):
        """Clear the event buffer."""
        self.event_buffer.clear()

    def _convert_react_to_agui(self, react_event: ReActEvent) -> Optional[Any]:
        """Convert a ReActEvent to an AG-UI event.

        This method is called automatically when the EventBus is in AG-UI mode.
        It maps ReAct event types to their AG-UI equivalents.

        Args:
            react_event: The ReActEvent to convert

        Returns:
            An AG-UI event, or None if the event should be skipped
        """
        if not self._agui_factory:
            return None

        from ..enums import ReActEventType
        import json

        event_type = react_event.event_type
        data = react_event.data

        if event_type == ReActEventType.STEP_START:
            return self._agui_factory.step_started(f"step_{react_event.step_number}")

        elif event_type == ReActEventType.THINKING_CHUNK:
            return self._agui_factory.thinking_chunk(
                data.get("delta", ""),
                data.get("content", "")
            )

        elif event_type == ReActEventType.THOUGHT:
            # Complete thought - emit as thinking custom event
            return self._agui_factory.custom("thought_complete", {
                "thought": data.get("thought", "")
            })

        elif event_type == ReActEventType.ACTION_PLANNED:
            action = data.get("action", "unknown")
            tool_description = data.get("tool_description")
            return self._agui_factory.tool_call_start(action, tool_description)

        elif event_type == ReActEventType.ACTION_EXECUTING:
            action = data.get("action", "unknown")
            args = data.get("action_input", {})
            # Emit tool_call_args with the arguments
            args_json = json.dumps(args) if isinstance(args, dict) else str(args)
            return self._agui_factory.tool_call_args(action, args_json)

        elif event_type == ReActEventType.OBSERVATION:
            action = data.get("action", "unknown")
            observation = data.get("observation", "")
            success = data.get("success", True)
            return self._agui_factory.tool_call_result(action, observation, success)

        elif event_type == ReActEventType.FINAL_ANSWER_CHUNK:
            delta = data.get("delta", "")
            # Start message if not already started
            if not self._agui_factory.message_started:
                # We need to emit message start first - caller will need to handle this
                # For now, just emit the content chunk
                pass
            return self._agui_factory.text_message_content(delta)

        elif event_type == ReActEventType.FINAL_ANSWER:
            # Final answer - emit message end
            return self._agui_factory.text_message_end()

        elif event_type == ReActEventType.ERROR:
            error_msg = data.get("error", "Unknown error")
            return self._agui_factory.run_error(error_msg)

        elif event_type == ReActEventType.STEP_COMPLETE:
            # Step complete with optional suggested actions
            suggested_actions = data.get("suggested_actions")
            if suggested_actions:
                return self._agui_factory.suggested_actions(suggested_actions)
            # Also emit step_finished
            return self._agui_factory.step_finished(f"step_{react_event.step_number}")

        elif event_type == ReActEventType.STOP_CONDITION:
            # Stop condition - emit as custom event
            return self._agui_factory.custom("stop_condition", {
                "stop_reason": data.get("stop_reason", ""),
                "description": data.get("description", "")
            })

        return None

    def _convert_plan_execute_to_agui(self, plan_event: PlanExecuteEvent) -> Optional[Any]:
        """Convert a PlanExecuteEvent to an AG-UI event.

        This method is called automatically when the EventBus is in AG-UI mode.
        It maps Plan & Execute event types to their AG-UI equivalents.

        Args:
            plan_event: The PlanExecuteEvent to convert

        Returns:
            An AG-UI event, or None if the event should be skipped
        """
        if not self._agui_factory:
            return None

        import json

        event_type = plan_event.event_type
        data = plan_event.data

        if event_type == PlanExecuteEventType.PLANNING_START:
            return self._agui_factory.step_started("planning")

        elif event_type == PlanExecuteEventType.PLANNING_THINKING_CHUNK:
            return self._agui_factory.thinking_chunk(
                data.get("delta", ""),
                data.get("accumulated", "")
            )

        elif event_type == PlanExecuteEventType.PLANNING_COMPLETE:
            # Emit planning status and step finished
            return self._agui_factory.planning_status("complete", data.get("plan"))

        elif event_type == PlanExecuteEventType.REPLANNING_START:
            return self._agui_factory.step_started("replanning")

        elif event_type == PlanExecuteEventType.REPLANNING_THINKING_CHUNK:
            return self._agui_factory.thinking_chunk(
                data.get("delta", ""),
                data.get("content", "")
            )

        elif event_type == PlanExecuteEventType.REPLANNING_COMPLETE:
            return self._agui_factory.step_finished("replanning")

        elif event_type == PlanExecuteEventType.SUBTASK_START:
            subtask = data.get("subtask", {})
            subtask_id = subtask.get("id", "unknown")
            return self._agui_factory.step_started(f"subtask_{subtask_id}")

        elif event_type == PlanExecuteEventType.SUBTASK_THINKING_CHUNK:
            # Check if this is a tool event
            if data.get("is_tool"):
                tool_name = data.get("tool_name", "unknown")
                if data.get("is_tool_planned"):
                    return self._agui_factory.tool_call_start(
                        tool_name,
                        data.get("tool_description")
                    )
                elif data.get("is_tool_executing"):
                    args = data.get("tool_args", {})
                    args_json = json.dumps(args) if isinstance(args, dict) else str(args)
                    return self._agui_factory.tool_call_args(tool_name, args_json)
            elif data.get("is_observation"):
                tool_name = data.get("tool_name", "unknown")
                observation = data.get("delta", "")
                success = data.get("success", True)
                return self._agui_factory.tool_call_result(tool_name, observation, success)
            else:
                # Regular thinking chunk
                return self._agui_factory.thinking_chunk(
                    data.get("delta", ""),
                    data.get("thought", "")
                )

        elif event_type == PlanExecuteEventType.SUBTASK_COMPLETE:
            subtask = data.get("subtask", {})
            subtask_id = subtask.get("id", "unknown")
            return self._agui_factory.step_finished(f"subtask_{subtask_id}")

        elif event_type == PlanExecuteEventType.SUBTASK_FAILED:
            error = data.get("error", "Unknown error")
            return self._agui_factory.custom("subtask_failed", {
                "subtask": data.get("subtask"),
                "error": error
            })

        elif event_type == PlanExecuteEventType.PLAN_PROGRESS:
            return self._agui_factory.progress(
                data.get("completed", 0),
                data.get("total", 0),
                data.get("progress_percentage", 0)
            )

        elif event_type == PlanExecuteEventType.SYNTHESIS_START:
            return self._agui_factory.step_started("synthesis")

        elif event_type == PlanExecuteEventType.FINAL_ANSWER_CHUNK:
            delta = data.get("delta", "")
            # Start message if not already started
            if not self._agui_factory.message_started:
                pass  # Caller needs to handle message start
            return self._agui_factory.text_message_content(delta)

        elif event_type == PlanExecuteEventType.FINAL_ANSWER:
            return self._agui_factory.text_message_end()

        elif event_type == PlanExecuteEventType.ERROR:
            error_msg = data.get("error", "Unknown error")
            return self._agui_factory.run_error(error_msg)

        return None

    async def publish_react_event(self, react_event: ReActEvent):
        """Publish a ReActEvent, converting to AG-UI format if in AG-UI mode.

        This is the preferred method for orchestrators to use. It automatically
        handles format conversion based on the EventBus mode.

        Args:
            react_event: The ReActEvent to publish
        """
        if self.is_agui_mode:
            # Convert to AG-UI format
            agui_event = self._convert_react_to_agui(react_event)
            if agui_event:
                await self.publish(agui_event)
        else:
            # Publish as-is
            await self.publish(react_event)


class EventFactory:
    """Factory for creating ReAct events - replaces duplicate emit_* methods."""
    
    @staticmethod
    def step_started(step_number: int) -> ReActEvent:
        """Create step start event."""
        return ReActEvent(
            event_type=ReActEventType.STEP_START,
            step_number=step_number,
            data={"step_number": step_number}
        )
    
    @staticmethod
    def thought(step_number: int, thought: str) -> ReActEvent:
        """Create thought event."""
        return ReActEvent(
            event_type=ReActEventType.THOUGHT,
            step_number=step_number,
            data={"thought": thought}
        )

    @staticmethod
    def thinking_chunk(step_number: int, delta: str, content: str) -> ReActEvent:
        """Create thinking chunk event for real-time streaming."""
        return ReActEvent(
            event_type=ReActEventType.THINKING_CHUNK,
            step_number=step_number,
            data={"delta": delta, "content": content}
        )

    @staticmethod
    def action_planned(step_number: int, action: str, action_input: dict, tool_description: str = None) -> ReActEvent:
        """Create action planned event."""
        return ReActEvent(
            event_type=ReActEventType.ACTION_PLANNED,
            step_number=step_number,
            data={
                "action": action,
                "action_input": action_input,
                "tool_description": tool_description,
            }
        )

    @staticmethod
    def action_executing(step_number: int, action: str, action_input: dict, tool_description: str = None) -> ReActEvent:
        """Create action executing event."""
        return ReActEvent(
            event_type=ReActEventType.ACTION_EXECUTING,
            step_number=step_number,
            data={
                "action": action,
                "action_input": action_input,
                "status": "executing",
                "tool_description": tool_description,
            }
        )
    
    @staticmethod
    def observation(step_number: int, observation: str, action: str, success: bool = True) -> ReActEvent:
        """Create observation event."""
        return ReActEvent(
            event_type=ReActEventType.OBSERVATION,
            step_number=step_number,
            data={"observation": observation, "action": action, "success": success}
        )
    
    @staticmethod
    def step_complete(step_number: int, step: ReActStep) -> ReActEvent:
        """Create step complete event."""
        return ReActEvent(
            event_type=ReActEventType.STEP_COMPLETE,
            step_number=step_number,
            data={"step": step.to_dict(), "execution_time": step.execution_time, "cost": step.cost}
        )
    
    @staticmethod
    def final_answer(step_number: int, answer: str) -> ReActEvent:
        """Create final answer event."""
        return ReActEvent(
            event_type=ReActEventType.FINAL_ANSWER,
            step_number=step_number,
            data={"answer": answer}
        )

    @staticmethod
    def final_answer_chunk(step_number: int, delta: str, content: str) -> ReActEvent:
        """Create final answer chunk event for real-time streaming."""
        return ReActEvent(
            event_type=ReActEventType.FINAL_ANSWER_CHUNK,
            step_number=step_number,
            data={"delta": delta, "content": content}
        )

    @staticmethod
    def error(step_number: int, error: str, error_type: str = "unknown") -> ReActEvent:
        """Create error event."""
        return ReActEvent(
            event_type=ReActEventType.ERROR,
            step_number=step_number,
            data={"error": error, "error_type": error_type}
        )
    
    @staticmethod
    def stop_condition(step_number: int, stop_reason: str, description: str) -> ReActEvent:
        """Create stop condition event."""
        return ReActEvent(
            event_type=ReActEventType.STOP_CONDITION,
            step_number=step_number,
            data={"stop_reason": stop_reason, "description": description}
        )
