"""Simplified event bus for ReAct events - eliminates duplicate emit methods."""

import logging
from typing import List, Optional, Callable, Any
from dataclasses import dataclass

from ..data import ReActEvent, ReActEventType, ReActStep

logger = logging.getLogger(__name__)


class EventBus:
    """Clean event bus with single publish method - no duplicate emit_* methods."""
    
    def __init__(self, buffer_size: int = 100):
        self.buffer_size = buffer_size
        self.event_buffer: List[ReActEvent] = []
        self.subscribers: List[Callable[[ReActEvent], None]] = []
        
    def subscribe(self, callback: Callable[[ReActEvent], None]):
        """Subscribe to events with a callback."""
        self.subscribers.append(callback)
    
    def unsubscribe(self, callback):
        """Remove a subscriber."""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    async def publish(self, event: ReActEvent):
        """Single publish method - replaces 9 duplicate emit_* methods."""
        # Add to buffer
        self.event_buffer.append(event)
        if len(self.event_buffer) > self.buffer_size:
            self.event_buffer.pop(0)
        
        # Notify all subscribers
        for callback in self.subscribers:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in event subscriber: {e}")
    
    def get_events(self, **filters) -> List[ReActEvent]:
        """Get filtered events from buffer using simple kwargs."""
        if not filters:
            return self.event_buffer.copy()
        
        result = []
        for event in self.event_buffer:
            if all(
                getattr(event, key) == value or 
                (isinstance(value, list) and getattr(event, key) in value)
                for key, value in filters.items()
                if hasattr(event, key)
            ):
                result.append(event)
        
        return result
    
    def clear_buffer(self):
        """Clear the event buffer."""
        self.event_buffer.clear()


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
    def action_planned(step_number: int, action: str, action_input: dict) -> ReActEvent:
        """Create action planned event."""
        return ReActEvent(
            event_type=ReActEventType.ACTION_PLANNED,
            step_number=step_number,
            data={"action": action, "action_input": action_input}
        )
    
    @staticmethod
    def action_executing(step_number: int, action: str, action_input: dict) -> ReActEvent:
        """Create action executing event."""
        return ReActEvent(
            event_type=ReActEventType.ACTION_EXECUTING,
            step_number=step_number,
            data={"action": action, "action_input": action_input, "status": "executing"}
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
