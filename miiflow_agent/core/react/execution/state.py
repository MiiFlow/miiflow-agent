"""Execution state management for ReAct orchestration.

This module provides a clean container for execution state, making it easy
to track progress and manage execution lifecycle.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExecutionState:
    """Encapsulates execution state for ReAct orchestration.

    This provides a clean container for all execution state, making it easy
    to track progress, reset between sessions, and reduce instance variable clutter.

    Attributes:
        current_step: Current step number (0-indexed, incremented before each step)
        steps: List of completed ReActStep objects
        start_time: Timestamp when execution started
        is_running: Whether execution is still in progress
        final_answer: The final answer when execution completes
        ready_for_answer: Flag indicating answer streaming has started
        accumulated_answer: Buffer for streaming answer chunks
    """

    current_step: int = 0
    steps: List[Any] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    is_running: bool = True
    final_answer: Optional[str] = None
    ready_for_answer: bool = False
    accumulated_answer: str = ""

    def increment_step(self) -> int:
        """Increment and return the new step number."""
        self.current_step += 1
        return self.current_step

    def set_final_answer(self, answer: str) -> None:
        """Set the final answer and stop execution."""
        self.final_answer = answer
        self.is_running = False

    def add_step(self, step: Any) -> None:
        """Add a completed step to the execution history."""
        self.steps.append(step)

    def reset_answer_buffer(self) -> None:
        """Reset the answer accumulation state."""
        self.ready_for_answer = False
        self.accumulated_answer = ""

    @property
    def total_cost(self) -> float:
        """Calculate total cost across all steps."""
        return sum(getattr(step, "cost", 0.0) for step in self.steps)

    @property
    def total_tokens(self) -> int:
        """Calculate total tokens used across all steps."""
        return sum(getattr(step, "tokens_used", 0) for step in self.steps)

    @property
    def total_execution_time(self) -> float:
        """Calculate total execution time from start."""
        return time.time() - self.start_time

    @property
    def step_count(self) -> int:
        """Get the number of completed steps."""
        return len(self.steps)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize execution state to dictionary."""
        return {
            "current_step": self.current_step,
            "step_count": self.step_count,
            "is_running": self.is_running,
            "has_final_answer": self.final_answer is not None,
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "total_execution_time": self.total_execution_time,
        }
