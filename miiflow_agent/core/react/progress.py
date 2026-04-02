"""Progress tracking for orchestration execution.

Provides structured progress snapshots that aggregate metrics across
orchestration steps, inspired by Claude Code's ProgressTracker pattern.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ToolActivity:
    """Record of a single tool execution."""

    tool_name: str
    description: str  # Human-readable: "Searching for Tesla news"
    timestamp: float
    success: bool
    execution_time: float


@dataclass
class ProgressSnapshot:
    """Point-in-time progress state."""

    current_step: int
    total_steps_limit: int
    tool_calls_count: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost: float
    elapsed_time: float
    recent_activities: List[ToolActivity]

    @property
    def progress_percentage(self) -> float:
        """Estimated progress based on steps used vs limit."""
        if self.total_steps_limit <= 0:
            return 0.0
        return min(100.0, (self.current_step / self.total_steps_limit) * 100)

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "current_step": self.current_step,
            "total_steps_limit": self.total_steps_limit,
            "tool_calls_count": self.tool_calls_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost": self.total_cost,
            "elapsed_time": round(self.elapsed_time, 2),
            "progress_percentage": round(self.progress_percentage, 1),
            "recent_activities": [
                {
                    "tool_name": a.tool_name,
                    "description": a.description,
                    "success": a.success,
                    "execution_time": round(a.execution_time, 3),
                }
                for a in self.recent_activities
            ],
        }


class ProgressTracker:
    """Accumulates progress metrics across orchestration steps.

    Tracks tool calls, token usage, cost, and maintains a sliding window
    of recent activities for UI display.
    """

    def __init__(self, max_steps: int = 25, max_activities: int = 5):
        """Initialize progress tracker.

        Args:
            max_steps: Maximum allowed steps (for progress percentage).
            max_activities: Number of recent activities to retain.
        """
        self.max_steps = max_steps
        self.max_activities = max_activities

        self._current_step: int = 0
        self._tool_calls_count: int = 0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_cost: float = 0.0
        self._start_time: float = time.time()
        self._recent_activities: deque = deque(maxlen=max_activities)

    def record_step(self, step) -> None:
        """Record a completed ReActStep.

        Args:
            step: A ReActStep instance with metrics.
        """
        self._current_step = step.step_number
        self._total_cost += step.cost
        self._total_input_tokens += step.tokens_used  # Approximate
        if step.is_action_step:
            self._tool_calls_count += 1

    def record_tool_call(
        self,
        tool_name: str,
        description: str,
        success: bool,
        execution_time: float,
    ) -> None:
        """Record a tool execution.

        Args:
            tool_name: Name of the tool executed.
            description: Human-readable description of the action.
            success: Whether the tool succeeded.
            execution_time: Time taken in seconds.
        """
        self._tool_calls_count += 1
        self._recent_activities.append(
            ToolActivity(
                tool_name=tool_name,
                description=description,
                timestamp=time.time(),
                success=success,
                execution_time=execution_time,
            )
        )

    def record_tokens(
        self, input_tokens: int = 0, output_tokens: int = 0, cost: float = 0.0
    ) -> None:
        """Record token usage from an LLM call.

        Args:
            input_tokens: Number of input tokens used.
            output_tokens: Number of output tokens used.
            cost: Cost in USD.
        """
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self._total_cost += cost

    def snapshot(self) -> ProgressSnapshot:
        """Get a point-in-time progress snapshot."""
        return ProgressSnapshot(
            current_step=self._current_step,
            total_steps_limit=self.max_steps,
            tool_calls_count=self._tool_calls_count,
            total_input_tokens=self._total_input_tokens,
            total_output_tokens=self._total_output_tokens,
            total_cost=self._total_cost,
            elapsed_time=time.time() - self._start_time,
            recent_activities=list(self._recent_activities),
        )
