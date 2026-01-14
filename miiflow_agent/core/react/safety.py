"""Safety mechanisms for ReAct loops."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from .enums import StopReason
from .models import ReActStep


def make_hashable(obj: Any) -> Any:
    """
    Recursively convert an object to a hashable type.

    This is needed for creating signatures from tool inputs that may contain
    lists, dicts, or other unhashable types.

    Args:
        obj: Object to convert

    Returns:
        Hashable version of the object
    """
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, dict):
        # Convert dict to frozenset of (key, hashable_value) tuples
        return frozenset((k, make_hashable(v)) for k, v in obj.items())
    elif isinstance(obj, (list, tuple)):
        # Convert list/tuple to tuple of hashable elements
        return tuple(make_hashable(item) for item in obj)
    elif isinstance(obj, set):
        # Convert set to frozenset of hashable elements
        return frozenset(make_hashable(item) for item in obj)
    elif isinstance(obj, frozenset):
        # Already hashable, but items might not be
        return frozenset(make_hashable(item) for item in obj)
    else:
        # For other types, try to convert to string as fallback
        return str(obj)


@dataclass
class StopCondition(ABC):
    """Abstract base class for stop conditions."""

    @abstractmethod
    def should_stop(self, steps: List[ReActStep], current_step: int) -> bool:
        """Check if the loop should stop."""
        pass

    @abstractmethod
    def get_stop_reason(self) -> StopReason:
        """Get the reason for stopping."""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Get human-readable description of this condition."""
        pass


@dataclass
class MaxStepsCondition(StopCondition):
    """Stop after maximum number of steps."""

    max_steps: int = 10

    def should_stop(self, steps: List[ReActStep], current_step: int) -> bool:
        return current_step >= self.max_steps

    def get_stop_reason(self) -> StopReason:
        return StopReason.MAX_STEPS

    def get_description(self) -> str:
        return f"Maximum {self.max_steps} steps reached"


@dataclass
class MaxBudgetCondition(StopCondition):
    """Stop when cost exceeds budget."""

    max_budget: float

    def should_stop(self, steps: List[ReActStep], current_step: int) -> bool:
        total_cost = sum(step.cost for step in steps)
        return total_cost >= self.max_budget

    def get_stop_reason(self) -> StopReason:
        return StopReason.MAX_BUDGET

    def get_description(self) -> str:
        return f"Budget limit of ${self.max_budget} exceeded"


@dataclass
class MaxTimeCondition(StopCondition):
    """Stop when execution time exceeds limit."""

    max_time_seconds: float
    start_time: float = None

    def __post_init__(self):
        if self.start_time is None:
            self.start_time = time.time()

    def should_stop(self, steps: List[ReActStep], current_step: int) -> bool:
        elapsed = time.time() - self.start_time
        return elapsed >= self.max_time_seconds

    def get_stop_reason(self) -> StopReason:
        return StopReason.MAX_TIME

    def get_description(self) -> str:
        return f"Time limit of {self.max_time_seconds} seconds exceeded"


@dataclass
class RepeatedActionsCondition(StopCondition):
    """Stop when agent repeatedly calls same action with same parameters."""

    max_repeats: int = 3
    lookback_steps: int = 5

    def should_stop(self, steps: List[ReActStep], current_step: int) -> bool:
        if len(steps) < self.max_repeats:
            return False

        # Look at recent steps for repeated patterns
        recent_steps = steps[-self.lookback_steps :] if self.lookback_steps else steps
        action_steps = [s for s in recent_steps if s.is_action_step]

        if len(action_steps) < self.max_repeats:
            return False

        # Check for identical actions in sequence
        last_actions = action_steps[-self.max_repeats :]

        if not last_actions:
            return False

        first_action = last_actions[0]
        for action_step in last_actions[1:]:
            if (
                action_step.action != first_action.action
                or action_step.action_input != first_action.action_input
            ):
                return False

        return True

    def get_stop_reason(self) -> StopReason:
        return StopReason.REPEATED_ACTIONS

    def get_description(self) -> str:
        return f"Repeated same action {self.max_repeats} times"


@dataclass
class ErrorThresholdCondition(StopCondition):
    """Stop when too many consecutive errors occur."""

    max_consecutive_errors: int = 3

    def should_stop(self, steps: List[ReActStep], current_step: int) -> bool:
        if len(steps) < self.max_consecutive_errors:
            return False

        # Check last N steps for consecutive errors
        recent_steps = steps[-self.max_consecutive_errors :]
        return all(step.is_error_step for step in recent_steps)

    def get_stop_reason(self) -> StopReason:
        return StopReason.ERROR_THRESHOLD

    def get_description(self) -> str:
        return f"Too many consecutive errors ({self.max_consecutive_errors})"


@dataclass
class InfiniteLoopDetector(StopCondition):
    """Detect various forms of infinite loops."""

    pattern_length: int = 3  # Minimum pattern length to detect
    min_repetitions: int = 2  # Minimum repetitions to consider a loop

    def should_stop(self, steps: List[ReActStep], current_step: int) -> bool:
        if len(steps) < self.pattern_length * self.min_repetitions:
            return False

        # Extract action signatures for pattern matching
        action_signatures = []
        for step in steps:
            if step.is_action_step:
                # Convert action_input to hashable form for signature matching
                hashable_input = make_hashable(step.action_input) if step.action_input else None
                signature = (step.action, hashable_input)
                action_signatures.append(signature)

        if len(action_signatures) < self.pattern_length * self.min_repetitions:
            return False

        # Look for repeating patterns
        for pattern_len in range(self.pattern_length, len(action_signatures) // 2 + 1):
            if self._has_repeating_pattern(action_signatures, pattern_len):
                return True

        return False

    def _has_repeating_pattern(self, signatures: List, pattern_length: int) -> bool:
        """Check if the list has a repeating pattern of given length."""
        if len(signatures) < pattern_length * self.min_repetitions:
            return False

        # Check if the last pattern_length * min_repetitions elements form a repeating pattern
        tail = signatures[-(pattern_length * self.min_repetitions) :]

        pattern = tail[:pattern_length]
        for i in range(self.min_repetitions):
            start_idx = i * pattern_length
            end_idx = start_idx + pattern_length
            if tail[start_idx:end_idx] != pattern:
                return False

        return True

    def get_stop_reason(self) -> StopReason:
        return StopReason.REPEATED_ACTIONS

    def get_description(self) -> str:
        return "Infinite loop pattern detected"


@dataclass
class ThinkingOnlyCondition(StopCondition):
    """Stop when agent repeatedly outputs only thinking without actions."""

    max_thinking_only_steps: int = 3

    def should_stop(self, steps: List[ReActStep], current_step: int) -> bool:
        if len(steps) < self.max_thinking_only_steps:
            return False

        # Check last N steps - if they all have thought but no action/answer, agent is stuck
        recent_steps = steps[-self.max_thinking_only_steps :]
        thinking_only_steps = [
            step for step in recent_steps if step.thought and not step.action and not step.answer
        ]

        return len(thinking_only_steps) >= self.max_thinking_only_steps

    def get_stop_reason(self) -> StopReason:
        return StopReason.FORCED_STOP

    def get_description(self) -> str:
        return f"Agent stuck in thinking-only mode for {self.max_thinking_only_steps} consecutive steps"


@dataclass
class EmptyResponseCondition(StopCondition):
    """Stop when LLM repeatedly returns completely empty responses."""

    max_empty_responses: int = 2

    def should_stop(self, steps: List[ReActStep], current_step: int) -> bool:
        if len(steps) < self.max_empty_responses:
            return False

        # Check last N steps - if they all have NO thought, NO action, and NO answer, LLM is stuck
        recent_steps = steps[-self.max_empty_responses :]
        empty_steps = [
            step
            for step in recent_steps
            if not step.thought and not step.action and not step.answer
        ]

        return len(empty_steps) >= self.max_empty_responses

    def get_stop_reason(self) -> StopReason:
        return StopReason.FORCED_STOP

    def get_description(self) -> str:
        return f"LLM returned {self.max_empty_responses} consecutive empty responses"


class SafetyManager:
    """Manages all safety conditions for ReAct loop."""

    def __init__(
        self,
        max_steps: int = 10,
        max_budget: Optional[float] = None,
        max_time_seconds: Optional[float] = None,
        max_repeated_actions: int = 3,
        max_consecutive_errors: int = 3,
        max_thinking_only_steps: int = 3,
        max_empty_responses: int = 2,
        enable_loop_detection: bool = True,
    ):
        self.conditions: List[StopCondition] = []

        # Always enforce max steps
        self.conditions.append(MaxStepsCondition(max_steps))

        # Optional conditions
        if max_budget:
            self.conditions.append(MaxBudgetCondition(max_budget))

        if max_time_seconds:
            self.conditions.append(MaxTimeCondition(max_time_seconds))

        if max_repeated_actions > 0:
            self.conditions.append(RepeatedActionsCondition(max_repeated_actions))

        if max_consecutive_errors > 0:
            self.conditions.append(ErrorThresholdCondition(max_consecutive_errors))

        if max_thinking_only_steps > 0:
            self.conditions.append(ThinkingOnlyCondition(max_thinking_only_steps))

        if max_empty_responses > 0:
            self.conditions.append(EmptyResponseCondition(max_empty_responses))

        if enable_loop_detection:
            self.conditions.append(InfiniteLoopDetector())

    def should_stop(self, steps: List[ReActStep], current_step: int) -> Optional[StopCondition]:
        """Check all conditions and return first triggered condition."""
        for condition in self.conditions:
            if condition.should_stop(steps, current_step):
                return condition
        return None

    def get_status_summary(self, steps: List[ReActStep], current_step: int) -> Dict[str, Any]:
        """Get current status against all safety conditions."""
        total_cost = sum(step.cost for step in steps)
        total_time = sum(step.execution_time for step in steps)
        error_count = sum(1 for step in steps if step.is_error_step)

        return {
            "current_step": current_step,
            "total_steps": len(steps),
            "total_cost": total_cost,
            "total_time": total_time,
            "error_count": error_count,
            "conditions": [
                {
                    "type": type(condition).__name__,
                    "description": condition.get_description(),
                    "triggered": condition.should_stop(steps, current_step),
                }
                for condition in self.conditions
            ],
        }
