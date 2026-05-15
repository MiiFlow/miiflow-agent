"""Event dataclass for the unified ReAct loop.

`PlanExecuteEvent`, `ParallelPlanEvent`, and `MultiAgentEvent` were
removed alongside their orchestrators in the unified-ReAct migration —
planning is now an `enter_plan_mode` / `exit_plan_mode` tool call inside
ReAct, and multi-agent dispatch is the `dispatch_assistant` tool. Both
emit ReActEventType events on the same bus.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict

from .enums import ReActEventType


@dataclass
class ReActEvent:
    """Event emitted during ReAct execution for streaming."""

    event_type: ReActEventType
    step_number: int
    data: Dict[str, Any]

    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for streaming."""
        return {
            "event_type": self.event_type.value,
            "step_number": self.step_number,
            "data": self.data,
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        """Convert event to JSON string."""
        import json

        return json.dumps(self.to_dict())
