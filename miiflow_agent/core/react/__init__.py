"""ReAct (Reasoning + Acting) — the unified agent loop.

Planning and multi-agent fan-out used to be separate orchestrators
(`PlanAndExecuteOrchestrator`, `MultiAgentOrchestrator`). They are now
emergent behaviors of the same ReAct loop:

  - Planning: the model calls `enter_plan_mode` / `exit_plan_mode`
    (deferred tools in `miiflow_agent.core.tools.plan_mode`). While in
    plan mode the executor refuses any tool that doesn't declare
    `is_read_only=True`.
  - Multi-agent dispatch: the model calls `dispatch_assistant` (the
    framework synthesizes it from `Agent.sub_agents`). Multiple calls
    in one turn run in parallel via the batch executor.

Usage:
    from miiflow_agent.core.react import ReActOrchestrator, ReActFactory

    orchestrator = ReActFactory.create_orchestrator(agent, max_steps=25)
    result = await orchestrator.execute("Find today's top news", context)
"""

from .orchestrator import ReActOrchestrator
from .factory import ReActFactory
from .events import EventBus, EventFactory

# Enums
from .enums import ReActEventType, StopReason

# Models
from .models import (
    ReActStep,
    ReActResult,
    ToolCall,
    ParseResult,
    ReasoningContext,
)

# Events
from .react_events import ReActEvent

# Exceptions
from .exceptions import ReActParsingError, ReActExecutionError, SafetyViolationError

# Safety
from .safety import StopCondition, SafetyManager

# Execution state
from .execution import ExecutionState

# SubAgent registry (still useful for sub-agent metadata even though
# dispatch is now a normal tool — see core/react/dispatch.py)
from .subagent_registry import (
    DynamicSubAgentConfig,
    SubAgentRegistry,
    get_global_registry,
)
from .model_selector import (
    ModelSelector,
    ModelTier,
    TaskComplexity,
    ComplexityDetector,
    select_model_for_task,
    detect_complexity,
)

__all__ = [
    # Main interfaces
    "ReActOrchestrator",
    "ReActFactory",
    # Event system
    "EventBus",
    "EventFactory",
    # Enums
    "ReActEventType",
    "StopReason",
    # Models
    "ReActStep",
    "ReActResult",
    "ToolCall",
    "ParseResult",
    "ReasoningContext",
    # Events
    "ReActEvent",
    # Exceptions
    "ReActParsingError",
    "ReActExecutionError",
    "SafetyViolationError",
    # Safety
    "StopCondition",
    "SafetyManager",
    # Execution state
    "ExecutionState",
    # SubAgent metadata
    "DynamicSubAgentConfig",
    "SubAgentRegistry",
    "get_global_registry",
    "ModelSelector",
    "ModelTier",
    "TaskComplexity",
    "ComplexityDetector",
    "select_model_for_task",
    "detect_complexity",
]

__version__ = "0.6.0"  # Migration to unified ReAct loop; legacy orchestrators removed
