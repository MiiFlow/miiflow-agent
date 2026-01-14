"""ReAct (Reasoning + Acting) and Plan & Execute - Architecture

Usage - ReAct:
    from miiflow_agent.core.react import ReActOrchestrator, ReActFactory

    orchestrator = ReActFactory.create_orchestrator(agent, max_steps=10)
    result = await orchestrator.execute("Find today's top news", context)

Usage - Plan & Execute:
    from miiflow_agent.core.react import PlanAndExecuteOrchestrator

    orchestrator = PlanAndExecuteOrchestrator(tool_executor, event_bus, safety_manager)
    result = await orchestrator.execute("Create Q4 sales report", context)
"""

# New clean architecture - no legacy imports
from .orchestrator import ReActOrchestrator
from .plan_execute_orchestrator import PlanAndExecuteOrchestrator
from .factory import ReActFactory
from .events import EventBus, EventFactory

# Enums
from .enums import ReActEventType, StopReason, PlanExecuteEventType

# Models
from .models import (
    ReActStep,
    ReActResult,
    ToolCall,
    ParseResult,
    ReasoningContext,
    SubTask,
    Plan,
    PlanExecuteResult,
)

# Events
from .react_events import ReActEvent, PlanExecuteEvent

# Exceptions
from .exceptions import ReActParsingError, ReActExecutionError, SafetyViolationError

# Parser and safety
from .parser import ReActParser
from .safety import StopCondition, SafetyManager

# Execution state
from .execution import ExecutionState

# Multi-agent components
from .multi_agent_orchestrator import MultiAgentOrchestrator
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
from .task_tool import TaskTool, TaskToolResult, create_task_tool

__all__ = [
    # Main interfaces
    "ReActOrchestrator",
    "PlanAndExecuteOrchestrator",
    "ReActFactory",
    # Event system
    "EventBus",
    "EventFactory",
    # Enums
    "ReActEventType",
    "StopReason",
    "PlanExecuteEventType",
    # Models
    "ReActStep",
    "ReActResult",
    "ToolCall",
    "ParseResult",
    "ReasoningContext",
    "SubTask",
    "Plan",
    "PlanExecuteResult",
    # Events
    "ReActEvent",
    "PlanExecuteEvent",
    # Exceptions
    "ReActParsingError",
    "ReActExecutionError",
    "SafetyViolationError",
    # Parser and safety
    "ReActParser",
    "StopCondition",
    "SafetyManager",
    # Execution state
    "ExecutionState",
    # Multi-agent components
    "MultiAgentOrchestrator",
    "DynamicSubAgentConfig",
    "SubAgentRegistry",
    "get_global_registry",
    "ModelSelector",
    "ModelTier",
    "TaskComplexity",
    "ComplexityDetector",
    "select_model_for_task",
    "detect_complexity",
    "TaskTool",
    "TaskToolResult",
    "create_task_tool",
]

__version__ = "0.5.0"  # Added dynamic multi-agent components
