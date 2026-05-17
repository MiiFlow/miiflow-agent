"""Enumeration types for the unified ReAct loop.

The legacy `PlanExecuteEventType`, `ParallelPlanEventType`, and
`MultiAgentEventType` enums were removed alongside their orchestrators
in the unified-ReAct migration. Telemetry that used to derive subtask
boundaries from those events now derives them from the flat
`ReActEventType` stream — see `core/react/events/` for the bus.
"""

from enum import Enum


class ReActEventType(Enum):
    """Types of events emitted during ReAct execution."""

    STEP_START = "step_start"
    THOUGHT = "thought"
    THINKING_CHUNK = "thinking_chunk"  # Streaming chunks during thinking
    ACTION_PLANNED = "action_planned"
    ACTION_EXECUTING = "action_executing"
    OBSERVATION = "observation"
    STEP_COMPLETE = "step_complete"
    FINAL_ANSWER = "final_answer"
    FINAL_ANSWER_CHUNK = "final_answer_chunk"  # Streaming chunks for final answer
    ERROR = "error"
    STOP_CONDITION = "stop_condition"
    CLARIFICATION_NEEDED = "clarification_needed"  # Agent needs user input
    TOOL_APPROVAL_NEEDED = "tool_approval_needed"  # Tool requires user approval before execution
    VISUALIZATION = "visualization"  # Tool returned a visualization result
    MEDIA = "media"  # Tool returned a media result (image/video/audio)
    ARTIFACT = "artifact"  # Tool returned a downloadable artifact (PDF, HTML, ...)
    PROGRESS = "progress"  # Progress snapshot update
    LLM_TRUNCATED = "llm_truncated"  # Model hit max_tokens (often mid-tool-call)
    SUBAGENT_DISPATCH = "subagent_dispatch"  # Sub-assistant dispatch (start/progress/complete/failed sub-events)
    PLAN_MODE_ENTERED = "plan_mode_entered"  # Model called enter_plan_mode; only read-only tools execute until exit
    PLAN_MODE_EXITED = "plan_mode_exited"  # User-approved exit_plan_mode; loop resumes with permission_mode=default
    PLAN_APPROVAL_NEEDED = "plan_approval_needed"  # exit_plan_mode raised — loop pauses for user decision
    TOOL_BLOCKED_BY_PLAN_MODE = "tool_blocked_by_plan_mode"  # Executor refused a non-read-only tool while in plan mode


class StopReason(Enum):
    """Reasons why ReAct loop terminated."""

    ANSWER_COMPLETE = "answer_complete"
    MAX_STEPS = "max_steps"
    MAX_BUDGET = "max_budget"
    MAX_TIME = "max_time"
    REPEATED_ACTIONS = "repeated_actions"
    ERROR_THRESHOLD = "error_threshold"
    FORCED_STOP = "forced_stop"
    USER_CANCELLED = "user_cancelled"
    NEEDS_CLARIFICATION = "needs_clarification"  # Agent needs user input to continue
    RECOVERY_EXHAUSTED = "recovery_exhausted"  # All recovery strategies exhausted

