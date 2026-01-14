"""Data models for ReAct and Plan & Execute systems."""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .enums import StopReason


@dataclass
class ReActStep:
    """Single step in ReAct reasoning loop."""

    step_number: int
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    answer: Optional[str] = None

    # Metadata
    timestamp: float = field(default_factory=time.time)
    execution_time: float = 0.0
    cost: float = 0.0
    tokens_used: int = 0

    # Error handling
    error: Optional[str] = None
    retry_count: int = 0

    # Removed tracing - stateless execution

    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_action_step(self) -> bool:
        """Whether this step involves a tool action."""
        return self.action is not None

    @property
    def is_final_step(self) -> bool:
        """Whether this step contains the final answer."""
        return self.answer is not None

    @property
    def is_error_step(self) -> bool:
        """Whether this step had an error."""
        return self.error is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary for serialization."""
        return {
            "step_number": self.step_number,
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation,
            "answer": self.answer,
            "timestamp": self.timestamp,
            "execution_time": self.execution_time,
            "cost": self.cost,
            "tokens_used": self.tokens_used,
            "error": self.error,
            "retry_count": self.retry_count,
            "metadata": self.metadata,
        }


@dataclass
class ReActResult:
    """Complete result of ReAct execution."""

    steps: List[ReActStep]
    final_answer: str
    stop_reason: StopReason

    # Performance metrics
    total_cost: float = 0.0
    total_execution_time: float = 0.0
    total_tokens: int = 0

    # Loop statistics
    steps_count: int = field(init=False)
    action_steps_count: int = field(init=False)
    error_steps_count: int = field(init=False)

    # Tracing

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate derived statistics."""
        self.steps_count = len(self.steps)
        self.action_steps_count = sum(1 for step in self.steps if step.is_action_step)
        self.error_steps_count = sum(1 for step in self.steps if step.is_error_step)

        if not self.total_cost:
            self.total_cost = sum(step.cost for step in self.steps)
        if not self.total_execution_time:
            self.total_execution_time = sum(step.execution_time for step in self.steps)
        if not self.total_tokens:
            self.total_tokens = sum(step.tokens_used for step in self.steps)

    @property
    def success_rate(self) -> float:
        """Percentage of steps that completed without errors."""
        if not self.steps:
            return 0.0
        return (self.steps_count - self.error_steps_count) / self.steps_count

    @property
    def avg_step_time(self) -> float:
        """Average execution time per step."""
        if not self.steps:
            return 0.0
        return self.total_execution_time / self.steps_count

    @property
    def tools_used(self) -> List[str]:
        """List of unique tools used during execution."""
        tools = set()
        for step in self.steps:
            if step.action:
                tools.add(step.action)
        return list(tools)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "steps": [step.to_dict() for step in self.steps],
            "final_answer": self.final_answer,
            "stop_reason": self.stop_reason.value,
            "total_cost": self.total_cost,
            "total_execution_time": self.total_execution_time,
            "total_tokens": self.total_tokens,
            "steps_count": self.steps_count,
            "action_steps_count": self.action_steps_count,
            "error_steps_count": self.error_steps_count,
            "success_rate": self.success_rate,
            "avg_step_time": self.avg_step_time,
            "tools_used": self.tools_used,
            "metadata": self.metadata,
        }


@dataclass
class ToolCall:
    """Represents a tool call action."""

    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_step(cls, step: ReActStep) -> "ToolCall":
        """Create ToolCall from ReActStep."""
        if not step.is_action_step:
            raise ValueError("Step is not an action step")
        return cls(name=step.action, arguments=step.action_input or {})


@dataclass
class ParseResult:
    """Result of parsing a ReAct response."""

    thought: str
    action_type: str  # "tool_call" or "final_answer"
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    answer: Optional[str] = None

    # Parsing metadata
    original_response: str = ""
    was_healed: bool = False
    healing_applied: str = ""
    confidence: float = 1.0


@dataclass
class ReasoningContext:
    """Context for reasoning operations."""

    current_step: int
    steps: List[ReActStep]
    total_cost: float = 0.0
    total_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def last_step(self) -> Optional[ReActStep]:
        """Get the last step if any."""
        return self.steps[-1] if self.steps else None

    @property
    def error_count(self) -> int:
        """Count of error steps."""
        return sum(1 for step in self.steps if step.is_error_step)


# Plan and Execute Data Structures


@dataclass
class SubTask:
    """A single subtask in a plan."""

    id: int
    description: str
    required_tools: List[str] = field(default_factory=list)
    dependencies: List[int] = field(default_factory=list)  # IDs of subtasks that must complete first
    success_criteria: Optional[str] = None

    # Execution results
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[str] = None
    error: Optional[str] = None

    # Metadata
    timestamp: float = field(default_factory=time.time)
    execution_time: float = 0.0
    cost: float = 0.0
    tokens_used: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert subtask to dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "required_tools": self.required_tools,
            "dependencies": self.dependencies,
            "success_criteria": self.success_criteria,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "timestamp": self.timestamp,
            "execution_time": self.execution_time,
            "cost": self.cost,
            "tokens_used": self.tokens_used,
        }


@dataclass
class Plan:
    """A structured plan with subtasks."""

    subtasks: List[SubTask]
    goal: str
    reasoning: str  # Why this plan was chosen

    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_subtasks(self) -> int:
        """Total number of subtasks."""
        return len(self.subtasks)

    @property
    def completed_subtasks(self) -> int:
        """Number of completed subtasks."""
        return sum(1 for st in self.subtasks if st.status == "completed")

    @property
    def failed_subtasks(self) -> int:
        """Number of failed subtasks."""
        return sum(1 for st in self.subtasks if st.status == "failed")

    @property
    def progress_percentage(self) -> float:
        """Progress as percentage."""
        if not self.subtasks:
            return 0.0
        return (self.completed_subtasks / self.total_subtasks) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert plan to dictionary."""
        return {
            "subtasks": [st.to_dict() for st in self.subtasks],
            "goal": self.goal,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp,
            "total_subtasks": self.total_subtasks,
            "completed_subtasks": self.completed_subtasks,
            "failed_subtasks": self.failed_subtasks,
            "progress_percentage": self.progress_percentage,
            "metadata": self.metadata,
        }


@dataclass
class PlanExecuteResult:
    """Result of Plan and Execute orchestration."""

    plan: Plan
    final_answer: str
    stop_reason: StopReason
    replans: int = 0  # Number of times we re-planned

    # Performance metrics
    total_cost: float = 0.0
    total_execution_time: float = 0.0
    total_tokens: int = 0

    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Percentage of subtasks that completed successfully."""
        if not self.plan.subtasks:
            return 0.0
        return (self.plan.completed_subtasks / self.plan.total_subtasks) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "plan": self.plan.to_dict(),
            "final_answer": self.final_answer,
            "stop_reason": self.stop_reason.value,
            "replans": self.replans,
            "total_cost": self.total_cost,
            "total_execution_time": self.total_execution_time,
            "total_tokens": self.total_tokens,
            "success_rate": self.success_rate,
            "metadata": self.metadata,
        }


# Parallel Plan Execution Data Structures


@dataclass
class ExecutionWave:
    """A wave of subtasks that can execute in parallel."""

    wave_number: int
    subtasks: List[SubTask]
    parallel_count: int = 0

    # Execution metrics
    execution_time: float = 0.0

    def __post_init__(self):
        """Calculate derived fields."""
        if not self.parallel_count:
            self.parallel_count = len(self.subtasks)

    def to_dict(self) -> Dict[str, Any]:
        """Convert wave to dictionary."""
        return {
            "wave_number": self.wave_number,
            "subtasks": [st.to_dict() for st in self.subtasks],
            "parallel_count": self.parallel_count,
            "execution_time": self.execution_time,
        }


# Multi-Agent Execution Data Structures


@dataclass
class SubAgentConfig:
    """Configuration for a specialized subagent."""

    name: str
    role: str  # "researcher", "analyzer", "coder", "summarizer", etc.
    focus: str  # Specific aspect this agent should focus on
    query: str  # The specific query/task for this subagent
    output_key: str  # Unique key for shared state

    # Optional customization
    system_prompt: Optional[str] = None
    max_steps: int = 10

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "name": self.name,
            "role": self.role,
            "focus": self.focus,
            "query": self.query,
            "output_key": self.output_key,
            "system_prompt": self.system_prompt,
            "max_steps": self.max_steps,
        }


@dataclass
class SubAgentResult:
    """Result from a single subagent execution."""

    agent_name: str
    role: str
    output_key: str
    result: Optional[str] = None
    success: bool = True
    error: Optional[str] = None

    # Performance metrics
    execution_time: float = 0.0
    tokens_used: int = 0
    cost: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "agent_name": self.agent_name,
            "role": self.role,
            "output_key": self.output_key,
            "result": self.result,
            "success": self.success,
            "error": self.error,
            "execution_time": self.execution_time,
            "tokens_used": self.tokens_used,
            "cost": self.cost,
        }


@dataclass
class SubAgentPlan:
    """Plan for subagent allocation."""

    reasoning: str
    subagent_configs: List[SubAgentConfig]

    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert plan to dictionary."""
        return {
            "reasoning": self.reasoning,
            "subagent_configs": [cfg.to_dict() for cfg in self.subagent_configs],
            "timestamp": self.timestamp,
        }


@dataclass
class MultiAgentResult:
    """Result of Multi-Agent orchestration."""

    subagent_results: List[SubAgentResult]
    final_answer: str
    stop_reason: StopReason

    # Performance metrics
    total_cost: float = 0.0
    total_execution_time: float = 0.0
    total_tokens: int = 0

    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success_count(self) -> int:
        """Number of successful subagents."""
        return sum(1 for r in self.subagent_results if r.success)

    @property
    def failure_count(self) -> int:
        """Number of failed subagents."""
        return sum(1 for r in self.subagent_results if not r.success)

    @property
    def success_rate(self) -> float:
        """Percentage of subagents that completed successfully."""
        if not self.subagent_results:
            return 0.0
        return (self.success_count / len(self.subagent_results)) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "subagent_results": [r.to_dict() for r in self.subagent_results],
            "final_answer": self.final_answer,
            "stop_reason": self.stop_reason.value,
            "total_cost": self.total_cost,
            "total_execution_time": self.total_execution_time,
            "total_tokens": self.total_tokens,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate,
            "metadata": self.metadata,
        }
