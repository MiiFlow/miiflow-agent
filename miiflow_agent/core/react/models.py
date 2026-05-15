"""Data models for ReAct and Plan & Execute systems."""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .enums import StopReason


@dataclass
class ToolInvocation:
    """One tool invocation in a (possibly parallel) batch within a step.

    Distinct from `ToolCall` below (which is a logical convenience class
    decoupled from a step). `ToolInvocation` captures the provider-supplied
    `tool_call_id` so transcripts can round-trip the matching
    `tool_result` block — required by Anthropic's API.
    """

    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    inputs: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    error: Optional[str] = None
    description: Optional[str] = None  # LLM-supplied __description for status UI

    @property
    def is_success(self) -> bool:
        return self.error is None


@dataclass
class ReActStep:
    """Single step in ReAct reasoning loop.

    A step corresponds to one LLM turn. The model may emit:
    - Final answer (terminal step): `answer` is set, no tool calls.
    - Single tool call (most common today): `action` / `action_input` /
      `observation` populated; equivalent to `tool_invocations[0]`.
    - Multiple parallel tool calls (post-refactor): `tool_invocations`
      holds N entries, one per tool_use block the model emitted.

    Back-compat: the legacy single-tool fields (`action`, `action_input`,
    `observation`, `error`) are kept and mirror `tool_invocations[0]` for
    callers that haven't migrated. Use `is_batch_step` to detect when a
    step has >1 invocation.
    """

    step_number: int
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    answer: Optional[str] = None

    # Parallel-tool fields (additive). Populated by the orchestrator when
    # the model emits >=1 tool_use block. For single-tool steps this list
    # has exactly one entry whose name/inputs/observation mirror the
    # legacy fields; for batch steps it has N. Empty for final-answer
    # steps and pure thinking steps.
    tool_invocations: List[ToolInvocation] = field(default_factory=list)

    # Metadata
    timestamp: float = field(default_factory=time.time)
    execution_time: float = 0.0
    cost: float = 0.0
    tokens_used: int = 0

    # Error handling. `error` reflects step-level failure (e.g. malformed
    # tool_call data, recovery exhausted, every tool in a parallel batch
    # failed). Per-invocation errors live on `tool_invocations[i].error`
    # and do NOT mark the step as `is_error_step` — the LLM sees the
    # error as a tool observation and reacts naturally on the next turn,
    # which is the desired behavior; surfacing it as a step-level error
    # would trigger recovery_manager unnecessarily.
    error: Optional[str] = None
    retry_count: int = 0

    # Removed tracing - stateless execution

    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_action_step(self) -> bool:
        """Whether this step involves a tool action."""
        return self.action is not None or bool(self.tool_invocations)

    @property
    def is_final_step(self) -> bool:
        """Whether this step contains the final answer."""
        return self.answer is not None

    @property
    def is_error_step(self) -> bool:
        """True for step-level failures only — NOT per-invocation errors.

        A parallel batch where some tools fail and others succeed is
        treated as a normal step that the LLM can react to; only when
        the whole step is broken (malformed tool_call, every tool in
        the batch failed) does `error` get set and recovery fire.
        """
        return self.error is not None

    @property
    def has_failed_invocations(self) -> bool:
        """True if any tool invocation in this step ended with an error.

        Use this for stats / observability — not for recovery decisions.
        Recovery fires on `is_error_step` (step-level), so a step where
        2/3 parallel tools succeeded continues to the next iteration
        without re-running the successful ones.
        """
        return any(inv.error is not None for inv in self.tool_invocations)

    @property
    def is_batch_step(self) -> bool:
        """True when this step ran >1 tool_use block in parallel/serial batch."""
        return len(self.tool_invocations) > 1

    @property
    def all_invocations(self) -> List[ToolInvocation]:
        """Canonical accessor for the invocations of this step.

        For single-tool steps that only populated the legacy fields,
        synthesizes a one-element list on the fly so consumers that
        always iterate keep working.
        """
        if self.tool_invocations:
            return self.tool_invocations
        if self.action is not None:
            return [
                ToolInvocation(
                    name=self.action,
                    inputs=self.action_input,
                    observation=self.observation,
                    error=self.error,
                )
            ]
        return []

    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary for serialization."""
        return {
            "step_number": self.step_number,
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation,
            "answer": self.answer,
            "tool_invocations": [
                {
                    "tool_call_id": inv.tool_call_id,
                    "name": inv.name,
                    "inputs": inv.inputs,
                    "observation": inv.observation,
                    "error": inv.error,
                    "description": inv.description,
                }
                for inv in self.tool_invocations
            ],
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

    # Clarification data (when stop_reason == NEEDS_CLARIFICATION)
    clarification_data: Optional[Dict[str, Any]] = None

    # Progress snapshot (from ProgressTracker)
    progress: Optional[Dict[str, Any]] = None

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
        result = {
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
        if self.clarification_data:
            result["clarification_data"] = self.clarification_data
        return result


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


