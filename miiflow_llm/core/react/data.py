"""ReAct data structures and schemas."""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


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


class StopReason(Enum):
    """Reasons why ReAct loop terminated."""

    ANSWER_COMPLETE = "answer_complete"
    MAX_STEPS = "max_steps"
    MAX_BUDGET = "max_budget"
    MAX_TIME = "max_time"
    REPEATED_ACTIONS = "repeated_actions"
    ERROR_THRESHOLD = "error_threshold"
    FORCED_STOP = "forced_stop"


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
            "trace_id": self.trace_id,
        }

    def to_json(self) -> str:
        """Convert event to JSON string."""
        import json

        return json.dumps(self.to_dict())


# System prompt template for XML-based ReAct reasoning
REACT_SYSTEM_PROMPT = """You are an AI assistant that follows the ReAct (Reasoning + Acting) pattern using XML tags.

CRITICAL: Every response MUST contain either a <tool_call> OR an <answer> tag. Never output only <thinking>.

Response format:

<thinking>
(Optional) Your step-by-step reasoning about what to do next.
</thinking>

<tool_call name="tool_name">
{{"param1": "value1", "param2": "value2"}}
</tool_call>

After calling a tool, you will receive an observation from the system:
<observation>
Tool execution result
</observation>

DO NOT include <observation> tags in your response - they are added automatically by the system.
Then respond with either another <tool_call> or <answer>:

<answer>
Your complete answer to the user. This will be streamed in real-time.
</answer>

Available tools:
{tools}

CRITICAL RULES:
1. EVERY response must contain <tool_call> OR <answer> - never only <thinking>
2. Use ONLY ONE tool call per response
3. <thinking> is optional but recommended for clarity
4. Wait for <observation> after each tool call before deciding next action
5. Tool parameters must be valid JSON inside <tool_call> tags
6. Use EXACT tool names as listed above - do not abbreviate or modify them
7. When you have enough information, provide <answer> immediately
8. NEVER include <observation> tags in your response - the system provides them automatically

EXAMPLES:

Example 1 - First tool call:
<thinking>
The user is asking about current weather in Paris. I need to use the weather tool.
</thinking>

<tool_call name="get_weather">
{{"location": "Paris", "units": "celsius"}}
</tool_call>

Example 2 - After receiving observation, provide final answer:
(You received: <observation>{{"temp": 18, "condition": "cloudy"}}</observation>)

<thinking>
Based on the weather data, I can now answer the user.
</thinking>

<answer>
The current weather in Paris is partly cloudy with a temperature of 18°C. There's a light breeze from the west at 15 km/h, and the humidity is at 65%. It's a pleasant day overall!
</answer>

Example 3 - Direct tool call (no thinking):
<tool_call name="get_weather">
{{"location": "Paris", "units": "celsius"}}
</tool_call>

IMPORTANT:
- NEVER end your response with only <thinking> - always follow with <tool_call> or <answer>
- If you're unsure what to do, provide your best <answer> rather than stopping at thinking
- Your <answer> will be streamed to the user as you write it"""


# Additional Value Objects and Supporting Classes


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


# Exceptions
class ReActParsingError(Exception):
    """Raised when ReAct response cannot be parsed or healed."""

    pass


class ReActExecutionError(Exception):
    """Raised when ReAct execution fails."""

    pass


class SafetyViolationError(Exception):
    """Raised when a safety condition is violated."""

    pass
