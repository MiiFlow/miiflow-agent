"""ReAct data structures and schemas."""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class ReActEventType(Enum):
    """Types of events emitted during ReAct execution."""
    STEP_START = "step_start"
    THOUGHT = "thought"
    ACTION_PLANNED = "action_planned" 
    ACTION_EXECUTING = "action_executing"
    OBSERVATION = "observation"
    STEP_COMPLETE = "step_complete"
    FINAL_ANSWER = "final_answer"
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
            "metadata": self.metadata
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
            "metadata": self.metadata
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
            "trace_id": self.trace_id
        }
    
    def to_json(self) -> str:
        """Convert event to JSON string."""
        import json
        return json.dumps(self.to_dict())


# JSON Schema for strict ReAct output validation
REACT_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["thought", "action_type"],
    "properties": {
        "thought": {
            "type": "string",
            "minLength": 1,
            "description": "Your current reasoning about the task"
        },
        "action_type": {
            "type": "string",
            "enum": ["tool_call", "final_answer"],
            "description": "Whether to call a tool or provide final answer"
        },
        "action": {
            "type": "string", 
            "description": "Tool name to call (required if action_type is tool_call)"
        },
        "action_input": {
            "type": "object",
            "description": "Parameters for the tool call (required if action_type is tool_call)"
        },
        "answer": {
            "type": "string",
            "minLength": 1,
            "description": "Final answer (required if action_type is final_answer)"
        }
    },
    "additionalProperties": False,
    "if": {
        "properties": {"action_type": {"const": "tool_call"}}
    },
    "then": {
        "required": ["action", "action_input"]
    },
    "else": {
        "required": ["answer"]
    }
}


# System prompt template for ReAct reasoning
REACT_SYSTEM_PROMPT = """You are an AI assistant that follows the ReAct (Reasoning + Acting) pattern.

CRITICAL: You must respond with ONLY valid JSON. No other text before or after.

For each step, respond with a JSON object containing:
1. "thought": Your reasoning about what to do next
2. "action_type": Either "tool_call" or "final_answer"
3. If "tool_call": Include "action" (tool name) and "action_input" (parameters)
4. If "final_answer": Include "answer" (your final response)

Available tools:
{tools}

EXAMPLES:

To use a tool:
{{"thought": "I need to search for current news to provide the user with today's top stories", "action_type": "tool_call", "action": "search_news", "action_input": {{"query": "today news"}}}}

To provide final answer:
{{"thought": "Based on the search results, I now have enough information to provide 5 bullet points", "action_type": "final_answer", "answer": "Here are today's top 5 news stories:\\n• Story 1\\n• Story 2\\n• Story 3\\n• Story 4\\n• Story 5"}}

REMEMBER: Respond with ONLY JSON. No explanations, no markdown, just the JSON object."""


# Additional Value Objects and Supporting Classes

@dataclass
class ToolCall:
    """Represents a tool call action."""
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_step(cls, step: ReActStep) -> 'ToolCall':
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