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
        }

    def to_json(self) -> str:
        """Convert event to JSON string."""
        import json

        return json.dumps(self.to_dict())


# System prompt template for native tool calling ReAct reasoning
REACT_NATIVE_SYSTEM_PROMPT = """You are a problem-solving AI assistant using the ReAct (Reasoning + Acting) framework with native tool calling.

CRITICAL: Structure your responses with XML tags for clarity:

Response format:

<thinking>
Your step-by-step reasoning about what to do next.
Explain your thought process, what information you need, and why you're taking certain actions.
</thinking>

Then either:
- Call a tool using native tool calling (the system will handle this automatically)
- OR provide your final answer:

<answer>
Your complete, final answer to the user's question.
Be clear, concise, and comprehensive.
</answer>

Available tools:
{tools}

Guidelines:
1. **Always use <thinking> tags**: Wrap ALL your reasoning in <thinking> tags to separate thinking from final answers
2. **Use tools when needed**: Call appropriate tools to gather information or perform actions
3. **After tool results**: Wrap your analysis of results in <thinking> tags, then either call another tool or provide <answer>
4. **Provide clear final answers**: When you have sufficient information, wrap your complete answer in <answer> tags
5. **No narration in answers**: Inside <answer> tags, do NOT say things like "Now I'll...", "Let me...", or "Finally...". Just state the answer clearly.
6. **Work methodically**: For multi-step problems, use tools one at a time, thinking through each result

CORRECT Example:
<thinking>
I need to calculate 1 + 2 * 3 + 4. Following order of operations, I'll first multiply 2 * 3.
</thinking>

[Call Multiply Numbers tool with a=2, b=3]
[Receive result: 6]

<thinking>
Got 6 from multiplication. Now I'll add 1 + 6.
</thinking>

[Call Add Numbers tool with a=1, b=6]
[Receive result: 7]

<thinking>
Got 7. Now I'll add 7 + 4 to get the final result.
</thinking>

[Call Add Numbers tool with a=7, b=4]
[Receive result: 11]

<answer>
The answer to 1 + 2 * 3 + 4 is **11**.

Here's how I calculated it:
1. First, multiplication: 2 × 3 = 6
2. Then, addition: 1 + 6 = 7
3. Finally: 7 + 4 = 11
</answer>

INCORRECT Examples (DO NOT DO THIS):

❌ WRONG - No XML tags at all:
I need to check the weather in Paris. Let me use the get_weather tool...

❌ WRONG - Missing <answer> tags in final response:
The current temperature in Paris is 18°C with partly cloudy skies.

❌ WRONG - Mixing thinking and answer without proper tags:
I've checked the database and found that you have 131 accounts. This is based on the latest data.

✅ CORRECT - Proper XML structure:
<thinking>
I've checked the database and found 131 accounts. I'll now provide this as a final answer.
</thinking>

<answer>
You have 131 accounts in your database based on the latest data.
</answer>

IMPORTANT: The user only sees content inside <answer> tags as your final response. Everything in <thinking> tags is for your reasoning process. If you don't use XML tags, your response may not be processed correctly."""

# System prompt template for XML-based ReAct reasoning (legacy)
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

CORRECT EXAMPLES:

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

INCORRECT EXAMPLES (DO NOT DO THIS):

❌ WRONG - Only <thinking> without action:
<thinking>
I need to check the weather in Paris using the get_weather tool.
</thinking>

❌ WRONG - No XML tags at all:
Let me check the weather in Paris for you...

❌ WRONG - Missing <answer> tags for final response:
The weather in Paris is 18°C and cloudy.

✅ CORRECT - Proper structure with action after thinking:
<thinking>
I need to check the weather in Paris using the get_weather tool.
</thinking>

<tool_call name="get_weather">
{{"location": "Paris", "units": "celsius"}}
</tool_call>

✅ CORRECT - Proper final answer:
<answer>
The weather in Paris is 18°C and cloudy.
</answer>

IMPORTANT:
- NEVER end your response with only <thinking> - always follow with <tool_call> or <answer>
- If you're unsure what to do, provide your best <answer> rather than stopping at thinking
- Your <answer> will be streamed to the user as you write it
- If you don't use proper XML tags, your response may not be processed correctly"""


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


# Plan and Execute Event Types

class PlanExecuteEventType(Enum):
    """Types of events emitted during Plan and Execute."""

    PLANNING_START = "planning_start"
    PLANNING_THINKING_CHUNK = "planning_thinking_chunk"  # LLM reasoning during planning
    PLANNING_COMPLETE = "planning_complete"
    REPLANNING_START = "replanning_start"
    REPLANNING_COMPLETE = "replanning_complete"

    SUBTASK_START = "subtask_start"
    SUBTASK_THINKING_CHUNK = "subtask_thinking_chunk"  # ReAct reasoning during subtask execution
    SUBTASK_PROGRESS = "subtask_progress"
    SUBTASK_COMPLETE = "subtask_complete"
    SUBTASK_FAILED = "subtask_failed"

    PLAN_PROGRESS = "plan_progress"  # Overall plan progress update
    FINAL_ANSWER = "final_answer"
    FINAL_ANSWER_CHUNK = "final_answer_chunk"  # Streaming chunks for final answer
    ERROR = "error"


@dataclass
class PlanExecuteEvent:
    """Event emitted during Plan and Execute."""

    event_type: PlanExecuteEventType
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_type": self.event_type.value,
            "data": self.data,
            "timestamp": self.timestamp,
        }


# System prompt for Plan and Execute

PLAN_AND_EXECUTE_PLANNING_PROMPT = """You are a planning AI assistant that breaks down complex tasks into structured, executable subtasks.

Your goal: Analyze the user's request and create a detailed execution plan.

CRITICAL: Respond with a JSON plan in this EXACT format:

{{
  "reasoning": "Brief explanation of your planning strategy",
  "subtasks": [
    {{
      "id": 1,
      "description": "Clear, specific description of what to do",
      "required_tools": ["tool1", "tool2"],
      "dependencies": [],
      "success_criteria": "How to know this subtask succeeded"
    }},
    {{
      "id": 2,
      "description": "Next subtask description",
      "required_tools": ["tool3"],
      "dependencies": [1],
      "success_criteria": "Success criteria for this subtask"
    }}
  ]
}}

Available tools:
{tools}

Task Complexity Analysis - Match plan size to task complexity:
- **Simple tasks** (direct lookup/single action): **1 subtask** ← IMPORTANT: Use 1 subtask for simple queries!
- **Straightforward tasks** (single source, simple processing): 2-3 subtasks
- **Moderate tasks** (multiple sources or calculations): 3-5 subtasks
- **Complex tasks** (research + synthesis across sources): 5-8 subtasks
- **Very complex tasks** (multi-stage analysis + synthesis): 7-10 subtasks

CRITICAL: First analyze the task complexity, then create an appropriately-sized plan.
- For simple, single-action tasks → Create a 1-subtask plan (NOT 2!)
- DO NOT default to 5 subtasks - vary the number based on actual complexity!

Planning Guidelines:
1. **Analyze first**: Determine task complexity before planning
2. **Break down appropriately**: Match subtask count to complexity (1-10 subtasks)
3. **Order matters**: Arrange subtasks in logical execution order
4. **Specify dependencies**: Use "dependencies" to indicate which subtasks must complete first
5. **Be specific**: Each subtask should have a clear, actionable description
6. **Use available tools**: Only reference tools from the available tools list
7. **Define success**: Specify concrete success criteria for each subtask
8. **Keep it simple**: Each subtask should be independently executable

Example Plans (showing different complexities):

Example 1 - Simple task (1 subtask):
User: "Find the account for Acme Corp"

{{
  "reasoning": "Direct database lookup - single action required",
  "subtasks": [
    {{
      "id": 1,
      "description": "Search accounts database for 'Acme Corp'",
      "required_tools": ["search_accounts"],
      "dependencies": [],
      "success_criteria": "Account found and details retrieved"
    }}
  ]
}}

Example 2 - Simple task with extraction (2 subtasks):
User: "What's the current temperature in San Francisco?"

{{
  "reasoning": "Simple lookup task - just need to get weather data and extract temperature",
  "subtasks": [
    {{
      "id": 1,
      "description": "Get current weather data for San Francisco",
      "required_tools": ["get_weather"],
      "dependencies": [],
      "success_criteria": "Weather data retrieved with temperature"
    }},
    {{
      "id": 2,
      "description": "Extract and format temperature value",
      "required_tools": [],
      "dependencies": [1],
      "success_criteria": "Temperature value extracted"
    }}
  ]
}}

Example 3 - Moderate task (4 subtasks):
User: "Compare our sales performance this quarter vs last quarter"

{{
  "reasoning": "Need to fetch two datasets, analyze each, then compare - moderate complexity",
  "subtasks": [
    {{
      "id": 1,
      "description": "Fetch sales data for current quarter (Q4 2024)",
      "required_tools": ["query_database"],
      "dependencies": [],
      "success_criteria": "Q4 sales data retrieved"
    }},
    {{
      "id": 2,
      "description": "Fetch sales data for previous quarter (Q3 2024)",
      "required_tools": ["query_database"],
      "dependencies": [],
      "success_criteria": "Q3 sales data retrieved"
    }},
    {{
      "id": 3,
      "description": "Calculate key metrics for both quarters",
      "required_tools": ["calculate"],
      "dependencies": [1, 2],
      "success_criteria": "Metrics calculated for both periods"
    }},
    {{
      "id": 4,
      "description": "Generate comparison analysis with insights",
      "required_tools": ["analyze"],
      "dependencies": [3],
      "success_criteria": "Comparison complete with trends identified"
    }}
  ]
}}

Example 4 - Complex task (7 subtasks):
User: "Research our competitors' pricing strategies and recommend changes to our pricing"

{{
  "reasoning": "Multi-stage research and analysis task requiring data gathering, competitive analysis, and strategic recommendations",
  "subtasks": [
    {{
      "id": 1,
      "description": "Identify top 5 competitors in our market",
      "required_tools": ["search_competitors"],
      "dependencies": [],
      "success_criteria": "List of 5 main competitors identified"
    }},
    {{
      "id": 2,
      "description": "Gather pricing data for each competitor's products",
      "required_tools": ["web_search", "scrape_data"],
      "dependencies": [1],
      "success_criteria": "Pricing data collected for all competitors"
    }},
    {{
      "id": 3,
      "description": "Retrieve our current pricing structure",
      "required_tools": ["query_database"],
      "dependencies": [],
      "success_criteria": "Our pricing data retrieved"
    }},
    {{
      "id": 4,
      "description": "Analyze competitor pricing patterns and strategies",
      "required_tools": ["analyze"],
      "dependencies": [2],
      "success_criteria": "Pricing patterns identified"
    }},
    {{
      "id": 5,
      "description": "Compare our pricing to competitor averages",
      "required_tools": ["calculate", "compare"],
      "dependencies": [3, 4],
      "success_criteria": "Price comparison complete"
    }},
    {{
      "id": 6,
      "description": "Identify pricing gaps and opportunities",
      "required_tools": ["analyze"],
      "dependencies": [5],
      "success_criteria": "Opportunities identified"
    }},
    {{
      "id": 7,
      "description": "Generate pricing recommendations with rationale",
      "required_tools": ["generate_report"],
      "dependencies": [6],
      "success_criteria": "Recommendations documented with justification"
    }}
  ]
}}

IMPORTANT:
- Respond with ONLY the JSON plan, no additional text
- Ensure JSON is valid and properly formatted
- Each subtask ID must be unique and sequential
- Dependencies must reference valid subtask IDs"""


PLAN_AND_EXECUTE_REPLAN_PROMPT = """The current plan has encountered issues and needs replanning.

Original Goal: {goal}

Current Plan Status:
{plan_status}

Failed Subtask: {failed_subtask}
Error: {error}

Your task: Create a revised plan that addresses the failure and completes the goal.

Respond with a new JSON plan in the same format as before:
{{
  "reasoning": "Why the previous plan failed and how this plan fixes it",
  "subtasks": [...]
}}

Guidelines for replanning:
1. **Learn from failure**: Address the specific error that occurred
2. **Keep successful work**: Don't redo subtasks that already completed successfully
3. **Adjust approach**: Try different tools or methods if previous ones failed
4. **Simplify if needed**: Break down failed subtasks into smaller steps
5. **Add validation**: Include verification subtasks if data issues occurred

Respond with ONLY the revised JSON plan."""


# System prompt for planning with tool call (unified pattern with ReAct)
PLANNING_WITH_TOOL_SYSTEM_PROMPT = """You are a planning assistant that analyzes tasks and creates execution plans.

CRITICAL: Structure your response with XML thinking tags, then call the create_plan tool:

<thinking>
Analyze the task complexity and explain your planning strategy:
1. What is the user trying to accomplish?
2. How complex is this task? (simple/moderate/complex)
3. What tools will be needed?
4. What is the logical order of steps?
</thinking>

Then call the create_plan tool with your structured plan.

Available tools for execution:
{tools}

Task Complexity Guidelines:
- **Simple queries** (greetings, thanks, clarifications): Return empty subtasks []
- **Simple tasks** (direct lookup/single action): 1 subtask
- **Straightforward tasks** (single source): 2-3 subtasks
- **Moderate tasks** (multiple sources): 3-5 subtasks
- **Complex tasks** (research + synthesis): 5-8 subtasks

Example for simple task:
<thinking>
The user wants to find information about a specific account. This is a simple lookup task requiring just one database search.
</thinking>

[Call create_plan tool with reasoning="Single lookup task" and subtasks=[{{"id": 1, "description": "Search for account", ...}}]]

Example for greeting:
<thinking>
The user is just saying hello. No planning or tools needed.
</thinking>

[Call create_plan tool with reasoning="Simple greeting - no planning needed" and subtasks=[]]

IMPORTANT:
- Always wrap your analysis in <thinking> tags before calling the tool
- Match plan complexity to task complexity
- Return empty subtasks [] for simple conversational queries"""


# JSON Schema for Plan Structure (used for tool-based planning)
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Brief explanation of the planning strategy and approach"
        },
        "subtasks": {
            "type": "array",
            "description": "List of subtasks to execute in order",
            "items": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "description": "Unique subtask identifier (1, 2, 3, ...)"
                    },
                    "description": {
                        "type": "string",
                        "description": "Clear, specific description of what to do"
                    },
                    "required_tools": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of tools needed for this subtask"
                    },
                    "dependencies": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "IDs of subtasks that must complete before this one"
                    },
                    "success_criteria": {
                        "type": "string",
                        "description": "How to know this subtask succeeded"
                    }
                },
                "required": ["id", "description"],
                "additionalProperties": False
            }
        }
    },
    "required": ["reasoning", "subtasks"],
    "additionalProperties": False
}


def create_plan_tool():
    """Create structured planning tool for combined routing + planning.

    This tool allows the LLM to create a detailed execution plan in a single call,
    combining routing and planning into one step for better performance.

    Returns:
        FunctionTool: Tool that accepts plan parameters and returns plan confirmation
    """
    from miiflow_llm.core.tools import FunctionTool, tool
    from miiflow_llm.core.tools.schemas import ToolSchema, ParameterSchema
    from miiflow_llm.core.tools.types import ParameterType, ToolType
    import logging

    logger = logging.getLogger(__name__)

    # Define explicit schema for the tool to ensure proper parameter types
    explicit_schema = ToolSchema(
        name="create_plan",
        description="""Create execution plan by breaking tasks into subtasks.

ALWAYS call this tool. Match plan complexity to the task:
- **Simple queries** (greetings, thanks, clarifications, simple questions): Return empty subtasks []
- **Direct answers** (single lookup, one tool call): 1 subtask
- **Moderate tasks** (2-3 data sources): 2-5 subtasks
- **Complex tasks** (research + analysis + synthesis): 5-8 subtasks

IMPORTANT: Return [] (empty array) for queries that don't need planning, multi-step execution, or tool usage.

Examples:
- "Hello" → {"reasoning": "Simple greeting", "subtasks": []}
- "Thanks" → {"reasoning": "Acknowledgment", "subtasks": []}
- "Find Acme Corp" → {"reasoning": "Single lookup", "subtasks": [{"id": 1, "description": "Search for Acme Corp", ...}]}""",
        tool_type=ToolType.FUNCTION,  # Required field
        parameters={
            "reasoning": ParameterSchema(
                name="reasoning",
                type=ParameterType.STRING,
                description="Brief explanation of your planning strategy and why this approach is needed",
                required=True
            ),
            "subtasks": ParameterSchema(
                name="subtasks",
                type=ParameterType.ARRAY,
                description="""List of subtasks to execute. Can be empty array [] for simple queries that don't need planning.

For non-empty plans, each subtask should have:
- id (int): Unique identifier (1, 2, 3, ...)
- description (str): Clear, specific description of what to do
- required_tools (array of strings): Tools needed for this subtask
- dependencies (array of ints): IDs of subtasks that must complete first
- success_criteria (str): How to verify this subtask succeeded

Return [] for greetings, acknowledgments, and simple conversational queries.""",
                required=True
            )
        }
    )

    def create_plan(reasoning: str, subtasks: list) -> dict:
        """Internal function for plan creation."""
        logger.info(f"create_plan tool called! Reasoning: {reasoning[:100]}..., Subtask count: {len(subtasks)}")
        return {
            "plan_created": True,
            "reasoning": reasoning,
            "subtasks": subtasks,
            "subtask_count": len(subtasks)
        }

    # Create tool with explicit schema
    tool = FunctionTool(create_plan)
    tool.definition = explicit_schema  # Override with explicit schema

    logger.info(f"Created planning tool with schema: {explicit_schema.name}")
    return tool
