"""Prompt templates for ReAct and Plan & Execute systems."""


# System prompt template for native tool calling ReAct reasoning
# NOTE: This prompt does NOT include tool descriptions because native tool calling
# sends tool schemas via the API's tools parameter. Including them here would be redundant.
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

PLAN_AND_EXECUTE_REPLAN_PROMPT = """The current plan has encountered issues and needs replanning.

Original Goal: {goal}

Current Plan Status:
{plan_status}

Failed Subtask: {failed_subtask}
Error: {error}
{completed_context}
Your task: Create a revised plan that addresses the failure and completes the goal.

Respond with a new JSON plan in the same format as before:
{{
  "reasoning": "Why the previous plan failed and how this plan fixes it",
  "subtasks": [...]
}}

Guidelines for replanning:
1. **Learn from failure**: Address the specific error that occurred
2. **Build on completed work**: Use results from completed subtasks (shown above) without re-executing them
3. **Adjust approach**: Try different tools or methods if previous ones failed
4. **Simplify if needed**: Break down failed subtasks into smaller steps
5. **Add validation**: Include verification subtasks if data issues occurred

Respond with ONLY the revised JSON plan."""


# System prompt for planning with tool call (unified pattern with ReAct)
# NOTE: This prompt does NOT include tool descriptions because native tool calling
# sends tool schemas via the API's tools parameter. Including them here would be redundant.
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


def create_plan_tool():
    """Create structured planning tool for combined routing + planning.

    This tool allows the LLM to create a detailed execution plan in a single call,
    combining routing and planning into one step for better performance.

    Returns:
        FunctionTool: Tool that accepts plan parameters and returns plan confirmation
    """
    from miiflow_agent.core.tools import FunctionTool
    from miiflow_agent.core.tools.schemas import ToolSchema, ParameterSchema
    from miiflow_agent.core.tools.types import ParameterType, ToolType
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
                required=True,
                items={
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "integer",
                            "description": "Unique identifier for the subtask (1, 2, 3, ...)"
                        },
                        "description": {
                            "type": "string",
                            "description": "Clear, specific description of what to do"
                        },
                        "required_tools": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tools needed for this subtask"
                        },
                        "dependencies": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "IDs of subtasks that must complete first"
                        },
                        "success_criteria": {
                            "type": "string",
                            "description": "How to verify this subtask succeeded"
                        }
                    },
                    "required": ["id", "description"]
                }
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


# Prompt for scoped subtask execution in Plan & Execute mode
# Used to constrain the ReAct agent to focus only on the current step
SUBTASK_EXECUTION_PROMPT = """You are executing Step {subtask_number} of {total_subtasks} in a multi-step plan.

CRITICAL - STAY FOCUSED ON THIS STEP ONLY:
- Complete ONLY the current subtask described below
- Do NOT perform any work that belongs to other steps
- Once this subtask is complete, provide your result and STOP

Current Subtask: {subtask_description}
{remaining_steps_warning}

Now execute ONLY this subtask:"""
