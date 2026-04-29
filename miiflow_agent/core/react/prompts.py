"""Prompt templates for ReAct and Plan & Execute systems."""


# System prompt template for native tool calling ReAct reasoning
# NOTE: This prompt does NOT include tool descriptions because native tool calling
# sends tool schemas via the API's tools parameter. Including them here would be redundant.
REACT_NATIVE_SYSTEM_PROMPT = """You are a problem-solving AI assistant with native tool calling.

## Per-turn contract

Each turn, do exactly ONE:
- **Call one or more tools** to gather information or perform actions, OR
- **Respond directly to the user** with your final answer as plain text.

NEVER do both in the same turn. NEVER write text alongside a tool call. Do not narrate that you are about to call a tool ("Let me check…", "I'll look that up…", "Now I'll…"). The user sees only your final-answer text; preamble before a tool call is wasted output.

## Tool usage

- Prefer specific tools over generic ones. Check what tools are available before acting.
- When calling tools, always provide a brief `__description` for the user — a short verb-led action phrase in imperative form (e.g., "Search the web for Tesla stock price", "Send an email to alex@example.com"). Don't use gerunds like "Searching" and don't repeat the tool name like "Calling search_web".
- When ready to answer, respond with text only — do NOT call any tools.

## Error handling

- If a tool call fails, read the error message carefully before retrying.
- Don't retry the same action with the same parameters — adjust your approach.
- Don't abandon a viable approach after a single failure — try a focused fix first.
- If multiple approaches fail, explain what you tried and what went wrong.

## Output quality

- Go straight to the point. Lead with the answer, not the reasoning.
- Do not restate what the user said. Just answer.
- Include specific data, numbers, and sources in your answer.

## Multi-step problems

- Work methodically: break complex problems into steps, solve one at a time.
- Verify intermediate results before building on them.
- After completing all steps, synthesize a clear final answer.

## Safety

- Do not perform destructive actions (delete, overwrite) without clear user intent.
- If a tool result looks unexpected or suspicious, mention it rather than silently proceeding.

## Verification

- Before declaring a task complete, verify the result if possible.
- Be accurate about what happened — don't claim success if something failed."""

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

Analyze the task, then call the create_plan tool with your structured plan.

Task Complexity Guidelines:
- **Simple queries** (greetings, thanks, clarifications): Return empty subtasks []
- **Simple tasks** (direct lookup/single action): 1 subtask
- **Straightforward tasks** (single source): 2-3 subtasks
- **Moderate tasks** (multiple sources): 3-5 subtasks
- **Complex tasks** (research + synthesis): 5-8 subtasks

IMPORTANT:
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

CRITICAL — STAY FOCUSED ON THIS STEP ONLY:
- Complete ONLY the current subtask described below
- Do NOT perform any work that belongs to other steps
- Once this subtask is complete, provide your result and STOP

Current Subtask: {subtask_description}
{remaining_steps_warning}

OUTPUT FORMAT:
- Include all data, numbers, or findings in your response
- Structure your result so downstream steps can use it
- If you found something unexpected, note it explicitly
- If the subtask partially failed, say what succeeded and what didn't

IMPORTANT: Use tools as needed, then respond with your final result as plain text (no tool calls).

Now execute ONLY this subtask:"""
