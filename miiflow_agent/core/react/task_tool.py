"""TaskTool for hierarchical agent spawning.

This module provides the Task tool that allows agents to spawn specialized
subagents during execution, enabling hierarchical delegation patterns.

Key features:
- Spawn subagents from within an agent using the Task tool
- Nesting depth limits to prevent infinite recursion
- Context isolation between parent and child agents
- Subagent result aggregation
- Support for all registered subagent types

Based on patterns from:
- Claude Agent SDK's Task tool
- Google ADK's agent delegation
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from .subagent_registry import DynamicSubAgentConfig, SubAgentRegistry, get_global_registry
from .model_selector import ModelSelector

if TYPE_CHECKING:
    from ..agent import RunContext
    from .orchestrator import ReActOrchestrator

logger = logging.getLogger(__name__)


# Maximum nesting depth to prevent infinite recursion
MAX_NESTING_DEPTH = 3

# Default timeout for subagent execution
DEFAULT_SUBAGENT_TIMEOUT = 120.0


@dataclass
class TaskToolResult:
    """Result from a Task tool invocation."""
    subagent_name: str
    subagent_type: str
    success: bool
    result: Optional[str] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    tokens_used: int = 0
    cost: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "subagent_name": self.subagent_name,
            "subagent_type": self.subagent_type,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "execution_time": self.execution_time,
            "tokens_used": self.tokens_used,
            "cost": self.cost,
        }


@dataclass
class TaskToolConfig:
    """Configuration for the TaskTool."""
    max_nesting_depth: int = MAX_NESTING_DEPTH
    default_timeout: float = DEFAULT_SUBAGENT_TIMEOUT
    allow_parallel: bool = True  # Allow multiple subagent calls in parallel


class TaskTool:
    """Tool for spawning specialized subagents from within an agent.

    The TaskTool enables hierarchical agent patterns where a parent agent
    can delegate subtasks to specialized child agents. This is the core
    mechanism for multi-agent orchestration.

    Features:
    - Spawn any registered subagent type
    - Automatic model selection based on subagent config
    - Context isolation (child agents don't see parent context)
    - Nesting depth limits to prevent infinite recursion
    - Timeout protection for child agent execution

    Usage:
        # Register the tool with an agent
        task_tool = TaskTool(
            registry=SubAgentRegistry(),
            orchestrator_factory=create_orchestrator,
        )

        # The agent can then call it like:
        # Task(subagent_type="explorer", prompt="Find files containing 'error'")

    Tool Schema:
        name: Task
        description: Delegate a subtask to a specialized subagent
        parameters:
            subagent_type: string (required) - Type of subagent to use
            prompt: string (required) - The task for the subagent
            description: string (optional) - Brief description of what the subagent will do
    """

    name: str = "Task"
    description: str = """Delegate a subtask to a specialized subagent.

Available subagent types:
- explorer: Find files, search code, understand codebase structure
- researcher: Search the web, gather information
- implementer: Write and modify code
- reviewer: Review code for bugs and issues
- planner: Plan implementation approaches

Use this tool when:
- A subtask requires specialized expertise
- You want to parallelize work
- The task would benefit from focused attention

The subagent will work independently and return its result."""

    def __init__(
        self,
        registry: Optional[SubAgentRegistry] = None,
        orchestrator_factory: Optional[Callable] = None,
        model_selector: Optional[ModelSelector] = None,
        config: Optional[TaskToolConfig] = None,
    ):
        """Initialize the TaskTool.

        Args:
            registry: SubAgent registry (uses global if not provided)
            orchestrator_factory: Factory function to create orchestrators for subagents
            model_selector: Model selector (uses default if not provided)
            config: TaskTool configuration
        """
        self.registry = registry or get_global_registry()
        self.orchestrator_factory = orchestrator_factory
        self.model_selector = model_selector or ModelSelector()
        self.config = config or TaskToolConfig()

        # Track active subagent executions
        self._active_subagents: Dict[str, asyncio.Task] = {}

    @property
    def parameters(self) -> Dict[str, Any]:
        """Get the JSON schema for tool parameters."""
        available_types = ", ".join(self.registry.names)

        return {
            "type": "object",
            "properties": {
                "subagent_type": {
                    "type": "string",
                    "description": f"Type of subagent to use. Available: {available_types}",
                    "enum": self.registry.names,
                },
                "prompt": {
                    "type": "string",
                    "description": "The task/query for the subagent to work on",
                },
                "description": {
                    "type": "string",
                    "description": "Optional brief description of what this subagent will do (3-5 words)",
                },
            },
            "required": ["subagent_type", "prompt"],
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to tool definition dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    async def execute(
        self,
        subagent_type: str,
        prompt: str,
        context: "RunContext",
        description: Optional[str] = None,
    ) -> str:
        """Execute a subagent and return its result.

        Args:
            subagent_type: Type of subagent from registry
            prompt: Task for the subagent
            context: Parent agent's run context
            description: Optional description (for logging)

        Returns:
            Subagent's result as a string
        """
        start_time = time.time()

        # Get subagent config
        config = self.registry.get(subagent_type)
        if not config:
            available = ", ".join(self.registry.names)
            error_msg = f"Unknown subagent type: {subagent_type}. Available: {available}"
            logger.warning(error_msg)
            return f"Error: {error_msg}"

        # Check nesting depth
        current_depth = getattr(context, "nesting_depth", 0)
        if current_depth >= self.config.max_nesting_depth:
            error_msg = f"Maximum nesting depth ({self.config.max_nesting_depth}) exceeded"
            logger.warning(error_msg)
            return f"Error: {error_msg}. Cannot spawn more subagents."

        # Create subagent config with nesting disabled if at max depth - 1
        effective_config = config
        if current_depth >= self.config.max_nesting_depth - 1:
            effective_config = config.copy(can_spawn_subagents=False)

        logger.info(
            f"Spawning subagent: {subagent_type} (depth={current_depth + 1})",
            extra={"prompt_preview": prompt[:100]},
        )

        try:
            result = await self._execute_subagent(
                config=effective_config,
                prompt=prompt,
                parent_context=context,
                nesting_depth=current_depth + 1,
            )

            execution_time = time.time() - start_time
            logger.info(
                f"Subagent {subagent_type} completed in {execution_time:.2f}s",
                extra={
                    "success": result.success,
                    "result_preview": (result.result or "")[:100],
                },
            )

            if result.success:
                return result.result or "Subagent completed but returned no result."
            else:
                return f"Subagent failed: {result.error}"

        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            error_msg = f"Subagent {subagent_type} timed out after {execution_time:.2f}s"
            logger.warning(error_msg)
            return f"Error: {error_msg}"

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Subagent {subagent_type} failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"Error: {error_msg}"

    async def _execute_subagent(
        self,
        config: DynamicSubAgentConfig,
        prompt: str,
        parent_context: "RunContext",
        nesting_depth: int,
    ) -> TaskToolResult:
        """Execute a subagent with the given configuration.

        Args:
            config: Subagent configuration
            prompt: Task for the subagent
            parent_context: Parent's run context
            nesting_depth: Current nesting depth

        Returns:
            TaskToolResult with execution results
        """
        start_time = time.time()

        if not self.orchestrator_factory:
            return TaskToolResult(
                subagent_name=f"subagent_{config.name}",
                subagent_type=config.name,
                success=False,
                error="No orchestrator factory configured. Cannot spawn subagents.",
            )

        try:
            # Create subagent context with nesting info
            subagent_context = self._create_subagent_context(
                parent_context=parent_context,
                config=config,
                nesting_depth=nesting_depth,
            )

            # Build the full subagent prompt
            full_prompt = self._build_subagent_prompt(config, prompt)

            # Create orchestrator for subagent
            orchestrator = self.orchestrator_factory(
                model=config.model,
                tools=config.tools,
                system_prompt=config.system_prompt,
                max_steps=config.max_steps,
            )

            # Execute with timeout
            result = await asyncio.wait_for(
                orchestrator.execute(full_prompt, subagent_context),
                timeout=config.timeout_seconds,
            )

            execution_time = time.time() - start_time

            return TaskToolResult(
                subagent_name=f"subagent_{config.name}",
                subagent_type=config.name,
                success=True,
                result=result.final_answer,
                execution_time=execution_time,
                tokens_used=result.total_tokens,
                cost=result.total_cost,
            )

        except asyncio.TimeoutError:
            return TaskToolResult(
                subagent_name=f"subagent_{config.name}",
                subagent_type=config.name,
                success=False,
                error=f"Subagent timed out after {config.timeout_seconds}s",
                execution_time=time.time() - start_time,
            )

        except Exception as e:
            return TaskToolResult(
                subagent_name=f"subagent_{config.name}",
                subagent_type=config.name,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time,
            )

    def _create_subagent_context(
        self,
        parent_context: "RunContext",
        config: DynamicSubAgentConfig,
        nesting_depth: int,
    ) -> "RunContext":
        """Create an isolated context for subagent execution.

        The subagent gets a fresh context to prevent context pollution
        while still inheriting necessary runtime settings.

        Args:
            parent_context: Parent's context
            config: Subagent configuration
            nesting_depth: Current nesting depth

        Returns:
            New RunContext for the subagent
        """
        # Import here to avoid circular imports
        from ..agent import RunContext

        # Create fresh context with inherited settings
        subagent_context = RunContext(
            max_steps=config.max_steps,
            cost_budget=parent_context.cost_budget / 2 if parent_context.cost_budget else None,
            time_budget=config.timeout_seconds,
            streaming=parent_context.streaming,
            messages=[],  # Fresh message history
        )

        # Add nesting depth tracking
        subagent_context.nesting_depth = nesting_depth

        # Mark as subagent context
        subagent_context.metadata = {
            "is_subagent": True,
            "parent_agent_type": getattr(parent_context, "agent_type", "unknown"),
            "subagent_type": config.name,
            "nesting_depth": nesting_depth,
        }

        return subagent_context

    def _build_subagent_prompt(
        self,
        config: DynamicSubAgentConfig,
        task_prompt: str,
    ) -> str:
        """Build the full prompt for subagent execution.

        Args:
            config: Subagent configuration
            task_prompt: The specific task prompt

        Returns:
            Full prompt for the subagent
        """
        parts = []

        # Add role context
        parts.append(f"You are a specialized {config.name} agent.")

        # Add focus area
        if config.description:
            parts.append(f"Your expertise: {config.description}")

        # Add the task
        parts.append(f"\n## Task\n{task_prompt}")

        # Add constraints
        parts.append("\n## Guidelines")
        parts.append("- Stay focused on your assigned task")
        parts.append("- Be thorough but efficient")
        parts.append("- Report your findings clearly")

        if config.tools:
            tools_str = ", ".join(config.tools)
            parts.append(f"- You have access to these tools: {tools_str}")

        return "\n".join(parts)

    async def execute_parallel(
        self,
        tasks: List[Dict[str, str]],
        context: "RunContext",
    ) -> List[TaskToolResult]:
        """Execute multiple subagent tasks in parallel.

        Args:
            tasks: List of task dicts with subagent_type and prompt
            context: Parent run context

        Returns:
            List of TaskToolResults
        """
        if not self.config.allow_parallel:
            # Execute sequentially
            results = []
            for task in tasks:
                result_str = await self.execute(
                    subagent_type=task["subagent_type"],
                    prompt=task["prompt"],
                    context=context,
                    description=task.get("description"),
                )
                results.append(TaskToolResult(
                    subagent_name=f"subagent_{task['subagent_type']}",
                    subagent_type=task["subagent_type"],
                    success="Error:" not in result_str,
                    result=result_str if "Error:" not in result_str else None,
                    error=result_str if "Error:" in result_str else None,
                ))
            return results

        # Execute in parallel
        async def execute_task(task: Dict[str, str]) -> TaskToolResult:
            result_str = await self.execute(
                subagent_type=task["subagent_type"],
                prompt=task["prompt"],
                context=context,
                description=task.get("description"),
            )
            return TaskToolResult(
                subagent_name=f"subagent_{task['subagent_type']}",
                subagent_type=task["subagent_type"],
                success="Error:" not in result_str,
                result=result_str if "Error:" not in result_str else None,
                error=result_str if "Error:" in result_str else None,
            )

        results = await asyncio.gather(
            *[execute_task(task) for task in tasks],
            return_exceptions=True,
        )

        # Convert exceptions to TaskToolResults
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(TaskToolResult(
                    subagent_name=f"subagent_{tasks[i]['subagent_type']}",
                    subagent_type=tasks[i]["subagent_type"],
                    success=False,
                    error=str(result),
                ))
            else:
                processed_results.append(result)

        return processed_results


def create_task_tool(
    registry: Optional[SubAgentRegistry] = None,
    orchestrator_factory: Optional[Callable] = None,
) -> TaskTool:
    """Factory function to create a TaskTool.

    Args:
        registry: SubAgent registry (uses global if not provided)
        orchestrator_factory: Factory to create orchestrators

    Returns:
        Configured TaskTool instance
    """
    return TaskTool(
        registry=registry,
        orchestrator_factory=orchestrator_factory,
    )
