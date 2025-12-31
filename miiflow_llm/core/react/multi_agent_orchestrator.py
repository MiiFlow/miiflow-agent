"""Multi-Agent orchestrator for parallel subagent execution."""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from ..message import Message, MessageRole
from .enums import MultiAgentEventType, StopReason
from .events import EventBus
from .models import MultiAgentResult, SubAgentConfig, SubAgentPlan, SubAgentResult
from .orchestrator import ReActOrchestrator
from .react_events import MultiAgentEvent
from .safety import SafetyManager
from .shared_state import SharedAgentState
from .tool_executor import AgentToolExecutor

if TYPE_CHECKING:
    from ..agent import Agent, RunContext

logger = logging.getLogger(__name__)


# Prompt for lead agent to plan subagent allocation
SUBAGENT_PLANNING_PROMPT = """You are a lead agent coordinating a team of specialized subagents.
Your task is to analyze the user's query and determine how to best allocate work to subagents.

User Query: {query}

Available Tools: {available_tools}

Analyze the query and create a plan for subagent allocation. Each subagent should:
1. Have a specific focus area or role
2. Be assigned a portion of the overall task
3. Work independently (no dependencies between subagents)

Respond with a JSON object in this exact format:
{{
    "reasoning": "Explanation of your allocation strategy",
    "subagents": [
        {{
            "name": "unique_name_for_this_agent",
            "role": "researcher|analyzer|coder|summarizer|custom_role",
            "focus": "Specific aspect this agent should focus on",
            "query": "The specific query/task for this subagent"
        }}
    ]
}}

Guidelines:
- Use 1-5 subagents based on query complexity
- Simple queries: 1-2 subagents
- Complex multi-faceted queries: 3-5 subagents
- Each subagent should have a distinct, non-overlapping focus
- Keep subagent queries focused and actionable
"""

SYNTHESIS_PROMPT = """You are synthesizing results from multiple specialized agents into a final answer.

Original Query: {query}

Subagent Results:
{subagent_results}

Based on these results, provide a comprehensive, well-structured answer that:
1. Addresses the original query directly
2. Integrates insights from all subagents
3. Resolves any conflicts between subagent findings
4. Is clear and actionable for the user
"""


class MultiAgentOrchestrator:
    """Orchestrate multiple specialized subagents working in parallel.

    This orchestrator implements the orchestrator-worker pattern where a lead agent
    coordinates multiple specialized subagents operating in parallel.

    Key features:
    - Lead agent plans subagent allocation based on query
    - Subagents execute in parallel using asyncio.gather
    - Shared state for collecting results (thread-safe)
    - Synchronous coordination (waits for all to complete)
    - All subagents share the same tool registry

    Based on patterns from:
    - Anthropic's multi-agent research system (90% latency reduction)
    - Google ADK's parallel fan-out/gather pattern
    - LangGraph's supervisor pattern

    Usage:
        orchestrator = MultiAgentOrchestrator(
            tool_executor=tool_executor,
            event_bus=event_bus,
            safety_manager=safety_manager,
            subagent_orchestrator=react_orchestrator,
        )
        result = await orchestrator.execute(query, context)
    """

    def __init__(
        self,
        tool_executor: AgentToolExecutor,
        event_bus: EventBus,
        safety_manager: SafetyManager,
        subagent_orchestrator: Optional[ReActOrchestrator] = None,
        max_subagents: int = 5,
        subagent_timeout_seconds: float = 120.0,
    ):
        """Initialize Multi-Agent orchestrator.

        Args:
            tool_executor: Tool execution adapter
            event_bus: Event bus for streaming events
            safety_manager: Safety condition checker
            subagent_orchestrator: ReAct orchestrator for subagent execution
            max_subagents: Maximum number of subagents to spawn (1-5)
            subagent_timeout_seconds: Timeout for each subagent execution
        """
        self.tool_executor = tool_executor
        self.event_bus = event_bus
        self.safety_manager = safety_manager
        self.subagent_orchestrator = subagent_orchestrator
        self.max_subagents = min(max_subagents, 5)  # Cap at 5
        self.subagent_timeout_seconds = subagent_timeout_seconds

        # Shared state for subagent results
        self.shared_state = SharedAgentState()

    async def execute(
        self, query: str, context: "RunContext", existing_plan: Optional[SubAgentPlan] = None
    ) -> MultiAgentResult:
        """Execute multi-agent workflow.

        Args:
            query: User's goal/query
            context: Run context with messages and state
            existing_plan: Optional pre-generated subagent plan

        Returns:
            MultiAgentResult with subagent results and final answer
        """
        start_time = time.time()

        try:
            # Phase 1: Plan subagent allocation
            await self._publish_event(
                MultiAgentEventType.PLANNING_START,
                {"query": query},
            )

            if existing_plan:
                plan = existing_plan
            else:
                plan = await self._plan_subagents(query, context)

            await self._publish_event(
                MultiAgentEventType.PLANNING_COMPLETE,
                {
                    "reasoning": plan.reasoning,
                    "subagent_count": len(plan.subagent_configs),
                    "subagents": [cfg.to_dict() for cfg in plan.subagent_configs],
                },
            )

            # Phase 2: Spawn and execute subagents in parallel
            await self._publish_event(
                MultiAgentEventType.EXECUTION_START,
                {"subagent_count": len(plan.subagent_configs)},
            )

            subagent_results = await self._execute_subagents(plan, query, context)

            # Phase 3: Synthesize results
            await self._publish_event(
                MultiAgentEventType.SYNTHESIS_START,
                {
                    "completed_subagents": len([r for r in subagent_results if r.success]),
                    "failed_subagents": len([r for r in subagent_results if not r.success]),
                },
            )

            final_answer = await self._synthesize_results(query, subagent_results, context)

            # Calculate totals
            total_time = time.time() - start_time
            total_cost = sum(r.cost for r in subagent_results)
            total_tokens = sum(r.tokens_used for r in subagent_results)

            # Determine stop reason
            failed_count = len([r for r in subagent_results if not r.success])
            if failed_count == 0:
                stop_reason = StopReason.ANSWER_COMPLETE
            elif failed_count < len(subagent_results):
                stop_reason = StopReason.ANSWER_COMPLETE  # Partial success
            else:
                stop_reason = StopReason.FORCED_STOP

            result = MultiAgentResult(
                subagent_results=subagent_results,
                final_answer=final_answer,
                stop_reason=stop_reason,
                total_cost=total_cost,
                total_execution_time=total_time,
                total_tokens=total_tokens,
            )

            # Emit final answer event
            await self._publish_event(
                MultiAgentEventType.FINAL_ANSWER,
                {"answer": final_answer, "result": result.to_dict()},
            )

            return result

        except Exception as e:
            logger.error(f"Multi-agent execution failed: {e}", exc_info=True)
            return MultiAgentResult(
                subagent_results=[],
                final_answer=f"Error occurred during multi-agent execution: {str(e)}",
                stop_reason=StopReason.FORCED_STOP,
            )

    async def _plan_subagents(
        self, query: str, context: "RunContext"
    ) -> SubAgentPlan:
        """Plan subagent allocation using lead agent.

        Args:
            query: User's query
            context: Run context

        Returns:
            SubAgentPlan with subagent configurations
        """
        # Get available tool names
        tool_names = list(self.tool_executor.available_tools.keys()) if hasattr(self.tool_executor, 'available_tools') else []

        # Build planning prompt
        planning_prompt = SUBAGENT_PLANNING_PROMPT.format(
            query=query,
            available_tools=", ".join(tool_names) if tool_names else "general tools",
        )

        messages = [Message(role=MessageRole.USER, content=planning_prompt)]

        # Stream planning for user feedback
        accumulated_content = ""
        async for chunk in self.tool_executor.stream_without_tools(
            messages=messages, temperature=0.3
        ):
            if chunk.delta:
                accumulated_content += chunk.delta
                await self._publish_event(
                    MultiAgentEventType.PLANNING_THINKING_CHUNK,
                    {"delta": chunk.delta, "content": accumulated_content},
                )

        # Parse subagent plan from response
        plan = self._parse_subagent_plan(accumulated_content, query)

        logger.info(f"Planned {len(plan.subagent_configs)} subagents: {plan.reasoning}")

        return plan

    def _parse_subagent_plan(self, response: str, query: str) -> SubAgentPlan:
        """Parse subagent plan from LLM response.

        Args:
            response: LLM response text
            query: Original query for fallback

        Returns:
            SubAgentPlan with parsed subagent configurations
        """
        try:
            # Extract JSON from response
            json_str = response.strip()

            # Try to extract from markdown code blocks
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
            else:
                # Find JSON object
                brace_idx = json_str.find("{")
                if brace_idx != -1:
                    json_str = json_str[brace_idx:]
                    brace_count = 0
                    for i, char in enumerate(json_str):
                        if char == "{":
                            brace_count += 1
                        elif char == "}":
                            brace_count -= 1
                            if brace_count == 0:
                                json_str = json_str[:i + 1]
                                break

            plan_data = json.loads(json_str.strip())

            # Convert to SubAgentConfig objects
            configs = []
            for i, agent_data in enumerate(plan_data.get("subagents", [])[:self.max_subagents]):
                config = SubAgentConfig(
                    name=agent_data.get("name", f"subagent_{i}"),
                    role=agent_data.get("role", "general"),
                    focus=agent_data.get("focus", "general task"),
                    query=agent_data.get("query", query),
                    output_key=f"result_{agent_data.get('name', f'subagent_{i}')}",
                )
                configs.append(config)

            return SubAgentPlan(
                reasoning=plan_data.get("reasoning", "Plan created"),
                subagent_configs=configs,
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse subagent plan: {e}, using fallback")
            # Fallback: single subagent for the whole task
            return SubAgentPlan(
                reasoning=f"Fallback plan (parse error: {e})",
                subagent_configs=[
                    SubAgentConfig(
                        name="primary_agent",
                        role="general",
                        focus="Complete the task",
                        query=query,
                        output_key="result_primary",
                    )
                ],
            )

    async def _execute_subagents(
        self,
        plan: SubAgentPlan,
        query: str,
        context: "RunContext",
    ) -> List[SubAgentResult]:
        """Execute all subagents in parallel.

        Args:
            plan: Subagent plan with configurations
            query: Original query
            context: Run context

        Returns:
            List of SubAgentResult objects
        """
        if not plan.subagent_configs:
            return []

        # Clear shared state for this run
        await self.shared_state.clear()

        async def execute_single_subagent(config: SubAgentConfig) -> SubAgentResult:
            """Execute a single subagent with timeout."""
            await self._publish_event(
                MultiAgentEventType.SUBAGENT_START,
                {
                    "name": config.name,
                    "role": config.role,
                    "focus": config.focus,
                    "query": config.query,
                },
            )

            start_time = time.time()

            try:
                if not self.subagent_orchestrator:
                    raise ValueError("ReAct orchestrator required for subagent execution")

                # Build subagent-specific context
                subagent_query = f"""You are a specialized agent with role: {config.role}
Your focus area: {config.focus}

Task: {config.query}

Complete this specific task. Stay focused on your assigned area."""

                # Execute with ReAct orchestrator (with timeout)
                result = await asyncio.wait_for(
                    self.subagent_orchestrator.execute(subagent_query, context),
                    timeout=self.subagent_timeout_seconds,
                )

                execution_time = time.time() - start_time

                # Write result to shared state
                await self.shared_state.write(
                    config.output_key,
                    {
                        "answer": result.final_answer,
                        "role": config.role,
                        "focus": config.focus,
                    },
                    agent_id=config.name,
                )

                subagent_result = SubAgentResult(
                    agent_name=config.name,
                    role=config.role,
                    output_key=config.output_key,
                    result=result.final_answer,
                    success=True,
                    execution_time=execution_time,
                    tokens_used=result.total_tokens,
                    cost=result.total_cost,
                )

                await self._publish_event(
                    MultiAgentEventType.SUBAGENT_COMPLETE,
                    {
                        "name": config.name,
                        "success": True,
                        "result_preview": result.final_answer[:200] if result.final_answer else "",
                        "execution_time": execution_time,
                    },
                )

                return subagent_result

            except asyncio.TimeoutError:
                execution_time = time.time() - start_time
                error_msg = f"Subagent timed out after {self.subagent_timeout_seconds}s"
                logger.warning(f"{config.name}: {error_msg}")

                await self._publish_event(
                    MultiAgentEventType.SUBAGENT_FAILED,
                    {
                        "name": config.name,
                        "error": error_msg,
                        "timeout": True,
                    },
                )

                return SubAgentResult(
                    agent_name=config.name,
                    role=config.role,
                    output_key=config.output_key,
                    result=None,
                    success=False,
                    error=error_msg,
                    execution_time=execution_time,
                )

            except Exception as e:
                execution_time = time.time() - start_time
                error_msg = str(e)
                logger.error(f"{config.name} failed: {error_msg}", exc_info=True)

                await self._publish_event(
                    MultiAgentEventType.SUBAGENT_FAILED,
                    {
                        "name": config.name,
                        "error": error_msg,
                    },
                )

                return SubAgentResult(
                    agent_name=config.name,
                    role=config.role,
                    output_key=config.output_key,
                    result=None,
                    success=False,
                    error=error_msg,
                    execution_time=execution_time,
                )

        # Execute all subagents in parallel
        logger.info(f"Executing {len(plan.subagent_configs)} subagents in parallel")
        results = await asyncio.gather(
            *[execute_single_subagent(config) for config in plan.subagent_configs],
            return_exceptions=False,  # Exceptions handled in execute_single_subagent
        )

        return list(results)

    async def _synthesize_results(
        self,
        query: str,
        subagent_results: List[SubAgentResult],
        context: "RunContext",
    ) -> str:
        """Synthesize subagent results into final answer.

        Args:
            query: Original user query
            subagent_results: List of subagent results
            context: Run context

        Returns:
            Final synthesized answer
        """
        # Format subagent results for synthesis
        results_text = []
        for result in subagent_results:
            if result.success and result.result:
                results_text.append(
                    f"**{result.agent_name}** ({result.role}):\n{result.result}\n"
                )
            elif not result.success:
                results_text.append(
                    f"**{result.agent_name}** ({result.role}): Failed - {result.error}\n"
                )

        if not results_text:
            return "No subagents completed successfully. Unable to provide an answer."

        # Build synthesis prompt
        synthesis_prompt = SYNTHESIS_PROMPT.format(
            query=query,
            subagent_results="\n".join(results_text),
        )

        messages = [Message(role=MessageRole.USER, content=synthesis_prompt)]

        # Stream synthesis
        final_answer = ""
        async for chunk in self.tool_executor.stream_without_tools(
            messages=messages, temperature=0.5
        ):
            if chunk.delta:
                final_answer += chunk.delta
                await self._publish_event(
                    MultiAgentEventType.FINAL_ANSWER_CHUNK,
                    {"delta": chunk.delta, "content": final_answer},
                )

        return final_answer

    async def _publish_event(
        self, event_type: MultiAgentEventType, data: Dict[str, Any]
    ) -> None:
        """Publish a multi-agent event.

        Args:
            event_type: Type of multi-agent event
            data: Event data
        """
        event = MultiAgentEvent(event_type=event_type, data=data)
        await self.event_bus.publish(event)

    def get_current_status(self) -> Dict[str, Any]:
        """Get current execution status."""
        return {
            "agent_type": "multi_agent_orchestrator",
            "max_subagents": self.max_subagents,
            "shared_state_keys": self.shared_state.keys,
        }
