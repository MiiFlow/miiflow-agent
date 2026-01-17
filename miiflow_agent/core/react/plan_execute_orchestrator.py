"""Plan and Execute orchestrator for complex multi-step tasks."""

import ast
import asyncio
import copy
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Set

from ..agent import RunContext
from ..message import Message, MessageRole
from .enums import ParallelPlanEventType, PlanExecuteEventType, ReActEventType, StopReason
from .models import ExecutionWave, Plan, PlanExecuteResult, SubTask
from .react_events import ParallelPlanEvent
from .prompts import PLAN_AND_EXECUTE_REPLAN_PROMPT, SUBTASK_EXECUTION_PROMPT
from .react_events import PlanExecuteEvent
from .events import EventBus
from .orchestrator import ReActOrchestrator
from .safety import SafetyManager
from .tool_executor import AgentToolExecutor

logger = logging.getLogger(__name__)


class PlanAndExecuteOrchestrator:
    """Plan and Execute orchestrator with composable ReAct execution.

    This orchestrator breaks down complex tasks into structured plans with subtasks,
    then executes each subtask (optionally using ReAct for complex subtasks).

    Workflow:
    1. Planning Phase: Generate structured plan with subtasks
    2. Execution Phase: Execute subtasks in dependency order
    3. Re-planning Phase: Adapt plan if subtasks fail
    4. Synthesis Phase: Combine results into final answer
    """

    def __init__(
        self,
        tool_executor: AgentToolExecutor,
        event_bus: EventBus,
        safety_manager: SafetyManager,
        subtask_orchestrator: Optional[ReActOrchestrator] = None,
        max_replans: int = 2,
        subtask_timeout_seconds: float = 120.0,
        parallel_execution: bool = False,
        max_parallel_subtasks: int = 5,
    ):
        """Initialize Plan and Execute orchestrator.

        Args:
            tool_executor: Tool execution adapter
            event_bus: Event bus for streaming events
            safety_manager: Safety condition checker
            subtask_orchestrator: ReAct orchestrator for subtask execution (required)
            max_replans: Maximum number of re-planning attempts
            subtask_timeout_seconds: Timeout for each subtask execution (default 120s)
            parallel_execution: Whether to execute independent subtasks in parallel waves
            max_parallel_subtasks: Maximum subtasks to run in parallel per wave (when parallel_execution=True)
        """
        self.tool_executor = tool_executor
        self.event_bus = event_bus
        self.safety_manager = safety_manager
        self.subtask_orchestrator = subtask_orchestrator
        self.max_replans = max_replans
        self.subtask_timeout_seconds = subtask_timeout_seconds
        self.parallel_execution = parallel_execution
        self.max_parallel_subtasks = max_parallel_subtasks

    def _create_isolated_orchestrator(self) -> "ReActOrchestrator":
        """Create an isolated ReAct orchestrator instance for parallel execution.

        During parallel subtask execution, each subtask needs:
        1. Its own event bus - to prevent event cross-contamination
        2. Its own parser - XMLReActParser has mutable state (buffer, in_thinking, etc.)
           that gets corrupted when shared between concurrent subtasks

        The tool_executor and safety_manager are stateless and can be safely shared.

        Returns:
            A new ReActOrchestrator instance with isolated event bus and parser.
        """
        from .events import EventBus
        from .parsing.xml_parser import XMLReActParser

        # Create isolated event bus for this orchestrator
        isolated_event_bus = EventBus()

        # Create isolated parser - CRITICAL: XMLReActParser has mutable state
        # (buffer, current_thinking, in_thinking, in_answer, etc.) that would
        # cause token interleaving if shared between parallel subtasks
        isolated_parser = XMLReActParser()

        # Create new orchestrator with isolated event bus AND parser
        # tool_executor and safety_manager are stateless, safe to share
        return ReActOrchestrator(
            tool_executor=self.subtask_orchestrator.tool_executor,
            event_bus=isolated_event_bus,
            safety_manager=self.subtask_orchestrator.safety_manager,
            parser=isolated_parser,
        )

    def _validate_plan(self, plan: Plan) -> list[str]:
        """Validate plan structure, return list of errors.

        Checks for:
        - Duplicate subtask IDs
        - Invalid dependency references
        - Self-dependencies
        - Dependency cycles
        - Empty descriptions

        Args:
            plan: Plan to validate

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        if not plan.subtasks:
            return errors  # Empty plan is valid (simple query)

        # Check subtask IDs are unique
        ids = [st.id for st in plan.subtasks]
        if len(ids) != len(set(ids)):
            errors.append("Duplicate subtask IDs detected")

        # Check dependencies reference valid IDs
        valid_ids = set(ids)
        for st in plan.subtasks:
            invalid_deps = set(st.dependencies) - valid_ids
            if invalid_deps:
                errors.append(f"Subtask {st.id} has invalid dependencies: {invalid_deps}")

            # Check for self-dependency
            if st.id in st.dependencies:
                errors.append(f"Subtask {st.id} depends on itself")

        # Check for cycles
        if self._has_dependency_cycle(plan):
            errors.append("Dependency cycle detected - would cause deadlock")

        # Check descriptions are non-empty
        for st in plan.subtasks:
            if not st.description or not st.description.strip():
                errors.append(f"Subtask {st.id} has empty description")

        return errors

    def _has_dependency_cycle(self, plan: Plan) -> bool:
        """Detect cycles in dependency graph using DFS.

        Args:
            plan: Plan to check

        Returns:
            True if cycle detected, False otherwise
        """
        if not plan.subtasks:
            return False

        # Build adjacency list: id -> list of dependent IDs
        graph = {st.id: st.dependencies for st in plan.subtasks}
        valid_ids = set(graph.keys())

        # Track visit states: 0=unvisited, 1=visiting (in stack), 2=visited
        state = {id: 0 for id in valid_ids}

        def dfs(node_id: int) -> bool:
            """Return True if cycle found."""
            if node_id not in valid_ids:
                return False  # Invalid dependency, handled elsewhere

            if state[node_id] == 1:
                return True  # Back edge = cycle
            if state[node_id] == 2:
                return False  # Already fully visited

            state[node_id] = 1  # Mark as visiting

            for dep_id in graph.get(node_id, []):
                if dfs(dep_id):
                    return True

            state[node_id] = 2  # Mark as visited
            return False

        # Check from each node
        for node_id in valid_ids:
            if state[node_id] == 0:
                if dfs(node_id):
                    return True

        return False

    def _generate_fallback_response(self, query: str, error: Optional[str] = None) -> str:
        """Generate context-aware fallback response based on query type.

        Args:
            query: Original user query
            error: Optional error message

        Returns:
            Appropriate fallback message
        """
        query_lower = query.lower().strip()

        # Detect query type and provide appropriate fallback
        if any(q in query_lower for q in ["calculate", "compute", "what is", "how much", "how many"]):
            return "I encountered an issue while processing your calculation. Please try rephrasing your question."

        if any(q in query_lower for q in ["debug", "error", "fix", "why", "broken"]):
            return "I ran into a problem while analyzing this. Could you provide more context?"

        if any(q in query_lower for q in ["search", "find", "look up", "lookup"]):
            return "I was unable to complete the search. Please try again or rephrase your query."

        if any(q in query_lower for q in ["summarize", "summary", "explain"]):
            return "I wasn't able to generate the summary. Please try again."

        # Generic but honest fallback
        if error:
            return f"I encountered an unexpected issue: {error}. Please try again."

        return "I wasn't able to process that request. Could you try rephrasing?"

    async def _execute_subtask_with_timeout(
        self,
        subtask: SubTask,
        context: RunContext,
        plan: Plan,
        orchestrator: Optional["ReActOrchestrator"] = None,
    ) -> Dict[str, Any]:
        """Execute subtask with timeout protection.

        Args:
            subtask: Subtask to execute
            context: Run context
            plan: Parent plan (for total subtask count)
            orchestrator: Optional isolated orchestrator for parallel execution.
                         If None, uses self.subtask_orchestrator.

        Returns:
            Dict with keys:
                - success: bool (True if successful, False if failed)
                - needs_clarification: bool (True if clarification requested)
                - clarification_data: Optional[Dict] (clarification details if needed)
        """
        # Get remaining subtask descriptions for boundary enforcement
        remaining_descriptions = [
            st.description for st in plan.subtasks
            if st.id > subtask.id and st.status == "pending"
        ]

        try:
            return await asyncio.wait_for(
                self._execute_subtask(
                    subtask,
                    context,
                    total_subtasks=len(plan.subtasks),
                    remaining_subtasks=remaining_descriptions,
                    orchestrator=orchestrator,
                ),
                timeout=self.subtask_timeout_seconds,
            )
        except asyncio.TimeoutError:
            subtask.status = "failed"
            subtask.error = f"Subtask timed out after {self.subtask_timeout_seconds}s"
            logger.error(f"Subtask {subtask.id} timed out: {subtask.description}")
            await self._publish_event(
                PlanExecuteEventType.SUBTASK_FAILED,
                {"subtask": subtask.to_dict(), "error": subtask.error, "timeout": True},
            )
            return {"success": False, "needs_clarification": False, "clarification_data": None}

    async def execute(
        self, query: str, context: RunContext, existing_plan: Optional[Plan] = None
    ) -> PlanExecuteResult:
        """Execute Plan and Execute workflow.

        Args:
            query: User's goal/query
            context: Run context with messages and state
            existing_plan: Optional pre-generated plan from combined routing step.
                          If provided, skips plan generation (saves ~2-5s)

        Returns:
            PlanExecuteResult with plan, results, and final answer
        """
        start_time = time.time()
        replans = 0

        try:
            # Phase 1: Initial Planning (or use existing plan)
            if existing_plan:
                plan = existing_plan
                logger.info(f"Using pre-generated plan with {len(plan.subtasks)} subtasks")

                # Still emit planning events for UI consistency
                await self._publish_event(PlanExecuteEventType.PLANNING_START, {"goal": query})
                await self._publish_event(
                    PlanExecuteEventType.PLANNING_COMPLETE,
                    {"plan": plan.to_dict(), "subtask_count": len(plan.subtasks)},
                )
            else:
                # Generate new plan (fallback for non-tool-calling providers)
                plan = await self._generate_plan(query, context)

            # Validate plan structure
            validation_errors = self._validate_plan(plan)
            if validation_errors:
                logger.warning(f"Plan validation errors: {validation_errors}")
                # Log but continue - the errors may be recoverable during execution

            # Check if LLM returned empty plan (simple query that doesn't need planning)
            if len(plan.subtasks) == 0:
                logger.info(
                    "Empty plan detected - generating direct response without subtask execution"
                )

                try:
                    final_answer = await self._generate_direct_response(query, context)
                    logger.info(
                        f"Direct response successful: '{final_answer[:100]}...' ({len(final_answer)} chars)"
                    )
                except Exception as e:
                    logger.error(f"Error generating direct response: {e}", exc_info=True)
                    final_answer = self._generate_fallback_response(query, str(e))

                # Ensure we have a valid answer
                if not final_answer or len(final_answer.strip()) == 0:
                    logger.warning("Empty final answer detected, using fallback")
                    final_answer = self._generate_fallback_response(query)

                # Emit final answer event to ensure it's accumulated
                logger.info(f"Emitting FINAL_ANSWER event with {len(final_answer)} chars")
                await self._publish_event(
                    PlanExecuteEventType.FINAL_ANSWER, {"answer": final_answer}
                )

                # Small delay to ensure event is processed
                await asyncio.sleep(0.05)

                # Return early with direct answer
                logger.info(
                    f"Returning PlanExecuteResult with final_answer: {final_answer[:100]}..."
                )
                return PlanExecuteResult(
                    plan=plan,
                    final_answer=final_answer,
                    stop_reason=StopReason.ANSWER_COMPLETE,
                    replans=0,
                    total_cost=0.0,
                    total_execution_time=time.time() - start_time,
                    total_tokens=0,
                )

            # Phase 2: Execute plan with re-planning on failures
            while replans <= self.max_replans:
                execution_success = await self._execute_plan(plan, context)

                # Check for clarification request - skip re-planning and synthesis
                if plan.metadata.get("needs_clarification"):
                    logger.info("Clarification requested, returning without synthesis")
                    total_time = time.time() - start_time
                    total_cost = sum(st.cost for st in plan.subtasks)
                    total_tokens = sum(st.tokens_used for st in plan.subtasks)

                    result = PlanExecuteResult(
                        plan=plan,
                        final_answer="",  # No answer - waiting for clarification
                        stop_reason=StopReason.NEEDS_CLARIFICATION,
                        replans=replans,
                        total_cost=total_cost,
                        total_execution_time=total_time,
                        total_tokens=total_tokens,
                        clarification_data=plan.metadata.get("clarification_data"),
                    )
                    return result

                if execution_success:
                    break

                # Re-planning needed
                if replans < self.max_replans:
                    replans += 1
                    logger.info(f"Re-planning (attempt {replans}/{self.max_replans})")
                    plan = await self._replan(plan, context, replan_attempt=replans)
                else:
                    logger.warning("Max replans reached, stopping execution")
                    break

            # Phase 3: Synthesize final answer
            final_answer = await self._synthesize_results(plan, query, context)

            # Calculate totals
            total_time = time.time() - start_time
            total_cost = sum(st.cost for st in plan.subtasks)
            total_tokens = sum(st.tokens_used for st in plan.subtasks)

            # Determine stop reason
            if plan.failed_subtasks == 0:
                stop_reason = StopReason.ANSWER_COMPLETE
            elif replans >= self.max_replans:
                stop_reason = StopReason.MAX_STEPS  # Reusing enum value
            else:
                stop_reason = StopReason.FORCED_STOP

            result = PlanExecuteResult(
                plan=plan,
                final_answer=final_answer,
                stop_reason=stop_reason,
                replans=replans,
                total_cost=total_cost,
                total_execution_time=total_time,
                total_tokens=total_tokens,
            )

            # Emit final answer event
            await self._publish_event(
                PlanExecuteEventType.FINAL_ANSWER,
                {"answer": final_answer, "result": result.to_dict()},
            )

            return result

        except Exception as e:
            logger.error(f"Plan and Execute execution failed: {e}", exc_info=True)
            # Return error result
            empty_plan = Plan(subtasks=[], goal=query, reasoning="Execution failed")
            return PlanExecuteResult(
                plan=empty_plan,
                final_answer=f"Error occurred during execution: {str(e)}",
                stop_reason=StopReason.FORCED_STOP,
            )

    async def _generate_plan(self, query: str, context: RunContext) -> Plan:
        """Generate initial plan using tool call with streaming thinking.

        Uses the same pattern as ReAct native:
        1. LLM streams thinking in <thinking> tags (readable text)
        2. LLM calls create_plan tool with structured plan data

        This provides:
        - Real-time streaming of planning reasoning (not raw JSON)
        - Structured plan output via tool call
        - Unified architecture with ReAct pattern

        Args:
            query: User's goal
            context: Run context with conversation history

        Returns:
            Plan with subtasks
        """
        from .prompts import PLANNING_WITH_TOOL_SYSTEM_PROMPT, create_plan_tool
        from .parsing.xml_parser import XMLReActParser, ParseEventType

        await self._publish_event(PlanExecuteEventType.PLANNING_START, {"goal": query})

        try:
            # Use planning prompt directly - tool schemas are sent via API's tools parameter,
            # so we don't need to include them in the system prompt
            planning_prompt = PLANNING_WITH_TOOL_SYSTEM_PROMPT

            logger.info("Generating plan using tool call with streaming thinking")

            # Build messages for LLM
            messages = []

            # 1. Add system prompt FIRST
            messages.append(Message(role=MessageRole.SYSTEM, content=planning_prompt))

            # 2. Add conversation history (USER/ASSISTANT only, no SYSTEM messages to avoid conflicts)
            conversation_history = [
                msg
                for msg in context.messages
                if msg.role in (MessageRole.USER, MessageRole.ASSISTANT)
            ]
            messages.extend(conversation_history)

            logger.info(
                f"Calling LLM with {len(messages)} messages for planning (1 system + {len(conversation_history)} conversation)"
            )

            # 3. Create the planning tool and get its schema
            plan_tool = create_plan_tool()
            tool_schema = self.tool_executor._client.client.convert_schema_to_provider_format(
                plan_tool.definition.to_universal_schema()
            )

            # 4. Stream LLM response with thinking + tool call
            accumulated_content = ""
            accumulated_tool_calls = {}  # index -> {id, function: {name, arguments}}
            parser = XMLReActParser()
            parser.reset()
            plan_data = None
            last_emitted_length = 0  # Track for non-XML fallback
            xml_thinking_detected = False

            async for chunk in self.tool_executor._client.client.astream_chat(
                messages=messages,
                tools=[tool_schema],
                temperature=0.2,
            ):
                # Stream thinking text as it arrives
                if chunk.delta:
                    accumulated_content += chunk.delta

                    # Parse XML incrementally to detect <thinking> tags
                    for parse_event in parser.parse_streaming(chunk.delta):
                        if parse_event.event_type == ParseEventType.THINKING:
                            # Thinking chunk detected - emit readable planning reasoning
                            xml_thinking_detected = True
                            delta = parse_event.data["delta"]
                            await self._publish_event(
                                PlanExecuteEventType.PLANNING_THINKING_CHUNK,
                                {"delta": delta, "accumulated": accumulated_content},
                            )
                            last_emitted_length = len(accumulated_content)

                    # Fallback: if no XML tags detected and we have new content, emit as reasoning
                    # Only emit if we haven't seen XML thinking tags and content looks like reasoning
                    if (
                        not xml_thinking_detected
                        and not parser.in_thinking
                        and len(accumulated_content) > last_emitted_length + 30
                    ):
                        new_content = accumulated_content[last_emitted_length:]
                        # Only emit if it doesn't look like JSON (which would be raw plan output)
                        if not new_content.strip().startswith("{"):
                            await self._publish_event(
                                PlanExecuteEventType.PLANNING_THINKING_CHUNK,
                                {
                                    "delta": new_content,
                                    "accumulated": accumulated_content,
                                    "is_fallback": True,
                                },
                            )
                            last_emitted_length = len(accumulated_content)

                # Accumulate tool calls if present in chunk
                if chunk.tool_calls:
                    for tool_call_dict in chunk.tool_calls:
                        idx = 0  # Planning uses single tool call

                        if idx not in accumulated_tool_calls:
                            accumulated_tool_calls[idx] = {
                                "id": None,
                                "type": "function",
                                "function": {"name": None, "arguments": None},
                            }

                        if tool_call_dict.get("id") is not None:
                            accumulated_tool_calls[idx]["id"] = tool_call_dict.get("id")

                        function_data = tool_call_dict.get("function", {})
                        if function_data.get("name") is not None:
                            accumulated_tool_calls[idx]["function"]["name"] = function_data.get("name")

                        new_args = function_data.get("arguments")
                        if new_args is not None:
                            if isinstance(new_args, str):
                                accumulated_tool_calls[idx]["function"]["arguments"] = new_args
                            elif isinstance(new_args, dict):
                                accumulated_tool_calls[idx]["function"]["arguments"] = new_args

            # 5. Extract plan from tool call
            if accumulated_tool_calls:
                tool_call = accumulated_tool_calls.get(0)
                if tool_call and tool_call["function"]["name"] == "create_plan":
                    args = tool_call["function"]["arguments"]
                    if isinstance(args, str):
                        plan_data = json.loads(args)
                    else:
                        plan_data = args
                    logger.info(f"Extracted plan from tool call: {len(plan_data.get('subtasks', []))} subtasks")

            # Fallback: try to parse JSON from accumulated content if no tool call
            if plan_data is None and accumulated_content:
                logger.warning("No tool call detected, attempting JSON fallback parse")
                plan_data = self._parse_plan_json_dict(accumulated_content)

            logger.info(f"Extracted plan: {len(plan_data.get('subtasks', []))} subtasks")
            logger.debug(f"Plan JSON: {accumulated_content}")

            # Convert to Plan object
            subtasks = []
            for idx, st_data in enumerate(plan_data.get("subtasks", []), start=1):
                # Ensure id is always a valid integer, fallback to index if not
                raw_id = st_data.get("id")
                try:
                    subtask_id = int(raw_id) if raw_id is not None else idx
                except (TypeError, ValueError):
                    subtask_id = idx
                subtask = SubTask(
                    id=subtask_id,
                    description=st_data.get("description", ""),
                    required_tools=st_data.get("required_tools", []),
                    dependencies=st_data.get("dependencies", []),
                    success_criteria=st_data.get("success_criteria", ""),
                )
                subtasks.append(subtask)

            plan = Plan(
                goal=query,
                reasoning=plan_data.get("reasoning", "Plan created"),
                subtasks=subtasks,
            )

            logger.info(f"Generated plan with {len(plan.subtasks)} subtasks using tool calling")

        except Exception as e:
            logger.error(f"Error generating plan with tool calling: {e}", exc_info=True)
            # Return minimal fallback plan on error
            plan = Plan(
                goal=query,
                reasoning=f"Plan creation failed: {str(e)}, creating fallback single-step plan",
                subtasks=[
                    SubTask(
                        id=1,
                        description=f"Execute task: {query}",
                        required_tools=[],
                        dependencies=[],
                        success_criteria="Task completed successfully",
                    )
                ],
            )

        await self._publish_event(
            PlanExecuteEventType.PLANNING_COMPLETE,
            {"plan": plan.to_dict(), "subtask_count": len(plan.subtasks)},
        )

        logger.info(f"Plan complete: {len(plan.subtasks)} subtasks, reasoning: {plan.reasoning}")

        return plan

    async def _generate_direct_response(self, query: str, context: RunContext) -> str:
        """Generate direct response for simple queries that don't need planning.

        Used when LLM returns empty plan (0 subtasks), indicating the query is simple
        enough to answer directly without multi-step execution.

        Args:
            query: User's original query
            context: Run context with conversation history

        Returns:
            Direct response string
        """
        logger.info(f"Generating direct response for query: {query}")

        # Build messages - use existing context messages as base
        messages = []

        # Copy existing conversation context (includes system prompt, previous messages)
        if context.messages:
            messages = [msg for msg in context.messages]
            logger.info(f"Using {len(messages)} context messages")

        # Ensure query is in messages (it should be from enhanced_response_generator)
        # Only add query if no user message exists AND query is not empty
        # Check if last message is a user message (it should be from enhanced_response_generator)
        last_user_msg = next(
            (msg for msg in reversed(messages) if msg.role == MessageRole.USER), None
        )

        # Only add query if no user message exists AND query is not empty
        if not last_user_msg and query and query.strip():
            logger.info(f"Adding query to messages: {query}")
            messages.append(Message(role=MessageRole.USER, content=query))

        logger.info(f"Calling LLM with {len(messages)} total messages")

        # Stream LLM response for real-time feedback (without tools)
        final_answer = ""
        async for chunk in self.tool_executor.stream_without_tools(
            messages=messages, temperature=0.7
        ):
            delta = chunk.delta
            if delta:
                final_answer += delta
                # Emit FINAL_ANSWER_CHUNK event for real-time streaming
                await self._publish_event(
                    PlanExecuteEventType.FINAL_ANSWER_CHUNK,
                    {"delta": delta, "content": final_answer},
                )

        if not final_answer or len(final_answer.strip()) == 0:
            logger.warning("LLM returned empty response, using fallback")
            final_answer = "Hello! How can I help you today?"

        logger.info(f"Direct response generated: {len(final_answer)} chars")

        return final_answer

    async def _replan(
        self, current_plan: Plan, context: RunContext, replan_attempt: int = 1
    ) -> Plan:
        """Re-generate plan after failure with streaming.

        Args:
            current_plan: Current plan with failures
            context: Run context
            replan_attempt: Current replan attempt number

        Returns:
            New revised plan
        """
        # Find failed subtasks (may be multiple in wave execution)
        failed_subtasks = [st for st in current_plan.subtasks if st.status == "failed"]
        failed_subtask = failed_subtasks[0] if failed_subtasks else None

        # Emit enhanced replanning start event with failure details
        await self._publish_event(
            PlanExecuteEventType.REPLANNING_START,
            {
                "current_plan": current_plan.to_dict(),
                "failed_subtask_id": failed_subtask.id if failed_subtask else None,
                "failed_subtask_description": failed_subtask.description if failed_subtask else None,
                "failure_reason": failed_subtask.error if failed_subtask else "Unknown",
                "failed_count": len(failed_subtasks),
                "replan_attempt": replan_attempt,
                "max_replans": self.max_replans,
            },
        )

        # Build plan status summary
        plan_status = self._format_plan_status(current_plan)

        # Build completed context to preserve successful work
        completed_results = [
            (st.id, st.description, st.result[:200] if st.result else "")
            for st in current_plan.subtasks
            if st.status == "completed" and st.result
        ]

        if completed_results:
            completed_context = "\n\nCompleted Work (use these results):\n" + "\n".join(
                f"- Subtask {id} ({desc}): {result}..."
                for id, desc, result in completed_results
            )
        else:
            completed_context = ""

        # Build replanning prompt
        replan_prompt = PLAN_AND_EXECUTE_REPLAN_PROMPT.format(
            goal=current_plan.goal,
            plan_status=plan_status,
            failed_subtask=failed_subtask.description if failed_subtask else "Unknown",
            error=failed_subtask.error if failed_subtask else "Unknown error",
            completed_context=completed_context,
        )

        # Create replanning messages
        messages = [Message(role=MessageRole.USER, content=replan_prompt)]

        # Stream replanning for user feedback
        accumulated_content = ""
        async for chunk in self.tool_executor.stream_without_tools(
            messages=messages, temperature=0.6
        ):
            if chunk.delta:
                accumulated_content += chunk.delta
                # Emit streaming chunk for UI visibility
                await self._publish_event(
                    PlanExecuteEventType.REPLANNING_THINKING_CHUNK,
                    {"delta": chunk.delta, "content": accumulated_content},
                )

        # Parse new plan from accumulated content
        plan_data = self._parse_plan_json_dict(accumulated_content)

        # Convert to Plan object
        subtasks = []
        for idx, st_data in enumerate(plan_data.get("subtasks", []), start=1):
            # Ensure id is always a valid integer, fallback to index if not
            raw_id = st_data.get("id")
            try:
                subtask_id = int(raw_id) if raw_id is not None else idx
            except (TypeError, ValueError):
                subtask_id = idx
            subtask = SubTask(
                id=subtask_id,
                description=st_data.get("description", ""),
                required_tools=st_data.get("required_tools", []),
                dependencies=st_data.get("dependencies", []),
                success_criteria=st_data.get("success_criteria", ""),
            )
            subtasks.append(subtask)

        new_plan = Plan(
            goal=current_plan.goal,
            reasoning=plan_data.get("reasoning", "Re-planned"),
            subtasks=subtasks,
        )

        await self._publish_event(
            PlanExecuteEventType.REPLANNING_COMPLETE,
            {"new_plan": new_plan.to_dict(), "subtask_count": len(new_plan.subtasks)},
        )

        logger.info(f"Re-planned with {len(new_plan.subtasks)} subtasks: {new_plan.reasoning}")

        return new_plan

    async def _execute_plan(self, plan: Plan, context: RunContext) -> bool:
        """Execute all subtasks in the plan.

        Dispatches to sequential or parallel execution based on parallel_execution flag.

        Args:
            plan: Plan to execute
            context: Run context

        Returns:
            True if all subtasks succeeded, False if any failed
        """
        if self.parallel_execution:
            return await self._execute_plan_parallel(plan, context)
        else:
            return await self._execute_plan_sequential(plan, context)

    async def _execute_plan_sequential(self, plan: Plan, context: RunContext) -> bool:
        """Execute all subtasks in the plan sequentially respecting dependencies.

        Args:
            plan: Plan to execute
            context: Run context

        Returns:
            True if all subtasks succeeded, False if any failed
        """
        logger.info(f"Executing plan with {len(plan.subtasks)} subtasks sequentially")

        completed_ids: Set[int] = set()

        for i, subtask in enumerate(plan.subtasks):
            # Check dependencies are satisfied
            if not self._dependencies_met(subtask, completed_ids):
                logger.error(
                    f"Subtask {subtask.id} has unmet dependencies: {subtask.dependencies}"
                )
                subtask.status = "failed"
                subtask.error = "Dependencies not satisfied"
                return False

            # Execute subtask with timeout
            result = await self._execute_subtask_with_timeout(subtask, context, plan)

            # Check for clarification request
            if result.get("needs_clarification"):
                logger.info(f"Subtask {subtask.id} needs clarification, stopping plan execution")
                # Store clarification data in plan metadata for later retrieval
                plan.metadata["needs_clarification"] = True
                plan.metadata["clarification_data"] = result.get("clarification_data")
                return False

            if result.get("success"):
                completed_ids.add(subtask.id)
            else:
                # Subtask failed - stop execution and trigger replan
                logger.warning(f"Subtask {subtask.id} failed, stopping plan execution")
                return False

            # Publish progress after each subtask
            await self._publish_event(
                PlanExecuteEventType.PLAN_PROGRESS,
                {
                    "completed": len(completed_ids),
                    "failed": 0,
                    "total": len(plan.subtasks),
                    "progress_percentage": (len(completed_ids) / len(plan.subtasks)) * 100,
                },
            )

        logger.info("Plan execution completed successfully")
        return True

    async def _execute_plan_parallel(self, plan: Plan, context: RunContext) -> bool:
        """Execute all subtasks in parallel waves respecting dependencies.

        Groups subtasks into waves where each wave contains subtasks that can
        run in parallel (all their dependencies are in completed waves).

        Args:
            plan: Plan to execute
            context: Run context

        Returns:
            True if all subtasks succeeded, False if any failed
        """
        logger.info(f"Executing plan with {len(plan.subtasks)} subtasks in parallel waves")

        # Build execution waves
        waves = self._build_execution_waves(plan.subtasks)

        if not waves:
            logger.info("No waves to execute (empty plan)")
            return True

        completed_ids: Set[int] = set()

        for wave in waves:
            # Emit wave start event
            await self._publish_parallel_event(
                ParallelPlanEventType.WAVE_START,
                {
                    "wave_number": wave.wave_number,
                    "subtask_ids": [st.id for st in wave.subtasks],
                    "parallel_count": wave.parallel_count,
                    "total_waves": len(waves),
                },
            )

            wave_start_time = time.time()

            # Execute all subtasks in this wave in parallel
            if wave.parallel_count == 1:
                # Single subtask - run directly
                subtask = wave.subtasks[0]
                result = await self._execute_subtask_with_timeout(subtask, context, plan)

                # Check for clarification request
                if result.get("needs_clarification"):
                    plan.metadata["needs_clarification"] = True
                    plan.metadata["clarification_data"] = result.get("clarification_data")
                    return False

                results = [result]
            else:
                # Multiple subtasks - run in parallel with asyncio.gather
                # CRITICAL: Each parallel subtask needs ISOLATED context AND orchestrator to avoid:
                # 1. Message corruption: When subtasks share context.messages, one subtask's
                #    tool_use can appear in another's messages without tool_result, causing
                #    Anthropic API errors.
                # 2. Event cross-contamination: When subtasks share event bus, all callbacks
                #    receive all events, causing duplicated/interleaved token streams.
                async def execute_with_event(
                    st: SubTask,
                    isolated_context: RunContext,
                    isolated_orchestrator: "ReActOrchestrator",
                ) -> Dict[str, Any]:
                    """Execute subtask with isolated context and orchestrator."""
                    await self._publish_parallel_event(
                        ParallelPlanEventType.PARALLEL_SUBTASK_START,
                        {
                            "subtask_id": st.id,
                            "wave_number": wave.wave_number,
                            "description": st.description,
                        },
                    )

                    # Pass isolated orchestrator to prevent event cross-contamination
                    result = await self._execute_subtask_with_timeout(
                        st, isolated_context, plan, orchestrator=isolated_orchestrator
                    )

                    success = result.get("success", False)
                    await self._publish_parallel_event(
                        ParallelPlanEventType.PARALLEL_SUBTASK_COMPLETE,
                        {
                            "subtask_id": st.id,
                            "wave_number": wave.wave_number,
                            "success": success,
                            "result": st.result if success else None,
                            "error": st.error if not success else None,
                            "needs_clarification": result.get("needs_clarification", False),
                        },
                    )

                    return result

                # Create isolated contexts for each parallel subtask
                # Each subtask gets a deep copy of messages to prevent cross-contamination
                def create_isolated_context(base_context: RunContext) -> RunContext:
                    """Create an isolated copy of the context for parallel execution."""
                    return RunContext(
                        deps=base_context.deps,  # Shared deps (read-only config)
                        messages=copy.deepcopy(base_context.messages),  # Deep copy messages!
                        retry=base_context.retry,
                        metadata=base_context.metadata.copy(),  # Shallow copy metadata
                    )

                # Run all subtasks in parallel with isolated contexts AND orchestrators
                # Each subtask gets its own event bus to prevent event cross-contamination
                results = await asyncio.gather(
                    *[
                        execute_with_event(
                            st,
                            create_isolated_context(context),
                            self._create_isolated_orchestrator(),
                        )
                        for st in wave.subtasks
                    ],
                    return_exceptions=True,
                )

            # Process results
            wave_success = True
            clarification_found = False
            for i, result in enumerate(results):
                subtask = wave.subtasks[i]
                if isinstance(result, Exception):
                    subtask.status = "failed"
                    subtask.error = str(result)
                    wave_success = False
                    logger.error(f"Subtask {subtask.id} raised exception: {result}")
                elif isinstance(result, dict):
                    # Check for clarification request
                    if result.get("needs_clarification"):
                        clarification_found = True
                        plan.metadata["needs_clarification"] = True
                        plan.metadata["clarification_data"] = result.get("clarification_data")
                        logger.info(f"Subtask {subtask.id} needs clarification")
                    elif result.get("success"):
                        completed_ids.add(subtask.id)
                    else:
                        wave_success = False
                else:
                    # Legacy: handle bool return (should not happen with updated code)
                    if result:
                        completed_ids.add(subtask.id)
                    else:
                        wave_success = False

            # If any subtask needs clarification, stop wave execution
            if clarification_found:
                return False

            wave.execution_time = time.time() - wave_start_time

            # Emit wave complete event
            await self._publish_parallel_event(
                ParallelPlanEventType.WAVE_COMPLETE,
                {
                    "wave_number": wave.wave_number,
                    "completed_ids": list(completed_ids),
                    "success": wave_success,
                    "execution_time": wave.execution_time,
                },
            )

            # Publish progress
            await self._publish_event(
                PlanExecuteEventType.PLAN_PROGRESS,
                {
                    "completed": len(completed_ids),
                    "failed": sum(1 for st in plan.subtasks if st.status == "failed"),
                    "total": len(plan.subtasks),
                    "progress_percentage": (len(completed_ids) / len(plan.subtasks)) * 100,
                    "current_wave": wave.wave_number,
                    "total_waves": len(waves),
                },
            )

            if not wave_success:
                # Wave had failures - stop and trigger replan
                logger.warning(f"Wave {wave.wave_number} had failures, stopping execution")
                return False

        logger.info(f"Parallel plan execution completed successfully in {len(waves)} waves")
        return True

    def _build_execution_waves(self, subtasks: List[SubTask]) -> List[ExecutionWave]:
        """Build parallel execution waves using topological sort.

        Groups subtasks into waves where:
        - Wave 0: All subtasks with no dependencies
        - Wave N: All subtasks whose dependencies are all in waves < N

        Args:
            subtasks: List of subtasks with dependencies

        Returns:
            List of ExecutionWave objects ordered by wave number
        """
        if not subtasks:
            return []

        # Build dependency graph
        id_to_subtask = {st.id: st for st in subtasks}
        remaining = set(st.id for st in subtasks)
        completed_ids: Set[int] = set()
        waves: List[ExecutionWave] = []
        wave_number = 0

        while remaining:
            # Find subtasks whose dependencies are all completed
            ready_ids = [
                st_id for st_id in remaining
                if all(dep_id in completed_ids for dep_id in id_to_subtask[st_id].dependencies)
            ]

            if not ready_ids:
                # Cycle detected or invalid dependencies - shouldn't happen if validation passed
                logger.error(f"No ready subtasks but {len(remaining)} remaining - possible cycle")
                # Put remaining in final wave as fallback
                ready_ids = list(remaining)

            # Limit wave size to max_parallel_subtasks
            wave_subtasks = [id_to_subtask[st_id] for st_id in ready_ids[:self.max_parallel_subtasks]]
            remaining_ready = ready_ids[self.max_parallel_subtasks:]

            wave = ExecutionWave(
                wave_number=wave_number,
                subtasks=wave_subtasks,
                parallel_count=len(wave_subtasks),
            )
            waves.append(wave)

            # Mark as completed and remove from remaining
            for st in wave_subtasks:
                completed_ids.add(st.id)
                remaining.discard(st.id)

            # If we had to limit wave size, remaining_ready goes to next wave
            # (they'll be picked up in the next iteration)

            wave_number += 1

        logger.info(f"Built {len(waves)} execution waves from {len(subtasks)} subtasks")
        for wave in waves:
            logger.debug(f"  Wave {wave.wave_number}: {[st.id for st in wave.subtasks]} ({wave.parallel_count} parallel)")

        return waves

    async def _publish_parallel_event(
        self, event_type: ParallelPlanEventType, data: Dict[str, Any]
    ) -> None:
        """Publish a parallel plan event.

        Args:
            event_type: Type of parallel plan event
            data: Event data
        """
        event = ParallelPlanEvent(event_type=event_type, data=data)
        await self.event_bus.publish(event)

    def _dependencies_met(self, subtask: SubTask, completed_ids: set) -> bool:
        """Check if all dependencies for a subtask are satisfied."""
        return all(dep_id in completed_ids for dep_id in subtask.dependencies)

    async def _execute_subtask(
        self,
        subtask: SubTask,
        context: RunContext,
        total_subtasks: int = 1,
        remaining_subtasks: Optional[list] = None,
        orchestrator: Optional["ReActOrchestrator"] = None,
    ) -> Dict[str, Any]:
        """Execute a single subtask.

        Args:
            subtask: Subtask to execute
            context: Run context
            total_subtasks: Total number of subtasks in the plan (for conditional rendering)
            remaining_subtasks: Descriptions of upcoming subtasks (for boundary enforcement)
            orchestrator: Optional isolated orchestrator for parallel execution.
                         If None, uses self.subtask_orchestrator.

        Returns:
            Dict with keys:
                - success: bool (True if successful, False if failed)
                - needs_clarification: bool (True if clarification requested)
                - clarification_data: Optional[Dict] (clarification details if needed)
        """
        subtask.status = "running"
        start_time = time.time()

        # Emit SUBTASK_START for all subtasks (including single-subtask plans)
        # The frontend handles single vs multi-subtask display differently
        if total_subtasks >= 1:
            await self._publish_event(
                PlanExecuteEventType.SUBTASK_START,
                {"subtask": subtask.to_dict(), "description": subtask.description},
            )

        # Use provided orchestrator or fall back to self.subtask_orchestrator
        active_orchestrator = orchestrator or self.subtask_orchestrator

        try:
            if not active_orchestrator:
                raise ValueError(
                    "ReAct orchestrator is required for subtask execution. "
                    "Please provide subtask_orchestrator in constructor or pass orchestrator parameter."
                )

            # Use ReAct orchestrator for subtask execution with event streaming
            logger.info(f"Executing subtask {subtask.id} with ReAct: {subtask.description}")

            # Build boundary-aware prompt for multi-step plans
            if total_subtasks > 1 and remaining_subtasks:
                # Warn about upcoming steps to avoid (show max 3)
                remaining_warning = "\nDo NOT perform these upcoming steps:\n" + "\n".join(
                    f"- {desc}" for desc in remaining_subtasks[:3]
                )

                scoped_query = SUBTASK_EXECUTION_PROMPT.format(
                    subtask_number=subtask.id,
                    total_subtasks=total_subtasks,
                    subtask_description=subtask.description,
                    remaining_steps_warning=remaining_warning,
                )
            elif total_subtasks > 1:
                # Multi-step plan but this is the last step
                scoped_query = SUBTASK_EXECUTION_PROMPT.format(
                    subtask_number=subtask.id,
                    total_subtasks=total_subtasks,
                    subtask_description=subtask.description,
                    remaining_steps_warning="",
                )
            else:
                # Single subtask plan - no boundary needed
                scoped_query = subtask.description

            # Create event forwarder to bubble up ReAct events as PlanExecute events
            # Using async to ensure events are published in order (no fire-and-forget race conditions)
            # Sequence counter ensures proper ordering of events when received out-of-order
            # due to parallel execution (asyncio.gather doesn't guarantee event order)
            subtask_sequence = 0

            async def forward_react_event(react_event):
                """Forward ReAct events from subtask as PlanExecute subtask events."""
                nonlocal subtask_sequence
                try:
                    # Increment sequence for each event to ensure proper ordering
                    subtask_sequence += 1
                    current_sequence = subtask_sequence

                    # Convert ReActEvent to PlanExecuteEvent with subtask context
                    if react_event.event_type == ReActEventType.THINKING_CHUNK:
                        # Create subtask thinking chunk event
                        await self._publish_event(
                            PlanExecuteEventType.SUBTASK_THINKING_CHUNK,
                            {
                                "subtask_id": subtask.id,
                                "sequence": current_sequence,
                                "delta": react_event.data.get("delta", ""),
                                "thought": react_event.data.get("content", ""),
                            },
                        )
                    elif react_event.event_type == ReActEventType.ACTION_PLANNED:
                        # Tool is about to be called
                        tool_name = react_event.data.get("action", "unknown")
                        tool_args = react_event.data.get("action_input", {})
                        tool_description = react_event.data.get("tool_description")
                        await self._publish_event(
                            PlanExecuteEventType.SUBTASK_THINKING_CHUNK,
                            {
                                "subtask_id": subtask.id,
                                "sequence": current_sequence,
                                "delta": "",
                                "is_tool": True,
                                "is_tool_planned": True,
                                "tool_name": tool_name,
                                "tool_args": tool_args,
                                "tool_description": tool_description,
                            },
                        )
                    elif react_event.event_type == ReActEventType.ACTION_EXECUTING:
                        # Tool is currently executing
                        tool_name = react_event.data.get("action", "unknown")
                        tool_args = react_event.data.get("action_input", {})
                        tool_description = react_event.data.get("tool_description")
                        await self._publish_event(
                            PlanExecuteEventType.SUBTASK_THINKING_CHUNK,
                            {
                                "subtask_id": subtask.id,
                                "sequence": current_sequence,
                                "delta": "",
                                "is_tool": True,
                                "is_tool_executing": True,
                                "tool_name": tool_name,
                                "tool_args": tool_args,
                                "tool_description": tool_description,
                            },
                        )
                    elif react_event.event_type == ReActEventType.OBSERVATION:
                        # Tool execution result
                        observation = str(react_event.data.get("observation", ""))
                        tool_name = react_event.data.get("action", "unknown")
                        await self._publish_event(
                            PlanExecuteEventType.SUBTASK_THINKING_CHUNK,
                            {
                                "subtask_id": subtask.id,
                                "sequence": current_sequence,
                                "delta": observation,
                                "is_observation": True,
                                "tool_name": tool_name,
                                "success": react_event.data.get("success", True),
                            },
                        )
                except Exception as e:
                    logger.error(f"Error forwarding ReAct event: {e}")

            # Subscribe to orchestrator's events (using active_orchestrator for isolation)
            active_orchestrator.event_bus.subscribe(forward_react_event)

            try:
                # CRITICAL: Create fresh context for subtask execution
                # The original context.messages contains the full user query, which would
                # cause the subtask to try to complete the entire task instead of just
                # its scoped part. We only keep system messages (assistant persona, etc.)
                # and let the scoped_query be the only user message.
                subtask_messages = [
                    msg for msg in context.messages if msg.role == MessageRole.SYSTEM
                ]
                subtask_context = RunContext(
                    deps=context.deps,
                    messages=subtask_messages,
                    retry=context.retry,
                    metadata=context.metadata.copy() if context.metadata else {},
                )

                # Execute subtask with scoped query - events will be forwarded automatically
                result = await active_orchestrator.execute(scoped_query, subtask_context)

                # Check if subtask needs clarification
                if result.stop_reason == StopReason.NEEDS_CLARIFICATION:
                    subtask.status = "needs_clarification"
                    subtask.result = result.final_answer
                    subtask.cost = result.total_cost
                    subtask.tokens_used = result.total_tokens
                    subtask.execution_time = time.time() - start_time

                    # Get clarification data from result
                    clarification_data = getattr(result, "clarification_data", None) or {}
                    clarification_data["subtask_id"] = subtask.id
                    clarification_data["subtask_description"] = subtask.description

                    # Emit clarification event
                    await self._publish_event(
                        PlanExecuteEventType.CLARIFICATION_NEEDED,
                        {
                            "subtask": subtask.to_dict(),
                            **clarification_data,
                        },
                    )

                    logger.info(f"Subtask {subtask.id} needs clarification")
                    return {
                        "success": False,
                        "needs_clarification": True,
                        "clarification_data": clarification_data,
                    }

                subtask.result = result.final_answer
                subtask.cost = result.total_cost
                subtask.tokens_used = result.total_tokens
                subtask.status = "completed"
            finally:
                # Unsubscribe to avoid memory leaks
                active_orchestrator.event_bus.unsubscribe(forward_react_event)

            subtask.execution_time = time.time() - start_time

            # Emit SUBTASK_COMPLETE for all plans with subtasks (including single-subtask)
            if total_subtasks >= 1:
                await self._publish_event(
                    PlanExecuteEventType.SUBTASK_COMPLETE,
                    {
                        "subtask": subtask.to_dict(),
                        "result": subtask.result,
                        "execution_time": subtask.execution_time,
                    },
                )

            return {"success": True, "needs_clarification": False, "clarification_data": None}

        except Exception as e:
            subtask.status = "failed"
            subtask.error = str(e)
            subtask.execution_time = time.time() - start_time

            # Emit SUBTASK_FAILED for all subtasks (including single-subtask plans)
            if total_subtasks >= 1:
                await self._publish_event(
                    PlanExecuteEventType.SUBTASK_FAILED,
                    {"subtask": subtask.to_dict(), "error": str(e)},
                )

            logger.error(f"Subtask {subtask.id} failed: {e}", exc_info=True)

            return {"success": False, "needs_clarification": False, "clarification_data": None}

    async def _synthesize_results(self, plan: Plan, query: str, context: RunContext) -> str:
        """Synthesize subtask results into final answer.

        Args:
            plan: Executed plan with results
            query: Original user query
            context: Run context

        Returns:
            Final answer string
        """
        # Emit synthesis start event for UI visibility
        await self._publish_event(
            PlanExecuteEventType.SYNTHESIS_START,
            {
                "completed_subtasks": plan.completed_subtasks,
                "total_subtasks": plan.total_subtasks,
                "results_preview": [
                    st.description for st in plan.subtasks if st.status == "completed"
                ],
            },
        )

        # Collect successful subtask results
        results = []
        for subtask in plan.subtasks:
            if subtask.status == "completed" and subtask.result:
                results.append(f"- {subtask.description}: {subtask.result}")

        if not results:
            return "No subtasks completed successfully. Unable to provide an answer."

        # Use LLM to synthesize final answer with streaming
        synthesis_prompt = f"""Based on the following subtask results, provide a comprehensive answer to the user's question.

Original Question: {query}

Subtask Results:
{chr(10).join(results)}

Provide a clear, well-formatted final answer that directly addresses the user's question:"""

        messages = [Message(role=MessageRole.USER, content=synthesis_prompt)]

        # Stream the final answer token by token (without tools)
        final_answer = ""
        async for chunk in self.tool_executor.stream_without_tools(
            messages=messages, temperature=0.5
        ):
            delta = chunk.delta
            if delta:
                final_answer += delta
                # Emit FINAL_ANSWER_CHUNK event for real-time streaming
                await self._publish_event(
                    PlanExecuteEventType.FINAL_ANSWER_CHUNK,
                    {"delta": delta, "content": final_answer},
                )

        return final_answer

    def _parse_plan_json_dict(self, json_str: str) -> dict:
        """Parse JSON plan from LLM response (fallback for providers without JSON schema).

        This method handles common LLM JSON output issues:
        - Markdown code blocks
        - Single quotes instead of double quotes
        - Trailing commas
        - Unquoted property names
        - Python dict literal format
        - Thinking/explanation text mixed with JSON

        Args:
            json_str: JSON string from LLM (may include markdown, thinking text, etc.)

        Returns:
            Parsed plan data as dict with 'reasoning' and 'subtasks' keys
        """
        original_str = json_str
        try:
            # Extract JSON from response (might have markdown code blocks or thinking text)
            json_str = json_str.strip()

            # Try to extract from markdown code blocks first
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
            else:
                # Try to find JSON object in the text (look for opening brace)
                brace_idx = json_str.find("{")
                if brace_idx != -1:
                    json_str = json_str[brace_idx:]
                    # Find matching closing brace by counting braces
                    brace_count = 0
                    for i, char in enumerate(json_str):
                        if char == "{":
                            brace_count += 1
                        elif char == "}":
                            brace_count -= 1
                            if brace_count == 0:
                                json_str = json_str[: i + 1]
                                break

            json_str = json_str.strip()
            if not json_str:
                raise ValueError("No JSON content found in response")

            # Try standard JSON parsing first
            try:
                plan_data = json.loads(json_str)
                return plan_data
            except json.JSONDecodeError:
                # Continue to fallback methods
                pass

            # Fallback 1: Sanitize common JSON issues
            sanitized = self._sanitize_json_string(json_str)
            try:
                plan_data = json.loads(sanitized)
                logger.info("Successfully parsed plan JSON after sanitization")
                return plan_data
            except json.JSONDecodeError:
                pass

            # Fallback 2: Try ast.literal_eval for Python dict notation
            try:
                plan_data = ast.literal_eval(json_str)
                if isinstance(plan_data, dict):
                    logger.info("Successfully parsed plan using ast.literal_eval")
                    return plan_data
            except (ValueError, SyntaxError):
                pass

            # Fallback 3: Try ast.literal_eval on sanitized string
            try:
                plan_data = ast.literal_eval(sanitized)
                if isinstance(plan_data, dict):
                    logger.info("Successfully parsed sanitized plan using ast.literal_eval")
                    return plan_data
            except (ValueError, SyntaxError):
                pass

            # All parsing methods failed
            raise ValueError(f"Could not parse JSON with any method")

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse plan JSON: {e}")
            logger.debug(f"Original string was: {original_str[:500]}...")
            # Return empty plan data on parse failure
            return {"reasoning": f"Failed to parse plan: {str(e)}", "subtasks": []}

    def _sanitize_json_string(self, json_str: str) -> str:
        """Sanitize common JSON formatting issues from LLM output.

        Handles:
        - Single quotes → double quotes
        - Trailing commas before closing brackets
        - Unquoted property names
        - JavaScript comments

        Args:
            json_str: Raw JSON-like string

        Returns:
            Sanitized JSON string
        """
        result = json_str

        # Remove JavaScript-style comments (// and /* */)
        result = re.sub(r"//.*?$", "", result, flags=re.MULTILINE)
        result = re.sub(r"/\*.*?\*/", "", result, flags=re.DOTALL)

        # Replace single quotes with double quotes (careful with nested quotes)
        # This regex handles the common case where properties use single quotes
        # Pattern: matches 'key': or 'value' patterns
        # Step 1: Replace single-quoted keys: 'key':
        result = re.sub(r"'(\w+)'\s*:", r'"\1":', result)

        # Step 2: Replace single-quoted string values that follow : or ,
        # This is trickier - we use a simple approach that works for most cases
        # Match : followed by optional whitespace and single-quoted string
        result = re.sub(r":\s*'([^']*)'", r': "\1"', result)
        # Match [ or , followed by optional whitespace and single-quoted string
        result = re.sub(r"([\[,])\s*'([^']*)'", r'\1 "\2"', result)

        # Remove trailing commas before closing brackets/braces
        # Pattern: comma followed by whitespace and then } or ]
        result = re.sub(r",\s*([\]\}])", r"\1", result)

        # Handle unquoted property names (e.g., {reasoning: "value"} → {"reasoning": "value"})
        # Pattern: start of object or comma, whitespace, unquoted word, colon
        result = re.sub(r"([\{,])\s*(\w+)\s*:", r'\1"\2":', result)

        return result

    def _dependencies_met(self, subtask: SubTask, completed_ids: set) -> bool:
        """Check if subtask dependencies are satisfied.

        Args:
            subtask: Subtask to check
            completed_ids: Set of completed subtask IDs

        Returns:
            True if all dependencies are met
        """
        return all(dep_id in completed_ids for dep_id in subtask.dependencies)

    def _format_plan_status(self, plan: Plan) -> str:
        """Format plan status for re-planning prompt.

        Args:
            plan: Current plan

        Returns:
            Formatted status string
        """
        lines = []
        for st in plan.subtasks:
            status_emoji = {"completed": "✓", "failed": "✗", "pending": "○", "running": "⟳"}.get(
                st.status, "?"
            )
            lines.append(f"{status_emoji} Subtask {st.id}: {st.description} [{st.status}]")
            if st.result:
                lines.append(f"  Result: {st.result[:100]}...")
            if st.error:
                lines.append(f"  Error: {st.error}")

        return "\n".join(lines)

    async def _publish_event(self, event_type: PlanExecuteEventType, data: Dict[str, Any]):
        """Publish event to event bus.

        Args:
            event_type: Type of event
            data: Event data
        """
        event = PlanExecuteEvent(event_type=event_type, data=data)
        await self.event_bus.publish(event)

    def get_current_status(self) -> Dict[str, Any]:
        """Get current execution status."""
        return {
            "agent_type": "plan_and_execute_orchestrator",
            "parallel_execution": self.parallel_execution,
            "max_parallel_subtasks": self.max_parallel_subtasks,
        }
