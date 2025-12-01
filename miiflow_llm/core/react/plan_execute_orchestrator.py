"""Plan and Execute orchestrator for complex multi-step tasks."""

import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional

from ..agent import RunContext
from ..message import Message, MessageRole
from .data import (
    PLAN_AND_EXECUTE_PLANNING_PROMPT,
    PLAN_AND_EXECUTE_REPLAN_PROMPT,
    Plan,
    PlanExecuteEvent,
    PlanExecuteEventType,
    PlanExecuteResult,
    ReActEventType,
    StopReason,
    SubTask,
)
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
        use_react_for_subtasks: bool = True,
    ):
        """Initialize Plan and Execute orchestrator.

        Args:
            tool_executor: Tool execution adapter
            event_bus: Event bus for streaming events
            safety_manager: Safety condition checker
            subtask_orchestrator: Optional ReAct orchestrator for complex subtasks
            max_replans: Maximum number of re-planning attempts
            use_react_for_subtasks: Whether to use ReAct for subtask execution
        """
        self.tool_executor = tool_executor
        self.event_bus = event_bus
        self.safety_manager = safety_manager
        self.subtask_orchestrator = subtask_orchestrator
        self.max_replans = max_replans
        self.use_react_for_subtasks = use_react_for_subtasks

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
                    final_answer = "Hello! How can I help you today?"

                # Ensure we have a valid answer
                if not final_answer or len(final_answer.strip()) == 0:
                    logger.warning("Empty final answer detected, using fallback")
                    final_answer = "Hello! How can I help you today?"

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

                if execution_success:
                    break

                # Re-planning needed
                if replans < self.max_replans:
                    replans += 1
                    logger.info(f"Re-planning (attempt {replans}/{self.max_replans})")
                    plan = await self._replan(plan, context)
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
        from .data import PLANNING_WITH_TOOL_SYSTEM_PROMPT, create_plan_tool
        from .parsing.xml_parser import XMLReActParser, ParseEventType

        await self._publish_event(PlanExecuteEventType.PLANNING_START, {"goal": query})

        try:
            # Build planning prompt with tool info
            tools_info = self.tool_executor.build_tools_description()
            planning_prompt = PLANNING_WITH_TOOL_SYSTEM_PROMPT.format(tools=tools_info)

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
                            delta = parse_event.data["delta"]
                            await self._publish_event(
                                PlanExecuteEventType.PLANNING_THINKING_CHUNK,
                                {"delta": delta, "accumulated": accumulated_content},
                            )

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
            for st_data in plan_data.get("subtasks", []):
                subtask = SubTask(
                    id=st_data.get("id"),
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

    async def _replan(self, current_plan: Plan, context: RunContext) -> Plan:
        """Re-generate plan after failure.

        Args:
            current_plan: Current plan with failures
            context: Run context

        Returns:
            New revised plan
        """
        await self._publish_event(
            PlanExecuteEventType.REPLANNING_START, {"current_plan": current_plan.to_dict()}
        )

        # Find failed subtask
        failed_subtask = next((st for st in current_plan.subtasks if st.status == "failed"), None)

        # Build plan status summary
        plan_status = self._format_plan_status(current_plan)

        # Build replanning prompt
        replan_prompt = PLAN_AND_EXECUTE_REPLAN_PROMPT.format(
            goal=current_plan.goal,
            plan_status=plan_status,
            failed_subtask=failed_subtask.description if failed_subtask else "Unknown",
            error=failed_subtask.error if failed_subtask else "Unknown error",
        )

        # Create replanning messages
        messages = [Message(role=MessageRole.USER, content=replan_prompt)]

        # Call LLM to replan (higher temperature for diverse plan sizes)
        response = await self.tool_executor._client.achat(messages=messages, temperature=0.6)

        # Parse new plan
        new_plan = self._parse_plan_json(response.message.content, current_plan.goal)

        await self._publish_event(
            PlanExecuteEventType.REPLANNING_COMPLETE,
            {"new_plan": new_plan.to_dict(), "subtask_count": len(new_plan.subtasks)},
        )

        logger.info(f"Re-planned with {len(new_plan.subtasks)} subtasks: {new_plan.reasoning}")

        return new_plan

    async def _execute_plan(self, plan: Plan, context: RunContext) -> bool:
        """Execute all subtasks in the plan.

        Args:
            plan: Plan to execute
            context: Run context

        Returns:
            True if all subtasks succeeded, False if any failed
        """
        logger.info(f"Executing plan with {len(plan.subtasks)} subtasks")

        # Track completed subtasks for dependency checking
        completed_subtask_ids = set()

        for subtask in plan.subtasks:
            # Check dependencies
            if not self._dependencies_met(subtask, completed_subtask_ids):
                logger.warning(
                    f"Subtask {subtask.id} dependencies not met, skipping: {subtask.dependencies}"
                )
                subtask.status = "failed"
                subtask.error = "Dependencies not satisfied"
                continue

            # Execute subtask (pass total count to conditionally show subtask headers)
            success = await self._execute_subtask(
                subtask, context, total_subtasks=len(plan.subtasks)
            )

            if success:
                completed_subtask_ids.add(subtask.id)
            else:
                # Subtask failed - stop execution and trigger re-planning
                logger.warning(f"Subtask {subtask.id} failed: {subtask.error}")
                return False

            # Publish progress
            await self._publish_event(
                PlanExecuteEventType.PLAN_PROGRESS,
                {
                    "completed": len(completed_subtask_ids),
                    "total": len(plan.subtasks),
                    "progress_percentage": (len(completed_subtask_ids) / len(plan.subtasks)) * 100,
                },
            )

        return True

    async def _execute_subtask(
        self, subtask: SubTask, context: RunContext, total_subtasks: int = 1
    ) -> bool:
        """Execute a single subtask.

        Args:
            subtask: Subtask to execute
            context: Run context
            total_subtasks: Total number of subtasks in the plan (for conditional rendering)

        Returns:
            True if successful, False otherwise
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

        try:
            if self.use_react_for_subtasks and self.subtask_orchestrator:
                # Use ReAct orchestrator for complex subtasks with event streaming
                logger.info(f"Executing subtask {subtask.id} with ReAct: {subtask.description}")

                # Create event forwarder to bubble up ReAct events as PlanExecute events
                def forward_react_event(react_event):
                    """Forward ReAct events from subtask as PlanExecute subtask events."""
                    try:
                        # Convert ReActEvent to PlanExecuteEvent with subtask context
                        import asyncio

                        if react_event.event_type == ReActEventType.THINKING_CHUNK:
                            # Create subtask thinking chunk event
                            asyncio.create_task(
                                self._publish_event(
                                    PlanExecuteEventType.SUBTASK_THINKING_CHUNK,
                                    {
                                        "subtask_id": subtask.id,
                                        "delta": react_event.data.get("delta", ""),
                                        "thought": react_event.data.get("content", ""),
                                    },
                                )
                            )
                        elif react_event.event_type == ReActEventType.ACTION_PLANNED:
                            # Tool is about to be called
                            tool_name = react_event.data.get("action", "unknown")
                            asyncio.create_task(
                                self._publish_event(
                                    PlanExecuteEventType.SUBTASK_THINKING_CHUNK,
                                    {
                                        "subtask_id": subtask.id,
                                        "delta": "",
                                        "is_tool": True,
                                        "is_tool_planned": True,
                                        "tool_name": tool_name,
                                    },
                                )
                            )
                        elif react_event.event_type == ReActEventType.ACTION_EXECUTING:
                            # Tool is currently executing
                            tool_name = react_event.data.get("action", "unknown")
                            asyncio.create_task(
                                self._publish_event(
                                    PlanExecuteEventType.SUBTASK_THINKING_CHUNK,
                                    {
                                        "subtask_id": subtask.id,
                                        "delta": "",
                                        "is_tool": True,
                                        "is_tool_executing": True,
                                        "tool_name": tool_name,
                                    },
                                )
                            )
                        elif react_event.event_type == ReActEventType.OBSERVATION:
                            # Tool execution result
                            observation = str(react_event.data.get("observation", ""))
                            tool_name = react_event.data.get("action", "unknown")
                            asyncio.create_task(
                                self._publish_event(
                                    PlanExecuteEventType.SUBTASK_THINKING_CHUNK,
                                    {
                                        "subtask_id": subtask.id,
                                        "delta": observation,
                                        "is_observation": True,
                                        "tool_name": tool_name,
                                        "success": react_event.data.get("success", True),
                                    },
                                )
                            )
                    except Exception as e:
                        logger.error(f"Error forwarding ReAct event: {e}")

                # Subscribe to subtask orchestrator's events
                self.subtask_orchestrator.event_bus.subscribe(forward_react_event)

                try:
                    # Execute subtask - events will be forwarded automatically
                    result = await self.subtask_orchestrator.execute(subtask.description, context)

                    subtask.result = result.final_answer
                    subtask.cost = result.total_cost
                    subtask.tokens_used = result.total_tokens
                    subtask.status = "completed"
                finally:
                    # Unsubscribe to avoid memory leaks
                    self.subtask_orchestrator.event_bus.unsubscribe(forward_react_event)

            else:
                # Direct tool execution (simpler, faster)
                logger.info(f"Executing subtask {subtask.id} directly: {subtask.description}")

                # For now, we'll use a simple approach: execute the first required tool
                # In a real implementation, you might parse the description to determine the tool and inputs
                if subtask.required_tools:
                    tool_name = subtask.required_tools[0]
                    # This is simplified - in practice, you'd need to extract tool inputs from the description
                    # or use an LLM to determine inputs
                    tool_result = await self.tool_executor.execute_tool(
                        tool_name, {}, context=context
                    )

                    if tool_result.success:
                        subtask.result = str(tool_result.output)
                        subtask.status = "completed"
                    else:
                        subtask.error = tool_result.error
                        subtask.status = "failed"
                else:
                    # No tools required - might be a reasoning or synthesis task
                    # Use LLM to complete the subtask
                    messages = [
                        Message(
                            role=MessageRole.USER,
                            content=f"Complete this task: {subtask.description}",
                        )
                    ]
                    response = await self.tool_executor._client.achat(messages=messages)
                    subtask.result = response.message.content
                    subtask.status = "completed"

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

            return True

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

            return False

    async def _synthesize_results(self, plan: Plan, query: str, context: RunContext) -> str:
        """Synthesize subtask results into final answer.

        Args:
            plan: Executed plan with results
            query: Original user query
            context: Run context

        Returns:
            Final answer string
        """
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

        Args:
            json_str: JSON string from LLM (may include markdown, thinking text, etc.)

        Returns:
            Parsed plan data as dict with 'reasoning' and 'subtasks' keys
        """
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

            plan_data = json.loads(json_str)
            return plan_data

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse plan JSON: {e}")
            logger.debug(f"JSON string was: {json_str[:500]}...")
            # Return empty plan data on parse failure
            return {"reasoning": f"Failed to parse plan: {str(e)}", "subtasks": []}

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
        return {"agent_type": "plan_and_execute_orchestrator"}
