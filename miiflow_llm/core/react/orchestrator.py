"""Focused ReAct orchestrator with clean separation of concerns."""

import logging
import time
from typing import Any, Dict, Optional

from ..agent import RunContext
from ..message import Message, MessageRole
from .data import REACT_SYSTEM_PROMPT, ReActResult, ReActStep, StopReason
from .events import EventBus, EventFactory
from .parsing.xml_parser import XMLReActParser
from .safety import SafetyManager
from .tool_executor import AgentToolExecutor

logger = logging.getLogger(__name__)


class ReActOrchestrator:
    """ReAct orchestrator with clean separation of concerns."""

    def __init__(
        self,
        tool_executor: AgentToolExecutor,
        event_bus: EventBus,
        safety_manager: SafetyManager,
        parser: XMLReActParser,
    ):
        self.tool_executor = tool_executor
        self.event_bus = event_bus
        self.safety_manager = safety_manager
        self.parser = parser

    async def execute(self, query: str, context: RunContext) -> ReActResult:
        execution_state = ExecutionState()

        try:
            self._setup_context(query, context)
            while execution_state.is_running:
                execution_state.current_step += 1
                if await self._should_stop(execution_state):
                    break
                step = await self._execute_reasoning_step(context, execution_state)
                execution_state.steps.append(step)

                if step.is_final_step:
                    execution_state.final_answer = step.answer
                    await self._publish_final_answer_event(step, execution_state)
                    break

                if step.is_error_step:
                    break

            return self._build_result(execution_state)

        except Exception as e:
            logger.error(f"ReAct execution failed: {e}", exc_info=True)
            return self._build_error_result(execution_state, e)

    def _setup_context(self, query: str, context: RunContext):
        """Setup context with system prompt and user query.

        Args:
            query: User query string (must be non-empty)
            context: Run context with messages list

        Raises:
            ValueError: If query is empty or context is invalid
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if not hasattr(context, "messages"):
            raise ValueError("Context must have a messages attribute")

        if context.messages is None:
            context.messages = []

        tools_info = self.tool_executor.build_tools_description()
        system_prompt = REACT_SYSTEM_PROMPT.format(tools=tools_info)

        messages = [Message(role=MessageRole.SYSTEM, content=system_prompt)]
        messages.extend(context.messages)
        messages.append(Message(role=MessageRole.USER, content=query))
        context.messages = messages

    async def _should_stop(self, state: "ExecutionState") -> bool:
        """Check safety conditions."""
        stop_condition = self.safety_manager.should_stop(state.steps, state.current_step)
        if stop_condition:
            event = EventFactory.stop_condition(
                state.current_step,
                stop_condition.get_stop_reason().value,
                stop_condition.get_description(),
            )
            await self.event_bus.publish(event)
            return True
        return False

    async def _execute_reasoning_step(
        self, context: RunContext, state: "ExecutionState"
    ) -> ReActStep:
        """Execute a single reasoning step with XML streaming parsing."""
        step = ReActStep(step_number=state.current_step, thought="")

        step_start_time = time.time()

        try:
            # DEBUG: Log conversation context before LLM call
            logger.debug(
                f"Step {state.current_step} - Conversation context has {len(context.messages)} messages:"
            )
            for i, msg in enumerate(context.messages[-5:]):  # Last 5 messages
                logger.debug(f"  [{i}] {msg.role}: {msg.content[:100]}...")

            # Publish step start event
            await self.event_bus.publish(EventFactory.step_started(state.current_step))

            # Stream LLM response with real-time XML parsing
            buffer = ""
            tokens_used = 0
            cost = 0.0
            answer_detected = False

            # Reset parser for new response
            self.parser.reset()

            async for chunk in self._stream_llm_response(context):
                if chunk.delta:
                    buffer += chunk.delta

                    # Debug: Log chunks as they arrive
                    if len(buffer) <= 50:
                        logger.debug(
                            f"Step {state.current_step} - Chunk received, buffer now: {buffer}"
                        )
                    elif len(buffer) == 51:
                        logger.debug(
                            f"Step {state.current_step} - Buffer growing, now {len(buffer)} chars..."
                        )

                    # Parse XML incrementally
                    from .parsing.xml_parser import ParseEventType

                    for parse_event in self.parser.parse_streaming(chunk.delta):
                        if parse_event.event_type == ParseEventType.THINKING:
                            # Thinking chunk detected - emit streaming chunk
                            delta = parse_event.data["delta"]
                            await self.event_bus.publish(
                                EventFactory.thinking_chunk(state.current_step, delta, buffer)
                            )

                        elif parse_event.event_type == ParseEventType.THINKING_COMPLETE:
                            # Complete thinking extracted
                            step.thought = parse_event.data["thought"]
                            # Publish complete thought event
                            await self.event_bus.publish(
                                EventFactory.thought(state.current_step, step.thought)
                            )

                        elif parse_event.event_type == ParseEventType.TOOL_CALL:
                            # Tool call detected
                            step.action = parse_event.data["tool_name"]
                            step.action_input = parse_event.data["parameters"]

                        elif parse_event.event_type == ParseEventType.ANSWER_START:
                            # <answer> tag detected - enter streaming answer mode
                            answer_detected = True
                            state.ready_for_answer = True

                        elif parse_event.event_type == ParseEventType.ANSWER_CHUNK:
                            # Stream answer chunks in real-time
                            delta = parse_event.data["delta"]
                            if not hasattr(state, "accumulated_answer"):
                                state.accumulated_answer = ""
                            state.accumulated_answer += delta
                            # Emit streaming chunk event
                            await self.event_bus.publish(
                                EventFactory.final_answer_chunk(
                                    state.current_step, delta, state.accumulated_answer
                                )
                            )

                        elif parse_event.event_type == ParseEventType.ANSWER_COMPLETE:
                            # Answer complete
                            step.answer = parse_event.data["answer"]

                # Accumulate metrics
                if chunk.usage:
                    tokens_used = chunk.usage.total_tokens
                if hasattr(chunk, "cost"):
                    cost += chunk.cost

            # DEBUG: Log what we received
            logger.debug(
                f"Step {state.current_step} - Stream completed. Buffer length: {len(buffer)}, tokens: {tokens_used}"
            )
            if buffer:
                logger.debug(f"Step {state.current_step} - Buffer content: {buffer[:500]}")

            # Set accumulated metrics
            step.tokens_used = tokens_used
            step.cost = cost

            # Check if parser successfully parsed any content (thinking, tool calls, or answer)
            has_content = self.parser.has_parsed_content
            has_valid_step = step.thought or step.action or step.answer
            has_actionable_content = step.action or step.answer

            logger.debug(
                f"Step {state.current_step} validation - "
                f"has_parsed_content: {has_content}, "
                f"has_valid_step: {has_valid_step}, "
                f"has_actionable: {has_actionable_content}, "
                f"buffer_len: {len(buffer)}"
            )

            # Warn if we have thinking but NO action or answer (stuck in thinking-only mode)
            if step.thought and not has_actionable_content:
                logger.warning(
                    f"Step {state.current_step}: LLM output only <thinking> without <tool_call> or <answer>. "
                    f"This violates the ReAct pattern. The agent may be stuck."
                )
            elif not has_content and not has_valid_step and not buffer.strip():
                # Completely empty response
                logger.warning(
                    f"Step {state.current_step}: LLM returned completely empty response. "
                    f"No thinking, action, or answer detected."
                )

            # CRITICAL FIX: Always preserve the assistant's response in conversation history
            # This ensures proper action-observation pairing for the LLM
            assistant_content = buffer.strip()

            # If buffer is empty but we parsed content, reconstruct from parsed data
            if not assistant_content and (step.thought or step.action or step.answer):
                logger.debug(f"Reconstructing assistant message from parsed components")
                parts = []
                if step.thought:
                    parts.append(f"<thinking>\n{step.thought}\n</thinking>")
                if step.action:
                    # Reconstruct tool call XML
                    params_str = ""
                    if step.action_input:
                        if isinstance(step.action_input, dict):
                            params_items = [f"<{k}>{v}</{k}>" for k, v in step.action_input.items()]
                            params_str = "\n".join(params_items)
                        else:
                            params_str = str(step.action_input)
                    parts.append(
                        f"<tool_call>\n<tool_name>{step.action}</tool_name>\n<parameters>\n{params_str}\n</parameters>\n</tool_call>"
                    )
                if step.answer:
                    parts.append(f"<answer>\n{step.answer}\n</answer>")
                assistant_content = "\n".join(parts)

            # Append assistant's message to maintain conversation flow
            if assistant_content:
                response_message = Message(role=MessageRole.ASSISTANT, content=assistant_content)
                context.messages.append(response_message)
                logger.debug(f"Added assistant message to context: {assistant_content[:100]}...")
            else:
                logger.warning(f"Step {state.current_step}: No assistant content to add to context")

            # Handle tool action if detected (AFTER adding assistant message)
            if step.action and not answer_detected:
                await self._handle_tool_action(step, context, state)

        except Exception as e:
            self._handle_step_error(step, e, state)

        finally:
            step.execution_time = time.time() - step_start_time
            await self.event_bus.publish(EventFactory.step_complete(state.current_step, step))

        # Add observation to context
        # IMPORTANT: Use USER role for observations, not SYSTEM
        # Anthropic API only allows SYSTEM messages at the start of conversation
        if step.observation:
            observation_message = Message(
                role=MessageRole.USER,
                content=f"<observation>\n{step.observation}\n</observation>",
            )
            context.messages.append(observation_message)
            logger.debug(
                f"Step {state.current_step} - Added observation to context: {step.observation[:100]}..."
            )
            logger.debug(
                f"Step {state.current_step} - Context now has {len(context.messages)} messages"
            )

        return step

    async def _get_llm_response(self, context: RunContext):
        """Get LLM response without tools (pure ReAct pattern)."""
        return await self.tool_executor.execute_without_tools(messages=context.messages)

    async def _stream_llm_response(self, context: RunContext):
        """Stream LLM response without tools (pure ReAct pattern)."""
        async for chunk in self.tool_executor.stream_without_tools(messages=context.messages):
            yield chunk

    async def _handle_tool_action(
        self, step: ReActStep, context: RunContext, state: "ExecutionState"
    ):
        """Handle tool action execution."""
        # step.action and step.action_input are already set from parsed data

        # Publish action events
        await self.event_bus.publish(
            EventFactory.action_planned(state.current_step, step.action, step.action_input)
        )

        await self.event_bus.publish(
            EventFactory.action_executing(state.current_step, step.action, step.action_input)
        )

        # Execute tool
        try:
            result = await self._execute_tool(step, context)

            if result.success:
                step.observation = str(result.output)
            else:
                step.error = result.error
                step.observation = f"Tool execution failed: {result.error}"

            # Update metrics
            step.cost += getattr(result, "cost", 0.0)
            step.execution_time += result.execution_time

            # Publish observation event
            await self.event_bus.publish(
                EventFactory.observation(
                    state.current_step, step.observation, step.action, result.success
                )
            )

        except Exception as e:
            step.error = f"Tool execution error: {str(e)}"
            step.observation = f"Tool '{step.action}' failed: {str(e)}"
            logger.error(f"Tool execution failed: {e}", exc_info=True)

            await self.event_bus.publish(
                EventFactory.observation(state.current_step, step.observation, step.action, False)
            )

    async def _execute_tool(self, step: ReActStep, context: RunContext):
        """Execute tool with proper context injection."""
        # Check if tool exists first
        if not self.tool_executor.has_tool(step.action):
            # Try fuzzy matching as fallback for common LLM hallucinations
            corrected_name = self._find_similar_tool(step.action)
            if corrected_name:
                logger.warning(
                    f"Tool '{step.action}' not found, auto-correcting to '{corrected_name}'"
                )
                step.action = corrected_name
            else:
                available_tools = self.tool_executor.list_tools()
                step.error = f"Tool '{step.action}' not found. Available: {available_tools}"
                raise Exception(step.error)

        if step.action_input is None:
            step.action_input = {}

        # Ensure action_input is a dictionary
        if not isinstance(step.action_input, dict):
            # For single-parameter tools, infer the parameter name
            tool_schema = self.tool_executor.get_tool_schema(step.action)
            params = tool_schema.get("parameters", {}).get("properties", {})
            if len(params) == 1:
                param_name = next(iter(params.keys()))
                step.action_input = {param_name: step.action_input}
            else:
                raise Exception(
                    f"Tool '{step.action}' expects dict input but got: {step.action_input}"
                )

        # Determine if tool needs context injection
        needs_context = self.tool_executor.tool_needs_context(step.action)

        # Execute tool with or without context based on tool's requirements
        return await self.tool_executor.execute_tool(
            step.action, step.action_input, context=context if needs_context else None
        )

    def _handle_step_error(self, step: ReActStep, error: Exception, state: "ExecutionState"):
        """Handle step execution errors."""
        step.error = f"Step execution failed: {str(error)}"
        step.observation = f"An error occurred: {str(error)}"
        logger.error(f"Step {state.current_step} failed: {error}", exc_info=True)

    async def _publish_final_answer_event(self, step: ReActStep, state: "ExecutionState"):
        """Publish final answer event.

        Note: With XML streaming, answer chunks are already emitted during parsing.
        This method is called for final confirmation only.
        """
        # If answer wasn't streamed (error case), publish complete answer
        if step.answer and not hasattr(state, "accumulated_answer"):
            await self.event_bus.publish(EventFactory.final_answer(state.current_step, step.answer))

    def _build_result(self, state: "ExecutionState") -> ReActResult:
        """Build successful result."""
        # Determine stop reason
        if state.final_answer:
            stop_reason = StopReason.ANSWER_COMPLETE
        else:
            stop_reason = StopReason.FORCED_STOP
            state.final_answer = self._generate_fallback_answer(state.steps)

        # Calculate totals
        total_time = time.time() - state.start_time
        total_cost = sum(step.cost for step in state.steps)
        total_tokens = sum(step.tokens_used for step in state.steps)

        return ReActResult(
            steps=state.steps,
            final_answer=state.final_answer,
            stop_reason=stop_reason,
            total_cost=total_cost,
            total_execution_time=total_time,
            total_tokens=total_tokens,
        )

    def _build_error_result(self, state: "ExecutionState", error: Exception) -> ReActResult:
        """Build error result."""
        return ReActResult(
            steps=state.steps,
            final_answer=f"Error occurred during execution: {str(error)}",
            stop_reason=StopReason.FORCED_STOP,
        )

    def _generate_fallback_answer(self, steps) -> str:
        """Generate fallback answer when no explicit answer is provided."""
        if not steps:
            return "No reasoning steps were completed."

        last_step = steps[-1]
        if last_step.observation:
            return f"Based on the available information: {last_step.observation}"
        elif last_step.thought:
            return f"My reasoning: {last_step.thought}"
        else:
            return "Unable to provide a complete answer due to execution issues."

    def _find_similar_tool(self, requested_name: str) -> Optional[str]:
        """Find a similar tool name using fuzzy matching.

        This helps auto-correct common LLM hallucinations like:
        - "Add" -> "Addition"
        - "Multiply" -> "Multiplication"
        - Case variations

        Args:
            requested_name: The tool name requested by the LLM

        Returns:
            Corrected tool name if a good match is found, None otherwise
        """
        available_tools = self.tool_executor.list_tools()
        requested_lower = requested_name.lower()

        # Strategy 1: Check if requested name is a substring of any available tool (case-insensitive)
        for tool_name in available_tools:
            tool_lower = tool_name.lower()
            # Check if one is a prefix/suffix of the other
            if requested_lower in tool_lower or tool_lower in requested_lower:
                # Prefer longer names (e.g., "Addition" over "Add")
                if len(tool_name) >= len(requested_name):
                    return tool_name

        # Strategy 2: Simple Levenshtein-inspired check for very similar names
        # (e.g., off by 1-2 characters due to typos)
        for tool_name in available_tools:
            if self._is_similar_enough(requested_name, tool_name):
                return tool_name

        return None

    def _is_similar_enough(self, s1: str, s2: str, threshold: int = 2) -> bool:
        """Check if two strings are similar enough (simple edit distance check).

        Args:
            s1: First string
            s2: Second string
            threshold: Maximum allowed differences

        Returns:
            True if strings are within threshold edits of each other
        """
        s1_lower = s1.lower()
        s2_lower = s2.lower()

        # Quick length check - if lengths differ by more than threshold, not similar
        if abs(len(s1_lower) - len(s2_lower)) > threshold:
            return False

        # Simple character difference count (not true edit distance, but faster)
        max_len = max(len(s1_lower), len(s2_lower))
        differences = sum(
            1
            for i in range(max_len)
            if i >= len(s1_lower) or i >= len(s2_lower) or s1_lower[i] != s2_lower[i]
        )

        return differences <= threshold

    def get_current_status(self) -> Dict[str, Any]:
        """Get current execution status."""
        return {"agent_type": "react_orchestrator"}


class ExecutionState:
    """Simple state container for execution tracking."""

    def __init__(self):
        self.current_step = 0
        self.steps = []
        self.start_time = time.time()
        self.is_running = True
        self.final_answer = None
