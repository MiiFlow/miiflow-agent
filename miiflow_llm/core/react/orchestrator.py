"""Focused ReAct orchestrator with clean separation of concerns."""

import logging
import re
import time
from typing import Any, Dict, Optional

from ..agent import RunContext
from ..message import Message, MessageRole
from .enums import StopReason
from .events import EventBus, EventFactory
from .execution import ExecutionState
from .models import ReActResult, ReActStep
from .parsing.xml_parser import XMLReActParser
from .safety import SafetyManager
from .tool_executor import AgentToolExecutor

logger = logging.getLogger(__name__)


def _strip_xml_tags_from_answer(content: str) -> str:
    """Strip XML thinking/answer tags from content to get clean answer text.

    This handles cases where LLM outputs raw XML tags that weren't parsed properly
    during streaming, ensuring users see clean content without internal markup.
    """
    if not content:
        return content

    # Remove <thinking>...</thinking> blocks entirely
    content = re.sub(r"<thinking>.*?</thinking>", "", content, flags=re.DOTALL | re.IGNORECASE)

    # Remove standalone opening/closing tags that might remain
    content = re.sub(r"</?thinking>", "", content, flags=re.IGNORECASE)
    content = re.sub(r"</?answer>", "", content, flags=re.IGNORECASE)

    # Clean up extra whitespace from removed tags
    content = re.sub(r"\n\s*\n\s*\n", "\n\n", content)

    return content.strip()


def _sanitize_error_message(error_msg: str) -> str:
    """Sanitize error messages by removing stack traces and technical details.

    Keeps the error message user-friendly while preserving enough context
    for the LLM to understand what went wrong.
    """
    if not error_msg:
        return "Unknown error occurred"

    # Split by common stack trace indicators
    lines = error_msg.split("\n")
    sanitized_lines = []

    for line in lines:
        # Skip lines that look like stack traces
        if any(
            indicator in line
            for indicator in [
                "Traceback (most recent call last)",
                'File "',
                "line ",
                "  at ",
                "Stack trace:",
                "^",  # Often used to point to error location
            ]
        ):
            continue

        # Skip lines with only whitespace or technical markers
        if not line.strip() or line.strip() in ["---", "==="]:
            continue

        sanitized_lines.append(line.strip())

    # If we filtered everything out, return the first line of the original
    if not sanitized_lines:
        return lines[0] if lines else "Unknown error occurred"

    # Join and limit length
    result = " ".join(sanitized_lines)
    if len(result) > 500:
        result = result[:500] + "..."

    return result


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

                # Execute reasoning step with native tool calling
                step = await self._execute_reasoning_step_native(context, execution_state)

                execution_state.steps.append(step)

                if step.is_final_step:
                    execution_state.final_answer = step.answer
                    await self._publish_final_answer_event(step, execution_state)
                    break

                # Don't break on error steps - let LLM see the error observation
                # and continue reasoning to provide a natural response
                # if step.is_error_step:
                #     break

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
            ValueError: If query is empty AND no user message in context
        """
        # Query can be empty if user message is already in context.messages
        # Check if there's at least a user message in context
        if not query or not query.strip():
            # Allow empty query if there's already a user message in context
            has_user_message = any(msg.role == MessageRole.USER for msg in (context.messages or []))
            if not has_user_message:
                raise ValueError("Query cannot be empty when no user message exists in context")

        if not hasattr(context, "messages"):
            raise ValueError("Context must have a messages attribute")

        if context.messages is None:
            context.messages = []

        # Native tool calling: tools are sent via API's tools parameter,
        # so we don't need to include them in the system prompt
        from .prompts import REACT_NATIVE_SYSTEM_PROMPT

        system_prompt = REACT_NATIVE_SYSTEM_PROMPT

        # Check for existing system prompt in context and merge if needed
        existing_system_prompts = [
            msg for msg in context.messages if msg.role == MessageRole.SYSTEM
        ]
        if existing_system_prompts:
            # Merge assistant's system prompt with ReAct prompt instead of having two separate ones
            assistant_prompt = existing_system_prompts[0].content
            merged_prompt = f"""{assistant_prompt}

---

{system_prompt}"""
            # Remove existing system prompts from context
            context_messages_without_system = [
                msg for msg in context.messages if msg.role != MessageRole.SYSTEM
            ]
            messages = [Message(role=MessageRole.SYSTEM, content=merged_prompt)]
            messages.extend(context_messages_without_system)
        else:
            messages = [Message(role=MessageRole.SYSTEM, content=system_prompt)]
            messages.extend(context.messages)

        # Only append query as a new user message if:
        # 1. Query is not empty AND
        # 2. No user message already exists at the end
        # This prevents duplicate messages when user message is already in context
        if query and query.strip():
            last_msg = messages[-1] if messages else None
            if not last_msg or last_msg.role != MessageRole.USER:
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

    async def _execute_reasoning_step_native(
        self, context: RunContext, state: "ExecutionState"
    ) -> ReActStep:
        """Execute a single reasoning step with native tool calling.

        Native tool calling uses a single LLM call where the model:
        1. Generates thinking/reasoning text in <thinking> tags
        2. Decides whether to call tools (optional)
        3. Provides final answer in <answer> tags when ready
        """
        step = ReActStep(step_number=state.current_step, thought="")
        step_start_time = time.time()

        try:
            # Publish step start event
            await self.event_bus.publish(EventFactory.step_started(state.current_step))

            # Single-phase: Stream LLM response WITH tools enabled
            buffer = ""
            tokens_used = 0
            cost = 0.0
            # Accumulate tool calls during streaming
            accumulated_tool_calls = {}  # index -> {id, function: {name, arguments}}

            # Reset parser for new response
            self.parser.reset()

            logger.debug(f"Step {state.current_step} - Calling LLM with tools enabled")

            async for chunk in self.tool_executor.stream_with_tools(messages=context.messages):
                # Stream text as it arrives
                if chunk.delta:
                    buffer += chunk.delta

                    # Parse XML incrementally to detect <thinking> and <answer> tags
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

                        elif parse_event.event_type == ParseEventType.ANSWER_START:
                            # <answer> tag detected - enter streaming answer mode
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
                            # Answer complete - sanitize to remove any residual XML tags
                            step.answer = _strip_xml_tags_from_answer(parse_event.data["answer"])

                    # Note: If chunk was not parsed (no XML tags), we accumulate in buffer
                    # but don't emit yet since we can't distinguish thinking from answer
                    # without XML tags. We'll classify and emit after streaming completes.

                # Accumulate tool calls if present in chunk
                # All providers now normalize to dict format via stream normalizers
                if chunk.tool_calls:
                    for tool_call_dict in chunk.tool_calls:
                        # All tool calls are now dicts thanks to provider normalizers
                        # Extract index (use 0 for first/only tool in ReAct single-action mode)
                        idx = len(accumulated_tool_calls) if len(accumulated_tool_calls) == 0 else 0

                        # Initialize on first chunk, merge on subsequent chunks
                        if idx not in accumulated_tool_calls:
                            # First chunk: initialize structure
                            accumulated_tool_calls[idx] = {
                                "id": None,
                                "type": "function",
                                "function": {
                                    "name": None,
                                    "arguments": None,  # Will be set based on provider format
                                },
                            }

                        # Update ID if present in this chunk
                        if tool_call_dict.get("id") is not None:
                            accumulated_tool_calls[idx]["id"] = tool_call_dict.get("id")

                        # Update type if present
                        if tool_call_dict.get("type") is not None:
                            accumulated_tool_calls[idx]["type"] = tool_call_dict.get("type")

                        # Update function name if present
                        function_data = tool_call_dict.get("function", {})
                        if function_data.get("name") is not None:
                            accumulated_tool_calls[idx]["function"]["name"] = function_data.get(
                                "name"
                            )

                        # Handle arguments based on format:
                        # - OpenAI: sends progressively longer strings in each chunk
                        # - Anthropic: sends complete dict in final chunk
                        new_args = function_data.get("arguments")
                        if new_args is not None:
                            current_args = accumulated_tool_calls[idx]["function"]["arguments"]

                            if isinstance(new_args, str):
                                # OpenAI format: string that grows with each chunk
                                # Provider already accumulates, so just use the latest value
                                accumulated_tool_calls[idx]["function"]["arguments"] = new_args
                            elif isinstance(new_args, dict):
                                # Anthropic format: dict (usually sent complete in one chunk)
                                if current_args is None or not isinstance(current_args, dict):
                                    accumulated_tool_calls[idx]["function"]["arguments"] = new_args
                                else:
                                    # Merge dicts if both exist (defensive)
                                    current_args.update(new_args)
                            else:
                                # Unexpected format, log and store as-is
                                logger.warning(
                                    f"Unexpected arguments type in chunk: {type(new_args)}"
                                )
                                accumulated_tool_calls[idx]["function"]["arguments"] = new_args

                        logger.debug(f"Tool call accumulated: {accumulated_tool_calls[idx]}")

                # Accumulate metrics
                if chunk.usage:
                    tokens_used = chunk.usage.total_tokens
                if hasattr(chunk, "cost"):
                    cost += chunk.cost

            step.tokens_used = tokens_used
            step.cost = cost

            # Finalize XML parser to flush any remaining buffered content
            # This is critical for handling LLMs that don't output closing </answer> tags
            # The parser holds back 10 chars to handle split tags, which need to be flushed
            from .parsing.xml_parser import ParseEventType

            for parse_event in self.parser.finalize():
                if parse_event.event_type == ParseEventType.ANSWER_CHUNK:
                    delta = parse_event.data["delta"]
                    if not hasattr(state, "accumulated_answer"):
                        state.accumulated_answer = ""
                    state.accumulated_answer += delta
                    await self.event_bus.publish(
                        EventFactory.final_answer_chunk(
                            state.current_step, delta, state.accumulated_answer
                        )
                    )
                elif parse_event.event_type == ParseEventType.ANSWER_COMPLETE:
                    step.answer = _strip_xml_tags_from_answer(parse_event.data["answer"])
                elif parse_event.event_type == ParseEventType.THINKING:
                    delta = parse_event.data["delta"]
                    await self.event_bus.publish(
                        EventFactory.thinking_chunk(state.current_step, delta, buffer)
                    )
                elif parse_event.event_type == ParseEventType.THINKING_COMPLETE:
                    step.thought = parse_event.data["thought"]
                    await self.event_bus.publish(
                        EventFactory.thought(state.current_step, step.thought)
                    )

            # Reconstruct assistant message from buffer
            # This preserves the complete response including XML tags for context
            assistant_content = buffer.strip()

            # Handle accumulated tool calls
            if accumulated_tool_calls:
                # Take first tool call (ReAct is single-action per step)
                tool_call_data = accumulated_tool_calls.get(0)
                if not tool_call_data:
                    tool_call_data = list(accumulated_tool_calls.values())[0]

                # Extract tool name, arguments, and ID from accumulated data
                step.action = tool_call_data["function"]["name"]
                tool_args = tool_call_data["function"]["arguments"]
                tool_call_id = tool_call_data["id"]

                # Parse arguments based on format:
                # - OpenAI: string (JSON) that needs parsing
                # - Anthropic: already a dict
                if tool_args is None:
                    logger.warning(
                        f"Step {state.current_step} - Tool '{step.action}' has None arguments "
                        "(streaming may be incomplete)"
                    )
                    step.action_input = {}
                elif isinstance(tool_args, str):
                    import json

                    # Handle empty string case
                    if not tool_args or tool_args.strip() == "":
                        logger.warning(
                            f"Step {state.current_step} - Tool '{step.action}' has empty arguments string"
                        )
                        step.action_input = {}
                    else:
                        try:
                            step.action_input = json.loads(tool_args)
                        except json.JSONDecodeError as e:
                            logger.error(
                                f"Step {state.current_step} - Failed to parse tool arguments as JSON. "
                                f"Error: {e}. Arguments preview: {tool_args[:200]}..."
                            )
                            step.error = f"Malformed tool arguments: Invalid JSON format"
                            step.action_input = {}
                elif isinstance(tool_args, dict):
                    # Already parsed (Anthropic format)
                    step.action_input = tool_args
                else:
                    logger.error(
                        f"Step {state.current_step} - Unexpected tool_args type: {type(tool_args)}, "
                        f"value: {tool_args}"
                    )
                    step.error = (
                        f"Malformed tool call: arguments type is {type(tool_args).__name__}"
                    )
                    step.action_input = {}

                # Extract __description from action_input (LLM-generated human-readable description)
                # This is used for displaying "Searching for Tesla news" instead of "search_web"
                tool_description = None
                if isinstance(step.action_input, dict):
                    tool_description = step.action_input.pop("__description", None)

                # Validate tool name is not None or empty
                if not step.action:
                    logger.warning(
                        f"Step {state.current_step} - Malformed tool call: function name is None or empty"
                    )
                    step.error = "Malformed tool call: function name is missing"
                    # Add assistant message to context anyway for history
                    response_message = Message(
                        role=MessageRole.ASSISTANT,
                        content=assistant_content,
                        tool_calls=None,
                    )
                    context.messages.append(response_message)
                    # Skip tool execution, continue to next step
                # Validate required parameters are present
                elif step.action_input is not None:
                    # Get tool schema to check required parameters
                    tool_schema = self.tool_executor.get_tool_schema(step.action)
                    required_params = tool_schema.get("parameters", {}).get("required", [])

                    # Check if any required parameters are missing
                    if required_params:
                        missing_params = [
                            param for param in required_params if param not in step.action_input
                        ]

                        if missing_params:
                            logger.error(
                                f"Step {state.current_step} - Tool '{step.action}' missing required parameters: "
                                f"{missing_params}. Provided parameters: {list(step.action_input.keys())}"
                            )
                            step.error = (
                                f"Tool '{step.action}' requires parameters: {', '.join(missing_params)}. "
                                f"This may indicate incomplete streaming or malformed LLM response."
                            )
                            # Add assistant message to context for history
                            response_message = Message(
                                role=MessageRole.ASSISTANT,
                                content=assistant_content,
                                tool_calls=None,
                            )
                            context.messages.append(response_message)
                            # Skip tool execution
                        else:
                            # All required parameters present, execute tool
                            # Add assistant message with both text and tool calls to context
                            tool_calls_list = list(accumulated_tool_calls.values())

                            response_message = Message(
                                role=MessageRole.ASSISTANT,
                                content=assistant_content,
                                tool_calls=tool_calls_list,
                            )
                            context.messages.append(response_message)

                            # Execute the tool
                            await self._handle_tool_action(
                                step, context, state, tool_call_id=tool_call_id, tool_description=tool_description
                            )
                    else:
                        # No required parameters, safe to execute
                        # Add assistant message with both text and tool calls to context
                        tool_calls_list = list(accumulated_tool_calls.values())

                        response_message = Message(
                            role=MessageRole.ASSISTANT,
                            content=assistant_content,
                            tool_calls=tool_calls_list,
                        )
                        context.messages.append(response_message)

                        # Execute the tool
                        await self._handle_tool_action(
                            step, context, state, tool_call_id=tool_call_id, tool_description=tool_description
                        )
                else:
                    # action_input is None (shouldn't happen, but defensive)
                    logger.error(
                        f"Step {state.current_step} - Tool '{step.action}' has None action_input"
                    )
                    step.error = "Internal error: action_input is None"
                    response_message = Message(
                        role=MessageRole.ASSISTANT,
                        content=assistant_content,
                        tool_calls=None,
                    )
                    context.messages.append(response_message)

            # No tool calls - this is a final answer (with XML tags parsed)
            else:
                logger.debug(f"Step {state.current_step} - No tool calls, final answer provided")

                # Add assistant message to context
                response_message = Message(role=MessageRole.ASSISTANT, content=assistant_content)
                context.messages.append(response_message)

                # If we parsed an answer from <answer> tags, it's already set
                # If not parsed but we have content, it might be a direct answer
                if not step.answer:
                    # Check step.thought first (XML-parsed thinking)
                    if step.thought and await self._is_final_answer(step.thought):
                        step.answer = _strip_xml_tags_from_answer(step.thought)
                    # If no thought but we have buffer content, check if it's a final answer
                    elif not step.thought and assistant_content:
                        if await self._is_final_answer(assistant_content):
                            # Sanitize to remove any <thinking> tags that weren't parsed
                            step.answer = _strip_xml_tags_from_answer(assistant_content)
                            # Note: Chunks were already emitted in real-time during streaming
                            # No need to emit again here

        except Exception as e:
            self._handle_step_error(step, e, state)

        finally:
            step.execution_time = time.time() - step_start_time
            await self.event_bus.publish(EventFactory.step_complete(state.current_step, step))

        # Add observation to context if present (from tool execution)
        # NOTE: This is actually added in _handle_tool_action now with proper tool_call_id
        return step

    async def _classify_response_type(self, content: str) -> str:
        """Use LLM to classify if content is thinking/reasoning or a final answer.

        This is a more robust approach than pattern matching, as it uses
        semantic understanding to classify the content.

        Args:
            content: The content to classify

        Returns:
            "THINKING" if content is reasoning/planning, "ANSWER" if it's a final answer
        """
        if not content or len(content.strip()) == 0:
            return "THINKING"

        # Truncate very long content to save tokens (keep first 500 chars)
        truncated_content = content[:500] + ("..." if len(content) > 500 else "")

        classification_prompt = f"""Classify this AI assistant response as either "THINKING" or "ANSWER".

THINKING: Internal reasoning, planning next steps, analyzing information, deciding what to do
Examples:
- "I need to check the database for this information"
- "Let me analyze the data from the tool result"
- "First, I should verify the user's credentials"

ANSWER: Direct response to user, complete solution, final conclusion, stating facts
Examples:
- "You have 131 accounts in your database"
- "The current stock price is $45.23"
- "Based on the analysis, here are the top 3 recommendations"

Response to classify:
{truncated_content}

Classification (respond with ONLY one word - either "THINKING" or "ANSWER"):"""

        try:
            # Create a simple message for classification
            messages = [Message(role=MessageRole.USER, content=classification_prompt)]

            # Use low temperature for deterministic classification
            # max_tokens=50 gives buffer for models that add extra formatting/whitespace
            response = await self.tool_executor._client.achat(
                messages=messages, temperature=0.0, max_tokens=50
            )

            classification = (response.message.content or "").strip().upper()

            # Validate response
            if "ANSWER" in classification:
                return "ANSWER"
            elif "THINKING" in classification:
                return "THINKING"
            else:
                # Fallback to heuristic if LLM returns empty or unexpected response
                # This is normal behavior, not a warning - some providers may return empty responses
                logger.debug(
                    f"Classification response '{classification}' not recognized, using heuristic"
                )
                return "ANSWER" if self._heuristic_is_final_answer(content) else "THINKING"

        except Exception as e:
            logger.warning(f"LLM classification failed: {e}, falling back to heuristic")
            return "ANSWER" if self._heuristic_is_final_answer(content) else "THINKING"

    def _heuristic_is_final_answer(self, thought: str) -> bool:
        """Fallback heuristic for detecting final answers (improved from old pattern matching).

        This is used as a last resort if LLM classification fails.
        Now uses multiple signals rather than just keywords.
        """
        if not thought or len(thought.strip()) == 0:
            return False

        thought_lower = thought.lower()

        # Signal 1: Strong indicators that this is a final answer
        final_answer_indicators = [
            "the answer is",
            "final answer",
            "in conclusion",
            "to summarize",
            "in summary",
            "therefore, the answer",
            "so the answer",
            "the result is",
            "to answer your question",
            "based on the",
        ]

        for indicator in final_answer_indicators:
            if indicator in thought_lower:
                return True

        # Signal 2: Strong indicators this is NOT a final answer (still thinking)
        needs_more_work_indicators = [
            "i need to",
            "i should",
            "let me",
            "i'll",
            "first, i",
            "next, i",
            "i will",
            "should check",
            "need to verify",
            "let's",
        ]

        for indicator in needs_more_work_indicators:
            if indicator in thought_lower:
                return False

        # Signal 3: Check for declarative statements with data
        import re

        declarative_patterns = [
            r"\byou have\b",
            r"\bthere (?:are|is)\b",
            r"\bthe (?:total|count|number|result) (?:is|are)\b",
            r"\b(?:has|have) \d+\b",
        ]

        for pattern in declarative_patterns:
            if re.search(pattern, thought_lower):
                return True

        # Signal 4: Length heuristic (answers typically longer than thinking)
        if len(thought) > 200:
            return True

        return False

    async def _is_final_answer(self, thought: str) -> bool:
        """Detect if the thought indicates a final answer.

        Uses intelligent LLM-based classification with heuristic fallback.
        This replaces the old brittle pattern matching approach.
        """
        classification = await self._classify_response_type(thought)
        return classification == "ANSWER"

    async def _handle_tool_action(
        self,
        step: ReActStep,
        context: RunContext,
        state: "ExecutionState",
        tool_call_id: Optional[str] = None,
        tool_description: Optional[str] = None,
    ):
        """Handle tool action execution."""
        # step.action and step.action_input are already set from parsed data

        # Resolve tool name BEFORE emitting events to ensure consistency
        # (fuzzy matching may correct LLM hallucinations)
        if not self.tool_executor.has_tool(step.action):
            corrected_name = self._find_similar_tool(step.action)
            if corrected_name:
                logger.warning(
                    f"Tool '{step.action}' not found, auto-correcting to '{corrected_name}'"
                )
                step.action = corrected_name

        # Publish action events with tool_description for human-readable display
        await self.event_bus.publish(
            EventFactory.action_planned(state.current_step, step.action, step.action_input, tool_description)
        )

        await self.event_bus.publish(
            EventFactory.action_executing(state.current_step, step.action, step.action_input, tool_description)
        )

        # Execute tool
        try:
            result = await self._execute_tool(step, context)

            if result.success:
                step.observation = str(result.output)
            else:
                # Sanitize error message for LLM consumption
                sanitized_error = _sanitize_error_message(result.error)
                step.error = result.error  # Keep full error for debugging
                step.observation = f"Tool execution failed: {sanitized_error}"

            # Update metrics
            step.cost += getattr(result, "cost", 0.0)
            step.execution_time += result.execution_time

            # Publish observation event
            await self.event_bus.publish(
                EventFactory.observation(
                    state.current_step, step.observation, step.action, result.success
                )
            )

            # Add tool result to context (required for native tool calling)
            if tool_call_id:
                observation_message = Message(
                    role=MessageRole.TOOL, content=step.observation, tool_call_id=tool_call_id
                )
                context.messages.append(observation_message)
                logger.debug(
                    f"Step {state.current_step} - Added tool result to context with ID: {tool_call_id}"
                )

        except Exception as e:
            # Sanitize error message for LLM consumption
            sanitized_error = _sanitize_error_message(str(e))
            step.error = f"Tool execution error: {str(e)}"  # Keep full error for debugging
            step.observation = f"Tool '{step.action}' failed: {sanitized_error}"
            logger.error(f"Tool execution failed: {e}", exc_info=True)

            await self.event_bus.publish(
                EventFactory.observation(state.current_step, step.observation, step.action, False)
            )

    async def _execute_tool(self, step: ReActStep, context: RunContext):
        """Execute tool with proper context injection."""
        # Tool name should already be resolved by _handle_tool_action
        # Just verify it exists (fuzzy matching was already done if needed)
        if not self.tool_executor.has_tool(step.action):
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
        This method publishes the complete final_answer event for consumers like agent.run()
        that need to capture the complete answer.
        """
        # Always publish final_answer event with complete answer
        # Chunks were streamed incrementally, but agent.run() needs the complete event
        if step.answer:
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
        # Guard against None or empty names
        if not requested_name:
            return None

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
