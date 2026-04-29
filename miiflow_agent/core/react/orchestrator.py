"""Focused ReAct orchestrator with clean separation of concerns."""

import json
import logging
import time
from typing import Any, Dict, List, Optional

from ..agent import RunContext
from ..message import Message, MessageRole
from .enums import ReActEventType, StopReason
from .exceptions import ToolApprovalRequired
from .events import EventBus, EventFactory
from .react_events import ReActEvent
from .execution import ExecutionState
from .models import ReActResult, ReActStep
from .safety import SafetyManager
from .tool_executor import AgentToolExecutor

logger = logging.getLogger(__name__)


def _preview(value: Any, max_len: int = 200) -> str:
    """Render a value for trace logs: single-line, truncated, length-annotated."""
    if value is None:
        return "None"
    if not isinstance(value, str):
        try:
            value = json.dumps(value, default=str, ensure_ascii=False)
        except Exception:
            value = str(value)
    collapsed = value.replace("\n", " ").replace("\r", " ")
    if len(collapsed) > max_len:
        return f"{collapsed[:max_len]!r}… (len={len(value)})"
    return repr(collapsed)


def _summarize_tool_call(tc: Any) -> str:
    """One-line summary of a Message.tool_call (supports dict or object form)."""
    if isinstance(tc, dict):
        tc_id = tc.get("id")
        fn = tc.get("function") or {}
        name = fn.get("name")
        args = fn.get("arguments")
    else:
        tc_id = getattr(tc, "id", None)
        name = getattr(getattr(tc, "function", None), "name", None) or getattr(tc, "name", None)
        args = getattr(getattr(tc, "function", None), "arguments", None) or getattr(tc, "arguments", None)
    return f"{name}(id={tc_id}, args={_preview(args, 160)})"


def _summarize_messages_for_trace(messages: List[Message]) -> str:
    """Render a messages list as a multi-line, truncated trace."""
    lines = []
    for i, msg in enumerate(messages):
        role = getattr(msg.role, "value", str(msg.role))
        tc_id = getattr(msg, "tool_call_id", None)
        tag = f"{role}/{tc_id}" if tc_id else role
        content = getattr(msg, "content", None)
        if isinstance(content, list):
            # Multimodal blocks — summarize their types.
            parts = []
            for b in content:
                btype = getattr(b, "__class__", type(b)).__name__
                text = getattr(b, "text", None)
                if text is not None:
                    parts.append(f"{btype}:{_preview(text, 80)}")
                else:
                    parts.append(btype)
            content_preview = "[" + ", ".join(parts) + "]"
        else:
            content_preview = _preview(content, 240)
        tool_calls = getattr(msg, "tool_calls", None) or []
        tc_preview = ""
        if tool_calls:
            tc_preview = " tool_calls=[" + ", ".join(_summarize_tool_call(tc) for tc in tool_calls) + "]"
        lines.append(f"  [{i}] {tag:<18}: {content_preview}{tc_preview}")
    return "\n".join(lines)


def _observation_with_citation_ref(output: Any) -> str:
    """Convert tool output to observation string, extracting citation ref from dicts.

    If the output is a dict containing a '_citation_ref' key, pop it and prepend
    [ref:label] to the stringified result so the LLM can cite it by label.
    For string results, the label is already prepended by the tool wrapper.
    """
    if isinstance(output, dict) and "_citation_ref" in output:
        ref = output.pop("_citation_ref")
        return f"[ref:{ref}]\n{str(output)}"
    return str(output)


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
    """Action-or-answer agent loop.

    Per turn the model emits exactly one of: tool calls (loop continues) or
    a text answer (loop exits). The prompt enforces the invariant; the
    orchestrator branches on which signal arrives during streaming.
    """

    def __init__(
        self,
        tool_executor: AgentToolExecutor,
        event_bus: EventBus,
        safety_manager: SafetyManager,
        recovery_manager=None,
        context_compressor=None,
        tool_filter=None,
    ):
        self.tool_executor = tool_executor
        self.event_bus = event_bus
        self.safety_manager = safety_manager
        self.recovery_manager = recovery_manager
        self.context_compressor = context_compressor
        self.tool_filter = tool_filter

    async def execute(self, query: str, context: RunContext) -> ReActResult:
        from .progress import ProgressTracker

        execution_state = ExecutionState()
        # Extract max_steps from safety manager (first MaxStepsCondition)
        max_steps = 25
        for cond in self.safety_manager.conditions:
            if hasattr(cond, 'max_steps'):
                max_steps = cond.max_steps
                break
        progress_tracker = ProgressTracker(max_steps=max_steps)

        try:
            self._setup_context(query, context)

            # Apply context compression before starting if available
            if self.context_compressor:
                result = await self.context_compressor.compress_if_needed(context.messages)
                if result.was_compressed:
                    context.messages = result.messages
                    logger.info(f"Pre-execution context compression: {result.original_count} -> {result.compressed_count} messages")

            while execution_state.is_running:
                execution_state.current_step += 1

                # Check for user cancellation
                if context.is_cancelled:
                    logger.info("Execution cancelled by user")
                    execution_state.final_answer = self.full_response if hasattr(self, 'full_response') else ""
                    break

                # If recovery just excluded a tool, give the LLM one turn to
                # respond to the reduced tool set before safety conditions
                # (e.g. ErrorThresholdCondition) can halt the loop. Without
                # this grace turn, SIMPLIFY_TOOLS' exclusion has no effect:
                # the loop halts at the top of the next iteration before the
                # LLM ever gets to produce a call with the new tool list.
                skip_stop_check = getattr(
                    execution_state, "_grace_turn_after_recovery", False
                )
                if skip_stop_check:
                    execution_state._grace_turn_after_recovery = False
                    logger.info(
                        "[ORCH] step=%d grace turn — bypassing safety check after SIMPLIFY_TOOLS",
                        execution_state.current_step,
                    )
                elif await self._should_stop(execution_state):
                    logger.info(
                        "[ORCH] step=%d STOP via safety condition (final_answer=%r, step_count=%d)",
                        execution_state.current_step,
                        execution_state.final_answer,
                        len(execution_state.steps),
                    )
                    break

                # Execute reasoning step with native tool calling.
                # Catch context-overflow / token-limit errors here so the
                # recovery manager (which knows how to compact) gets a chance
                # before we abort the whole run.
                try:
                    step = await self._execute_reasoning_step_native(context, execution_state)
                except Exception as _llm_err:
                    from .recovery import is_context_overflow_error

                    if not is_context_overflow_error(_llm_err) or not self.recovery_manager:
                        raise
                    logger.warning(
                        "LLM call hit context-overflow error; attempting compaction recovery: %s",
                        _llm_err,
                    )
                    recovery_action = await self.recovery_manager.attempt_recovery(
                        error=_llm_err,
                        context=context,
                        step=None,
                        tool_name=None,
                    )
                    if not recovery_action.should_continue:
                        logger.warning(
                            "Context-overflow recovery exhausted; stopping execution"
                        )
                        break
                    if recovery_action.guidance_message:
                        context.messages.append(
                            Message(
                                role=MessageRole.USER,
                                content=recovery_action.guidance_message,
                            )
                        )
                    # Skip the rest of this iteration; the next loop turn will
                    # retry the reasoning step against the (now compacted)
                    # message history.
                    continue

                execution_state.steps.append(step)

                # Update progress tracker and emit progress event
                progress_tracker.record_step(step)
                await self.event_bus.publish(
                    EventFactory.progress(execution_state.current_step, progress_tracker.snapshot())
                )

                # Check if clarification was requested during this step
                if execution_state.needs_clarification:
                    logger.info("Breaking execution loop - clarification requested")
                    break

                # Trace: one line per step summarizing the outcome so the log
                # reads like a ReAct transcript.
                logger.info(
                    "[ORCH] step=%d outcome action=%s error=%s is_final=%s answer=%s obs=%s",
                    execution_state.current_step,
                    step.action,
                    _preview(step.error, 200) if step.error else None,
                    step.is_final_step,
                    _preview(step.answer, 200) if step.answer else None,
                    _preview(step.observation, 200) if step.observation else None,
                )

                if step.is_final_step:
                    execution_state.final_answer = step.answer
                    await self._publish_final_answer_event(step, execution_state)
                    break

                # Recovery: if step had an error, try recovery strategies
                if step.is_error_step and self.recovery_manager:
                    recovery_action = await self.recovery_manager.attempt_recovery(
                        error=Exception(step.error or "Unknown error"),
                        context=context,
                        step=step,
                        tool_name=step.action,
                    )
                    logger.info(
                        "[ORCH] step=%d recovery strategy=%s attempt=%s continue=%s "
                        "excluded_tools=%s guidance=%s",
                        execution_state.current_step,
                        getattr(recovery_action.strategy_used, "value", recovery_action.strategy_used),
                        recovery_action.attempt_number,
                        recovery_action.should_continue,
                        sorted(recovery_action.excluded_tools) if recovery_action.excluded_tools else None,
                        _preview(recovery_action.guidance_message, 240) if recovery_action.guidance_message else None,
                    )
                    if not recovery_action.should_continue:
                        logger.warning("Recovery exhausted, stopping execution")
                        break
                    if recovery_action.guidance_message:
                        context.messages.append(
                            Message(role=MessageRole.USER, content=recovery_action.guidance_message)
                        )
                    if recovery_action.excluded_tools:
                        if self.tool_filter:
                            for tool_name in recovery_action.excluded_tools:
                                self.tool_filter.add_denied(tool_name)
                        # One grace turn so the LLM gets to respond to the
                        # SIMPLIFY_TOOLS guidance (and the narrowed tool set,
                        # if tool_filter is wired) before ErrorThresholdCondition
                        # halts the loop. Must fire even when tool_filter is
                        # absent — the guidance message alone is the only
                        # meaningful signal the model has to break the pattern.
                        execution_state._grace_turn_after_recovery = True
                elif not (step.is_error_step) and self.recovery_manager:
                    self.recovery_manager.record_success()

            result = await self._build_result(execution_state, context)
            result.progress = progress_tracker.snapshot().to_dict()
            return result

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
            # Framework prompt first (higher priority), assistant prompt second (context)
            assistant_prompt = existing_system_prompts[0].content
            merged_prompt = f"""{system_prompt}

---

{assistant_prompt}"""
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

        The model either calls tools (continue loop) or responds with
        plain text (final answer). No explicit thinking mechanism needed.
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
            finish_reason = None

            logger.debug(f"Step {state.current_step} - Calling LLM with tools enabled")

            # Trace: dump everything going INTO the LLM so we can audit the
            # exact messages, tool schemas, and generation settings on each
            # turn. Kept at INFO level (single prefix [LLM_TURN]) so it's easy
            # to grep and filter.
            try:
                _trace_tools = self.tool_executor._build_native_tool_schemas()
                _trace_tool_names = []
                for _t in _trace_tools:
                    # Providers vary: OpenAI nests under function.name, others use top-level name.
                    _fn = _t.get("function") if isinstance(_t, dict) else None
                    _trace_tool_names.append(
                        (_fn or {}).get("name") if _fn else (_t.get("name") if isinstance(_t, dict) else str(_t))
                    )
            except Exception as _trace_err:
                _trace_tool_names = [f"<unavailable: {_trace_err}>"]
            logger.info(
                "[LLM_TURN] step=%d OUT messages=%d tools=%d temp=%s max_tokens=%s\n%s\n  tool_names=%s",
                state.current_step,
                len(context.messages),
                len(_trace_tool_names),
                getattr(self.tool_executor.agent, "temperature", None),
                getattr(self.tool_executor.agent, "max_tokens", None),
                _summarize_messages_for_trace(context.messages),
                _trace_tool_names,
            )

            async for chunk in self.tool_executor.stream_with_tools(messages=context.messages):
                # Native extended thinking (Anthropic thinking blocks, OpenAI
                # reasoning tokens) arrive on a dedicated channel separate
                # from response text. Pass through to the UI's thinking panel;
                # never enters the answer buffer.
                if getattr(chunk, "thinking_delta", None):
                    await self.event_bus.publish(
                        EventFactory.thinking_chunk(
                            state.current_step, chunk.thinking_delta, buffer
                        )
                    )

                # Text deltas. Per the per-turn invariant in prompts.py the
                # model emits text only on answer turns, so stream text live
                # into the final-answer bubble. If a tool call later appears
                # in the same turn (prompt violation), we log and switch
                # routing for any subsequent text — the small amount already
                # streamed is a tolerable UI artifact.
                if chunk.delta:
                    buffer += chunk.delta
                    if accumulated_tool_calls:
                        # Action turn already declared; treat further text as
                        # preamble narration. Emit as thinking_chunk.
                        await self.event_bus.publish(
                            EventFactory.thinking_chunk(
                                state.current_step, chunk.delta, buffer
                            )
                        )
                    else:
                        await self.event_bus.publish(
                            EventFactory.final_answer_chunk(
                                state.current_step, chunk.delta, buffer
                            )
                        )

                # Accumulate tool calls if present in chunk
                # All providers now normalize to dict format via stream normalizers
                if chunk.tool_calls:
                    if buffer and not accumulated_tool_calls:
                        # Frontier models often narrate before a tool call
                        # ("I'll do X"). Logged at INFO for observability;
                        # not actionable on its own.
                        logger.info(
                            "Step %d - Tool-call preamble: model emitted text "
                            "before tool call. text_preview=%s",
                            state.current_step,
                            _preview(buffer, 200),
                        )
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

                        # Preserve function_call_metadata (e.g. Gemini thought_signature)
                        if tool_call_dict.get("function_call_metadata"):
                            accumulated_tool_calls[idx]["function_call_metadata"] = (
                                tool_call_dict["function_call_metadata"]
                            )

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
                if getattr(chunk, "finish_reason", None):
                    finish_reason = chunk.finish_reason

            step.tokens_used = tokens_used
            step.cost = cost

            assistant_content = buffer.strip()

            # Trace: dump the LLM response we got back, including any accumulated
            # tool calls. Pairs with the OUT line emitted before stream_with_tools.
            logger.info(
                "[LLM_TURN] step=%d IN finish_reason=%s tokens=%s cost=%s text=%s tool_calls=[%s]",
                state.current_step,
                finish_reason,
                tokens_used,
                cost,
                _preview(assistant_content, 240),
                ", ".join(_summarize_tool_call(tc) for tc in accumulated_tool_calls.values()),
            )

            # === TURN ROUTING ===
            # Tool calls present = action turn, execute and loop.
            # finish_reason == "length" = response truncated, nudge model.
            # Otherwise = answer turn, accumulated text is the final answer.

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

                # Extract __description from action_input (LLM-generated human-readable description).
                # Format: short verb-led imperative phrase, e.g. "Search the web for Tesla news"
                # — used as a UI label and as the question text for tool approval consent.
                tool_description = None
                if isinstance(step.action_input, dict):
                    tool_description = step.action_input.pop("__description", None)

                # Validate tool name is not None or empty
                if not step.action:
                    logger.warning(
                        f"Step {state.current_step} - Malformed tool call: function name is None or empty"
                    )
                    step.error = "Malformed tool call: function name is missing"
                    # Drop tool_calls: we can't emit a valid tool_use block
                    # without a name. Append a user nudge so the conversation
                    # ends on a user turn (Anthropic rejects trailing
                    # assistant messages on models that don't support prefill).
                    response_message = Message(
                        role=MessageRole.ASSISTANT,
                        content=assistant_content,
                        tool_calls=None,
                    )
                    context.messages.append(response_message)
                    step.observation = (
                        "Previous tool call was malformed (missing function name). "
                        "Retry with a valid tool call or provide a final answer."
                    )
                    context.messages.append(
                        Message(role=MessageRole.USER, content=step.observation)
                    )
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
                            step.observation = step.error
                            # Preserve tool_calls + append a synthetic tool_result
                            # so the LLM sees the error as a normal tool failure
                            # and the conversation ends on a user turn. Fall back
                            # to dropping tool_calls if we have no id to pair
                            # with (Anthropic rejects tool_use blocks without ids).
                            if tool_call_id:
                                tool_calls_list = list(accumulated_tool_calls.values())
                                context.messages.append(
                                    Message(
                                        role=MessageRole.ASSISTANT,
                                        content=assistant_content,
                                        tool_calls=tool_calls_list,
                                    )
                                )
                                context.messages.append(
                                    Message(
                                        role=MessageRole.TOOL,
                                        content=step.observation,
                                        tool_call_id=tool_call_id,
                                    )
                                )
                            else:
                                context.messages.append(
                                    Message(
                                        role=MessageRole.ASSISTANT,
                                        content=assistant_content,
                                        tool_calls=None,
                                    )
                                )
                                context.messages.append(
                                    Message(role=MessageRole.USER, content=step.observation)
                                )
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
                    step.observation = step.error
                    if tool_call_id:
                        tool_calls_list = list(accumulated_tool_calls.values())
                        context.messages.append(
                            Message(
                                role=MessageRole.ASSISTANT,
                                content=assistant_content,
                                tool_calls=tool_calls_list,
                            )
                        )
                        context.messages.append(
                            Message(
                                role=MessageRole.TOOL,
                                content=step.observation,
                                tool_call_id=tool_call_id,
                            )
                        )
                    else:
                        context.messages.append(
                            Message(
                                role=MessageRole.ASSISTANT,
                                content=assistant_content,
                                tool_calls=None,
                            )
                        )
                        context.messages.append(
                            Message(role=MessageRole.USER, content=step.observation)
                        )

            # No tool calls — check if this is truly a final answer or an incomplete response
            elif finish_reason == "length":
                # Response was truncated by max_tokens — LLM was likely about to call a tool
                logger.warning(
                    f"Step {state.current_step} - Response truncated (finish_reason='length'). "
                    f"Continuing loop to give LLM another turn. Preview: {assistant_content[:200]}..."
                )
                # Append the partial response plus a user nudge so the
                # conversation ends on a user turn (Opus 4.7 and similar
                # models reject trailing assistant messages as prefill).
                # Skip the assistant append entirely when there's no content
                # to preserve, so we don't emit an empty message.
                if assistant_content:
                    context.messages.append(
                        Message(role=MessageRole.ASSISTANT, content=assistant_content)
                    )
                context.messages.append(
                    Message(
                        role=MessageRole.USER,
                        content=(
                            "Previous response was truncated by max_tokens. "
                            "Continue from where you left off, call the tool you intended, "
                            "or provide a final answer."
                        ),
                    )
                )
                # Don't set step.answer — loop will continue

            else:
                # Answer turn: no tool calls, no truncation. Streamed text
                # deltas already went out as final_answer_chunk events;
                # set step.answer so is_final_step returns True and the
                # main loop publishes the closing final_answer event.
                logger.debug(
                    f"Step {state.current_step} - No tool calls; treating as final answer"
                )
                context.messages.append(
                    Message(role=MessageRole.ASSISTANT, content=assistant_content)
                )
                step.answer = assistant_content

        except Exception as e:
            self._handle_step_error(step, e, state)

        finally:
            step.execution_time = time.time() - step_start_time
            await self.event_bus.publish(EventFactory.step_complete(state.current_step, step))

        # Add observation to context if present (from tool execution)
        # NOTE: This is actually added in _handle_tool_action now with proper tool_call_id
        return step

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
            result = await self._execute_tool(step, context, state)

            if result.success:
                # Check if this is a visualization result BEFORE stringification
                # This is critical because str(VisualizationResult) returns [VIZ:uuid]
                # which loses the actual chart data
                from miiflow_agent.visualization import is_visualization_result, extract_visualization_data
                from miiflow_agent.visualization.types import (
                    is_media_result, extract_media_data,
                    is_media_collection, extract_media_collection, extract_collection_metadata,
                    is_llm_block_injection, extract_llm_blocks,
                )
                from miiflow_agent.artifacts import (
                    extract_artifact_data,
                    format_artifact_observation,
                    is_artifact_result,
                )

                if is_llm_block_injection(result.output):
                    # Tool wants the LLM to actually see pixels on the next turn.
                    # Queue the raw block dicts; the TOOL-message construction below
                    # will materialize them as multimodal content.
                    inj = extract_llm_blocks(result.output) or {}
                    blocks = inj.get("blocks") or []
                    summary = inj.get("summary") or (
                        f"Injected {len(blocks)} content block(s) for visual analysis."
                    )
                    step.observation = summary
                    state.pending_llm_blocks = list(blocks)
                    logger.info(
                        f"Step {state.current_step} - Queued {len(blocks)} LLM blocks for next turn"
                    )
                elif is_media_collection(result.output):
                    media_items = extract_media_collection(result.output) or []
                    metadata_items = extract_collection_metadata(result.output) or []
                    observation_lines = []
                    for idx, media_data in enumerate(media_items):
                        await self.event_bus.publish(
                            EventFactory.media(state.current_step, media_data, step.action)
                        )
                        media_id = media_data['id']
                        media_url = media_data.get('url', '')
                        if media_url and not media_url.startswith('data:'):
                            state.media_store[media_id] = media_url
                        # Correlate metadata entries by index (tools return parallel lists)
                        meta = metadata_items[idx] if idx < len(metadata_items) else None
                        if meta is not None:
                            try:
                                meta_json = json.dumps(meta, default=str)
                            except Exception:
                                meta_json = str(meta)
                            observation_lines.append(f"[MEDIA:{media_id}] {meta_json}")
                        else:
                            observation_lines.append(
                                f"[MEDIA:{media_id}] media_type={media_data.get('media_type')} "
                                f"url={media_url}"
                            )
                    step.observation = (
                        f"Returned {len(media_items)} media item(s). "
                        "Reference any of them with media_ref:<id>.\n"
                        + "\n".join(observation_lines)
                    )
                    logger.info(
                        f"Step {state.current_step} - Emitted {len(media_items)} media events (collection)"
                    )
                elif is_media_result(result.output):
                    media_data = extract_media_data(result.output)
                    await self.event_bus.publish(
                        EventFactory.media(state.current_step, media_data, step.action)
                    )
                    media_id = media_data['id']
                    media_url = media_data.get('url', '')

                    # Store media URL in execution state so subsequent tool calls
                    # (e.g. image editing) can resolve media_ref:<id> to actual URL.
                    # Only store actual URLs, not data URIs (which can be MBs of base64).
                    # System tools already persist to S3 before reaching here, so
                    # media_url should be an S3 URL for normal image gen flows.
                    if media_url and not media_url.startswith('data:'):
                        state.media_store[media_id] = media_url

                    # Always include media_ref so LLM can reference this image
                    # in subsequent tool calls (e.g. edit_gpt_image_1)
                    if media_url and not media_url.startswith('data:'):
                        step.observation = (
                            f"[MEDIA:{media_id}] Image generated successfully. "
                            f"To edit this image, use media_ref:{media_id} as the image parameter. "
                            f"Image URL: {media_url}"
                        )
                    else:
                        step.observation = (
                            f"[MEDIA:{media_id}] Image generated successfully. "
                            f"To edit this image, use media_ref:{media_id} as the image parameter."
                        )
                    logger.info(
                        f"Step {state.current_step} - Emitted media event: "
                        f"id={media_id}, type={media_data.get('media_type')}"
                    )
                elif is_visualization_result(result.output):
                    viz_data = extract_visualization_data(result.output)
                    if viz_data:
                        # Emit visualization event with full data BEFORE stringification
                        await self.event_bus.publish(
                            EventFactory.visualization(state.current_step, viz_data, step.action)
                        )
                        # Store marker for observation (what gets sent to LLM context)
                        # For auth_prompt visualizations, include context so LLM knows tool was blocked
                        if viz_data.get("type") == "auth_prompt":
                            provider_name = viz_data.get("data", {}).get("providerName", "the provider")
                            step.observation = (
                                f"[VIZ:{viz_data['id']}] "
                                f"Tool was blocked: authentication required for {provider_name}. "
                                f"A connect prompt has been shown to the user. "
                                f"Do not proceed with this provider's tools until the user connects their account."
                            )
                        else:
                            step.observation = f"[VIZ:{viz_data['id']}]"
                        logger.info(
                            f"Step {state.current_step} - Emitted visualization event: "
                            f"type={viz_data.get('type')}, id={viz_data.get('id')}"
                        )
                    else:
                        # Extraction failed, fall back to string
                        step.observation = _observation_with_citation_ref(result.output)
                elif is_artifact_result(result.output):
                    artifact_data = extract_artifact_data(result.output)
                    if artifact_data:
                        await self.event_bus.publish(
                            EventFactory.artifact(state.current_step, artifact_data, step.action)
                        )
                        step.observation = format_artifact_observation(artifact_data)
                        logger.info(
                            f"Step {state.current_step} - Emitted artifact event: "
                            f"kind={artifact_data.get('kind')}, id={artifact_data.get('id')}"
                        )
                    else:
                        step.observation = _observation_with_citation_ref(result.output)
                else:
                    step.observation = _observation_with_citation_ref(result.output)

                # Check if this is a clarification request
                from ..tools.clarification import is_clarification_result, extract_clarification_data

                if is_clarification_result(result):
                    clarification = extract_clarification_data(result)
                    if clarification:
                        # Mark state for clarification
                        state.needs_clarification = True
                        state.clarification_data = clarification.to_dict()
                        state.clarification_data["tool_call_id"] = tool_call_id
                        logger.info(
                            f"Clarification requested: {clarification.question}"
                        )

                        # Emit clarification event
                        await self.event_bus.publish(
                            ReActEvent(
                                event_type=ReActEventType.CLARIFICATION_NEEDED,
                                step_number=state.current_step,
                                data={
                                    "step": state.current_step,
                                    "question": clarification.question,
                                    "options": clarification.options,
                                    "context": clarification.context,
                                    "allow_free_text": clarification.allow_free_text,
                                    "tool_call_id": tool_call_id,
                                },
                            )
                        )
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

            # Add tool result to context (required for native tool calling).
            # If the tool returned LlmBlockInjection, the orchestrator stashed
            # raw block dicts on state.pending_llm_blocks — attach them as
            # multimodal TOOL-message content so the next LLM turn sees the
            # actual pixels rather than a URL string.
            if tool_call_id:
                if state.pending_llm_blocks:
                    from ..message import ImageBlock, TextBlock, VideoBlock

                    content_blocks: List[Any] = [TextBlock(text=step.observation or "")]
                    for b in state.pending_llm_blocks:
                        btype = b.get("type")
                        if btype == "text":
                            content_blocks.append(TextBlock(text=b.get("text", "")))
                        elif btype == "image_url":
                            content_blocks.append(
                                ImageBlock(
                                    image_url=b.get("image_url", ""),
                                    detail=b.get("detail", "auto"),
                                )
                            )
                        elif btype == "video_url":
                            content_blocks.append(
                                VideoBlock(
                                    video_url=b.get("video_url", ""),
                                    mime_type=b.get("mime_type"),
                                )
                            )
                    observation_message = Message(
                        role=MessageRole.TOOL,
                        content=content_blocks,
                        tool_call_id=tool_call_id,
                    )
                    state.pending_llm_blocks = []
                else:
                    observation_message = Message(
                        role=MessageRole.TOOL,
                        content=step.observation,
                        tool_call_id=tool_call_id,
                    )
                context.messages.append(observation_message)
                logger.debug(
                    f"Step {state.current_step} - Added tool result to context with ID: {tool_call_id}"
                )

        except ToolApprovalRequired as e:
            # Tool requires user approval before execution
            # NOTE: __description was already popped from action_input at line 523
            # and stored in `tool_description`, so use that instead of re-popping
            state.needs_clarification = True  # Reuse existing pause mechanism
            state.clarification_data = {
                "type": "tool_approval",
                "tool_name": e.tool_name,
                "tool_inputs": e.tool_inputs or {},
                "tool_description": tool_description or "",
                "tool_schema": self.tool_executor.get_tool_schema(e.tool_name),
                "tool_call_id": tool_call_id,
                "reason": e.reason,
            }

            # Emit approval event for SSE
            await self.event_bus.publish(
                ReActEvent(
                    event_type=ReActEventType.TOOL_APPROVAL_NEEDED,
                    step_number=state.current_step,
                    data=state.clarification_data,
                )
            )

            logger.info(
                f"Tool '{e.tool_name}' requires approval - pausing execution"
            )

            # CRITICAL: Must add a tool result to context, otherwise Anthropic API
            # rejects with "tool_use ids were found without tool_result blocks"
            if tool_call_id:
                observation_message = Message(
                    role=MessageRole.TOOL,
                    content="Tool execution paused - waiting for user approval.",
                    tool_call_id=tool_call_id,
                )
                context.messages.append(observation_message)

        except Exception as e:
            # Sanitize error message for LLM consumption
            sanitized_error = _sanitize_error_message(str(e))
            step.error = f"Tool execution error: {str(e)}"  # Keep full error for debugging
            step.observation = f"Tool '{step.action}' failed: {sanitized_error}"
            logger.error(f"Tool execution failed: {e}", exc_info=True)

            await self.event_bus.publish(
                EventFactory.observation(state.current_step, step.observation, step.action, False)
            )

            # Add tool result to context even on exception (required for native tool calling)
            # Without this, Anthropic API will reject subsequent calls with:
            # "tool_use ids were found without tool_result blocks"
            if tool_call_id:
                observation_message = Message(
                    role=MessageRole.TOOL, content=step.observation, tool_call_id=tool_call_id
                )
                context.messages.append(observation_message)
                logger.debug(
                    f"Step {state.current_step} - Added error tool result to context with ID: {tool_call_id}"
                )

    async def _execute_tool(self, step: ReActStep, context: RunContext, state: "ExecutionState" = None):
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

        # Resolve media_ref:<id> references in tool inputs to actual URLs
        # This enables image editing tools to reference previously generated images
        if isinstance(step.action_input, dict) and state:
            step.action_input = self._resolve_media_refs(step.action_input, state)

        # Expose media_store on ctx.deps so tools (e.g. analyze_creative) can
        # resolve media_ref IDs without re-implementing the resolution logic.
        # Safe in-place assignment because the orchestrator owns ctx lifecycle
        # for the duration of a ReAct run.
        if state is not None and context is not None:
            try:
                if isinstance(context.deps, dict):
                    context.deps["media_store"] = state.media_store
            except Exception:
                # deps may be a custom type; tools that need media_store use
                # ctx.deps.get which would silently miss — acceptable v1.
                pass

        # Determine if tool needs context injection
        needs_context = self.tool_executor.tool_needs_context(step.action)

        # Execute tool with or without context based on tool's requirements
        return await self.tool_executor.execute_tool(
            step.action, step.action_input, context=context if needs_context else None
        )

    def _resolve_media_refs(self, inputs: dict, state: "ExecutionState") -> dict:
        """Resolve media references in tool inputs to actual URLs.

        Handles multiple reference formats:
        - media_ref:<id>  — explicit reference (preferred)
        - /mnt/data/<uuid>.png — hallucinated sandbox paths (common with GPT-based models)
        - Any non-URL string containing a UUID that matches a stored media ID

        When image generation tools produce media results, the URLs are stored
        in execution state's media_store. This method resolves those references
        to the actual image URLs before tool execution.
        """
        import re

        media_store = state.media_store
        if not media_store:
            return inputs

        resolved = {}
        media_ref_pattern = re.compile(r'^media_ref:(.+)$')
        # Match UUIDs in hallucinated file paths like /mnt/data/<uuid>.png
        uuid_pattern = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', re.IGNORECASE)

        for key, value in inputs.items():
            if isinstance(value, str):
                stripped = value.strip()

                # 1. Explicit media_ref:<id>
                match = media_ref_pattern.match(stripped)
                if match:
                    media_id = match.group(1)
                    stored_url = media_store.get(media_id)
                    if stored_url:
                        resolved[key] = stored_url
                        logger.info(f"Resolved media_ref:{media_id} to stored URL")
                        continue
                    else:
                        logger.warning(f"media_ref:{media_id} not found in media store")

                # 2. Non-URL string (hallucinated path like /mnt/data/..., or bare filename)
                #    Try to find a UUID in the string that matches a stored media ID
                if not stripped.startswith(('http://', 'https://', 'data:')):
                    uuid_matches = uuid_pattern.findall(stripped)
                    resolved_from_uuid = False
                    for uuid_str in uuid_matches:
                        stored_url = media_store.get(uuid_str)
                        if stored_url:
                            resolved[key] = stored_url
                            logger.info(
                                f"Resolved hallucinated path '{stripped}' to stored URL "
                                f"via media ID {uuid_str}"
                            )
                            resolved_from_uuid = True
                            break
                    if resolved_from_uuid:
                        continue

                    # 3. If only one media exists and the value looks like a file path,
                    #    assume it refers to the most recent generated image
                    if len(media_store) == 1 and (
                        stripped.startswith('/') or stripped.startswith('file://')
                    ):
                        only_url = next(iter(media_store.values()))
                        resolved[key] = only_url
                        logger.info(
                            f"Resolved file path '{stripped}' to only available media URL"
                        )
                        continue

                resolved[key] = value
            else:
                resolved[key] = value

        return resolved

    def _handle_step_error(self, step: ReActStep, error: Exception, state: "ExecutionState"):
        """Handle step execution errors."""
        step.error = f"Step execution failed: {str(error)}"
        step.observation = f"An error occurred: {str(error)}"
        logger.error(f"Step {state.current_step} failed: {error}", exc_info=True)

    async def _publish_final_answer_event(self, step: ReActStep, state: "ExecutionState"):
        """Publish the closing final_answer event for the complete answer.

        Answer chunks are streamed live as they arrive in the streaming loop;
        this single event signals completion and gives consumers like
        agent.run() the full answer string in one place.
        """
        if step.answer:
            await self.event_bus.publish(EventFactory.final_answer(state.current_step, step.answer))

    async def _build_result(self, state: "ExecutionState", context: RunContext = None) -> ReActResult:
        """Build successful result."""
        # Determine stop reason
        if state.needs_clarification:
            stop_reason = StopReason.NEEDS_CLARIFICATION
            # Don't generate fallback - we're waiting for user input
            state.final_answer = ""
        elif context is not None and context.is_cancelled:
            stop_reason = StopReason.USER_CANCELLED
        elif state.final_answer:
            stop_reason = StopReason.ANSWER_COMPLETE
        else:
            stop_reason = StopReason.FORCED_STOP
            state.final_answer = self._generate_fallback_answer(state.steps)
            logger.warning(
                "[ORCH] FALLBACK answer generated (no step produced a final answer). steps=%d final_answer=%s",
                len(state.steps),
                _preview(state.final_answer, 300),
            )
            # Publish fallback as FINAL_ANSWER event so streaming_service captures it
            if state.final_answer:
                await self.event_bus.publish(
                    EventFactory.final_answer(state.current_step, state.final_answer)
                )

        logger.info(
            "[ORCH] result stop_reason=%s steps=%d final_answer=%s",
            getattr(stop_reason, "value", stop_reason),
            len(state.steps),
            _preview(state.final_answer, 300),
        )

        # Calculate totals
        total_time = time.time() - state.start_time
        total_cost = sum(step.cost for step in state.steps)
        total_tokens = sum(step.tokens_used for step in state.steps)

        result = ReActResult(
            steps=state.steps,
            final_answer=state.final_answer,
            stop_reason=stop_reason,
            total_cost=total_cost,
            total_execution_time=total_time,
            total_tokens=total_tokens,
        )

        # Attach clarification data if present
        if state.clarification_data:
            result.clarification_data = state.clarification_data

        return result

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

        # Detect the "halted on consecutive tool errors" case. If the tail of
        # the step history is made of error steps, the last observation is a
        # raw tool-execution error string — leaking that to the user surfaces
        # internals like parameter names and schema mismatches. Return a
        # user-facing apology instead, keeping any successful observations
        # out of the leak path too.
        recent_errors = [s for s in steps[-3:] if getattr(s, "is_error_step", False)]
        if recent_errors and getattr(last_step, "is_error_step", False):
            return (
                "I ran into repeated issues while trying to fulfill this request and "
                "wasn't able to produce a complete answer. Please try rephrasing your "
                "question, or try again in a moment."
            )

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
