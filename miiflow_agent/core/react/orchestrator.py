"""Focused ReAct orchestrator with clean separation of concerns."""

import json
import logging
import time
from typing import Any, Dict, List, Optional

from ..agent import RunContext
from ..checkpoint import DispatchLedgerEntry, PendingInterrupt, stable_json_hash
from ..interrupt import mint_interrupt_id
from ..message import Message, MessageRole
from .enums import ReActEventType, StopReason
from .exceptions import PlanApprovalRequired, ToolApprovalRequired
from .events import EventBus, EventFactory
from .react_events import ReActEvent
from .execution import ExecutionState
from .models import ReActResult, ReActStep, ToolInvocation
from .safety import SafetyManager
from .tool_executor import AgentToolExecutor, ToolCall

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
        name = getattr(getattr(tc, "function", None), "name", None) or getattr(
            tc, "name", None
        )
        args = getattr(getattr(tc, "function", None), "arguments", None) or getattr(
            tc, "arguments", None
        )
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
            tc_preview = (
                " tool_calls=["
                + ", ".join(_summarize_tool_call(tc) for tc in tool_calls)
                + "]"
            )
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


def _coerce_dict_payload(value: Any) -> Optional[Dict[str, Any]]:
    """Best-effort parse of structured tool output."""
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None

    stripped = value.strip()
    if not stripped or not stripped.startswith("{"):
        return None
    try:
        parsed = json.loads(stripped)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _approved_action_failure_payload(
    *,
    success: bool,
    observation: Any,
    output: Any = None,
    error: Any = None,
) -> Optional[Dict[str, Any]]:
    """Return structured failure data when an approved action did not complete."""
    payload = _coerce_dict_payload(output) or _coerce_dict_payload(observation)
    if payload is not None:
        status = str(payload.get("status") or "").lower()
        if status in {"failed", "failure", "error"} or payload.get("success") is False:
            return dict(payload)

    if not success:
        failure: Dict[str, Any] = {}
        if payload is not None:
            failure.update(payload)
        failure["error"] = str(error or observation or "Unknown error")
        return failure

    if isinstance(observation, str) and observation.startswith(
        ("Tool execution failed:", "Tool execution error:")
    ):
        return {"error": observation}
    return None


def _approved_action_failure_guidance(
    *,
    tool_name: str,
    failure_payload: Dict[str, Any],
) -> str:
    """Steering note for the model's reporting turn after an approved action failed.

    Raw failure payloads (proto field paths, resource-name JSON) must never reach
    the user verbatim — the model translates them. Auto-retry safety does NOT
    depend on halting the run: the approval gate's one-shot pass is spent, so any
    mutation the model proposes next pauses for a fresh user approval.
    """
    lines = [
        f"SYSTEM NOTE: the user-approved `{tool_name}` action FAILED. The tool "
        "result above contains the structured details (failed step, error, any "
        "partially created resources).",
        "",
        "Do not re-run it as-is. Report the outcome to the user in plain language:",
        "- Say what succeeded and what failed. If resources were partially "
        "created, mention them in plain terms and note they can be reused on a "
        "retry — do not paste raw resource names or JSON.",
        "- Translate the error into its plain-language cause. Never show raw "
        "API field paths or error codes.",
        "- Recommend the concrete fix.",
        "If you can correct the inputs yourself, you may propose the corrected "
        "action now — it will pause for a fresh user approval. Otherwise, ask "
        "the user how to proceed.",
    ]
    failed_step = failure_payload.get("failed_step")
    if failed_step:
        lines.insert(
            1, f"(Failed step: `{failed_step}`.)"
        )
    return "\n".join(lines)


def _approved_action_success_guidance(tool_name: str) -> str:
    """Steering note after a deterministic approved-action execution SUCCEEDED.

    Without this, the model sees only a bare success tool_result next to the
    pause turn's frozen "waiting for user approval" dispatch observation — and
    has been observed concluding the work is still unconfirmed, re-dispatching
    an identical mutation (and popping ANOTHER approval modal). Success must be
    as explicit as failure.
    """
    return (
        f"SYSTEM NOTE: the user APPROVED the pending `{tool_name}` action and "
        "it has ALREADY been executed — its result is in the tool result "
        "above. The task step it implements is DONE. Do not run it again, do "
        "not re-dispatch it, and do not re-verify it with more tool calls. "
        "Report the outcome to the user in plain language now (include the "
        "key created/changed entities), and only continue with further work "
        "if the original request clearly requires more."
    )


def _resolve_stale_dispatch_placeholder(
    messages: List[Message],
    *,
    parent_tool_call_id: Optional[str],
    tool_name: str,
    success: bool,
) -> None:
    """Rewrite the pause turn's frozen dispatch observation after resume.

    When a child's approval pauses the run, the parent's ``dispatch_assistant``
    tool_result is persisted as "Tool execution paused - waiting for user
    approval." and reconstructed verbatim on every later turn — permanently
    claiming the dispatch is still waiting even after the approved action ran.
    That stale claim directly contradicts the real result and has misled the
    model into re-dispatching. Resolve it in place once the outcome is known.
    """
    if not parent_tool_call_id:
        return
    for msg in reversed(messages):
        if (
            getattr(msg, "role", None) == MessageRole.TOOL
            and getattr(msg, "tool_call_id", None) == parent_tool_call_id
        ):
            outcome = "executed" if success else "executed but FAILED"
            msg.content = (
                f"This dispatch paused for user approval of `{tool_name}`; the "
                f"user approved and the action has since been {outcome} — see "
                f"the `{tool_name}` tool result that follows. This dispatch is "
                "complete; do not repeat it."
            )
            return


def _recover_approval_action_route(
    action: Dict[str, Any],
    checkpoint: Any,
) -> Dict[str, Any]:
    """Enrich legacy approval descriptors with checkpoint interrupt routing."""
    if not checkpoint or action.get("raised_by_path"):
        return action

    tool_call_id = action.get("tool_call_id")
    interrupt_id = action.get("interrupt_id")
    candidates = []
    try:
        active = checkpoint.active_interrupt()
        if active is not None:
            candidates.append(active)
    except Exception:
        pass

    interrupts = getattr(checkpoint, "interrupts", {}) or {}
    if interrupt_id and interrupt_id in interrupts:
        candidates.append(interrupts[interrupt_id])
    candidates.extend(interrupts.values())

    seen = set()
    for interrupt in candidates:
        iid = getattr(interrupt, "interrupt_id", None)
        if not iid or iid in seen:
            continue
        seen.add(iid)
        if getattr(interrupt, "kind", None) != "tool_approval":
            continue
        if interrupt_id and iid != interrupt_id:
            continue
        if tool_call_id and getattr(interrupt, "tool_call_id", None) != tool_call_id:
            continue

        payload = dict(getattr(interrupt, "payload", None) or {})
        recovered = dict(action)
        recovered.setdefault("tool_name", payload.get("tool_name"))
        recovered.setdefault("tool_call_id", getattr(interrupt, "tool_call_id", None))
        recovered.setdefault("interrupt_id", iid)
        recovered.setdefault("inputs", payload.get("tool_inputs") or {})
        recovered["raised_by_path"] = list(
            getattr(interrupt, "raised_by_path", None) or []
        )
        if payload.get("parent_tool_call_id") is not None:
            recovered.setdefault(
                "parent_tool_call_id", payload.get("parent_tool_call_id")
            )
        return recovered

    return action


def _consume_approval_checkpoint_state(
    checkpoint: Any,
    *,
    interrupt_id: Optional[str],
    raised_by_path: Optional[List[str]],
) -> None:
    """Clear one-shot approval resume state after the approved action is attempted."""
    if checkpoint is None:
        return
    try:
        checkpoint.pending_approved_action = None
        checkpoint.resume = None
        if raised_by_path:
            (getattr(checkpoint, "agent_frames", {}) or {}).pop(
                "/".join(raised_by_path), None
            )
        if interrupt_id:
            if getattr(
                checkpoint, "active_interrupt_id", None
            ) == interrupt_id and hasattr(checkpoint, "clear_active_interrupt"):
                checkpoint.clear_active_interrupt()
            else:
                (getattr(checkpoint, "interrupts", {}) or {}).pop(interrupt_id, None)
                queue = getattr(checkpoint, "interrupt_queue", None)
                if isinstance(queue, list):
                    checkpoint.interrupt_queue = [i for i in queue if i != interrupt_id]
                pending = getattr(checkpoint, "pending_interrupt", None)
                if getattr(pending, "interrupt_id", None) == interrupt_id:
                    checkpoint.pending_interrupt = None
    except Exception:
        logger.debug("Failed to clear approval checkpoint state", exc_info=True)


def _preparse_tool_args_string(
    tool_call_data: Dict[str, Any], step_number: int, tool_name: str
) -> None:
    """Parse OpenAI/Gemini-style string tool args into a dict in-place.

    On JSONDecodeError, attaches a `_truncation_error` marker so the
    orchestrator's truncation handler treats it the same as Anthropic's
    upstream truncation signal (which the stream_normalizer attaches at
    content_block_stop). No-op if args are already a dict, None, or empty.
    """
    args = tool_call_data.get("function", {}).get("arguments")
    if not isinstance(args, str) or not args.strip():
        return
    import json as _json_mod

    try:
        tool_call_data["function"]["arguments"] = _json_mod.loads(args)
    except _json_mod.JSONDecodeError as parse_exc:
        logger.warning(
            "Step %d - Tool '%s' args failed to parse as JSON "
            "(len=%d, error=%s) — routing to truncation handler",
            step_number,
            tool_name,
            len(args),
            parse_exc,
        )
        tool_call_data["_truncation_error"] = {
            "kind": "json_parse_failed",
            "message": str(parse_exc),
            "accumulated_length": len(args),
            "raw_prefix": args[:500],
        }
        tool_call_data["function"]["arguments"] = {}


def _format_missing_params_error(
    tool_name: str,
    missing_params: List[str],
    provided_params: List[str],
    tool_schema: Dict[str, Any],
) -> str:
    """Build an actionable <tool_use_error> for missing required params.

    Includes per-field schema hints so the model can correct in one retry
    rather than guessing again. Mirrors claude-code's tool_use_error pattern.
    """
    parameters = tool_schema.get("parameters", {}) or {}
    properties = parameters.get("properties", {}) or {}
    required = set(parameters.get("required", []) or [])

    def _field_hint(name: str, spec: Dict[str, Any]) -> str:
        type_ = spec.get("type", "any")
        desc = (spec.get("description") or "").strip()
        enum = spec.get("enum")
        marker = "REQUIRED" if name in required else "optional"
        parts = [f"  - {name} ({marker}, {type_}"]
        if enum:
            parts.append(f", one of: {', '.join(map(str, enum))}")
        parts.append(")")
        if desc:
            parts.append(f": {desc}")
        return "".join(parts)

    # Show missing required first, then the rest, so attention lands on the gap.
    ordered = list(missing_params) + [n for n in properties if n not in missing_params]
    schema_lines = [_field_hint(n, properties[n]) for n in ordered if n in properties]

    return (
        "<tool_use_error>\n"
        f"InputValidationError: tool '{tool_name}' is missing required parameters "
        f"{missing_params}. Provided: {provided_params}.\n\n"
        f"Schema for '{tool_name}':\n"
        + "\n".join(schema_lines)
        + "\n\nRetry the call with every REQUIRED field populated. Do not omit "
        "data-bearing fields (arrays, structured objects) — supplying only "
        "cosmetic fields (title, layout, etc.) will fail again.\n"
        "</tool_use_error>"
    )


def _extract_failure_metadata(
    steps: List[ReActStep],
    *,
    stop_reason: str,
    description: str,
    truncation: int = 800,
    input_truncation: int = 500,
) -> Dict[str, Any]:
    """Scan the recent step history for the last failing tool invocation
    and return a structured diagnostic payload.

    Used when a safety condition halts the loop (e.g. RepeatedToolError)
    so callers like the dispatch envelope can surface a real cause to
    the parent agent instead of the canned "repeated issues" string.

    The payload is bounded — both the error observation and the input
    snapshot are truncated — so a failing query with a long body doesn't
    bloat the parent's tool observation.
    """
    last_failing = None
    failing_attempts = 0
    # Walk the recent invocations newest-first. Bounded scan (8 steps)
    # mirrors RepeatedToolErrorCondition's lookback window so the count
    # we return matches what the safety condition was actually counting.
    for step in reversed(steps[-8:]):
        for inv in reversed(step.all_invocations):
            is_failure = inv.error is not None or _observation_looks_like_error(
                inv.observation
            )
            if not is_failure:
                continue
            failing_attempts += 1
            if last_failing is None:
                last_failing = inv

    payload: Dict[str, Any] = {
        "stop_reason": stop_reason,
        "description": description,
    }
    if last_failing is not None:
        observation = last_failing.observation or last_failing.error or ""
        payload["last_tool"] = last_failing.name
        payload["last_tool_error"] = observation[:truncation]
        if last_failing.inputs:
            payload["last_tool_input"] = _truncate_input(
                last_failing.inputs, input_truncation
            )
        payload["attempts_seen"] = failing_attempts
    return payload


def _observation_looks_like_error(observation: Optional[str]) -> bool:
    """Heuristic: tool observations that carry a soft error.

    Mirrors the marker logic in ``RepeatedToolErrorCondition._error_key``
    so the metadata we surface lines up with the condition that fired.
    The full regex lives in ``safety.py``; we accept a small duplication
    here to avoid an import cycle.
    """
    if not isinstance(observation, str) or not observation:
        return False
    # Match the framework's `Tool execution failed:` prefix or a dict
    # carrying a truthy `error` value.
    if observation.startswith("Tool execution failed:"):
        return True
    return "'error':" in observation and "'error': None" not in observation


def _truncate_input(inputs: Dict[str, Any], limit: int) -> Dict[str, Any]:
    """Truncate string values in a tool input dict so the failure payload
    stays bounded even when a query string is very long."""
    truncated: Dict[str, Any] = {}
    for key, value in inputs.items():
        if isinstance(value, str) and len(value) > limit:
            truncated[key] = (
                value[:limit] + f"... [truncated {len(value) - limit} chars]"
            )
        else:
            truncated[key] = value
    return truncated


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
            if hasattr(cond, "max_steps"):
                max_steps = cond.max_steps
                break
        progress_tracker = ProgressTracker(max_steps=max_steps)

        try:
            self._setup_context(query, context)

            # Provision a per-turn DispatchCounter on ctx.deps so the
            # synthesized dispatch_assistant tool can do race-safe budget
            # accounting under parallel dispatch. Counter is fresh per
            # ReAct turn — cross-turn budgets (if needed) are a Django
            # adapter concern. Safe in-place assignment because the
            # orchestrator owns ctx lifecycle for the duration of this run.
            if context is not None:
                from .dispatch import DispatchCounter

                # Provision a per-turn DispatchCounter so the synthesized
                # dispatch_assistant tool can do race-safe budget accounting
                # under parallel dispatch. Counter is fresh per ReAct turn —
                # cross-turn budgets (if needed) are a Django adapter concern.
                # Honor a counter the caller already installed on EITHER
                # surface (lets Django swap in a cross-turn counter, and lets
                # test fixtures that still seed ctx.deps continue to work).
                deps_is_dict = isinstance(getattr(context, "deps", None), dict)
                preseeded_counter = (
                    context.deps.get("dispatch_counter") if deps_is_dict else None
                )
                if preseeded_counter is not None:
                    context.run_state.dispatch_counter = preseeded_counter
                elif context.run_state.dispatch_counter is None:
                    context.run_state.dispatch_counter = DispatchCounter()

                # Attach this orchestrator's event_bus so tools that publish
                # back to the parent's stream (notably dispatch_assistant
                # forwarding subagent lifecycle events) find it regardless
                # of whether they're invoked through the single-tool path
                # (_execute_tool) or the parallel batch path
                # (_handle_parallel_tool_batch → execute_many). The bus is
                # invariant for the run, so set-once here beats per-step
                # injection that the batch path was missing.
                context.run_state.event_bus = self.event_bus

                # Legacy dual-write to ctx.deps for callers that haven't
                # migrated to ctx.run_state.* yet. Remove once every reader
                # (the dispatch closures, batch_executor, memory_fs_tools)
                # has been switched. New code should NOT read these keys.
                if deps_is_dict:
                    context.deps["dispatch_counter"] = (
                        context.run_state.dispatch_counter
                    )
                    context.deps["event_bus"] = self.event_bus

            # Deterministic approval resume: if the user just approved a tool
            # call, EXECUTE it here (control flow owns continuation) rather than
            # asking the model to re-emit it. Runs before the first LLM call so
            # the model sees the real result and only reports it. A REJECTED
            # approval is acknowledged deterministically too — the loop below
            # never runs, so the model cannot re-request the declined action.
            validated_resume = self._apply_resume_command(context)
            await self._acknowledge_rejected_approval(
                validated_resume, context, execution_state
            )
            await self._execute_pending_approved_action(context, execution_state)

            # Apply context compression before starting if available
            if self.context_compressor:
                result = await self.context_compressor.compress_if_needed(
                    context.messages
                )
                if result.was_compressed:
                    context.messages = result.messages
                    logger.info(
                        f"Pre-execution context compression: {result.original_count} -> {result.compressed_count} messages"
                    )

            while execution_state.is_running:
                execution_state.current_step += 1

                # Check for user cancellation
                if context.is_cancelled:
                    logger.info("Execution cancelled by user")
                    execution_state.final_answer = (
                        self.full_response if hasattr(self, "full_response") else ""
                    )
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
                    step = await self._execute_reasoning_step_native(
                        context, execution_state
                    )
                except Exception as _llm_err:
                    from .recovery import is_context_overflow_error

                    if (
                        not is_context_overflow_error(_llm_err)
                        or not self.recovery_manager
                    ):
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
                    EventFactory.progress(
                        execution_state.current_step, progress_tracker.snapshot()
                    )
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
                    from .recovery import FailureKind

                    kind_str = step.metadata.get("failure_kind", "runtime")
                    try:
                        failure_kind = FailureKind(kind_str)
                    except ValueError:
                        failure_kind = FailureKind.RUNTIME
                    recovery_action = await self.recovery_manager.attempt_recovery(
                        error=Exception(step.error or "Unknown error"),
                        context=context,
                        step=step,
                        tool_name=step.action,
                        failure_kind=failure_kind,
                    )
                    logger.info(
                        "[ORCH] step=%d recovery strategy=%s attempt=%s continue=%s "
                        "excluded_tools=%s guidance=%s",
                        execution_state.current_step,
                        getattr(
                            recovery_action.strategy_used,
                            "value",
                            recovery_action.strategy_used,
                        ),
                        recovery_action.attempt_number,
                        recovery_action.should_continue,
                        (
                            sorted(recovery_action.excluded_tools)
                            if recovery_action.excluded_tools
                            else None
                        ),
                        (
                            _preview(recovery_action.guidance_message, 240)
                            if recovery_action.guidance_message
                            else None
                        ),
                    )
                    if not recovery_action.should_continue:
                        logger.warning("Recovery exhausted, stopping execution")
                        break
                    if recovery_action.guidance_message:
                        context.messages.append(
                            Message(
                                role=MessageRole.USER,
                                content=recovery_action.guidance_message,
                            )
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
            has_user_message = any(
                msg.role == MessageRole.USER for msg in (context.messages or [])
            )
            if not has_user_message:
                raise ValueError(
                    "Query cannot be empty when no user message exists in context"
                )

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
        stop_condition = self.safety_manager.should_stop(
            state.steps, state.current_step
        )
        if stop_condition:
            # Log which condition fired — invaluable when an
            # unexpected stop happens in production. The stop_reason
            # alone (event.data.reason) doesn't distinguish e.g.
            # ThinkingOnlyCondition from EmptyResponseCondition since
            # both map to FORCED_STOP.
            logger.warning(
                "[ORCH] safety condition fired class=%s reason=%s description=%r at step=%d",
                type(stop_condition).__name__,
                stop_condition.get_stop_reason().value,
                stop_condition.get_description(),
                state.current_step,
            )
            failure = _extract_failure_metadata(
                state.steps,
                stop_reason=stop_condition.get_stop_reason().value,
                description=stop_condition.get_description(),
            )
            # Stash on state so _build_result can attach it to the
            # ReActResult.metadata for callers that consume the result
            # directly (the dispatch envelope reads it via the event;
            # both paths are wired so neither side has to know about
            # the other).
            state.failure_metadata = failure
            event = EventFactory.stop_condition(
                state.current_step,
                stop_condition.get_stop_reason().value,
                stop_condition.get_description(),
                failure=failure,
            )
            await self.event_bus.publish(event)
            return True
        return False

    async def _steer_report_after_approved_action_failure(
        self,
        *,
        context: RunContext,
        state: "ExecutionState",
        tool_name: str,
        tool_call_id: str,
        observation: str,
        failure_payload: Dict[str, Any],
        raised_by_path: Optional[List[str]] = None,
    ) -> None:
        """Steer the model to report an approved side-effect failure — never halt.

        The failure observation is already paired into the transcript as the
        tool_result; the run continues so the model's next turn translates it
        into plain language and actionable next steps. Halting here (the old
        behavior) shipped the raw API error to the user verbatim. The model
        cannot silently retry: the approval gate's one-shot pass is spent, so
        any corrected mutation it proposes pauses for a fresh user approval.
        """
        logger.info(
            "[ORCH] approved action '%s' failed (id=%s, path=%s, step=%s) — "
            "steering a reporting turn",
            tool_name,
            tool_call_id,
            raised_by_path,
            failure_payload.get("failed_step"),
        )
        context.messages.append(
            Message(
                role=MessageRole.USER,
                content=_approved_action_failure_guidance(
                    tool_name=tool_name,
                    failure_payload=failure_payload,
                ),
            )
        )

    async def _execute_pending_approved_action(
        self, context: RunContext, state: "ExecutionState"
    ) -> None:
        """Deterministically execute a tool call the user just approved.

        FOUNDATIONAL: continuation across an approval pause is CONTROL FLOW, not
        reasoning. The action was already decided by the model and approved by
        the user, so the orchestrator completes it here — with the one-shot gate
        pass and the approved/edited inputs — and surfaces the real result. The
        model is then invoked only to *report* that result, never to re-emit the
        call (which it unreliably does: it would re-ask clarification or lose its
        own proposal).

        Idempotent within the run: the descriptor is consumed up front and the
        gate's one-shot pass is spent by ``execute_tool``, so the model cannot
        double-fire the same call. Degrades gracefully: with no descriptor, the
        reconstructed "approved — call it again" placeholder remains and the
        legacy model-driven path applies. Execution does NOT depend on tool
        VISIBILITY (the tool is registered regardless of ToolSearch gating), so
        this is immune to the schema-cap / pinning concerns entirely.
        """
        deps = getattr(context, "deps", None)
        checkpoint = getattr(context, "checkpoint", None)
        cp_action = getattr(checkpoint, "pending_approved_action", None)
        if not isinstance(deps, dict) and cp_action is None:
            return
        action = cp_action.to_dict() if hasattr(cp_action, "to_dict") else None
        if action is None and isinstance(deps, dict):
            action = deps.get("pending_approved_action")
        if not action:
            return

        action = _recover_approval_action_route(action, checkpoint)
        tool_name = action.get("tool_name")
        tool_call_id = action.get("tool_call_id")
        inputs = action.get("inputs") or {}
        raised_by_path = list(action.get("raised_by_path") or [])
        interrupt_id = action.get("interrupt_id")
        if not tool_name or not tool_call_id:
            return

        # Dispatch safety: execute locally only for tools THIS agent owns. For
        # child-owned tools, route through an adapter-supplied deterministic
        # child resumer. Never fall back to "let the LLM re-dispatch"; approved
        # side effects are control flow.
        registry = getattr(self.tool_executor, "_tool_registry", None)
        if registry is not None and hasattr(registry, "_has_registered_tool"):
            if not registry._has_registered_tool(tool_name):
                frame = None
                if checkpoint is not None and raised_by_path:
                    frame = (getattr(checkpoint, "agent_frames", {}) or {}).get(
                        "/".join(raised_by_path)
                    )
                child_resumer = (
                    deps.get("child_approval_resumer")
                    if isinstance(deps, dict)
                    else None
                )
                if frame is not None and child_resumer is not None:
                    if isinstance(deps, dict):
                        deps["pending_approved_action"] = None
                    _consume_approval_checkpoint_state(
                        checkpoint,
                        interrupt_id=interrupt_id,
                        raised_by_path=raised_by_path,
                    )
                    raw_output = None
                    raw_error = None
                    try:
                        import inspect

                        outcome = child_resumer(
                            action=action,
                            frame=frame,
                            context=context,
                        )
                        if inspect.isawaitable(outcome):
                            outcome = await outcome
                        if hasattr(outcome, "success"):
                            success = bool(getattr(outcome, "success", False))
                            out = getattr(outcome, "output", None)
                            err = getattr(outcome, "error", None)
                            raw_output = out
                            raw_error = err
                            observation = (
                                (
                                    out
                                    if isinstance(out, str)
                                    else json.dumps(out, default=str)
                                )
                                if success
                                else f"Tool execution failed: {_sanitize_error_message(str(err or 'Unknown error'))}"
                            )
                        else:
                            success = bool((outcome or {}).get("success", True))
                            raw_output = (outcome or {}).get("output")
                            raw_error = (outcome or {}).get("error")
                            raw_obs = (outcome or {}).get("observation")
                            if raw_obs is None:
                                raw_obs = raw_output
                            observation = (
                                raw_obs
                                if isinstance(raw_obs, str)
                                else json.dumps(raw_obs, default=str)
                            )
                    except Exception as e:  # noqa: BLE001
                        success = False
                        raw_error = e
                        observation = f"Tool execution error: {e}"

                    failure_payload = _approved_action_failure_payload(
                        success=success,
                        observation=observation,
                        output=raw_output,
                        error=raw_error,
                    )
                    if failure_payload is not None:
                        success = False

                    logger.info(
                        "[ORCH] deterministic child approval-resume executed '%s' "
                        "(success=%s, id=%s, path=%s)",
                        tool_name,
                        success,
                        tool_call_id,
                        raised_by_path,
                    )
                    if checkpoint is not None and hasattr(checkpoint, "merge_ledger"):
                        checkpoint.merge_ledger(
                            [
                                DispatchLedgerEntry(
                                    kind="tool_call",
                                    success=success,
                                    observation=observation,
                                    turn=state.current_step,
                                    tool_name=tool_name,
                                    inputs_hash=stable_json_hash(inputs or {}),
                                    produced_by_path=raised_by_path or ["root"],
                                )
                            ]
                        )
                    for msg in reversed(context.messages):
                        if (
                            getattr(msg, "role", None) == MessageRole.TOOL
                            and getattr(msg, "tool_call_id", None) == tool_call_id
                        ):
                            msg.content = observation
                            break
                    else:
                        context.messages.append(
                            Message(
                                role=MessageRole.TOOL,
                                content=observation,
                                tool_call_id=tool_call_id,
                            )
                        )
                    try:
                        await self.event_bus.publish(
                            EventFactory.observation(
                                state.current_step,
                                observation,
                                tool_name,
                                success,
                                tool_call_id=tool_call_id,
                            )
                        )
                    except Exception as evt_err:  # noqa: BLE001
                        logger.debug(
                            "Failed to publish child approval-resume observation: %s",
                            evt_err,
                        )
                    _resolve_stale_dispatch_placeholder(
                        context.messages,
                        parent_tool_call_id=action.get("parent_tool_call_id"),
                        tool_name=tool_name,
                        success=success,
                    )
                    if failure_payload is not None:
                        await self._steer_report_after_approved_action_failure(
                            context=context,
                            state=state,
                            tool_name=tool_name,
                            tool_call_id=tool_call_id,
                            observation=observation,
                            failure_payload=failure_payload,
                            raised_by_path=raised_by_path or ["root"],
                        )
                    else:
                        context.messages.append(
                            Message(
                                role=MessageRole.USER,
                                content=_approved_action_success_guidance(tool_name),
                            )
                        )
                    return

                message = (
                    f"Approved tool '{tool_name}' belongs to a child agent, but "
                    "the saved child approval frame could not be resumed. "
                    "Stopped without executing the approved action."
                )
                logger.error(
                    "[ORCH] approval-resume fail-closed: tool=%s path=%s frame=%s resumer=%s",
                    tool_name,
                    raised_by_path,
                    bool(frame),
                    bool(child_resumer),
                )
                if isinstance(deps, dict):
                    deps["pending_approved_action"] = None
                _consume_approval_checkpoint_state(
                    checkpoint,
                    interrupt_id=interrupt_id,
                    raised_by_path=raised_by_path,
                )
                state.is_running = False
                state.final_answer = message
                state.failure_metadata = {
                    "stop_reason": "child_approval_resume_missing",
                    "last_tool": tool_name,
                    "description": message,
                    "raised_by_path": raised_by_path,
                }
                try:
                    await self.event_bus.publish(
                        EventFactory.final_answer(state.current_step, message)
                    )
                except Exception as evt_err:  # noqa: BLE001
                    logger.debug("Failed to publish fail-closed answer: %s", evt_err)
                return

        # Consume now (we will execute) so nothing else re-runs it this turn.
        if isinstance(deps, dict):
            deps["pending_approved_action"] = None
        _consume_approval_checkpoint_state(
            checkpoint,
            interrupt_id=interrupt_id,
            raised_by_path=raised_by_path,
        )

        raw_output = None
        raw_error = None
        try:
            result = await self.tool_executor.execute_tool(
                tool_name, inputs, context=context
            )
            success = getattr(result, "success", True)
            if not success:
                raw_error = getattr(result, "error", None)
                observation = (
                    "Tool execution failed: "
                    f"{_sanitize_error_message(str(raw_error or 'Unknown error'))}"
                )
            else:
                out = getattr(result, "output", None)
                raw_output = out
                observation = (
                    out if isinstance(out, str) else json.dumps(out, default=str)
                )
        except (
            Exception
        ) as e:  # noqa: BLE001 — surface as observation, never crash the run
            success = False
            raw_error = e
            observation = f"Tool execution error: {e}"

        failure_payload = _approved_action_failure_payload(
            success=success,
            observation=observation,
            output=raw_output,
            error=raw_error,
        )
        if failure_payload is not None:
            success = False

        logger.info(
            "[ORCH] deterministic approval-resume executed '%s' (success=%s, id=%s)",
            tool_name,
            success,
            tool_call_id,
        )
        ReActOrchestrator._record_tool_ledger_entry(
            self,
            context,
            state,
            tool_name=tool_name,
            inputs=inputs,
            observation=observation,
            success=success,
        )

        # Pair the result with the approved tool_use: replace the reconstructed
        # "approved — call it again" placeholder for this tool_call_id, or append
        # a fresh TOOL message if no placeholder exists (keeps the tool_use/
        # tool_result pairing valid for the next LLM call).
        replaced = False
        for msg in reversed(context.messages):
            if (
                getattr(msg, "role", None) == MessageRole.TOOL
                and getattr(msg, "tool_call_id", None) == tool_call_id
            ):
                msg.content = observation
                replaced = True
                break
        if not replaced:
            context.messages.append(
                Message(
                    role=MessageRole.TOOL,
                    content=observation,
                    tool_call_id=tool_call_id,
                )
            )

        try:
            await self.event_bus.publish(
                EventFactory.observation(
                    state.current_step,
                    observation,
                    tool_name,
                    success,
                    tool_call_id=tool_call_id,
                )
            )
        except Exception as evt_err:  # noqa: BLE001
            logger.debug("Failed to publish approval-resume observation: %s", evt_err)

        _resolve_stale_dispatch_placeholder(
            context.messages,
            parent_tool_call_id=action.get("parent_tool_call_id"),
            tool_name=tool_name,
            success=success,
        )
        if failure_payload is not None:
            await self._steer_report_after_approved_action_failure(
                context=context,
                state=state,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                observation=observation,
                failure_payload=failure_payload,
                raised_by_path=raised_by_path or None,
            )
        else:
            context.messages.append(
                Message(
                    role=MessageRole.USER,
                    content=_approved_action_success_guidance(tool_name),
                )
            )

    def _apply_resume_command(self, context: RunContext) -> Optional[Any]:
        """Translate an authoritative checkpoint resume into runtime action.

        This is deliberately deterministic and runs before the first LLM call.
        The adapter may still dual-write legacy ``ctx.deps`` fields during the
        migration, but checkpoint/resume wins when present.

        Returns the resume command iff it validated against the active
        interrupt THIS turn — callers must never act on a stale persisted
        resume from a prior turn. The checkpoint's stored copy is consumed
        (cleared) on success so it can never replay.
        """
        checkpoint = getattr(context, "checkpoint", None)
        resume = getattr(context, "resume", None) or getattr(checkpoint, "resume", None)
        if checkpoint is None or resume is None:
            return None

        interrupt = None
        if hasattr(checkpoint, "active_interrupt"):
            interrupt = checkpoint.active_interrupt()
        if interrupt is None:
            interrupt = (getattr(checkpoint, "interrupts", {}) or {}).get(
                resume.interrupt_id
            )
        if interrupt is None or interrupt.interrupt_id != resume.interrupt_id:
            logger.warning(
                "[ORCH] ignoring resume for unknown interrupt_id=%s",
                getattr(resume, "interrupt_id", None),
            )
            return None
        if interrupt.kind != resume.kind:
            logger.warning(
                "[ORCH] ignoring resume kind mismatch interrupt=%s resume=%s",
                interrupt.kind,
                resume.kind,
            )
            return None

        if resume.kind == "tool_approval":
            payload = interrupt.payload or {}
            tool_name = payload.get("tool_name")
            tool_call_id = interrupt.tool_call_id
            if resume.decision == "approved" and tool_name and tool_call_id:
                approved_inputs = (
                    resume.value.get("modified_inputs")
                    or resume.value.get("tool_inputs")
                    or payload.get("tool_inputs")
                    or {}
                )
                from ..checkpoint import PendingApprovedAction

                action = PendingApprovedAction(
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    inputs=dict(approved_inputs or {}),
                    interrupt_id=interrupt.interrupt_id,
                    raised_by_path=list(interrupt.raised_by_path or []),
                    parent_tool_call_id=payload.get("parent_tool_call_id"),
                )
                checkpoint.pending_approved_action = action
                if isinstance(getattr(context, "deps", None), dict):
                    context.deps["pending_approved_action"] = action.to_dict()
            elif resume.decision == "rejected":
                checkpoint.extra.setdefault("approval_rejections", []).append(
                    {
                        "interrupt_id": interrupt.interrupt_id,
                        "tool_name": payload.get("tool_name"),
                        "tool_inputs": payload.get("tool_inputs") or {},
                        "reason": resume.value.get("reason", ""),
                    }
                )
            checkpoint.clear_active_interrupt()
        elif resume.kind in ("clarification", "plan_approval"):
            checkpoint.clear_active_interrupt()

        # If clearing promoted a queued interrupt (a second pause raised in the
        # same batch as the one just answered), flag it so the adapter can
        # re-surface its UI prompt at the end of this turn — the frontend only
        # ever rendered the LAST pause of that batch.
        promoted = checkpoint.active_interrupt()
        if promoted is not None:
            checkpoint.extra["resurface_interrupt_id"] = promoted.interrupt_id
            logger.info(
                "[ORCH] promoted queued interrupt %s after resume of %s",
                promoted.interrupt_id,
                resume.interrupt_id,
            )

        # Consume: the stored resume answered THIS interrupt. Leaving it on the
        # checkpoint would make every later turn re-see a stale "rejected"/
        # "approved" command.
        checkpoint.resume = None
        return resume

    async def _acknowledge_rejected_approval(
        self,
        resume: Optional[Any],
        context: RunContext,
        state: "ExecutionState",
    ) -> bool:
        """Deterministically acknowledge a rejected tool approval — no LLM call.

        A decline needs no reasoning: confirm the cancellation and ask what the
        user wants instead. Handing the turn to the model instead has repeatedly
        produced re-requests of the rejected action (the standing task in the
        transcript outweighs any steering text — observed 12× consecutively in
        the reject harness). Any follow-up arrives as a fresh user message with
        full tool freedom, so nothing is lost by not invoking the model here.
        """
        if (
            resume is None
            or getattr(resume, "kind", None) != "tool_approval"
            or getattr(resume, "decision", None) != "rejected"
        ):
            return False

        reason = str((getattr(resume, "value", None) or {}).get("reason") or "").strip()
        # UI-default placeholder, not user-authored text (set by the approval
        # modal when the user declines without typing a reason).
        if reason.lower() in {"", "no reason given", "user declined this change"}:
            reason = ""

        lines = [
            "Understood — I've cancelled that proposed change. Nothing was modified."
        ]
        if reason:
            lines.append(f"Noted: {reason}")
        lines.append(
            "What would you like to do instead? I can adjust the proposal and "
            "run it past you again, or we can leave it as is."
        )
        message = "\n\n".join(lines)

        logger.info(
            "[ORCH] deterministic rejection acknowledgement (interrupt_id=%s)",
            getattr(resume, "interrupt_id", None),
        )
        state.final_answer = message
        state.is_running = False
        await self.event_bus.publish(
            EventFactory.final_answer(state.current_step, message)
        )
        return True

    async def _record_interrupt(
        self,
        context: RunContext,
        state: "ExecutionState",
        *,
        kind: str,
        payload: Dict[str, Any],
        tool_call_id: Optional[str] = None,
        raised_by_path: Optional[List[str]] = None,
    ) -> PendingInterrupt:
        """Persist and publish the canonical runtime interrupt.

        Legacy pause paths still emit their specialized events for the current
        UI, but this typed checkpoint record is the authoritative control-plane
        state used by deterministic resume.
        """
        interrupt_id = payload.get("interrupt_id") or mint_interrupt_id(
            kind, tool_call_id
        )
        interrupt = PendingInterrupt(
            interrupt_id=interrupt_id,
            kind=kind,
            raised_by_path=raised_by_path or ["root"],
            payload=dict(payload),
            tool_call_id=tool_call_id,
        )
        # Tolerates stand-in states (tests) that predate the field.
        raised_this_run = getattr(state, "raised_interrupt_ids", None)
        if raised_this_run is None:
            raised_this_run = []
            try:
                state.raised_interrupt_ids = raised_this_run
            except AttributeError:
                pass
        checkpoint = getattr(context, "checkpoint", None)
        if checkpoint is not None and hasattr(checkpoint, "set_interrupt"):
            # Parallel pauses in one run (dispatch_assistant is parallelizable):
            # the previously-active interrupt must be QUEUED, not dropped, or
            # the earlier child is stranded forever — it stays in `interrupts`
            # but nothing ever activates it again. Demote only interrupts this
            # run raised: a stale active from an old turn keeps today's
            # replace-and-forget behavior instead of being resurrected.
            prior_active = getattr(checkpoint, "active_interrupt_id", None)
            checkpoint.set_interrupt(interrupt)
            if (
                prior_active
                and prior_active != interrupt.interrupt_id
                and prior_active in raised_this_run
                and prior_active in (getattr(checkpoint, "interrupts", {}) or {})
                and prior_active not in (getattr(checkpoint, "interrupt_queue", []) or [])
            ):
                checkpoint.interrupt_queue.append(prior_active)
        raised_this_run.append(interrupt_id)
        await self.event_bus.publish(
            EventFactory.interrupt_requested(state.current_step, interrupt)
        )
        return interrupt

    async def _handle_tool_approval_marker_result(
        self,
        context: RunContext,
        state: "ExecutionState",
        result: Any,
        *,
        parent_tool_call_id: Optional[str],
    ) -> bool:
        """Pause on a child-owned tool approval surfaced as a marker result.

        ``dispatch_assistant`` returns this marker when a sub-agent pauses on a
        write tool. The parent must persist the CHILD interrupt and stop; it
        must not throw a new parent-owned ``ToolApprovalRequired`` or ask the
        model to re-dispatch.
        """
        from ..tools.clarification import is_tool_approval_result

        if not is_tool_approval_result(result):
            return False

        output = result.output if isinstance(result.output, dict) else {}
        child_tool_call_id = output.get("tool_call_id")
        subagent_path = output.get("subagent_path") or []
        raised_by_path = output.get("raised_by_path") or ["root"] + list(subagent_path)
        payload = {
            "interrupt_id": output.get("interrupt_id"),
            "tool_name": output.get("tool_name"),
            "tool_inputs": output.get("tool_inputs") or {},
            "tool_description": output.get("tool_description") or "",
            "tool_schema": output.get("tool_schema") or {},
            "reason": output.get("reason"),
            "handle": output.get("handle"),
            "child_assistant_id": output.get("child_assistant_id"),
            "child_thread_id": output.get("child_thread_id"),
            "subagent_id": output.get("subagent_id"),
            "subagent_path": subagent_path,
            "parent_tool_call_id": parent_tool_call_id,
        }

        interrupt = await self._record_interrupt(
            context,
            state,
            kind="tool_approval",
            payload=payload,
            tool_call_id=child_tool_call_id,
            raised_by_path=raised_by_path,
        )

        checkpoint = getattr(context, "checkpoint", None)
        frame_key = "/".join(interrupt.raised_by_path or [])
        frame = (
            checkpoint.agent_frames.get(frame_key)
            if checkpoint is not None and hasattr(checkpoint, "agent_frames")
            else None
        )
        if frame is not None:
            frame.pending_interrupt = interrupt
            frame.metadata.setdefault("parent_tool_call_id", parent_tool_call_id)
            frame.metadata.setdefault("child_thread_id", payload.get("child_thread_id"))
            frame.metadata.setdefault(
                "child_assistant_id", payload.get("child_assistant_id")
            )

        state.needs_clarification = True
        state.clarification_data = {
            "type": "tool_approval",
            "tool_name": payload["tool_name"],
            "tool_inputs": payload["tool_inputs"],
            "tool_description": payload["tool_description"],
            "tool_schema": payload["tool_schema"],
            "tool_call_id": child_tool_call_id,
            "interrupt_id": interrupt.interrupt_id,
            "raised_by_path": interrupt.raised_by_path,
            "reason": payload["reason"],
            "parent_tool_call_id": parent_tool_call_id,
            "child_assistant_id": payload.get("child_assistant_id"),
            "child_thread_id": payload.get("child_thread_id"),
            "subagent_id": payload.get("subagent_id"),
            "subagent_path": subagent_path,
        }
        await self.event_bus.publish(
            ReActEvent(
                event_type=ReActEventType.TOOL_APPROVAL_NEEDED,
                step_number=state.current_step,
                data=state.clarification_data,
            )
        )
        return True

    def _record_tool_ledger_entry(
        self,
        context: RunContext,
        state: "ExecutionState",
        *,
        tool_name: Optional[str],
        inputs: Optional[Dict[str, Any]],
        observation: Optional[str],
        success: bool,
    ) -> None:
        """Reduce a completed tool call into the checkpoint ledger."""
        if not tool_name:
            return
        checkpoint = getattr(context, "checkpoint", None)
        if checkpoint is None or not hasattr(checkpoint, "merge_ledger"):
            return
        checkpoint.merge_ledger(
            [
                DispatchLedgerEntry(
                    kind="tool_call",
                    success=bool(success),
                    observation=observation or "",
                    turn=state.current_step,
                    tool_name=tool_name,
                    inputs_hash=stable_json_hash(inputs or {}),
                    produced_by_path=["root"],
                )
            ]
        )

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
            pending_answer_deltas: List[str] = []
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
                        (_fn or {}).get("name")
                        if _fn
                        else (_t.get("name") if isinstance(_t, dict) else str(_t))
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

            async for chunk in self.tool_executor.stream_with_tools(
                messages=context.messages
            ):
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

                # Text deltas are not final until the provider turn is classified.
                # Frontier models often emit narration before tool_use blocks; if we
                # stream that as FINAL_ANSWER_CHUNK and a tool appears later, runtime
                # control-flow text leaks into the final answer. Buffer answer
                # candidates and replay them only after the stream closes with no
                # tool calls.
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
                        pending_answer_deltas.append(chunk.delta)

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
                        if pending_answer_deltas:
                            preamble = "".join(pending_answer_deltas)
                            pending_answer_deltas = []
                            await self.event_bus.publish(
                                EventFactory.thinking_chunk(
                                    state.current_step, preamble, buffer
                                )
                            )
                    for tool_call_dict in chunk.tool_calls:
                        # All tool calls are now dicts thanks to provider normalizers.
                        # Resolve the index for this tool call chunk so parallel
                        # tool_use blocks accumulate as distinct entries.
                        #
                        # Resolution order:
                        # 1. Use the provider-supplied `index` field when present
                        #    (Anthropic streams it on every content_block_start;
                        #    OpenAI's delta also carries it).
                        # 2. Else, key off the chunk's `id` — first time we see
                        #    an id, allocate the next slot; subsequent chunks
                        #    with the same id reuse it.
                        # 3. Else, fall back to slot 0 (legacy single-tool path
                        #    when no index/id signal is available).
                        provider_idx = tool_call_dict.get("index")
                        chunk_id = tool_call_dict.get("id")
                        if isinstance(provider_idx, int):
                            idx = provider_idx
                        elif chunk_id is not None:
                            # Find existing slot by id, else allocate next.
                            idx = None
                            for slot, slot_data in accumulated_tool_calls.items():
                                if slot_data.get("id") == chunk_id:
                                    idx = slot
                                    break
                            if idx is None:
                                idx = len(accumulated_tool_calls)
                        else:
                            # No index, no id — keep merging into slot 0.
                            # Either we haven't seen the id yet (first chunk
                            # of the only tool call) or this is a single-tool
                            # turn that streamed across multiple chunks.
                            idx = 0 if accumulated_tool_calls else 0

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
                            accumulated_tool_calls[idx]["type"] = tool_call_dict.get(
                                "type"
                            )

                        # Preserve function_call_metadata (e.g. Gemini thought_signature)
                        if tool_call_dict.get("function_call_metadata"):
                            accumulated_tool_calls[idx]["function_call_metadata"] = (
                                tool_call_dict["function_call_metadata"]
                            )

                        # Update function name if present
                        function_data = tool_call_dict.get("function", {})
                        if function_data.get("name") is not None:
                            accumulated_tool_calls[idx]["function"]["name"] = (
                                function_data.get("name")
                            )

                        # Handle arguments based on format:
                        # - OpenAI: sends progressively longer strings in each chunk
                        # - Anthropic: sends complete dict in final chunk
                        new_args = function_data.get("arguments")
                        if new_args is not None:
                            current_args = accumulated_tool_calls[idx]["function"][
                                "arguments"
                            ]

                            if isinstance(new_args, str):
                                # OpenAI format: string that grows with each chunk
                                # Provider already accumulates, so just use the latest value
                                accumulated_tool_calls[idx]["function"][
                                    "arguments"
                                ] = new_args
                            elif isinstance(new_args, dict):
                                # Anthropic format: dict (usually sent complete in one chunk)
                                if current_args is None or not isinstance(
                                    current_args, dict
                                ):
                                    accumulated_tool_calls[idx]["function"][
                                        "arguments"
                                    ] = new_args
                                else:
                                    # Merge dicts if both exist (defensive)
                                    current_args.update(new_args)
                            else:
                                # Unexpected format, log and store as-is
                                logger.warning(
                                    f"Unexpected arguments type in chunk: {type(new_args)}"
                                )
                                accumulated_tool_calls[idx]["function"][
                                    "arguments"
                                ] = new_args

                        logger.debug(
                            f"Tool call accumulated: {accumulated_tool_calls[idx]}"
                        )

                # Accumulate metrics
                output_tokens = None
                input_tokens = None
                if chunk.usage:
                    tokens_used = chunk.usage.total_tokens
                    output_tokens = getattr(chunk.usage, "completion_tokens", None)
                    input_tokens = getattr(chunk.usage, "prompt_tokens", None)
                if hasattr(chunk, "cost"):
                    cost += chunk.cost
                if getattr(chunk, "finish_reason", None):
                    finish_reason = chunk.finish_reason

            step.tokens_used = tokens_used
            step.cost = cost

            assistant_content = buffer.strip()
            if not accumulated_tool_calls and finish_reason != "length":
                replayed = ""
                for delta in pending_answer_deltas:
                    replayed += delta
                    await self.event_bus.publish(
                        EventFactory.final_answer_chunk(
                            state.current_step, delta, replayed
                        )
                    )
            elif not accumulated_tool_calls and pending_answer_deltas:
                await self.event_bus.publish(
                    EventFactory.thinking_chunk(
                        state.current_step, "".join(pending_answer_deltas), buffer
                    )
                )

            agent_max_tokens = getattr(self.tool_executor.agent, "max_tokens", None)

            # Trace: dump the LLM response we got back, including any accumulated
            # tool calls. Pairs with the OUT line emitted before stream_with_tools.
            logger.info(
                "[LLM_TURN] step=%d IN finish_reason=%s in_tokens=%s out_tokens=%s/max=%s total=%s cost=%s text=%s tool_calls=[%s]",
                state.current_step,
                finish_reason,
                input_tokens,
                output_tokens,
                agent_max_tokens,
                tokens_used,
                cost,
                _preview(assistant_content, 240),
                ", ".join(
                    _summarize_tool_call(tc) for tc in accumulated_tool_calls.values()
                ),
            )

            # Dedicated warning + event when the model hit max_tokens. Carries
            # enough context for postmortem (tool, json length, buffer prefix)
            # without log scraping.
            if finish_reason == "length":
                tool_names = [
                    (tc.get("function", {}) or {}).get("name") or "<unnamed>"
                    for tc in accumulated_tool_calls.values()
                ]
                # Pull truncation details from the first tool call that has them.
                # The stream_normalizer attaches _truncation_error when JSON parse
                # fails at content_block_stop; an _truncation_error-less tool call
                # means the model emitted a complete tool_use block but max_tokens
                # cut off subsequent content (or the buffer was empty).
                trunc_meta = (
                    next(
                        (
                            tc.get("_truncation_error")
                            for tc in accumulated_tool_calls.values()
                            if tc.get("_truncation_error")
                        ),
                        None,
                    )
                    or {}
                )
                accumulated_json_length = trunc_meta.get("accumulated_length")
                raw_prefix = trunc_meta.get("raw_prefix")

                logger.warning(
                    "[LLM_TRUNCATED] step=%d in_tokens=%s out_tokens=%s/max=%s "
                    "tool_calls=%s json_len=%s raw_prefix=%r "
                    "— bump max_tokens or narrow scope.",
                    state.current_step,
                    input_tokens,
                    output_tokens,
                    agent_max_tokens,
                    tool_names,
                    accumulated_json_length,
                    (raw_prefix or "")[:200],
                )

                # Stash on step.metadata so a serialized step trace contains
                # everything needed to diagnose without re-running.
                step.metadata["truncation"] = {
                    "finish_reason": finish_reason,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "max_tokens": agent_max_tokens,
                    "tool_names": tool_names,
                    "accumulated_json_length": accumulated_json_length,
                    "raw_prefix": raw_prefix,
                }

                try:
                    await self.event_bus.publish(
                        EventFactory.llm_truncated(
                            step_number=state.current_step,
                            finish_reason=finish_reason,
                            output_tokens=output_tokens,
                            max_tokens=agent_max_tokens,
                            input_tokens=input_tokens,
                            tool_names=tool_names,
                            accumulated_json_length=accumulated_json_length,
                            raw_prefix=raw_prefix,
                        )
                    )
                except Exception as _evt_err:
                    logger.debug("Failed to publish llm_truncated event: %s", _evt_err)

            # === TURN ROUTING ===
            # Tool calls present = action turn, execute and loop.
            # finish_reason == "length" = response truncated, nudge model.
            # Otherwise = answer turn, accumulated text is the final answer.

            if accumulated_tool_calls and len(accumulated_tool_calls) > 1:
                # Parallel batch path — 2+ tool_use blocks emitted by the model
                # in this assistant turn. Routes through the batch executor
                # which applies the all-or-nothing parallelism rule (parallel
                # only when every tool is parallelizable=True and not
                # require_approval; otherwise serial in input order).
                tool_names = [
                    accumulated_tool_calls[k].get("function", {}).get("name", "?")
                    for k in sorted(accumulated_tool_calls.keys())
                ]
                logger.info(
                    "[ORCH] step=%d PARALLEL_BATCH tools=%s",
                    state.current_step,
                    tool_names,
                )
                await self._handle_parallel_tool_batch(
                    step=step,
                    context=context,
                    state=state,
                    accumulated_tool_calls=accumulated_tool_calls,
                    assistant_content=assistant_content,
                )
            elif accumulated_tool_calls:
                # Take first tool call (ReAct is single-action per step)
                tool_call_data = accumulated_tool_calls.get(0)
                if not tool_call_data:
                    tool_call_data = list(accumulated_tool_calls.values())[0]

                # Extract tool name, arguments, and ID from accumulated data
                step.action = tool_call_data["function"]["name"]
                tool_args = tool_call_data["function"]["arguments"]
                tool_call_id = tool_call_data["id"]

                # Pre-parse string args *before* the truncation check so a
                # JSONDecodeError gets routed through the truncation handler
                # (mirrors Anthropic, where stream_normalizer attaches
                # _truncation_error at content_block_stop).
                _preparse_tool_args_string(
                    tool_call_data, state.current_step, step.action
                )
                tool_args = tool_call_data["function"]["arguments"]

                # Detect tool_use truncation. Two signals:
                # - stream_normalizer attached _truncation_error (JSON didn't parse)
                # - finish_reason="length" with a tool call in flight
                # Either way, the args are incomplete — running schema validation
                # against them surfaces a misleading "missing required params" error.
                # Skip validation and feed the truncation back to the LLM directly.
                truncation_error = tool_call_data.get("_truncation_error")
                if truncation_error or finish_reason == "length":
                    logger.warning(
                        "Step %d - Tool '%s' truncated mid-stream "
                        "(finish_reason=%s, json_parse_failed=%s). Asking model to retry.",
                        state.current_step,
                        step.action,
                        finish_reason,
                        bool(truncation_error),
                    )
                    retry_msg = (
                        f"<tool_use_error>\n"
                        f"The tool call to '{step.action}' was truncated mid-stream by the "
                        f"model's max_tokens limit before the input JSON was complete. The "
                        f"arguments are unusable. Retry with a smaller scope (e.g. fewer rows, "
                        f"narrower date range, or split into multiple smaller calls). If a "
                        f"tool needs to emit a large data array inline, prefer summarising "
                        f"or paginating.\n"
                        f"</tool_use_error>"
                    )
                    step.error = retry_msg
                    step.observation = retry_msg
                    step.metadata["failure_kind"] = "truncation"
                    # Backfill truncation debug fields for the case where the
                    # parse failed without finish_reason=="length" (rare but
                    # possible) — the [LLM_TRUNCATED] block above only fires
                    # on finish_reason=="length".
                    if "truncation" not in step.metadata and truncation_error:
                        step.metadata["truncation"] = {
                            "finish_reason": finish_reason,
                            "tool_names": [step.action],
                            "accumulated_json_length": truncation_error.get(
                                "accumulated_length"
                            ),
                            "raw_prefix": truncation_error.get("raw_prefix"),
                            "parse_error": truncation_error.get("message"),
                        }
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
                        if assistant_content:
                            context.messages.append(
                                Message(
                                    role=MessageRole.ASSISTANT,
                                    content=assistant_content,
                                )
                            )
                        context.messages.append(
                            Message(role=MessageRole.USER, content=step.observation)
                        )
                    # Skip the rest of this step's tool processing on truncation —
                    # outer loop will iterate, model retries with the feedback above.
                    # The finally: block at the end of this method publishes
                    # step_complete and sets execution_time.
                    return step

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
                            step.error = (
                                f"Malformed tool arguments: Invalid JSON format"
                            )
                            step.action_input = {}
                elif isinstance(tool_args, dict):
                    # Already parsed (Anthropic format)
                    step.action_input = tool_args
                else:
                    logger.error(
                        f"Step {state.current_step} - Unexpected tool_args type: {type(tool_args)}, "
                        f"value: {tool_args}"
                    )
                    step.error = f"Malformed tool call: arguments type is {type(tool_args).__name__}"
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
                    required_params = tool_schema.get("parameters", {}).get(
                        "required", []
                    )

                    # Check if any required parameters are missing
                    if required_params:
                        missing_params = [
                            param
                            for param in required_params
                            if param not in step.action_input
                        ]

                        if missing_params:
                            logger.error(
                                f"Step {state.current_step} - Tool '{step.action}' missing required parameters: "
                                f"{missing_params}. Provided parameters: {list(step.action_input.keys())}"
                            )
                            step.error = _format_missing_params_error(
                                tool_name=step.action,
                                missing_params=missing_params,
                                provided_params=list(step.action_input.keys()),
                                tool_schema=tool_schema,
                            )
                            step.observation = step.error
                            step.metadata["failure_kind"] = "schema"
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
                                    Message(
                                        role=MessageRole.USER, content=step.observation
                                    )
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
                                step,
                                context,
                                state,
                                tool_call_id=tool_call_id,
                                tool_description=tool_description,
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
                            step,
                            context,
                            state,
                            tool_call_id=tool_call_id,
                            tool_description=tool_description,
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
                # Answer turn: no tool calls, no truncation. Buffered text
                # deltas were replayed as final_answer_chunk events above;
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
            await self.event_bus.publish(
                EventFactory.step_complete(state.current_step, step)
            )

        # Add observation to context if present (from tool execution)
        # NOTE: This is actually added in _handle_tool_action now with proper tool_call_id
        return step

    async def _handle_parallel_tool_batch(
        self,
        step: ReActStep,
        context: RunContext,
        state: "ExecutionState",
        accumulated_tool_calls: Dict[int, Dict[str, Any]],
        assistant_content: str,
    ) -> None:
        """Execute 2+ tool_use blocks the model emitted in one assistant turn.

        Parses each tool call's args, validates schema, appends ONE
        ASSISTANT message preserving Anthropic's tool_use/tool_result
        pairing invariant, dispatches the batch through
        ``executor.execute_many`` (which applies the all-or-nothing
        parallelism rule), and appends N TOOL messages with each
        observation paired by ``tool_call_id``.

        Per-invocation results are stored on ``step.tool_invocations``.
        For back-compat with single-action consumers, ``step.action``,
        ``step.action_input``, and ``step.observation`` mirror the FIRST
        invocation's fields. Per-call errors land on
        ``step.tool_invocations[i].error``; ``step.error`` is set only
        for step-level failures (e.g. malformed transcript), not for
        individual tool failures (those still surface as observations
        the LLM can react to).

        Rich result types (visualizations, artifacts, media collections,
        LLM block injections) get FULL observation processing when the
        executor's serial fallback fires (mixed-parallelizability or
        approval-required batches). When the batch runs in true parallel
        mode (every tool is ``parallelizable=True``), only visualization
        markers + simple stringification are handled — by construction
        parallelizable tools shouldn't return media/artifact/llm-block
        results, so this is a documented v1 trade-off rather than a
        correctness gap.
        """
        from ..tools import ToolResult

        # ── Phase 1: parse, validate, and build invocations ─────────────
        ordered_keys = sorted(accumulated_tool_calls.keys())
        invocations: List[ToolInvocation] = []
        tool_call_dicts: List[Dict[str, Any]] = []  # for the ASSISTANT message
        pre_exec_errors: Dict[str, str] = {}  # tool_call_id -> error string

        for key in ordered_keys:
            tool_call_data = accumulated_tool_calls[key]
            name = tool_call_data.get("function", {}).get("name")
            raw_args = tool_call_data.get("function", {}).get("arguments")
            tool_call_id = (
                tool_call_data.get("id") or f"call_{state.current_step}_{key}"
            )

            # Parse args (OpenAI sends string; Anthropic sends dict)
            _preparse_tool_args_string(tool_call_data, state.current_step, name)
            raw_args = tool_call_data.get("function", {}).get("arguments")

            inputs: Optional[Dict[str, Any]]
            if raw_args is None:
                inputs = {}
            elif isinstance(raw_args, str):
                if not raw_args or raw_args.strip() == "":
                    inputs = {}
                else:
                    try:
                        inputs = json.loads(raw_args)
                    except json.JSONDecodeError:
                        inputs = {}
                        pre_exec_errors[tool_call_id] = (
                            f"Malformed tool arguments for '{name}': invalid JSON"
                        )
            elif isinstance(raw_args, dict):
                inputs = raw_args
            else:
                inputs = {}
                pre_exec_errors[tool_call_id] = (
                    f"Malformed tool arguments for '{name}': "
                    f"unexpected type {type(raw_args).__name__}"
                )

            # Extract __description for UI labels
            description = None
            if isinstance(inputs, dict):
                description = inputs.pop("__description", None)

            # Truncation / malformed name guards (mirror single-tool path)
            if tool_call_data.get("_truncation_error"):
                pre_exec_errors[tool_call_id] = (
                    f"Tool call to '{name}' was truncated mid-stream. "
                    "Retry with a narrower scope or split into smaller calls."
                )
            if not name:
                pre_exec_errors[tool_call_id] = (
                    "Malformed tool call: function name is missing."
                )

            # Schema validation — required-params check (mirrors single-tool path)
            if name and tool_call_id not in pre_exec_errors and inputs is not None:
                try:
                    tool_schema = self.tool_executor.get_tool_schema(name)
                    required_params = (
                        tool_schema.get("parameters", {}).get("required") or []
                    )
                    if required_params:
                        missing = [p for p in required_params if p not in inputs]
                        if missing:
                            pre_exec_errors[tool_call_id] = (
                                _format_missing_params_error(
                                    tool_name=name,
                                    missing_params=missing,
                                    provided_params=list(inputs.keys()),
                                    tool_schema=tool_schema,
                                )
                            )
                except Exception as exc:
                    # Schema lookup failure is non-fatal; let the
                    # executor surface the real error.
                    logger.debug(
                        "Step %d - Schema lookup failed for '%s': %s",
                        state.current_step,
                        name,
                        exc,
                    )

            invocations.append(
                ToolInvocation(
                    tool_call_id=tool_call_id,
                    name=name,
                    inputs=inputs or {},
                    description=description,
                )
            )
            tool_call_dicts.append(tool_call_data)

        # ── Phase 2: append the assistant message with ALL tool_calls ──
        # This satisfies the Anthropic invariant that every tool_use block
        # in an assistant message must be followed (across messages) by a
        # matching tool_result keyed on the same tool_call_id.
        context.messages.append(
            Message(
                role=MessageRole.ASSISTANT,
                content=assistant_content,
                tool_calls=tool_call_dicts,
            )
        )

        # ── Phase 3: execute the batch ─────────────────────────────────
        # Tools that pre-failed (schema/truncation/missing-name) get a
        # synthetic error observation and don't get sent to the executor.
        # Successfully-parsed tools go through execute_many, which decides
        # parallel vs serial based on the all-or-nothing rule.
        runnable: List[ToolCall] = []
        runnable_indices: List[int] = []
        for i, inv in enumerate(invocations):
            if inv.tool_call_id in pre_exec_errors or not inv.name:
                continue
            runnable.append(
                ToolCall(
                    tool_call_id=inv.tool_call_id, name=inv.name, inputs=inv.inputs
                )
            )
            runnable_indices.append(i)

        # Publish action_planned events for visibility (one per invocation).
        for inv in invocations:
            if inv.name:
                try:
                    await self.event_bus.publish(
                        EventFactory.action_planned(
                            state.current_step,
                            inv.name,
                            inv.inputs,
                            inv.description,
                            tool_call_id=inv.tool_call_id,
                        )
                    )
                except Exception as evt_err:
                    logger.debug("Failed to publish action_planned: %s", evt_err)

        results: List[Optional[ToolResult]] = [None] * len(invocations)
        # Set when an approval-required tool in the batch trips the approval
        # gate. ``execute_many`` forces a batch serial when any tool is
        # ``require_approval=True`` and raises ``ToolApprovalRequired`` at that
        # tool. We catch it here (the single-tool path catches it too, but the
        # batch path previously let it escape to the generic step-error handler,
        # which routed an unsatisfiable approval through the recovery ladder AND
        # left the just-appended tool_use blocks without matching tool_results —
        # the next LLM call then 400s with "tool_use ids were found without
        # tool_result blocks"). Pausing here keeps the transcript valid.
        approval_pause: Optional[ToolApprovalRequired] = None
        if runnable:
            # Mirror the per-step ctx.run_state injection that _execute_tool
            # does on the single-tool path. Without this, tools running in a
            # parallel batch (notably dispatch_assistant) see stale or absent
            # step_number / media_store. Each parallel branch then gets its
            # OWN Context copy via _execute_parallel's create_task, so this
            # write is visible to all branches at the moment of fan-out.
            if context is not None:
                context.run_state.media_store = state.media_store
                context.run_state.step_number = state.current_step
                context.run_state.event_bus = self.event_bus
                # Legacy dual-write — remove once readers move to run_state.
                if isinstance(getattr(context, "deps", None), dict):
                    context.deps["media_store"] = state.media_store
                    context.deps["step_number"] = state.current_step
                    context.deps["event_bus"] = self.event_bus

            # Provide context only if any tool needs it (single-tool path
            # checks per-tool; here we pass it always — execute_tool
            # internally handles the per-tool needs_context check).
            try:
                batch_results = await self.tool_executor.execute_many(
                    runnable, context=context
                )
                for runnable_pos, idx in enumerate(runnable_indices):
                    results[idx] = batch_results[runnable_pos]
            except ToolApprovalRequired as e:
                approval_pause = e
                logger.info(
                    "Parallel batch contains approval-required tool '%s' - "
                    "pausing execution for user approval",
                    e.tool_name,
                )

        # ── Phase 4: process results into observations ─────────────────
        # Per-invocation observation handling: basic stringification +
        # visualization marker detection. More exotic result types
        # (media/artifact/llm_block_injection) fall through to str() in
        # batch mode — see method docstring for the v1 trade-off.
        from miiflow_agent.visualization import (
            extract_visualization_data,
            is_visualization_result,
        )

        if approval_pause is not None:
            # An approval-required tool tripped the gate. Pause the run (mirrors
            # the single-tool ToolApprovalRequired handler) and give EVERY
            # invocation a placeholder observation so Phase 5 can pair each
            # tool_use block with a tool_result. ``execute_many`` raised before
            # returning, so no tool in the batch actually ran.
            approval_inv = next(
                (iv for iv in invocations if iv.name == approval_pause.tool_name),
                None,
            )
            state.needs_clarification = True
            state.clarification_data = {
                "type": "tool_approval",
                "tool_name": approval_pause.tool_name,
                "tool_inputs": approval_pause.tool_inputs
                or (approval_inv.inputs if approval_inv else {}),
                "tool_description": (approval_inv.description if approval_inv else "")
                or "",
                "tool_schema": self.tool_executor.get_tool_schema(
                    approval_pause.tool_name
                ),
                "tool_call_id": approval_inv.tool_call_id if approval_inv else None,
                "reason": approval_pause.reason,
            }
            interrupt = await self._record_interrupt(
                context,
                state,
                kind="tool_approval",
                payload={
                    "tool_name": approval_pause.tool_name,
                    "tool_inputs": approval_pause.tool_inputs
                    or (approval_inv.inputs if approval_inv else {}),
                    "tool_description": (
                        approval_inv.description if approval_inv else ""
                    )
                    or "",
                    "tool_schema": self.tool_executor.get_tool_schema(
                        approval_pause.tool_name
                    ),
                    "reason": approval_pause.reason,
                },
                tool_call_id=approval_inv.tool_call_id if approval_inv else None,
            )
            state.clarification_data["interrupt_id"] = interrupt.interrupt_id
            state.clarification_data["raised_by_path"] = interrupt.raised_by_path
            try:
                await self.event_bus.publish(
                    ReActEvent(
                        event_type=ReActEventType.TOOL_APPROVAL_NEEDED,
                        step_number=state.current_step,
                        data=state.clarification_data,
                    )
                )
            except Exception as evt_err:
                logger.debug("Failed to publish approval event: %s", evt_err)
            for inv in invocations:
                if inv.name == approval_pause.tool_name:
                    inv.observation = (
                        "Tool execution paused - waiting for user approval."
                    )
                else:
                    inv.observation = (
                        f"Not executed - batch paused pending approval of "
                        f"'{approval_pause.tool_name}'."
                    )
            # Skip normal result processing; fall through to Phase 5 which
            # appends one TOOL message per invocation (pairing invariant).
            invocations_to_process: List[ToolInvocation] = []
        else:
            invocations_to_process = invocations

        for i, inv in enumerate(invocations_to_process):
            # Pre-execution error case
            if inv.tool_call_id in pre_exec_errors:
                err = pre_exec_errors[inv.tool_call_id]
                inv.error = err
                inv.observation = err
                continue

            result = results[i]
            if result is None:
                inv.error = "Tool execution returned no result"
                inv.observation = inv.error
                continue

            if not result.success:
                sanitized = _sanitize_error_message(result.error or "Unknown error")
                inv.error = result.error
                inv.observation = f"Tool execution failed: {sanitized}"
                # Propagate the validation marker the registry stamped from the
                # raised exception's `is_tool_validation_error` attribute. Used
                # below to classify an all-failed step as schema-kind so the
                # recovery_manager skips the runtime ladder.
                inv.is_validation_error = bool(
                    (result.metadata or {}).get("is_validation_error")
                )
            else:
                # Visualization → emit event + use [VIZ:id] marker
                if is_visualization_result(result.output):
                    viz_data = extract_visualization_data(result.output)
                    if viz_data:
                        try:
                            await self.event_bus.publish(
                                EventFactory.visualization(
                                    state.current_step, viz_data, inv.name
                                )
                            )
                        except Exception as evt_err:
                            logger.debug(
                                "Failed to publish visualization event: %s", evt_err
                            )
                        inv.observation = f"[VIZ:{viz_data['id']}]"
                    else:
                        inv.observation = _observation_with_citation_ref(result.output)
                else:
                    inv.observation = _observation_with_citation_ref(result.output)

            if await self._handle_tool_approval_marker_result(
                context,
                state,
                result,
                parent_tool_call_id=inv.tool_call_id,
            ):
                inv.observation = "Tool execution paused - waiting for user approval."

            # Per-invocation observation event for downstream consumers.
            self._record_tool_ledger_entry(
                context,
                state,
                tool_name=inv.name,
                inputs=inv.inputs,
                observation=inv.observation,
                success=inv.error is None,
            )
            try:
                await self.event_bus.publish(
                    EventFactory.observation(
                        state.current_step,
                        inv.observation,
                        inv.name,
                        inv.error is None,
                        tool_call_id=inv.tool_call_id,
                    )
                )
            except Exception as evt_err:
                logger.debug("Failed to publish observation event: %s", evt_err)

        # ── Phase 5: append N TOOL messages, paired by tool_call_id ────
        for inv in invocations:
            context.messages.append(
                Message(
                    role=MessageRole.TOOL,
                    content=inv.observation or "",
                    tool_call_id=inv.tool_call_id,
                )
            )

        # ── Phase 6: write to step (canonical + back-compat) ───────────
        step.tool_invocations = invocations
        # Back-compat: mirror the first invocation's fields on legacy
        # singular attributes so consumers that haven't migrated still
        # observe sensible values.
        first = invocations[0] if invocations else None
        if first is not None:
            step.action = first.name
            step.action_input = first.inputs
            step.observation = first.observation
            # step.error is set ONLY if every invocation failed — this
            # signals "this whole step failed" to recovery_manager.
            # Per-invocation failures stay on each invocation.error and
            # don't trigger step-level recovery (the LLM sees the errors
            # as tool observations and can react).
            if all(inv.error is not None for inv in invocations):
                step.error = (
                    f"All {len(invocations)} parallel tool calls failed. "
                    f"First error: {first.error}"
                )
                # If every failure was a deterministic input-shape rejection
                # (e.g. GAQL preflight), classify as schema so recovery_manager
                # short-circuits: the per-invocation tool observations already
                # carry the corrective hint the LLM needs to fix its next call.
                # Without this, two parallel preflight failures would burn the
                # 3-attempt recovery ladder and force a fallback answer.
                if all(
                    getattr(inv, "is_validation_error", False) for inv in invocations
                ):
                    step.metadata["failure_kind"] = "schema"
                else:
                    step.metadata["failure_kind"] = "all_failed"

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
            EventFactory.action_planned(
                state.current_step,
                step.action,
                step.action_input,
                tool_description,
                tool_call_id=tool_call_id,
            )
        )

        await self.event_bus.publish(
            EventFactory.action_executing(
                state.current_step,
                step.action,
                step.action_input,
                tool_description,
                tool_call_id=tool_call_id,
            )
        )

        # Execute tool
        try:
            result = await self._execute_tool(step, context, state)

            if result.success:
                # Check if this is a visualization result BEFORE stringification
                # This is critical because str(VisualizationResult) returns [VIZ:uuid]
                # which loses the actual chart data
                from miiflow_agent.visualization import (
                    is_visualization_result,
                    extract_visualization_data,
                )
                from miiflow_agent.visualization.types import (
                    is_media_result,
                    extract_media_data,
                    is_media_collection,
                    extract_media_collection,
                    extract_collection_metadata,
                    is_llm_block_injection,
                    extract_llm_blocks,
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
                            EventFactory.media(
                                state.current_step, media_data, step.action
                            )
                        )
                        media_id = media_data["id"]
                        media_url = media_data.get("url", "")
                        if media_url and not media_url.startswith("data:"):
                            state.media_store[media_id] = media_url
                        # Correlate metadata entries by index (tools return parallel lists)
                        meta = (
                            metadata_items[idx] if idx < len(metadata_items) else None
                        )
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
                    media_id = media_data["id"]
                    media_url = media_data.get("url", "")

                    # Store media URL in execution state so subsequent tool calls
                    # (e.g. image editing) can resolve media_ref:<id> to actual URL.
                    # Only store actual URLs, not data URIs (which can be MBs of base64).
                    # System tools already persist to S3 before reaching here, so
                    # media_url should be an S3 URL for normal image gen flows.
                    if media_url and not media_url.startswith("data:"):
                        state.media_store[media_id] = media_url

                    # Always include media_ref so LLM can reference this image
                    # in subsequent tool calls (e.g. edit_gpt_image_1)
                    if media_url and not media_url.startswith("data:"):
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
                            EventFactory.visualization(
                                state.current_step, viz_data, step.action
                            )
                        )
                        # Store marker for observation (what gets sent to LLM context)
                        # For auth_prompt visualizations, include context so LLM knows tool was blocked
                        if viz_data.get("type") == "auth_prompt":
                            provider_name = viz_data.get("data", {}).get(
                                "providerName", "the provider"
                            )
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
                            EventFactory.artifact(
                                state.current_step, artifact_data, step.action
                            )
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

                if await self._handle_tool_approval_marker_result(
                    context,
                    state,
                    result,
                    parent_tool_call_id=tool_call_id,
                ):
                    step.observation = (
                        "Tool execution paused - waiting for user approval."
                    )

                # Check if this is a clarification request
                from ..tools.clarification import (
                    is_clarification_result,
                    extract_clarification_data,
                )

                if not state.needs_clarification and is_clarification_result(result):
                    clarification = extract_clarification_data(result)
                    if clarification:
                        # Phase 1: deterministic established-facts short-circuit (R4).
                        # Check the asked questions against facts already resolved this
                        # run (threaded in via deps["established_facts"] by the adapter —
                        # absent ⇒ behaviour identical to before). Questions whose stable
                        # key is already answered NEVER re-pause; if every question is
                        # resolved we skip the pause entirely and hand the model the
                        # known answers as the observation.
                        from ..checkpoint import EstablishedFact
                        from ..interrupt import decide_clarification

                        facts_by_key = {}
                        deps = getattr(context, "deps", None)
                        if isinstance(deps, dict):
                            for fd in deps.get("established_facts") or []:
                                try:
                                    f = EstablishedFact.from_dict(fd)
                                    facts_by_key[f.key] = f
                                except Exception:
                                    continue

                        question_dicts = [q.to_dict() for q in clarification.questions]
                        clarification_round = 0
                        if isinstance(deps, dict):
                            clarification_round = int(
                                deps.get("clarification_round", 0) or 0
                            )
                        decision = decide_clarification(
                            question_dicts,
                            facts_by_key,
                            interrupt_count=clarification_round,
                        )

                        if not decision.should_pause:
                            # Everything was already answered — do NOT pause; the model
                            # proceeds deterministically with the known answers.
                            step.observation = (
                                decision.resolved_observation or step.observation
                            )
                            logger.info(
                                "Clarification short-circuited: all question(s) already settled"
                            )
                        else:
                            state.needs_clarification = True
                            clarification_data = clarification.to_dict()
                            clarification_data["questions"] = decision.pause_questions
                            clarification_data["tool_call_id"] = tool_call_id
                            raw_clarification_output = (
                                result.output if isinstance(result.output, dict) else {}
                            )
                            for meta_key in (
                                "handle",
                                "child_assistant_id",
                                "subagent_id",
                                "status",
                                "subagent_path",
                            ):
                                if meta_key in raw_clarification_output:
                                    clarification_data[meta_key] = (
                                        raw_clarification_output[meta_key]
                                    )
                            subagent_path = raw_clarification_output.get(
                                "subagent_path"
                            )
                            raised_by_path = ["root"] + list(subagent_path or [])
                            interrupt = await self._record_interrupt(
                                context,
                                state,
                                kind="clarification",
                                payload={
                                    "questions": decision.pause_questions,
                                    "context": clarification.context,
                                    **{
                                        k: v
                                        for k, v in clarification_data.items()
                                        if k
                                        in (
                                            "handle",
                                            "child_assistant_id",
                                            "subagent_id",
                                            "status",
                                            "subagent_path",
                                        )
                                    },
                                },
                                tool_call_id=tool_call_id,
                                raised_by_path=raised_by_path,
                            )
                            clarification_data["interrupt_id"] = interrupt.interrupt_id
                            clarification_data["raised_by_path"] = (
                                interrupt.raised_by_path
                            )
                            state.clarification_data = clarification_data
                            logger.info(
                                f"Clarification requested: {len(decision.pause_questions)} question(s)"
                            )

                            # Emit clarification event
                            await self.event_bus.publish(
                                ReActEvent(
                                    event_type=ReActEventType.CLARIFICATION_NEEDED,
                                    step_number=state.current_step,
                                    data={
                                        "step": state.current_step,
                                        "questions": decision.pause_questions,
                                        "context": clarification.context,
                                        "tool_call_id": tool_call_id,
                                        "interrupt_id": interrupt.interrupt_id,
                                        "raised_by_path": interrupt.raised_by_path,
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
            self._record_tool_ledger_entry(
                context,
                state,
                tool_name=step.action,
                inputs=step.action_input,
                observation=step.observation,
                success=result.success,
            )
            await self.event_bus.publish(
                EventFactory.observation(
                    state.current_step,
                    step.observation,
                    step.action,
                    result.success,
                    tool_call_id=tool_call_id,
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
            interrupt = await self._record_interrupt(
                context,
                state,
                kind="tool_approval",
                payload={
                    "tool_name": e.tool_name,
                    "tool_inputs": e.tool_inputs or {},
                    "tool_description": tool_description or "",
                    "tool_schema": self.tool_executor.get_tool_schema(e.tool_name),
                    "reason": e.reason,
                },
                tool_call_id=tool_call_id,
            )
            state.clarification_data["interrupt_id"] = interrupt.interrupt_id
            state.clarification_data["raised_by_path"] = interrupt.raised_by_path

            # Emit approval event for SSE
            await self.event_bus.publish(
                ReActEvent(
                    event_type=ReActEventType.TOOL_APPROVAL_NEEDED,
                    step_number=state.current_step,
                    data=state.clarification_data,
                )
            )

            logger.info(f"Tool '{e.tool_name}' requires approval - pausing execution")

            # CRITICAL: Must add a tool result to context, otherwise Anthropic API
            # rejects with "tool_use ids were found without tool_result blocks"
            if tool_call_id:
                observation_message = Message(
                    role=MessageRole.TOOL,
                    content="Tool execution paused - waiting for user approval.",
                    tool_call_id=tool_call_id,
                )
                context.messages.append(observation_message)

        except PlanApprovalRequired as e:
            # `exit_plan_mode` raised — pause the loop while the user
            # decides whether to approve the proposed plan. Reuses the
            # same `state.needs_clarification` pause mechanism the
            # tool-approval path uses; the streaming service
            # distinguishes them by `clarification_data["type"]`.
            state.needs_clarification = True
            state.clarification_data = {
                "type": "plan_approval",
                "plan": e.plan_text,
                "tool_call_id": tool_call_id or e.tool_call_id,
            }
            interrupt = await self._record_interrupt(
                context,
                state,
                kind="plan_approval",
                payload={"plan": e.plan_text},
                tool_call_id=tool_call_id or e.tool_call_id,
            )
            state.clarification_data["interrupt_id"] = interrupt.interrupt_id
            state.clarification_data["raised_by_path"] = interrupt.raised_by_path

            await self.event_bus.publish(
                ReActEvent(
                    event_type=ReActEventType.PLAN_APPROVAL_NEEDED,
                    step_number=state.current_step,
                    data=state.clarification_data,
                )
            )

            logger.info(
                "exit_plan_mode raised — pausing for plan approval "
                f"(plan length: {len(e.plan_text)} chars)"
            )

            # Same Anthropic tool_use/tool_result pairing invariant as
            # the ToolApprovalRequired path: a tool_use without a
            # matching tool_result block makes the next API call 4xx.
            if tool_call_id:
                observation_message = Message(
                    role=MessageRole.TOOL,
                    content="Plan submitted — waiting for user approval.",
                    tool_call_id=tool_call_id,
                )
                context.messages.append(observation_message)

        except Exception as e:
            # Sanitize error message for LLM consumption
            sanitized_error = _sanitize_error_message(str(e))
            step.error = (
                f"Tool execution error: {str(e)}"  # Keep full error for debugging
            )
            step.observation = f"Tool '{step.action}' failed: {sanitized_error}"
            logger.error(f"Tool execution failed: {e}", exc_info=True)

            await self.event_bus.publish(
                EventFactory.observation(
                    state.current_step,
                    step.observation,
                    step.action,
                    False,
                    tool_call_id=tool_call_id,
                )
            )

            # Add tool result to context even on exception (required for native tool calling)
            # Without this, Anthropic API will reject subsequent calls with:
            # "tool_use ids were found without tool_result blocks"
            if tool_call_id:
                observation_message = Message(
                    role=MessageRole.TOOL,
                    content=step.observation,
                    tool_call_id=tool_call_id,
                )
                context.messages.append(observation_message)
                logger.debug(
                    f"Step {state.current_step} - Added error tool result to context with ID: {tool_call_id}"
                )

    async def _execute_tool(
        self, step: ReActStep, context: RunContext, state: "ExecutionState" = None
    ):
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

        # Expose media_store + event_bus + step_number to tools so they can:
        #   - resolve media_ref IDs without re-implementing resolution logic
        #     (e.g. analyze_creative)
        #   - publish events back to the parent's stream during execution
        #     (e.g. dispatch_assistant streaming a sub-assistant's progress)
        # Safe in-place assignment because the orchestrator owns ctx lifecycle
        # for the duration of a ReAct run.
        if context is not None:
            if state is not None:
                context.run_state.media_store = state.media_store
                context.run_state.step_number = state.current_step
            context.run_state.event_bus = self.event_bus

            # Legacy dual-write to ctx.deps (see RunState docstring). Remove
            # once every reader of these keys has been switched to
            # ctx.run_state.*.
            if isinstance(getattr(context, "deps", None), dict):
                if state is not None:
                    context.deps["media_store"] = state.media_store
                    context.deps["step_number"] = state.current_step
                context.deps["event_bus"] = self.event_bus

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
        media_ref_pattern = re.compile(r"^media_ref:(.+)$")
        # Match UUIDs in hallucinated file paths like /mnt/data/<uuid>.png
        uuid_pattern = re.compile(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            re.IGNORECASE,
        )

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
                if not stripped.startswith(("http://", "https://", "data:")):
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
                        stripped.startswith("/") or stripped.startswith("file://")
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

    def _handle_step_error(
        self, step: ReActStep, error: Exception, state: "ExecutionState"
    ):
        """Handle step execution errors."""
        step.error = f"Step execution failed: {str(error)}"
        step.observation = f"An error occurred: {str(error)}"
        logger.error(f"Step {state.current_step} failed: {error}", exc_info=True)

    async def _publish_final_answer_event(
        self, step: ReActStep, state: "ExecutionState"
    ):
        """Publish the closing final_answer event for the complete answer.

        Answer chunks are streamed live as they arrive in the streaming loop;
        this single event signals completion and gives consumers like
        agent.run() the full answer string in one place.
        """
        if step.answer:
            await self.event_bus.publish(
                EventFactory.final_answer(state.current_step, step.answer)
            )

    async def _build_result(
        self, state: "ExecutionState", context: RunContext = None
    ) -> ReActResult:
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

        if context is not None:
            checkpoint = getattr(context, "checkpoint", None)
            interrupt = (
                checkpoint.active_interrupt()
                if checkpoint is not None and hasattr(checkpoint, "active_interrupt")
                else None
            )
            if interrupt is not None:
                result.metadata["pending_interrupt"] = interrupt.to_dict()

        # Carry the structured failure (set by ``_should_stop`` when a
        # safety condition halts the loop) so the dispatch envelope can
        # surface a real cause to the parent agent.
        if state.failure_metadata is not None:
            result.metadata["failure"] = state.failure_metadata

        return result

    def _build_error_result(
        self, state: "ExecutionState", error: Exception
    ) -> ReActResult:
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

        return (
            "I wasn't able to produce a complete answer from this run. "
            "Please try again, or narrow the request if it involves a lot of work."
        )

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
