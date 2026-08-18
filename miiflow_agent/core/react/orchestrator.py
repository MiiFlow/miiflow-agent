"""Focused ReAct orchestrator with clean separation of concerns."""

import asyncio
import inspect
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from ..agent import RunContext
from ..checkpoint import (
    DispatchLedgerEntry,
    PendingInterrupt,
    make_ledger_digest,
    stable_json_hash,
)
from ..interrupt import mint_interrupt_id
from ..observation import (
    ObservationRecord,
    RecordedObservation,
    bound_observation_for_llm,
    get_observation_sink,
)
from ..context import CompressionVerdict
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


def _optimistic_answer_streaming_enabled() -> bool:
    """Kill switch: MIIFLOW_OPTIMISTIC_ANSWER_STREAMING=0 reverts to
    buffer-and-replay of answer deltas (no live streaming, no retractions)."""
    return os.environ.get("MIIFLOW_OPTIMISTIC_ANSWER_STREAMING", "1").lower() not in (
        "0",
        "false",
    )


class _LazyTrace:
    """Defer expensive trace-string rendering until the log record is emitted."""

    def __init__(self, render):
        self._render = render

    def __str__(self) -> str:
        try:
            return self._render()
        except Exception as exc:  # noqa: BLE001
            return f"<trace failed: {exc}>"


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


def _attach_search_blocks(
    messages: List[Message], blocks: List[Dict[str, Any]], since_index: int
) -> bool:
    """Record provider tool-search blocks on this step's assistant message.

    The provider streams a ``server_tool_use`` / ``tool_search_tool_result``
    pair whenever it resolves a ``defer_loading`` tool. Nothing executes on our
    side, but the pair is the model's only record that the discovery happened —
    drop it and the model re-runs the same search on every later turn, paying a
    round-trip each time.

    ``since_index`` is the length of ``messages`` when the step began, and the
    search is bounded to messages at or after it. Without that bound the caller
    (a ``finally``, which also runs on the error, truncation and empty-turn
    paths where this step appended NO assistant message) would walk back into a
    previous step's — or a previous turn's — assistant message and mutate it.
    That message was already sent verbatim in the cached prefix, so editing it
    diverges the history, misses the cache tier and re-bills the whole prefix:
    precisely the cost this feature exists to avoid. It would also show the
    model a search result on a turn where it never searched.

    Only complete pairs are kept: the API rejects a ``server_tool_use`` replayed
    without its matching result, which would 400 every remaining turn of the
    thread. An orphan half is discarded, costing one repeated search instead.
    Ids must be non-empty — two id-less halves would otherwise "pair" with each
    other on the empty string, and a blank id is itself rejected by the API.

    Returns True if the blocks were attached.
    """
    used = {
        b.get("id") for b in blocks if b.get("type") == "server_tool_use" and b.get("id")
    }
    answered = {
        b.get("tool_use_id")
        for b in blocks
        if b.get("type") == "tool_search_tool_result" and b.get("tool_use_id")
    }
    paired = used & answered
    complete = [
        b
        for b in blocks
        if (b.get("id") if b.get("type") == "server_tool_use" else b.get("tool_use_id"))
        in paired
    ]
    if not complete:
        logger.debug(
            "Discarding %d unpaired tool-search block(s); model will re-search",
            len(blocks),
        )
        return False

    for index in range(len(messages) - 1, max(since_index, 0) - 1, -1):
        message = messages[index]
        if message.role != MessageRole.ASSISTANT:
            continue
        existing = message.metadata if isinstance(message.metadata, dict) else {}
        # Copy the list too, not just the dict: `{**existing}` is shallow, so a
        # setdefault-then-extend would mutate a list shared with the original
        # metadata (and with any other message copied from it).
        merged = {**existing}
        merged["tool_search_blocks"] = [
            *(merged.get("tool_search_blocks") or []),
            *complete,
        ]
        message.metadata = merged
        return True

    logger.debug(
        "No assistant message from this step to carry %d tool-search block(s); "
        "model will re-search",
        len(complete),
    )
    return False


#: Appended to a rendered visualization's ``[VIZ:id]`` observation. Stated
#: here, at the moment the model holds the handle, rather than only in a
#: system prompt: it is what makes revising a visual safe. Without it the model
#: had no way to know what became of a render it did NOT embed — one
#: production strategist rendered the same KPI card three times, refining the
#: labels, and embedded only the last; the host appended the two abandoned
#: drafts under the answer as duplicates. The contract is now explicit and
#: symmetric with the host's finalize step (unembedded renders are dropped
#: whenever the answer embeds any marker), so "render again, embed only the
#: final one" is the documented way to revise.
VISUALIZATION_MARKER_CONTRACT = (
    "Rendered. It appears in the answer only where this marker is embedded; "
    "a render that is not embedded is dropped, so to revise it, render again "
    "and embed only the final marker."
)


def visualization_observation(viz_data: Dict[str, Any]) -> str:
    """What the model is told after a tool returned a visualization.

    For a chart it is the ``[VIZ:id]`` handle plus the one rule the model
    needs to hold it correctly (:data:`VISUALIZATION_MARKER_CONTRACT`). It is
    wrong for an ``auth_prompt``, where the marker is the ONLY record that the
    tool did no work — read back as "visualization generated", it reports a
    blocked call as a success, and the model carries on as though it had
    data. The auth card is also not something the model places: the host shows
    it regardless of embedding, so it gets its own wording rather than the
    embed rule.

    Shared because the two result paths had drifted: the single-tool path
    spelled the auth case out while the batch path emitted the bare marker, so
    the same blocked tool explained itself or didn't depending on whether the
    model happened to call it alongside another one.
    """
    marker = f"[VIZ:{viz_data.get('id', 'unknown')}]"
    if viz_data.get("type") != "auth_prompt":
        return f"{marker} {VISUALIZATION_MARKER_CONTRACT}"
    provider_name = (viz_data.get("data") or {}).get("providerName") or "the provider"
    return (
        f"{marker} No data was returned: {provider_name} is not connected. "
        f"A connect prompt has been shown to the user. Do not retry this or any "
        f"other {provider_name} tool in this run — tell the user what you needed "
        f"{provider_name} for, and continue with anything that does not need it."
    )


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


def _extract_partial_results(
    steps: List[ReActStep],
    *,
    max_results: int = 12,
    excerpt: int = 400,
    input_truncation: int = 300,
) -> List[Dict[str, Any]]:
    """The successful half of `_extract_failure_metadata`.

    Walks the step history for tool invocations that SUCCEEDED and returns a
    bounded description of each: what was called, with what, the durable
    `observation_ref`, and a short excerpt.

    This exists because a run that ends without an answer was, until now,
    reported as if it had done nothing. On 2026-08-02 a `google_ads_specialist`
    dispatch ran 738s, completed six `google_ads_query` calls holding the entire
    dataset its parent asked for, then force-stopped on consecutive empty model
    turns and returned a fixed apology — three times in a row, until the parent
    exhausted its dispatch budget. The data was retrieved and then discarded at
    the exit. The refs are durable rows, so naming them lets the caller
    `read_observation` the full payload rather than re-fetching from the vendor.

    Oldest-first, because a caller reconstructing what happened wants the
    sequence in the order it ran. Bounded on every axis (`observation_policy`
    rules apply to anything the model will be shown).
    """
    results: List[Dict[str, Any]] = []
    for step in steps:
        for inv in step.all_invocations:
            if inv.error is not None or _observation_looks_like_error(inv.observation):
                continue
            if not inv.name:
                continue
            entry: Dict[str, Any] = {"tool": inv.name}
            if inv.description:
                entry["description"] = inv.description
            if inv.inputs:
                entry["inputs"] = _truncate_input(inv.inputs, input_truncation)
            if inv.observation_ref:
                entry["observation_ref"] = inv.observation_ref
            if inv.observation:
                entry["excerpt"] = inv.observation[:excerpt]
            results.append(entry)

    # Keep the most recent when a long run overflows: later calls are the ones
    # that refined earlier ones, and the caller is trying to finish the job.
    if len(results) > max_results:
        results = results[-max_results:]
    return results


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

        # Collaborators: cohesive method clusters extracted from this class
        # (which keeps thin delegates so the call surface is unchanged).
        from .answer_synthesis import AnswerSynthesis
        from .approval_flow import ApprovalResumeFlow
        from .context_coordinator import ContextCoordinator
        from .recording import OutcomeRecording

        from .step_streaming import StepStreamer

        from .tool_actions import ToolActionHandler

        self._context_coord = ContextCoordinator(self)
        self._answers = AnswerSynthesis(self)
        self._recording = OutcomeRecording(self)
        self._approval_flow = ApprovalResumeFlow(self)
        self._step_streamer = StepStreamer(self)
        self._tool_actions = ToolActionHandler(self)

        # Recovery decides *when* to compact; only the orchestrator can build
        # the request shape the engine needs to do it, so it owns the *how*.
        # Without this wiring a ContextEngine handed to RecoveryManager makes
        # COMPRESS_AND_RETRY a no-op (the engine has no compress_if_needed).
        if (
            self.recovery_manager is not None
            and getattr(self.recovery_manager, "compress_fn", None) is None
        ):
            self.recovery_manager.compress_fn = self._compress_for_recovery

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

                # Seed this run's media_ref -> URL store from a map the caller
                # pre-installed on ctx (the Django adapter rebuilds it from the
                # thread's prior media). Without this the store starts empty
                # every run, so a media_ref the model saw in an EARLIER turn no
                # longer resolves and "save the image you showed me last turn"
                # is forced to regenerate a fresh, non-identical image. Merge
                # (not replace): media generated THIS run writes into the same
                # dict. Then point BOTH ctx surfaces at the run's store now — as
                # with event_bus/dispatch_counter above — so the seed is visible
                # to the very first tool, not only after the per-step refresh.
                preseeded_media = (
                    context.deps.get("media_store") if deps_is_dict else None
                )
                if not preseeded_media:
                    preseeded_media = getattr(
                        context.run_state, "media_store", None
                    )
                if isinstance(preseeded_media, dict) and preseeded_media:
                    execution_state.media_store.update(preseeded_media)
                context.run_state.media_store = execution_state.media_store

                # Legacy dual-write to ctx.deps for callers that haven't
                # migrated to ctx.run_state.* yet. Remove once every reader
                # (the dispatch closures, batch_executor, memory_fs_tools)
                # has been switched. New code should NOT read these keys.
                if deps_is_dict:
                    context.deps["dispatch_counter"] = (
                        context.run_state.dispatch_counter
                    )
                    context.deps["event_bus"] = self.event_bus
                    context.deps["media_store"] = execution_state.media_store

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

            if self.context_compressor and hasattr(
                self.context_compressor, "on_session_start"
            ):
                try:
                    await self.context_compressor.on_session_start(
                        self.tool_executor.build_request_shape(context.messages)
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug("[ORCH] context engine session start failed: %s", exc)

            while execution_state.is_running:
                execution_state.current_step += 1

                # Size the request before every LLM call, not just once at the
                # start. A run that accumulates twenty tool results grows past
                # the window mid-loop; checking only up front means the first
                # sign of trouble is a provider 400, handled reactively by the
                # recovery manager after the latency has already been paid.
                # The check itself is local and cheap — that is precisely why
                # `should_compress` does no I/O.
                if self.context_compressor:
                    await self._maybe_compress(
                        context,
                        phase=f"step={execution_state.current_step}",
                        state=execution_state,
                    )

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
                    # Convert the work already in hand into an answer before
                    # leaving. Without this, EVERY safety halt discards the
                    # whole run — see _answer_after_halt.
                    await self._answer_after_halt(context, execution_state)
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
                    from .message_repair import (
                        is_structural_message_error,
                        repair_tool_pairing,
                    )
                    from .recovery import is_context_overflow_error

                    # A structural 400 (tool_use/tool_result pairing) means
                    # the provider will reject EVERY subsequent request with
                    # this history — the session is stuck, not just this
                    # step. Repair once and resend; a second rejection
                    # propagates so a producer bug can't hide behind an
                    # infinite repair loop.
                    if (
                        is_structural_message_error(_llm_err)
                        and not execution_state.structural_repair_attempted
                    ):
                        execution_state.structural_repair_attempted = True
                        repaired, anomalies = repair_tool_pairing(context.messages)
                        if anomalies:
                            logger.exception(
                                "[ORCH] provider rejected message structure "
                                "(%s); repaired and resending: %s",
                                _preview(str(_llm_err), 200),
                                "; ".join(anomalies),
                            )
                            context.messages = repaired
                            continue
                        logger.exception(
                            "[ORCH] provider rejected message structure but "
                            "repair found nothing to fix: %s",
                            _preview(str(_llm_err), 200),
                        )

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
                        await self._record_recovery_halt(
                            execution_state,
                            stop_reason="context_overflow",
                            description=str(_llm_err)[:300],
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

                # A provider that rejected the REQUEST (not the model's
                # answer) will reject the identical resend the ladder makes;
                # repair the history first and let the model take a turn.
                if step.is_error_step and step.action is None:
                    if await self._repair_rejected_request(step, context, execution_state):
                        continue

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
                        await self._record_recovery_halt(
                            execution_state,
                            stop_reason="recovery_exhausted",
                            description=str(step.error or "recovery exhausted")[:800],
                        )
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

    def _reconcile_context_usage(self, usage, state: "ExecutionState") -> None:
        """Delegates to _context_coord.reconcile_context_usage — see that module."""
        from .context_coordinator import ContextCoordinator

        collab = getattr(self, "_context_coord", None) or ContextCoordinator(self)
        return collab.reconcile_context_usage(usage, state)

    async def _compress_for_recovery(self, context: RunContext, overflow: bool = False) -> bool:
        """Delegates to _context_coord.compress_for_recovery — see that module."""
        from .context_coordinator import ContextCoordinator

        collab = getattr(self, "_context_coord", None) or ContextCoordinator(self)
        return await collab.compress_for_recovery(context, overflow)

    async def _maybe_compress(self, context: RunContext, phase: str, state: "ExecutionState" = None) -> None:
        """Delegates to _context_coord.maybe_compress — see that module."""
        from .context_coordinator import ContextCoordinator

        collab = getattr(self, "_context_coord", None) or ContextCoordinator(self)
        return await collab.maybe_compress(context, phase, state)

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
            state.halt_description = stop_condition.get_description()
            event = EventFactory.stop_condition(
                state.current_step,
                stop_condition.get_stop_reason().value,
                stop_condition.get_description(),
                failure=failure,
                partial_results=_extract_partial_results(state.steps),
            )
            await self.event_bus.publish(event)
            return True
        return False

    async def _steer_report_after_approved_action_failure(self, *, context: RunContext, state: "ExecutionState", tool_name: str, tool_call_id: str, observation: str, failure_payload: Dict[str, Any], raised_by_path: Optional[List[str]] = None) -> None:
        """Delegates to _approval_flow.steer_report_after_approved_action_failure — see that module."""
        from .approval_flow import ApprovalResumeFlow

        collab = getattr(self, "_approval_flow", None) or ApprovalResumeFlow(self)
        return await collab.steer_report_after_approved_action_failure(context=context, state=state, tool_name=tool_name, tool_call_id=tool_call_id, observation=observation, failure_payload=failure_payload, raised_by_path=raised_by_path)

    async def _execute_pending_approved_action(self, context: RunContext, state: "ExecutionState") -> None:
        """Delegates to _approval_flow.execute_pending_approved_action — see that module."""
        from .approval_flow import ApprovalResumeFlow

        collab = getattr(self, "_approval_flow", None) or ApprovalResumeFlow(self)
        return await collab.execute_pending_approved_action(context, state)

    def _apply_resume_command(self, context: RunContext) -> Optional[Any]:
        """Delegates to _approval_flow.apply_resume_command — see that module."""
        from .approval_flow import ApprovalResumeFlow

        collab = getattr(self, "_approval_flow", None) or ApprovalResumeFlow(self)
        return collab.apply_resume_command(context)

    async def _acknowledge_rejected_approval(self, resume: Optional[Any], context: RunContext, state: "ExecutionState") -> bool:
        """Delegates to _approval_flow.acknowledge_rejected_approval — see that module."""
        from .approval_flow import ApprovalResumeFlow

        collab = getattr(self, "_approval_flow", None) or ApprovalResumeFlow(self)
        return await collab.acknowledge_rejected_approval(resume, context, state)

    async def _record_interrupt(self, context: RunContext, state: "ExecutionState", *, kind: str, payload: Dict[str, Any], tool_call_id: Optional[str] = None, raised_by_path: Optional[List[str]] = None) -> PendingInterrupt:
        """Delegates to _recording.record_interrupt — see that module."""
        from .recording import OutcomeRecording

        collab = getattr(self, "_recording", None) or OutcomeRecording(self)
        return await collab.record_interrupt(context, state, kind=kind, payload=payload, tool_call_id=tool_call_id, raised_by_path=raised_by_path)

    async def _handle_tool_approval_marker_result(self, context: RunContext, state: "ExecutionState", result: Any, *, parent_tool_call_id: Optional[str]) -> bool:
        """Delegates to _approval_flow.handle_tool_approval_marker_result — see that module."""
        from .approval_flow import ApprovalResumeFlow

        collab = getattr(self, "_approval_flow", None) or ApprovalResumeFlow(self)
        return await collab.handle_tool_approval_marker_result(context, state, result, parent_tool_call_id=parent_tool_call_id)

    async def _record_tool_observation(self, context: RunContext, state: "ExecutionState", *, tool_name: Optional[str], inputs: Optional[Dict[str, Any]], observation: Optional[str], success: bool, tool_call_id: Optional[str] = None, raw_output: Any = None, error: Optional[str] = None, execution_time_ms: Optional[int] = None, produced_by_path: Optional[List[str]] = None, source: str = "react") -> RecordedObservation:
        """Delegates to _recording.record_tool_observation — see that module."""
        from .recording import OutcomeRecording

        collab = getattr(self, "_recording", None) or OutcomeRecording(self)
        return await collab.record_tool_observation(context, state, tool_name=tool_name, inputs=inputs, observation=observation, success=success, tool_call_id=tool_call_id, raw_output=raw_output, error=error, execution_time_ms=execution_time_ms, produced_by_path=produced_by_path, source=source)

    async def _record_provider_executed_calls(self, context: RunContext, state: "ExecutionState", *, calls: Dict[str, Dict[str, Any]], results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Delegates to _recording.record_provider_executed_calls — see that module."""
        from .recording import OutcomeRecording

        collab = getattr(self, "_recording", None) or OutcomeRecording(self)
        return await collab.record_provider_executed_calls(context, state, calls=calls, results=results)

    async def _execute_reasoning_step_native(self, context: RunContext, state: "ExecutionState") -> ReActStep:
        """Delegates to _step_streamer.execute_reasoning_step_native — see that module."""
        from .step_streaming import StepStreamer

        collab = getattr(self, "_step_streamer", None) or StepStreamer(self)
        return await collab.execute_reasoning_step_native(context, state)

    async def _handle_parallel_tool_batch(self, step: ReActStep, context: RunContext, state: "ExecutionState", accumulated_tool_calls: Dict[int, Dict[str, Any]], assistant_content: str) -> None:
        """Delegates to _tool_actions.handle_parallel_tool_batch — see that module."""
        from .tool_actions import ToolActionHandler

        collab = getattr(self, "_tool_actions", None) or ToolActionHandler(self)
        return await collab.handle_parallel_tool_batch(step, context, state, accumulated_tool_calls, assistant_content)

    async def _handle_tool_action(self, step: ReActStep, context: RunContext, state: "ExecutionState", tool_call_id: Optional[str] = None, tool_description: Optional[str] = None):
        """Delegates to _tool_actions.handle_tool_action — see that module."""
        from .tool_actions import ToolActionHandler

        collab = getattr(self, "_tool_actions", None) or ToolActionHandler(self)
        return await collab.handle_tool_action(step, context, state, tool_call_id, tool_description)

    async def _execute_tool(self, step: ReActStep, context: RunContext, state: "ExecutionState" = None):
        """Delegates to _tool_actions.execute_tool — see that module."""
        from .tool_actions import ToolActionHandler

        collab = getattr(self, "_tool_actions", None) or ToolActionHandler(self)
        return await collab.execute_tool(step, context, state)

    def _resolve_media_refs(
        self,
        inputs: dict,
        state: "ExecutionState",
        tool_name: str | None = None,
    ) -> dict:
        """Resolve symbolic media references in tool inputs to stored URLs.

        The rewriting logic lives in ``react.media_refs``; this wrapper
        supplies the run's media store and the passthrough set, which comes
        from the tool's own schema (``ParameterSchema.media_ref_passthrough``)
        — a table of application tool names used to live here, and an
        application's tool names have no business inside the framework.
        """
        from .media_refs import declared_passthrough_params, resolve_media_refs

        try:
            schema = self.tool_executor._get_tool_schema_obj(tool_name)
        except Exception:  # noqa: BLE001 — resolution must not fail the call
            schema = None
        passthrough = (
            declared_passthrough_params(schema) if schema is not None else set()
        )

        return resolve_media_refs(
            inputs, state.media_store, passthrough_params=passthrough
        )

    # Bad media blocks a single run may strip before it stops trying. Each
    # rejection costs one error step, and ErrorThresholdCondition halts on
    # three consecutive, so this is a documentation of intent more than a
    # binding limit — but it must exist so a repair that never helps cannot
    # loop for the whole step budget.
    MAX_MEDIA_REPAIRS = 3

    async def _repair_rejected_request(
        self, step: ReActStep, context: RunContext, state: "ExecutionState"
    ) -> bool:
        """Edit the history after a provider rejected the request itself.

        The streaming step turns every provider exception into an error step
        (`_handle_step_error`), so a 400 about the request's CONTENT lands
        here rather than in the ``except`` around the step call. Two classes
        are repairable and both are deterministic — retrying without editing
        the history just repeats the 400 (2026-08-18: four identical
        rejections of one xlsx-as-image block, then a halt):

        * structural — tool_use/tool_result pairing (once per run);
        * unprocessable media — an image/document block the provider cannot
          decode (bounded by MAX_MEDIA_REPAIRS).

        Returns True when the history changed and the step should be resent.
        The error step stays on ``state.steps`` so the consecutive-error
        threshold still bounds a repair that does not help.
        """
        from .message_repair import (
            is_structural_message_error,
            is_unsupported_media_error,
            repair_tool_pairing,
            strip_unprocessable_media,
        )

        error_text = step.error or ""

        if is_unsupported_media_error(error_text):
            if state.media_repairs >= self.MAX_MEDIA_REPAIRS:
                logger.warning(
                    "[ORCH] provider rejected media again after %d repairs; "
                    "leaving it to the recovery ladder: %s",
                    state.media_repairs,
                    _preview(error_text, 200),
                )
                return False
            repaired, anomalies = strip_unprocessable_media(context.messages)
            if not anomalies:
                logger.warning(
                    "[ORCH] provider rejected a media block but the history "
                    "holds none to strip: %s",
                    _preview(error_text, 200),
                )
                return False
            state.media_repairs += 1
            logger.warning(
                "[ORCH] provider rejected a media block (%s); stripped and "
                "resending (repair %d/%d): %s",
                _preview(error_text, 200),
                state.media_repairs,
                self.MAX_MEDIA_REPAIRS,
                "; ".join(anomalies),
            )
            context.messages = repaired
            return True

        if is_structural_message_error(error_text) and not state.structural_repair_attempted:
            state.structural_repair_attempted = True
            repaired, anomalies = repair_tool_pairing(context.messages)
            if not anomalies:
                logger.warning(
                    "[ORCH] provider rejected message structure but repair "
                    "found nothing to fix: %s",
                    _preview(error_text, 200),
                )
                return False
            logger.warning(
                "[ORCH] provider rejected message structure (%s); repaired and "
                "resending: %s",
                _preview(error_text, 200),
                "; ".join(anomalies),
            )
            context.messages = repaired
            return True

        return False

    def _handle_step_error(self, step: ReActStep, error: Exception, state: "ExecutionState"):
        """Delegates to _tool_actions.handle_step_error — see that module."""
        from .tool_actions import ToolActionHandler

        collab = getattr(self, "_tool_actions", None) or ToolActionHandler(self)
        return collab.handle_step_error(step, error, state)

    async def _record_recovery_halt(
        self, state: "ExecutionState", *, stop_reason: str, description: str
    ) -> None:
        """Make a recovery-ladder halt look like every other halt.

        Safety-condition halts stamp `failure_metadata` and publish a
        STOP_CONDITION event, which is what the dispatch envelope, the root
        agent span's status and the halt wrap-up all read. A halt because the
        recovery manager gave up (fatal provider error, overflow that will not
        compact) used to just `break` — no failure metadata, no event — so a
        run killed by an out-of-credit account reported nothing more specific
        than the canned "repeated issues" answer, and its span read as a
        success in the trace list.
        """
        failure = _extract_failure_metadata(
            state.steps, stop_reason=stop_reason, description=description
        )
        state.failure_metadata = failure
        state.halt_description = description
        try:
            await self.event_bus.publish(
                EventFactory.stop_condition(
                    state.current_step,
                    stop_reason,
                    description,
                    failure=failure,
                    partial_results=_extract_partial_results(state.steps),
                )
            )
        except Exception:  # noqa: BLE001 — never let telemetry fail the halt
            logger.exception("failed to publish recovery-halt stop condition")

    async def _publish_final_answer_event(self, step: ReActStep, state: "ExecutionState"):
        """Delegates to _answers.publish_final_answer_event — see that module."""
        from .answer_synthesis import AnswerSynthesis

        collab = getattr(self, "_answers", None) or AnswerSynthesis(self)
        return await collab.publish_final_answer_event(step, state)

    async def _build_result(self, state: "ExecutionState", context: RunContext = None) -> ReActResult:
        """Delegates to _answers.build_result — see that module."""
        from .answer_synthesis import AnswerSynthesis

        collab = getattr(self, "_answers", None) or AnswerSynthesis(self)
        return await collab.build_result(state, context)

    def _build_error_result(self, state: "ExecutionState", error: Exception) -> ReActResult:
        """Delegates to _answers.build_error_result — see that module."""
        from .answer_synthesis import AnswerSynthesis

        collab = getattr(self, "_answers", None) or AnswerSynthesis(self)
        return collab.build_error_result(state, error)

    async def _answer_after_halt(self, context: RunContext, state: "ExecutionState") -> bool:
        """Delegates to _answers.answer_after_halt — see that module."""
        from .answer_synthesis import AnswerSynthesis

        collab = getattr(self, "_answers", None) or AnswerSynthesis(self)
        return await collab.answer_after_halt(context, state)

    def _generate_fallback_answer(self, steps) -> str:
        """Delegates to _answers.generate_fallback_answer — see that module."""
        from .answer_synthesis import AnswerSynthesis

        collab = getattr(self, "_answers", None) or AnswerSynthesis(self)
        return collab.generate_fallback_answer(steps)

    def _find_similar_tool(self, requested_name: str) -> Optional[str]:
        """Fuzzy-match a hallucinated tool name (see ``react.fuzzy_tools``)."""
        from .fuzzy_tools import find_similar_tool

        return find_similar_tool(requested_name, self.tool_executor.list_tools())

    def _is_similar_enough(self, s1: str, s2: str, threshold: int = 2) -> bool:
        """Delegates to ``react.fuzzy_tools.is_similar_enough``."""
        from .fuzzy_tools import is_similar_enough

        return is_similar_enough(s1, s2, threshold)

    def get_current_status(self) -> Dict[str, Any]:
        """Get current execution status."""
        return {"agent_type": "react_orchestrator"}
