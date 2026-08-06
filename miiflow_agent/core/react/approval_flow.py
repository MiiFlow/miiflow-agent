"""Deterministic approval/resume flow.

When a run pauses for tool approval, the approved action is re-executed by
CONTROL FLOW on resume — never by asking the model to re-emit the call
(durable-execution design; the model is only asked to report the outcome).
This collaborator owns that spine: applying the resume command, replaying or
acknowledging the user's decision, translating approval-marker tool results
into pauses, and steering the report turn after an approved action fails.
Methods were moved verbatim from ReActOrchestrator (which keeps thin
delegates); ``self._orch`` is the orchestrator.
"""

from __future__ import annotations

import inspect
import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..message import Message, MessageRole
from .enums import ReActEventType
from .events import EventFactory
from .react_events import ReActEvent
from .orchestrator import (
    _approved_action_failure_guidance,
    _approved_action_failure_payload,
    _approved_action_success_guidance,
    _consume_approval_checkpoint_state,
    _recover_approval_action_route,
    _resolve_stale_dispatch_placeholder,
    _sanitize_error_message,
)

if TYPE_CHECKING:
    from ..agent import RunContext
    from .execution import ExecutionState
    from .orchestrator import ReActOrchestrator

logger = logging.getLogger(__name__)


class ApprovalResumeFlow:
    """The approval pause/resume spine of the loop."""

    def __init__(self, orch: "ReActOrchestrator"):
        self._orch = orch

    def apply_resume_command(self, context: RunContext) -> Optional[Any]:
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

    async def acknowledge_rejected_approval(
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
        await self._orch.event_bus.publish(
            EventFactory.final_answer(state.current_step, message)
        )
        return True

    async def handle_tool_approval_marker_result(
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

        interrupt = await self._orch._record_interrupt(
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
        await self._orch.event_bus.publish(
            ReActEvent(
                event_type=ReActEventType.TOOL_APPROVAL_NEEDED,
                step_number=state.current_step,
                data=state.clarification_data,
            )
        )
        return True

    async def steer_report_after_approved_action_failure(
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

    async def execute_pending_approved_action(
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
        registry = getattr(self._orch.tool_executor, "_tool_registry", None)
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
                    _recorded = await self._orch._record_tool_observation(
                        context,
                        state,
                        tool_name=tool_name,
                        inputs=inputs,
                        observation=observation,
                        success=success,
                        tool_call_id=tool_call_id,
                        raw_output=raw_output,
                        error=str(raw_error) if raw_error else None,
                        produced_by_path=raised_by_path or ["root"],
                        source="resume",
                    )
                    child_observation_ref = _recorded.ref
                    observation = _recorded.observation
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
                        await self._orch.event_bus.publish(
                            EventFactory.observation(
                                state.current_step,
                                observation,
                                tool_name,
                                success,
                                tool_call_id=tool_call_id,
                                observation_ref=child_observation_ref,
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
                        await self._orch._steer_report_after_approved_action_failure(
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
                    await self._orch.event_bus.publish(
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
            result = await self._orch.tool_executor.execute_tool(
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
        _recorded = await self._orch._record_tool_observation(
            context,
            state,
            tool_name=tool_name,
            inputs=inputs,
            observation=observation,
            success=success,
            tool_call_id=tool_call_id,
            raw_output=raw_output,
            error=str(raw_error) if raw_error else None,
            source="resume",
        )
        approved_observation_ref = _recorded.ref
        observation = _recorded.observation

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
            await self._orch.event_bus.publish(
                EventFactory.observation(
                    state.current_step,
                    observation,
                    tool_name,
                    success,
                    tool_call_id=tool_call_id,
                    observation_ref=approved_observation_ref,
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
            await self._orch._steer_report_after_approved_action_failure(
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
