"""Durable recording of tool outcomes: observation store, ledger, interrupts.

The single seams through which a finalized tool execution becomes durable
state: the adapter ObservationSink write (awaited inline — see
core/observation.py for why never fire-and-forget), the checkpoint ledger
reduce, provider-executed (native MCP) call registration, and interrupt
recording with the parallel-batch demotion queue. Methods were moved
verbatim from ReActOrchestrator (which keeps thin delegates); ``self._orch``
is the orchestrator.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..checkpoint import DispatchLedgerEntry, PendingInterrupt, make_ledger_digest
from ..interrupt import mint_interrupt_id
from ..observation import (
    ObservationRecord,
    RecordedObservation,
    bound_observation_for_llm,
    get_observation_sink,
)
from .events import EventFactory

if TYPE_CHECKING:
    from ..agent import RunContext
    from .execution import ExecutionState
    from .orchestrator import ReActOrchestrator

logger = logging.getLogger(__name__)


class OutcomeRecording:
    """The durable-state seams for finalized tool outcomes."""

    def __init__(self, orch: "ReActOrchestrator"):
        self._orch = orch

    async def record_tool_observation(
        self,
        context: RunContext,
        state: "ExecutionState",
        *,
        tool_name: Optional[str],
        inputs: Optional[Dict[str, Any]],
        observation: Optional[str],
        success: bool,
        tool_call_id: Optional[str] = None,
        raw_output: Any = None,
        error: Optional[str] = None,
        execution_time_ms: Optional[int] = None,
        produced_by_path: Optional[List[str]] = None,
        source: str = "react",
    ) -> RecordedObservation:
        """Single seam for a finalized tool call: persist the canonical
        observation via the adapter sink (awaited inline — see
        core/observation.py for why never fire-and-forget), then reduce it
        into the checkpoint ledger.

        Returns the ref (None when no sink is wired / the write failed) paired
        with the BOUNDED observation string. Callers must use the returned
        ``.observation`` for anything the model will see — the raw argument may
        be arbitrarily large (a 2.35 MB GAQL dump took a production run past
        the 1M-token ceiling), and the ref needed to write the "read the rest"
        marker only exists here.
        """
        if not tool_name:
            # Nothing to store, but still bound: this must not be an escape
            # hatch that returns an arbitrarily large string to a caller that
            # is about to put it in front of the model.
            return RecordedObservation(
                ref=None,
                observation=bound_observation_for_llm(
                    get_observation_sink(context),
                    observation,
                    tool_name=tool_name,
                    ref=None,
                ),
            )

        # Default producer address: the per-run stamp the adapter set
        # (["root"] for the root run, ["child", <thread_id>] for a
        # dispatched child) — lets a session continuation exclude the
        # child's own prior entries from its worklog. Explicit
        # produced_by_path (resume paths) wins.
        if not produced_by_path:
            run_deps = getattr(context, "deps", None)
            if isinstance(run_deps, dict):
                produced_by_path = run_deps.get("ledger_producer_path")

        ref: Optional[str] = None
        sink = get_observation_sink(context)
        if sink is not None:
            try:
                ref = await sink.record(
                    ObservationRecord(
                        tool_name=tool_name,
                        tool_call_id=tool_call_id,
                        inputs=inputs or {},
                        observation_text=observation or "",
                        raw_output=raw_output,
                        success=bool(success),
                        error=error,
                        execution_time_ms=execution_time_ms,
                        step_number=state.current_step,
                        produced_by_path=list(produced_by_path or ["root"]),
                        source=source,
                    )
                )
            except Exception as sink_err:  # noqa: BLE001 — never fail the run
                logger.debug("Observation sink record failed: %s", sink_err)
                ref = None

        # Ledger writes target the ROOT thread's checkpoint when this run is a
        # dispatched child (deps["root_checkpoint"], shared by reference
        # in-process) — the root checkpoint is the single durable blackboard
        # for the whole dispatch tree. Falls back to this run's own checkpoint.
        deps = getattr(context, "deps", None)
        checkpoint = None
        if isinstance(deps, dict):
            checkpoint = deps.get("root_checkpoint")
        if checkpoint is None:
            checkpoint = getattr(context, "checkpoint", None)
        if checkpoint is not None and hasattr(checkpoint, "merge_ledger"):
            # Identity MUST match the dedupe gate's read-side computation
            # (same exclusions + scope dims), or the gate never hits.
            from .dedupe import dedupe_identity

            schema = None
            try:
                schema = self._orch.tool_executor._get_tool_schema_obj(tool_name)
            except Exception:  # noqa: BLE001
                schema = None
            inputs_hash = dedupe_identity(
                tool_name,
                inputs,
                scope_dims=getattr(schema, "dedupe_scope_dims", None),
                deps=deps if isinstance(deps, dict) else None,
            )
            checkpoint.merge_ledger(
                [
                    DispatchLedgerEntry(
                        kind="tool_call",
                        success=bool(success),
                        digest=make_ledger_digest(observation or ""),
                        observation_ref=ref,
                        produced_at=time.time(),
                        turn_index=getattr(checkpoint, "turn_index", 0),
                        tool_name=tool_name,
                        inputs_hash=inputs_hash,
                        produced_by_path=list(produced_by_path or ["root"]),
                    )
                ]
            )

        # Bound AFTER the store write and the ledger reduce: the row keeps the
        # fullest copy the adapter is willing to retain (that is what the ref
        # serves), and the digest summarizes the real output. Only the string
        # travelling into the next request is clamped.
        bounded = bound_observation_for_llm(
            sink, observation, tool_name=tool_name, ref=ref
        )
        if observation and len(bounded) != len(observation):
            logger.warning(
                "[ORCH] step=%d observation for '%s' bounded for context: "
                "%d -> %d chars (ref=%s)",
                state.current_step,
                tool_name,
                len(observation),
                len(bounded),
                ref,
            )
        return RecordedObservation(ref=ref, observation=bounded)

    async def record_provider_executed_calls(
        self,
        context: RunContext,
        state: "ExecutionState",
        *,
        calls: Dict[str, Dict[str, Any]],
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Write the action+observation pair for tools the provider ran itself.

        These never pass through the local dispatcher, so nothing else on the
        run would record them — without this the timeline shows an answer that
        cites data no visible tool call produced.

        Each call emits ACTION_PLANNED *before* its OBSERVATION even though the
        work is already finished. Consumers build their tool row on the action
        event and only update it on the observation (streaming_service's
        execution_timeline, both frontend chunk reducers), so an
        observation-only call is invisible: no reasoning-panel row live, an
        empty execution_timeline on reload, and `has_tool_events` left False so
        the turn is mislabeled `single_hop`. The event pair is the contract —
        keep it whole here rather than teaching every consumer about MCP.

        Returns the metadata to attach to the assistant message so
        `_convert_message` can replay the mcp_tool_use/mcp_tool_result pairs.
        """
        results_by_id = {r.get("tool_use_id"): r for r in results if r.get("tool_use_id")}

        for call_id, call in calls.items():
            fn = call.get("function", {}) or {}
            arguments = fn.get("arguments") if isinstance(fn.get("arguments"), dict) else {}
            result = results_by_id.get(call_id) or {}

            await self._orch.event_bus.publish(
                EventFactory.action_planned(
                    state.current_step,
                    fn.get("name"),
                    arguments,
                    tool_call_id=call_id,
                )
            )
            # A call with no matching result means the provider never reported
            # one (turn cut short, or a paused turn). Record it as a failure
            # rather than dropping it — a silent gap in the trail is worse than
            # a visible incomplete call.
            has_result = call_id in results_by_id
            is_error = bool(result.get("is_error", False)) or not has_result
            observation = result.get("content") or (
                "" if has_result else "No result returned by the provider for this call."
            )

            recorded = await self._orch._record_tool_observation(
                context,
                state,
                tool_name=fn.get("name"),
                inputs=arguments,
                observation=observation,
                success=not is_error,
                tool_call_id=call_id,
                error=observation if is_error else None,
            )
            observation = recorded.observation
            # These results are replayed to the provider verbatim by
            # `_convert_message` (as mcp_tool_result blocks), so the bound has
            # to land on the payload itself — not just on the event.
            if has_result and isinstance(result, dict):
                result["content"] = observation

            await self._orch.event_bus.publish(
                EventFactory.observation(
                    state.current_step,
                    observation,
                    fn.get("name"),
                    not is_error,
                    tool_call_id=call_id,
                    observation_ref=recorded.ref,
                )
            )

        return {"mcp_tool_results": results} if results else {}

    async def record_interrupt(
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
        await self._orch.event_bus.publish(
            EventFactory.interrupt_requested(state.current_step, interrupt)
        )
        return interrupt
