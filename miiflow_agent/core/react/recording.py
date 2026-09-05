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

import json
import logging
import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

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


def _call_arguments(raw: Any) -> Dict[str, Any]:
    """The arguments of a provider-executed call, whatever shape it arrived in.

    Anthropic hands back a parsed dict; OpenAI's Responses API hands back the
    JSON *string* it sent (`mcp_call.arguments`). Keeping only the dict left
    every native-MCP call on the OpenAI path recorded with `input: {}` — which
    is how a run that shipped a bad argument to a connected MCP server showed a
    validation failure in the server's logs and an empty, blameless call in our
    own timeline. Record what was actually sent.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def short_tool_name(tool_name: str) -> str:
    """`meta_ads_insights` -> `ads_insights`; short names pass through.

    THE citation-label shortening rule. Three places depend on it agreeing
    exactly: this module mints labels for the ReAct path, the server's
    `tool_config_converter._generate_reference_label` mints them for its own
    tool path, and the server's `citation_processor` reconstructs a label the
    model invented by applying the same rule to the tool name it wrote. If a
    copy of this drifts, citations stop resolving silently — the marker simply
    ships to the reader as raw `[ref:...]` text. Hence one function, imported.
    """
    parts = tool_name.split("_")
    return "_".join(parts[-2:]) if len(parts) > 2 else tool_name


class OutcomeRecording:
    """The durable-state seams for finalized tool outcomes."""

    def __init__(self, orch: "ReActOrchestrator"):
        self._orch = orch

    #: An observation that already opens with a label (a resumed run, a tool
    #: that injects its own) must not be labelled twice.
    _REF_PREFIX_RE = re.compile(r"^\[ref:[\w]+\]")

    #: Artifact receipts, not data sources. A `render_*` call returns a
    #: `[VIZ:id]` marker that the model places in its answer and the host
    #: swaps for the card; citing it would mean citing the chart rather than
    #: the numbers behind it. It is also unsafe: the host's
    #: `is_visualization_result` matches a bare marker with an ANCHORED
    #: `^\[VIZ:...\]$`, so a `[ref:...]` prefix in front of one would stop
    #: the visualization from being recognised at all.
    _ARTIFACT_MARKER_RE = re.compile(r"^\[(?:VIZ|SA|MEDIA):")

    #: Kept as a method for the call sites already using it; the rule itself
    #: lives at module level so every minter and the resolver share one copy.
    _short_tool_name = staticmethod(short_tool_name)

    @classmethod
    def _is_artifact_observation(cls, observation: Optional[str]) -> bool:
        """Whether this observation is a render receipt rather than a source."""
        return bool(observation) and bool(cls._ARTIFACT_MARKER_RE.match(observation))

    def _mint_reference_label(self, state: "ExecutionState", tool_name: str) -> str:
        """Allocate this run's next citation label for ``tool_name``.

        ``state`` is a duck-typed seam (resume paths and tests pass stand-ins),
        so the counter store is created on demand rather than assumed — the
        alternative is numbering that silently restarts at 1 mid-run and hands
        two different calls the same citation key.
        """
        counters = getattr(state, "reference_label_counters", None)
        if counters is None:
            counters = {}
            try:
                state.reference_label_counters = counters
            except AttributeError:  # frozen/slotted stand-in — label anyway
                pass
        short_name = self._short_tool_name(tool_name)
        counters[short_name] = counters.get(short_name, 0) + 1
        return f"{short_name}_{counters[short_name]}"

    def _apply_reference_label(
        self, reference_label: Optional[str], observation: str
    ) -> str:
        """Open the observation with the `[ref:LABEL]` tag that cites it.

        The prompt already instructs the model that "tool results open with a
        reference tag like [ref:tool_name_N] ... append that exact tag", and
        `citation_processor` already resolves markers by scanning the timeline
        for a matching `reference_label` — which `streaming_service`
        `_extract_reference_label` reads back off this very prefix. Only the
        tag itself was never emitted on this path, so the model invented
        plausible-looking labels from tool names, nothing resolved them, and
        the raw marker shipped to the reader: "...blending Smart Bidding's
        target [ref:google_ads_query_2]." Minting here — the one seam every
        finalized observation passes through — closes the loop for local and
        provider-executed calls alike.
        """
        if not reference_label:
            return observation
        if observation and self._REF_PREFIX_RE.match(observation):
            return observation
        if not observation:
            return f"[ref:{reference_label}]"
        return f"[ref:{reference_label}]\n{observation}"

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

        # Minted before the sink write so the stored row, the string the model
        # reads, and the citation the model writes back all name the same call.
        # Artifact receipts are skipped outright — no label, and no number
        # consumed, so citation keys stay dense over the calls that produced
        # actual data.
        reference_label = (
            None
            if self._is_artifact_observation(observation)
            else self._mint_reference_label(state, tool_name)
        )

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
                        reference_label=reference_label,
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
        # Prefix AFTER bounding so the tag survives truncation — it is the
        # only thing tying the model's citation to this call.
        return RecordedObservation(
            ref=ref,
            observation=self._apply_reference_label(reference_label, bounded),
        )

    async def publish_provider_call_planned(
        self,
        state: "ExecutionState",
        *,
        call: Dict[str, Any],
        call_id: str,
    ) -> None:
        """Announce a call the provider is running server-side.

        These never pass through the local dispatcher, so nothing else on the
        run would record them — without this the timeline shows an answer that
        cites data no visible tool call produced.

        Published the moment the ``mcp_tool_use`` block FINALIZES (its
        `content_block_stop`, where the arguments have been parsed), not when
        it starts: the normalizer never parses MCP arguments mid-block, so the
        start-event placeholder still carries ``{}`` and a row built from it
        shows an argument-less, blameless call.

        ACTION_PLANNED must precede the call's OBSERVATION even though the work
        is already finished by the time we see either. Consumers build their
        tool row on the action event and only update it on the observation
        (streaming_service's execution_timeline, both frontend chunk reducers),
        so an observation-only call is invisible: no reasoning-panel row live,
        an empty execution_timeline on reload, and `has_tool_events` left False
        so the turn is mislabeled `single_hop`. The event pair is the contract —
        keep it whole here rather than teaching every consumer about MCP.
        """
        fn = call.get("function", {}) or {}
        await self._orch.event_bus.publish(
            EventFactory.action_planned(
                state.current_step,
                fn.get("name"),
                _call_arguments(fn.get("arguments")),
                tool_call_id=call_id,
                executor="native_mcp",
                server_name=call.get("server_name"),
            )
        )

    async def record_provider_call_observation(
        self,
        context: RunContext,
        state: "ExecutionState",
        *,
        call: Dict[str, Any],
        call_id: str,
        result: Optional[Dict[str, Any]],
        execution_time_ms: Optional[int] = None,
    ) -> None:
        """Close one provider-executed call: store it, then publish OBSERVATION.

        ``result`` is None when the provider never reported one (turn cut
        short, or a paused turn). Record it as a failure rather than dropping
        it — a silent gap in the trail is worse than a visible incomplete call.
        """
        fn = call.get("function", {}) or {}
        arguments = _call_arguments(fn.get("arguments"))

        has_result = result is not None
        is_error = bool((result or {}).get("is_error", False)) or not has_result
        observation = (result or {}).get("content") or (
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
            execution_time_ms=execution_time_ms,
        )
        observation = recorded.observation
        # These results are replayed to the provider verbatim by
        # `_convert_message` (as mcp_tool_result blocks), so the bound has
        # to land on the payload itself — not just on the event.
        #
        # The citation key is stripped back off for the replay copy.
        # Labels are numbered per RUN, so a stale one riding into a later
        # turn would collide with that turn's freshly minted key of the
        # same name — two different results wearing one tag, in front of
        # the model and in the resolver. The label's job is done once this
        # turn's event carries it.
        if has_result and isinstance(result, dict):
            result["content"] = self._REF_PREFIX_RE.sub(
                "", observation, count=1
            ).lstrip("\n")

        await self._orch.event_bus.publish(
            EventFactory.observation(
                state.current_step,
                observation,
                fn.get("name"),
                not is_error,
                tool_call_id=call_id,
                observation_ref=recorded.ref,
                executor="native_mcp",
                server_name=call.get("server_name"),
            )
        )

    async def record_provider_executed_calls(
        self,
        context: RunContext,
        state: "ExecutionState",
        *,
        calls: Dict[str, Dict[str, Any]],
        results: List[Dict[str, Any]],
        already_planned: Optional[Set[str]] = None,
        already_recorded: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """Sweep whatever the streaming pass did not already record, and return
        the metadata to attach to the assistant message.

        The pair for a provider-executed call is normally published DURING the
        stream, at the block that carries it — see
        `step_streaming._handle_provider_executed_call`. Anything left here is
        a call the provider never answered: the turn was cut short mid-block,
        or paused. Recording it as a visible failure beats a silent gap.

        This ran as the only recording path until it was found to invert the
        wire order: a turn whose tools all ran inside the provider streamed its
        whole answer first and published every tool event afterwards, so the
        reasoning panel materialized above an already-finished answer and the
        late ACTION_PLANNED made `streaming_service` discard (then re-emit) the
        real answer. Kept as the sweep, never as the main path.

        The pair carries `executor="native_mcp"` and the `server_name` because a
        consumer that only RENDERS the call can ignore who ran it, but one that
        REPLAYS it into a later request cannot: a provider-executed call has to
        go back as an mcp_tool_use/mcp_tool_result pair. Dropping that bit here
        is what let a host persist the call and replay it as a local `tool_use`,
        so the model reissued GitHub's `search_code` client-side on the next
        turn and got "Tool 'search_code' not found" from a registry that never
        holds native-MCP tools.

        Returns the metadata to attach to the assistant message so
        `_convert_message` can replay the mcp_tool_use/mcp_tool_result pairs.
        """
        planned = already_planned if already_planned is not None else frozenset()
        recorded = already_recorded if already_recorded is not None else frozenset()
        results_by_id = {r.get("tool_use_id"): r for r in results if r.get("tool_use_id")}

        for call_id, call in calls.items():
            if call_id in recorded:
                continue
            if call_id not in planned:
                await self.publish_provider_call_planned(
                    state, call=call, call_id=call_id
                )
            await self.record_provider_call_observation(
                context,
                state,
                call=call,
                call_id=call_id,
                result=results_by_id.get(call_id),
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
