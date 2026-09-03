"""Dispatch lifecycle — framework-side guardrails and event bubbling
for sub-agent dispatches.

This module owns everything the orchestrator does *around* a sub-agent's
own `stream()`:

  - **Guardrails**: depth cap, cycle detection, per-handle budget,
    per-turn global budget. Pure functions plus an asyncio-lock-protected
    counter so the budget is correct under parallel dispatch (the lossy
    JSONField counter at thread.metadata["dispatch_counts"] races when
    two dispatches fire in the same gather).
  - **Event bubbling**: forward the child's FINAL_ANSWER_CHUNK events as
    SUBAGENT_DISPATCH/progress on the parent's bus, and re-publish the
    child's SUBAGENT_DISPATCH events with our subagent_id prepended to
    the path so depth-2+ dispatches nest correctly in the UI.
  - **Lifecycle**: emit start/complete/failed events on the parent's bus
    around the child's stream.

The Django side (Stage 3) drops its duplicated copy of this logic and
calls `dispatch_subagent()` from the orchestrator. Until then, both code
paths can coexist because they emit identical SUBAGENT_DISPATCH event
shapes.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
)

from ..tools.schemas import ToolFailure
from .enums import ReActEventType
from .events.bus import EventFactory
from .react_events import ReActEvent

if TYPE_CHECKING:
    from ..subagent import SubAgent, SubAgentHandoff, SubAgentResult
    from ..tools.function.function_tool import FunctionTool
    from .events.bus import EventBus


logger = logging.getLogger(__name__)


# Guardrail defaults. These match the values the Django-side
# `dispatch_assistant_tool` ships with so the alias path keeps working
# while Stage 3 migrates.
MAX_NESTING_DEPTH = 3
DEFAULT_MAX_CALLS_PER_HANDLE = 3
DEFAULT_MAX_DISPATCHES_PER_TURN = 30


class DispatchGuardrailError(Exception):
    """Raised when a dispatch would violate a guardrail.

    Carries `kind` so callers can render a structured tool observation
    back to the LLM. The orchestrator catches this, packages the error
    as a tool_result, and lets the model retry with a different handle
    or give up.
    """

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


@dataclass
class DispatchCounter:
    """In-memory, asyncio-Lock-protected counter for a single parent turn.

    Replaces the lossy JSONField counter at
    `thread.metadata["dispatch_counts"]`. Under parallel dispatch
    (N gather'd children launched from one parent step), two dispatches
    can read-then-write the same `counts[handle]` value and lose one of
    the increments. This counter holds an `asyncio.Lock` around the
    read-modify-write so the check-and-increment is atomic.

    Instantiated once per parent turn by the orchestrator. The Django
    adapter can serialize the final counts back to thread.metadata at
    turn end for cross-turn diagnostic purposes if needed; the framework
    itself does not persist.
    """

    max_per_handle: int = DEFAULT_MAX_CALLS_PER_HANDLE
    max_total: int = DEFAULT_MAX_DISPATCHES_PER_TURN

    counts: Dict[str, int] = field(default_factory=dict)
    total: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    async def reserve(
        self, handle: str, *, per_handle_limit: Optional[int] = None
    ) -> None:
        """Reserve one dispatch slot for `handle`, raising if over budget.

        ``per_handle_limit`` lets the adapter pass a TIGHTER per-edge cap
        (AssistantHandoff.max_calls_per_turn); the effective limit is the
        min of it and the counter's global per-handle cap, checked and
        incremented under the same lock — dispatch_assistant is
        parallelizable, so any check outside this lock is racy.

        On success, the per-handle and total counters are both
        incremented before returning. On failure, the counters are
        unchanged. Either way, the lock is released before raising.
        """
        effective_limit = self.max_per_handle
        if per_handle_limit is not None:
            effective_limit = min(effective_limit, int(per_handle_limit))
        async with self._lock:
            per_handle = self.counts.get(handle, 0)
            if per_handle >= effective_limit:
                raise DispatchGuardrailError(
                    "per_handle_budget_exceeded",
                    (
                        f"Already dispatched to '{handle}' "
                        f"{per_handle} times this turn "
                        f"(limit: {effective_limit})."
                    ),
                )
            if self.total >= self.max_total:
                raise DispatchGuardrailError(
                    "max_dispatches_exceeded",
                    (
                        f"Already dispatched {self.total} times this "
                        f"turn (global limit: {self.max_total})."
                    ),
                )
            self.counts[handle] = per_handle + 1
            self.total += 1

    def snapshot(self) -> Dict[str, Any]:
        """Return a serializable snapshot of the counter state.

        Safe to call without the lock — readers get a self-consistent
        view because Python dict assignment is atomic. The Django
        adapter persists this to `thread.metadata["dispatch_counts"]`
        at turn end.
        """
        return {
            "counts": dict(self.counts),
            "total": self.total,
            "max_per_handle": self.max_per_handle,
            "max_total": self.max_total,
        }


def enforce_static_guardrails(
    handle: str,
    *,
    child_id: str,
    dispatch_chain: Sequence[str],
    depth: int,
    max_depth: int = MAX_NESTING_DEPTH,
) -> None:
    """Check guardrails that don't need the per-turn counter.

    Specifically: cycle detection (child can't appear on its own
    ancestor chain) and depth cap. These are pure functions of the
    incoming handoff — no shared state.

    Raises `DispatchGuardrailError` on violation. Returns None on
    success. Caller is responsible for the counter check via
    `DispatchCounter.reserve()`.
    """
    if child_id in dispatch_chain:
        raise DispatchGuardrailError(
            "cycle_detected",
            (
                f"Dispatching to '{handle}' would form a cycle — "
                f"the sub-agent is already on the dispatch chain."
            ),
        )
    if depth > max_depth:
        raise DispatchGuardrailError(
            "max_depth_exceeded",
            f"Cannot dispatch deeper than {max_depth} levels (depth={depth}).",
        )


async def forward_subagent_events(
    child_events: AsyncIterator[ReActEvent],
    *,
    parent_event_bus: "EventBus",
    parent_step_number: int,
    subagent_id: str,
    own_path: List[str],
    handle: Optional[str] = None,
    started_at_monotonic: Optional[float] = None,
    surface_interrupts: bool = False,
    promote_to_final: bool = False,
) -> Optional[Dict[str, Any]]:
    """Consume the child's event stream and forward the relevant events
    to the parent's bus.

    Forwarded events (all surface in the parent's SubagentPanel inside
    the child's nestedChunks):

      1. FINAL_ANSWER_CHUNK → subagent_dispatch/progress — the child's
         final answer, streamed token-by-token into ``result``.
      2. THINKING_CHUNK → subagent_dispatch/thinking — the child's
         intermediate reasoning, appended to a nested thinking chunk.
      3. ACTION_PLANNED → subagent_dispatch/tool (status="planned") —
         a tool the child is about to call.
      4. ACTION_EXECUTING → subagent_dispatch/tool (status="executing")
         — same tool flipped to running.
      5. OBSERVATION → subagent_dispatch/observation — tool result;
         marks the matching tool chunk completed on the FE.
      6. SUBAGENT_DISPATCH → re-published as-is, with our subagent_id
         prepended to ``subagent_path`` so depth-2+ dispatches nest
         correctly.

    When ``surface_interrupts`` is True (Phase 2, R1), CLARIFICATION_NEEDED /
    TOOL_APPROVAL_NEEDED events are **captured and forwarded up** the parent
    bus (path-prefixed) and returned to the caller, so a sub-agent that needs
    the user no longer pauses into a void — ``dispatch_subagent`` surfaces the
    captured interrupt to the parent instead of reading an empty
    ``final_result()``. When False, behaviour is unchanged: these events stay
    private to the child (legacy default).

    Other event types stay in the child's run (STEP_START / STEP_COMPLETE /
    VISUALIZATION / MEDIA / ARTIFACT / etc. are private to the child).

    ``promote_to_final`` switches this into TRANSFER mode: the child owns
    the user-facing turn, so its answer is no longer a nested panel's
    ``result`` but *the* answer. In that mode:

      - FINAL_ANSWER_CHUNK is re-published as a real parent
        FINAL_ANSWER_CHUNK, so it lands in the main message stream (and
        therefore in the parent's persisted message body) instead of
        subagent_dispatch/progress.
      - VISUALIZATION / MEDIA / ARTIFACT are forwarded verbatim rather
        than dropped — a transferred child renders its own output, so
        those events have to reach the parent's stream.
      - Everything else (thinking, tool calls, observations) still goes
        out as subagent_dispatch, so the panel keeps showing the child's
        work — just not a duplicate copy of the answer text.

    Nested transfer composes without special handling: a grandchild's
    promoted chunks arrive here as the child's own FINAL_ANSWER_CHUNKs
    and get promoted again.

    This is a *consumer*: it iterates the entire generator before
    returning. The caller is responsible for calling
    `SubAgent.final_result()` afterward. Returns the captured interrupt data
    (path-prefixed) when a surfaced interrupt was seen, else None.

    ``handle`` + ``started_at_monotonic`` are optional and used only for
    the ``[DISPATCH_TIMING] phase=first_chunk`` log line — emitted on
    the first non-empty FINAL_ANSWER_CHUNK so log scraping can tell
    whether parallel dispatches actually started producing output in
    parallel or stacked end-to-start.
    """
    first_chunk_logged = False
    captured_interrupt: Optional[Dict[str, Any]] = None
    # Running accumulation of the promoted answer. EventFactory.final_answer_chunk
    # carries both the delta and the content-so-far; only the delta is consumed
    # downstream today, but we keep `content` honest rather than passing the
    # delta twice.
    promoted_answer = ""
    async for child_event in child_events:
        if not isinstance(child_event, ReActEvent):
            continue

        et = child_event.event_type
        data = child_event.data or {}

        if surface_interrupts and et in (
            ReActEventType.CLARIFICATION_NEEDED,
            ReActEventType.TOOL_APPROVAL_NEEDED,
        ):
            # Capture the child's pause and forward it up (path-prefixed) so it
            # reaches the user instead of being dropped. The child's stream ends
            # right after this (its orchestrator suspended); dispatch_subagent
            # reads the captured interrupt rather than an empty final_result().
            captured_interrupt = {
                "kind": (
                    "tool_approval"
                    if et == ReActEventType.TOOL_APPROVAL_NEEDED
                    else "clarification"
                ),
                "subagent_id": subagent_id,
                "subagent_path": own_path,
                "data": dict(data),
            }
            await parent_event_bus.publish(
                ReActEvent(
                    event_type=et,
                    step_number=parent_step_number,
                    data={**data, "subagent_id": subagent_id, "subagent_path": own_path},
                )
            )
            continue

        if et == ReActEventType.FINAL_ANSWER_CHUNK:
            chunk = data.get("delta") or data.get("chunk") or data.get("content") or ""
            if chunk:
                if not first_chunk_logged and started_at_monotonic is not None:
                    now = time.monotonic()
                    logger.info(
                        "[DISPATCH_TIMING] phase=first_chunk sub=%s handle=%s elapsed_ms=%d t=%.6f",
                        subagent_id,
                        handle or "?",
                        int((now - started_at_monotonic) * 1000),
                        now,
                    )
                    first_chunk_logged = True
                if promote_to_final:
                    # TRANSFER: the child owns the user-facing turn. Emit a real
                    # FINAL_ANSWER_CHUNK so the chunk lands in the main message
                    # stream (and therefore in the parent's persisted message
                    # body) rather than in the nested panel's `result`.
                    promoted_answer += chunk
                    await parent_event_bus.publish(
                        EventFactory.final_answer_chunk(
                            parent_step_number, chunk, promoted_answer
                        )
                    )
                else:
                    await parent_event_bus.publish(
                        EventFactory.subagent_dispatch(
                            parent_step_number,
                            "progress",
                            {
                                "subagent_id": subagent_id,
                                "subagent_path": own_path,
                                "chunk": chunk,
                            },
                        )
                    )
        elif et == ReActEventType.ANSWER_RETRACTED:
            if promote_to_final:
                # TRANSFER: the child's optimistically streamed chunks were
                # re-published as real parent FINAL_ANSWER_CHUNKs, so the
                # retraction must bubble too — child tool calls never emit a
                # parent ACTION_PLANNED, making this the only signal that
                # clears the streamed preamble from consumers and the server
                # accumulator.
                promoted_answer = ""
                await parent_event_bus.publish(
                    EventFactory.answer_retracted(
                        parent_step_number,
                        data.get("retracted_text", ""),
                        data.get("reason", "tool_call"),
                    )
                )
            # Non-transfer: the child's chunks only fed the nested panel's
            # transient progress description; nothing to retract.
        elif et == ReActEventType.THINKING_CHUNK:
            # Stream the child's intermediate reasoning into a nested
            # thinking chunk on the parent's panel. The FE appends deltas
            # to the most recent thinking chunk in nestedChunks; a
            # non-thinking event (tool/observation) closes the streak so
            # subsequent thinking starts a new chunk.
            delta = data.get("delta") or data.get("content") or ""
            if delta:
                await parent_event_bus.publish(
                    EventFactory.subagent_dispatch(
                        parent_step_number,
                        "thinking",
                        {
                            "subagent_id": subagent_id,
                            "subagent_path": own_path,
                            "chunk": delta,
                        },
                    )
                )
        elif et == ReActEventType.ACTION_PLANNED:
            await parent_event_bus.publish(
                EventFactory.subagent_dispatch(
                    parent_step_number,
                    "tool",
                    {
                        "subagent_id": subagent_id,
                        "subagent_path": own_path,
                        "tool_name": data.get("action"),
                        "tool_description": data.get("tool_description"),
                        "status": "planned",
                    },
                )
            )
        elif et == ReActEventType.ACTION_EXECUTING:
            await parent_event_bus.publish(
                EventFactory.subagent_dispatch(
                    parent_step_number,
                    "tool",
                    {
                        "subagent_id": subagent_id,
                        "subagent_path": own_path,
                        "tool_name": data.get("action"),
                        "tool_description": data.get("tool_description"),
                        "status": "executing",
                    },
                )
            )
        elif et == ReActEventType.OBSERVATION:
            # OBSERVATION carries the tool result. The FE pushes an
            # observation chunk AND marks the matching tool chunk
            # completed in one branch. `observation_ref` points at the
            # canonically stored observation row so parent traces can keep
            # a ref + bounded excerpt instead of the raw payload.
            await parent_event_bus.publish(
                EventFactory.subagent_dispatch(
                    parent_step_number,
                    "observation",
                    {
                        "subagent_id": subagent_id,
                        "subagent_path": own_path,
                        "tool_name": data.get("action"),
                        "chunk": data.get("observation") or "",
                        "success": data.get("success", True),
                        "observation_ref": data.get("observation_ref"),
                    },
                )
            )
        elif promote_to_final and et in (
            ReActEventType.VISUALIZATION,
            ReActEventType.MEDIA,
            ReActEventType.ARTIFACT,
        ):
            # Only reachable in TRANSFER mode. In report mode these stay
            # private to the child because the child runs HEADLESS and the
            # parent owns rendering; a transferred child runs on the parent's
            # surface and renders its own output, so its render events have to
            # reach the parent's stream or the chart silently disappears.
            await parent_event_bus.publish(
                ReActEvent(
                    event_type=et,
                    step_number=parent_step_number,
                    data=dict(data),
                )
            )
        elif et == ReActEventType.SUBAGENT_DISPATCH:
            inner = dict(data)
            nested_path = list(inner.get("subagent_path") or [])
            inner["subagent_path"] = [*own_path, *nested_path]
            await parent_event_bus.publish(
                ReActEvent(
                    event_type=ReActEventType.SUBAGENT_DISPATCH,
                    step_number=parent_step_number,
                    data=inner,
                )
            )

    return captured_interrupt


async def dispatch_subagent(
    sub_agent: "SubAgent",
    handoff: "SubAgentHandoff",
    *,
    parent_event_bus: Optional["EventBus"],
    parent_step_number: int,
    parent_assistant_id: str,
    child_id: str,
    counter: DispatchCounter,
    max_depth: int = MAX_NESTING_DEPTH,
    surface_interrupts: bool = False,
    per_handle_limit: Optional[int] = None,
    transfer: bool = False,
) -> "SubAgentResult":
    """Run one parent → child dispatch under the framework's lifecycle.

    Wraps the child's `SubAgent.stream()` with:

      1. Static guardrails (cycle, depth) and counter reservation.
      2. `subagent_dispatch/start` on the parent's bus.
      3. Event forwarding (FINAL_ANSWER_CHUNK → progress, nested
         SUBAGENT_DISPATCH → re-publish with path prefix).
      4. `subagent_dispatch/complete|failed` on the parent's bus.
      5. Returns the SubAgentResult for the orchestrator to package as
         a tool observation.

    If `parent_event_bus` is None (e.g., a test calling this directly),
    events aren't emitted — the dispatch still runs and the result is
    returned. This matches today's `dispatch_assistant_tool` behavior
    when ctx.deps lacks an event_bus.

    ``transfer`` selects TRANSFER mode: the child owns the user-facing turn
    rather than reporting back. Its answer chunks are promoted into the
    parent's own FINAL_ANSWER_CHUNK stream (see ``forward_subagent_events``),
    the lifecycle `complete` event carries an empty ``result`` so the panel
    doesn't render a duplicate copy of the answer, and
    ``result.metadata["transferred"]`` is set so the caller can end the
    parent's turn instead of looping back to the LLM. The dispatch itself is
    otherwise identical — same guardrails, same counter, same budget.
    """
    # Late import to avoid the agent/config/subagent circular at module load.
    from ..subagent import SubAgentResult

    handle = sub_agent.handle

    # Phase 1: guardrails. Failures here surface as the tool observation,
    # not as exceptions to the parent's orchestrator.
    enforce_static_guardrails(
        handle,
        child_id=child_id,
        dispatch_chain=handoff.dispatch_chain,
        depth=handoff.depth,
        max_depth=max_depth,
    )
    await counter.reserve(handle, per_handle_limit=per_handle_limit)

    # Phase 2: subagent_id + lifecycle start.
    subagent_id = f"sub_{uuid.uuid4().hex[:12]}"
    own_path: List[str] = [subagent_id]
    started_at = time.monotonic()

    # [DISPATCH_TIMING] — grep these three phase lines together (start,
    # first_chunk, complete) to tell parallel vs serial under
    # asyncio.gather. All three carry the same `sub=` id so a single
    # dispatch can be threaded across log lines, and `t=` is a
    # process-monotonic timestamp so log lines from different gather
    # branches can be ordered without relying on log arrival order.
    logger.info(
        "[DISPATCH_TIMING] phase=start sub=%s handle=%s parent=%s child=%s depth=%d t=%.6f",
        subagent_id,
        handle,
        parent_assistant_id,
        child_id,
        handoff.depth,
        started_at,
    )

    if parent_event_bus is not None:
        await parent_event_bus.publish(
            EventFactory.subagent_dispatch(
                parent_step_number,
                "start",
                {
                    "subagent_id": subagent_id,
                    "subagent_path": own_path,
                    "handle": handle,
                    "name": sub_agent.name,
                    "parent_assistant_id": parent_assistant_id,
                    "child_assistant_id": child_id,
                },
            )
        )

    # Phase 3: run the child + forward events.
    error: Optional[str] = None
    result: Optional[SubAgentResult] = None
    captured_interrupt: Optional[Dict[str, Any]] = None
    try:
        child_stream = sub_agent.stream(handoff)
        # The Protocol declares stream() as an async iterator; defensively
        # accept either a bare coroutine returning the iterator OR the
        # iterator itself, since some implementations get this wrong.
        if asyncio.iscoroutine(child_stream):
            child_stream = await child_stream  # type: ignore[assignment]
        if parent_event_bus is not None:
            captured_interrupt = await forward_subagent_events(
                child_stream,
                parent_event_bus=parent_event_bus,
                parent_step_number=parent_step_number,
                subagent_id=subagent_id,
                own_path=own_path,
                handle=handle,
                started_at_monotonic=started_at,
                surface_interrupts=surface_interrupts,
                promote_to_final=transfer,
            )
        else:
            # No bus — just drain the stream so the child completes.
            async for _ in child_stream:
                pass
        result = sub_agent.final_result()
    except DispatchGuardrailError:
        # These are pre-flight; we should never reach here with one, but
        # propagate intact if a SubAgent itself raises one internally.
        raise
    except Exception as exc:  # noqa: BLE001 — we surface everything as a failed dispatch
        error = str(exc)
        logger.exception("dispatch_subagent: child stream raised")
        result = SubAgentResult(
            answer="",
            status="failed",
            duration_ms=int((time.monotonic() - started_at) * 1000),
            error=error,
        )

    # Phase 4: lifecycle complete / failed.
    finished_at = time.monotonic()
    logger.info(
        "[DISPATCH_TIMING] phase=complete sub=%s handle=%s status=%s elapsed_ms=%d t=%.6f",
        subagent_id,
        handle,
        result.status,
        int((finished_at - started_at) * 1000),
        finished_at,
    )

    if parent_event_bus is not None:
        sub_event = "complete" if result.status == "completed" else "failed"
        await parent_event_bus.publish(
            EventFactory.subagent_dispatch(
                parent_step_number,
                sub_event,
                {
                    "subagent_id": subagent_id,
                    "subagent_path": own_path,
                    # In TRANSFER mode the answer already streamed into the
                    # main message; echoing it here too would render the same
                    # text twice (once in the panel, once in the message body).
                    # The panel keeps the child's tool work, just not a
                    # duplicate of the answer.
                    "result": "" if transfer else (result.answer or ""),
                    "status": result.status,
                    "error": result.error,
                    "duration_ms": result.duration_ms,
                    "child_assistant_id": child_id,
                    "child_thread_id": result.child_run_id,
                    "transferred": transfer,
                },
            )
        )

    # Attach the subagent_id we minted so the orchestrator can include it
    # in the tool observation back to the parent LLM.
    if not result.metadata.get("subagent_id"):
        result.metadata["subagent_id"] = subagent_id

    # TRANSFER: tell the caller the child owns the answer, so it can end the
    # parent's turn instead of looping back to the LLM for a synthesis pass.
    # Only claim it when the child actually produced something — an empty or
    # failed transfer must fall back to the normal report path, or the turn
    # ends with silence.
    if transfer and result.status == "completed" and (result.answer or "").strip():
        result.metadata["transferred"] = True

    # Phase 2 (R1): if the child paused for the user, surface the interrupt on the
    # result so the dispatch tool returns a clarification-marker observation and the
    # parent pauses too — instead of swallowing an empty final_result().
    if captured_interrupt is not None:
        result.metadata["pending_child_interrupt"] = captured_interrupt

    return result


DISPATCH_TOOL_NAME = "dispatch_assistant"


def _render_dispatcher_description(
    sub_agents: "Sequence[SubAgent]",
    *,
    extra_note: Optional[str] = None,
    transfer_allowed: bool = False,
) -> str:
    """Render the LLM-facing dispatcher tool description from the
    sub-agent list.

    Includes a per-sub-agent block with name + description + ``when_to_use``
    so the LLM has enough surface area to pick the right handle. Mirrors
    the format the Django-side `dispatch_assistant_tool._render_description`
    produces today so the prompt rewrite isn't a hard cut.
    """
    parts = [
        "Delegate this turn's subtask to a sub-agent.",
        "",
        "Each sub-agent has its own focused tools and prompt — picking the "
        "right one for a subtask lets you keep your own context narrow.",
        "",
        "Available sub-agents:",
        "",
    ]
    for sub in sub_agents:
        parts.append(f"### {sub.handle}")
        parts.append(f"**Name**: {sub.name}")
        if sub.description:
            parts.append(f"**Description**: {sub.description.strip()}")
        if sub.when_to_use:
            parts.append(f"**When to use**: {sub.when_to_use.strip()}")
        parts.append("")

    parts.append(
        "You may call `dispatch_assistant` multiple times in one turn for "
        "independent subtasks (the orchestrator runs them in parallel when "
        "possible). The sub-agent sees only your `task`, `intent_summary`, "
        "and — depending on its forward_transcript_window — possibly your "
        "most recent user message. Include any other context the child "
        "needs in `structured_handoff_data`."
    )
    parts.append("")
    parts.append(
        "Runtime policy limits how many times each handle may be dispatched in "
        "one turn. Treat remaining calls as a recovery budget, not a planning "
        "target: combine independent questions for the same sub-agent into one "
        "self-contained task, and reuse completed findings instead of spending "
        "another call. A rejection reports the effective limit."
    )
    if transfer_allowed:
        parts.append("")
        parts.append(
            "**`mode` — who answers the user.**\n"
            "\n"
            "- `report`: the sub-agent's findings come back to you as a tool "
            "result and YOU write the user-facing answer. Use this whenever you "
            "dispatch more than one sub-agent, or when you must combine the "
            "result with data only you have.\n"
            "- `transfer`: you hand the turn over. The sub-agent's reply goes "
            "straight to the user in its own voice, and your turn ENDS. Use it "
            "when ONE sub-agent fully owns the question and you would add "
            "nothing but a re-wording. Visuals are NOT lost: a transferred "
            "sub-agent runs on your surface, calls the same `render_*` tools, "
            "and its output reaches the user — so needing a chart or table is "
            "not a reason to pick `report`.\n"
            "\n"
            "`transfer` is only honored on a lone `dispatch_assistant` call. If "
            "you emit it alongside any other tool call it silently degrades to "
            "`report`, so don't plan a turn around a transfer running in "
            "parallel with anything else."
        )
    if extra_note:
        parts.append("")
        parts.append(extra_note)
    return "\n".join(parts)


def _build_dispatcher_schema(
    sub_agents: "Sequence[SubAgent]", *, transfer_allowed: bool = False
):
    """Build the `ToolSchema` for the synthesized dispatcher.

    The `handle` parameter is an enum constrained to the registered
    sub-agent handles, so the LLM can only call into known children.

    ``transfer_allowed`` adds the optional `mode` parameter. When it's False
    the parameter is omitted from the schema ENTIRELY rather than being
    accepted-and-ignored — a run with nobody to talk to (headless: the
    suggestion pipeline, schedules, `run_assistant_once`) should not even be
    able to express a transfer, and a model can't misuse a parameter it was
    never shown.
    """
    from ..tools.schemas import ParameterSchema, ToolSchema
    from ..tools.types import ParameterType, ToolType

    handles = [sub.handle for sub in sub_agents]
    mode_param = (
        {
            "mode": ParameterSchema(
                name="mode",
                type=ParameterType.STRING,
                description=(
                    "Who answers the user. 'report' — the sub-agent reports "
                    "back and YOU write the user-facing answer; required when "
                    "you dispatch more than one sub-agent or must combine the "
                    "result with your own data. 'transfer' — hand the turn to "
                    "the sub-agent: its reply goes directly to the user and "
                    "your turn ends; use it when ONE sub-agent fully owns the "
                    "question and you would only be re-wording its answer. "
                    "Valid only as the sole tool call in a message."
                ),
                # REQUIRED on purpose. As an optional field with a documented
                # default, models simply omitted it — measured: the router
                # never once set it across several live runs, so transfer was
                # unreachable in practice. Required forces an explicit choice
                # per dispatch, which is the decision we actually want made.
                required=True,
                enum=["report", "transfer"],
            )
        }
        if transfer_allowed
        else {}
    )
    return ToolSchema(
        name=DISPATCH_TOOL_NAME,
        description=_render_dispatcher_description(
            sub_agents, transfer_allowed=transfer_allowed
        ),
        tool_type=ToolType.FUNCTION,
        # Parallel dispatch is gather-safe: each child runs under its own
        # `subagent_id` + `subagent_path`, and the dispatch lifecycle owns
        # the per-turn counter via asyncio.Lock so concurrent reserves are
        # race-free.
        parallelizable=True,
        parameters={
            "handle": ParameterSchema(
                name="handle",
                type=ParameterType.STRING,
                description=(
                    "Which sub-agent should handle this subtask. "
                    "Must be exactly one of the handles listed in the tool description."
                ),
                required=True,
                enum=handles,
            ),
            "task": ParameterSchema(
                name="task",
                type=ParameterType.STRING,
                description=(
                    "The self-contained subtask to delegate. The sub-agent "
                    "sees only this — phrase it so a fresh specialist could "
                    "act on it without further context."
                ),
                required=True,
            ),
            "intent_summary": ParameterSchema(
                name="intent_summary",
                type=ParameterType.STRING,
                description=(
                    "One or two sentences capturing the user's overall intent "
                    "for this turn. Helps the sub-agent frame its work even "
                    "when the subtask itself is narrow."
                ),
                required=True,
            ),
            "structured_handoff_data": ParameterSchema(
                name="structured_handoff_data",
                type=ParameterType.OBJECT,
                description=(
                    "Optional structured payload (free-form object). Use this "
                    "when the sub-agent needs typed inputs — campaign IDs, "
                    "date ranges, prior results — that don't fit naturally in "
                    "a prose task."
                ),
                required=False,
            ),
            **mode_param,
        },
    )


def _err(kind: str, message: str) -> "ToolFailure[Dict[str, Any]]":
    """Return an explicit failure with a structured diagnostic payload."""
    output = {
        "status": "rejected",
        "error_kind": kind,
        "error": message,
        "answer": None,
    }
    return ToolFailure(error=message, output=output, error_type=kind)


def make_subagent_dispatcher_tool(
    sub_agents: "Sequence[SubAgent]",
    *,
    parent_assistant_id: str,
    counter: Optional[DispatchCounter] = None,
    child_id_resolver: Optional[Callable[["SubAgent"], Any]] = None,
    transfer_allowed: bool = False,
) -> "FunctionTool":
    """Build a `dispatch_assistant`-shaped FunctionTool from a SubAgent list.

    The returned tool's `parallelizable=True` flag means Stage 1's parallel
    tool batch executor will gather multiple dispatcher calls from one
    parent assistant turn — that's how this design unlocks parallel
    dispatch without orchestrator surgery.

    Counter semantics:
        - If a counter is passed in, it's the parent's per-turn counter
          (typical for the Django adapter, which constructs one per turn).
        - If ``None``, the closure creates ONE counter that lives for the
          lifetime of the resulting tool. That's correct for unit tests
          and single-turn usage; for production use, callers should
          construct a fresh counter per parent turn and pass it in.

    Child identity:
        - The framework needs a stable identifier per child for cycle
          detection. By default we use the handle itself (handles are
          unique within an agent). The Django adapter overrides this via
          `child_id_resolver` to use the child Assistant's row id, so
          cycle detection survives across handle renames.

    Per-call context (read from ctx.deps when available, falls back to
    sensible defaults so unit tests don't need to wire the full
    orchestrator):
        - ``event_bus`` (Optional[EventBus]) — parent's event bus
        - ``step_number`` (int) — parent's current step
        - ``dispatch_chain`` (List[str]) — ancestor chain
        - ``depth`` (int) — current dispatch nesting depth
        - ``parent_user_message`` (Optional[str]) — last user message
          for handoffs that forward it

    ``transfer_allowed`` exposes the `mode` parameter so the model can hand
    the turn to a sub-agent (see `_build_dispatcher_schema`). Callers derive
    it from the run's surface: interactive surfaces can transfer, headless
    runs cannot.
    """
    from ..tools.function.function_tool import FunctionTool

    schema = _build_dispatcher_schema(sub_agents, transfer_allowed=transfer_allowed)
    # Always exposed: routing to a sub-agent is the parent's primary action, so
    # the tool must be callable without a tool_search round-trip. Set on the
    # factory, not on any one caller — this tool is BUILT rather than registered
    # with the @tool decorator, so it never picks up the decorator's
    # `always_load` kwarg, and that is a property of the factory shared by every
    # call site (Django adapter, Agent(sub_agents=...), configured_subagent).
    schema.metadata["always_load"] = True
    child_map: Dict[str, "SubAgent"] = {sub.handle: sub for sub in sub_agents}
    closure_counter = counter or DispatchCounter()

    def _resolve_child_id(sub: "SubAgent") -> str:
        if child_id_resolver is not None:
            return str(child_id_resolver(sub))
        return sub.handle

    async def _dispatch(
        ctx,
        handle: str,
        task: str,
        intent_summary: str,
        structured_handoff_data: Optional[Dict[str, Any]] = None,
        mode: str = "report",
    ) -> Dict[str, Any] | ToolFailure[Dict[str, Any]]:
        from ..subagent import SubAgentHandoff

        child = child_map.get(handle)
        if child is None:
            return _err(
                "unknown_handle",
                (
                    f"Handle '{handle}' is not a valid sub-agent. "
                    f"Valid handles: {sorted(child_map.keys())}."
                ),
            )

        # Read per-call context from ctx.deps. ctx may be a dict (some
        # tests) or a RunContext (production); handle both.
        deps: Dict[str, Any] = {}
        if hasattr(ctx, "deps") and isinstance(ctx.deps, dict):
            deps = ctx.deps
        elif isinstance(ctx, dict):
            deps = ctx

        event_bus = deps.get("event_bus")
        step_number = int(deps.get("step_number", 0) or 0)
        # The counter on ctx.deps takes precedence over the closure copy
        # so the Django adapter can swap in a per-turn counter without
        # having to rebuild the tool.
        runtime_counter = deps.get("dispatch_counter") or closure_counter
        dispatch_chain: List[str] = list(deps.get("dispatch_chain") or [])
        depth = int(deps.get("dispatch_depth", 1) or 1)
        parent_user_message = deps.get("parent_user_message")

        handoff = SubAgentHandoff(
            task=task,
            intent_summary=intent_summary,
            structured_payload=structured_handoff_data,
            depth=depth,
            dispatch_chain=dispatch_chain,
            parent_user_message=parent_user_message,
        )

        child_id = _resolve_child_id(child)

        # A transfer is only coherent as the sole tool call in the message —
        # "hand over the turn AND also do these other things" has no meaning.
        # The orchestrator sets `batch_size` for parallel batches; anything
        # above 1 degrades to a report rather than failing the call, so a
        # mis-planned turn still produces an answer.
        transfer = transfer_allowed and mode == "transfer"
        if transfer and int(deps.get("batch_size", 1) or 1) > 1:
            logger.info(
                "dispatch: transfer downgraded to report — not a lone tool call "
                "(handle=%s batch_size=%s)",
                handle,
                deps.get("batch_size"),
            )
            transfer = False

        try:
            result = await dispatch_subagent(
                child,
                handoff,
                parent_event_bus=event_bus,
                parent_step_number=step_number,
                parent_assistant_id=parent_assistant_id,
                child_id=child_id,
                counter=runtime_counter,
                surface_interrupts=bool(deps.get("surface_child_interrupts")),
                transfer=transfer,
            )
        except DispatchGuardrailError as guard:
            return _err(guard.kind, str(guard))

        # Phase 2 (R1/R2): the child paused for the user. Return a private
        # marker observation so the parent's pause detection fires with the
        # CHILD interrupt id/path — instead of handing the model an empty answer
        # or fabricating a parent-owned approval.
        child_interrupt = result.metadata.get("pending_child_interrupt")
        if child_interrupt and child_interrupt.get("kind") == "tool_approval":
            from ..checkpoint import AgentFrame, PendingInterrupt
            from ..interrupt import mint_interrupt_id
            from ..tools.clarification import child_tool_approval_observation

            idata = child_interrupt.get("data") or {}
            frame_path = ["root"] + list(child_interrupt.get("subagent_path") or [])
            child_tool_call_id = idata.get("tool_call_id")
            interrupt_id = idata.get("interrupt_id") or mint_interrupt_id(
                "tool_approval", child_tool_call_id
            )
            payload = {
                "tool_name": idata.get("tool_name"),
                "tool_inputs": idata.get("tool_inputs") or {},
                "tool_description": idata.get("tool_description") or "",
                "tool_schema": idata.get("tool_schema") or {},
                "reason": idata.get("reason"),
                "handle": handle,
                "child_assistant_id": child_id,
                "child_thread_id": result.child_run_id,
                "subagent_id": result.metadata.get("subagent_id"),
                "subagent_path": child_interrupt.get("subagent_path"),
            }
            checkpoint = getattr(ctx, "checkpoint", None)
            if checkpoint is not None:
                pending = PendingInterrupt(
                    interrupt_id=interrupt_id,
                    kind="tool_approval",
                    raised_by_path=frame_path,
                    payload=payload,
                    tool_call_id=child_tool_call_id,
                )
                checkpoint.agent_frames["/".join(frame_path)] = AgentFrame(
                    path=frame_path,
                    pending_interrupt=pending,
                    metadata={
                        "handle": handle,
                        "child_assistant_id": child_id,
                        "child_thread_id": result.child_run_id,
                        "subagent_id": result.metadata.get("subagent_id"),
                        "subagent_path": child_interrupt.get("subagent_path"),
                    },
                )
            return child_tool_approval_observation(
                tool_name=payload["tool_name"],
                tool_inputs=payload["tool_inputs"],
                tool_call_id=child_tool_call_id,
                interrupt_id=interrupt_id,
                reason=payload["reason"],
                tool_description=payload["tool_description"],
                tool_schema=payload["tool_schema"],
                dispatch_meta={
                    "handle": handle,
                    "child_assistant_id": child_id,
                    "child_thread_id": result.child_run_id,
                    "subagent_id": result.metadata.get("subagent_id"),
                    "status": "awaiting_tool_approval",
                    "subagent_path": child_interrupt.get("subagent_path"),
                    "raised_by_path": frame_path,
                },
            )

        if child_interrupt and child_interrupt.get("kind") == "clarification":
            from ..checkpoint import AgentFrame, PendingInterrupt
            from ..interrupt import mint_interrupt_id
            from ..tools.clarification import child_clarification_observation

            idata = child_interrupt.get("data") or {}
            frame_path = ["root"] + list(child_interrupt.get("subagent_path") or [])
            checkpoint = getattr(ctx, "checkpoint", None)
            if checkpoint is not None:
                pending = PendingInterrupt(
                    interrupt_id=idata.get("interrupt_id")
                    or mint_interrupt_id("clarification", idata.get("tool_call_id")),
                    kind="clarification",
                    raised_by_path=frame_path,
                    payload={
                        "questions": idata.get("questions") or [],
                        "context": idata.get("context"),
                        "handle": handle,
                        "child_assistant_id": child_id,
                        "subagent_id": result.metadata.get("subagent_id"),
                        "subagent_path": child_interrupt.get("subagent_path"),
                    },
                    tool_call_id=idata.get("tool_call_id"),
                )
                checkpoint.agent_frames["/".join(frame_path)] = AgentFrame(
                    path=frame_path,
                    pending_interrupt=pending,
                    metadata={
                        "handle": handle,
                        "child_assistant_id": child_id,
                        "subagent_id": result.metadata.get("subagent_id"),
                        "subagent_path": child_interrupt.get("subagent_path"),
                    },
                )
            return child_clarification_observation(
                questions=idata.get("questions") or [],
                context=idata.get("context"),
                dispatch_meta={
                    "handle": handle,
                    "child_assistant_id": child_id,
                    "subagent_id": result.metadata.get("subagent_id"),
                    "status": "awaiting_clarification",
                    "subagent_path": child_interrupt.get("subagent_path"),
                },
            )

        transferred = bool(result.metadata.get("transferred"))
        if transferred:
            # Hand the answer to the orchestrator out-of-band and keep the tool
            # observation TINY. The turn ends here, so the model never reads
            # this — but it still has to exist, because the assistant message
            # already carries a tool_use block and a tool_use without a matching
            # tool_result breaks replay of this thread on the next turn.
            # The answer itself reaches history as the final assistant message.
            deps["pending_transfer"] = {
                "answer": result.answer,
                "handle": handle,
                "child_assistant_id": child_id,
                "child_thread_id": result.child_run_id,
                "subagent_id": result.metadata.get("subagent_id"),
            }
            observation = {
                "handle": handle,
                "status": "transferred",
                "note": (
                    "Turn handed to the sub-agent; its reply went directly to "
                    "the user."
                ),
            }
        else:
            observation = {
                "handle": handle,
                "child_assistant_id": child_id,
                "subagent_id": result.metadata.get("subagent_id"),
                "status": result.status,
                "answer": result.answer,
                "error": result.error,
                "duration_ms": result.duration_ms,
            }
        # Ledger entry targets the root checkpoint (shared blackboard for the
        # dispatch tree) and stays bounded: digest only — the full dispatch
        # observation is recorded by the parent's observation seam.
        deps = getattr(ctx, "deps", None)
        checkpoint = deps.get("root_checkpoint") if isinstance(deps, dict) else None
        if checkpoint is None:
            checkpoint = getattr(ctx, "checkpoint", None)
        if checkpoint is not None and hasattr(checkpoint, "merge_ledger"):
            import time as _time

            from ..checkpoint import (
                DispatchLedgerEntry,
                make_ledger_digest,
                stable_json_hash,
            )

            checkpoint.merge_ledger(
                [
                    DispatchLedgerEntry(
                        kind="dispatch",
                        success=result.status == "completed",
                        digest=make_ledger_digest(str(result.answer or result.error or "")),
                        produced_at=_time.time(),
                        turn_index=getattr(checkpoint, "turn_index", 0),
                        handle=handle,
                        task_hash=stable_json_hash(
                            {
                                "task": task,
                                "intent_summary": intent_summary,
                                "structured_handoff_data": structured_handoff_data,
                            }
                        ),
                        produced_by_path=["root"],
                    )
                ]
            )

        # Tool observation back to the parent LLM. Match the shape the
        # Django-side dispatch_assistant tool returns so consumers of the
        # observation (recovery, citation, the parent's own reasoning)
        # don't need to fork.
        if result.status == "failed":
            return ToolFailure(
                error=result.error or f"Sub-agent '{handle}' failed.",
                output=observation,
                error_type="subagent_failed",
            )
        return observation

    _dispatch._tool_schema = schema  # type: ignore[attr-defined]
    return FunctionTool(_dispatch)


__all__ = [
    "DEFAULT_MAX_CALLS_PER_HANDLE",
    "DEFAULT_MAX_DISPATCHES_PER_TURN",
    "DISPATCH_TOOL_NAME",
    "MAX_NESTING_DEPTH",
    "DispatchCounter",
    "DispatchGuardrailError",
    "dispatch_subagent",
    "enforce_static_guardrails",
    "forward_subagent_events",
    "make_subagent_dispatcher_tool",
]
