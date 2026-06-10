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

    async def reserve(self, handle: str) -> None:
        """Reserve one dispatch slot for `handle`, raising if over budget.

        On success, the per-handle and total counters are both
        incremented before returning. On failure, the counters are
        unchanged. Either way, the lock is released before raising.
        """
        async with self._lock:
            per_handle = self.counts.get(handle, 0)
            if per_handle >= self.max_per_handle:
                raise DispatchGuardrailError(
                    "per_handle_budget_exceeded",
                    (
                        f"Already dispatched to '{handle}' "
                        f"{per_handle} times this turn "
                        f"(limit: {self.max_per_handle})."
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
            # completed in one branch.
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
                    },
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
    await counter.reserve(handle)

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
                    "result": result.answer or "",
                    "status": result.status,
                    "error": result.error,
                    "duration_ms": result.duration_ms,
                    "child_assistant_id": child_id,
                    "child_thread_id": result.child_run_id,
                },
            )
        )

    # Attach the subagent_id we minted so the orchestrator can include it
    # in the tool observation back to the parent LLM.
    if not result.metadata.get("subagent_id"):
        result.metadata["subagent_id"] = subagent_id

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
    if extra_note:
        parts.append("")
        parts.append(extra_note)
    return "\n".join(parts)


def _build_dispatcher_schema(sub_agents: "Sequence[SubAgent]"):
    """Build the `ToolSchema` for the synthesized dispatcher.

    The `handle` parameter is an enum constrained to the registered
    sub-agent handles, so the LLM can only call into known children.
    """
    from ..tools.schemas import ParameterSchema, ToolSchema
    from ..tools.types import ParameterType, ToolType

    handles = [sub.handle for sub in sub_agents]
    return ToolSchema(
        name=DISPATCH_TOOL_NAME,
        description=_render_dispatcher_description(sub_agents),
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
        },
    )


def _err(kind: str, message: str) -> Dict[str, Any]:
    """Synthesize a structured tool observation for the parent LLM."""
    return {
        "status": "rejected",
        "error_kind": kind,
        "error": message,
        "answer": None,
    }


def make_subagent_dispatcher_tool(
    sub_agents: "Sequence[SubAgent]",
    *,
    parent_assistant_id: str,
    counter: Optional[DispatchCounter] = None,
    child_id_resolver: Optional[Callable[["SubAgent"], Any]] = None,
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
    """
    from ..tools.function.function_tool import FunctionTool

    schema = _build_dispatcher_schema(sub_agents)
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
    ) -> Dict[str, Any]:
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

        observation = {
            "handle": handle,
            "child_assistant_id": child_id,
            "subagent_id": result.metadata.get("subagent_id"),
            "status": result.status,
            "answer": result.answer,
            "error": result.error,
            "duration_ms": result.duration_ms,
        }
        checkpoint = getattr(ctx, "checkpoint", None)
        if checkpoint is not None and hasattr(checkpoint, "merge_ledger"):
            from ..checkpoint import DispatchLedgerEntry, stable_json_hash

            checkpoint.merge_ledger(
                [
                    DispatchLedgerEntry(
                        kind="dispatch",
                        success=result.status == "completed",
                        observation=str(observation),
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
