"""Durable, typed run checkpoint — the spine of pause/resume + multi-agent context.

This is the single, serializable source of truth for everything that must survive
*across* a pause (clarification / tool approval) or *cross* an agent boundary
(parent ↔ sub-agent). It replaces three scattered, untyped stores that the system
historically reconstructed from the model each turn:

  1. ad-hoc ``ctx.deps`` magic keys (``pending_approved_action`` …)
  2. the ``thread.metadata["pending_clarification"]`` blob
  3. the by-reference ``parent_run_context`` mutable scratchpad
     (deleted July 2026 — the ledger + facts reducers replaced it)

The design principle is *durable execution* (cf. LangGraph checkpoints): control-flow
state is owned by code and persisted, never re-derived by the LLM. See
``~/.claude/plans/you-are-an-experienced-tingly-cook.md`` for the full architecture
and the root-cause theory.

Phase 0 introduces this model as an **inert, additive** artifact: the Django adapter
round-trips it alongside the legacy stores (dual-write) but nothing reads it to drive
behavior yet. Phases 1–3 progressively flip readers onto it and delete the heuristics.

Everything here is pure data with tolerant ``to_dict`` / ``from_dict`` so it can be
persisted as JSON (initially ``thread.metadata["agent_checkpoint"]``) and evolved
without breaking old rows. Unknown keys are preserved in ``extra`` for
forward-compatibility.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from .message import Message

# Bump when the serialized shape changes incompatibly. ``from_dict`` tolerates
# older payloads; this is for diagnostics / future migration shims.
CHECKPOINT_VERSION = 1


def stable_json_hash(value: Any) -> str:
    """Deterministic short hash for reducer identities."""
    try:
        raw = json.dumps(value or {}, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:
        raw = str(value)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class EstablishedFact:
    """A clarification the user has already resolved — the deterministic answer store.

    Keyed by ``key`` (a stable, model-supplied slug such as ``"daily_budget"``), so a
    re-asked question short-circuits to the known answer by exact-match lookup instead
    of the legacy 50%-token-overlap heuristic. Injected into every agent's prompt as an
    "already settled — do not ask again" block (Pillar 3).
    """

    key: str
    answer: Any
    question_text: str = ""
    options: List[Any] = field(default_factory=list)
    # Dispatch address of the agent that asked (and to whom the answer belongs).
    answered_by_path: List[str] = field(default_factory=list)
    # The turn index on which it was resolved (diagnostics / ordering).
    turn: int = 0
    # Provenance, with a precedence lattice enforced by ``Checkpoint.upsert_fact``:
    #   "user" — an explicit clarification answer. Never overwritten by tools.
    #   "tool" — promoted from a tool observation (fact extractor). May refresh
    #            other tool facts, never a user answer. Tool-fact keys are
    #            namespaced ("tool:...") so collision with clarification keys
    #            is structurally impossible on top of the lattice.
    source: str = "user"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "answer": self.answer,
            "question_text": self.question_text,
            "options": self.options,
            "answered_by_path": self.answered_by_path,
            "turn": self.turn,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EstablishedFact":
        return cls(
            key=data["key"],
            answer=data.get("answer"),
            question_text=data.get("question_text", ""),
            options=list(data.get("options") or []),
            answered_by_path=list(data.get("answered_by_path") or []),
            turn=int(data.get("turn", 0)),
            source=data.get("source", "user"),
        )


@dataclass
class PendingInterrupt:
    """A unified human-in-the-loop pause (clarification | tool_approval | plan_approval).

    Collapses what were three divergent pause paths into one primitive. ``raised_by_path``
    is the dispatch address of the agent that paused (e.g.
    ``["root", "<dispatch_tool_call_id>"]``) so a ``Resume`` routes deterministically back
    to the exact frame — including a sub-agent buried in the dispatch tree. Phase 1 uses
    this at the root; Phase 2 makes it tree-aware.
    """

    interrupt_id: str
    kind: str  # "clarification" | "tool_approval" | "plan_approval"
    raised_by_path: List[str] = field(default_factory=list)
    # Clarification: ``{"questions": [...], "context": ...}``.
    # Approval: ``{"tool_name", "tool_inputs", "tool_schema", "tool_description", "preview"}``.
    payload: Dict[str, Any] = field(default_factory=dict)
    tool_call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interrupt_id": self.interrupt_id,
            "kind": self.kind,
            "raised_by_path": self.raised_by_path,
            "payload": self.payload,
            "tool_call_id": self.tool_call_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PendingInterrupt":
        return cls(
            interrupt_id=data["interrupt_id"],
            kind=data["kind"],
            raised_by_path=list(data.get("raised_by_path") or []),
            payload=dict(data.get("payload") or {}),
            tool_call_id=data.get("tool_call_id"),
        )


@dataclass
class ResumeCommand:
    """A deterministic response to a previously persisted human interrupt.

    This is the SDK-facing resume primitive. Adapters translate UI/message
    payloads into this shape, then the runtime validates ``interrupt_id`` against
    the checkpoint before continuing. The model never reconstructs or chooses
    the resume path.
    """

    interrupt_id: str
    kind: Literal["clarification", "tool_approval", "plan_approval"]
    value: Dict[str, Any] = field(default_factory=dict)
    decision: Literal["answered", "approved", "rejected"] = "answered"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interrupt_id": self.interrupt_id,
            "kind": self.kind,
            "value": self.value,
            "decision": self.decision,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResumeCommand":
        return cls(
            interrupt_id=data["interrupt_id"],
            kind=data.get("kind", "clarification"),
            value=dict(data.get("value") or {}),
            decision=data.get("decision", "answered"),
        )


@dataclass
class PendingApprovedAction:
    """A tool call the user has approved, to be executed deterministically on resume.

    Formalizes the existing ``ctx.deps["pending_approved_action"]`` descriptor consumed
    by ``orchestrator._execute_pending_approved_action``. ``inputs`` may be the user's
    edited values (approve-with-edits).
    """

    tool_name: str
    tool_call_id: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    interrupt_id: Optional[str] = None
    raised_by_path: List[str] = field(default_factory=list)
    parent_tool_call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "tool_call_id": self.tool_call_id,
            "inputs": self.inputs,
            "interrupt_id": self.interrupt_id,
            "raised_by_path": self.raised_by_path,
            "parent_tool_call_id": self.parent_tool_call_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PendingApprovedAction":
        return cls(
            tool_name=data["tool_name"],
            tool_call_id=data["tool_call_id"],
            inputs=dict(data.get("inputs") or {}),
            interrupt_id=data.get("interrupt_id"),
            raised_by_path=list(data.get("raised_by_path") or []),
            parent_tool_call_id=data.get("parent_tool_call_id"),
        )


# Hard cap on a ledger entry's digest — SERIALIZER-ENFORCED, not a convention.
# The ledger persists inside the thread checkpoint forever-ish; unbounded
# "digests" are how the v1 ledger grew to 246KB blobs in production.
LEDGER_DIGEST_MAX_CHARS = 400


def make_ledger_digest(text: str) -> str:
    """One-line, capped digest of an observation for ledger entries."""
    if not text:
        return ""
    collapsed = " ".join(str(text).split())
    if len(collapsed) <= LEDGER_DIGEST_MAX_CHARS:
        return collapsed
    return collapsed[: LEDGER_DIGEST_MAX_CHARS - 1] + "…"


@dataclass
class DispatchLedgerEntry:
    """A structured record of work already done — the deterministic blackboard.

    Replaces the LLM-trusted ``prior_tool_calls`` string blob. Two shapes share one
    typed entry, discriminated by ``kind``:

      * ``"dispatch"``  — a completed sub-agent dispatch (``handle``, ``task_hash``).
      * ``"tool_call"`` — a tool the tree already ran (``tool_name``, ``inputs_hash``);
        deduped by ``(tool_name, inputs_hash)``.

    v2 shape: entries are BOUNDED. The full observation lives in the canonical
    observation store behind ``observation_ref``; the entry carries only a capped
    ``digest``. ``produced_at`` (wall clock) is the TTL basis for the read-through
    dedupe gate; ``turn_index`` is the ROOT thread's turn counter (NOT the intra-run
    step number) and drives pruning.
    """

    kind: str  # "dispatch" | "tool_call"
    success: bool = True
    digest: str = ""
    observation_ref: Optional[str] = None
    produced_at: float = 0.0
    turn_index: int = 0
    # dispatch
    handle: Optional[str] = None
    task_hash: Optional[str] = None
    # tool_call
    tool_name: Optional[str] = None
    inputs_hash: Optional[str] = None
    # Dispatch address of the agent that produced this entry.
    produced_by_path: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.digest = make_ledger_digest(self.digest)

    def dedupe_key(self) -> str:
        """Deterministic identity used by the reducer to merge / drop duplicates."""
        if self.kind == "tool_call":
            return f"tool_call::{self.tool_name}::{self.inputs_hash}"
        return f"dispatch::{self.handle}::{self.task_hash}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "success": self.success,
            "digest": make_ledger_digest(self.digest),
            "observation_ref": self.observation_ref,
            "produced_at": self.produced_at,
            "turn_index": self.turn_index,
            "handle": self.handle,
            "task_hash": self.task_hash,
            "tool_name": self.tool_name,
            "inputs_hash": self.inputs_hash,
            "produced_by_path": self.produced_by_path,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DispatchLedgerEntry":
        # v1 tolerance: legacy entries carried the FULL observation and an
        # intra-run step number in ``turn``. Fold the observation into a capped
        # digest (this is what shrinks old 246KB blobs on first load) and map
        # ``turn`` → ``turn_index``.
        digest = data.get("digest")
        if digest is None:
            digest = data.get("observation", "")
        turn_index = data.get("turn_index")
        if turn_index is None:
            turn_index = data.get("turn", 0)
        return cls(
            kind=data["kind"],
            success=bool(data.get("success", True)),
            digest=make_ledger_digest(str(digest or "")),
            observation_ref=data.get("observation_ref"),
            produced_at=float(data.get("produced_at", 0.0) or 0.0),
            turn_index=int(turn_index or 0),
            handle=data.get("handle"),
            task_hash=data.get("task_hash"),
            tool_name=data.get("tool_name"),
            inputs_hash=data.get("inputs_hash"),
            produced_by_path=list(data.get("produced_by_path") or []),
        )


@dataclass
class AgentFrame:
    """A suspended agent's saved position so it can resume exactly where it paused.

    Stored per dispatch-path so a paused sub-agent (Phase 2) rehydrates its own
    transcript + step pointer rather than being re-dispatched fresh. The checkpoint
    granularity is the *step boundary*: on resume the interrupted step re-runs from its
    start, which is safe because writes are interrupt-gated and reads are ledger-deduped
    (the R5 idempotency invariant).
    """

    path: List[str]
    transcript: List[Message] = field(default_factory=list)
    step_pointer: int = 0
    pending_interrupt: Optional[PendingInterrupt] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "transcript": [m.to_dict() for m in self.transcript],
            "step_pointer": self.step_pointer,
            "pending_interrupt": (
                self.pending_interrupt.to_dict() if self.pending_interrupt else None
            ),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentFrame":
        pi = data.get("pending_interrupt")
        return cls(
            path=list(data.get("path") or []),
            transcript=[Message.from_dict(m) for m in (data.get("transcript") or [])],
            step_pointer=int(data.get("step_pointer", 0)),
            pending_interrupt=PendingInterrupt.from_dict(pi) if pi else None,
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class Checkpoint:
    """The durable, typed run state persisted per thread.

    Owned by the SDK, written by the orchestrator, persisted/loaded by the adapter
    around each turn. Everything that must survive a pause or cross an agent boundary
    lives here — and only here, once the legacy stores are retired (Phase 4).
    """

    version: int = CHECKPOINT_VERSION
    thread_id: Optional[str] = None
    # NOTE: a ``transcript`` field existed here through July 2026 but never
    # had a writer — the conversation record lives in Message rows and the
    # per-message execution timelines (excerpt+ref). Removed; ``from_dict``
    # drops the key from old blobs.
    # Deterministic answer store (Defect 1).
    established_facts: List[EstablishedFact] = field(default_factory=list)
    # Canonical HITL interrupts. Only one is active in the current UI contract;
    # additional surfaced interrupts can be queued by id and activated
    # deterministically after the current one resolves.
    interrupts: Dict[str, PendingInterrupt] = field(default_factory=dict)
    active_interrupt_id: Optional[str] = None
    interrupt_queue: List[str] = field(default_factory=list)
    # The resume command supplied by the adapter for the active interrupt.
    resume: Optional[ResumeCommand] = None
    # Legacy mirror of ``active_interrupt()``. Kept so old rows/tests and the
    # current Django projection can continue to round-trip while readers flip
    # to the authoritative fields above.
    pending_interrupt: Optional[PendingInterrupt] = None
    # A user-approved tool call awaiting deterministic execution on resume.
    pending_approved_action: Optional[PendingApprovedAction] = None
    # Monotonic ROOT-turn counter: incremented by the root run at turn start.
    # Ledger entries stamp it (pruning basis); NOT the intra-run step number.
    turn_index: int = 0
    # Structured blackboard of already-done work (Defect 3).
    dispatch_ledger: List[DispatchLedgerEntry] = field(default_factory=list)
    # Session child threads for CONTINUE_THREAD handoff edges, keyed by
    # handle: {"child_thread_id", "created_turn", "last_used_turn",
    # "turns_used"}. The busy lease lives on the child Thread row itself
    # (DB-level, survives crashes), NOT here — this checkpoint persists only
    # at turn end, so two concurrent turns would both read "free".
    child_sessions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Suspended agent frames keyed by joined dispatch-path (Defect 2 / Phase 2).
    agent_frames: Dict[str, AgentFrame] = field(default_factory=dict)
    # Forward-compat catch-all: unknown top-level keys are preserved here so an
    # older binary never destroys state written by a newer one.
    extra: Dict[str, Any] = field(default_factory=dict)

    # --- established-facts helpers (used from Phase 1) ----------------------------

    def fact(self, key: str) -> Optional[EstablishedFact]:
        """Exact-match lookup of an already-resolved clarification by its stable key."""
        for f in self.established_facts:
            if f.key == key:
                return f
        return None

    def facts_by_key(self) -> Dict[str, EstablishedFact]:
        # Last write wins, matching append-then-supersede semantics.
        out: Dict[str, EstablishedFact] = {}
        for f in self.established_facts:
            out[f.key] = f
        return out

    # Bound on retained facts; superseded/oldest tool facts drop first.
    FACTS_MAX = 100

    def upsert_fact(self, fact: EstablishedFact) -> None:
        """Append a fact, replacing any prior answer for the same key.

        Precedence lattice: a user-answered fact is never downgraded by a
        tool-promoted one (``user > tool``). Tool facts refresh freely.
        """
        existing = self.fact(fact.key)
        if (
            existing is not None
            and existing.source == "user"
            and fact.source == "tool"
        ):
            return
        self.established_facts = [f for f in self.established_facts if f.key != fact.key]
        self.established_facts.append(fact)
        if len(self.established_facts) > self.FACTS_MAX:
            # Drop the oldest tool facts first; user answers survive longest.
            tool_facts = [f for f in self.established_facts if f.source == "tool"]
            if tool_facts:
                self.established_facts.remove(tool_facts[0])
            else:
                self.established_facts.pop(0)

    # --- dispatch-ledger reducer (used from Phase 3) ------------------------------

    def merge_ledger(self, entries: List[DispatchLedgerEntry]) -> None:
        """Deterministically merge new entries, deduping by ``dedupe_key``.

        Last write wins so a later, successful re-run supersedes an earlier failure for
        the same logical work item. This is the reducer that replaces the racy,
        by-reference scratchpad it replaced. Synchronous on purpose:
        no await inside ⇒ atomic on the event loop even under parallel dispatch.
        """
        index = {e.dedupe_key(): i for i, e in enumerate(self.dispatch_ledger)}
        for entry in entries:
            k = entry.dedupe_key()
            if k in index:
                self.dispatch_ledger[index[k]] = entry
            else:
                index[k] = len(self.dispatch_ledger)
                self.dispatch_ledger.append(entry)

    # Ledger retention: entries older than this many root turns are dropped at
    # persist time; the total entry count is capped regardless.
    LEDGER_KEEP_TURNS = 5
    LEDGER_MAX_ENTRIES = 200

    def prune_ledger(self) -> None:
        """Bound the persisted ledger: drop stale turns, cap total entries.

        Called by the adapter immediately before serialization so the blackboard
        stays useful for cross-turn dedupe without recreating the unbounded v1
        blobs. Oldest entries drop first when the count cap binds.
        """
        keep_after = self.turn_index - self.LEDGER_KEEP_TURNS
        kept = [e for e in self.dispatch_ledger if e.turn_index >= keep_after]
        if len(kept) > self.LEDGER_MAX_ENTRIES:
            kept = kept[-self.LEDGER_MAX_ENTRIES :]
        self.dispatch_ledger = kept

    def ledger_lookup(self, dedupe_key: str) -> Optional[DispatchLedgerEntry]:
        """Exact-match lookup by reducer identity (read side of the gate)."""
        for entry in self.dispatch_ledger:
            if entry.dedupe_key() == dedupe_key:
                return entry
        return None

    # Worklog rendering bounds (the handoff block a dispatched child receives).
    WORKLOG_TURN_WINDOW = 2
    WORKLOG_MAX_ENTRIES = 20

    def render_worklog_block(
        self,
        *,
        exclude_produced_by: Optional[List[str]] = None,
    ) -> str:
        """Render the deterministic "[work_already_done]" block for a handoff.

        Replaces the LLM-trusted ``[prior_tool_calls]`` scratchpad blob with a
        bounded, ledger-backed digest of the tree's recent work: tool calls
        AND completed dispatches, successes AND failures (a child that can
        see a sibling's error doesn't repeat the mistake). Each entry carries
        its ``ref`` so any agent can pull the full stored observation via
        ``read_observation`` — even for tools it doesn't own.

        ``exclude_produced_by`` drops entries produced by the given dispatch
        path (don't feed a continued session its own work back).
        """
        window_start = self.turn_index - self.WORKLOG_TURN_WINDOW + 1
        entries = [
            e for e in self.dispatch_ledger if e.turn_index >= window_start
        ]
        if exclude_produced_by:
            entries = [
                e for e in entries if e.produced_by_path != list(exclude_produced_by)
            ]
        if not entries:
            return ""
        entries = entries[-self.WORKLOG_MAX_ENTRIES :]

        lines = [
            "[work_already_done]",
            "Work the agent tree already completed recently (successes AND "
            "failures — don't repeat a failed call unchanged). Digests below "
            "are truncated; fetch any full output with read_observation(ref=…). "
            "Re-issuing an identical fresh read is served from cache, but "
            "prefer reusing these results.",
        ]
        for e in entries:
            status = "OK" if e.success else "ERROR"
            if e.kind == "dispatch":
                label = f"dispatch→{e.handle}"
            else:
                label = e.tool_name or "?"
            ref_part = f" (ref: {e.observation_ref})" if e.observation_ref else ""
            lines.append(f"- [{status}] {label} → {e.digest}{ref_part}")
        return "\n".join(lines)

    # --- interrupt helpers -----------------------------------------------------------

    def active_interrupt(self) -> Optional[PendingInterrupt]:
        """Return the active interrupt using the canonical fields, with legacy fallback."""
        if self.active_interrupt_id and self.active_interrupt_id in self.interrupts:
            return self.interrupts[self.active_interrupt_id]
        return self.pending_interrupt

    def set_interrupt(self, interrupt: PendingInterrupt) -> None:
        """Make ``interrupt`` the single active UI interrupt.

        Existing active interrupts are retained in ``interrupts`` for audit/resume
        validation but only this id is active. This mirrors the current frontend
        contract of one human prompt at a time. Callers that must not lose the
        prior active (parallel pauses in one run) demote it into
        ``interrupt_queue`` first — see the orchestrator's ``_record_interrupt``.
        """
        self.interrupts[interrupt.interrupt_id] = interrupt
        self.active_interrupt_id = interrupt.interrupt_id
        self.pending_interrupt = interrupt
        # An interrupt can't be both active and queued (a re-raised child pause
        # mints the same id deterministically from its tool_call_id).
        if interrupt.interrupt_id in self.interrupt_queue:
            self.interrupt_queue = [
                i for i in self.interrupt_queue if i != interrupt.interrupt_id
            ]

    def enqueue_interrupt(self, interrupt: PendingInterrupt) -> None:
        """Queue an interrupt without making it active."""
        self.interrupts[interrupt.interrupt_id] = interrupt
        if interrupt.interrupt_id not in self.interrupt_queue:
            self.interrupt_queue.append(interrupt.interrupt_id)
        if not self.active_interrupt_id:
            self.activate_next_interrupt()

    def activate_next_interrupt(self) -> Optional[PendingInterrupt]:
        """Promote the next queued interrupt to active, if any."""
        while self.interrupt_queue:
            next_id = self.interrupt_queue.pop(0)
            interrupt = self.interrupts.get(next_id)
            if interrupt is not None:
                self.active_interrupt_id = next_id
                self.pending_interrupt = interrupt
                return interrupt
        self.active_interrupt_id = None
        self.pending_interrupt = None
        return None

    def clear_active_interrupt(self) -> None:
        """Clear the active interrupt and promote the next queued interrupt."""
        if self.active_interrupt_id:
            self.interrupts.pop(self.active_interrupt_id, None)
        self.resume = None
        self.activate_next_interrupt()

    # --- serialization ------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "version": self.version,
            "thread_id": self.thread_id,
            "established_facts": [f.to_dict() for f in self.established_facts],
            "interrupts": {k: v.to_dict() for k, v in self.interrupts.items()},
            "active_interrupt_id": self.active_interrupt_id,
            "interrupt_queue": self.interrupt_queue,
            "resume": self.resume.to_dict() if self.resume else None,
            "pending_interrupt": (
                self.pending_interrupt.to_dict() if self.pending_interrupt else None
            ),
            "pending_approved_action": (
                self.pending_approved_action.to_dict()
                if self.pending_approved_action
                else None
            ),
            "turn_index": self.turn_index,
            "dispatch_ledger": [e.to_dict() for e in self.dispatch_ledger],
            "child_sessions": {k: dict(v) for k, v in self.child_sessions.items()},
            "agent_frames": {k: v.to_dict() for k, v in self.agent_frames.items()},
        }
        # Preserve any forward-compat keys without letting them shadow known fields.
        for k, v in self.extra.items():
            data.setdefault(k, v)
        return data

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Checkpoint":
        if not data:
            return cls()
        known = {
            "version",
            "thread_id",
            "transcript",
            "established_facts",
            "interrupts",
            "active_interrupt_id",
            "interrupt_queue",
            "resume",
            "pending_interrupt",
            "pending_approved_action",
            "turn_index",
            "dispatch_ledger",
            "child_sessions",
            "agent_frames",
        }
        pi = data.get("pending_interrupt")
        ppa = data.get("pending_approved_action")
        resume = data.get("resume")
        interrupts = {
            k: PendingInterrupt.from_dict(v)
            for k, v in (data.get("interrupts") or {}).items()
            if isinstance(v, dict)
        }
        pending_interrupt = PendingInterrupt.from_dict(pi) if pi else None
        active_interrupt_id = data.get("active_interrupt_id")
        if pending_interrupt and not interrupts:
            interrupts[pending_interrupt.interrupt_id] = pending_interrupt
            active_interrupt_id = active_interrupt_id or pending_interrupt.interrupt_id
        if active_interrupt_id and active_interrupt_id in interrupts and pending_interrupt is None:
            pending_interrupt = interrupts[active_interrupt_id]
        return cls(
            version=int(data.get("version", CHECKPOINT_VERSION)),
            thread_id=data.get("thread_id"),
            established_facts=[
                EstablishedFact.from_dict(f) for f in (data.get("established_facts") or [])
            ],
            interrupts=interrupts,
            active_interrupt_id=active_interrupt_id,
            interrupt_queue=list(data.get("interrupt_queue") or []),
            resume=ResumeCommand.from_dict(resume) if isinstance(resume, dict) else None,
            pending_interrupt=pending_interrupt,
            pending_approved_action=PendingApprovedAction.from_dict(ppa) if ppa else None,
            turn_index=int(data.get("turn_index", 0) or 0),
            dispatch_ledger=[
                DispatchLedgerEntry.from_dict(e) for e in (data.get("dispatch_ledger") or [])
            ],
            child_sessions={
                k: dict(v)
                for k, v in (data.get("child_sessions") or {}).items()
                if isinstance(v, dict)
            },
            agent_frames={
                k: AgentFrame.from_dict(v) for k, v in (data.get("agent_frames") or {}).items()
            },
            extra={k: v for k, v in data.items() if k not in known},
        )
