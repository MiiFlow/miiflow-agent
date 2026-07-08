"""Read-through dedupe gate: serve identical reads from the tree blackboard.

The gate sits in front of tool execution. When a tool that has DECLARED a
serve contract (``ToolSchema.idempotency_class != "none"``) is called with
inputs identical to a ledger entry that is fresh (per-class TTL), valid, and
backed by a stored observation, the gate serves the stored observation text
instead of re-executing — deterministic dedupe replacing the old
prompt-guidance approach ("Do NOT call list_all_ad_accounts again").

Safety properties (each one closes a verified failure mode):

* **Opt-in only.** ``is_read_only`` is NOT sufficient: time-relative queries
  (``DURING LAST_30_DAYS``) hash identically across days, and observations
  carrying run-scoped ``_data_id`` render refs poison a different RunContext.
  Tools declare a class; TTLs are code-owned per class.
* **Validity predicate.** Business errors returned as successful payloads
  (``{"error": "account not found"}``) and payloads with ``_data_id`` refs
  are never served.
* **Never dispatches.** ``dispatch`` ledger entries are excluded — specialist
  output recording and child-thread linkage depend on real execution.
* **Single-flight.** Concurrent identical calls (parallel siblings in one
  gather batch) coalesce onto one executor; waiters share the leader's
  result, including failures (the model retries next step). This CLOSES the
  sibling duplicate-read race in-process.
* **Scope dims.** Tools whose result depends on more than their inputs fold
  the declared deps keys into the dedupe key. Org scoping is implicit: the
  ledger lives on one org's root thread and the observation store org-guards
  serving.
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional

from ..checkpoint import stable_json_hash
from ..observation import ObservationSink

logger = logging.getLogger(__name__)

# TTLs are per-class constants owned by code — not per-tool configuration.
IDEMPOTENCY_TTLS_SECONDS: Dict[str, float] = {
    # Structural discovery (account lists, campaign catalogs): stable for hours.
    "discovery": 3600.0,
    # Metrics reads restate intraday; serve only within the freshness window
    # of a single turn's work.
    "performance_read": 300.0,
}

# Inputs keys injected by the framework that must not affect identity.
_IDENTITY_EXCLUDED_INPUT_KEYS = {"__description", "__reasoning"}


def dedupe_identity(
    tool_name: str,
    inputs: Optional[Dict[str, Any]],
    *,
    scope_dims: Optional[list] = None,
    deps: Optional[Dict[str, Any]] = None,
) -> str:
    """Stable identity hash for a tool call, matching the ledger's inputs_hash
    plus any declared scope dimensions."""
    clean_inputs = {
        k: v
        for k, v in (inputs or {}).items()
        if k not in _IDENTITY_EXCLUDED_INPUT_KEYS
    }
    scope = {}
    for dim in scope_dims or []:
        scope[dim] = (deps or {}).get(dim)
    if scope:
        return stable_json_hash({"inputs": clean_inputs, "scope": scope})
    return stable_json_hash(clean_inputs)


def observation_is_servable(text: str) -> bool:
    """Reject stored observations that are unsafe to serve into a new context."""
    if not text:
        return False
    # Run-scoped render refs resolve only inside the RunContext that made
    # them; serving one hands the consumer a dangling pointer.
    if "_data_id" in text:
        return False
    # Business errors recorded as transport-successes: platform tools return
    # {"error": ...} payloads with success=True. Cheap structural check on
    # the head of the payload.
    head = text[:200].lstrip()
    for marker in ('{"error"', "{'error'", '{"errors"', "{'errors'"):
        if head.startswith(marker):
            return False
    return True


class LedgerDedupeGate:
    """Per-root-turn gate instance, shared by reference across the dispatch tree."""

    def __init__(self, root_checkpoint: Any, sink: Optional[ObservationSink]):
        self._checkpoint = root_checkpoint
        self._sink = sink
        self._inflight: Dict[str, asyncio.Future] = {}

    # ── Serving ────────────────────────────────────────────────────────────

    async def try_serve(
        self,
        tool_name: str,
        inputs: Optional[Dict[str, Any]],
        schema: Any,
        deps: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Return a served result dict for an identical fresh read, else None.

        The returned value is ``{"output": str, "observation_ref": str}``;
        the executor wraps it into its ToolResult shape.
        """
        idem_class = getattr(schema, "idempotency_class", "none") or "none"
        ttl = IDEMPOTENCY_TTLS_SECONDS.get(idem_class)
        if ttl is None or self._checkpoint is None or self._sink is None:
            return None

        identity = dedupe_identity(
            tool_name,
            inputs,
            scope_dims=getattr(schema, "dedupe_scope_dims", None),
            deps=deps,
        )
        entry = self._checkpoint.ledger_lookup(f"tool_call::{tool_name}::{identity}")
        if entry is None or entry.kind != "tool_call":
            return None
        if not entry.success or not entry.observation_ref:
            return None
        if (time.time() - (entry.produced_at or 0)) > ttl:
            return None

        stored = await self._sink.fetch(entry.observation_ref)
        if stored is None or not stored.success:
            return None
        if not observation_is_servable(stored.observation_text):
            return None

        logger.info(
            "[DEDUPE] served '%s' from ledger (ref=%s, age=%.0fs)",
            tool_name,
            entry.observation_ref,
            time.time() - (entry.produced_at or 0),
        )
        return {
            "output": stored.observation_text,
            "observation_ref": stored.ref,
        }

    # ── Single-flight ──────────────────────────────────────────────────────

    def inflight_key(
        self,
        tool_name: str,
        inputs: Optional[Dict[str, Any]],
        schema: Any,
        deps: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Identity key for single-flight coalescing, or None when the tool
        has no serve contract (no coalescing for non-idempotent tools)."""
        idem_class = getattr(schema, "idempotency_class", "none") or "none"
        if idem_class not in IDEMPOTENCY_TTLS_SECONDS:
            return None
        identity = dedupe_identity(
            tool_name,
            inputs,
            scope_dims=getattr(schema, "dedupe_scope_dims", None),
            deps=deps,
        )
        return f"{tool_name}::{identity}"

    def claim(self, key: str) -> Optional[asyncio.Future]:
        """Synchronous check-and-set (atomic on the event loop): returns None
        when the caller becomes the leader (a future is registered for it),
        or the leader's future to await when one is already in flight."""
        existing = self._inflight.get(key)
        if existing is not None:
            return existing
        self._inflight[key] = asyncio.get_event_loop().create_future()
        return None

    def resolve(self, key: str, result: Any) -> None:
        fut = self._inflight.pop(key, None)
        if fut is not None and not fut.done():
            fut.set_result(result)

    def resolve_error(self, key: str, error: BaseException) -> None:
        fut = self._inflight.pop(key, None)
        if fut is not None and not fut.done():
            fut.set_exception(error)


def get_dedupe_gate(context: Any) -> Optional[LedgerDedupeGate]:
    """Resolve the per-turn gate from ``context.deps``; None when not wired."""
    deps = getattr(context, "deps", None)
    if not isinstance(deps, dict):
        return None
    gate = deps.get("dedupe_gate")
    return gate if isinstance(gate, LedgerDedupeGate) else None
