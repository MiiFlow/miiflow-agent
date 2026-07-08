"""Canonical tool-observation port — one stored record per tool execution.

The orchestrator invokes an adapter-supplied ``ObservationSink`` at the moment a
tool observation is finalized (serial step, parallel batch, deterministic
approval-resume). The sink persists the full LLM-facing observation once and
returns an opaque ``ref``; every other surface (execution timeline, dispatch
ledger, SSE, sub-agent traces) carries ``{excerpt, observation_ref}`` instead of
a second full copy.

Contract notes for adapters:

* The sink is injected via ``RunContext.deps["observation_sink"]``. Absent sink
  ⇒ every call site degrades to prior behavior (no ref, nothing stored).
* ``record``/``fetch`` MUST never raise into the run loop — swallow and return
  ``None``. They are awaited inline from the orchestrator (never fire-and-forget,
  never from the callback registry: callbacks don't carry ``RunContext`` and
  fire-and-forget writes are lost on event-loop teardown in worker contexts).
* ``fetch`` MUST enforce the adapter's tenancy boundary (an org-scoped guard):
  refs travel across agents in ledgers and prompts, so serving is where leaks
  would happen.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

OBSERVATION_SINK_DEPS_KEY = "observation_sink"


@dataclass
class ObservationRecord:
    """Everything known about one finalized tool execution."""

    tool_name: str
    tool_call_id: Optional[str]
    inputs: Dict[str, Any]
    observation_text: str
    raw_output: Any = None
    success: bool = True
    error: Optional[str] = None
    execution_time_ms: Optional[int] = None
    step_number: int = 0
    produced_by_path: List[str] = field(default_factory=list)
    source: str = "react"  # "react" | "resume"


@dataclass
class StoredObservation:
    """A previously persisted observation served back by ``fetch``."""

    ref: str
    observation_text: str
    tool_name: str
    success: bool
    created_at_ts: float
    truncated: bool = False


@runtime_checkable
class ObservationSink(Protocol):
    async def record(self, rec: ObservationRecord) -> Optional[str]:
        """Persist one observation; return an opaque ref, or None on failure."""
        ...

    async def fetch(self, ref: str) -> Optional[StoredObservation]:
        """Load a stored observation by ref; None if missing or out of scope."""
        ...


def get_observation_sink(context: Any) -> Optional[ObservationSink]:
    """Resolve the sink from ``context.deps``; None when not wired."""
    deps = getattr(context, "deps", None)
    if not isinstance(deps, dict):
        return None
    sink = deps.get(OBSERVATION_SINK_DEPS_KEY)
    if sink is None:
        return None
    if not (hasattr(sink, "record") and hasattr(sink, "fetch")):
        return None
    return sink
