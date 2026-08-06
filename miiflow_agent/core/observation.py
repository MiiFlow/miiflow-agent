"""Canonical tool-observation port — one stored record per tool execution.

The orchestrator invokes an adapter-supplied ``ObservationSink`` at the moment a
tool observation is finalized (serial step, parallel batch, deterministic
approval-resume). The sink persists the observation once and returns an opaque
``ref``; every other surface (execution timeline, dispatch ledger, SSE,
sub-agent traces) carries ``{excerpt, observation_ref}`` instead of a second
full copy.

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
* OPTIONAL ``llm_excerpt(text, tool_name, ref) -> str`` bounds what the model
  sees (see ``bound_observation_for_llm``). Adapters that store a truncated
  copy MUST implement it so their store cap and their context cap agree —
  otherwise the ref cannot serve back what the model was already shown.

Why the LLM-facing string is bounded (production incident, 2026-07-31):
a single ``google_ads_query`` returned 2,349,866 chars and the very next
request was 1,110,727 tokens against a 1,000,000-token ceiling — a hard 400
that killed the run at step 8. Every *derived* surface was already bounded;
the live ``context.messages`` append was the one path with no ceiling, so the
model routinely saw more than the store retained. Depth on demand goes through
``read_observation``, never through an unbounded inline paste.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

OBSERVATION_SINK_DEPS_KEY = "observation_sink"

# Framework fallback ceiling on the observation string handed to the model,
# used when no sink is wired or the sink declares no policy of its own. Chosen
# to be generous — this is a blast-radius guard against pathological tool
# output, not a tuning knob for normal analysis. Adapters with a store cap
# should override via ``llm_excerpt`` and keep the two numbers equal.
LLM_OBSERVATION_MAX_CHARS = 200_000

# No ref to point at (no sink wired / write failed), so the guidance is
# "re-run narrower" rather than "read the rest" — telling the model to follow
# a ref that cannot resolve is worse than telling it the data is gone.
_FALLBACK_TRUNCATION_MARKER = (
    "\n…[truncated {omitted} chars to fit the context window. Re-run the tool "
    "with a narrower scope — fewer rows, a shorter date range, or fewer "
    "fields — if you need the rest.]"
)

# A ref exists (the sink stored the full output) but the sink declares no
# excerpt policy of its own: the framework bound applies, and the marker
# points at the stored copy instead of telling the model to re-run the tool
# and pay for data that is already persisted.
_FALLBACK_TRUNCATION_MARKER_WITH_REF = (
    "\n…[truncated {omitted} chars to fit the context window. The full output "
    'is stored — call read_observation(ref="{ref}") to fetch it.]'
)


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
class RecordedObservation:
    """What the orchestrator gets back from finalizing one tool observation.

    ``observation`` — NOT the caller's original string — is what may be shown
    to the model: it is the bounded form produced by
    ``bound_observation_for_llm``. Returning the two together is deliberate;
    the recording seam is the only place that knows the ref, and a caller that
    took the ref while keeping its own unbounded text is exactly the bug this
    type exists to make unwritable.
    """

    ref: Optional[str]
    observation: str


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


def bound_observation_for_llm(
    sink: Optional[ObservationSink],
    text: Optional[str],
    *,
    tool_name: Optional[str],
    ref: Optional[str],
) -> str:
    """Return the observation string that may enter ``context.messages``.

    Delegates to the sink's ``llm_excerpt`` when it declares one (the adapter
    owns the marker wording and the cap that agrees with its store), and falls
    back to the framework ceiling otherwise. A sink whose ``llm_excerpt``
    raises or returns a non-string is treated as declaring no policy — same
    fail-soft rule as ``record``/``fetch``: an adapter defect must degrade the
    bound, never fail the run.
    """
    if not text:
        return text or ""

    excerpt = getattr(sink, "llm_excerpt", None) if sink is not None else None
    if callable(excerpt):
        try:
            bounded = excerpt(text=text, tool_name=tool_name, ref=ref)
            if isinstance(bounded, str):
                return bounded
            logger.debug(
                "observation sink llm_excerpt returned %s, not str; "
                "falling back to the framework bound",
                type(bounded).__name__,
            )
        except Exception:  # noqa: BLE001 — never fail the run on adapter policy
            logger.debug("observation sink llm_excerpt failed", exc_info=True)

    if len(text) <= LLM_OBSERVATION_MAX_CHARS:
        return text
    omitted = len(text) - LLM_OBSERVATION_MAX_CHARS
    if ref:
        return text[:LLM_OBSERVATION_MAX_CHARS] + (
            _FALLBACK_TRUNCATION_MARKER_WITH_REF.format(omitted=omitted, ref=ref)
        )
    return text[:LLM_OBSERVATION_MAX_CHARS] + _FALLBACK_TRUNCATION_MARKER.format(
        omitted=omitted
    )
