"""SubAgent primitives — the framework-side contract for delegating a turn
to a specialist agent.

A `SubAgent` is the parent's *view* of a child agent. It bundles:

  - identity (`handle`, `name`, `description`)
  - per-edge policy (`when_to_use`, `handoff_schema`,
    `forward_transcript_window`, `clarification_policy`,
    `auto_approve_child_tools`)
  - a `stream()` coroutine that runs the child given a `SubAgentHandoff`

The same child agent dispatched-to from two different parents produces
two different SubAgent INSTANCES (potentially with different policies).
Per-edge fields live on the SubAgent itself, not on the child Agent.

The framework owns the dispatch *lifecycle* (guardrails, event bubbling,
counter); the SubAgent owns the *side effects* (thread creation, billing,
ORM writes). Stage 3 migrates Django's `AssistantInvoker` to a concrete
`SubAgent` implementation.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:
    from .react.events.bus import EventBus
    from .react.events.types import ReActEvent
    from .tools.schemas import ToolSchema


class ClarificationPolicy(str, enum.Enum):
    """How clarification questions raised by the child should be handled.

    - ``ALWAYS_BUBBLE``: any child clarification is forwarded to the user
      (default for cross-team handoffs).
    - ``BUBBLE_ONLY_BLOCKING``: only clarifications the child marks
      blocking are forwarded; informational ones are silently treated as
      "go with your best guess".
    - ``NEVER_BUBBLE``: child runs to completion without user
      interaction (suitable for fully-automated specialists).
    """

    ALWAYS_BUBBLE = "always_bubble"
    BUBBLE_ONLY_BLOCKING = "bubble_only_blocking"
    NEVER_BUBBLE = "never_bubble"


class ForwardWindow(str, enum.Enum):
    """How much of the parent's transcript to forward to the child.

    - ``NONE``: child sees only the handoff payload.
    - ``LAST_USER``: child sees the most recent user message (default).
    - ``FULL``: child sees the entire parent transcript.
    """

    NONE = "none"
    LAST_USER = "last_user"
    FULL = "full"


@dataclass
class SubAgentHandoff:
    """The parent-to-child handoff payload.

    Constructed by the orchestrator when a dispatcher tool call fires.
    Passed to `SubAgent.stream()`. Carries enough context for the child
    to act without seeing the parent's full conversation.

    Attributes:
        task: Self-contained subtask description. The child sees only
            this — phrase it so a fresh specialist could act on it
            without further context.
        intent_summary: One or two sentences capturing the user's overall
            intent for this turn. Helps the child frame its work even
            when the subtask itself is narrow.
        structured_payload: Optional typed payload (campaign IDs, date
            ranges, prior results) that doesn't fit naturally in prose.
        depth: Nesting depth of this dispatch (root parent = 0,
            first-level child = 1, ...). Used by the dispatch guardrails.
        dispatch_chain: Ordered list of parent identifiers leading down
            to this dispatch. The child's *own* identifier is NOT yet on
            the chain when it receives the handoff; the framework appends
            it before deeper dispatches.
        parent_user_message: Optional copy of the parent's most recent
            user message, included when the SubAgent's
            `forward_transcript_window` is `LAST_USER` or wider.
    """

    task: str
    intent_summary: str
    structured_payload: Optional[Dict[str, Any]] = None
    depth: int = 1
    dispatch_chain: List[str] = field(default_factory=list)
    parent_user_message: Optional[str] = None


@dataclass
class SubAgentResult:
    """What the child returns to the parent after streaming completes.

    Returned by `SubAgent.final_result()` after the `stream()` generator
    is exhausted. The orchestrator passes a serialized form back to the
    parent LLM as the dispatcher tool's observation.

    Attributes:
        answer: The child's final answer to the subtask, or empty
            string on failure.
        status: ``"completed"`` if the child reached a final answer,
            ``"failed"`` if it raised or aborted.
        duration_ms: Wall-clock duration of the child's turn.
        error: Optional error string when ``status == "failed"``.
        child_run_id: Opaque identifier the parent can include in the
            observation so the UI can link to the child's transcript.
            Framework treats this as a string; Django sets it to the
            child Thread.id.
        metadata: Optional free-form bag for child-specific extras
            (token usage, model name, etc.). The framework doesn't
            interpret it.
    """

    answer: str
    status: Literal["completed", "failed"]
    duration_ms: int = 0
    error: Optional[str] = None
    child_run_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SubAgent(Protocol):
    """Protocol every framework sub-agent implements.

    Implementations are typically thin: they hold a reference to the
    underlying child Agent (or a factory for one) plus the per-edge
    policy, then in `stream()` they construct a `RunContext` from the
    handoff and delegate to the child's own orchestrator.

    The framework calls implementations via `dispatch.dispatch_subagent`
    which wraps the stream in the lifecycle guardrails (depth, cycle,
    budget, event bubbling). Implementations themselves don't need to
    enforce those — they just run the child.

    Required attributes (read by the framework):
        handle: LLM-facing slug. Must be a valid identifier-style string
            since the framework renders it into a tool enum.
        name: Human-readable name (shown in the dispatch tool
            description).
        description: One-paragraph description rendered into the
            dispatch tool description.
        schema: Optional custom dispatch tool schema. ``None`` means
            "use the default dispatch_assistant schema with this
            sub-agent's handle on the enum". Custom schemas are how a
            sub-agent advertises typed handoff payloads.
        when_to_use: Short hint rendered into the dispatch tool
            description ("call this when ...").
        handoff_schema: Optional JSON schema for the
            ``structured_payload`` field, used by the dispatch tool's
            parameter description when present.
        forward_transcript_window: How much of the parent transcript
            to include in the handoff.
        clarification_policy: How to handle child clarifications.
        auto_approve_child_tools: When True, tools the child invokes
            that would normally require approval auto-approve under
            this parent (useful when the user has already approved at
            the dispatch level).

    Lifecycle methods:
        stream(handoff): async generator yielding ReActEvent objects
            from the child's run. Must terminate by exhausting the
            generator; the framework calls `final_result()` afterward.
        final_result(): returns the SubAgentResult after stream() has
            been fully consumed.
    """

    handle: str
    name: str
    description: str
    schema: Optional["ToolSchema"]
    when_to_use: str
    handoff_schema: Optional[Dict[str, Any]]
    forward_transcript_window: ForwardWindow
    clarification_policy: ClarificationPolicy
    auto_approve_child_tools: bool

    def stream(self, handoff: SubAgentHandoff) -> AsyncIterator["ReActEvent"]:
        """Run the child agent given the handoff payload.

        Yields ReActEvent objects from the child's orchestrator. The
        framework forwards a subset of these to the parent's event bus
        (FINAL_ANSWER_CHUNK → subagent_dispatch/progress, nested
        SUBAGENT_DISPATCH → re-published with path prefix).
        """
        ...

    def final_result(self) -> SubAgentResult:
        """Return the child's final result. Called after `stream()` is exhausted."""
        ...


@dataclass
class SubAgentDescriptor:
    """Default mutable struct that satisfies the SubAgent protocol's data
    fields but leaves `stream()` / `final_result()` to subclasses.

    Most concrete SubAgents will subclass this rather than implementing
    the Protocol from scratch — saves boilerplate for the eight or so
    attribute fields and gives a stable place to attach helpers.
    """

    handle: str
    name: str
    description: str
    schema: Optional["ToolSchema"] = None
    when_to_use: str = ""
    handoff_schema: Optional[Dict[str, Any]] = None
    forward_transcript_window: ForwardWindow = ForwardWindow.LAST_USER
    clarification_policy: ClarificationPolicy = ClarificationPolicy.ALWAYS_BUBBLE
    auto_approve_child_tools: bool = False

    async def stream(
        self, handoff: SubAgentHandoff
    ) -> AsyncIterator["ReActEvent"]:  # pragma: no cover - abstract
        raise NotImplementedError(
            f"{type(self).__name__}.stream() must be implemented by subclasses"
        )
        # Unreachable: makes mypy / runtime accept this as a generator.
        if False:
            yield  # type: ignore[unreachable]

    def final_result(self) -> SubAgentResult:  # pragma: no cover - abstract
        raise NotImplementedError(
            f"{type(self).__name__}.final_result() must be implemented by subclasses"
        )


__all__ = [
    "ClarificationPolicy",
    "ForwardWindow",
    "SubAgent",
    "SubAgentDescriptor",
    "SubAgentHandoff",
    "SubAgentResult",
]
