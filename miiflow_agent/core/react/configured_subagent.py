"""Adapter: ``DynamicSubAgentConfig`` + child ``Agent`` → ``SubAgent``.

The framework already ships everything around dispatch — the
``SubAgent`` Protocol (``core/subagent.py``), the lifecycle guardrails
and event forwarding (``core/react/dispatch.py``), and the
``dispatch_assistant`` FunctionTool factory
(``make_subagent_dispatcher_tool``). What it has *not* shipped is a
concrete way to turn one of the registry's ``DynamicSubAgentConfig``
entries into a runnable child. Django's ``AssistantInvoker`` is the only
implementation today; everyone else gets ``NotImplementedError`` from
``SubAgentDescriptor.stream``.

``ConfiguredSubAgent`` closes that gap. It wraps a config plus a
caller-built ``Agent`` instance, delegates ``stream()`` to the child
agent's own streaming orchestrator, and accumulates the
``FINAL_ANSWER_CHUNK`` deltas into the ``SubAgentResult`` that
``final_result()`` returns.

Typical wiring:

    from miiflow_agent import Agent, AgentType, LLMClient
    from miiflow_agent.core.react import (
        ConfiguredSubAgent,
        DynamicSubAgentConfig,
        make_registry_dispatcher_tool,
        SubAgentRegistry,
    )

    registry = SubAgentRegistry()  # ships empty after the default-template removal
    registry.register(DynamicSubAgentConfig(
        name="explorer",
        description="Find relevant files",
        system_prompt="You are a codebase exploration specialist...",
        tools=["file_read"],
        max_steps=5,
    ))

    client = LLMClient.create("anthropic", model="claude-sonnet-4-5-20251029")

    def child_factory(config):
        return Agent(
            client=client,
            agent_type=AgentType.REACT,
            system_prompt=config.system_prompt,
            tools=[file_read_tool],  # caller picks the matching tool instances
            max_iterations=config.max_steps,
        )

    dispatcher_tool = make_registry_dispatcher_tool(
        registry,
        child_agent_factory=child_factory,
        parent_assistant_id="my_parent_agent",
    )
    parent_agent.add_tool(dispatcher_tool)

The caller owns tool resolution and model selection because the SDK
doesn't know the consumer's tool registry layout or provider
preferences. Keeping that decision on the caller side preserves
``miiflow-agent``'s lean, BYO-everything posture.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Dict, List, Optional

from ..subagent import (
    ClarificationPolicy,
    ForwardWindow,
    SubAgentHandoff,
    SubAgentResult,
)
from .enums import ReActEventType


if TYPE_CHECKING:
    from ..agent import Agent, RunContext
    from ..tools.function.function_tool import FunctionTool
    from .dispatch import DispatchCounter
    from .react_events import ReActEvent
    from .subagent_registry import DynamicSubAgentConfig, SubAgentRegistry


logger = logging.getLogger(__name__)


# Hook for constructing the child's ``RunContext.deps``. Receives the
# ``SubAgentHandoff`` so callers can route ``structured_payload`` or the
# parent user message into the child's deps as they see fit.
DepsFactory = Callable[[SubAgentHandoff], Any]

# Caller-supplied function that builds a fresh child ``Agent`` for a
# given config. Called once per registered config when the dispatcher
# tool is assembled — the resulting Agent is then reused across every
# dispatch to that handle within the same parent.
ChildAgentFactory = Callable[["DynamicSubAgentConfig"], "Agent"]


def _default_deps_factory(handoff: SubAgentHandoff) -> dict:
    """Empty-dict deps with the structured payload threaded through.

    Callers that want richer deps (org id, thread id, …) override this
    when constructing the ConfiguredSubAgent. The default surfaces the
    handoff's structured payload under a stable key so tools running
    inside the child can find it without needing a custom factory.
    """
    return {"handoff_payload": handoff.structured_payload or {}}


def _compose_query(handoff: SubAgentHandoff) -> str:
    """Combine task + intent_summary + parent_user_message into one prompt.

    The child agent's orchestrator only takes a single ``query`` string,
    so the structured handoff has to collapse to text. We keep the task
    in the foreground (it's the self-contained subtask) and tack on the
    parent's intent + last user message as bracketed context blocks so
    the child can see the why without losing the what.
    """
    parts: List[str] = [handoff.task.strip()]
    if handoff.intent_summary:
        parts.append("")
        parts.append(f"[Parent intent: {handoff.intent_summary.strip()}]")
    if handoff.parent_user_message:
        parts.append("")
        parts.append(
            "[Most recent user message to the parent:]\n"
            f"{handoff.parent_user_message.strip()}"
        )
    return "\n".join(parts)


def _summarize_failure(failure: Dict[str, Any]) -> str:
    """Build a one-line error summary from a structured failure payload.

    Used as ``SubAgentResult.error`` so callers that only inspect ``error``
    still see something meaningful — the dispatch envelope carries the
    full ``failure`` dict alongside.
    """
    stop_reason = failure.get("stop_reason") or "stopped"
    last_tool = failure.get("last_tool")
    last_tool_error = failure.get("last_tool_error") or ""
    if last_tool:
        # Trim the error to keep this single-line.
        snippet = last_tool_error.splitlines()[0][:200] if last_tool_error else ""
        if snippet:
            return f"{stop_reason}: {last_tool} failed with: {snippet}"
        return f"{stop_reason}: {last_tool} failed"
    return f"{stop_reason}: {failure.get('description') or 'no answer produced'}"


class ConfiguredSubAgent:
    """SubAgent that runs a pre-built child ``Agent`` per dispatch.

    Satisfies the ``SubAgent`` Protocol via duck typing — we don't
    inherit from ``SubAgentDescriptor`` because it's a dataclass and
    field redeclaration in a subclass with custom ``__init__`` is more
    friction than it's worth here.

    The child agent is constructed by the caller (via the
    ``ChildAgentFactory`` passed to ``make_registry_dispatcher_tool``)
    and shared across all dispatches to this handle. If a caller needs
    per-dispatch fresh agents (e.g. to reset internal state), they
    construct a new ConfiguredSubAgent per parent turn rather than
    reusing one.
    """

    def __init__(
        self,
        config: "DynamicSubAgentConfig",
        child_agent: "Agent",
        *,
        when_to_use: Optional[str] = None,
        forward_transcript_window: ForwardWindow = ForwardWindow.LAST_USER,
        clarification_policy: ClarificationPolicy = ClarificationPolicy.ALWAYS_BUBBLE,
        auto_approve_child_tools: bool = False,
        deps_factory: Optional[DepsFactory] = None,
    ) -> None:
        self._config = config
        self._child_agent = child_agent
        self._deps_factory: DepsFactory = deps_factory or _default_deps_factory
        self._last_result: Optional[SubAgentResult] = None

        # SubAgent Protocol fields, populated from the config so the
        # dispatcher tool description can render them in the LLM-facing
        # tool schema.
        self.handle: str = config.name
        self.name: str = config.name
        self.description: str = config.description
        self.schema = None  # use default dispatch_assistant schema
        self.when_to_use: str = (
            when_to_use
            if when_to_use is not None
            else f"Delegate to '{config.name}' when: {config.description}"
        )
        self.handoff_schema = config.output_schema  # output_schema reused as handoff hint
        self.forward_transcript_window = forward_transcript_window
        self.clarification_policy = clarification_policy
        self.auto_approve_child_tools = auto_approve_child_tools

    async def stream(
        self, handoff: SubAgentHandoff
    ) -> AsyncIterator["ReActEvent"]:
        """Run the child agent and yield its ReAct events.

        The parent's ``dispatch_subagent`` lifecycle wraps this with
        guardrails (depth, cycle, budget) and forwards a subset of the
        yielded events onto the parent's bus — see
        ``forward_subagent_events`` in dispatch.py for which event types
        bubble up. We yield everything the child produces; the lifecycle
        decides what to forward.

        We accumulate ``FINAL_ANSWER_CHUNK`` deltas locally so
        ``final_result()`` can return a populated ``SubAgentResult``
        once the stream is exhausted, matching the contract the test
        suite's ``_FakeSubAgent`` and the Django ``AssistantInvoker``
        both implement.
        """
        # Late import to avoid the agent/config/subagent circular at
        # module load (matches the late-import in dispatch.dispatch_subagent).
        from ..agent import RunContext

        start = time.monotonic()
        answer_parts: List[str] = []
        captured_failure: Optional[Dict[str, Any]] = None

        context = RunContext(
            deps=self._deps_factory(handoff),
            messages=[],
        )
        query = _compose_query(handoff)

        # We let exceptions propagate up to ``dispatch_subagent``, which
        # already synthesizes a failed ``SubAgentResult`` and emits the
        # ``subagent_dispatch/failed`` event. Setting ``_last_result``
        # in a ``finally`` is belt-and-suspenders for callers that
        # invoke ``stream()`` directly without going through dispatch.
        try:
            async for event in self._child_agent.stream(
                query=query,
                context=context,
                max_steps=self._config.max_steps,
                max_time_seconds=self._config.timeout_seconds,
            ):
                event_type = getattr(event, "event_type", None)
                if event_type == ReActEventType.FINAL_ANSWER_CHUNK:
                    data = getattr(event, "data", {}) or {}
                    delta = (
                        data.get("delta")
                        or data.get("chunk")
                        or data.get("content")
                        or ""
                    )
                    if delta:
                        answer_parts.append(delta)
                elif event_type == ReActEventType.STOP_CONDITION:
                    # Carries structured failure info when the child loop
                    # halted via a safety condition (e.g. repeated tool
                    # errors). Without this capture, the parent only sees
                    # the canned "I ran into repeated issues" fallback
                    # answer and has nothing actionable to report.
                    data = getattr(event, "data", {}) or {}
                    failure = data.get("failure")
                    if isinstance(failure, dict):
                        captured_failure = failure
                yield event
        finally:
            duration_ms = int((time.monotonic() - start) * 1000)
            answer = "".join(answer_parts)
            error: Optional[str]
            if captured_failure:
                # Build a one-line summary from the structured payload —
                # callers that only look at ``error`` (e.g. legacy logging
                # paths) still see a real cause rather than ``None``.
                error = _summarize_failure(captured_failure)
            elif answer:
                error = None
            else:
                error = "child produced no final answer"
            self._last_result = SubAgentResult(
                answer=answer,
                # If the stream completed without ever producing a final-
                # answer chunk, treat it as a failure so the parent gets
                # a non-empty error rather than a silent empty answer.
                status="completed" if answer else "failed",
                duration_ms=duration_ms,
                error=error,
                failure=captured_failure,
            )

    def final_result(self) -> SubAgentResult:
        """Return the captured result from the most recent ``stream()`` run.

        Returns a synthetic failed result if ``stream()`` was never
        consumed — this can only happen when callers misuse the
        Protocol, but raising would surface as an exception in the
        parent's tool observation rather than a structured error.
        """
        if self._last_result is None:
            return SubAgentResult(
                answer="",
                status="failed",
                error="stream() was never consumed before final_result()",
            )
        return self._last_result


def configured_subagents_from_registry(
    registry: "SubAgentRegistry",
    child_agent_factory: ChildAgentFactory,
    *,
    deps_factory: Optional[DepsFactory] = None,
) -> List[ConfiguredSubAgent]:
    """Wrap every config in ``registry`` as a ``ConfiguredSubAgent``.

    Order follows ``registry.get_all()`` (priority-descending). The
    ``child_agent_factory`` is called once per config — failures
    propagate, since a misconfigured child is a setup-time error and
    silently dropping a handle would surprise the caller later when
    the model tries to dispatch to it.
    """
    return [
        ConfiguredSubAgent(
            config=config,
            child_agent=child_agent_factory(config),
            deps_factory=deps_factory,
        )
        for config in registry.get_all()
    ]


def make_registry_dispatcher_tool(
    registry: "SubAgentRegistry",
    child_agent_factory: ChildAgentFactory,
    *,
    parent_assistant_id: str,
    counter: Optional["DispatchCounter"] = None,
    deps_factory: Optional[DepsFactory] = None,
) -> "FunctionTool":
    """End-to-end convenience: registry + factory → ``dispatch_assistant`` tool.

    Equivalent to:

        sub_agents = configured_subagents_from_registry(registry, factory)
        return make_subagent_dispatcher_tool(
            sub_agents,
            parent_assistant_id=parent_assistant_id,
            counter=counter,
        )

    Use this when you want every config in the registry exposed as a
    dispatch handle. When you need finer control (skip some handles,
    apply per-handle policy overrides), build the
    ``ConfiguredSubAgent`` list yourself and call
    ``make_subagent_dispatcher_tool`` directly.
    """
    from .dispatch import make_subagent_dispatcher_tool

    sub_agents = configured_subagents_from_registry(
        registry,
        child_agent_factory,
        deps_factory=deps_factory,
    )
    return make_subagent_dispatcher_tool(
        sub_agents,
        parent_assistant_id=parent_assistant_id,
        counter=counter,
    )


__all__ = [
    "ChildAgentFactory",
    "ConfiguredSubAgent",
    "DepsFactory",
    "configured_subagents_from_registry",
    "make_registry_dispatcher_tool",
]
