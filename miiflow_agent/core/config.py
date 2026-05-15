"""AgentConfig — a single dataclass that captures everything you need to
construct an `Agent`, including `sub_agents`.

This is the **canonical** way to construct an Agent going forward. The
legacy `Agent(client, tools=..., agent_type=..., ...)` kwargs constructor
still works (and internally builds an AgentConfig), but the dataclass form
is what callers should reach for when they have more than two parameters
to pass — particularly when wiring sub-agents.

Why a dataclass and not a config builder / dict / pydantic model:

  - Static field list is auditable. Adding a knob is a one-line dataclass
    edit, not "go grep the constructor and the four places that read kwargs".
  - Default values are explicit. The legacy constructor defaults are
    preserved here verbatim so `AgentConfig(client=c)` produces the same
    agent as `Agent(c)`.
  - No runtime dependency on pydantic / validators. The framework already
    has its own schema layer (`ToolSchema`); we don't need a second one
    for agent construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)

if TYPE_CHECKING:
    from .agent import AgentType, RunContext
    from .client import LLMClient
    from .subagent import SubAgent
    from .tools import FunctionTool


@dataclass
class AgentConfig:
    """Everything needed to construct an `Agent`.

    Required:
        client: LLM client this agent talks to.

    Behavior:
        agent_type: Orchestration mode (single-hop / ReAct / plan-and-
            execute / parallel-plan / multi-agent). Default SINGLE_HOP.
        system_prompt: String OR callable that receives a RunContext and
            returns a string. The callable form lets the prompt vary by
            turn (e.g., inject memory, current time).
        retries: How many full-run retries before propagating the error.
            Default 1 — most agents shouldn't retry blind; if you want
            recovery, plug in a RecoveryManager via the ReAct path.
        max_iterations: Maximum ReAct steps (also caps replans for
            plan-and-execute via ``max_iterations // 5``). Default 10.
        temperature: Sampling temperature passed to the LLM. Default 0.7.
        max_tokens: Optional cap on completion tokens. ``None`` = provider
            default.
        json_schema: Optional response schema to force structured output
            in single-hop mode.

    Capabilities:
        tools: FunctionTool instances registered with the agent's tool
            registry at construction time.
        sub_agents: SubAgent instances this agent can dispatch to. The
            framework synthesizes a `dispatch_assistant`-shaped tool
            whose `handle` enum covers every sub-agent's handle and
            routes dispatcher calls through the framework's dispatch
            lifecycle (depth, cycle, budget, event bubbling).

    Memory & context:
        context_compression: When True (and `agent_type` is a multi-step
            mode), enables on-the-fly transcript compression once the
            context budget is exceeded. Default True.
        max_context_tokens: Token budget for the compressor. ``None`` =
            provider's default model window minus a safety margin.

    The constructor keeps signatures aligned with Agent.__init__: anything
    you can pass to `Agent(client, **kwargs)` you can pass to
    `AgentConfig(client=..., **kwargs)`, and vice versa.
    """

    client: "LLMClient"
    agent_type: Optional["AgentType"] = None
    system_prompt: Optional[Union[str, Callable[["RunContext"], str]]] = None
    retries: int = 1
    max_iterations: int = 10
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    tools: List["FunctionTool"] = field(default_factory=list)
    sub_agents: List["SubAgent"] = field(default_factory=list)
    json_schema: Optional[Dict[str, Any]] = None
    context_compression: bool = True
    max_context_tokens: Optional[int] = None

    # Identifier the synthesized dispatcher tool uses as
    # ``parent_assistant_id`` in event payloads. When ``None``, the
    # framework substitutes a synthetic ``"framework_agent"`` placeholder.
    # Stage 3's Django adapter sets this to the real Assistant.id so the
    # SSE events the FE consumes carry the right linkage.
    parent_assistant_id: Optional[str] = None

    def __post_init__(self) -> None:
        # Resolve the AgentType default lazily so we don't have to import
        # at module load (would create a circular: agent.py → config.py →
        # agent.py via the AgentType type hint).
        if self.agent_type is None:
            from .agent import AgentType

            self.agent_type = AgentType.SINGLE_HOP

        # Normalize tools: callers may pass either a FunctionTool OR a
        # raw `@tool`-decorated function (the ToolRegistry accepts both,
        # so AgentConfig must mirror that). Decorated functions carry
        # the underlying FunctionTool at ``func._function_tool``.
        normalized: List[Any] = []
        for t in self.tools:
            if hasattr(t, "_function_tool"):
                normalized.append(t._function_tool)
            else:
                normalized.append(t)
        self.tools = normalized

        # No-name-conflict invariant: a sub-agent handle collides with a
        # tool name → the LLM would resolve "dispatch_to_research" before
        # the framework synthesizes the dispatcher. Cheaper to fail at
        # construction than to debug a "wrong tool was called" bug.
        tool_names = {getattr(t, "name", None) for t in self.tools}
        tool_names.discard(None)
        for sub in self.sub_agents:
            if sub.handle in tool_names:
                raise ValueError(
                    f"Sub-agent handle '{sub.handle}' collides with a "
                    f"tool of the same name. Rename one of them."
                )

        handles = [s.handle for s in self.sub_agents]
        if len(handles) != len(set(handles)):
            duplicates = [h for h in handles if handles.count(h) > 1]
            raise ValueError(
                f"Duplicate sub-agent handles: {sorted(set(duplicates))}. "
                f"Each sub-agent must have a unique handle."
            )

    def to_kwargs(self) -> Dict[str, Any]:
        """Render this config as keyword arguments for the legacy
        `Agent(client, **kwargs)` constructor.

        Used by `Agent.__init__` when it receives a config so the body
        of the old constructor doesn't have to fork. ``client`` is
        deliberately excluded — pass it positionally.
        """
        return {
            "agent_type": self.agent_type,
            "system_prompt": self.system_prompt,
            "retries": self.retries,
            "max_iterations": self.max_iterations,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "tools": list(self.tools),
            "json_schema": self.json_schema,
            "context_compression": self.context_compression,
            "max_context_tokens": self.max_context_tokens,
        }

    def with_overrides(self, **overrides: Any) -> "AgentConfig":
        """Return a copy of this config with fields replaced.

        Avoids `dataclasses.replace` import noise at call sites and
        makes copy-with-mutation idiomatic.
        """
        from dataclasses import replace

        return replace(self, **overrides)


__all__ = ["AgentConfig"]
