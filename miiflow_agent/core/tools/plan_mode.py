"""Plan-mode deferred tools.

Two tools that together let the model self-escalate into a planning mode
where every side-effectful tool is automatically refused until it
commits to a plan. Mirrors Claude Code's ``EnterPlanMode`` /
``ExitPlanMode`` pattern: there is no structured Plan data class — the
"plan" is text the model passes to ``exit_plan_mode``.

- ``enter_plan_mode(reasoning)``: flips ``ctx.run_state.permission_mode``
  to ``"plan"`` and surfaces ``exit_plan_mode`` via the ToolSearch
  session so the model can discover it on the next turn even if its
  registry didn't always-load it.
- ``exit_plan_mode(plan)``: flips ``permission_mode`` back to ``"default"``
  and emits ``PLAN_MODE_EXITED`` on the event bus carrying the proposed
  plan text. The streaming service can intercept that event to pause
  for user approval (same pattern as ``TOOL_APPROVAL_NEEDED``).

Both tools register with ``is_read_only=True`` so they remain callable
while plan mode is active.

Registration:

    from miiflow_agent.core.tools.plan_mode import register_plan_mode_tools

    register_plan_mode_tools(agent.tool_registry)

The Agent constructor wires this automatically when
``AgentConfig.enable_plan_mode=True`` (or when the legacy kwargs
constructor receives ``enable_plan_mode=True``).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .function import FunctionTool
from .schemas import ParameterSchema, ToolSchema
from .types import ParameterType, ToolType


logger = logging.getLogger(__name__)


ENTER_PLAN_MODE_TOOL_NAME = "enter_plan_mode"
EXIT_PLAN_MODE_TOOL_NAME = "exit_plan_mode"


_ENTER_DESCRIPTION = """Enter plan mode.

Call this when the user's request needs investigation, design, or research
BEFORE you start making changes. While in plan mode:
  - You can still call read-only tools (search, list, describe, inspect).
  - Any tool with side effects (write, edit, send, post, dispatch_assistant,
    spawn, etc.) is refused with a "blocked — plan mode active" tool result.
    Don't fight the refusal: gather information, decide on an approach, then
    call `exit_plan_mode` with the plan text.

Use this for: multi-step coding tasks, anything touching shared state,
anything where committing to an approach matters more than speed.

DO NOT use this for: simple lookups, single-tool answers, conversational
replies. Those should answer directly without planning."""


_EXIT_DESCRIPTION = """Exit plan mode with a proposed plan.

Call this when you've finished investigating and have a concrete plan the
user (or the downstream service) should approve before you act. Pass the
plan as a markdown string in `plan`.

After this tool returns, plan mode is cleared and side-effectful tools
become callable again. The streaming service intercepts the
PLAN_MODE_EXITED event the executor emits and may pause the run to
collect user approval before continuing — exactly like the existing
tool-approval flow. From the model's perspective, simply continue the
turn: if you receive a system message indicating rejection, treat its
feedback as new instructions and re-plan."""


async def _enter_plan_mode(ctx, reasoning: str, __description: Optional[str] = None) -> Dict[str, Any]:
    """Flip the run into plan mode and announce the entry."""
    # Mutate run-state permission mode. The executor (AgentToolExecutor)
    # reads this on every subsequent tool call.
    run_state = getattr(ctx, "run_state", None)
    if run_state is not None:
        run_state.permission_mode = "plan"

    # Surface exit_plan_mode in the active ToolSearch session so the
    # model sees it on the next turn even if it's a deferred tool. No-op
    # when no session is active (legacy callers that don't open one).
    try:
        from .tool_search import mark_tools_enabled

        mark_tools_enabled([EXIT_PLAN_MODE_TOOL_NAME])
    except Exception as exc:  # noqa: BLE001 — we never want tool execution to crash on bookkeeping
        logger.debug("enter_plan_mode: mark_tools_enabled skipped: %s", exc)

    # Emit a lifecycle event on the run's event bus when one is wired
    # (streaming runs from the orchestrator) so the UI can render a
    # "plan mode active" indicator. No-op for tests that don't set a bus.
    event_bus = getattr(run_state, "event_bus", None) if run_state is not None else None
    if event_bus is not None:
        try:
            from ..react.enums import ReActEventType
            from ..react.react_events import ReActEvent

            await event_bus.publish(
                ReActEvent(
                    event_type=ReActEventType.PLAN_MODE_ENTERED,
                    step_number=int(getattr(run_state, "step_number", 0) or 0),
                    data={"reasoning": reasoning},
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("enter_plan_mode: event emission skipped: %s", exc)

    return {
        "status": "entered_plan_mode",
        "reasoning": reasoning,
        "message": (
            "Plan mode is active. Only read-only tools (search/list/describe) "
            "will execute. Investigate as needed, then call `exit_plan_mode` "
            "with your proposed plan as markdown."
        ),
    }


async def _exit_plan_mode(ctx, plan: str, __description: Optional[str] = None) -> Dict[str, Any]:
    """Halt the loop and request user approval of the proposed plan.

    This tool does NOT flip ``permission_mode`` back to default itself —
    that flip is the user's decision, made out-of-band. Raises
    ``PlanApprovalRequired`` so the orchestrator can catch it, emit
    ``PLAN_APPROVAL_NEEDED``, persist the pending plan on thread
    metadata, and end the stream. The user's approve/reject decision
    arrives in a subsequent turn (see Django ``views.py`` resume
    handler). On approve, the resume injects a confirmation and
    ``permission_mode`` defaults back to ``"default"``. On reject, the
    resume keeps ``permission_mode = "plan"`` and feeds rejection text
    in as a user message so the model has to re-plan.
    """
    from ..react.exceptions import PlanApprovalRequired

    raise PlanApprovalRequired(plan_text=plan)


def _build_enter_plan_mode_tool() -> FunctionTool:
    schema = ToolSchema(
        name=ENTER_PLAN_MODE_TOOL_NAME,
        description=_ENTER_DESCRIPTION,
        tool_type=ToolType.FUNCTION,
        # The act of flipping a permission flag is read-only itself; the
        # gating it produces only affects *subsequent* tool calls. Keeping
        # is_read_only=True here means the model can still escalate into
        # plan mode even mid-plan-mode (no-op idempotent), which matches
        # Claude Code semantics.
        is_read_only=True,
        # Side-effectful only on the run state — safe to run in parallel
        # alongside read-only sibling calls if the model batches.
        parallelizable=True,
        parameters={
            "reasoning": ParameterSchema(
                name="reasoning",
                type=ParameterType.STRING,
                description=(
                    "One sentence explaining why this turn needs a plan before "
                    "acting. Surfaced in the UI so the user sees what the "
                    "agent is about to investigate."
                ),
                required=True,
            ),
        },
    )
    # Mark the underlying coroutine with the schema so FunctionTool picks
    # it up directly (mirrors the dispatcher tool pattern in dispatch.py).
    _enter_plan_mode._tool_schema = schema  # type: ignore[attr-defined]
    return FunctionTool(_enter_plan_mode)


def _build_exit_plan_mode_tool() -> FunctionTool:
    schema = ToolSchema(
        name=EXIT_PLAN_MODE_TOOL_NAME,
        description=_EXIT_DESCRIPTION,
        tool_type=ToolType.FUNCTION,
        is_read_only=True,
        # Two ``exit_plan_mode`` calls in one turn would be a model bug;
        # keep this serial so the second one observes the first's state
        # flip rather than racing it.
        parallelizable=False,
        parameters={
            "plan": ParameterSchema(
                name="plan",
                type=ParameterType.STRING,
                description=(
                    "The proposed plan as markdown. Make it concrete: what "
                    "files/services/state will change, in what order, and how "
                    "you'll verify each step. The streaming service may pause "
                    "the run to collect user approval on this text."
                ),
                required=True,
            ),
        },
    )
    _exit_plan_mode._tool_schema = schema  # type: ignore[attr-defined]
    return FunctionTool(_exit_plan_mode)


def build_plan_mode_tools() -> list:
    """Return freshly-built (enter, exit) FunctionTool instances.

    Caller is responsible for registering them with a `ToolRegistry`.
    Use `register_plan_mode_tools(registry)` for the common case.
    """
    return [_build_enter_plan_mode_tool(), _build_exit_plan_mode_tool()]


def register_plan_mode_tools(tool_registry) -> list:
    """Register `enter_plan_mode` and `exit_plan_mode` on a ToolRegistry.

    Idempotent: silently skips tools already present (matches Agent's
    sub-agent dispatcher registration pattern).

    Returns the list of FunctionTool instances that were registered (or
    already existed) so callers can mirror them onto `Agent._tools`.
    """
    tools: list = []
    existing = set(tool_registry.tools.keys()) if hasattr(tool_registry, "tools") else set()
    for tool in build_plan_mode_tools():
        if tool.name in existing:
            tools.append(tool_registry.tools[tool.name])
            continue
        tool_registry.register(tool)
        tools.append(tool)
    return tools


__all__ = [
    "ENTER_PLAN_MODE_TOOL_NAME",
    "EXIT_PLAN_MODE_TOOL_NAME",
    "build_plan_mode_tools",
    "register_plan_mode_tools",
]
