"""Regression: a tool argument named like a dispatch parameter must not crash.

Some models emit a tool argument literally named ``context`` (or ``tool_name``).
``execute_safe_with_context(tool_name, context, **kwargs)`` used to bind those
positionally AND receive them in ``**kwargs``, raising
``got multiple values for argument 'context'`` at the call boundary — before
the unknown-parameter filter inside could drop them. Making those params
positional-only routes the same-named tool args into ``**kwargs`` instead,
where the existing filter drops them as unknown.
"""

import pytest

from miiflow_agent.core.agent import RunContext
from miiflow_agent.core.tools import ToolRegistry, tool


@tool(description="Search using injected context")
async def search(ctx: RunContext, query: str) -> dict:
    # `ctx` is the first-param context-injection slot; `query` is the only
    # model-facing parameter.
    return {"echoed": query}


@pytest.mark.asyncio
async def test_with_context_tolerates_stray_context_kwarg():
    """A model arg named ``context`` must not collide with the dispatch param."""
    registry = ToolRegistry()
    registry.register(search)

    # Previously raised: TypeError got multiple values for argument 'context'.
    result = await registry.execute_safe_with_context(
        "search", object(), **{"query": "hi", "context": "stray"}
    )

    assert result.success, result.error
    assert result.output == {"echoed": "hi"}


@pytest.mark.asyncio
async def test_with_context_tolerates_stray_tool_name_kwarg():
    """Same for a model arg named ``tool_name`` (the other dispatch param)."""
    registry = ToolRegistry()
    registry.register(search)

    result = await registry.execute_safe_with_context(
        "search", object(), **{"query": "hi", "tool_name": "search"}
    )

    assert result.success, result.error
    assert result.output == {"echoed": "hi"}


@pytest.mark.asyncio
async def test_execute_safe_tolerates_tool_name_kwarg():
    """The no-context path's ``tool_name`` is positional-only too."""

    @tool(description="Plain echo")
    def echo(query: str) -> dict:
        return {"echoed": query}

    registry = ToolRegistry()
    registry.register(echo)

    result = await registry.execute_safe("echo", **{"query": "hi"})
    assert result.success, result.error
    assert result.output == {"echoed": "hi"}
