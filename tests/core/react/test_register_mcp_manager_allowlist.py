"""Regression tests for ``ToolRegistry.register_mcp_manager`` allowlist semantics.

Production bug ("MCP stopped working"): the caller in
``enhanced_response_generator`` computed an allowlist as
``{t.name for t in mcp_tools}`` and passed it to ``register_mcp_manager``. When
``mcp_tools`` was empty (e.g. a cached-DB-name vs live-tool-name drift filtered
everything out), an EMPTY SET was passed. The old guard
``if allowed_names is not None and tool.name not in allowed_names`` treats an
empty set as "restrict to nothing", so EVERY discovered tool was skipped — the
MCP server was connected but advertised zero tools to the LLM.

Fix: a falsy ``allowed_names`` (None *or* empty set) means "no restriction →
register all". Only a non-empty set actually restricts.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from miiflow_agent.core.tools.mcp.mcp_tool import MCPTool
from miiflow_agent.core.tools.registry import ToolRegistry


def _make_stub_mcp_tool(server_name: str, tool_name: str) -> MCPTool:
    mcp_definition = SimpleNamespace(
        name=tool_name,
        description=f"Stub {tool_name}",
        inputSchema={"type": "object", "properties": {}},
    )
    return MCPTool(mcp_definition, MagicMock(), server_name)


def _make_manager(*tools: MCPTool):
    manager = MagicMock()
    manager.get_all_tools.return_value = list(tools)
    return manager


def test_empty_allowlist_registers_all_tools():
    """An EMPTY allowlist must NOT black out the server (the prod regression)."""
    registry = ToolRegistry()
    t1 = _make_stub_mcp_tool("Supabase", "execute_sql")
    t2 = _make_stub_mcp_tool("Supabase", "list_tables")
    manager = _make_manager(t1, t2)

    registry.register_mcp_manager(manager, allowed_names=set())

    assert t1.name in registry.mcp_tools
    assert t2.name in registry.mcp_tools, (
        "Empty allowlist wiped all MCP tools — connected-but-invisible regression"
    )


def test_none_allowlist_registers_all_tools():
    """``None`` keeps the documented 'register everything' behavior."""
    registry = ToolRegistry()
    t1 = _make_stub_mcp_tool("Supabase", "execute_sql")
    manager = _make_manager(t1)

    registry.register_mcp_manager(manager, allowed_names=None)

    assert t1.name in registry.mcp_tools


def test_nonempty_allowlist_still_restricts():
    """A genuine selection must still filter to only the chosen tools."""
    registry = ToolRegistry()
    keep = _make_stub_mcp_tool("Supabase", "execute_sql")
    drop = _make_stub_mcp_tool("Supabase", "list_tables")
    manager = _make_manager(keep, drop)

    registry.register_mcp_manager(manager, allowed_names={keep.name})

    assert keep.name in registry.mcp_tools
    assert drop.name not in registry.mcp_tools
