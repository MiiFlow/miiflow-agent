"""Regression test for tool_search → MCP tool surfacing into LLM schemas.

Pre-fix bug: ``AgentToolExecutor._build_native_tool_schemas()`` looked up each
candidate name only in ``registry.tools`` (FunctionTools), so MCP and HTTP tools
were silently dropped from the schema list sent to the LLM. A ``tool_search``
call would advertise an MCP tool as available, but the next turn's schema list
excluded it — leaving the model to either hallucinate a call or invent a
text-format pseudo-call.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from miiflow_agent.core.react.tool_executor import AgentToolExecutor
from miiflow_agent.core.tools.mcp.mcp_tool import MCPTool
from miiflow_agent.core.tools.registry import ToolRegistry
from miiflow_agent.core.tools.tool_search import mark_tools_enabled, tool_search_session


def _make_stub_mcp_tool(server_name: str, tool_name: str, description: str) -> MCPTool:
    """Build an MCPTool without standing up a real server connection.

    Only schema/lookup paths are exercised here; execute() is not called.
    """
    mcp_definition = SimpleNamespace(
        name=tool_name,
        description=description,
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "SQL to run."},
            },
            "required": ["query"],
        },
    )
    connection = MagicMock()  # unused for schema building
    return MCPTool(mcp_definition, connection, server_name)


def _make_mock_agent_with_registry(registry: ToolRegistry, provider_name: str = "openai"):
    """Mirror the pattern in test_orchestrator.TestNativeToolCallingMode.

    Defaults to a non-Anthropic provider so the in-process meta-tool gating
    path is exercised (undiscovered tools are omitted from the schema list).
    Anthropic uses first-party server-side tool search, which sends the full
    list with ``defer_loading`` instead of hiding — covered separately by
    ``test_unenabled_mcp_tool_deferred_under_native_search``.
    """
    mock_model_client = MagicMock()
    mock_model_client.provider_name = provider_name
    # Identity formatter — keeps the universal schema shape so assertions
    # can read top-level "name" directly.
    mock_model_client.convert_schema_to_provider_format = MagicMock(
        side_effect=lambda x: x
    )

    mock_llm_client = MagicMock()
    mock_llm_client.client = mock_model_client
    mock_llm_client._client = mock_model_client
    mock_llm_client.tool_registry = registry

    agent = MagicMock()
    agent.client = mock_llm_client
    agent.tool_registry = registry
    agent.temperature = 0.7
    agent.max_tokens = None
    agent._tools = []
    return agent


def test_mcp_tool_enabled_via_tool_search_appears_in_native_schemas():
    """An MCP tool surfaced by tool_search must reach the LLM's tool schemas.

    Repro for the silent-drop bug at
    ``tool_executor._build_native_tool_schemas`` where ``registry.tools.get``
    skipped every name not in the FunctionTool dict.
    """
    registry = ToolRegistry(tool_search_enabled=True, tool_search_threshold=0)
    mcp_tool = _make_stub_mcp_tool(
        server_name="Supabase",
        tool_name="execute_sql",
        description="Executes raw SQL in the Postgres database.",
    )
    registry.register_mcp_tool(mcp_tool)
    namespaced = mcp_tool.name  # "Supabase__execute_sql"
    assert namespaced in registry.mcp_tools

    agent = _make_mock_agent_with_registry(registry)
    executor = AgentToolExecutor(agent)

    with tool_search_session():
        # Simulates what tool_search.mark_tools_enabled does after the model
        # invokes the meta-tool and matches the MCP tool by keyword.
        mark_tools_enabled([namespaced])
        schemas = executor._build_native_tool_schemas()

    surfaced_names = {
        s.get("name") or (s.get("function") or {}).get("name")
        for s in schemas
    }
    assert namespaced in surfaced_names, (
        f"MCP tool '{namespaced}' was enabled via tool_search but did not "
        f"appear in _build_native_tool_schemas output. Surfaced: {surfaced_names}"
    )


def test_unenabled_mcp_tool_stays_hidden_under_tool_search():
    """Counter-test: tool_search gating still works after the fix.

    Without ``mark_tools_enabled``, the MCP tool must NOT appear in the
    schemas — only the meta-tool plus any ``always_load`` entries should.
    """
    registry = ToolRegistry(tool_search_enabled=True, tool_search_threshold=0)
    mcp_tool = _make_stub_mcp_tool(
        server_name="Supabase",
        tool_name="execute_sql",
        description="Executes raw SQL in the Postgres database.",
    )
    registry.register_mcp_tool(mcp_tool)

    agent = _make_mock_agent_with_registry(registry)
    executor = AgentToolExecutor(agent)

    with tool_search_session():
        schemas = executor._build_native_tool_schemas()

    surfaced_names = {
        s.get("name") or (s.get("function") or {}).get("name")
        for s in schemas
    }
    assert mcp_tool.name not in surfaced_names, (
        "MCP tool should stay hidden until tool_search surfaces it; got "
        f"{surfaced_names}"
    )


def test_unenabled_mcp_tool_deferred_under_native_search():
    """Anthropic native search hides tools via ``defer_loading``, not omission.

    Unlike the in-process meta-tool path, the first-party server-side search
    (``provider_name == "anthropic"``) sends the FULL tool list every turn so
    the cache prefix stays stable, marking undiscovered tools ``defer_loading``
    so the API strips them from the prompt. The MCP tool must therefore be
    present in the schema list but flagged deferred until surfaced.
    """
    registry = ToolRegistry(tool_search_enabled=True, tool_search_threshold=0)
    mcp_tool = _make_stub_mcp_tool(
        server_name="Supabase",
        tool_name="execute_sql",
        description="Executes raw SQL in the Postgres database.",
    )
    registry.register_mcp_tool(mcp_tool)

    agent = _make_mock_agent_with_registry(registry, provider_name="anthropic")
    executor = AgentToolExecutor(agent)

    with tool_search_session():
        schemas = executor._build_native_tool_schemas()

    mcp_schema = next(
        (s for s in schemas if s.get("name") == mcp_tool.name), None
    )
    assert mcp_schema is not None, (
        "Native search must keep the MCP tool in the schema list (deferred), "
        f"not drop it; got {[s.get('name') for s in schemas]}"
    )
    assert mcp_schema.get("defer_loading") is True, (
        "Undiscovered MCP tool must be marked defer_loading under native "
        f"search; got {mcp_schema!r}"
    )
