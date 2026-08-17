"""Native MCP servers ride the connector's `mcp_toolset` — and are deferred.

Production `thread_rQCbwiTgYvPjYSFZTl1DrFbd` (2026-08-17): the root assistant's
turn-1 prompt was 696,009 tokens (cache_read 0, 30.3 s to first token) against
~35K four days earlier. The org had attached a whole Klaviyo MCP server (263
tools, ~1.58 MB of schemas) to the root; on Anthropic we registered it through
the MCP connector under the deprecated `mcp-client-2025-04-04` shape, which
loads EVERY tool of every server into context and offers no way to defer.
Our `defer_loading` path only ever covered local schemas, so the connector's
tools were the one part of the request it could not shield.

Under `mcp-client-2025-11-20` tool selection moves into an `mcp_toolset` entry
in `tools`, and that entry's `default_config.defer_loading` is the ONLY
deferral mechanism for connector tools. These tests pin the wire shape and the
three invariants the surrounding pipeline relies on: the toolset is deferred
exactly when the request carries a tool-search tool, the search tool stays
last (it is the cache-breakpoint target), and the reactive
"tool search unsupported" fallback strips the toolset flags too.
"""

from unittest.mock import AsyncMock, patch

import pytest

from miiflow_agent.core import Message
from miiflow_agent.core.tools.mcp import NativeMCPServerConfig
from miiflow_agent.providers.anthropic_client import AnthropicClient

SEARCH_TOOL = {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"}


def _local(name, **extra):
    return {"name": name, "description": name, "input_schema": {"type": "object"}, **extra}


def _klaviyo(**overrides):
    kwargs = dict(
        name="Klaviyo",
        url="https://mcp.klaviyo.com/mcp",
        authorization_token="tok",
        known_tools=["get_campaigns", "list_flows"],
    )
    kwargs.update(overrides)
    return NativeMCPServerConfig(**kwargs)


def _toolsets(params):
    return [t for t in params["tools"] if t.get("type") == "mcp_toolset"]


# ── NativeMCPServerConfig wire shapes ────────────────────────────────────────


def test_server_entry_carries_connection_details_only():
    entry = _klaviyo(allowed_tools=["get_campaigns"]).to_anthropic_format()
    assert entry == {
        "type": "url",
        "url": "https://mcp.klaviyo.com/mcp",
        "name": "Klaviyo",
        "authorization_token": "tok",
    }
    # The deprecated per-server field is never sent under the new beta.
    assert "tool_configuration" not in entry


def test_toolset_defaults_to_enable_everything_eagerly():
    assert _klaviyo().to_anthropic_toolset() == {
        "type": "mcp_toolset",
        "mcp_server_name": "Klaviyo",
    }


def test_toolset_defers_the_whole_server_when_asked():
    assert _klaviyo().to_anthropic_toolset(defer_loading=True) == {
        "type": "mcp_toolset",
        "mcp_server_name": "Klaviyo",
        "default_config": {"defer_loading": True},
    }


def test_allowed_tools_become_the_documented_allowlist_shape():
    toolset = _klaviyo(allowed_tools=["get_campaigns", "list_flows"]).to_anthropic_toolset(
        defer_loading=True
    )
    assert toolset["default_config"] == {"enabled": False, "defer_loading": True}
    assert toolset["configs"] == {
        "get_campaigns": {"enabled": True},
        "list_flows": {"enabled": True},
    }


def test_legacy_tool_configuration_folds_into_the_toolset():
    cfg = _klaviyo(tool_configuration={"enabled": False, "allowed_tools": ["list_flows"]})
    toolset = cfg.to_anthropic_toolset()
    assert toolset["default_config"] == {"enabled": False}
    assert toolset["configs"] == {"list_flows": {"enabled": True}}
    assert "tool_configuration" not in cfg.to_anthropic_format()


# ── request assembly ─────────────────────────────────────────────────────────


def test_toolset_is_deferred_and_sits_before_the_search_tool():
    params = {
        "tools": [_local("a", defer_loading=True), _local("render_kpi"), dict(SEARCH_TOOL)],
        "betas": ["structured-outputs-2025-11-13"],
    }
    AnthropicClient._apply_native_mcp(params, [_klaviyo()])

    assert params["mcp_servers"] == [_klaviyo().to_anthropic_format()]
    assert _toolsets(params) == [
        {"type": "mcp_toolset", "mcp_server_name": "Klaviyo", "default_config": {"defer_loading": True}}
    ]
    # Search tool stays LAST so the final-tool cache breakpoint lands on it —
    # a deferred entry may not carry cache_control.
    assert params["tools"][-1] == SEARCH_TOOL
    assert params["tools"][-2]["type"] == "mcp_toolset"
    AnthropicClient._apply_prompt_caching(params)
    assert params["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in params["tools"][-2]


def test_without_a_search_tool_the_toolset_loads_eagerly_and_goes_last():
    """No search tool means nothing could discover a deferred tool, so the
    pre-existing eager behaviour is kept — and the (non-deferred) toolset can
    safely be the cache-breakpoint target."""
    params = {"tools": [_local("a")]}
    AnthropicClient._apply_native_mcp(params, [_klaviyo()])
    assert params["tools"][-1] == {"type": "mcp_toolset", "mcp_server_name": "Klaviyo"}
    AnthropicClient._apply_prompt_caching(params)
    assert params["tools"][-1]["cache_control"] == {"type": "ephemeral"}


def test_native_mcp_with_no_local_tools_still_sends_the_toolset():
    params = {}
    AnthropicClient._apply_native_mcp(params, [_klaviyo()])
    assert params["tools"] == [{"type": "mcp_toolset", "mcp_server_name": "Klaviyo"}]


def test_beta_header_is_the_current_revision_and_added_once():
    params = {"tools": [_local("a")], "betas": ["structured-outputs-2025-11-13"]}
    AnthropicClient._apply_native_mcp(params, [_klaviyo()])
    AnthropicClient._apply_native_mcp(params, [_klaviyo(name="Other", url="https://x/mcp")])
    assert params["betas"].count("mcp-client-2025-11-20") == 1
    assert "mcp-client-2025-04-04" not in params["betas"]
    assert "structured-outputs-2025-11-13" in params["betas"]


def test_every_server_gets_exactly_one_toolset():
    """The API rejects a server with zero or two toolsets."""
    servers = [_klaviyo(), _klaviyo(name="GitHub", url="https://api.githubcopilot.com/mcp/")]
    params = {"tools": [dict(SEARCH_TOOL)]}
    AnthropicClient._apply_native_mcp(params, servers)
    assert [t["mcp_server_name"] for t in _toolsets(params)] == ["Klaviyo", "GitHub"]
    assert [s["name"] for s in params["mcp_servers"]] == ["Klaviyo", "GitHub"]


# ── the reactive "tool search unsupported" fallback ──────────────────────────


def test_disable_tool_deferral_strips_toolset_flags_too():
    """Dropping the search tool while leaving `defer_loading` on the toolset
    would leave those tools unreachable for the whole retry."""
    params = {"tools": [_local("a", defer_loading=True), dict(SEARCH_TOOL)]}
    AnthropicClient._apply_native_mcp(
        params, [_klaviyo(allowed_tools=["get_campaigns"])]
    )
    assert AnthropicClient._disable_tool_deferral(params) is True
    assert params["tools"] == [
        _local("a"),
        {
            "type": "mcp_toolset",
            "mcp_server_name": "Klaviyo",
            "default_config": {"enabled": False},
            "configs": {"get_campaigns": {"enabled": True}},
        },
    ]
    assert AnthropicClient._disable_tool_deferral(params) is False


def test_disable_tool_deferral_drops_an_empty_default_config():
    params = {"tools": [dict(SEARCH_TOOL)]}
    AnthropicClient._apply_native_mcp(params, [_klaviyo()])
    AnthropicClient._disable_tool_deferral(params)
    assert params["tools"] == [{"type": "mcp_toolset", "mcp_server_name": "Klaviyo"}]


# ── stale tool_reference pruning knows about connector tools ─────────────────


def _search_pair(*tool_names, use_id="srvtoolu_9"):
    return [
        {"type": "server_tool_use", "id": use_id, "name": "tool_search_tool_regex",
         "input": {"query": "klaviyo"}},
        {"type": "tool_search_tool_result", "tool_use_id": use_id,
         "content": {"type": "tool_search_tool_search_result",
                     "tool_references": [
                         {"type": "tool_reference", "tool_name": n} for n in tool_names
                     ]}},
    ]


def test_prune_keeps_references_to_tools_a_present_server_is_known_to_serve():
    """A deferred connector tool the model discovered is replayed as a
    tool_reference like any local one; pruning it because it is not in
    `tools` would erase the discovery on every later turn. The API spells the
    reference "<server name>_<tool>" (verbatim, unsanitised — observed live);
    the bare name is accepted too."""
    pair = _search_pair("Klaviyo_get_campaigns", "list_flows", "google_ads_query")
    params = {
        "tools": [_local("google_ads_query"), dict(SEARCH_TOOL)],
        "messages": [{"role": "assistant", "content": pair}],
    }
    AnthropicClient._apply_native_mcp(params, [_klaviyo()])
    assert AnthropicClient._prune_stale_tool_references(params, [_klaviyo()]) == 0
    assert params["messages"][0]["content"] is pair


def test_prune_matches_the_servers_name_verbatim():
    server = _klaviyo(name="Deep Wiki-X", known_tools=["read_wiki_structure"])
    pair = _search_pair("Deep Wiki-X_read_wiki_structure")
    params = {"tools": [dict(SEARCH_TOOL)], "messages": [{"role": "assistant", "content": pair}]}
    assert AnthropicClient._prune_stale_tool_references(params, [server]) == 0


def test_prune_still_drops_a_reference_no_present_server_claims():
    """Detaching the server (or an unknown spelling) prunes — the safe
    direction: one extra search beats the 400 the pruner exists to prevent."""
    pair = _search_pair("get_campaigns")
    params = {
        "tools": [_local("google_ads_query")],
        "messages": [{"role": "assistant", "content": pair}],
    }
    assert AnthropicClient._prune_stale_tool_references(params, None) == 1
    assert params["messages"][0]["content"] == [{"type": "text", "text": "[no content]"}]


# ── end to end through achat ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_achat_sends_toolset_and_new_beta(mock_anthropic_response):
    client = AnthropicClient(model="claude-sonnet-4-5", api_key="test-key")
    with patch.object(client.client.beta.messages, "create", new_callable=AsyncMock) as create:
        create.return_value = mock_anthropic_response
        await client.achat(
            [Message.user("hi")],
            tools=[_local("a", defer_loading=True), dict(SEARCH_TOOL)],
            mcp_servers=[_klaviyo()],
        )
    kwargs = create.call_args.kwargs
    assert "mcp-client-2025-11-20" in kwargs["betas"]
    assert "mcp-client-2025-04-04" not in kwargs["betas"]
    assert kwargs["mcp_servers"] == [_klaviyo().to_anthropic_format()]
    types = [t.get("type") for t in kwargs["tools"]]
    assert types.index("mcp_toolset") < types.index(SEARCH_TOOL["type"])
    toolset = next(t for t in kwargs["tools"] if t.get("type") == "mcp_toolset")
    assert toolset["default_config"] == {"defer_loading": True}
