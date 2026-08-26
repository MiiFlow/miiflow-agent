"""Tests for bridge-model tool disclosure.

The load-bearing property is **the tools array never grows**. That is the
whole reason the bridge exists: the in-process meta-tool enabled discovered
tools, which grew the array on the next iteration, invalidated the tools cache
tier, cascaded to system and messages, and re-billed the entire prefix
uncached — the ~40s parent stall. Everything else here is in service of that.
"""

import pytest

from miiflow_agent.core.tools import ParameterSchema, tool
from miiflow_agent.core.tools.registry import ToolRegistry
from miiflow_agent.core.tools.tool_bridge import (
    BRIDGE_TOOL_NAMES,
    TOOL_CALL_NAME,
    TOOL_DESCRIBE_NAME,
    TOOL_SEARCH_NAME,
    build_catalog_listing,
    unwrap_bridge_call,
)
from miiflow_agent.core.tools.types import ParameterType

pytestmark = pytest.mark.unit


def _make_tool(name: str):
    @tool(
        name=name,
        description=f"Query {name} and return performance rows.",
        parameters={
            "customer_id": ParameterSchema(
                name="customer_id",
                type=ParameterType.STRING,
                description="Account id",
                required=True,
            )
        },
    )
    async def _fn(customer_id: str):
        return {"tool": name, "customer_id": customer_id}

    return _fn


@pytest.fixture
def registry():
    reg = ToolRegistry(tool_bridge_enabled=True)
    for group in ("google_ads", "meta_ads", "tiktok_ads"):
        for i in range(10):
            reg.register(_make_tool(f"{group}_report_{i}"))
    return reg


class TestUnwrap:
    def test_unwraps_nested_arguments(self):
        assert unwrap_bridge_call(
            TOOL_CALL_NAME, {"name": "meta_ads_report_1", "arguments": {"customer_id": "9"}}
        ) == ("meta_ads_report_1", {"customer_id": "9"})

    def test_passes_through_non_bridge_calls(self):
        """Runs on every dispatch — the miss path must be a plain no-op."""
        assert unwrap_bridge_call("google_ads_report_0", {"customer_id": "1"}) is None

    def test_accepts_json_string_arguments(self):
        """Some models emit `arguments` as a JSON string. Recovering it costs
        one parse and saves a wasted turn."""
        assert unwrap_bridge_call(
            TOOL_CALL_NAME, {"name": "x", "arguments": '{"customer_id": "7"}'}
        ) == ("x", {"customer_id": "7"})

    def test_accepts_flattened_arguments(self):
        """Models sometimes flatten arguments into the call itself."""
        assert unwrap_bridge_call(
            TOOL_CALL_NAME, {"name": "x", "customer_id": "7"}
        ) == ("x", {"customer_id": "7"})

    def test_nested_arguments_win_over_flattened(self):
        assert unwrap_bridge_call(
            TOOL_CALL_NAME, {"name": "x", "arguments": {"a": 1}, "a": 2}
        ) == ("x", {"a": 1})

    def test_recursion_guard(self):
        """A confused model can otherwise drive tool_call into itself and burn
        the whole iteration budget on a no-op recursion."""
        for target in BRIDGE_TOOL_NAMES:
            assert unwrap_bridge_call(TOOL_CALL_NAME, {"name": target}) is None

    def test_missing_name_does_not_unwrap(self):
        assert unwrap_bridge_call(TOOL_CALL_NAME, {"arguments": {}}) is None
        assert unwrap_bridge_call(TOOL_CALL_NAME, {"name": "  "}) is None

    def test_malformed_json_arguments_degrade_to_empty(self):
        assert unwrap_bridge_call(
            TOOL_CALL_NAME, {"name": "x", "arguments": "not json"}
        ) == ("x", {})


class TestCatalog:
    def test_full_form_lists_names_and_descriptions(self):
        text, form = build_catalog_listing(
            [("alpha_one", "Does the alpha thing"), ("alpha_two", "Does another")],
            max_tokens=4000,
        )
        assert form == "full"
        assert "alpha_one" in text and "Does the alpha thing" in text

    def test_degrades_rather_than_truncating(self):
        """A listing cut off mid-name is worse than a shorter complete one —
        the model will confidently call the fragment."""
        entries = [(f"g{i}_tool_{j}", "description text " * 5) for i in range(20) for j in range(20)]
        text, form = build_catalog_listing(entries, max_tokens=900)
        assert form in {"names", "mixed", "groups"}
        assert len(text) / 3.5 <= 900

    def test_huge_group_does_not_starve_small_ones(self):
        """Attaching one big MCP server must not make every other tool
        undiscoverable-by-browsing."""
        entries = [("small_a", "x"), ("small_b", "y")]
        entries += [(f"huge_t{i}", "z") for i in range(3000)]
        text, _ = build_catalog_listing(entries, max_tokens=600)
        assert "small_a" in text
        assert "huge" in text  # present as a summary line

    def test_deterministic_under_input_reordering(self):
        """A listing that reorders between assemblies defeats the caching this
        exists to protect."""
        import random

        entries = [(f"t{i}", f"desc {i}") for i in range(30)]
        shuffled = entries[:]
        random.shuffle(shuffled)
        assert build_catalog_listing(entries)[0] == build_catalog_listing(shuffled)[0]

    def test_empty_catalog(self):
        assert build_catalog_listing([]) == ("", "none")


class TestRegistryIntegration:
    def test_bridge_tools_are_off_registry(self):
        """They must not leak into list_tools()/get_schemas() or any code that
        iterates the registry."""
        reg = ToolRegistry(tool_bridge_enabled=True)
        reg.register(_make_tool("alpha_one"))
        reg.get_bridge_tools()
        assert not BRIDGE_TOOL_NAMES & set(reg.list_tools())

    def test_bridge_is_cached_across_calls(self, registry):
        """Re-rendering the catalog every turn would produce a different tools
        array each iteration — the exact invalidation this prevents."""
        assert registry.get_bridge_tools() is registry.get_bridge_tools()

    def test_bridge_rebuilds_when_catalog_changes(self, registry):
        first = registry.get_bridge_tools()
        registry.register(_make_tool("brand_new_tool"))
        second = registry.get_bridge_tools()
        assert second is not first
        assert "brand_new_tool" in second[0].schema.description

    def test_excluded_tools_are_not_in_the_catalog(self, registry):
        """Listing an already-loaded tool as 'available on demand' invites a
        pointless describe/call round-trip."""
        bridge = registry.get_bridge_tools(exclude={"google_ads_report_0"})
        assert "google_ads_report_0:" not in bridge[0].schema.description

    @pytest.mark.asyncio
    async def test_tool_search_does_not_enable_anything(self, registry):
        """Enabling is what grew the array. The bridge must not do it."""
        from miiflow_agent.core.tools.tool_search import (
            get_enabled_tool_names,
            tool_search_session,
        )

        registry.get_bridge_tools()
        with tool_search_session():
            await registry.execute_safe(TOOL_SEARCH_NAME, query="meta ads")
            assert not get_enabled_tool_names()

    @pytest.mark.asyncio
    async def test_tool_describe_returns_full_schema(self, registry):
        registry.get_bridge_tools()
        result = await registry.execute_safe(
            TOOL_DESCRIBE_NAME, name="meta_ads_report_3"
        )
        assert result.success
        assert result.output["name"] == "meta_ads_report_3"
        assert "customer_id" in result.output["parameters"]["properties"]

    @pytest.mark.asyncio
    async def test_tool_describe_suggests_on_typo(self, registry):
        registry.get_bridge_tools()
        result = await registry.execute_safe(
            TOOL_DESCRIBE_NAME, name="meta_ads_reprt_3"
        )
        assert "meta_ads_report_3" in result.output["hint"]

    @pytest.mark.asyncio
    async def test_tool_call_handler_refuses_to_execute(self, registry):
        """The handler is unreachable by design — dispatching there would skip
        approval, context injection and tracing. If it runs, it must fail
        loudly rather than quietly executing on the wrong path."""
        registry.get_bridge_tools()
        result = await registry.execute_safe(
            TOOL_CALL_NAME, name="meta_ads_report_3", arguments={"customer_id": "1"}
        )
        assert "not unwrapped" in str(result.output)


class TestArrayStability:
    """The property the whole design exists for."""

    def test_array_does_not_grow_after_discovery(self, registry):
        from unittest.mock import MagicMock

        from miiflow_agent.core.react.tool_executor import AgentToolExecutor
        from miiflow_agent.core.tools.tool_search import (
            mark_tools_enabled,
            tool_search_session,
        )

        registry.tool_search_threshold = 0
        model_client = MagicMock()
        model_client.provider_name = "openai"
        model_client.model = "gpt-4o"
        model_client.convert_schema_to_provider_format = lambda s: dict(s)

        llm_client = MagicMock()
        llm_client.client = model_client
        llm_client.tool_registry = registry

        agent = MagicMock()
        agent.client = llm_client
        agent.tool_registry = registry
        agent._tools = []
        executor = AgentToolExecutor(agent)

        with tool_search_session():
            before = executor._build_native_tool_schemas()
            # Simulate the model discovering several tools mid-run — under the
            # old meta-tool this is exactly what grew the array.
            mark_tools_enabled(
                ["meta_ads_report_1", "meta_ads_report_2", "google_ads_report_5"]
            )
            after = executor._build_native_tool_schemas()

        names_before = [s.get("name") for s in before]
        names_after = [s.get("name") for s in after]
        assert names_before == names_after, (
            "tools array changed after discovery — this invalidates the tools "
            f"cache tier. before={names_before} after={names_after}"
        )
        assert set(names_after) == BRIDGE_TOOL_NAMES


class TestBridgeToolsAreCallableThroughTheExecutor:
    """Shown-to-the-model and callable must be the same set.

    `_build_native_tool_schemas` put `tool_search` / `tool_describe` /
    `tool_call` in the array and `ToolRegistry.execute_safe` routed them, but
    the executor's PRE-execution checks (`has_tool`, the schema lookup, the
    context decision) only consulted the registry dicts — so the action
    handler rejected the model's very first `tool_search` with
    "Tool 'tool_search' not found". Found on the first live run against
    OpenAI (2026-08-18), the day the bridge became the default; every unit
    test until then stopped at the array or at the registry, never at the
    executor gate in between.
    """

    def _executor(self, registry):
        from unittest.mock import MagicMock

        from miiflow_agent.core.react.tool_executor import AgentToolExecutor

        registry.tool_search_threshold = 0
        model_client = MagicMock()
        model_client.provider_name = "openai"
        model_client.model = "gpt-4o"
        model_client.convert_schema_to_provider_format = lambda s: dict(s)
        llm_client = MagicMock()
        llm_client.client = model_client
        llm_client.tool_registry = registry
        agent = MagicMock()
        agent.client = llm_client
        agent.tool_registry = registry
        agent._tools = []
        executor = AgentToolExecutor(agent)
        # What the loop does before the first model call: this is what builds
        # the bridge tools.
        shown = {s.get("name") for s in executor._build_native_tool_schemas()}
        assert shown == BRIDGE_TOOL_NAMES
        return executor

    def test_every_tool_shown_to_the_model_passes_has_tool(self, registry):
        executor = self._executor(registry)
        for name in BRIDGE_TOOL_NAMES:
            assert executor.has_tool(name), name
            assert executor.get_tool_schema(name).get("name") == name
            assert executor.tool_needs_context(name) is False

    @pytest.mark.asyncio
    async def test_search_describe_call_execute_through_the_executor(self, registry):
        executor = self._executor(registry)

        found = await executor.execute_tool(TOOL_SEARCH_NAME, {"query": "meta ads report 3"})
        assert found.success, found.error
        assert any(m["name"] == "meta_ads_report_3" for m in found.output["results"])

        described = await executor.execute_tool(TOOL_DESCRIBE_NAME, {"name": "meta_ads_report_3"})
        assert described.success, described.error
        assert "customer_id" in str(described.output)

        called = await executor.execute_tool(
            TOOL_CALL_NAME, {"name": "meta_ads_report_3", "arguments": {"customer_id": "9"}}
        )
        assert called.success, called.error
        assert called.output == {"tool": "meta_ads_report_3", "customer_id": "9"}

    def test_a_registry_without_bridge_tools_still_reports_them_absent(self, registry):
        """The check is against the BUILT bridge, not the names — a run that
        never built the bridge (bridge off) must not accept `tool_search`."""
        from unittest.mock import MagicMock

        from miiflow_agent.core.react.tool_executor import AgentToolExecutor

        registry.tool_bridge_enabled = False
        agent = MagicMock()
        agent.tool_registry = registry
        agent.client = MagicMock()
        executor = AgentToolExecutor(agent)
        assert not executor.has_tool(TOOL_SEARCH_NAME)


class TestBridgeCallInjectsContext:
    """A `first_param` tool invoked through `tool_call` must still get ctx.

    The single-tool action path used to decide context injection on the
    pre-unwrap name — `tool_needs_context("tool_call")` is False, so every
    client tool reached through the bridge executed with `context=None` and
    died with "missing 1 required positional argument: 'ctx'" (production,
    from the day the bridge became the default). The context decision has
    exactly one owner: `_execute_tool_inner`, which runs after unwrapping;
    callers pass the context through unconditionally.
    """

    @pytest.mark.asyncio
    async def test_first_param_tool_through_tool_call_receives_ctx(self, registry):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from miiflow_agent.core.react.models import ReActStep
        from miiflow_agent.core.react.tool_actions import ToolActionHandler
        from miiflow_agent.core.react.tool_executor import AgentToolExecutor

        @tool(
            name="ctx_probe",
            description="Echo the org id from the injected context.",
            parameters={
                "label": ParameterSchema(
                    name="label",
                    type=ParameterType.STRING,
                    description="Echoed back",
                    required=True,
                )
            },
        )
        async def _ctx_probe(ctx, label: str):
            return {"label": label, "org": ctx.deps.get("organization_id")}

        registry.register(_ctx_probe)
        registry.tool_search_threshold = 0
        model_client = MagicMock()
        model_client.provider_name = "openai"
        model_client.model = "gpt-4o"
        model_client.convert_schema_to_provider_format = lambda s: dict(s)
        llm_client = MagicMock()
        llm_client.client = model_client
        llm_client.tool_registry = registry
        agent = MagicMock()
        agent.client = llm_client
        agent.tool_registry = registry
        agent._tools = []
        executor = AgentToolExecutor(agent)
        executor._build_native_tool_schemas()  # builds the bridge tools

        orch = MagicMock()
        orch.tool_executor = executor
        handler = ToolActionHandler(orch)

        context = SimpleNamespace(
            run_state=SimpleNamespace(),
            deps={"organization_id": "org_guard"},
        )
        step = ReActStep(
            step_number=1,
            thought="",
            action=TOOL_CALL_NAME,
            action_input={"name": "ctx_probe", "arguments": {"label": "x"}},
        )
        result = await handler.execute_tool(step, context)
        assert result.success, result.error
        assert result.output == {"label": "x", "org": "org_guard"}
