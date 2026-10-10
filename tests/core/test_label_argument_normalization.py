"""Names and titles a model hands a tool still HTML-escaped are stored as text.

Production: the agent named a workflow "Cross-Platform Paid Media Monitor &amp;
Optimizer" and a report "Spend Trend &amp; CPA" with no escaped text in its
context, and both entities were saved with the entity.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from miiflow_agent.core.callbacks import (
    CallbackEvent,
    CallbackEventType,
    get_global_registry,
)
from miiflow_agent.core.react.tool_executor import AgentToolExecutor
from miiflow_agent.core.tools import ToolRegistry, ToolResult
from miiflow_agent.core.tools.argument_normalization import normalize_label_arguments
from miiflow_agent.visualization.text_normalization import normalize_label


class TestNormalizeLabel:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            (
                "Cross-Platform Paid Media Monitor &amp; Optimizer",
                "Cross-Platform Paid Media Monitor & Optimizer",
            ),
            ("Spend Trend &amp;amp; CPA", "Spend Trend & CPA"),
            ("Brand &#39;26 &lt;test&gt;", "Brand '26 <test>"),
            ("Performance Overview \\u2014 Q3", "Performance Overview — Q3"),
        ],
    )
    def test_decodes_escaped_labels(self, raw, expected):
        assert normalize_label(raw) == expected

    @pytest.mark.parametrize(
        "label", ["R&D budget", "AT&T", "Q&A &copy2024", "Fish & Chips &unknown;", ""]
    )
    def test_leaves_plain_labels_alone(self, label):
        assert normalize_label(label) == label

    def test_is_idempotent(self):
        once = normalize_label("A &amp;amp;amp; B")
        assert normalize_label(once) == once == "A & B"


class TestNormalizeLabelArguments:
    def test_repairs_top_level_labels_and_column_names(self):
        args = {
            "name": "Leads &amp; Deals",
            "display_name": "Ops &amp; Pacing",
            "columns": [{"name": "Spend &amp; CPA", "key": "spend_cpa"}],
        }
        assert normalize_label_arguments(args) == {
            "name": "Leads & Deals",
            "display_name": "Ops & Pacing",
            "columns": [{"name": "Spend & CPA", "key": "spend_cpa"}],
        }

    def test_leaves_content_lookups_user_data_and_json_strings_untouched(self):
        args = {
            "html_content": "<p>Tom &amp; Jerry</p>",
            "path": "/reports/A &amp; B.md",
            "confirm_name": "A &amp; B",
            "name_contains": "&amp;",
            "node_config": '[{"data": {"name": "A &amp; B"}}]',
            # A row's "name" cell is the user's data, and upsert matches on it.
            "rows": [{"name": "Tom &amp; Jerry"}],
            "inputs": {"title": "A &amp; B"},
        }
        assert normalize_label_arguments(args) == args

    def test_does_not_mutate_the_callers_dict(self):
        args = {"name": "A &amp; B", "columns": [{"name": "C &amp; D"}]}
        normalize_label_arguments(args)
        assert args == {"name": "A &amp; B", "columns": [{"name": "C &amp; D"}]}

    def test_non_dict_arguments_pass_through(self):
        assert normalize_label_arguments(None) is None


class TestExecutorRepairsLabels:
    async def test_tool_and_tool_executed_event_see_plain_text(self):
        registry = ToolRegistry()
        agent = MagicMock()
        agent.client.tool_registry = registry
        agent.tool_registry = registry
        registry.execute_safe = AsyncMock(
            return_value=ToolResult(
                name="create_workflow", input={}, output="ok", success=True
            )
        )
        events: list[CallbackEvent] = []

        async def capture(event: CallbackEvent):
            events.append(event)

        callbacks = get_global_registry()
        callbacks.register(CallbackEventType.TOOL_EXECUTED, capture)
        try:
            await AgentToolExecutor(agent).execute_tool(
                "create_workflow", {"name": "Monitor &amp; Optimizer", "preview": False}
            )
        finally:
            callbacks.unregister(CallbackEventType.TOOL_EXECUTED, capture)

        registry.execute_safe.assert_awaited_once_with(
            "create_workflow", name="Monitor & Optimizer", preview=False
        )
        assert events[0].tool_inputs == {
            "name": "Monitor & Optimizer",
            "preview": False,
        }
