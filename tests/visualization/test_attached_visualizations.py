"""Editable cards can accompany domain data without erasing IDs or versions."""

from unittest.mock import MagicMock

import pytest

from miiflow_agent import Agent, RunContext
from miiflow_agent.core.react.enums import ReActEventType
from miiflow_agent.core.react.execution import ExecutionState
from miiflow_agent.core.react.factory import ReActFactory
from miiflow_agent.visualization import VisualizationResult


def _handler():
    agent = MagicMock(spec=Agent)
    agent.client = MagicMock(provider_name="openai")
    agent.tool_registry = MagicMock()
    agent.tool_registry.list_tools.return_value = []
    agent._tools = []
    orch = ReActFactory.create_orchestrator(agent=agent, max_steps=5, max_budget=None, max_time_seconds=None)
    events = []
    orch.event_bus.subscribe(lambda event: events.append(event))
    return orch, events


@pytest.mark.asyncio
async def test_attached_cards_emit_events_and_preserve_authoring_data():
    orch, events = _handler()
    card = VisualizationResult(type="custom_editor", data={"document_id": "doc_1"}).to_dict()
    output = {"document_id": "doc_1", "version": 7, "__visualizations__": [card]}
    observation, blocks = await orch._tool_actions.process_result(
        output, ExecutionState(current_step=1), "create_document", RunContext(deps={}, messages=[]),
    )
    rendered = [event for event in events if event.event_type == ReActEventType.VISUALIZATION]
    assert len(rendered) == 1
    assert rendered[0].data["visualization"] == card
    assert "'document_id': 'doc_1'" in observation
    assert "'version': 7" in observation
    assert f"[VIZ:{card['id']}]" in observation
    assert "__visualizations__" not in observation
    assert "__visualizations__" in output
    assert blocks == []


@pytest.mark.asyncio
async def test_ordinary_results_keep_their_existing_observation():
    orch, events = _handler()
    output = {"document_id": "doc_1", "version": 7}
    observation, blocks = await orch._tool_actions.process_result(
        output, ExecutionState(current_step=1), "get_document", RunContext(deps={}, messages=[]),
    )
    assert observation == str(output)
    assert blocks == []
    assert events == []


@pytest.mark.asyncio
async def test_invalid_attached_cards_do_not_erase_domain_data():
    orch, events = _handler()
    observation, _ = await orch._tool_actions.process_result(
        {"version": 7, "__visualizations__": [None, "junk", {"id": "missing-marker"}]},
        ExecutionState(current_step=1), "get_document", RunContext(deps={}, messages=[]),
    )
    assert observation == str({"version": 7})
    assert events == []
