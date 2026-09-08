"""A tool result that ALSO delivers files.

`create_artifact` returns an ArtifactResult and nothing else, so its whole
observation is the `[ARTIFACT:...]` marker. A report tool is different: its
result is data the model needs (`document_id`, `builder_path`, a `file` dict)
AND a file the person should receive. Before the `__artifacts__` side-channel
the tool had no way to say both, so the file reached the model only as a URL —
which it pasted into the answer (a presigned bucket link, production
2026-09-08). These pin the seam: the result stays the observation, one
artifact event is published per entry, and the marker line tells the model the
person already has the file and must not be handed a link.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from miiflow_agent import Agent, RunContext
from miiflow_agent.artifacts import (
    ATTACHED_ARTIFACTS_KEY,
    format_artifact_observation,
    is_file_backed_artifact,
    pop_attached_artifacts,
)
from miiflow_agent.core.react.enums import ReActEventType
from miiflow_agent.core.react.execution import ExecutionState
from miiflow_agent.core.react.factory import ReActFactory


def _file_artifact(**metadata):
    return {
        "__artifact__": True,
        "id": "a-1",
        "kind": "pdf",
        "title": "Weekly check-in",
        "description": "",
        "source_html": "",
        "metadata": {"file_asset_id": "file_asset_1", **metadata},
    }


class TestPopAttachedArtifacts:
    def test_non_dicts_and_dicts_without_the_key_pass_through(self):
        assert pop_attached_artifacts("text") == ("text", [])
        assert pop_attached_artifacts(["x"]) == (["x"], [])
        assert pop_attached_artifacts({"a": 1}) == ({"a": 1}, [])

    def test_the_key_is_removed_and_the_input_is_not_mutated(self):
        result = {"document_id": "rdoc_1", ATTACHED_ARTIFACTS_KEY: [_file_artifact()]}
        rest, attached = pop_attached_artifacts(result)
        assert rest == {"document_id": "rdoc_1"}
        assert [a["id"] for a in attached] == ["a-1"]
        assert ATTACHED_ARTIFACTS_KEY in result  # caller's dict untouched

    def test_entries_that_are_not_artifacts_are_dropped_not_raised(self):
        result = {ATTACHED_ARTIFACTS_KEY: ["junk", {"id": "no-marker"}, _file_artifact()]}
        _, attached = pop_attached_artifacts(result)
        assert len(attached) == 1

    def test_a_non_list_value_is_left_alone(self):
        result = {ATTACHED_ARTIFACTS_KEY: "nope"}
        assert pop_attached_artifacts(result) == (result, [])


class TestFileBackedObservation:
    def test_file_backed_is_decided_by_the_asset_id_alone(self):
        assert is_file_backed_artifact(_file_artifact())
        rendered = {**_file_artifact(), "metadata": {}}
        assert not is_file_backed_artifact(rendered)
        assert not is_file_backed_artifact({"metadata": None})

    def test_file_backed_observation_forbids_links_and_revision(self):
        text = format_artifact_observation(_file_artifact())
        assert text.startswith("[ARTIFACT:a-1] PDF file titled 'Weekly check-in'")
        assert "Do NOT paste, quote, or invent a link" in text
        assert "the person already has it" in text
        # No revision instruction: there is no HTML to edit.
        assert 'edit_artifact("a-1"' not in text
        assert 'get_artifact("a-1"' not in text

    def test_rendered_artifacts_keep_the_revision_instruction(self):
        text = format_artifact_observation({**_file_artifact(), "metadata": {}})
        assert 'get_artifact("a-1")' in text
        assert 'edit_artifact("a-1"' in text


def _handler():
    agent = MagicMock(spec=Agent)
    agent.client = MagicMock(provider_name="openai")
    agent.tool_registry = MagicMock()
    agent.tool_registry.list_tools.return_value = []
    agent._tools = []
    orch = ReActFactory.create_orchestrator(
        agent=agent, max_steps=5, max_budget=None, max_time_seconds=None
    )
    events = []
    orch.event_bus.subscribe(lambda e: events.append(e))
    return orch, events


@pytest.mark.asyncio
async def test_process_result_publishes_each_attached_artifact_and_keeps_the_result():
    orch, events = _handler()
    output = {
        "document_id": "rdoc_1",
        "builder_path": "/adlyse/reports/rdoc_1",
        "file": {"url": "https://app.example/api/files/file_asset_1/content"},
        ATTACHED_ARTIFACTS_KEY: [_file_artifact(), {**_file_artifact(), "id": "a-2", "kind": "csv"}],
    }

    observation, blocks = await orch._tool_actions.process_result(
        output, ExecutionState(current_step=1), "render_report_document",
        RunContext(deps={}, messages=[]),
    )

    artifact_events = [e for e in events if e.event_type == ReActEventType.ARTIFACT]
    assert [e.data["artifact"]["id"] for e in artifact_events] == ["a-1", "a-2"]
    assert all(e.data["action"] == "render_report_document" for e in artifact_events)
    assert blocks == []
    # The model still reads the result — minus the side-channel — and then
    # one marker line per file.
    assert "'document_id': 'rdoc_1'" in observation
    assert ATTACHED_ARTIFACTS_KEY not in observation
    assert "[ARTIFACT:a-1] PDF file" in observation
    assert "[ARTIFACT:a-2] CSV file" in observation
    assert observation.index("rdoc_1") < observation.index("[ARTIFACT:a-1]")


@pytest.mark.asyncio
async def test_a_result_without_the_key_is_unchanged_by_the_seam():
    orch, events = _handler()
    output = {"document_id": "rdoc_1"}

    observation, _ = await orch._tool_actions.process_result(
        output, ExecutionState(current_step=1), "get_report_document",
        RunContext(deps={}, messages=[]),
    )

    assert observation == str(output)
    assert not [e for e in events if e.event_type == ReActEventType.ARTIFACT]
