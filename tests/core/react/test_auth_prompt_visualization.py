"""Tests for auth_prompt visualization in the unified ReAct loop.

Verifies that when a tool returns a dict with __visualization__: True (as
happens when the PRE_TOOL_USE auth callback blocks a tool), the
VISUALIZATION event is emitted by the ReAct orchestrator. Plan & Execute
and Multi-Agent forwarding tests were removed alongside their
orchestrators in the unified-ReAct migration.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from miiflow_agent import Agent, RunContext
from miiflow_agent.core.react.enums import ReActEventType
from miiflow_agent.core.react.events import EventBus, EventFactory
from miiflow_agent.core.react.execution import ExecutionState
from miiflow_agent.core.react.factory import ReActFactory
from miiflow_agent.core.react.models import ReActStep
from miiflow_agent.core.react.react_events import ReActEvent
from miiflow_agent.core.tools.schemas import ToolResult
from miiflow_agent.visualization.types import (
    extract_visualization_data,
    is_visualization_result,
)


# --- Shared test data ---

AUTH_PROMPT_VIZ_DICT = {
    "__visualization__": True,
    "id": "test-viz-id",
    "type": "auth_prompt",
    "title": "Connect your account",
    "data": {
        "provider": "google_ads",
        "providerName": "Google Ads",
        "providerLogo": "https://example.com/logo.png",
        "reason": "Authentication is required to use Google Ads.",
        "serviceProviderId": "sp_123",
        "authMethods": [{"id": "am_1", "name": "OAuth", "authType": "oauth2"}],
    },
    "config": {},
}

AUTH_BLOCK_REASON_JSON = (
    '{"auth_required": true, "provider": {'
    '"serviceProviderId": "sp_123", "name": "Google Ads", '
    '"logo": "https://example.com/logo.png", '
    '"authMethods": [{"id": "am_1", "name": "OAuth", "authType": "oauth2"}]}}'
)


# --- Fixtures ---


@pytest.fixture
def mock_agent():
    """Create a mock agent with one tool registered."""
    mock_client = MagicMock()
    mock_client.provider_name = "openai"
    agent = MagicMock(spec=Agent)
    agent.client = mock_client
    agent.tool_registry = MagicMock()
    agent.tool_registry.list_tools.return_value = ["test_tool"]
    agent._tools = []
    return agent


@pytest.fixture
def auth_blocked_tool_result():
    """ToolResult simulating a tool blocked by auth callback."""
    return ToolResult(
        name="test_tool",
        input={},
        output=AUTH_PROMPT_VIZ_DICT,
        success=True,
    )


@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
def collected_events(event_bus):
    """Subscribe to event bus and collect all events."""
    events = []
    event_bus.subscribe(lambda e: events.append(e))
    return events


# --- Test: is_visualization_result detects auth_prompt dict ---


class TestVisualizationDetection:
    """Test that is_visualization_result detects auth_prompt dicts."""

    def test_detects_auth_prompt_dict(self):
        assert is_visualization_result(AUTH_PROMPT_VIZ_DICT) is True

    def test_extracts_auth_prompt_data(self):
        data = extract_visualization_data(AUTH_PROMPT_VIZ_DICT)
        assert data is not None
        assert data["type"] == "auth_prompt"
        assert data["data"]["providerName"] == "Google Ads"

    def test_rejects_plain_blocked_dict(self):
        blocked_dict = {"error": "Tool blocked", "blocked": True}
        assert is_visualization_result(blocked_dict) is False

    def test_rejects_string(self):
        assert is_visualization_result("not a viz") is False

    def test_rejects_none(self):
        assert is_visualization_result(None) is False


# --- Test: ReAct orchestrator emits VISUALIZATION event ---


class TestReActAuthPromptVisualization:
    """Test that ReAct orchestrator emits VISUALIZATION event for auth_prompt."""

    @pytest.mark.asyncio
    async def test_handle_tool_action_emits_visualization_event(
        self, mock_agent, auth_blocked_tool_result
    ):
        """When tool returns __visualization__ dict, orchestrator should emit
        a VISUALIZATION event and set observation to [VIZ:id] marker."""
        orchestrator = ReActFactory.create_orchestrator(
            agent=mock_agent,
            max_steps=5,
            max_budget=None,
            max_time_seconds=None,
        )

        # Collect events from the orchestrator's event bus
        events = []
        orchestrator.event_bus.subscribe(lambda e: events.append(e))

        # Mock tool executor to return the auth-blocked result
        orchestrator.tool_executor.execute_tool = AsyncMock(
            return_value=auth_blocked_tool_result
        )
        orchestrator.tool_executor.has_tool = MagicMock(return_value=True)
        orchestrator.tool_executor.tool_needs_context = MagicMock(return_value=False)

        # Create step and context
        step = ReActStep(
            step_number=1,
            thought="I need to use this tool",
            action="test_tool",
            action_input={"query": "test"},
        )
        context = RunContext(deps=None, messages=[])
        state = ExecutionState()
        state.current_step = 1

        # Execute tool action
        await orchestrator._handle_tool_action(
            step, context, state, tool_call_id="call_123"
        )

        # Filter for VISUALIZATION events
        viz_events = [
            e
            for e in events
            if isinstance(e, ReActEvent)
            and e.event_type == ReActEventType.VISUALIZATION
        ]

        assert len(viz_events) == 1, (
            f"Expected 1 VISUALIZATION event, got {len(viz_events)}. "
            f"All events: {[e.event_type for e in events if isinstance(e, ReActEvent)]}"
        )

        viz_data = viz_events[0].data.get("visualization")
        assert viz_data is not None
        assert viz_data["type"] == "auth_prompt"
        assert viz_data["data"]["providerName"] == "Google Ads"

        # Observation should be the VIZ marker with auth block context
        assert step.observation == (
            "[VIZ:test-viz-id] Tool was blocked: authentication required for Google Ads. "
            "A connect prompt has been shown to the user. "
            "Do not proceed with this provider's tools until the user connects their account."
        )

    @pytest.mark.asyncio
    async def test_non_viz_tool_result_does_not_emit_visualization(
        self, mock_agent
    ):
        """Normal tool results should NOT emit VISUALIZATION events."""
        orchestrator = ReActFactory.create_orchestrator(
            agent=mock_agent,
            max_steps=5,
            max_budget=None,
            max_time_seconds=None,
        )

        events = []
        orchestrator.event_bus.subscribe(lambda e: events.append(e))

        normal_result = ToolResult(
            name="test_tool",
            input={},
            output="Some normal result",
            success=True,
        )
        orchestrator.tool_executor.execute_tool = AsyncMock(
            return_value=normal_result
        )
        orchestrator.tool_executor.has_tool = MagicMock(return_value=True)
        orchestrator.tool_executor.tool_needs_context = MagicMock(return_value=False)

        step = ReActStep(
            step_number=1,
            thought="Using tool",
            action="test_tool",
            action_input={},
        )
        context = RunContext(deps=None, messages=[])
        state = ExecutionState()
        state.current_step = 1

        await orchestrator._handle_tool_action(
            step, context, state, tool_call_id="call_456"
        )

        viz_events = [
            e
            for e in events
            if isinstance(e, ReActEvent)
            and e.event_type == ReActEventType.VISUALIZATION
        ]
        assert len(viz_events) == 0


class TestEventFactoryVisualization:
    """Test EventFactory.visualization for auth_prompt data."""

    def test_creates_visualization_event(self):
        event = EventFactory.visualization(
            step_number=1,
            viz_data=AUTH_PROMPT_VIZ_DICT,
            action="test_tool",
        )

        assert event.event_type == ReActEventType.VISUALIZATION
        assert event.step_number == 1
        assert event.data["visualization"]["type"] == "auth_prompt"
        assert event.data["action"] == "test_tool"
        assert event.data["visualization"]["__visualization__"] is True

    def test_visualization_event_preserves_auth_methods(self):
        event = EventFactory.visualization(
            step_number=1,
            viz_data=AUTH_PROMPT_VIZ_DICT,
            action="test_tool",
        )

        auth_methods = event.data["visualization"]["data"]["authMethods"]
        assert len(auth_methods) == 1
        assert auth_methods[0]["authType"] == "oauth2"
