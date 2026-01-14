"""Tests for event streaming in ReAct and Plan & Execute orchestrators."""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from dataclasses import dataclass
from typing import List

from miiflow_agent import RunContext
from miiflow_agent.core.react.events import EventBus, EventFactory
from miiflow_agent.core.react.enums import ReActEventType, PlanExecuteEventType, StopReason
from miiflow_agent.core.react.models import ReActStep
from miiflow_agent.core.react.react_events import ReActEvent, PlanExecuteEvent


class TestEventFactory:
    """Test EventFactory for creating ReAct events."""

    def test_step_started_event(self):
        """Test creating step started event."""
        event = EventFactory.step_started(step_number=1)

        assert event.event_type == ReActEventType.STEP_START
        assert event.step_number == 1
        assert event.data["step_number"] == 1

    def test_thought_event(self):
        """Test creating thought event."""
        event = EventFactory.thought(step_number=1, thought="I need to analyze this")

        assert event.event_type == ReActEventType.THOUGHT
        assert event.step_number == 1
        assert event.data["thought"] == "I need to analyze this"

    def test_thinking_chunk_event(self):
        """Test creating thinking chunk event for streaming."""
        event = EventFactory.thinking_chunk(
            step_number=1,
            delta="thinking",
            content="I am thinking"
        )

        assert event.event_type == ReActEventType.THINKING_CHUNK
        assert event.data["delta"] == "thinking"
        assert event.data["content"] == "I am thinking"

    def test_action_planned_event(self):
        """Test creating action planned event."""
        event = EventFactory.action_planned(
            step_number=1,
            action="search",
            action_input={"query": "test"}
        )

        assert event.event_type == ReActEventType.ACTION_PLANNED
        assert event.data["action"] == "search"
        assert event.data["action_input"] == {"query": "test"}

    def test_action_executing_event(self):
        """Test creating action executing event."""
        event = EventFactory.action_executing(
            step_number=1,
            action="search",
            action_input={"query": "test"}
        )

        assert event.event_type == ReActEventType.ACTION_EXECUTING
        assert event.data["status"] == "executing"

    def test_observation_event(self):
        """Test creating observation event."""
        event = EventFactory.observation(
            step_number=1,
            observation="Found 10 results",
            action="search",
            success=True
        )

        assert event.event_type == ReActEventType.OBSERVATION
        assert event.data["observation"] == "Found 10 results"
        assert event.data["success"] is True

    def test_step_complete_event(self):
        """Test creating step complete event."""
        step = ReActStep(
            step_number=1,
            thought="Done thinking",
            action="search",
            execution_time=0.5,
            cost=0.001
        )
        event = EventFactory.step_complete(step_number=1, step=step)

        assert event.event_type == ReActEventType.STEP_COMPLETE
        assert event.data["execution_time"] == 0.5
        assert event.data["cost"] == 0.001

    def test_final_answer_event(self):
        """Test creating final answer event."""
        event = EventFactory.final_answer(
            step_number=1,
            answer="The answer is 42"
        )

        assert event.event_type == ReActEventType.FINAL_ANSWER
        assert event.data["answer"] == "The answer is 42"

    def test_final_answer_chunk_event(self):
        """Test creating final answer chunk event for streaming."""
        event = EventFactory.final_answer_chunk(
            step_number=1,
            delta="42",
            content="The answer is 42"
        )

        assert event.event_type == ReActEventType.FINAL_ANSWER_CHUNK
        assert event.data["delta"] == "42"
        assert event.data["content"] == "The answer is 42"

    def test_error_event(self):
        """Test creating error event."""
        event = EventFactory.error(
            step_number=1,
            error="Tool execution failed",
            error_type="tool_error"
        )

        assert event.event_type == ReActEventType.ERROR
        assert event.data["error"] == "Tool execution failed"
        assert event.data["error_type"] == "tool_error"

    def test_stop_condition_event(self):
        """Test creating stop condition event."""
        event = EventFactory.stop_condition(
            step_number=3,
            stop_reason="max_steps",
            description="Maximum steps reached"
        )

        assert event.event_type == ReActEventType.STOP_CONDITION
        assert event.data["stop_reason"] == "max_steps"


class TestEventBusStreaming:
    """Test EventBus for real-time event streaming."""

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self):
        """Test multiple subscribers receive events."""
        bus = EventBus()
        events1 = []
        events2 = []

        def handler1(event):
            events1.append(event)

        def handler2(event):
            events2.append(event)

        bus.subscribe(handler1)
        bus.subscribe(handler2)

        event = EventFactory.step_started(step_number=1)
        await bus.publish(event)

        assert len(events1) == 1
        assert len(events2) == 1

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        """Test unsubscribing from events."""
        bus = EventBus()
        events = []

        def handler(event):
            events.append(event)

        bus.subscribe(handler)
        await bus.publish(EventFactory.step_started(1))
        assert len(events) == 1

        bus.unsubscribe(handler)
        await bus.publish(EventFactory.step_started(2))
        assert len(events) == 1  # No new events

    @pytest.mark.asyncio
    async def test_event_buffer_limit(self):
        """Test event buffer respects size limit."""
        bus = EventBus(buffer_size=5)

        for i in range(10):
            await bus.publish(EventFactory.step_started(i))

        # Buffer should only keep last 5 events
        assert len(bus.event_buffer) == 5
        # First event should be step 5 (oldest kept)
        assert bus.event_buffer[0].step_number == 5

    @pytest.mark.asyncio
    async def test_clear_buffer(self):
        """Test clearing event buffer."""
        bus = EventBus()

        await bus.publish(EventFactory.step_started(1))
        await bus.publish(EventFactory.step_started(2))
        assert len(bus.event_buffer) == 2

        bus.clear_buffer()
        assert len(bus.event_buffer) == 0

    @pytest.mark.asyncio
    async def test_get_events_no_filter(self):
        """Test getting all events without filter."""
        bus = EventBus()

        await bus.publish(EventFactory.step_started(1))
        await bus.publish(EventFactory.thought(1, "thinking"))
        await bus.publish(EventFactory.step_started(2))

        events = bus.get_events()
        assert len(events) == 3

    @pytest.mark.asyncio
    async def test_get_events_by_type(self):
        """Test filtering events by type."""
        bus = EventBus()

        await bus.publish(EventFactory.step_started(1))
        await bus.publish(EventFactory.thought(1, "thinking"))
        await bus.publish(EventFactory.step_started(2))
        await bus.publish(EventFactory.final_answer(2, "done"))

        step_events = bus.get_events(event_type=ReActEventType.STEP_START)
        assert len(step_events) == 2

        thought_events = bus.get_events(event_type=ReActEventType.THOUGHT)
        assert len(thought_events) == 1

    @pytest.mark.asyncio
    async def test_get_events_by_step(self):
        """Test filtering events by step number."""
        bus = EventBus()

        await bus.publish(EventFactory.step_started(1))
        await bus.publish(EventFactory.thought(1, "thinking"))
        await bus.publish(EventFactory.step_started(2))
        await bus.publish(EventFactory.thought(2, "more thinking"))

        step1_events = bus.get_events(step_number=1)
        assert len(step1_events) == 2

    @pytest.mark.asyncio
    async def test_subscriber_error_isolation(self):
        """Test that one subscriber's error doesn't affect others."""
        bus = EventBus()
        events = []

        def failing_handler(event):
            raise RuntimeError("Handler failed")

        def good_handler(event):
            events.append(event)

        bus.subscribe(failing_handler)
        bus.subscribe(good_handler)

        # Should not raise, and good_handler should still receive event
        await bus.publish(EventFactory.step_started(1))
        assert len(events) == 1


class TestReActEventTypes:
    """Test ReAct event type coverage."""

    def test_all_event_types_exist(self):
        """Test all expected ReAct event types exist."""
        expected_types = [
            "STEP_START",
            "THOUGHT",
            "THINKING_CHUNK",
            "ACTION_PLANNED",
            "ACTION_EXECUTING",
            "OBSERVATION",
            "STEP_COMPLETE",
            "FINAL_ANSWER",
            "FINAL_ANSWER_CHUNK",
            "ERROR",
            "STOP_CONDITION",
        ]

        for type_name in expected_types:
            assert hasattr(ReActEventType, type_name), f"Missing event type: {type_name}"

    def test_event_type_values_unique(self):
        """Test event type values are unique."""
        values = [e.value for e in ReActEventType]
        assert len(values) == len(set(values)), "Event type values not unique"


class TestPlanExecuteEventTypes:
    """Test Plan & Execute event type coverage."""

    def test_all_event_types_exist(self):
        """Test all expected Plan & Execute event types exist."""
        expected_types = [
            "PLANNING_START",
            "PLANNING_COMPLETE",
            "PLANNING_THINKING_CHUNK",
            "SUBTASK_START",
            "SUBTASK_COMPLETE",
            "SUBTASK_FAILED",
            "SUBTASK_THINKING_CHUNK",
            "PLAN_PROGRESS",
            "REPLANNING_START",
            "REPLANNING_COMPLETE",
            "FINAL_ANSWER",
            "FINAL_ANSWER_CHUNK",
            "ERROR",
        ]

        for type_name in expected_types:
            assert hasattr(PlanExecuteEventType, type_name), f"Missing event type: {type_name}"


class TestStreamingPatterns:
    """Test common streaming patterns."""

    @pytest.mark.asyncio
    async def test_incremental_thinking_stream(self):
        """Test streaming thinking chunks incrementally."""
        bus = EventBus()
        accumulated = []

        def handler(event):
            if event.event_type == ReActEventType.THINKING_CHUNK:
                accumulated.append(event.data["content"])

        bus.subscribe(handler)

        # Simulate streaming "I am thinking about the problem"
        chunks = ["I ", "am ", "thinking ", "about ", "the ", "problem"]
        content = ""
        for chunk in chunks:
            content += chunk
            await bus.publish(EventFactory.thinking_chunk(1, chunk, content))

        assert len(accumulated) == 6
        assert accumulated[-1] == "I am thinking about the problem"

    @pytest.mark.asyncio
    async def test_full_react_step_event_sequence(self):
        """Test full sequence of events for a ReAct step."""
        bus = EventBus()
        event_sequence = []

        def handler(event):
            event_sequence.append(event.event_type)

        bus.subscribe(handler)

        # Step 1: Start
        await bus.publish(EventFactory.step_started(1))

        # Step 2: Thinking (streamed)
        await bus.publish(EventFactory.thinking_chunk(1, "I ", "I "))
        await bus.publish(EventFactory.thinking_chunk(1, "need to search", "I need to search"))

        # Step 3: Complete thought
        await bus.publish(EventFactory.thought(1, "I need to search"))

        # Step 4: Action planned
        await bus.publish(EventFactory.action_planned(1, "search", {"query": "test"}))

        # Step 5: Action executing
        await bus.publish(EventFactory.action_executing(1, "search", {"query": "test"}))

        # Step 6: Observation
        await bus.publish(EventFactory.observation(1, "Found results", "search", True))

        # Step 7: Step complete
        step = ReActStep(step_number=1, thought="I need to search", action="search")
        await bus.publish(EventFactory.step_complete(1, step))

        expected_sequence = [
            ReActEventType.STEP_START,
            ReActEventType.THINKING_CHUNK,
            ReActEventType.THINKING_CHUNK,
            ReActEventType.THOUGHT,
            ReActEventType.ACTION_PLANNED,
            ReActEventType.ACTION_EXECUTING,
            ReActEventType.OBSERVATION,
            ReActEventType.STEP_COMPLETE,
        ]

        assert event_sequence == expected_sequence

    @pytest.mark.asyncio
    async def test_final_answer_streaming(self):
        """Test streaming final answer chunks."""
        bus = EventBus()
        chunks_received = []

        def handler(event):
            if event.event_type == ReActEventType.FINAL_ANSWER_CHUNK:
                chunks_received.append(event.data)

        bus.subscribe(handler)

        # Stream final answer "The answer is 42"
        answer_parts = ["The ", "answer ", "is ", "42"]
        content = ""
        for part in answer_parts:
            content += part
            await bus.publish(EventFactory.final_answer_chunk(1, part, content))

        assert len(chunks_received) == 4
        assert chunks_received[-1]["content"] == "The answer is 42"
        assert chunks_received[-1]["delta"] == "42"


class TestEventDataIntegrity:
    """Test event data structure integrity."""

    def test_react_event_has_timestamp(self):
        """Test ReActEvent has timestamp."""
        event = EventFactory.step_started(1)
        assert hasattr(event, "timestamp")
        assert event.timestamp > 0

    def test_react_step_serialization(self):
        """Test ReActStep can be serialized in events."""
        step = ReActStep(
            step_number=1,
            thought="Test thought",
            action="test_action",
            action_input={"key": "value"},
            observation="Test observation",
            execution_time=0.5,
            cost=0.001,
            tokens_used=100,
        )

        event = EventFactory.step_complete(1, step)
        step_dict = event.data["step"]

        assert step_dict["step_number"] == 1
        assert step_dict["thought"] == "Test thought"
        assert step_dict["action"] == "test_action"

    def test_plan_execute_event_creation(self):
        """Test PlanExecuteEvent creation."""
        event = PlanExecuteEvent(
            event_type=PlanExecuteEventType.PLANNING_START,
            data={"goal": "Test goal"}
        )

        assert event.event_type == PlanExecuteEventType.PLANNING_START
        assert event.data["goal"] == "Test goal"
        assert hasattr(event, "timestamp")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
