#!/usr/bin/env python3
"""Manual end-to-end test for the callback system.

This test verifies that callbacks fire correctly for:
1. LLMClient.achat() - POST_CALL callback with token usage
2. LLMClient.astream_chat() - POST_CALL callback after streaming completes
3. Agent.run() - AGENT_RUN_START and AGENT_RUN_END callbacks
4. Agent.stream() with REACT - callbacks for multi-hop reasoning
5. Context passing via callback_context()

Usage:
    cd packages/miiflow-llm
    OPENAI_API_KEY=sk-xxx python tests/manual/test_callbacks_manual.py
"""

import asyncio
import os
import sys
from dataclasses import dataclass, field
from typing import List

# Add the package to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from miiflow_llm import (
    Agent,
    AgentType,
    CallbackContext,
    CallbackEvent,
    CallbackEventType,
    LLMClient,
    Message,
    RunContext,
    callback_context,
    clear,
    get_callback_context,
    on_agent_run_end,
    on_agent_run_start,
    on_error,
    on_post_call,
)
from miiflow_llm.core.tools import tool


@dataclass
class CallbackTracker:
    """Tracks all callback events for verification."""
    events: List[CallbackEvent] = field(default_factory=list)

    def reset(self):
        self.events.clear()

    def get_events_by_type(self, event_type: CallbackEventType) -> List[CallbackEvent]:
        return [e for e in self.events if e.event_type == event_type]

    def print_summary(self):
        print("\n" + "=" * 60)
        print("CALLBACK EVENTS SUMMARY")
        print("=" * 60)
        for i, event in enumerate(self.events, 1):
            print(f"\n[Event {i}] {event.event_type.value}")
            print(f"  Provider: {event.provider}")
            print(f"  Model: {event.model}")
            if event.tokens:
                print(f"  Tokens: prompt={event.tokens.prompt_tokens}, completion={event.tokens.completion_tokens}, total={event.tokens.total_tokens}")
            if event.latency_ms:
                print(f"  Latency: {event.latency_ms:.2f}ms")
            if event.agent_type:
                print(f"  Agent Type: {event.agent_type}")
            if event.query:
                print(f"  Query: {event.query[:50]}...")
            if event.context:
                print(f"  Context: org_id={event.context.organization_id}, node_run_id={event.context.agent_node_run_id}")
            if event.error:
                print(f"  Error: {event.error_type}: {event.error}")
            print(f"  Success: {event.success}")
        print("\n" + "=" * 60)


# Global tracker
tracker = CallbackTracker()


# Register callbacks
@on_post_call
async def track_post_call(event: CallbackEvent):
    """Track POST_CALL events."""
    tracker.events.append(event)
    print(f"[CALLBACK] POST_CALL: {event.provider}:{event.model} - {event.tokens.total_tokens if event.tokens else 0} tokens")


@on_error
async def track_error(event: CallbackEvent):
    """Track ON_ERROR events."""
    tracker.events.append(event)
    print(f"[CALLBACK] ON_ERROR: {event.error_type}: {event.error}")


@on_agent_run_start
async def track_agent_start(event: CallbackEvent):
    """Track AGENT_RUN_START events."""
    tracker.events.append(event)
    print(f"[CALLBACK] AGENT_RUN_START: {event.agent_type}")


@on_agent_run_end
async def track_agent_end(event: CallbackEvent):
    """Track AGENT_RUN_END events."""
    tracker.events.append(event)
    print(f"[CALLBACK] AGENT_RUN_END: {event.agent_type} - success={event.success}")


# Define a simple tool for ReAct testing
@tool("calculate", "Perform a mathematical calculation")
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


async def test_llm_client_achat():
    """Test 1: Basic LLMClient.achat() callback."""
    print("\n" + "=" * 60)
    print("TEST 1: LLMClient.achat() with callback context")
    print("=" * 60)

    tracker.reset()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return False

    client = LLMClient.create("openai", model="gpt-4o-mini", api_key=api_key)

    # Create context for billing tracking
    ctx = CallbackContext(
        organization_id="org_test_123",
        agent_node_run_id="run_test_456",
        metadata={"owner_type": "platform", "test": "achat"}
    )

    # Make LLM call with context
    with callback_context(ctx):
        response = await client.achat([
            Message.user("Say 'Hello callback test!' in exactly 5 words.")
        ])

    print(f"\nResponse: {response.message.content}")

    # Verify callback fired
    post_call_events = tracker.get_events_by_type(CallbackEventType.POST_CALL)

    assert len(post_call_events) == 1, f"Expected 1 POST_CALL event, got {len(post_call_events)}"

    event = post_call_events[0]
    assert event.provider == "openai", f"Expected provider 'openai', got '{event.provider}'"
    assert event.model == "gpt-4o-mini", f"Expected model 'gpt-4o-mini', got '{event.model}'"
    assert event.tokens is not None, "Expected tokens to be set"
    assert event.tokens.prompt_tokens > 0, "Expected prompt_tokens > 0"
    assert event.tokens.completion_tokens > 0, "Expected completion_tokens > 0"
    assert event.context is not None, "Expected context to be set"
    assert event.context.organization_id == "org_test_123", "Expected organization_id to match"
    assert event.context.agent_node_run_id == "run_test_456", "Expected agent_node_run_id to match"
    assert event.success is True, "Expected success to be True"

    print("\n✅ TEST 1 PASSED: achat() callback works correctly")
    return True


async def test_llm_client_streaming():
    """Test 2: LLMClient.astream_chat() callback."""
    print("\n" + "=" * 60)
    print("TEST 2: LLMClient.astream_chat() with callback context")
    print("=" * 60)

    tracker.reset()

    api_key = os.environ.get("OPENAI_API_KEY")
    client = LLMClient.create("openai", model="gpt-4o-mini", api_key=api_key)

    ctx = CallbackContext(
        organization_id="org_stream_123",
        agent_node_run_id="run_stream_456",
        metadata={"owner_type": "platform", "test": "streaming"}
    )

    content = ""
    with callback_context(ctx):
        async for chunk in client.astream_chat([
            Message.user("Count from 1 to 5, one number per line.")
        ]):
            if chunk.delta:
                content += chunk.delta
                print(chunk.delta, end="", flush=True)

    # Allow pending callback tasks to complete
    await asyncio.sleep(0.1)

    print(f"\n\nFull response: {content[:100]}...")

    # Verify callback fired after streaming completed
    post_call_events = tracker.get_events_by_type(CallbackEventType.POST_CALL)

    assert len(post_call_events) == 1, f"Expected 1 POST_CALL event, got {len(post_call_events)}"

    event = post_call_events[0]
    assert event.context is not None, "Expected context to be set"
    assert event.context.organization_id == "org_stream_123", "Expected organization_id to match"
    assert event.tokens is not None, "Expected tokens to be set"
    assert event.success is True, "Expected success to be True"

    print("\n✅ TEST 2 PASSED: astream_chat() callback works correctly")
    return True


async def test_agent_run():
    """Test 3: Agent.run() with AGENT_RUN_START and AGENT_RUN_END callbacks."""
    print("\n" + "=" * 60)
    print("TEST 3: Agent.run() with agent lifecycle callbacks")
    print("=" * 60)

    tracker.reset()

    api_key = os.environ.get("OPENAI_API_KEY")
    client = LLMClient.create("openai", model="gpt-4o-mini", api_key=api_key)

    agent = Agent(
        client,
        agent_type=AgentType.SINGLE_HOP,
        system_prompt="You are a helpful assistant. Be brief.",
    )

    ctx = CallbackContext(
        organization_id="org_agent_123",
        assistant_id="asst_test_789",
        metadata={"owner_type": "platform", "test": "agent_run"}
    )

    with callback_context(ctx):
        result = await agent.run("What is 2+2? Answer with just the number.")

    # Allow pending callback tasks to complete
    await asyncio.sleep(0.1)

    print(f"\nAgent result: {result.data}")

    # Verify agent callbacks fired
    start_events = tracker.get_events_by_type(CallbackEventType.AGENT_RUN_START)
    end_events = tracker.get_events_by_type(CallbackEventType.AGENT_RUN_END)
    post_call_events = tracker.get_events_by_type(CallbackEventType.POST_CALL)

    assert len(start_events) == 1, f"Expected 1 AGENT_RUN_START, got {len(start_events)}"
    assert len(end_events) == 1, f"Expected 1 AGENT_RUN_END, got {len(end_events)}"
    assert len(post_call_events) >= 1, f"Expected at least 1 POST_CALL, got {len(post_call_events)}"

    # Verify start event
    start_event = start_events[0]
    assert start_event.agent_type == "single_hop", f"Expected agent_type 'single_hop', got '{start_event.agent_type}'"
    assert start_event.context is not None, "Expected context in start event"
    assert start_event.context.organization_id == "org_agent_123", "Expected organization_id to match"

    # Verify end event
    end_event = end_events[0]
    assert end_event.success is True, "Expected agent run to succeed"
    assert end_event.context is not None, "Expected context in end event"

    print("\n✅ TEST 3 PASSED: Agent.run() lifecycle callbacks work correctly")
    return True


async def test_agent_stream_react():
    """Test 4: Agent.stream() with REACT and tools - verifies AgentToolExecutor fix."""
    print("\n" + "=" * 60)
    print("TEST 4: Agent.stream() REACT with tools (AgentToolExecutor fix)")
    print("=" * 60)

    tracker.reset()

    api_key = os.environ.get("OPENAI_API_KEY")
    client = LLMClient.create("openai", model="gpt-4o-mini", api_key=api_key)

    agent = Agent(
        client,
        agent_type=AgentType.REACT,
        system_prompt="You are a helpful assistant that can do calculations. Use the calculate tool when needed.",
        max_iterations=5,
    )
    agent.add_tool(calculate)

    ctx = CallbackContext(
        organization_id="org_react_123",
        assistant_id="asst_react_789",
        agent_node_run_id="run_react_456",
        metadata={"owner_type": "platform", "test": "react_stream"}
    )

    run_context = RunContext(deps=None, messages=[])

    events_received = []
    with callback_context(ctx):
        async for event in agent.stream(
            "What is 15 * 7? Use the calculate tool to find out.",
            run_context,
            agent_type=AgentType.REACT,
            max_steps=5,
        ):
            events_received.append(event)
            if hasattr(event, 'event_type'):
                print(f"  [ReAct Event] {event.event_type.value}")

    # Allow pending callback tasks to complete
    await asyncio.sleep(0.1)

    print(f"\nReceived {len(events_received)} ReAct events")

    # Verify callbacks fired
    start_events = tracker.get_events_by_type(CallbackEventType.AGENT_RUN_START)
    end_events = tracker.get_events_by_type(CallbackEventType.AGENT_RUN_END)
    post_call_events = tracker.get_events_by_type(CallbackEventType.POST_CALL)

    assert len(start_events) == 1, f"Expected 1 AGENT_RUN_START, got {len(start_events)}"
    assert len(end_events) == 1, f"Expected 1 AGENT_RUN_END, got {len(end_events)}"

    # ReAct should make multiple LLM calls (reasoning steps)
    print(f"\nPOST_CALL events: {len(post_call_events)} (one per LLM call in ReAct)")
    assert len(post_call_events) >= 1, f"Expected at least 1 POST_CALL for ReAct, got {len(post_call_events)}"

    # Verify context is passed through all calls
    for event in post_call_events:
        assert event.context is not None, "Expected context in POST_CALL"
        assert event.context.organization_id == "org_react_123", f"Expected org_id 'org_react_123', got '{event.context.organization_id}'"
        assert event.tokens is not None, "Expected tokens in POST_CALL"

    # Calculate total tokens across all LLM calls
    total_prompt = sum(e.tokens.prompt_tokens for e in post_call_events if e.tokens)
    total_completion = sum(e.tokens.completion_tokens for e in post_call_events if e.tokens)
    print(f"Total tokens across all ReAct calls: prompt={total_prompt}, completion={total_completion}")

    print("\n✅ TEST 4 PASSED: Agent.stream() REACT callbacks work correctly")
    print("   (This verifies the AgentToolExecutor fix - callbacks fire for all reasoning steps)")
    return True


async def test_context_without_callback():
    """Test 5: Verify callbacks work without context (context is optional)."""
    print("\n" + "=" * 60)
    print("TEST 5: Callbacks without context (context should be None)")
    print("=" * 60)

    tracker.reset()

    api_key = os.environ.get("OPENAI_API_KEY")
    client = LLMClient.create("openai", model="gpt-4o-mini", api_key=api_key)

    # Make call WITHOUT callback_context()
    response = await client.achat([
        Message.user("Say 'no context test' in 3 words.")
    ])

    print(f"\nResponse: {response.message.content}")

    # Verify callback fired with None context
    post_call_events = tracker.get_events_by_type(CallbackEventType.POST_CALL)

    assert len(post_call_events) == 1, f"Expected 1 POST_CALL, got {len(post_call_events)}"

    event = post_call_events[0]
    assert event.context is None, f"Expected context to be None, got {event.context}"
    assert event.tokens is not None, "Expected tokens to be set"
    assert event.success is True, "Expected success to be True"

    print("\n✅ TEST 5 PASSED: Callbacks work without context (context is None)")
    return True


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("MIIFLOW-LLM CALLBACK SYSTEM - END-TO-END TESTS")
    print("=" * 60)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\nERROR: OPENAI_API_KEY environment variable not set")
        print("Usage: OPENAI_API_KEY=sk-xxx python tests/manual/test_callbacks_manual.py")
        sys.exit(1)

    print(f"\nUsing API key: {api_key[:20]}...")

    results = []

    try:
        # Test 1: Basic achat()
        results.append(("LLMClient.achat()", await test_llm_client_achat()))

        # Test 2: Streaming
        results.append(("LLMClient.astream_chat()", await test_llm_client_streaming()))

        # Test 3: Agent.run()
        results.append(("Agent.run()", await test_agent_run()))

        # Test 4: Agent.stream() with REACT
        results.append(("Agent.stream() REACT", await test_agent_stream_react()))

        # Test 5: Without context
        results.append(("Without context", await test_context_without_callback()))

    except Exception as e:
        print(f"\n❌ TEST FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Exception", False))

    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    # Print all tracked events
    tracker.print_summary()

    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print("\n💥 SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
