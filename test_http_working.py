#!/usr/bin/env python3
"""Test function tools using decorated tools from examples with Phoenix observability."""

import asyncio
import os
import time

# Enable Phoenix observability using the cleaned-up API
print("🔍 Setting up Phoenix observability...")

# Import the new cleaned-up observability system
from miiflow_llm.core.observability import (
    enable_phoenix_tracing,
    launch_local_phoenix,
    check_instrumentation_status,
    get_logger
)

# Setup observability
print("⚙️  Initializing observability...")
try:
    # Launch Phoenix session
    session = launch_local_phoenix()
    if session:
        print(f"✓ Phoenix started: {session.url}")
        
        # Enable tracing
        success = enable_phoenix_tracing(endpoint="http://localhost:6006")
        print(f"✓ Phoenix tracing enabled: {success}")
    else:
        print("⚠️  Failed to start Phoenix")
    
    # Check instrumentation status
    instrumentation_status = check_instrumentation_status()
    for provider, status in instrumentation_status.items():
        available = "✓" if status.get("available", False) else "✗"
        instrumented = "✓" if status.get("instrumented", False) else "✗"
        print(f"  {provider}: Available {available}, Instrumented {instrumented}")
        
except Exception as e:
    print(f"⚠️  Observability setup failed: {e}")
    print("Continuing without observability...")

from miiflow_llm.agents import create_agent, AgentConfig, AgentContext
from miiflow_llm.core.agent import AgentType


import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'examples', 'tools'))
from calculator import calculate
from posts_api import get_user_posts, get_all_users
from miiflow_llm.core.tools.decorators import get_tool_from_function

async def test_with_working_config():
    """Test both function and API tools using decorated functions."""
    print("Testing Function Tools with Working Agent Config")
    print("=" * 50)

    # Extract FunctionTool instances from decorated functions
    function_tools = []
    for func in [calculate, get_user_posts, get_all_users]:
        tool = get_tool_from_function(func)
        if tool:
            function_tools.append(tool)
    
    agent = create_agent(AgentConfig(
        provider='openai',
        model='gpt-4o-mini',
        agent_type=AgentType.SINGLE_HOP,
        tools=function_tools,
        max_iterations=20,
        system_prompt="""You are a professional assistant with access to calculation and data retrieval tools.

TOOLS AVAILABLE:
- calculate: Evaluate mathematical expressions (supports +, -, *, /, **, sqrt, sin, cos, tan, log, etc.)
- get_user_posts: Fetch user posts from JSONPlaceholder API by user ID
- get_all_users: Fetch all users from JSONPlaceholder API

GUIDELINES:
- Think step by step and show your reasoning process
- When you need data or calculations, explain why you're using each tool
- For complex questions, break them down into logical steps
- Be thorough in your analysis but efficient in execution"""
    ))

    print(f"Agent created with {len(agent.list_tools())} tools: {agent.list_tools()}")
    schemas = agent.agent.tool_registry.get_schemas('openai')
    print(f"\nSchemas sent to OpenAI API:")
    for schema in schemas:
        name = schema['function']['name']
        desc = schema['function']['description']
        params = list(schema['function']['parameters']['properties'].keys())
        print(f"  {name}: {desc} | params: {params}")
    print()

    context = AgentContext()

    # Test 1: Function tool (known to work)
    print("\nTest 1: Function Tool")
    result1 = await agent.run("What is 6 times 72?", context=context)
    response1 = result1.get('response', str(result1))
    tool_calls1 = result1.get('tool_calls_made', 0)
    print(f"Agent: {response1}")
    print(f"Tools used: {tool_calls1}")

    # Test 2: HTTP tool with same agent
    print("\nTest 2: HTTP Tool")
    result2 = await agent.run("Get posts for user 2, then calculate 45* 45", context=context)
    response2 = result2.get('response', str(result2))
    tool_calls2 = result2.get('tool_calls_made', 0)
    print(f"Agent: {response2}")
    print(f"Tools used: {tool_calls2}")

    if tool_calls1 > 0 and tool_calls2 > 0:
        print("\nSUCCESS: Both function and HTTP tools work!")
    elif tool_calls1 > 0:
        print(f"\nFunction tool works ({tool_calls1} calls) but HTTP tool doesn't ({tool_calls2} calls)")
    else:
        print(f"\nNeither tool worked properly")

    # Show observability status after tests
    print("\n" + "="*50)
    print("📊 Observability Status After Tests")
    print("="*50)
    
    # Check if Phoenix tracing is working
    print("✓ Phoenix tracing should be capturing LLM calls")
    print("✓ Check Phoenix Dashboard at http://localhost:6006/ for traces")
    print("✓ Structured logging with trace correlation is active")
    
    # Show current instrumentation status
    try:
        instrumentation_status = check_instrumentation_status()
        print("✓ Current instrumentation status:")
        for provider, status in instrumentation_status.items():
            available = "✓" if status.get("available", False) else "✗"
            instrumented = "✓" if status.get("instrumented", False) else "✗"
            print(f"    {provider}: Available {available}, Instrumented {instrumented}")
    except Exception as e:
        print(f"⚠️  Could not check instrumentation status: {e}")

async def test_streaming_react():
    """Test real-time ReAct streaming with the same tools."""
    print("\n" + "="*60)
    print(" Testing Real-Time ReAct Streaming")
    print("="*60)

    # Same tool setup as working test
    function_tools = []
    for func in [calculate, get_user_posts, get_all_users]:
        tool = get_tool_from_function(func)
        if tool:
            function_tools.append(tool)

    # Create ReAct agent
    agent = create_agent(AgentConfig(
        provider='openai',
        model='gpt-4o-mini',
        agent_type=AgentType.REACT,  # Enable ReAct mode for streaming
        tools=function_tools,
        max_iterations=10,
        system_prompt="""You are an AI assistant using ReAct (Reasoning + Acting) pattern.

TOOLS AVAILABLE:
- calculate: Evaluate mathematical expressions
- get_user_posts: Fetch user posts from API by user ID
- get_all_users: Fetch all users from API

Think step by step and use tools when needed."""
    ))

    print(f"ReAct Agent created with {len(agent.list_tools())} tools")
    print(f"Agent type: {agent.agent.agent_type}")

    context = AgentContext()
    query = "Get posts for user 2, then calculate how many posts they have multiplied by 3"

    print(f"\nQuery: '{query}'")
    print("Starting real-time streaming...\n")

    start_time = time.time()
    event_count = 0

    try:
        # Stream ReAct events in real-time
        async for event in agent.stream_react(query, context):
            elapsed = time.time() - start_time
            event_count += 1

            print(f" [{elapsed:.3f}s] {event.event_type.value}")

            # Show relevant event details
            if event.event_type.value == "thought":
                thought = event.data.get("thought", "")
                print(f"   💭 {thought}")
            elif event.event_type.value == "action_planned":
                action = event.data.get("action", "")
                print(f"   📋 Planning: {action}")
            elif event.event_type.value == "observation":
                obs = event.data.get("observation", "")
                action = event.data.get("action", "")
                success = "SUCCESS" if event.data.get("success", True) else "❌"
                print(f"   {success} {action}: {obs}...")
            elif event.event_type.value == "final_answer":
                answer = event.data.get("answer", "")
                print(f"{answer}")

            print()

        total_time = time.time() - start_time
        print(f" Streaming complete! {event_count} events in {total_time:.3f}s")

    except Exception as e:
        print(f" Streaming error: {e}")

if __name__ == "__main__":
    print("🚀 Starting tests with Phoenix observability enabled...")
    print("="*60)
    
    # Run both tests
    asyncio.run(test_with_working_config())
    asyncio.run(test_streaming_react())
    
    # Keep session open until user input
    print("\n🎉 Tests completed!")
    print("🌐 Phoenix Dashboard should be available at: http://localhost:6006/")
    print("📊 You can now explore the traces and observability data.")
    print("💡 Keep this session running to maintain Phoenix access.")
    
    try:
        input("\nPress Enter to exit and stop the session...")
    except KeyboardInterrupt:
        print("\n👋 Session terminated.")
    
    print("✅ Session ended.")
