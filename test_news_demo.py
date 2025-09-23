#!/usr/bin/env python3
"""News CLI Demo: Find today's top news and summarize in 5 bullets."""

import asyncio
import time
import sys
import os
from miiflow_llm.agents import create_agent, AgentConfig, AgentContext
from miiflow_llm.core.agent import AgentType


# # Add examples/tools to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'examples', 'tools'))
from news import get_top_news, search_news_by_topic
from miiflow_llm.core.tools.decorators import get_tool_from_function


async def demo_news_cli():
    """Demo CLI: Find today's top news and summarize in 5 bullets."""
    
    print("Find today's top news and summarize in 5 bullets")
    print("=" * 70)

    news_tools = []
    for func in [get_top_news, search_news_by_topic]:
        tool = get_tool_from_function(func)
        if tool:
            news_tools.append(tool)

    # Create ReAct agent for streaming
    agent = create_agent(AgentConfig(
        provider='openai',
        model='gpt-4o-mini',
        agent_type=AgentType.REACT,  # Enable ReAct mode for streaming
        tools=news_tools,
        max_iterations=8,
        system_prompt="""You are an AI news analyst using ReAct (Reasoning + Acting) pattern.

TOOLS AVAILABLE:
- get_top_news: Fetch top N news headlines from Google News
- search_news_by_topic: Search news by specific topics

TASK: Find today's top news and summarize in exactly 5 bullet points.

INSTRUCTIONS:
1. First, get the latest top news headlines
2. Analyze the most important stories
3. Summarize in exactly 5 concise bullet points
4. Focus on different categories (politics, economy, technology, world, etc.)
5. Each bullet should be 1-2 sentences maximum"""
    ))

    print(f"Agent created with {len(agent.list_tools())} tools: {agent.list_tools()}")
    print()

    context = AgentContext()
    query = "Find today's top news and summarize in 5 bullets"

    print(f" Query: '{query}'")
    print("Starting...\n")

    start_time = time.time()

    try:
        async for event in agent.stream_react(query, context):
            elapsed = time.time() - start_time
            print(f"[{elapsed:.2f}s] {event.event_type.value}")
            if event.event_type.value == "thought":
                thought = event.data.get("thought", "")
                print(f"    {thought}")
            elif event.event_type.value == "action_planned":
                action = event.data.get("action", "")
                print(f"    Planning: {action}")
            elif event.event_type.value == "observation":
                obs = event.data.get("observation", "")
                action = event.data.get("action", "")
                success = "SUCCESS" if event.data.get("success", True) else "FAILURE"
                
                print(f"   {success} {action}: {obs}")
            elif event.event_type.value == "final_answer":
                answer = event.data.get("answer", "")
                print(f"\n FINAL NEWS SUMMARY:")
                print(f"{answer}")

            print()

        total_time = time.time() - start_time
        print(f"\n Demo complete in {total_time:.2f}s")

    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()


async def test_basic_functionality():
    print(" Basic Functionality Test")
    news_tools = []
    for func in [get_top_news, search_news_by_topic]:
        tool = get_tool_from_function(func)
        if tool:
            news_tools.append(tool)
    
    agent = create_agent(AgentConfig(
        provider='openai',
        model='gpt-4o-mini',
        agent_type=AgentType.SINGLE_HOP,  # Simple single call
        tools=news_tools,
        max_iterations=5,
        system_prompt="""You are a news assistant. Use the available tools to fetch news when requested."""
    ))

    context = AgentContext()
    result = await agent.run("Get 3 top news headlines", context=context)
    response = result.get('response', str(result))
    tool_calls = result.get('tool_calls_made', 0)
    
    print(f"Response: {response}")
    print(f"Tools used: {tool_calls}")


if __name__ == "__main__":
    # asyncio.run(test_basic_functionality())
    asyncio.run(demo_news_cli())
