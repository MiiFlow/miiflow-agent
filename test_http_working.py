#!/usr/bin/env python3
"""Test function tools using decorated tools from examples."""

import asyncio
from miiflow_llm.agents import create_agent, AgentConfig, ContextType, AgentContext


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
        context_type=ContextType.USER,
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

    context = AgentContext(ContextType.USER)

    # Test 1: Function tool (known to work)
    print("\nTest 1: Function Tool")
    result1 = await agent.run("What is 6 times 72?", context=context)
    response1 = result1.get('response', str(result1))
    tool_calls1 = result1.get('tool_calls_made', 0)
    print(f"Agent: {response1}")
    print(f"Tools used: {tool_calls1}")

    # Test 2: HTTP tool with same agent
    print("\nTest 2: HTTP Tool")
    result2 = await agent.run("Get posts for user 1+4 and what is 2*46", context=context)
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

if __name__ == "__main__":
    asyncio.run(test_with_working_config())
