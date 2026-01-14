"""ReAct Agent example.

This example demonstrates the ReAct (Reasoning + Acting) pattern:
1. Basic usage - Creating an agent with tools
2. Streaming - Real-time event streaming
"""

import asyncio
import math

from miiflow_agent import LLMClient, Agent, AgentType, RunContext, tool
from miiflow_agent.core.react import ReActEventType


# Define simple tools using the @tool decorator
@tool("calculate", "Evaluate mathematical expressions safely")
def calculate(expression: str) -> str:
    """Calculate the result of a mathematical expression.

    Args:
        expression: A mathematical expression like "2 + 2" or "sqrt(16)"
    """
    allowed = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "pi": math.pi,
        "e": math.e,
    }

    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


@tool("get_weather", "Get current weather for a location")
def get_weather(location: str) -> str:
    """Get weather information for a location.

    Args:
        location: City name or location
    """
    # Simulated weather data
    weather_data = {
        "new york": "Sunny, 72°F (22°C)",
        "london": "Cloudy, 59°F (15°C)",
        "tokyo": "Rainy, 68°F (20°C)",
        "paris": "Partly cloudy, 65°F (18°C)",
    }

    location_lower = location.lower()
    if location_lower in weather_data:
        return f"Weather in {location}: {weather_data[location_lower]}"
    return f"Weather data not available for {location}"


async def basic_react_example():
    """Basic ReAct agent with tools."""
    # Create client
    client = LLMClient.create("openai", model="gpt-4o-mini")

    # Create agent with tools
    agent = Agent(
        client,
        agent_type=AgentType.REACT,
        max_iterations=5,
        system_prompt="You are a helpful assistant with access to tools.",
    )

    # Add tools
    agent.add_tool(calculate)
    agent.add_tool(get_weather)

    # Run agent
    print("Query: What is 25 * 4 + 100?")
    result = await agent.run("What is 25 * 4 + 100?")
    print(f"Answer: {result.data}\n")


async def streaming_example():
    """Agent with real-time event streaming."""
    client = LLMClient.create("openai", model="gpt-4o-mini")

    agent = Agent(
        client,
        agent_type=AgentType.REACT,
        max_iterations=5,
    )

    agent.add_tool(calculate)
    agent.add_tool(get_weather)

    print("Query: What's the weather in Tokyo, and calculate sqrt(144) + 8")
    print("=" * 50)

    context = RunContext(deps=None, messages=[])

    async for event in agent.stream(
        "What's the weather in Tokyo, and calculate sqrt(144) + 8", context
    ):
        if event.event_type == ReActEventType.THINKING_CHUNK:
            print(event.data.get("delta", ""), end="", flush=True)
        elif event.event_type == ReActEventType.ACTION_PLANNED:
            action = event.data.get("action", "")
            print(f"\n[Calling tool: {action}]")
        elif event.event_type == ReActEventType.OBSERVATION:
            obs = str(event.data.get("observation", ""))[:100]
            print(f"[Tool result: {obs}]")
        elif event.event_type == ReActEventType.FINAL_ANSWER:
            print("\n" + "=" * 50)
            print("FINAL ANSWER:")
            print(event.data.get("answer", ""))


if __name__ == "__main__":
    print("=== Basic ReAct Example ===")
    asyncio.run(basic_react_example())

    print("\n" + "=" * 50 + "\n")

    print("=== Streaming Example ===")
    asyncio.run(streaming_example())
