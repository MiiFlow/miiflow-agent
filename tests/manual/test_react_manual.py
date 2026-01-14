#!/usr/bin/env python3
"""
Manual test script for ReAct mode.

Usage:
    # OpenAI (default)
    export OPENAI_API_KEY="sk-..."
    python tests/manual/test_react_manual.py

    # Google Gemini
    export GOOGLE_API_KEY="..."
    python tests/manual/test_react_manual.py --provider gemini

    # Anthropic Claude
    export ANTHROPIC_API_KEY="sk-ant-..."
    python tests/manual/test_react_manual.py --provider anthropic

    # Run all providers
    python tests/manual/test_react_manual.py --provider all

Tests core ReAct functionality:
1. Basic tool usage
2. Streaming events
"""

import argparse
import asyncio
import math
import os
import sys

from miiflow_agent import Agent, AgentType, LLMClient, RunContext
from miiflow_agent.core.react import ReActEventType
from miiflow_agent.core.tools import tool


# =============================================================================
# Provider Configuration
# =============================================================================

PROVIDERS = {
    "openai": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "env_var": "OPENAI_API_KEY",
    },
    "gemini": {
        "provider": "gemini",
        "model": "gemini-2.0-flash",
        "env_var": "GOOGLE_API_KEY",
    },
    "anthropic": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-5-20250929",
        "env_var": "ANTHROPIC_API_KEY",
    },
}


# =============================================================================
# Tool Definitions (simple mock tools)
# =============================================================================


@tool("calculate", "Evaluate mathematical expressions")
def calculate(expression: str) -> str:
    """Safely evaluate a math expression.

    Args:
        expression: A mathematical expression like '2 + 2' or 'sqrt(16)'
    """
    allowed = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sqrt": math.sqrt,
        "pow": pow,
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
    weather_data = {
        "new york": "Sunny, 72°F (22°C)",
        "london": "Cloudy, 59°F (15°C)",
        "tokyo": "Rainy, 68°F (20°C)",
        "paris": "Partly cloudy, 65°F (18°C)",
        "san francisco": "Foggy, 60°F (16°C)",
    }

    location_lower = location.lower()
    if location_lower in weather_data:
        return f"Weather in {location}: {weather_data[location_lower]}"
    return f"Weather data not available for {location}"


# =============================================================================
# Test Infrastructure
# =============================================================================


def check_environment(provider_name: str) -> bool:
    """Check that required environment variables are set for the provider."""
    config = PROVIDERS[provider_name]
    env_var = config["env_var"]
    api_key = os.environ.get(env_var)

    if not api_key:
        print(f"SKIP: {env_var} not set for {provider_name}")
        return False

    print(f"Environment check passed: {env_var} is set")
    return True


def create_agent(provider_name: str):
    """Create the ReAct agent with tools for the specified provider."""
    config = PROVIDERS[provider_name]

    client = LLMClient.create(
        config["provider"],
        model=config["model"],
        api_key=os.environ.get(config["env_var"]),
    )

    agent = Agent(
        client,
        agent_type=AgentType.REACT,
        max_iterations=5,
        system_prompt="You are a helpful assistant. Use tools to answer questions.",
    )

    agent.add_tool(calculate)
    agent.add_tool(get_weather)

    return agent


# =============================================================================
# Test Cases
# =============================================================================


async def test_basic(agent):
    """Test 1: Basic tool usage"""
    print("\n" + "=" * 60)
    print("TEST 1: Basic Tool Usage")
    print("Query: What is sqrt(144) + 8?")
    print("=" * 60)

    result = await agent.run("What is sqrt(144) + 8?")

    print(f"\nResult:\n{result.data}")

    # Validate result
    assert result.data is not None, "Result should not be None"
    assert len(result.data) > 10, "Result should contain content"
    # Result should contain 20 (sqrt(144) = 12, 12 + 8 = 20)
    assert "20" in result.data, "Result should contain the answer 20"

    print("\nPASSED")


async def test_streaming(agent):
    """Test 2: Streaming events"""
    print("\n" + "=" * 60)
    print("TEST 2: Streaming Events")
    print("Query: What's the weather in Tokyo?")
    print("=" * 60)

    context = RunContext(deps=None, messages=[])
    events_received = []
    final_answer = None

    async for event in agent.stream("What's the weather in Tokyo?", context):
        events_received.append(event.event_type)

        if event.event_type == ReActEventType.THINKING_CHUNK:
            print(event.data.get("delta", ""), end="", flush=True)
        elif event.event_type == ReActEventType.ACTION_PLANNED:
            action = event.data.get("action", "")
            print(f"\n[Calling tool: {action}]")
        elif event.event_type == ReActEventType.OBSERVATION:
            obs = str(event.data.get("observation", ""))[:100]
            print(f"[Tool result: {obs}]")
        elif event.event_type == ReActEventType.FINAL_ANSWER:
            final_answer = event.data.get("answer", "")
            print(f"\n\nFINAL ANSWER:\n{final_answer}")

    # Validate streaming worked
    assert len(events_received) > 0, "Should receive streaming events"
    assert ReActEventType.FINAL_ANSWER in events_received, "Should receive FINAL_ANSWER event"
    assert final_answer is not None, "Should have final answer"
    # Should mention Tokyo weather
    assert "tokyo" in final_answer.lower() or "rainy" in final_answer.lower(), \
        "Answer should mention Tokyo or weather"

    print("\nPASSED")


# =============================================================================
# Main Runner
# =============================================================================


async def run_tests_for_provider(provider_name: str) -> tuple[int, int, list]:
    """Run all tests for a specific provider.

    Returns:
        Tuple of (passed, failed, failures list)
    """
    config = PROVIDERS[provider_name]
    print("\n" + "=" * 60)
    print(f"Testing with: {provider_name.upper()} ({config['model']})")
    print("=" * 60)

    if not check_environment(provider_name):
        return 0, 0, []  # Skipped

    print(f"\nCreating agent with {provider_name}...")
    agent = create_agent(provider_name)
    print("Agent created with tools: calculate, get_weather")

    tests = [
        ("Basic Tool Usage", test_basic),
        ("Streaming Events", test_streaming),
    ]

    passed = 0
    failed = 0
    failures = []

    for name, test_fn in tests:
        try:
            await test_fn(agent)
            passed += 1
        except AssertionError as e:
            failed += 1
            failures.append((f"{provider_name}/{name}", str(e)))
            print(f"\nFAILED: {e}")
        except Exception as e:
            failed += 1
            failures.append((f"{provider_name}/{name}", f"Exception: {e}"))
            print(f"\nFAILED with exception: {e}")

    return passed, failed, failures


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Manual test script for ReAct mode with multiple providers"
    )
    parser.add_argument(
        "--provider",
        "-p",
        choices=["openai", "gemini", "anthropic", "all"],
        default="openai",
        help="LLM provider to test (default: openai)",
    )
    return parser.parse_args()


async def main():
    """Run all tests and report results."""
    args = parse_args()

    print("=" * 60)
    print("ReAct Agent Manual Test Suite")
    print("=" * 60)

    # Determine which providers to test
    if args.provider == "all":
        providers_to_test = list(PROVIDERS.keys())
    else:
        providers_to_test = [args.provider]

    total_passed = 0
    total_failed = 0
    total_skipped = 0
    all_failures = []

    for provider_name in providers_to_test:
        passed, failed, failures = await run_tests_for_provider(provider_name)
        if passed == 0 and failed == 0:
            total_skipped += 1
        else:
            total_passed += passed
            total_failed += failed
            all_failures.extend(failures)

    # Final Summary
    print("\n" + "=" * 60)
    print("FINAL TEST SUMMARY")
    print("=" * 60)
    print(f"Providers tested: {len(providers_to_test) - total_skipped}/{len(providers_to_test)}")
    print(f"Total Passed: {total_passed}")
    print(f"Total Failed: {total_failed}")

    if all_failures:
        print("\nAll Failures:")
        for name, error in all_failures:
            print(f"  - {name}: {error}")

    print("=" * 60)

    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
