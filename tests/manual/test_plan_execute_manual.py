#!/usr/bin/env python3
"""
Manual test script for Plan & Execute mode.

Usage:
    # OpenAI (default)
    export OPENAI_API_KEY="sk-..."
    python tests/manual/test_plan_execute_manual.py

    # Google Gemini
    export GOOGLE_API_KEY="..."
    python tests/manual/test_plan_execute_manual.py --provider gemini

    # Anthropic Claude
    export ANTHROPIC_API_KEY="sk-ant-..."
    python tests/manual/test_plan_execute_manual.py --provider anthropic

    # Run all providers
    python tests/manual/test_plan_execute_manual.py --provider all

Tests core Plan & Execute functionality:
1. Basic plan creation and execution
2. Streaming events
"""

import argparse
import asyncio
import os
import sys

from miiflow_agent import Agent, AgentType, LLMClient, RunContext
from miiflow_agent.core.react import PlanExecuteEventType
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


@tool("search_web", "Search the web for information")
def search_web(query: str) -> str:
    """Search for information on the web.

    Args:
        query: Search query
    """
    return f"""
    Search results for '{query}':
    1. Wikipedia article about {query}
    2. Recent news: {query} trends
    3. Expert analysis of {query}
    """


@tool("analyze_data", "Analyze data and provide insights")
def analyze_data(data: str, analysis_type: str = "summary") -> str:
    """Analyze provided data.

    Args:
        data: Data to analyze
        analysis_type: Type of analysis (summary, comparison, trend)
    """
    return f"""
    Analysis ({analysis_type}):
    - Data received: {data[:100]}...
    - Key findings: 3 main patterns identified
    - Confidence: High
    """


@tool("generate_report", "Generate a formatted report")
def generate_report(title: str, sections: str) -> str:
    """Generate a report from sections.

    Args:
        title: Report title
        sections: Report sections
    """
    return f"""
    # {title}

    ## Executive Summary
    This report covers: {sections}

    ## Key Findings
    - Finding 1: Significant trends identified
    - Finding 2: Areas for improvement noted

    ## Conclusion
    Based on analysis, actionable steps are recommended.
    """


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
    """Create the Plan & Execute agent with tools for the specified provider."""
    config = PROVIDERS[provider_name]

    client = LLMClient.create(
        config["provider"],
        model=config["model"],
        api_key=os.environ.get(config["env_var"]),
    )

    agent = Agent(
        client,
        agent_type=AgentType.PLAN_AND_EXECUTE,
        max_iterations=10,
        system_prompt="You are an analyst. Create structured plans for research tasks.",
    )

    agent.add_tool(search_web)
    agent.add_tool(analyze_data)
    agent.add_tool(generate_report)

    return agent


# =============================================================================
# Test Cases
# =============================================================================


async def test_basic(agent):
    """Test 1: Basic Plan & Execute"""
    print("\n" + "=" * 60)
    print("TEST 1: Basic Plan & Execute")
    print("Query: Research AI trends and create a brief report")
    print("=" * 60)

    task = """
    Research the latest trends in artificial intelligence,
    analyze the key developments, and create a brief report.
    """

    result = await agent.run(task)

    print(f"\nResult:\n{result.data[:500]}..." if len(result.data) > 500 else f"\nResult:\n{result.data}")

    # Validate result
    assert result.data is not None, "Result should not be None"
    assert len(result.data) > 50, "Result should contain substantial content"
    # Result should be about AI
    result_lower = result.data.lower()
    assert any(word in result_lower for word in ["ai", "artificial", "intelligence", "trends"]), \
        "Result should mention AI or trends"

    print("\nPASSED")


async def test_streaming(agent):
    """Test 2: Streaming events"""
    print("\n" + "=" * 60)
    print("TEST 2: Streaming Events")
    print("Query: Find info about Python frameworks and compare them")
    print("=" * 60)

    task = "Find information about Python web frameworks and compare them"

    context = RunContext(deps=None, messages=[])
    events_received = []
    final_answer = None
    planning_complete = False

    async for event in agent.stream(task, context):
        event_type = event.event_type
        events_received.append(event_type)

        if event_type == PlanExecuteEventType.PLANNING_START:
            print("\n[PHASE 1: PLANNING]")
            print("Creating execution plan...")

        elif event_type == PlanExecuteEventType.PLANNING_COMPLETE:
            subtask_count = event.data.get("subtask_count", 0)
            print(f"\nPlan created with {subtask_count} subtasks!")
            print("\n[PHASE 2: EXECUTION]")
            planning_complete = True

        elif event_type == PlanExecuteEventType.SUBTASK_START:
            desc = event.data.get("description", "")[:60]
            print(f"\n  Subtask: {desc}...")

        elif event_type == PlanExecuteEventType.SUBTASK_COMPLETE:
            result_preview = str(event.data.get("result", ""))[:80]
            print(f"  Done: {result_preview}...")

        elif event_type == PlanExecuteEventType.FINAL_ANSWER:
            final_answer = event.data.get("answer", "")
            print("\n" + "=" * 50)
            print("[PHASE 3: FINAL ANSWER]")
            print("=" * 50)
            print(final_answer[:300] + "..." if len(final_answer) > 300 else final_answer)

    # Validate streaming worked
    assert len(events_received) > 0, "Should receive streaming events"
    assert planning_complete, "Should complete planning phase"
    assert PlanExecuteEventType.FINAL_ANSWER in events_received, "Should receive FINAL_ANSWER"
    assert final_answer is not None, "Should have final answer"

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
    print("Agent created with tools: search_web, analyze_data, generate_report")

    tests = [
        ("Basic Plan & Execute", test_basic),
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
            import traceback
            traceback.print_exc()

    return passed, failed, failures


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Manual test script for Plan & Execute mode with multiple providers"
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
    print("Plan & Execute Agent Manual Test Suite")
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
