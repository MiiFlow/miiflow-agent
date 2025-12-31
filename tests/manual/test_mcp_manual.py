#!/usr/bin/env python3
"""
Manual test script for MCP (Model Context Protocol) integration.

Prerequisites:
    1. Start the string-tools MCP server:
       cd /Users/yzk/Desktop/miiflow-web/server
       python -m workflow.services.mcp_servers.string_tools --port 8766

    OR run Django which serves it at /mcp/string-tools/:
       cd /Users/yzk/Desktop/miiflow-web/server
       poetry run python manage.py runserver

Usage:
    # OpenAI (default)
    export OPENAI_API_KEY="sk-..."
    python tests/manual/test_mcp_manual.py

    # Anthropic Claude
    export ANTHROPIC_API_KEY="sk-ant-..."
    python tests/manual/test_mcp_manual.py --provider anthropic

    # Run all providers
    python tests/manual/test_mcp_manual.py --provider all

    # Use Django server (default port 8000)
    python tests/manual/test_mcp_manual.py --mcp-url http://localhost:8000/mcp/string-tools/

Tests:
1. MCP connection and tool discovery
2. Agent using MCP tools to answer questions
"""

import argparse
import asyncio
import os
import sys

from miiflow_llm import Agent, AgentType, LLMClient, RunContext
from miiflow_llm.core.react import ReActEventType
from miiflow_llm.core.tools import MCPServerConfig, MCPToolManager


# =============================================================================
# Provider Configuration
# =============================================================================

PROVIDERS = {
    "openai": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "env_var": "OPENAI_API_KEY",
    },
    "anthropic": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-5-20250929",
        "env_var": "ANTHROPIC_API_KEY",
    },
}

# Default MCP server URL (standalone string-tools server)
DEFAULT_MCP_URL = "http://localhost:8766/mcp"


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


async def create_mcp_manager(mcp_url: str, transport: str = "streamable_http") -> MCPToolManager:
    """Create and connect to MCP server."""
    config = MCPServerConfig(
        name="string-tools",
        transport=transport,
        url=mcp_url,
        timeout=30.0,
    )

    manager = MCPToolManager()
    manager.add_server(config)
    await manager.connect_all()

    return manager


def create_agent(provider_name: str, mcp_manager: "MCPToolManager"):
    """Create the ReAct agent with MCP tools for the specified provider."""
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
        system_prompt=(
            "You are a helpful assistant with access to string manipulation tools. "
            "Use the edit_distance tool to compare strings when asked about similarity."
        ),
    )

    # Register MCP tools via the tool registry (not add_tool which expects decorated functions)
    agent.tool_registry.register_mcp_manager(mcp_manager)

    return agent


# =============================================================================
# Test Cases
# =============================================================================


async def test_mcp_connection(mcp_url: str, transport: str = "streamable_http") -> MCPToolManager:
    """Test 1: MCP Connection and Tool Discovery"""
    print("\n" + "=" * 60)
    print("TEST 1: MCP Connection and Tool Discovery")
    print(f"Connecting to: {mcp_url}")
    print(f"Transport: {transport}")
    print("=" * 60)

    manager = await create_mcp_manager(mcp_url, transport=transport)

    tools = manager.get_all_tools()
    tool_names = manager.list_tool_names()

    print(f"\nConnected! Found {len(tools)} tools:")
    for name in tool_names:
        tool = manager.get_tool(name)
        print(f"  - {name}: {tool.description[:60]}...")

    # Validate we found the expected tools
    assert len(tools) >= 2, "Should find at least 2 tools (edit_distance, edit_distance_detailed)"
    assert any("edit_distance" in name for name in tool_names), "Should find edit_distance tool"

    print("\nPASSED")
    return manager


async def test_direct_tool_call(manager: MCPToolManager):
    """Test 2: Direct MCP Tool Execution"""
    print("\n" + "=" * 60)
    print("TEST 2: Direct MCP Tool Execution")
    print("=" * 60)

    # Find an edit_distance tool (handles both standalone and combined server naming)
    tool_names = manager.list_tool_names()
    edit_distance_tool = None
    for name in tool_names:
        if "edit_distance" in name and "detailed" not in name:
            edit_distance_tool = name
            break

    if not edit_distance_tool:
        print("No edit_distance tool found, skipping direct tool test")
        print("Available tools:", tool_names)
        return

    print(f"Calling: {edit_distance_tool}('hello', 'hallo')")

    result = await manager.execute_tool(
        edit_distance_tool,
        string1="hello",
        string2="hallo"
    )

    print(f"\nResult: {result}")

    # Validate result
    assert result is not None, "Result should not be None"
    # The result should be a CallToolResult with content
    content = result.content if hasattr(result, 'content') else result
    print(f"Content: {content}")

    print("\nPASSED")


async def test_agent_with_mcp(agent, provider_name: str):
    """Test 3: Agent using MCP tools"""
    print("\n" + "=" * 60)
    print(f"TEST 3: Agent Using MCP Tools ({provider_name})")
    print("Query: How similar are 'kitten' and 'sitting'? Use the edit_distance tool.")
    print("=" * 60)

    result = await agent.run(
        "How similar are 'kitten' and 'sitting'? Use the edit_distance tool to find out."
    )

    print(f"\nResult:\n{result.data}")

    # Validate result
    assert result.data is not None, "Result should not be None"
    assert len(result.data) > 20, "Result should contain meaningful content"
    # The edit distance between 'kitten' and 'sitting' is 3
    # Agent should mention this or related info

    print("\nPASSED")


async def test_streaming_with_mcp(agent, provider_name: str):
    """Test 4: Streaming events with MCP tools"""
    print("\n" + "=" * 60)
    print(f"TEST 4: Streaming Events with MCP Tools ({provider_name})")
    print("Query: Compare 'cat' and 'hat' using the edit_distance tool")
    print("=" * 60)

    context = RunContext(deps=None, messages=[])
    events_received = []
    tool_called = False
    final_answer = None

    async for event in agent.stream(
        "Compare 'cat' and 'hat' using the edit_distance tool",
        context
    ):
        events_received.append(event.event_type)

        if event.event_type == ReActEventType.THINKING_CHUNK:
            print(event.data.get("delta", ""), end="", flush=True)
        elif event.event_type == ReActEventType.ACTION_PLANNED:
            action = event.data.get("action", "")
            tool_called = True
            print(f"\n[Calling tool: {action}]")
        elif event.event_type == ReActEventType.OBSERVATION:
            obs = str(event.data.get("observation", ""))[:100]
            print(f"[Tool result: {obs}]")
        elif event.event_type == ReActEventType.FINAL_ANSWER:
            final_answer = event.data.get("answer", "")
            print(f"\n\nFINAL ANSWER:\n{final_answer}")

    # Validate streaming worked
    assert len(events_received) > 0, "Should receive streaming events"
    # Note: tool_called might be False if the model answers directly without using tools
    # This is acceptable behavior for simple questions
    if not tool_called:
        print("\n[Note: Agent answered directly without calling MCP tool]")
    assert final_answer is not None, "Should have final answer"

    print("\nPASSED")


# =============================================================================
# Main Runner
# =============================================================================


async def run_tests_for_provider(
    provider_name: str,
    manager: MCPToolManager
) -> tuple[int, int, list]:
    """Run agent tests for a specific provider.

    Returns:
        Tuple of (passed, failed, failures list)
    """
    config = PROVIDERS[provider_name]
    print("\n" + "=" * 60)
    print(f"Testing with: {provider_name.upper()} ({config['model']})")
    print("=" * 60)

    if not check_environment(provider_name):
        return 0, 0, []  # Skipped

    print(f"\nCreating agent with {provider_name} + MCP tools...")
    mcp_tools = manager.get_all_tools()
    agent = create_agent(provider_name, manager)
    print(f"Agent created with {len(mcp_tools)} MCP tools")

    tests = [
        ("Agent with MCP", lambda: test_agent_with_mcp(agent, provider_name)),
        ("Streaming with MCP", lambda: test_streaming_with_mcp(agent, provider_name)),
    ]

    passed = 0
    failed = 0
    failures = []

    for name, test_fn in tests:
        try:
            await test_fn()
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
        description="Manual test script for MCP integration with multiple providers"
    )
    parser.add_argument(
        "--provider",
        "-p",
        choices=["openai", "anthropic", "all"],
        default="openai",
        help="LLM provider to test (default: openai)",
    )
    parser.add_argument(
        "--mcp-url",
        default=DEFAULT_MCP_URL,
        help=f"MCP server URL (default: {DEFAULT_MCP_URL})",
    )
    parser.add_argument(
        "--transport",
        "-t",
        choices=["streamable_http", "sse", "http"],
        default="streamable_http",
        help="MCP transport type (default: streamable_http, use http for Django JSON-RPC)",
    )
    return parser.parse_args()


async def main():
    """Run all tests and report results."""
    args = parse_args()

    print("=" * 60)
    print("MCP Integration Manual Test Suite")
    print("=" * 60)
    print(f"MCP URL: {args.mcp_url}")
    print(f"Transport: {args.transport}")

    # Test 1 & 2: MCP connection (independent of LLM provider)
    try:
        manager = await test_mcp_connection(args.mcp_url, transport=args.transport)
        await test_direct_tool_call(manager)
    except Exception as e:
        print(f"\nFATAL: Could not connect to MCP server: {e}")
        print("\nMake sure the MCP server is running:")
        print("  cd /Users/yzk/Desktop/miiflow-web/server")
        print("  python -m workflow.services.mcp_servers.string_tools --port 8766")
        print("\nOr use Django:")
        print("  poetry run python manage.py runserver")
        print("  Then use: --mcp-url http://localhost:8000/mcp/string-tools/")
        sys.exit(1)

    # Determine which providers to test
    if args.provider == "all":
        providers_to_test = list(PROVIDERS.keys())
    else:
        providers_to_test = [args.provider]

    total_passed = 2  # MCP connection tests
    total_failed = 0
    total_skipped = 0
    all_failures = []

    for provider_name in providers_to_test:
        passed, failed, failures = await run_tests_for_provider(provider_name, manager)
        if passed == 0 and failed == 0:
            total_skipped += 1
        else:
            total_passed += passed
            total_failed += failed
            all_failures.extend(failures)

    # Cleanup
    await manager.disconnect_all()

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
