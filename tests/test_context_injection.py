#!/usr/bin/env python3
"""
Test that context injection works correctly for tools that need it.

This test verifies that the orchestrator correctly detects which tools need
context and passes it appropriately.
"""

import asyncio
import sys
import os

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from miiflow_agent.core.tools import tool, get_tool_from_function
from miiflow_agent.core.react.tool_executor import AgentToolExecutor
from miiflow_agent.core.tools.registry import ToolRegistry


# Tool without context (like test_react_streaming tools)
@tool("add_simple", "Add two numbers")
def add_simple(a: float, b: float) -> float:
    """Add two numbers without context."""
    return a + b


# Tool WITH context injection (like CRM tools)
@tool("add_with_context", "Add two numbers with context")
def add_with_context(ctx, a: float, b: float) -> dict:
    """Add two numbers with context injection.

    Args:
        ctx: Context (auto-injected)
        a: First number
        b: Second number
    """
    # Access context
    org_id = ctx.deps.get("organization_id", "unknown")
    result = a + b
    return {
        "success": True,
        "result": result,
        "organization_id": org_id
    }


def test_context_detection():
    """Test that tool_needs_context correctly detects context patterns."""
    print("\n" + "=" * 80)
    print("TEST: Context Detection")
    print("=" * 80)

    # Create mock agent with registry
    class MockAgent:
        def __init__(self):
            self.tool_registry = ToolRegistry()
            self.temperature = 0.7
            self.client = None  # Mock client

    mock_agent = MockAgent()

    # Register tools
    tool_simple = get_tool_from_function(add_simple)
    tool_with_ctx = get_tool_from_function(add_with_context)

    mock_agent.tool_registry.register(tool_simple)
    mock_agent.tool_registry.register(tool_with_ctx)

    # Create executor
    executor = AgentToolExecutor(mock_agent)

    # Check context detection
    needs_ctx_simple = executor.tool_needs_context("add_simple")
    needs_ctx_with = executor.tool_needs_context("add_with_context")

    print(f"\n✓ Tool 'add_simple' needs context: {needs_ctx_simple}")
    print(f"✓ Tool 'add_with_context' needs context: {needs_ctx_with}")

    assert needs_ctx_simple == False, "add_simple should NOT need context"
    assert needs_ctx_with == True, "add_with_context SHOULD need context"

    print("\n✅ Context detection working correctly!")
    print("  - Tools without ctx parameter: detected as not needing context")
    print("  - Tools with ctx parameter: detected as needing context")


async def test_tool_execution():
    """Test that tools execute correctly with/without context."""
    print("\n" + "=" * 80)
    print("TEST: Tool Execution")
    print("=" * 80)

    # Create mock agent with registry
    class MockAgent:
        def __init__(self):
            self.tool_registry = ToolRegistry()
            self.temperature = 0.7
            self.client = None  # Mock client

    mock_agent = MockAgent()

    # Register tools
    tool_simple = get_tool_from_function(add_simple)
    tool_with_ctx = get_tool_from_function(add_with_context)

    mock_agent.tool_registry.register(tool_simple)
    mock_agent.tool_registry.register(tool_with_ctx)

    # Create executor
    executor = AgentToolExecutor(mock_agent)

    # Test 1: Execute simple tool without context
    print("\n1. Testing simple tool (no context needed):")
    result1 = await executor.execute_tool("add_simple", {"a": 10, "b": 20}, context=None)
    print(f"   Result: {result1.output}")
    print(f"   Success: {result1.success}")
    assert result1.success == True
    assert result1.output == 30
    print("   ✅ Simple tool works without context")

    # Test 2: Execute context tool WITH context
    print("\n2. Testing context tool (context needed):")

    # Create mock context
    class MockContext:
        def __init__(self):
            self.deps = {"organization_id": "org_12345"}

    mock_context = MockContext()

    result2 = await executor.execute_tool("add_with_context", {"a": 15, "b": 27}, context=mock_context)
    print(f"   Result: {result2.output}")
    print(f"   Success: {result2.success}")
    assert result2.success == True
    assert result2.output["result"] == 42
    assert result2.output["organization_id"] == "org_12345"
    print("   ✅ Context tool works with context injection")
    print(f"   ✅ Context was injected: org_id = {result2.output['organization_id']}")

    return True


async def main():
    """Run all tests."""
    print("\n╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "CONTEXT INJECTION TESTS" + " " * 35 + "║")
    print("╚" + "=" * 78 + "╝")

    try:
        # Test 1: Context detection
        test_context_detection()

        # Test 2: Tool execution
        await test_tool_execution()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nSummary:")
        print("  - Context detection works correctly ✓")
        print("  - Tools without context execute correctly ✓")
        print("  - Tools with context receive context injection ✓")
        print("\nThe orchestrator will correctly:")
        print("  - Pass context=None to simple tools (test_react_streaming)")
        print("  - Pass context=RunContext to CRM tools (assistant)")
        print()

        return 0

    except AssertionError as e:
        print("\n" + "=" * 80)
        print("❌ TEST FAILED!")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ UNEXPECTED ERROR!")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
