#!/usr/bin/env python3
"""Test fuzzy tool name matching for hallucination resilience."""

import sys
import os

# Add package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "packages/miiflow-llm"))

from miiflow_llm.core.react.orchestrator import ReActOrchestrator
from miiflow_llm.core.react.tool_executor import AgentToolExecutor
from miiflow_llm.core.react.events import EventBus
from miiflow_llm.core.react.safety import SafetyManager
from miiflow_llm.core.react.parser import ReActParser


class MockToolExecutor:
    """Mock tool executor for testing."""

    def list_tools(self):
        return ["Addition", "Multiply", "First level encoding", "Second level encoding",
                "Find Accounts", "Import Contact to Account", "Create Account"]


def test_fuzzy_matching():
    """Test that fuzzy matching works for common hallucinations."""

    # Create a minimal orchestrator for testing
    mock_executor = MockToolExecutor()
    event_bus = EventBus()
    safety_manager = SafetyManager()
    parser = ReActParser()

    orchestrator = ReActOrchestrator(
        tool_executor=mock_executor,
        event_bus=event_bus,
        safety_manager=safety_manager,
        parser=parser
    )

    # Test cases: (hallucinated_name, expected_correction)
    test_cases = [
        ("Add", "Addition"),  # Substring match - main hallucination case
        ("add", "Addition"),  # Case insensitive
        ("Multiply", "Multiply"),  # Exact match
        ("multiply", "Multiply"),  # Case insensitive exact
        ("First level", "First level encoding"),  # Substring match
        ("encoding", "First level encoding"),  # Substring match (should pick first)
        ("Additoin", "Addition"),  # 1 char swap typo
        ("NonExistent", None),  # Should return None
    ]

    print("Testing fuzzy tool name matching...\n")

    all_passed = True
    for hallucinated, expected in test_cases:
        result = orchestrator._find_similar_tool(hallucinated)
        passed = result == expected
        all_passed = all_passed and passed

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: '{hallucinated}' -> '{result}' (expected: '{expected}')")

    print(f"\n{'='*60}")
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    test_fuzzy_matching()
