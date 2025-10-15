#!/usr/bin/env python3
"""
Test script for ReAct orchestrator streaming behavior.

This script demonstrates real-time streaming of thinking chunks during ReAct execution.
It creates a simple agent with a calculator tool and shows how chunks are streamed
as the agent reasons about the problem.

Usage:
    poetry run python examples/test_react_streaming.py
"""

import asyncio
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miiflow_llm import AgentConfig, AgentContext, create_agent
from miiflow_llm.core.agent import AgentType
from miiflow_llm.core.tools import get_tool_from_function, tool


# ANSI color codes for pretty output
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"


def print_colored(text: str, color: str = Colors.ENDC):
    """Print colored text."""
    print(f"{color}{text}{Colors.ENDC}")


def print_event_header(event_type: str, step_number: int):
    """Print a formatted event header."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print_colored(
        f"\n[{timestamp}] {event_type.upper()} (Step {step_number})", Colors.BOLD + Colors.CYAN
    )


def print_thinking_chunk(delta: str, show_inline: bool = True):
    """Print a thinking chunk (inline or newline)."""
    if show_inline:
        print(f"{Colors.DIM}{delta}{Colors.ENDC}", end="", flush=True)
    else:
        print_colored(f"  Chunk: {delta}", Colors.DIM)


def print_json(data: Dict[str, Any], indent: int = 2):
    """Print formatted JSON data."""
    import json

    formatted = json.dumps(data, indent=indent)
    print_colored(formatted, Colors.GREEN)


# Define simple calculator tools
@tool("add", "Add two numbers together")
def add_tool(a: float, b: float) -> float:
    """Add two numbers together.

    Args:
        a: First number
        b: Second number

    Returns:
        The sum of a and b
    """
    result = a + b
    print_colored(f"\n  🔧 Tool Execution: add({a}, {b}) = {result}", Colors.YELLOW)
    return result


@tool("multiply", "Multiply two numbers together")
def multiply_tool(a: float, b: float) -> float:
    """Multiply two numbers together.

    Args:
        a: First number
        b: Second number

    Returns:
        The product of a and b
    """
    result = a * b
    print_colored(f"\n  🔧 Tool Execution: multiply({a}, {b}) = {result}", Colors.YELLOW)
    return result


async def test_react_streaming(prompt: str, max_steps: int = 10, show_inline_chunks: bool = True):
    """
    Test ReAct orchestrator streaming with a given prompt.

    Args:
        prompt: The user query
        max_steps: Maximum reasoning steps
        show_inline_chunks: If True, print chunks inline; otherwise print each chunk on new line
    """
    print_colored("=" * 80, Colors.HEADER)
    print_colored("🚀 ReAct Orchestrator Streaming Test", Colors.BOLD + Colors.HEADER)
    print_colored("=" * 80, Colors.HEADER)

    # Check for API key
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print_colored("\n❌ Error: No API key found!", Colors.RED)
        print("Please set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        return

    # Configure agent
    provider = "openai" if os.getenv("OPENAI_API_KEY") else "anthropic"
    model = "gpt-4o-mini" if provider == "openai" else "claude-3-5-sonnet-20241022"

    print_colored(f"\n📋 Configuration:", Colors.BOLD)
    print_colored(f"  Provider: {provider}", Colors.BLUE)
    print_colored(f"  Model: {model}", Colors.BLUE)
    print_colored(f"  Max Steps: {max_steps}", Colors.BLUE)
    print_colored(f"  Query: {prompt}", Colors.BLUE)

    # Create tools using the decorated functions
    from miiflow_llm.core.tools import FunctionTool

    add = get_tool_from_function(add_tool)
    multiply = get_tool_from_function(multiply_tool)

    # Create agent config
    config = AgentConfig(
        provider=provider,
        model=model,
        agent_type=AgentType.REACT,
        tools=[add, multiply],
        system_prompt="You are a helpful calculator assistant. Use the provided tools to solve math problems step by step. Use XML tags (<thinking>, <tool_call>, <answer>) to structure your responses.",
        temperature=0.7,
        max_iterations=max_steps,
    )

    # Create agent
    agent = create_agent(config)
    context = AgentContext()

    # Track events
    event_counts = defaultdict(int)
    thinking_chunks = []
    start_time = datetime.now()
    current_step_chunks = []

    print_colored(f"\n{'='*80}", Colors.HEADER)
    print_colored("📊 STREAMING EVENTS", Colors.BOLD + Colors.HEADER)
    print_colored(f"{'='*80}", Colors.HEADER)

    try:
        # Stream ReAct execution
        async for event in agent.stream_react(prompt=prompt, context=context):
            event_type = (
                event.event_type.value
                if hasattr(event.event_type, "value")
                else str(event.event_type)
            )
            event_counts[event_type] += 1

            if event_type == "step_start":
                print_event_header("Step Started", event.step_number)
                current_step_chunks = []

            elif event_type == "thinking_chunk":
                # This is the key streaming event!
                delta = event.data.get("delta", "")
                thinking_chunks.append(delta)
                current_step_chunks.append(delta)
                print_thinking_chunk(delta, show_inline_chunks)

            elif event_type == "thought":
                # Complete thought after all chunks
                if not show_inline_chunks:
                    print_colored(f"\n  💭 Complete Thought:", Colors.GREEN)
                    print_colored(f"     {event.data.get('thought', '')}", Colors.GREEN)
                else:
                    # Just add a newline after inline chunks
                    print()

                if current_step_chunks:
                    print_colored(
                        f"  ✨ Streamed {len(current_step_chunks)} chunks for this thought",
                        Colors.DIM,
                    )

            elif event_type == "action_planned":
                print_colored(f"\n  🎯 Action Planned:", Colors.YELLOW)
                action = event.data.get("action", "")
                action_input = event.data.get("action_input", {})
                print_colored(f"     Tool: {action}", Colors.YELLOW)
                print_colored(f"     Input: {action_input}", Colors.YELLOW)

            elif event_type == "observation":
                print_colored(f"\n  👁️  Observation:", Colors.GREEN)
                observation = event.data.get("observation", "")
                success = event.data.get("success", True)
                status = "✅ Success" if success else "❌ Failed"
                print_colored(f"     {status}: {observation}", Colors.GREEN)

            elif event_type == "final_answer":
                print_event_header("Final Answer", event.step_number)
                answer = event.data.get("answer", "")
                print_colored(f"\n  🎉 {answer}", Colors.BOLD + Colors.GREEN)

            elif event_type == "final_answer_chunk":
                # For future streaming of final answer
                delta = event.data.get("delta", "")
                print_thinking_chunk(delta, show_inline_chunks)

            elif event_type == "error":
                print_event_header("Error", event.step_number)
                error = event.data.get("error", "Unknown error")
                print_colored(f"  ❌ {error}", Colors.RED)

            elif event_type == "stop_condition":
                stop_reason = event.data.get("stop_reason", "unknown")
                print_colored(f"\n  ⏹️  Stopped: {stop_reason}", Colors.YELLOW)

    except Exception as e:
        print_colored(f"\n❌ Error during execution: {e}", Colors.RED)
        import traceback

        traceback.print_exc()
        return

    # Print summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print_colored(f"\n{'='*80}", Colors.HEADER)
    print_colored("📈 EXECUTION SUMMARY", Colors.BOLD + Colors.HEADER)
    print_colored(f"{'='*80}", Colors.HEADER)

    print_colored(f"\n⏱️  Duration: {duration:.2f} seconds", Colors.BLUE)
    print_colored(f"🔢 Total Events: {sum(event_counts.values())}", Colors.BLUE)

    print_colored(f"\n📊 Event Breakdown:", Colors.BOLD)
    for event_type, count in sorted(event_counts.items()):
        icon = "✨" if event_type == "thinking_chunk" else "📌"
        color = Colors.GREEN if event_type == "thinking_chunk" else Colors.BLUE
        print_colored(f"  {icon} {event_type}: {count}", color)

    # Highlight streaming performance
    chunk_count = event_counts.get("thinking_chunk", 0)
    if chunk_count > 0:
        print_colored(f"\n💡 Streaming Performance:", Colors.BOLD + Colors.GREEN)
        print_colored(f"  ✅ {chunk_count} thinking chunks streamed in real-time", Colors.GREEN)
        print_colored(f"  ✅ Average: {chunk_count / duration:.1f} chunks/second", Colors.GREEN)
        print_colored(f"  ✅ Users saw real-time thinking progress!", Colors.GREEN)
    else:
        print_colored(f"\n⚠️  WARNING: No thinking chunks detected!", Colors.RED)
        print_colored(f"  Streaming may not be working correctly", Colors.RED)


async def main():
    """Run test scenarios."""
    # Test 1: Simple arithmetic
    await test_react_streaming(
        prompt="What is 15 + 27?",
        max_steps=5,
        show_inline_chunks=True,  # Show chunks inline for realistic UX
    )

    print("\n\n")

    # Test 2: Multi-step reasoning
    await test_react_streaming(
        prompt="Calculate (12 + 8) * 5 + 10 * 20 + 30. Show me the steps.",
        max_steps=10,
        show_inline_chunks=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
