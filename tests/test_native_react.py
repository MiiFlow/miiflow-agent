"""Test script for native tool calling ReAct agent."""

import asyncio
import logging
import os
from typing import Dict, Any

import pytest

from miiflow_agent import LLMClient
from miiflow_agent.core import Agent, AgentType
from miiflow_agent.core.tools import tool

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Define test tools
@tool(name="multiply", description="Multiply two numbers together")
def multiply(a: float, b: float) -> Dict[str, Any]:
    """Multiply two numbers."""
    result = a * b
    return {"success": True, "result": result}


@tool(name="add", description="Add two numbers together")
def add(a: float, b: float) -> Dict[str, Any]:
    """Add two numbers."""
    result = a + b
    return {"success": True, "result": result}


@tool(name="subtract", description="Subtract second number from first")
def subtract(a: float, b: float) -> Dict[str, Any]:
    """Subtract two numbers."""
    result = a - b
    return {"success": True, "result": result}


def has_openai_key() -> bool:
    """Check if OpenAI API key is available."""
    return bool(os.getenv("OPENAI_API_KEY"))


@pytest.mark.skipif(not has_openai_key(), reason="OPENAI_API_KEY not set - skipping real LLM tests")
@pytest.mark.asyncio
async def test_native_react():
    """Test ReAct agent with native tool calling."""

    logger.info("=" * 80)
    logger.info("Testing Native Tool Calling ReAct Agent")
    logger.info("=" * 80)

    # Create LLM client (OpenAI)
    client = LLMClient.create("openai", model="gpt-4o-mini")

    # Create agent
    agent = Agent(
        client=client,
        agent_type=AgentType.REACT,
        tools=[multiply, add, subtract],
        max_iterations=10,
        temperature=0.7
    )

    logger.info("\n✓ Agent created")

    # Get tool names safely
    tool_names = []
    for tool in agent._tools:
        if hasattr(tool, 'name'):
            tool_names.append(tool.name)
        elif hasattr(tool, '__name__'):
            tool_names.append(tool.__name__)
        else:
            tool_names.append(str(tool))

    logger.info(f"✓ Registered tools: {tool_names}")

    # Test query
    query = "What is 15 multiplied by 27, and then add 100 to that result?"

    logger.info(f"\n📝 Query: {query}")
    logger.info("\n" + "=" * 80)
    logger.info("Starting ReAct execution...")
    logger.info("=" * 80 + "\n")

    # Track events
    thinking_chunks_count = 0
    tool_calls = []
    final_answer = None

    try:
        # Stream ReAct execution
        from miiflow_agent.core.agent import RunContext
        context = RunContext(deps=None)

        async for event in agent.stream(query, context, max_steps=10):
            event_type = event.event_type.value

            if event_type == "step_start":
                logger.info(f"\n🔄 Step {event.step_number} starting...")

            elif event_type == "thinking_chunk":
                thinking_chunks_count += 1
                delta = event.data.get('delta', '')
                print(delta, end='', flush=True)

            elif event_type == "thought":
                thought = event.data.get('thought', '')
                logger.info(f"\n\n💭 Complete thought: {thought[:200]}...")

            elif event_type == "action_planned":
                action = event.data.get('action', '')
                inputs = event.data.get('inputs', {})
                logger.info(f"\n🔧 Tool call planned: {action}({inputs})")
                tool_calls.append((action, inputs))

            elif event_type == "observation":
                observation = event.data.get('observation', '')
                logger.info(f"📊 Observation: {observation}")

            elif event_type == "final_answer":
                final_answer = event.data.get('answer', '')
                logger.info(f"\n\n✅ Final Answer: {final_answer}")

            elif event_type == "step_complete":
                logger.info(f"✓ Step {event.step_number} completed")

    except Exception as e:
        logger.error(f"\n❌ Error during execution: {e}", exc_info=True)
        return False

    # Results summary
    logger.info("\n" + "=" * 80)
    logger.info("Execution Summary")
    logger.info("=" * 80)
    logger.info(f"✓ Thinking chunks streamed: {thinking_chunks_count}")
    logger.info(f"✓ Tool calls executed: {len(tool_calls)}")
    for tool_name, inputs in tool_calls:
        logger.info(f"  - {tool_name}: {inputs}")
    logger.info(f"✓ Final answer: {final_answer[:100] if final_answer else 'None'}...")

    # Validate results
    success = True
    if thinking_chunks_count == 0:
        logger.error("❌ FAIL: No thinking chunks were streamed")
        success = False
    else:
        logger.info(f"✅ PASS: Thinking chunks were streamed ({thinking_chunks_count} chunks)")

    if len(tool_calls) == 0:
        logger.error("❌ FAIL: No tool calls were executed")
        success = False
    else:
        logger.info(f"✅ PASS: Tool calls were executed ({len(tool_calls)} calls)")

    if not final_answer:
        logger.error("❌ FAIL: No final answer received")
        success = False
    else:
        logger.info("✅ PASS: Final answer received")

        # Check if answer contains expected result (15 * 27 = 405, 405 + 100 = 505)
        if "505" in final_answer:
            logger.info("✅ PASS: Answer contains correct result (505)")
        else:
            logger.warning(f"⚠️  WARNING: Answer might not contain expected result (505)")

    logger.info("\n" + "=" * 80)
    if success:
        logger.info("🎉 All tests passed!")
    else:
        logger.error("❌ Some tests failed")
    logger.info("=" * 80)

    return success


async def main():
    """Run all tests."""

    # Test native tool calling
    native_success = await test_native_react()

    logger.info("\n\n" + "=" * 80)
    logger.info("Final Results")
    logger.info("=" * 80)
    logger.info(f"ReAct Agent: {'✅ PASS' if native_success else '❌ FAIL'}")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
