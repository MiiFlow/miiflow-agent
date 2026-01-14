"""Plan & Execute example.

This example demonstrates the Plan & Execute pattern for complex tasks:
1. Basic usage - Breaking down complex tasks into subtasks
2. Streaming - Real-time event streaming during planning and execution
"""

import asyncio

from miiflow_agent import Agent, AgentType, LLMClient, RunContext, tool
from miiflow_agent.core.react import PlanExecuteEventType


# Define simple tools for the agent
@tool("search_web", "Search the web for information")
def search_web(query: str) -> str:
    """Search for information on the web."""
    # Simulated search results
    return f"""
    Search results for '{query}':
    1. Wikipedia article about {query}
    2. Recent news: {query} trends in 2024
    3. Expert analysis of {query}
    """


@tool("analyze_data", "Analyze data and provide insights")
def analyze_data(data: str, analysis_type: str = "summary") -> str:
    """Analyze provided data."""
    return f"""
    Analysis ({analysis_type}):
    - Data received: {data[:100]}...
    - Key findings: 3 main patterns identified
    - Confidence: High
    """


@tool("generate_report", "Generate a formatted report")
def generate_report(title: str, sections: str) -> str:
    """Generate a report from sections."""
    return f"""
    # {title}

    ## Executive Summary
    This report covers: {sections}

    ## Key Findings
    - Finding 1: Significant trends identified
    - Finding 2: Areas for improvement noted
    - Finding 3: Recommendations provided

    ## Conclusion
    Based on analysis, actionable steps are recommended.
    """


async def basic_plan_execute():
    """Basic Plan & Execute example."""
    client = LLMClient.create("openai", model="gpt-4o-mini")

    # Create agent with PLAN_EXECUTE type
    agent = Agent(
        client,
        agent_type=AgentType.PLAN_EXECUTE,
        max_iterations=10,
    )

    agent.add_tool(search_web)
    agent.add_tool(analyze_data)
    agent.add_tool(generate_report)

    # Complex task that benefits from planning
    task = """
    Research the latest trends in artificial intelligence,
    analyze the key developments, and create a brief report.
    """

    print(f"Task: {task.strip()}")
    print("\nExecuting with Plan & Execute pattern...\n")

    result = await agent.run(task)
    print(f"Result: {result.data}")


async def streaming_example():
    """Plan & Execute with real-time event streaming."""
    client = LLMClient.create("openai", model="gpt-4o-mini")

    agent = Agent(
        client,
        agent_type=AgentType.PLAN_EXECUTE,
        max_iterations=10,
    )

    agent.add_tool(search_web)
    agent.add_tool(analyze_data)

    task = "Find information about Python web frameworks and compare them"
    print(f"Task: {task}")
    print("=" * 50)

    context = RunContext(deps=None, messages=[])

    async for event in agent.stream(task, context):
        event_type = event.event_type

        if event_type == PlanExecuteEventType.PLANNING_START:
            print("\n[PHASE 1: PLANNING]")
            print("Creating execution plan...")

        elif event_type == PlanExecuteEventType.PLANNING_COMPLETE:
            subtask_count = event.data.get("subtask_count", 0)
            print(f"\nPlan created with {subtask_count} subtasks!")
            print("\n[PHASE 2: EXECUTION]")

        elif event_type == PlanExecuteEventType.SUBTASK_START:
            desc = event.data.get("description", "")[:60]
            print(f"\n  Subtask: {desc}...")

        elif event_type == PlanExecuteEventType.SUBTASK_COMPLETE:
            result_preview = str(event.data.get("result", ""))[:80]
            print(f"  Done: {result_preview}...")

        elif event_type == PlanExecuteEventType.FINAL_ANSWER:
            print("\n" + "=" * 50)
            print("[PHASE 3: FINAL ANSWER]")
            print("=" * 50)
            answer = event.data.get("answer", "")
            print(answer[:500] + "..." if len(answer) > 500 else answer)


if __name__ == "__main__":
    print("=== Basic Plan & Execute ===")
    asyncio.run(basic_plan_execute())

    print("\n" + "=" * 50 + "\n")

    print("=== Streaming Example ===")
    asyncio.run(streaming_example())
