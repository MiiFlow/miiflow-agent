"""Plan & Execute example.

This example demonstrates the Plan & Execute pattern for complex tasks:
- Breaking down complex tasks into subtasks
- Executing subtasks with dependencies
- Re-planning on failure
- Synthesizing results
"""

import asyncio
from miiflow_llm import LLMClient, Agent, AgentType, RunContext, tool
from miiflow_llm.core.react import (
    ReActFactory,
    PlanAndExecuteOrchestrator,
    EventBus,
    PlanExecuteEventType,
)


# Define tools for the agent
@tool("search_web", "Search the web for information")
def search_web(query: str) -> str:
    """Search for information on the web."""
    # Simulated search
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

    # Create agent with tools
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


async def plan_execute_with_events():
    """Plan & Execute with real-time event streaming."""
    client = LLMClient.create("openai", model="gpt-4o-mini")

    agent = Agent(
        client,
        agent_type=AgentType.PLAN_EXECUTE,
        max_iterations=10,
    )

    agent.add_tool(search_web)
    agent.add_tool(analyze_data)

    # Event handler for real-time updates
    def on_event(event):
        event_type = event.event_type

        if event_type == PlanExecuteEventType.PLANNING_START:
            print("[Planning] Starting to create execution plan...")

        elif event_type == PlanExecuteEventType.PLANNING_COMPLETE:
            plan_data = event.data.get("plan", {})
            subtask_count = event.data.get("subtask_count", 0)
            print(f"[Planning] Plan created with {subtask_count} subtasks")

        elif event_type == PlanExecuteEventType.SUBTASK_START:
            desc = event.data.get("description", "")
            print(f"[Executing] Starting subtask: {desc[:50]}...")

        elif event_type == PlanExecuteEventType.SUBTASK_COMPLETE:
            result = event.data.get("result", "")[:50]
            print(f"[Executing] Subtask completed: {result}...")

        elif event_type == PlanExecuteEventType.FINAL_ANSWER:
            print("[Complete] Final answer generated")

    # Note: In practice, subscribe to orchestrator's event bus
    # This is a demonstration of the event types

    task = "Find information about Python web frameworks and compare them"
    print(f"Task: {task}\n")

    result = await agent.run(task)
    print(f"\nFinal Result: {result.data[:200]}...")


async def plan_execute_manual_orchestrator():
    """Using PlanAndExecuteOrchestrator directly for more control."""
    from miiflow_llm.core.react.tool_executor import AgentToolExecutor
    from miiflow_llm.core.react.safety import SafetyManager

    client = LLMClient.create("openai", model="gpt-4o-mini")

    # Create agent
    agent = Agent(client)
    agent.add_tool(search_web)
    agent.add_tool(analyze_data)
    agent.add_tool(generate_report)

    # Create components manually
    event_bus = EventBus()
    safety_manager = SafetyManager(max_steps=10)
    tool_executor = AgentToolExecutor(agent, agent.tool_registry)

    # Create ReAct orchestrator for subtask execution
    react_orchestrator = ReActFactory.create_orchestrator(
        agent=agent,
        max_steps=5,
        use_native_tools=True,
    )

    # Create Plan & Execute orchestrator
    orchestrator = PlanAndExecuteOrchestrator(
        tool_executor=tool_executor,
        event_bus=event_bus,
        safety_manager=safety_manager,
        subtask_orchestrator=react_orchestrator,
        max_replans=2,
        use_react_for_subtasks=True,
    )

    # Subscribe to events
    events_received = []

    def track_events(event):
        events_received.append(event.event_type)

    event_bus.subscribe(track_events)

    # Execute
    context = RunContext(deps=None, messages=[])
    result = await orchestrator.execute(
        "Create a comparison of Python and JavaScript for web development",
        context,
    )

    print(f"Final answer: {result.final_answer[:200]}...")
    print(f"Stop reason: {result.stop_reason}")
    print(f"Replans: {result.replans}")
    print(f"Events received: {len(events_received)}")


async def plan_execute_with_existing_plan():
    """Provide a pre-generated plan to skip planning phase."""
    from miiflow_llm.core.react import Plan, SubTask
    from miiflow_llm.core.react.tool_executor import AgentToolExecutor
    from miiflow_llm.core.react.safety import SafetyManager

    client = LLMClient.create("openai", model="gpt-4o-mini")

    agent = Agent(client)
    agent.add_tool(search_web)
    agent.add_tool(generate_report)

    # Pre-define a plan
    existing_plan = Plan(
        goal="Research and report on cloud computing",
        reasoning="User wants a structured report on cloud computing",
        subtasks=[
            SubTask(
                id=1,
                description="Search for cloud computing trends",
                required_tools=["search_web"],
                dependencies=[],
                success_criteria="Found relevant information",
            ),
            SubTask(
                id=2,
                description="Generate report from findings",
                required_tools=["generate_report"],
                dependencies=[1],
                success_criteria="Report created",
            ),
        ],
    )

    # Create orchestrator
    event_bus = EventBus()
    safety_manager = SafetyManager(max_steps=10)
    tool_executor = AgentToolExecutor(agent, agent.tool_registry)

    orchestrator = PlanAndExecuteOrchestrator(
        tool_executor=tool_executor,
        event_bus=event_bus,
        safety_manager=safety_manager,
        max_replans=1,
        use_react_for_subtasks=False,  # Direct tool execution
    )

    context = RunContext(deps=None, messages=[])

    print("Executing with pre-defined plan...")
    print(f"Plan has {len(existing_plan.subtasks)} subtasks\n")

    # Pass existing plan to skip planning phase
    result = await orchestrator.execute(
        "Research cloud computing",
        context,
        existing_plan=existing_plan,
    )

    print(f"Result: {result.final_answer[:200]}...")


if __name__ == "__main__":
    print("=== Basic Plan & Execute ===")
    asyncio.run(basic_plan_execute())

    print("\n" + "=" * 50 + "\n")

    print("=== Plan & Execute with Events ===")
    asyncio.run(plan_execute_with_events())

    print("\n" + "=" * 50 + "\n")

    print("=== Manual Orchestrator Usage ===")
    asyncio.run(plan_execute_manual_orchestrator())

    print("\n" + "=" * 50 + "\n")

    print("=== Pre-defined Plan ===")
    asyncio.run(plan_execute_with_existing_plan())
