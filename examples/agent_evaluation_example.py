"""Example demonstrating agent evaluation with Phoenix observability."""

import os
import asyncio
from typing import List

# Set up Phoenix before importing miiflow-agent
os.environ["PHOENIX_ENABLED"] = "true"
os.environ["PHOENIX_ENDPOINT"] = "http://localhost:6006"
os.environ["STRUCTURED_LOGGING"] = "true"

# Import miiflow-agent
from miiflow_agent import LLMClient, Message, Agent
from miiflow_agent.core.observability.evaluation import (
    AgentEvaluator,
    EvaluatedAgent,
    EvaluationMetric,
    create_default_evaluator,
    create_evaluated_agent
)


async def basic_evaluation_example():
    """Demonstrate basic agent evaluation."""
    print("\n=== Basic Agent Evaluation Example ===")

    # Create an agent
    client = LLMClient.create("openai", model="gpt-4o-mini")
    agent = Agent[dict](
        client=client,
        system_prompt="You are a helpful assistant that provides accurate information."
    )

    # Create an evaluator with default metrics
    evaluator = create_default_evaluator()

    # Create an evaluated agent
    evaluated_agent = create_evaluated_agent(agent, evaluator)

    # Test queries
    test_queries = [
        "What is the capital of France?",
        "How do you make a paper airplane?",
        "Explain quantum computing in simple terms.",
        "What's 2 + 2?",
        "Tell me about the weather on Mars."
    ]

    print(f"Running {len(test_queries)} test queries...")

    for i, query in enumerate(test_queries, 1):
        try:
            print(f"\n--- Query {i}: {query} ---")

            # Run the agent (this will automatically evaluate)
            result = await evaluated_agent.run(query, deps={})

            # Print response
            print(f"Response: {str(result.data)[:100]}...")

            # Print evaluation results
            if "evaluation" in result.metadata:
                eval_data = result.metadata["evaluation"]
                print(f"Evaluation ID: {eval_data['run_id']}")
                print("Metrics:")
                for metric_name, value in eval_data["metrics"].items():
                    print(f"  - {metric_name}: {value}")

        except Exception as e:
            print(f"Error: {e}")

    # Get evaluation summary
    summary = evaluated_agent.get_evaluation_summary()
    print(f"\n=== Evaluation Summary ===")
    print(f"Total evaluations: {summary.get('total_evaluations', 0)}")

    if "metric_summaries" in summary:
        for metric_name, stats in summary["metric_summaries"].items():
            print(f"\n{metric_name.upper()}:")
            if stats["type"] == "boolean":
                print(f"  Success rate: {stats['success_rate']:.2%}")
                print(f"  Success count: {stats['success_count']}/{stats['total_count']}")
            elif stats["type"] == "numeric":
                print(f"  Average: {stats['mean']:.2f}")
                print(f"  Range: {stats['min']:.2f} - {stats['max']:.2f}")


async def custom_evaluation_example():
    """Demonstrate custom evaluation metrics."""
    print("\n=== Custom Evaluation Metrics Example ===")

    # Create an agent
    client = LLMClient.create("openai", model="gpt-4o-mini")
    agent = Agent[dict](
        client=client,
        system_prompt="You are a concise assistant. Keep responses under 100 words."
    )

    # Create custom evaluator
    evaluator = AgentEvaluator()

    # Add custom length metric
    def length_evaluator(response: str, context: dict) -> bool:
        word_count = len(response.split())
        return word_count <= 100

    evaluator.add_metric(EvaluationMetric(
        name="conciseness",
        description="Response is under 100 words",
        evaluator=length_evaluator,
        metadata={"max_words": 100}
    ))

    # Add custom politeness metric
    def politeness_evaluator(response: str, context: dict) -> bool:
        polite_words = ["please", "thank you", "you're welcome", "excuse me", "sorry"]
        response_lower = response.lower()
        return any(word in response_lower for word in polite_words)

    evaluator.add_metric(EvaluationMetric(
        name="politeness",
        description="Response contains polite language",
        evaluator=politeness_evaluator
    ))

    # Add factual accuracy check (with ground truth)
    ground_truth = {
        "What is the capital of France?": "Paris",
        "What's 2 + 2?": "4",
        "Who wrote Romeo and Juliet?": "Shakespeare"
    }

    def accuracy_evaluator(response: str, context: dict) -> bool:
        query = context.get("user_query", "")
        expected = ground_truth.get(query)
        if not expected:
            return True  # No ground truth available, assume correct

        return expected.lower() in response.lower()

    evaluator.add_metric(EvaluationMetric(
        name="factual_accuracy",
        description="Response contains correct factual information",
        evaluator=accuracy_evaluator,
        metadata={"ground_truth_queries": len(ground_truth)}
    ))

    # Create evaluated agent
    evaluated_agent = EvaluatedAgent(agent, evaluator)

    # Test with queries that have ground truth
    test_queries = [
        "What is the capital of France?",
        "What's 2 + 2?",
        "Who wrote Romeo and Juliet?",
        "Can you help me understand machine learning?"
    ]

    print(f"Testing custom metrics with {len(test_queries)} queries...")

    for query in test_queries:
        try:
            print(f"\n--- Query: {query} ---")
            result = await evaluated_agent.run(query, deps={})

            response = str(result.data)
            print(f"Response ({len(response.split())} words): {response}")

            # Print custom evaluation results
            if "evaluation" in result.metadata:
                eval_data = result.metadata["evaluation"]
                print("Custom Metrics:")
                for metric_name, value in eval_data["metrics"].items():
                    print(f"  - {metric_name}: {value}")

        except Exception as e:
            print(f"Error: {e}")

    # Print summary
    summary = evaluated_agent.get_evaluation_summary()
    print(f"\n=== Custom Metrics Summary ===")
    for metric_name, stats in summary.get("metric_summaries", {}).items():
        if stats["type"] == "boolean":
            print(f"{metric_name}: {stats['success_rate']:.2%} success rate")


async def evaluation_with_tools_example():
    """Demonstrate evaluation of agents with tools."""
    print("\n=== Agent with Tools Evaluation Example ===")

    # Define tools
    def calculator(expression: str) -> str:
        """Calculate mathematical expressions."""
        try:
            # Simple eval (don't use in production!)
            result = eval(expression.replace("^", "**"))
            return str(result)
        except:
            return "Error in calculation"

    def weather(location: str) -> str:
        """Get weather information (mock)."""
        return f"The weather in {location} is sunny and 75°F"

    # Create agent with tools
    client = LLMClient.create("openai", model="gpt-4o-mini")
    agent = Agent[dict](
        client=client,
        system_prompt="You have access to calculator and weather tools. Use them when appropriate."
    )

    # Add tools
    from miiflow_agent.core.tools import tool
    calc_tool = tool("calculator", "Calculate mathematical expressions")(calculator)
    weather_tool = tool("weather", "Get weather information")(weather)

    agent.add_tool(calc_tool)
    agent.add_tool(weather_tool)

    # Create evaluator with tool-specific metrics
    evaluator = AgentEvaluator()

    # Add tool usage metric
    def tool_usage_evaluator(response: str, context: dict) -> bool:
        query = context.get("user_query", "").lower()

        # Check if agent should have used tools
        needs_calculation = any(word in query for word in ["calculate", "compute", "add", "multiply", "divide"])
        needs_weather = any(word in query for word in ["weather", "temperature", "forecast"])

        if not (needs_calculation or needs_weather):
            return True  # No tools needed

        # Check if response suggests tool was used
        response_lower = response.lower()
        tool_indicators = ["calculated", "computed", "weather", "temperature", "result"]

        return any(indicator in response_lower for indicator in tool_indicators)

    evaluator.add_metric(EvaluationMetric(
        name="tool_usage",
        description="Agent uses tools when appropriate",
        evaluator=tool_usage_evaluator
    ))

    # Add calculation accuracy metric
    def calc_accuracy_evaluator(response: str, context: dict) -> bool:
        query = context.get("user_query", "").lower()

        # Only check calculation queries
        if not any(word in query for word in ["calculate", "what is", "compute"]):
            return True

        # Simple checks for common calculations
        calculation_checks = {
            "2 + 2": "4",
            "10 * 5": "50",
            "100 / 4": "25"
        }

        for calc, expected in calculation_checks.items():
            if calc in query:
                return expected in response

        return True  # Can't verify, assume correct

    evaluator.add_metric(EvaluationMetric(
        name="calculation_accuracy",
        description="Mathematical calculations are correct",
        evaluator=calc_accuracy_evaluator
    ))

    # Create evaluated agent
    evaluated_agent = EvaluatedAgent(agent, evaluator)

    # Test queries requiring tools
    test_queries = [
        "What is 15 * 23?",
        "Calculate 2 + 2",
        "What's the weather like in Paris?",
        "Tell me about quantum physics",  # No tools needed
        "Compute 100 divided by 4"
    ]

    print(f"Testing agent with tools using {len(test_queries)} queries...")

    for query in test_queries:
        try:
            print(f"\n--- Query: {query} ---")
            result = await evaluated_agent.run(query, deps={})

            response = str(result.data)
            print(f"Response: {response}")

            # Print evaluation results
            if "evaluation" in result.metadata:
                eval_data = result.metadata["evaluation"]
                print("Tool Evaluation:")
                for metric_name, value in eval_data["metrics"].items():
                    print(f"  - {metric_name}: {value}")

        except Exception as e:
            print(f"Error: {e}")

    # Print final summary
    summary = evaluated_agent.get_evaluation_summary()
    print(f"\n=== Tool Usage Evaluation Summary ===")
    for metric_name, stats in summary.get("metric_summaries", {}).items():
        if stats["type"] == "boolean":
            print(f"{metric_name}: {stats['success_rate']:.2%} success rate")


def setup_phoenix_for_evaluation():
    """Set up Phoenix dashboard for evaluation viewing."""
    print("\n=== Phoenix Dashboard Setup for Evaluation ===")

    try:
        import phoenix as px

        # Start Phoenix session
        session = px.launch_app(host="localhost", port=6006)
        print(f"✓ Phoenix dashboard started at {session.url}")
        print("  - Agent evaluations will appear as traces")
        print("  - Evaluation metrics will be visible as span attributes")
        print("  - Use the Phoenix dashboard to analyze agent performance")

    except ImportError:
        print("✗ Phoenix not installed. Install with:")
        print("  pip install 'miiflow-agent[observability]'")
    except Exception as e:
        print(f"✗ Failed to start Phoenix: {e}")


async def main():
    """Run all evaluation examples."""
    print("Miiflow LLM Agent Evaluation with Phoenix")
    print("=" * 50)

    # Set up Phoenix dashboard
    setup_phoenix_for_evaluation()

    # Wait for Phoenix to be ready
    await asyncio.sleep(2)

    try:
        # Run examples
        await basic_evaluation_example()
        await custom_evaluation_example()
        await evaluation_with_tools_example()

        print("\n=== Examples Complete ===")
        print("Check the Phoenix dashboard at http://localhost:6006 to view:")
        print("  - Agent execution traces with evaluation spans")
        print("  - Evaluation metrics as span attributes")
        print("  - Performance analytics across multiple runs")
        print("  - Tool usage patterns and effectiveness")

    except Exception as e:
        print(f"Error in examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Check dependencies
    try:
        import phoenix as px
        import opentelemetry

        # Run the examples
        asyncio.run(main())

    except ImportError as e:
        print(f"Missing dependencies: {e}")
        print("\nInstall evaluation dependencies with:")
        print("pip install 'miiflow-agent[observability]'")
    except KeyboardInterrupt:
        print("\nExamples interrupted by user")
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()