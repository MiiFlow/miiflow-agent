"""Example demonstrating Phoenix observability with miiflow-agent."""

import os
import asyncio
from typing import List

# Set up Phoenix before importing miiflow-agent
os.environ["PHOENIX_ENABLED"] = "true"
os.environ["PHOENIX_ENDPOINT"] = "http://localhost:6006"
os.environ["STRUCTURED_LOGGING"] = "true"

# Import miiflow-agent (this will initialize observability)
from miiflow_agent import LLMClient, Message, Agent, RunContext
from miiflow_agent.core import print_observability_status


async def basic_llm_tracing_example():
    """Demonstrate basic LLM request tracing."""
    print("\n=== Basic LLM Tracing Example ===")

    # Create client (automatically instrumented)
    client = LLMClient.create("openai", model="gpt-4o-mini")

    # Simple chat request (will be traced)
    messages = [Message.user("What is the capital of France?")]

    try:
        response = await client.achat(messages)
        print(f"Response: {response.message.content}")
        print(f"Tokens used: {response.usage.total_tokens}")
    except Exception as e:
        print(f"Error: {e}")


async def streaming_tracing_example():
    """Demonstrate streaming request tracing."""
    print("\n=== Streaming Tracing Example ===")

    client = LLMClient.create("openai", model="gpt-4o-mini")
    messages = [Message.user("Tell me a short story about a robot.")]

    try:
        print("Streaming response: ", end="")
        async for chunk in client.astream_chat(messages):
            print(chunk.delta, end="", flush=True)
        print()  # New line after streaming
    except Exception as e:
        print(f"Error: {e}")


async def agent_tracing_example():
    """Demonstrate agent execution tracing with tools."""
    print("\n=== Agent Tracing Example ===")

    # Define a simple tool
    def weather_tool(location: str) -> str:
        """Get weather information for a location."""
        return f"The weather in {location} is sunny and 75°F"

    # Create agent with tool
    client = LLMClient.create("openai", model="gpt-4o-mini")

    agent = Agent[dict](
        client=client,
        system_prompt="You are a helpful assistant with access to weather information."
    )

    # Add the tool
    from miiflow_agent.core.tools import tool
    tool_decorator = tool("weather", "Get weather information")
    instrumented_tool = tool_decorator(weather_tool)
    agent.add_tool(instrumented_tool)

    # Run agent (will create nested traces)
    try:
        result = await agent.run(
            "What's the weather like in Paris?",
            deps={}
        )
        print(f"Agent result: {result.data}")
    except Exception as e:
        print(f"Error: {e}")


async def error_tracing_example():
    """Demonstrate error tracing."""
    print("\n=== Error Tracing Example ===")

    # Create client with invalid API key to trigger error
    client = LLMClient.create("openai", model="gpt-4o-mini", api_key="invalid-key")
    messages = [Message.user("This will fail")]

    try:
        response = await client.achat(messages)
        print(f"Unexpected success: {response.message.content}")
    except Exception as e:
        print(f"Expected error traced: {type(e).__name__}: {e}")


def demonstrate_metrics_integration():
    """Demonstrate metrics integration with Phoenix."""
    print("\n=== Metrics Integration Example ===")

    from miiflow_agent.core.observability.metrics_bridge import ObservabilityMetricsCollector
    from miiflow_agent.core.metrics import UsageData, TokenCount
    from datetime import datetime

    # Create enhanced metrics collector
    collector = ObservabilityMetricsCollector()

    # Record some sample usage
    usage_data = [
        UsageData(
            provider="openai",
            model="gpt-4o-mini",
            operation="chat",
            tokens=TokenCount(prompt_tokens=50, completion_tokens=100, total_tokens=150),
            latency_ms=1200.0,
            success=True,
            timestamp=datetime.now()
        ),
        UsageData(
            provider="anthropic",
            model="claude-3-haiku",
            operation="chat",
            tokens=TokenCount(prompt_tokens=30, completion_tokens=80, total_tokens=110),
            latency_ms=800.0,
            success=True,
            timestamp=datetime.now()
        )
    ]

    for usage in usage_data:
        collector.record_usage(usage)

    # Get metrics summary
    metrics = collector.get_metrics()
    print(f"Recorded metrics for {len(metrics)} provider:model combinations")

    # Export to Phoenix format
    phoenix_export = collector.export_metrics_to_phoenix()
    print(f"Phoenix export contains {len(phoenix_export)} sections")


def setup_phoenix_dashboard():
    """Set up Phoenix dashboard if not running."""
    print("\n=== Phoenix Dashboard Setup ===")

    try:
        import phoenix as px
        import httpx

        # Check if Phoenix is already running
        try:
            response = httpx.get("http://localhost:6006", timeout=2)
            print("✓ Phoenix dashboard is already running at http://localhost:6006")
            return
        except (httpx.ConnectError, httpx.TimeoutException):
            pass

        # Start Phoenix session
        session = px.launch_app(host="localhost", port=6006)
        print(f"✓ Phoenix dashboard started at {session.url}")
        print("  - Open this URL in your browser to view traces")
        print("  - Leave this running while executing the examples")

    except ImportError:
        print("✗ Phoenix not installed. Install with:")
        print("  pip install 'miiflow-agent[observability]'")
    except Exception as e:
        print(f"✗ Failed to start Phoenix: {e}")


async def main():
    """Run all observability examples."""
    print("Miiflow LLM Phoenix Observability Examples")
    print("=" * 50)

    # Show observability status
    print_observability_status()

    # Set up Phoenix dashboard
    setup_phoenix_dashboard()

    # Wait a moment for Phoenix to be ready
    await asyncio.sleep(2)

    # Run examples
    await basic_llm_tracing_example()
    await streaming_tracing_example()

    # Note: Agent examples require valid API keys
    print("\n--- Advanced Examples (require valid API keys) ---")
    try:
        await agent_tracing_example()
    except Exception as e:
        print(f"Agent example skipped: {e}")

    await error_tracing_example()

    # Demonstrate metrics
    demonstrate_metrics_integration()

    print("\n=== Examples Complete ===")
    print("Check the Phoenix dashboard at http://localhost:6006 to view traces!")
    print("Look for:")
    print("  - LLM request spans with token counts and latencies")
    print("  - Agent execution spans with tool calls")
    print("  - Error spans with exception details")
    print("  - Nested spans showing request flow")


if __name__ == "__main__":
    # Check if we have required dependencies
    try:
        import phoenix as px
        import opentelemetry

        # Run the examples
        asyncio.run(main())

    except ImportError as e:
        print(f"Missing dependencies: {e}")
        print("\nInstall observability dependencies with:")
        print("pip install 'miiflow-agent[observability]'")
        print("\nOr install individual packages:")
        print("pip install arize-phoenix opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp")
    except KeyboardInterrupt:
        print("\nExamples interrupted by user")
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()