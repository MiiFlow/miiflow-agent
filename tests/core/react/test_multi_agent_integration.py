"""Integration tests for Multi-Agent orchestrator across providers.

These tests verify multi-agent execution works correctly across:
- OpenAI (gpt-4o-mini)
- Anthropic/Claude (claude-3-5-haiku)
- Google/Gemini (gemini-2.0-flash)

Run with:
    # All providers (requires all API keys)
    cd packages/miiflow-agent
    poetry run pytest tests/core/react/test_multi_agent_integration.py -v -m smoke -s

    # Specific provider
    poetry run pytest tests/core/react/test_multi_agent_integration.py -v -k "openai" -m smoke -s
    poetry run pytest tests/core/react/test_multi_agent_integration.py -v -k "anthropic" -m smoke -s
    poetry run pytest tests/core/react/test_multi_agent_integration.py -v -k "gemini" -m smoke -s

    # Quick run (first available provider)
    poetry run python tests/core/react/test_multi_agent_integration.py

Required environment variables:
    OPENAI_API_KEY - For OpenAI tests
    ANTHROPIC_API_KEY - For Claude tests
    GOOGLE_API_KEY - For Gemini tests
"""

import asyncio
import logging
import os
import pytest
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from unittest.mock import MagicMock, AsyncMock

from miiflow_agent import LLMClient
from miiflow_agent.core import Agent, AgentType
from miiflow_agent.core.agent import RunContext
from miiflow_agent.core.message import Message
from miiflow_agent.core.tools import tool, ToolRegistry
from miiflow_agent.core.react.enums import MultiAgentEventType, StopReason
from miiflow_agent.core.react.events import EventBus
from miiflow_agent.core.react.models import MultiAgentResult, SubAgentConfig, SubAgentPlan
from miiflow_agent.core.react.multi_agent_orchestrator import MultiAgentOrchestrator
from miiflow_agent.core.react.orchestrator import ReActOrchestrator
from miiflow_agent.core.react.parser import ReActParser
from miiflow_agent.core.react.react_events import MultiAgentEvent
from miiflow_agent.core.react.safety import SafetyManager
from miiflow_agent.core.react.tool_executor import AgentToolExecutor

logger = logging.getLogger(__name__)


# =============================================================================
# Provider Configurations
# =============================================================================

@dataclass
class MultiAgentProviderConfig:
    """Configuration for a provider in multi-agent tests."""
    name: str
    model: str
    api_key_env: str

    def has_api_key(self) -> bool:
        """Check if API key is configured."""
        key = os.getenv(self.api_key_env)
        return bool(key and not key.startswith('your-'))

    def get_api_key(self) -> Optional[str]:
        """Get API key if available."""
        if self.has_api_key():
            return os.getenv(self.api_key_env)
        return None

    def skip_if_no_key(self):
        """Skip test if API key not available."""
        if not self.has_api_key():
            pytest.skip(f"{self.api_key_env} not configured")


MULTI_AGENT_PROVIDERS = [
    MultiAgentProviderConfig(
        name="openai",
        model="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
    ),
    MultiAgentProviderConfig(
        name="anthropic",
        model="claude-3-5-haiku-20241022",
        api_key_env="ANTHROPIC_API_KEY",
    ),
    MultiAgentProviderConfig(
        name="gemini",
        model="gemini-2.0-flash-exp",
        api_key_env="GEMINI_API_KEY",  # LLMClient expects GEMINI_API_KEY
    ),
]


# =============================================================================
# Test Tools
# =============================================================================

@tool(name="search_news", description="Search for recent news on a topic")
def search_news(topic: str) -> Dict[str, Any]:
    """Search for recent news on a topic.

    Args:
        topic: The topic to search news for

    Returns:
        Recent news headlines and summaries
    """
    # Mock implementation for testing
    return {
        "success": True,
        "result": f"""Recent news about {topic}:
1. {topic} shows significant growth in Q4 2024
2. Analysts predict strong performance for {topic} sector
3. New developments announced for {topic} initiatives
4. Market experts weigh in on {topic} trends"""
    }


@tool(name="analyze_sentiment", description="Analyze the sentiment of text")
def analyze_sentiment(text: str) -> Dict[str, Any]:
    """Analyze the sentiment of text.

    Args:
        text: The text to analyze

    Returns:
        Sentiment analysis result
    """
    # Mock implementation
    return {
        "success": True,
        "result": f"Sentiment analysis of '{text[:50]}...': Overall positive sentiment (75% confidence). Key themes: growth, optimism, market interest."
    }


@tool(name="get_financial_data", description="Get financial data for a company")
def get_financial_data(company: str) -> Dict[str, Any]:
    """Get financial data for a company.

    Args:
        company: Company name or ticker

    Returns:
        Financial metrics and data
    """
    # Mock implementation
    return {
        "success": True,
        "result": f"""Financial data for {company}:
- Revenue: $50B (up 15% YoY)
- Operating margin: 22%
- Market cap: $200B
- P/E ratio: 25
- Recent performance: Positive trend over last 6 months"""
    }


# =============================================================================
# Event Collector for Testing
# =============================================================================

class EventCollector:
    """Collect events for test verification."""

    def __init__(self):
        self.events: List[Any] = []
        self.event_types: List[MultiAgentEventType] = []

    async def collect(self, event: Any):
        """Collect an event."""
        # Handle different event types
        if hasattr(event, 'event_type'):
            self.events.append(event)
            self.event_types.append(event.event_type)
            logger.info(f"Event: {event.event_type.value} - {getattr(event, 'data', {})}")

    def has_event_type(self, event_type: MultiAgentEventType) -> bool:
        """Check if an event type was collected."""
        return event_type in self.event_types

    def get_events_of_type(self, event_type: MultiAgentEventType) -> List[Any]:
        """Get all events of a specific type."""
        return [e for e in self.events if hasattr(e, 'event_type') and e.event_type == event_type]

    def count_event_type(self, event_type: MultiAgentEventType) -> int:
        """Count occurrences of an event type."""
        return self.event_types.count(event_type)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture(params=MULTI_AGENT_PROVIDERS, ids=lambda p: p.name)
def provider_config(request) -> MultiAgentProviderConfig:
    """Fixture that provides each provider configuration."""
    return request.param


@pytest.fixture
def test_tools() -> List:
    """Create a list of test tools."""
    return [search_news, analyze_sentiment, get_financial_data]


@pytest.fixture
def event_collector() -> EventCollector:
    """Create an event collector."""
    return EventCollector()


# =============================================================================
# Helper Functions
# =============================================================================

def create_client(config: MultiAgentProviderConfig):
    """Create an LLM client instance from config."""
    api_key = config.get_api_key()
    if not api_key:
        pytest.skip(f"{config.api_key_env} not configured")

    return LLMClient.create(config.name, model=config.model)


async def create_multi_agent_orchestrator(
    config: MultiAgentProviderConfig,
    test_tools: List,
    event_bus: EventBus,
    max_subagents: int = 3,
) -> MultiAgentOrchestrator:
    """Create a multi-agent orchestrator for testing."""
    client = create_client(config)

    # Create agent for subagent execution
    agent = Agent(
        client=client,
        agent_type=AgentType.REACT,
        tools=test_tools,
        max_iterations=5,  # Limit steps for testing
    )

    # Create tool executor from agent
    tool_executor = AgentToolExecutor(agent)

    # Create ReAct orchestrator for subagents (directly, not via factory, to share event_bus)
    subagent_safety = SafetyManager(max_steps=5, max_time_seconds=60)
    subagent_orchestrator = ReActOrchestrator(
        tool_executor=tool_executor,
        event_bus=event_bus,
        safety_manager=subagent_safety,
        parser=ReActParser(),
    )

    # Create safety manager for multi-agent orchestrator
    safety_manager = SafetyManager(
        max_steps=10,
        max_time_seconds=120,
    )

    return MultiAgentOrchestrator(
        tool_executor=tool_executor,
        event_bus=event_bus,
        safety_manager=safety_manager,
        subagent_orchestrator=subagent_orchestrator,
        max_subagents=max_subagents,
        subagent_timeout_seconds=60.0,
    )


# =============================================================================
# Test Cases - Multi-Agent Execution
# =============================================================================

@pytest.mark.smoke
@pytest.mark.asyncio
class TestMultiAgentExecution:
    """Test multi-agent execution across providers."""

    async def test_basic_multi_agent_query(
        self,
        provider_config: MultiAgentProviderConfig,
        test_tools: List,
        event_collector: EventCollector,
    ):
        """Test basic multi-agent execution with a multi-faceted query.

        This test verifies:
        - Lead agent successfully plans subagent allocation
        - Multiple subagents execute in parallel
        - Results are synthesized into a final answer
        - Events are emitted correctly throughout execution
        """
        provider_config.skip_if_no_key()

        # Create event bus and subscribe collector
        event_bus = EventBus()
        event_bus.subscribe(event_collector.collect)

        # Create orchestrator
        orchestrator = await create_multi_agent_orchestrator(
            provider_config, test_tools, event_bus
        )

        # Execute multi-faceted query
        query = "Research Tesla: find recent news, analyze market sentiment, and provide financial overview."
        context = RunContext(deps=None, messages=[])

        result = await orchestrator.execute(query, context)

        # Verify result structure
        assert isinstance(result, MultiAgentResult)
        assert result.final_answer is not None
        assert len(result.final_answer) > 0
        assert result.stop_reason in [StopReason.ANSWER_COMPLETE, StopReason.FORCED_STOP]

        # Verify events were emitted
        assert event_collector.has_event_type(MultiAgentEventType.PLANNING_START)
        assert event_collector.has_event_type(MultiAgentEventType.PLANNING_COMPLETE)
        assert event_collector.count_event_type(MultiAgentEventType.SUBAGENT_START) >= 1
        assert event_collector.has_event_type(MultiAgentEventType.SYNTHESIS_START)
        assert event_collector.has_event_type(MultiAgentEventType.FINAL_ANSWER)

        # Log result for manual verification
        logger.info(f"\n{'='*60}")
        logger.info(f"Provider: {provider_config.name}")
        logger.info(f"Query: {query}")
        logger.info(f"Subagent count: {len(result.subagent_results)}")
        logger.info(f"Final answer length: {len(result.final_answer)}")
        logger.info(f"Total execution time: {result.total_execution_time:.2f}s")
        logger.info(f"Stop reason: {result.stop_reason}")
        logger.info(f"{'='*60}\n")

    async def test_multi_agent_with_existing_plan(
        self,
        provider_config: MultiAgentProviderConfig,
        test_tools: List,
        event_collector: EventCollector,
    ):
        """Test multi-agent execution with a pre-defined plan.

        This test verifies:
        - Orchestrator correctly uses an existing plan
        - Skips planning phase when plan is provided
        - Executes all subagents defined in the plan
        """
        provider_config.skip_if_no_key()

        # Create event bus and subscribe collector
        event_bus = EventBus()
        event_bus.subscribe(event_collector.collect)

        # Create orchestrator
        orchestrator = await create_multi_agent_orchestrator(
            provider_config, test_tools, event_bus
        )

        # Create a pre-defined plan
        existing_plan = SubAgentPlan(
            reasoning="Pre-defined plan for testing: one researcher and one analyst",
            subagent_configs=[
                SubAgentConfig(
                    name="news_researcher",
                    role="researcher",
                    focus="Recent news and developments",
                    query="Search for recent news about Apple and summarize key developments.",
                    output_key="result_news",
                ),
                SubAgentConfig(
                    name="financial_analyst",
                    role="analyzer",
                    focus="Financial performance analysis",
                    query="Get financial data for Apple and provide key metrics analysis.",
                    output_key="result_financial",
                ),
            ],
        )

        # Execute with existing plan
        query = "Provide a comprehensive overview of Apple."
        context = RunContext(deps=None, messages=[])

        result = await orchestrator.execute(query, context, existing_plan=existing_plan)

        # Verify result
        assert isinstance(result, MultiAgentResult)
        assert result.final_answer is not None
        assert len(result.subagent_results) == 2

        # Verify subagent names match plan
        subagent_names = [r.agent_name for r in result.subagent_results]
        assert "news_researcher" in subagent_names
        assert "financial_analyst" in subagent_names

        # Verify events
        assert event_collector.count_event_type(MultiAgentEventType.SUBAGENT_START) == 2

        logger.info(f"\n{'='*60}")
        logger.info(f"Provider: {provider_config.name}")
        logger.info(f"Pre-defined plan executed successfully")
        logger.info(f"Subagents: {subagent_names}")
        logger.info(f"{'='*60}\n")

    async def test_single_subagent_fallback(
        self,
        provider_config: MultiAgentProviderConfig,
        test_tools: List,
        event_collector: EventCollector,
    ):
        """Test that simple queries result in single subagent execution.

        This test verifies:
        - Simple queries don't create unnecessary subagents
        - Execution still completes successfully
        """
        provider_config.skip_if_no_key()

        # Create event bus and subscribe collector
        event_bus = EventBus()
        event_bus.subscribe(event_collector.collect)

        # Create orchestrator with low max_subagents
        orchestrator = await create_multi_agent_orchestrator(
            provider_config, test_tools, event_bus, max_subagents=1
        )

        # Execute simple query
        query = "What is the capital of France?"
        context = RunContext(deps=None, messages=[])

        result = await orchestrator.execute(query, context)

        # Verify result
        assert isinstance(result, MultiAgentResult)
        assert result.final_answer is not None

        # Should have at most 1 subagent
        assert len(result.subagent_results) <= 1

        logger.info(f"\n{'='*60}")
        logger.info(f"Provider: {provider_config.name}")
        logger.info(f"Simple query handled with {len(result.subagent_results)} subagent(s)")
        logger.info(f"{'='*60}\n")


@pytest.mark.smoke
@pytest.mark.asyncio
class TestMultiAgentEventStreaming:
    """Test event streaming during multi-agent execution."""

    async def test_event_sequence_order(
        self,
        provider_config: MultiAgentProviderConfig,
        test_tools: List,
        event_collector: EventCollector,
    ):
        """Test that events are emitted in the correct sequence.

        Expected sequence:
        1. PLANNING_START
        2. PLANNING_COMPLETE
        3. EXECUTION_START
        4. SUBAGENT_START (1 or more)
        5. SUBAGENT_COMPLETE or SUBAGENT_FAILED (1 or more)
        6. SYNTHESIS_START
        7. FINAL_ANSWER
        """
        provider_config.skip_if_no_key()

        # Create event bus and subscribe collector
        event_bus = EventBus()
        event_bus.subscribe(event_collector.collect)

        # Create orchestrator
        orchestrator = await create_multi_agent_orchestrator(
            provider_config, test_tools, event_bus
        )

        # Execute query
        query = "Compare the recent performance of Microsoft and Google."
        context = RunContext(deps=None, messages=[])

        await orchestrator.execute(query, context)

        # Verify event sequence
        assert len(event_collector.events) > 0

        # First event should be PLANNING_START
        assert event_collector.events[0].event_type == MultiAgentEventType.PLANNING_START

        # PLANNING_COMPLETE should come before any SUBAGENT_START
        planning_complete_idx = event_collector.event_types.index(MultiAgentEventType.PLANNING_COMPLETE)
        first_subagent_start_idx = next(
            (i for i, t in enumerate(event_collector.event_types) if t == MultiAgentEventType.SUBAGENT_START),
            -1
        )
        if first_subagent_start_idx >= 0:
            assert planning_complete_idx < first_subagent_start_idx

        # SYNTHESIS_START should come after all subagent events
        if event_collector.has_event_type(MultiAgentEventType.SYNTHESIS_START):
            synthesis_idx = event_collector.event_types.index(MultiAgentEventType.SYNTHESIS_START)
            last_subagent_complete_idx = max(
                (i for i, t in enumerate(event_collector.event_types)
                 if t in [MultiAgentEventType.SUBAGENT_COMPLETE, MultiAgentEventType.SUBAGENT_FAILED]),
                default=-1
            )
            if last_subagent_complete_idx >= 0:
                assert synthesis_idx > last_subagent_complete_idx

        # FINAL_ANSWER should be last
        assert event_collector.events[-1].event_type == MultiAgentEventType.FINAL_ANSWER

        logger.info(f"\n{'='*60}")
        logger.info(f"Provider: {provider_config.name}")
        logger.info(f"Event sequence verified: {len(event_collector.events)} events")
        logger.info(f"Event types: {[e.value for e in event_collector.event_types]}")
        logger.info(f"{'='*60}\n")

    async def test_subagent_events_contain_required_data(
        self,
        provider_config: MultiAgentProviderConfig,
        test_tools: List,
        event_collector: EventCollector,
    ):
        """Test that subagent events contain all required data fields."""
        provider_config.skip_if_no_key()

        # Create event bus and subscribe collector
        event_bus = EventBus()
        event_bus.subscribe(event_collector.collect)

        # Create orchestrator
        orchestrator = await create_multi_agent_orchestrator(
            provider_config, test_tools, event_bus
        )

        # Execute query
        query = "Research recent AI developments and analyze their market impact."
        context = RunContext(deps=None, messages=[])

        await orchestrator.execute(query, context)

        # Verify SUBAGENT_START events
        start_events = event_collector.get_events_of_type(MultiAgentEventType.SUBAGENT_START)
        for event in start_events:
            assert "name" in event.data, "SUBAGENT_START should have 'name'"
            assert "role" in event.data, "SUBAGENT_START should have 'role'"
            assert "query" in event.data, "SUBAGENT_START should have 'query'"

        # Verify SUBAGENT_COMPLETE events
        complete_events = event_collector.get_events_of_type(MultiAgentEventType.SUBAGENT_COMPLETE)
        for event in complete_events:
            assert "name" in event.data, "SUBAGENT_COMPLETE should have 'name'"
            assert "success" in event.data, "SUBAGENT_COMPLETE should have 'success'"
            assert "execution_time" in event.data, "SUBAGENT_COMPLETE should have 'execution_time'"

        logger.info(f"\n{'='*60}")
        logger.info(f"Provider: {provider_config.name}")
        logger.info(f"Verified {len(start_events)} SUBAGENT_START events")
        logger.info(f"Verified {len(complete_events)} SUBAGENT_COMPLETE events")
        logger.info(f"{'='*60}\n")


@pytest.mark.smoke
@pytest.mark.asyncio
class TestMultiAgentContextIsolation:
    """Test context isolation in parallel subagent execution."""

    async def test_parallel_subagents_dont_share_messages(
        self,
        provider_config: MultiAgentProviderConfig,
        test_tools: List,
        event_collector: EventCollector,
    ):
        """Test that parallel subagents have isolated message contexts.

        This test verifies the fix for the Anthropic API error:
        "tool_use ids found without tool_result blocks immediately after"

        Each subagent should have its own isolated context to prevent
        message contamination during parallel execution.
        """
        provider_config.skip_if_no_key()

        # Create event bus and subscribe collector
        event_bus = EventBus()
        event_bus.subscribe(event_collector.collect)

        # Create orchestrator with 3 subagents to test parallel execution
        orchestrator = await create_multi_agent_orchestrator(
            provider_config, test_tools, event_bus, max_subagents=3
        )

        # Execute query that will spawn multiple subagents using tools
        query = """Analyze three companies in parallel:
        1. Search for Tesla news
        2. Get Apple financial data
        3. Analyze Microsoft sentiment
        Provide a comprehensive comparison."""
        context = RunContext(deps=None, messages=[])

        # Execute - this should NOT raise Anthropic API errors
        result = await orchestrator.execute(query, context)

        # Verify execution completed without errors
        assert isinstance(result, MultiAgentResult)
        assert result.stop_reason != StopReason.FORCED_STOP or len(result.subagent_results) > 0

        # Verify no subagents failed due to API errors
        failed_events = event_collector.get_events_of_type(MultiAgentEventType.SUBAGENT_FAILED)
        for event in failed_events:
            error = event.data.get("error", "")
            # Should not have tool_use/tool_result mismatch errors
            assert "tool_use" not in error.lower() or "tool_result" not in error.lower(), \
                f"Context isolation failed - got API error: {error}"

        logger.info(f"\n{'='*60}")
        logger.info(f"Provider: {provider_config.name}")
        logger.info(f"Parallel execution completed successfully")
        logger.info(f"Subagents: {len(result.subagent_results)}")
        logger.info(f"Failed: {len(failed_events)}")
        logger.info(f"{'='*60}\n")


@pytest.mark.smoke
@pytest.mark.asyncio
class TestMultiAgentErrorHandling:
    """Test error handling in multi-agent execution."""

    async def test_returns_partial_results_on_failures(
        self,
        provider_config: MultiAgentProviderConfig,
        test_tools: List,
        event_collector: EventCollector,
    ):
        """Test that partial results are returned even when some subagents fail."""
        provider_config.skip_if_no_key()

        # Create event bus and subscribe collector
        event_bus = EventBus()
        event_bus.subscribe(event_collector.collect)

        # Create orchestrator
        orchestrator = await create_multi_agent_orchestrator(
            provider_config, test_tools, event_bus
        )

        # Create a plan with one normal and one problematic subagent
        existing_plan = SubAgentPlan(
            reasoning="Test plan with potential failures",
            subagent_configs=[
                SubAgentConfig(
                    name="good_agent",
                    role="researcher",
                    focus="Simple task",
                    query="What is 2+2?",
                    output_key="result_good",
                ),
                SubAgentConfig(
                    name="complex_agent",
                    role="analyzer",
                    focus="Complex analysis",
                    query="Perform an extremely detailed analysis of global market trends.",
                    output_key="result_complex",
                ),
            ],
        )

        query = "Test query."
        context = RunContext(deps=None, messages=[])

        result = await orchestrator.execute(query, context, existing_plan=existing_plan)

        # Verify we still get a final answer even if some failed
        assert isinstance(result, MultiAgentResult)
        assert result.final_answer is not None

        # Verify we have results (success or failure)
        assert len(result.subagent_results) == 2

        logger.info(f"\n{'='*60}")
        logger.info(f"Provider: {provider_config.name}")
        logger.info(f"Partial results test completed")
        logger.info(f"Successful: {len([r for r in result.subagent_results if r.success])}")
        logger.info(f"Failed: {len([r for r in result.subagent_results if not r.success])}")
        logger.info(f"{'='*60}\n")


# =============================================================================
# Direct Execution for Manual Testing
# =============================================================================

if __name__ == "__main__":
    """Run a quick manual test."""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    async def run_manual_test():
        print("\n" + "="*60)
        print("MANUAL MULTI-AGENT TEST")
        print("="*60 + "\n")

        # Check for API keys
        available_providers = []
        for config in MULTI_AGENT_PROVIDERS:
            if config.has_api_key():
                available_providers.append(config)
                print(f"✓ {config.name}: API key found")
            else:
                print(f"✗ {config.name}: {config.api_key_env} not set")

        if not available_providers:
            print("\nNo API keys configured. Set at least one of:")
            print("  OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY")
            sys.exit(1)

        # Use first available provider
        config = available_providers[0]
        print(f"\nUsing provider: {config.name} ({config.model})")

        # Create tools
        tools = [search_news, analyze_sentiment, get_financial_data]

        # Create event collector
        collector = EventCollector()

        # Create event bus and subscribe collector
        event_bus = EventBus()
        event_bus.subscribe(collector.collect)

        # Create orchestrator
        print("\nCreating multi-agent orchestrator...")
        orchestrator = await create_multi_agent_orchestrator(
            config, tools, event_bus
        )

        # Execute test query
        query = "Research Tesla: find recent news, analyze market sentiment, and provide financial overview."
        print(f"\nQuery: {query}")
        print("\nExecuting...")
        print("-"*60)

        context = RunContext(deps=None, messages=[])
        result = await orchestrator.execute(query, context)

        # Print results
        print("-"*60)
        print("\nRESULTS:")
        print(f"  Stop reason: {result.stop_reason}")
        print(f"  Subagents: {len(result.subagent_results)}")
        print(f"  Execution time: {result.total_execution_time:.2f}s")
        print(f"  Total tokens: {result.total_tokens}")

        print("\nSUBAGENT RESULTS:")
        for r in result.subagent_results:
            status = "✓" if r.success else "✗"
            print(f"  {status} {r.agent_name} ({r.role}): {r.execution_time:.2f}s")

        print("\nEVENTS:")
        for event in collector.events:
            print(f"  {event.event_type.value}")

        print("\nFINAL ANSWER:")
        print("-"*60)
        print(result.final_answer[:1000] if result.final_answer else "No answer")
        if result.final_answer and len(result.final_answer) > 1000:
            print(f"\n... (truncated, total length: {len(result.final_answer)})")
        print("-"*60)

        print("\n" + "="*60)
        print("TEST COMPLETE")
        print("="*60 + "\n")

    asyncio.run(run_manual_test())
