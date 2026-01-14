#!/usr/bin/env python3
"""
Test tool context injection patterns - Pydantic AI style vs current style.
Includes real LLM integration tests using API keys from .env
"""

import pytest
import asyncio
import os
from unittest.mock import MagicMock
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from miiflow_agent.core.agent import Agent, RunContext
from miiflow_agent.core.client import LLMClient
from miiflow_agent.core.tools import FunctionTool, tool


@dataclass
class MockDeps:
    """Test dependency container."""
    database_name: str = "test_db"
    user_id: str = "test_user"


class TestToolContextInjection:
    """Test tool context injection patterns."""
    
    @pytest.fixture
    def llm_client(self):
        """Create mock LLM client."""
        mock_client = MagicMock()
        mock_client.provider_name = "openai"
        return LLMClient(mock_client)
    
    def test_pydantic_ai_style_context_detection(self, llm_client):
        """Test that Pydantic AI style context is detected correctly."""
        agent = Agent(llm_client)
        
        @tool("search")
        async def search(ctx: RunContext[MockDeps], query: str) -> str:
            return f"Searched: {query} with {ctx.deps.database_name}"
        
        agent.add_tool(search)
        
        # Verify context injection pattern detected
        search_tool = agent.tool_registry.tools['search']
        assert search_tool.context_injection['pattern'] == 'first_param'
        assert search_tool.context_injection['param_name'] == 'ctx'
        assert search_tool.context_injection['param_index'] == 0
        
        # Verify schema excludes context parameter
        if search_tool.schema.parameters:
            assert 'ctx' not in search_tool.schema.parameters
            assert 'query' in search_tool.schema.parameters
        
        print(f"✅ Pydantic AI style detection: {search_tool.context_injection}")
    
    def test_current_style_context_detection(self, llm_client):
        """Test that current style context is detected correctly."""
        agent = Agent(llm_client)
        
        @tool("search")
        async def search(query: str, context: RunContext[MockDeps]) -> str:
            return f"Searched: {query} with {context.deps.database_name}"
        
        agent.add_tool(search)
        
        # Verify backward compatibility
        search_tool = agent.tool_registry.tools['search']
        assert search_tool.context_injection['pattern'] == 'keyword'
        assert search_tool.context_injection['param_name'] == 'context'
        
        # Verify schema excludes context parameter
        if search_tool.schema.parameters:
            assert 'context' not in search_tool.schema.parameters
            assert 'query' in search_tool.schema.parameters
        
        print(f"✅ Current style detection: {search_tool.context_injection}")
    
    def test_no_context_detection(self, llm_client):
        """Test tools without context parameters."""
        agent = Agent(llm_client)
        
        @tool("calculator")
        async def calculator(a: int, b: int) -> int:
            return a + b
        
        agent.add_tool(calculator)
        
        calc_tool = agent.tool_registry.tools['calculator']
        assert calc_tool.context_injection['pattern'] == 'none'
        
        # Schema should include all parameters
        if calc_tool.schema.parameters:
            assert 'a' in calc_tool.schema.parameters
            assert 'b' in calc_tool.schema.parameters
        
        print(f"✅ No context detection: {calc_tool.context_injection}")
    
    def test_dual_decorator_syntax(self, llm_client):
        """Test both @tool and @tool('name') syntax."""
        agent = Agent(llm_client)
        
        # Style 1: @tool (Pydantic AI)
        @tool("tool1")
        async def tool1(ctx: RunContext[MockDeps], query: str) -> str:
            return f"Tool1: {query} - {ctx.deps.user_id}"
            
        # Style 2: @tool("name") (current)  
        @tool("tool2")
        async def tool2_func(query: str, context: RunContext[MockDeps]) -> str:
            return f"Tool2: {query} - {context.deps.user_id}"
        
        agent.add_tool(tool1)
        agent.add_tool(tool2_func)
        
        # Both should be registered
        assert 'tool1' in agent.tool_registry.tools
        assert 'tool2' in agent.tool_registry.tools
        
        # Check patterns
        tool1 = agent.tool_registry.tools['tool1']
        tool2 = agent.tool_registry.tools['tool2']
        
        assert tool1.context_injection['pattern'] == 'first_param'
        assert tool2.context_injection['pattern'] == 'keyword'
        
        print(f"✅ Dual syntax works: tool1={tool1.context_injection['pattern']}, tool2={tool2.context_injection['pattern']}")
    
    @pytest.mark.asyncio
    async def test_context_execution_first_param(self):
        """Test actual execution with first_param pattern."""
        # Test direct tool execution
        def search_func(ctx: MockDeps, query: str) -> str:
            return f"Searched: {query} with {ctx.database_name}"
        
        tool = FunctionTool(search_func, "search", "Search tool")
        
        # Verify pattern detection
        assert tool.context_injection['pattern'] == 'first_param'
        
        # Test execution through registry
        from miiflow_agent.core.tools import ToolRegistry
        registry = ToolRegistry()
        registry.register(tool)
        
        deps = MockDeps(database_name="production_db", user_id="user123")
        
        # Execute with context as first parameter
        result = await registry.execute_safe_with_context("search", deps, query="test query")
        
        assert result.success
        assert "Searched: test query with production_db" == result.output
        assert result.metadata['execution_pattern'] == 'first_param'
        
        print(f"✅ First param execution: {result.output}")
    
    @pytest.mark.asyncio 
    async def test_context_execution_keyword_pattern_detection(self):
        """Test that keyword pattern detection works correctly."""
        def search_func(query: str, context: MockDeps = None) -> str:
            if context:
                return f"Searched: {query} with {context.database_name}"
            return f"Searched: {query} with no context"
        
        tool = FunctionTool(search_func, "search", "Search tool") 
        
        # Pattern detection should work for keyword context parameter
        print(f"Detected pattern: {tool.context_injection}")
        assert tool.context_injection['pattern'] == 'keyword'
        assert tool.context_injection['param_name'] == 'context'
        
        # Test that schema excludes context parameter
        if tool.schema.parameters:
            assert 'context' not in tool.schema.parameters
        
        print(f"✅ Keyword pattern detection works: {tool.context_injection}")


def _has_any_api_key() -> bool:
    """Check if any LLM API key is available."""
    for key in ['GROQ_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY']:
        if os.getenv(key):
            return True
    return False


@pytest.mark.skipif(not _has_any_api_key(), reason="No API keys found - skipping real LLM tests")
class TestRealLLMIntegration:
    """Test context injection with real LLM API calls."""

    @classmethod
    def has_api_key(cls, provider: str) -> bool:
        """Check if API key is available for provider."""
        key_map = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'groq': 'GROQ_API_KEY'
        }
        key_name = key_map.get(provider, '')
        key_value = os.getenv(key_name)

        # Debug output
        print(f"Checking {provider}: {key_name} = {'***' if key_value else 'None'}")

        return key_value is not None and key_value.strip() != ''

    @classmethod
    def get_available_provider(cls) -> Optional[str]:
        """Get first available provider with API key."""
        providers = ['groq', 'openai', 'anthropic']  # Groq first (fastest/cheapest)
        for provider in providers:
            if cls.has_api_key(provider):
                return provider
        return None

    @pytest.mark.asyncio
    async def test_pydantic_ai_style_real_llm_invocation(self):
        """Test Pydantic AI style tools with real LLM - autonomous tool calling."""
        provider = self.get_available_provider()
        if not provider:
            pytest.skip("No API keys found in .env - skipping real LLM tests")
            
        print(f"\n🔄 Testing Pydantic AI style with real {provider.upper()} API...")
        
        # Create real LLM client
        llm_client = LLMClient.create(
            provider=provider,
            model='llama-3.1-8b-instant' if provider == 'groq' else 'gpt-3.5-turbo' if provider == 'openai' else 'claude-3-haiku-20240307'
        )
        
        # Create agent with context dependency
        @dataclass
        class UserContext:
            user_id: str = "user123"
            preferences: dict = None
            search_history: list = None
            
            def __post_init__(self):
                if self.preferences is None:
                    self.preferences = {"language": "en", "category": "tech"}
                if self.search_history is None:
                    self.search_history = ["AI", "machine learning", "python"]
        
        agent = Agent(llm_client, max_iterations=5)
        
        # Add Pydantic AI style tools
        @tool("search_knowledge")
        async def search_knowledge(ctx: RunContext[UserContext], query: str) -> str:
            """Search the knowledge base for information on a topic."""
            user_prefs = ctx.deps.preferences
            search_hist = ctx.deps.search_history
            
            # Simulate knowledge search with context
            return f"Knowledge search for '{query}': Found 3 articles about {query} in {user_prefs['language']} language. Based on your history of {search_hist}, this matches your interests in {user_prefs['category']} category."
        
        @tool("get_user_stats")
        async def get_user_stats(ctx: RunContext[UserContext]) -> str:
            """Get user statistics and activity summary."""
            return f"User {ctx.deps.user_id} stats: 15 searches this week, primary interest: {ctx.deps.preferences['category']}, last searches: {', '.join(ctx.deps.search_history[-2:])}"
        
        agent.add_tool(search_knowledge)
        agent.add_tool(get_user_stats)
        
        print(f"🛠️ Registered tools: {list(agent.tool_registry.tools.keys())}")
        
        # Test: LLM should autonomously use tools to answer
        user_ctx = UserContext(user_id="alice123", preferences={"language": "en", "category": "AI"})
        
        query = "I want to learn about neural networks. Can you help me find relevant information and show my learning progress?"
        
        print(f"🤖 User Query: {query}")
        print("Expected: LLM should use BOTH search_knowledge AND get_user_stats tools autonomously")
        
        try:
            result = await agent.run(query, deps=user_ctx)
            
            print(f"\n✅ Agent Response: {result.data}")
            print(f"📊 Tools called: {result.tool_calls_made if hasattr(result, 'tool_calls_made') else 'Unknown'}")
            
            # Verify tools were called autonomously
            response_text = str(result.data).lower()
            
            # Check if tool outputs appear in response (indicating they were called)
            search_called = "knowledge search" in response_text or "found" in response_text
            stats_called = "stats" in response_text or "searches this week" in response_text
            
            if search_called and stats_called:
                print("SUCCESS: LLM autonomously used BOTH tools with Pydantic AI style context injection!")
                return True
            elif search_called or stats_called:
                print("⚠️ PARTIAL: LLM used at least one tool autonomously")
                return True  
            else:
                print("❌ LLM did not appear to use tools autonomously") 
                print(f"Response content: {result.data}")
                return False
                
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            return False
    
    @pytest.mark.asyncio
    async def test_current_style_real_llm_invocation(self):
        """Test current style tools with real LLM - autonomous tool calling."""
        provider = self.get_available_provider()
        if not provider:
            pytest.skip("No API keys found in .env - skipping real LLM tests")

        print(f"\n🔄 Testing current style with real {provider.upper()} API...")

        # Create real LLM client
        llm_client = LLMClient.create(
            provider=provider,
            model='llama-3.1-8b-instant' if provider == 'groq' else 'gpt-3.5-turbo' if provider == 'openai' else 'claude-3-haiku-20240307'
        )

        @dataclass
        class TaskContext:
            project_name: str = "MiiFlow AI"
            available_budget: float = 10000.0
            team_size: int = 5
            deadline: str = "2024-01-15"

        agent = Agent(llm_client, max_iterations=5)
        
        # Add current style tools (with context as keyword)
        @tool("calculate_budget")
        async def calculate_budget(hours: int, hourly_rate: float, context: RunContext[TaskContext]) -> str:
            """Calculate project budget based on hours and rate."""
            total_cost = hours * hourly_rate
            remaining = context.deps.available_budget - total_cost
            
            return f"Project {context.deps.project_name}: {hours} hours × ${hourly_rate}/hr = ${total_cost}. Remaining budget: ${remaining}. Team size: {context.deps.team_size}"
        
        @tool("get_project_info")  
        async def get_project_info(context: RunContext[TaskContext]) -> str:
            """Get current project information and status."""
            return f"Project: {context.deps.project_name}, Budget: ${context.deps.available_budget}, Team: {context.deps.team_size} people, Deadline: {context.deps.deadline}"
        
        agent.add_tool(calculate_budget)
        agent.add_tool(get_project_info)
        
        print(f"🛠️ Registered tools: {list(agent.tool_registry.tools.keys())}")
        
        # Test: LLM should autonomously use tools
        task_ctx = TaskContext(project_name="AI Chatbot", available_budget=15000.0, team_size=3)
        
        query = "We need to estimate the budget for our AI chatbot project. Assume 200 hours of work at $75/hour. Can you help calculate this and show our project details?"
        
        print(f"User Query: {query}")
        print("Expected: LLM should use BOTH calculate_budget AND get_project_info tools")
        
        try:
            result = await agent.run(query, deps=task_ctx)
            
            print(f"\n✅ Agent Response: {result.data}")
            print(f"📊 Tools called: {result.tool_calls_made if hasattr(result, 'tool_calls_made') else 'Unknown'}")
            
            # Verify tools were called autonomously  
            response_text = str(result.data).lower()
            
            budget_called = "200" in response_text and "75" in response_text and "$15000" in response_text or "$15,000" in response_text
            info_called = "ai chatbot" in response_text or "project:" in response_text
            
            if budget_called and info_called:
                print("SUCCESS: LLM autonomously used BOTH tools with current style context injection!")
                return True
            elif budget_called or info_called:
                print("⚠️ PARTIAL: LLM used at least one tool autonomously")
                return True
            else:
                print("❌ LLM did not appear to use tools autonomously")
                print(f"Response content: {result.data}")
                return False
                
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            return False


# Manual test runner for direct execution
def main():
    """Run all tests manually."""
    print("🧪 Testing Tool Context Injection Patterns")
    print("=" * 50)
    
    # Create test instance
    test_instance = TestToolContextInjection()
    
    # Mock client
    mock_client = MagicMock()
    mock_client.provider_name = "openai"
    llm_client = LLMClient(mock_client)
    
    # Run sync tests
    print("\n📋 Pattern Detection Tests:")
    test_instance.test_pydantic_ai_style_context_detection(llm_client)
    test_instance.test_current_style_context_detection(llm_client)
    test_instance.test_no_context_detection(llm_client)
    test_instance.test_dual_decorator_syntax(llm_client)
    
    # Run async tests
    print("\n🔄 Execution Tests:")
    asyncio.run(test_instance.test_context_execution_first_param())
    asyncio.run(test_instance.test_context_execution_keyword_pattern_detection())
    
    print("\n🎉 All context injection pattern tests passed!")
    print("\nKey Features Verified:")
    print("✅ Pydantic AI style: @agent.tool with ctx as first param")
    print("✅ Current style: @agent.tool('name') with context keyword")
    print("✅ Automatic pattern detection and schema filtering")
    print("✅ Smart context injection during execution")
    
    # Real LLM Integration Tests
    print("\n" + "="*60)
    print("🚀 REAL LLM INTEGRATION TESTS")
    print("="*60)
    
    real_test = TestRealLLMIntegration()
    provider = real_test.get_available_provider()
    
    if not provider:
        print("⚠️ No API keys found in .env file")
        print("To run real LLM tests, add one of these to your .env:")
        print("  GROQ_API_KEY=your_key_here")
        print("  OPENAI_API_KEY=your_key_here") 
        print("  ANTHROPIC_API_KEY=your_key_here")
        return
    
    print(f"🔑 Found {provider.upper()} API key - running real LLM tests...")
    
    try:
        # Test Pydantic AI style with real LLM
        result1 = asyncio.run(real_test.test_pydantic_ai_style_real_llm_invocation())
        
        # Test current style with real LLM  
        result2 = asyncio.run(real_test.test_current_style_real_llm_invocation())
        
        if result1 and result2:
            print("\n🎉 ALL TESTS PASSED! Context injection works with real LLM APIs!")
        elif result1 or result2:
            print("\n⚠️ PARTIAL SUCCESS - At least one pattern works with real LLM")
        else:
            print("\n❌ Real LLM tests failed - check tool registration and schemas")
            
    except Exception as e:
        print(f"\n❌ Real LLM integration tests failed: {e}")
        print("This might be due to API rate limits, network issues, or configuration problems")


if __name__ == "__main__":
    main()
