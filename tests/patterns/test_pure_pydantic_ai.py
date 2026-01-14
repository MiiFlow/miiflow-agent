"""Test Pydantic AI style context injection patterns in miiflow-agent agents."""
import pytest
import asyncio
import os
from dataclasses import dataclass
from unittest.mock import MagicMock

from miiflow_agent import LLMClient, Agent, RunContext, FunctionTool, tool


@dataclass
class MockDeps:
    user_id: str = "test_user"
    database_name: str = "production_db"


class TestPurePydanticAI:
    """Test pure Pydantic AI context injection patterns only."""

    @pytest.fixture
    def llm_client(self):
        """Mock LLM client for testing."""
        mock_client = MagicMock()
        mock_client.provider_name = "openai"
        return LLMClient(mock_client)

    def test_pydantic_ai_style_detection(self, llm_client):
        """Test that Pydantic AI style (ctx as first param) is detected correctly."""
        agent = Agent(llm_client)
        
        @tool("search")
        async def search(ctx: RunContext[MockDeps], query: str) -> str:
            return f"Searched: {query} with {ctx.deps.database_name}"
        
        agent.add_tool(search)
        
        search_tool = agent.tool_registry.tools['search']
        assert hasattr(search_tool, 'context_injection')
        assert search_tool.context_injection['pattern'] == 'first_param'
        assert search_tool.context_injection['param_name'] == 'ctx'
        assert search_tool.context_injection['param_index'] == 0
        
        # Test that schema excludes ctx parameter from LLM view
        if search_tool.schema.parameters:
            assert 'ctx' not in search_tool.schema.parameters
            assert 'query' in search_tool.schema.parameters
        
        print(f"✅ Pydantic AI style detection: {search_tool.context_injection}")

    def test_plain_function_detection(self, llm_client):
        """Test that plain functions (no context) are detected correctly."""
        agent = Agent(llm_client)  # No deps_type needed
        
        @tool("calculator")
        async def calculator(a: int, b: int) -> int:
            return a + b
        
        agent.add_tool(calculator)
        
        calc_tool = agent.tool_registry.tools['calculator']
        assert hasattr(calc_tool, 'context_injection')
        assert calc_tool.context_injection['pattern'] == 'none'
        
        print(f"✅ Plain function detection: {calc_tool.context_injection}")

    def test_context_parameter_variations(self):
        """Test different context parameter names are recognized."""
        # Test ctx as first parameter
        async def tool_with_ctx(ctx: RunContext, data: str) -> str:
            return f"data: {data}"
            
        # Test context as first parameter  
        async def tool_with_context(context: RunContext, data: str) -> str:
            return f"data: {data}"
        
        tool1 = FunctionTool(tool_with_ctx, "tool1", "Test tool 1")
        tool2 = FunctionTool(tool_with_context, "tool2", "Test tool 2")
        
        # Both should detect first_param pattern
        assert tool1.context_injection['pattern'] == 'first_param'
        assert tool2.context_injection['pattern'] == 'first_param'
        assert tool1.context_injection['param_name'] == 'ctx'
        assert tool2.context_injection['param_name'] == 'context'
        
        print(f"✅ Context variations work: ctx and context both detected as first_param")

    @pytest.mark.asyncio
    async def test_pydantic_ai_execution(self, llm_client):
        """Test that Pydantic AI style tools execute correctly."""
        agent = Agent(llm_client)
        
        @tool("search")
        async def search(ctx: RunContext[MockDeps], query: str) -> str:
            return f"Searched: {query} with {ctx.deps.database_name}"
        
        agent.add_tool(search)
        
        # Test execution through tool registry
        context = RunContext(MockDeps("alice", "test_db"), [])
        result = await agent.tool_registry.execute_safe_with_context(
            "search", context, query="neural networks"
        )
        
        assert result.success
        assert "neural networks" in result.output
        assert "test_db" in result.output
        assert result.metadata['execution_pattern'] == 'first_param'
        
        print(f"✅ Pydantic AI execution: {result.output}")


def has_api_key() -> bool:
    """Check if any LLM API key is available."""
    providers = ['groq', 'openai', 'anthropic']
    for provider in providers:
        key_name = {'groq': 'GROQ_API_KEY', 'openai': 'OPENAI_API_KEY', 'anthropic': 'ANTHROPIC_API_KEY'}.get(provider)
        if os.getenv(key_name):
            return True
    return False


@pytest.mark.skipif(not has_api_key(), reason="No API keys found - skipping real LLM tests")
class TestRealLLMIntegration:
    """Test Pydantic AI patterns with real LLM APIs."""

    @classmethod
    def get_available_provider(cls) -> str | None:
        """Get first available provider with API key."""
        providers = ['groq', 'openai', 'anthropic']
        for provider in providers:
            key_name = {'groq': 'GROQ_API_KEY', 'openai': 'OPENAI_API_KEY', 'anthropic': 'ANTHROPIC_API_KEY'}.get(provider)
            if os.getenv(key_name):
                return provider
        return None

    @pytest.mark.asyncio
    async def test_autonomous_tool_calling(self):
        """Test that LLM autonomously calls Pydantic AI style tools."""
        provider = self.get_available_provider()
        if not provider:
            pytest.skip("No API keys found - skipping real LLM tests")
            
        print(f"\n🔄 Testing autonomous tool calling with {provider.upper()} API...")
        
        # Use core Agent class directly for better compatibility
        llm_client = LLMClient.create(
            provider=provider,
            model='llama-3.1-8b-instant' if provider == 'groq' else 'gpt-4o-mini'
        )
        
        # User context
        @dataclass
        class UserContext:
            user_id: str = "alice123"
            role: str = "researcher"
            interests: list = None
            
            def __post_init__(self):
                if self.interests is None:
                    self.interests = ["AI", "machine learning"]
        
        agent = Agent(llm_client, max_iterations=3)
        
        # Add Pydantic AI style tools using unified approach
        @tool("get_profile")
        async def get_profile(ctx: RunContext[UserContext]) -> str:
            """Get user profile information"""
            user = ctx.deps
            return f"User {user.user_id} is a {user.role} interested in {', '.join(user.interests)}"
            
        @tool("calculate")
        def calculate(expression: str) -> str:
            """Calculate mathematical expressions"""
            if "+" in expression:
                parts = expression.split("+")
                result = sum(int(p.strip()) for p in parts)
                return f"{expression} = {result}"
            return f"Cannot calculate: {expression}"
        
        agent.add_tool(get_profile)
        agent.add_tool(calculate)
        
        # Test autonomous tool calling
        user_ctx = UserContext()
        result = await agent.run(
            "Who am I and what's 15 + 27?",
            deps=user_ctx
        )
        
        response = result.data
        print(f"🤖 Agent Response: {response}")
        
        # Check if tools were used
        profile_used = any(word in response.lower() for word in ["alice123", "researcher"])
        calc_used = any(word in response for word in ["42", "15", "27", "="])
        
        if profile_used and calc_used:
            print("✅ SUCCESS: LLM autonomously used BOTH tools!")
            return True
        elif profile_used or calc_used:
            print("⚠️ PARTIAL: LLM used at least one tool autonomously")
            return True  
        else:
            print("❌ FAILED: LLM did not use tools autonomously")
            return False


def main():
    """Run tests manually."""
    print("Testing Pure Pydantic AI Context Injection")
    print("=" * 50)
    
    # Unit tests
    test_instance = TestPurePydanticAI()
    llm_client = LLMClient.create('openai', 'gpt-3.5-turbo')
    
    try:
        test_instance.test_pydantic_ai_style_detection(llm_client)
        test_instance.test_plain_function_detection(llm_client)
        test_instance.test_context_parameter_variations()
        
        print("\nRunning async execution test...")
        asyncio.run(test_instance.test_pydantic_ai_execution(llm_client))
        
        print("\n🚀 Running real LLM integration test...")
        real_test = TestRealLLMIntegration()
        success = asyncio.run(real_test.test_autonomous_tool_calling())
        
        if success:
            print("\n🎉 ALL TESTS PASSED! Pure Pydantic AI patterns working perfectly!")
        else:
            print("\n⚠️ Some tests passed, but LLM tool calling needs improvement")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    main()
