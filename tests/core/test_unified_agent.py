"""Tests for the unified Agent architecture (stateless)."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from miiflow_llm.core.agent import (
    Agent, RunContext, RunResult, AgentType
)
from miiflow_llm.core.client import LLMClient, ChatResponse
from miiflow_llm.core.message import Message, MessageRole
from miiflow_llm.core.metrics import TokenCount
from dataclasses import dataclass
from typing import Optional


# Test-specific dependency container (stateless)
@dataclass
class MockDeps:
    """Test dependency container for stateless operations."""
    
    api_key: Optional[str] = None
    user_role: Optional[str] = None
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TestUnifiedAgentArchitecture:
    """Test the unified agent system (stateless)."""
    
    @pytest.fixture
    def mock_client(self):
        """Mock LLMClient for testing."""
        client = MagicMock()
        client.provider_name = "openai"
        client.achat = AsyncMock()
        
        # Mock the convert_schema_to_provider_format method to return proper schemas
        def mock_convert_schema(schema):
            """Mock schema conversion that returns properly formatted tool schemas."""
            if isinstance(schema, dict):
                # Return OpenAI-style function calling schema
                return {
                    "type": "function",
                    "function": {
                        "name": schema.get('name', 'unknown_tool'),
                        "description": schema.get('description', 'A tool function'),
                        "parameters": schema.get('parameters', {
                            "type": "object",
                            "properties": {},
                            "required": []
                        })
                    }
                }
            return schema
        
        client.convert_schema_to_provider_format = MagicMock(side_effect=mock_convert_schema)
        return client
    
    @pytest.fixture
    def llm_client(self, mock_client):
        """LLMClient instance with mock."""
        return LLMClient(mock_client)
    
    @pytest.fixture
    def test_deps(self):
        """MockDeps instance for testing."""
        return MockDeps(
            api_key="test_key_123",
            user_role="admin",
            metadata={"environment": "test"}
        )
    
    def test_agent_type_configuration(self, llm_client):
        """Test that different agent types are configured properly."""
        single_hop_agent = Agent(llm_client, agent_type=AgentType.SINGLE_HOP)
        assert single_hop_agent.agent_type == AgentType.SINGLE_HOP
        
        react_agent = Agent(llm_client, agent_type=AgentType.REACT)
        assert react_agent.agent_type == AgentType.REACT
        assert react_agent.max_iterations == 10  # Default value
    
    def test_run_context_stateless(self, test_deps):
        """Test RunContext with stateless dependency injection."""
        context = RunContext(
            deps=test_deps,
            metadata={"request_id": "12345"}
        )
        
        assert context.deps.user_role == "admin"
        assert context.metadata.get("request_id") == "12345"
        assert len(context.messages) == 0  # No persisted history
    
    def test_stateless_agent_creation(self, llm_client):
        """Test stateless agent creation for different types."""
        single_hop_agent = Agent(llm_client, agent_type=AgentType.SINGLE_HOP, deps_type=MockDeps)
        assert single_hop_agent.agent_type == AgentType.SINGLE_HOP
        assert single_hop_agent.deps_type == MockDeps
        assert len(single_hop_agent._tools) == 0
        
        react_agent = Agent(llm_client, agent_type=AgentType.REACT, deps_type=MockDeps)
        assert react_agent.agent_type == AgentType.REACT
        assert react_agent.max_iterations == 10
    
    @pytest.mark.asyncio
    async def test_agent_with_dependency_injection(self, llm_client, mock_client, test_deps):
        """Test agent with stateless dependency injection."""
        agent = Agent(llm_client, agent_type=AgentType.SINGLE_HOP, deps_type=MockDeps)
        
        @agent.tool(name="get_user_info", description="Get user information from deps")
        async def get_user_info(context: RunContext[MockDeps]) -> str:
            """Get user info from dependency container."""
            try:
                if context.deps and context.deps.user_role:
                    return f"User role: {context.deps.user_role}, API key: {context.deps.api_key[:8]}..."
                return "No user info available"
            except Exception as e:
                return f"Error getting user info: {e}"
        
        tool_call_response = ChatResponse(
            message=Message(
                role=MessageRole.ASSISTANT,
                content="",
                tool_calls=[{
                    "id": "call_123",
                    "function": {
                        "name": "get_user_info",
                        "arguments": '{}'
                    }
                }]
            ),
            usage=TokenCount(prompt_tokens=20, completion_tokens=10, total_tokens=30),
            model="gpt-4",
            provider="openai"
        )
        
        final_response = ChatResponse(
            message=Message(role=MessageRole.ASSISTANT, content="Hello! I can see your user information."),
            usage=TokenCount(prompt_tokens=25, completion_tokens=8, total_tokens=33),
            model="gpt-4",
            provider="openai"
        )
        
        mock_client.achat.side_effect = [tool_call_response, final_response]
        
        result = await agent.run("Hello", deps=test_deps)
        
        assert isinstance(result, RunResult)
        assert result.data == "Hello! I can see your user information."
    
    @pytest.mark.asyncio
    async def test_react_agent_with_tools(self, llm_client, mock_client, test_deps):
        """Test REACT agent using tools."""
        agent = Agent(llm_client, agent_type=AgentType.REACT, deps_type=MockDeps)
        
        @agent.tool(name="calculate", description="Perform calculations")
        async def calculate(context: RunContext[MockDeps], expression: str) -> str:
            """Simple calculator tool."""
            try:
                # Safe eval for simple expressions
                if all(c in "0123456789+-*/.()" for c in expression.replace(" ", "")):
                    result = eval(expression)
                    return f"Result: {result}"
                return "Invalid expression"
            except Exception as e:
                return f"Calculation error: {e}"
        
        tool_call_response = ChatResponse(
            message=Message(
                role=MessageRole.ASSISTANT,
                content="",
                tool_calls=[{
                    "id": "call_456", 
                    "function": {
                        "name": "calculate",
                        "arguments": '{"expression": "15 * 4"}'
                    }
                }]
            ),
            usage=TokenCount(prompt_tokens=30, completion_tokens=15, total_tokens=45),
            model="gpt-4",
            provider="openai"
        )
        
        final_response = ChatResponse(
            message=Message(role=MessageRole.ASSISTANT, content="The result is 60."),
            usage=TokenCount(prompt_tokens=40, completion_tokens=12, total_tokens=52),
            model="gpt-4", 
            provider="openai"
        )
        
        mock_client.achat.side_effect = [tool_call_response, final_response]
        
        result = await agent.run("What is 15 * 4?", deps=test_deps)
        
        assert result.data == "The result is 60."
    
    def test_agent_with_custom_prompt(self, llm_client):
        """Test agent creation with custom configuration."""
        single_hop_agent = Agent(llm_client, agent_type=AgentType.SINGLE_HOP, deps_type=MockDeps)
        assert single_hop_agent.agent_type == AgentType.SINGLE_HOP
        
        react_agent = Agent(llm_client, agent_type=AgentType.REACT, deps_type=MockDeps)
        assert react_agent.agent_type == AgentType.REACT
        
        custom_agent = Agent(
            llm_client, 
            agent_type=AgentType.REACT,
            deps_type=MockDeps,
            system_prompt="You are a helpful assistant specialized in calculations."
        )
        assert custom_agent.system_prompt == "You are a helpful assistant specialized in calculations."
    
    def test_message_history_passthrough(self, llm_client):
        """Test that message history can be passed through but not persisted."""
        agent = Agent(llm_client, agent_type=AgentType.SINGLE_HOP)
        
        # Create some message history
        message_history = [
            Message(role=MessageRole.USER, content="Hello"),
            Message(role=MessageRole.ASSISTANT, content="Hi there!"),
        ]
        
        context = RunContext(
            deps=None,
            messages=message_history
        )
        
        assert len(context.messages) == 2
        assert context.last_user_message().content == "Hello"
        assert context.last_agent_message().content == "Hi there!"
    
    @pytest.mark.asyncio 
    async def test_stateless_tool_execution(self, llm_client, mock_client):
        """Test that tools execute without any memory persistence."""
        agent = Agent(llm_client, agent_type=AgentType.SINGLE_HOP)
        
        @agent.tool(name="echo", description="Echo the input")
        async def echo_tool(text: str) -> str:
            """Simple echo tool without context injection."""
            return f"Echo: {text}"
        
        tool_call_response = ChatResponse(
            message=Message(
                role=MessageRole.ASSISTANT,
                content="",
                tool_calls=[{
                    "id": "call_echo",
                    "function": {
                        "name": "echo",
                        "arguments": '{"text": "Hello World"}'
                    }
                }]
            ),
            usage=TokenCount(prompt_tokens=15, completion_tokens=5, total_tokens=20),
            model="gpt-4",
            provider="openai"
        )
        
        final_response = ChatResponse(
            message=Message(role=MessageRole.ASSISTANT, content="I echoed your message."),
            usage=TokenCount(prompt_tokens=20, completion_tokens=6, total_tokens=26),
            model="gpt-4",
            provider="openai"
        )
        
        mock_client.achat.side_effect = [tool_call_response, final_response]
        
        result = await agent.run("Please echo 'Hello World'")
        
        assert result.data == "I echoed your message."
        # Verify tool was registered
        assert "echo" in agent.tool_registry.tools


class TestStatelessPatterns:
    """Test patterns for stateless agent operations."""
    
    def test_tool_registration_patterns(self):
        """Test various tool registration patterns."""
        from miiflow_llm.core.client import LLMClient
        mock_provider = MagicMock()
        mock_provider.provider_name = "test"
        llm_client = LLMClient(mock_provider)
        
        agent = Agent(llm_client, agent_type=AgentType.REACT)
        
        @agent.tool("search")
        async def search_tool(query: str) -> str:
            return f"Searching for: {query}"
            
        @agent.tool(name="compute", description="Perform computation")  
        async def compute_tool(expression: str) -> str:
            return f"Computing: {expression}"
        
        assert "search" in agent.tool_registry.tools
        assert "compute" in agent.tool_registry.tools
    
    def test_agent_types_behavior(self):
        """Test that different agent types behave correctly."""
        from miiflow_llm.core.client import LLMClient
        mock_provider = MagicMock()
        mock_provider.provider_name = "test"
        llm_client = LLMClient(mock_provider)
        
        single_hop = Agent(llm_client, agent_type=AgentType.SINGLE_HOP)
        react = Agent(llm_client, agent_type=AgentType.REACT)
        
        assert single_hop.max_iterations == 10  # Default
        assert react.max_iterations == 10
        assert single_hop.agent_type != react.agent_type
    
    def test_context_injection_patterns(self):
        """Test context injection works without memory persistence."""
        from miiflow_llm.core.client import LLMClient
        mock_provider = MagicMock()  
        mock_provider.provider_name = "test"
        llm_client = LLMClient(mock_provider)
        
        agent = Agent(llm_client, agent_type=AgentType.SINGLE_HOP, deps_type=MockDeps)
        
        @agent.tool("get_context")
        async def get_context_tool(context: RunContext[MockDeps]) -> str:
            """Tool that uses context injection."""
            if context.deps:
                return f"Context role: {context.deps.user_role}"
            return "No context"
        
        # Verify tool was registered with context injection
        tool = agent.tool_registry.tools["get_context"]
        assert hasattr(tool, 'context_injection')


class TestImportPatterns:
    """Test that imports work correctly."""
    
    def test_core_imports(self):
        """Test that applications can import everything they need."""
        from miiflow_llm.core import Agent, RunContext, RunResult, AgentType
        
        assert Agent is not None
        assert RunContext is not None
        assert AgentType.SINGLE_HOP is not None
        assert AgentType.REACT is not None
    
    def test_agent_typing_support(self):
        """Test that proper type annotations work.""" 
        from miiflow_llm.core import Agent
        
        def create_typed_agent(client) -> Agent[MockDeps, str]:
            return Agent(client, deps_type=MockDeps, result_type=str)
        
        agent_type = Agent[MockDeps, str]
        assert agent_type is not None
