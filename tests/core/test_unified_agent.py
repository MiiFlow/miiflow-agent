"""Tests for the unified Agent architecture (stateless)."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from miiflow_llm.core.agent import (
    Agent, RunContext, RunResult, AgentType
)
from miiflow_llm.core.client import LLMClient, ChatResponse
from miiflow_llm.core.message import Message, MessageRole
from miiflow_llm.core.metrics import TokenCount
from miiflow_llm.core.tools import tool
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
        single_hop_agent = Agent(llm_client, agent_type=AgentType.SINGLE_HOP)
        assert single_hop_agent.agent_type == AgentType.SINGLE_HOP
        assert len(single_hop_agent._tools) == 0

        react_agent = Agent(llm_client, agent_type=AgentType.REACT)
        assert react_agent.agent_type == AgentType.REACT
        assert react_agent.max_iterations == 10



    def test_agent_with_custom_prompt(self, llm_client):
        """Test agent creation with custom configuration."""
        single_hop_agent = Agent(llm_client, agent_type=AgentType.SINGLE_HOP)
        assert single_hop_agent.agent_type == AgentType.SINGLE_HOP

        react_agent = Agent(llm_client, agent_type=AgentType.REACT)
        assert react_agent.agent_type == AgentType.REACT

        custom_agent = Agent(
            llm_client,
            agent_type=AgentType.REACT,
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
    


class TestStatelessPatterns:
    """Test patterns for stateless agent operations."""
    
    def test_tool_registration_patterns(self):
        """Test various tool registration patterns using unified @tool approach."""
        
        
        
        mock_provider = MagicMock()
        mock_provider.provider_name = "test"
        llm_client = LLMClient(mock_provider)
        
        agent = Agent(llm_client, agent_type=AgentType.REACT)
        
        @tool("search")
        async def search_tool(query: str) -> str:
            return f"Searching for: {query}"
            
        @tool(name="compute", description="Perform computation")  
        async def compute_tool(expression: str) -> str:
            return f"Computing: {expression}"
        
        # Register tools with agent using unified approach
        agent.add_tool(search_tool)
        agent.add_tool(compute_tool)
        
        assert "search" in agent.tool_registry.tools
        assert "compute" in agent.tool_registry.tools
    
    def test_agent_types_behavior(self):
        """Test that different agent types behave correctly."""
        
        mock_provider = MagicMock()
        mock_provider.provider_name = "test"
        llm_client = LLMClient(mock_provider)
        
        single_hop = Agent(llm_client, agent_type=AgentType.SINGLE_HOP)
        react = Agent(llm_client, agent_type=AgentType.REACT)
        
        assert single_hop.max_iterations == 10  # Default
        assert react.max_iterations == 10
        assert single_hop.agent_type != react.agent_type
    
    def test_context_injection_patterns(self):
        """Test context injection works without memory persistence using unified @tool approach."""

        mock_provider = MagicMock()
        mock_provider.provider_name = "test"
        llm_client = LLMClient(mock_provider)

        agent = Agent(llm_client, agent_type=AgentType.SINGLE_HOP)

        @tool("get_context")
        async def get_context_tool(context: RunContext[MockDeps]) -> str:
            """Tool that uses context injection."""
            if context.deps:
                return f"Context role: {context.deps.user_role}"
            return "No context"

        # Register tool with agent using unified approach
        agent.add_tool(get_context_tool)

        # Verify tool was registered with context injection
        context_tool = agent.tool_registry.tools["get_context"]
        assert hasattr(context_tool, 'context_injection')


class TestStreamSingleHop:
    """Tests for _stream_single_hop method."""

    @pytest.mark.asyncio
    async def test_stream_single_hop_adds_user_message(self):
        """Test that _stream_single_hop adds user message to context."""
        from unittest.mock import MagicMock
        from miiflow_llm.core.agent import Agent, RunContext
        from miiflow_llm.core.message import Message, MessageRole
        from miiflow_llm.core.client import StreamChunk

        # Create a mock client
        mock_client = MagicMock()

        # Setup mock streaming response
        async def mock_stream(*args, **kwargs):
            yield StreamChunk(content="Hello", delta="Hello", finish_reason=None)
            yield StreamChunk(content="Hello!", delta="!", finish_reason="stop")

        mock_client.astream_chat = mock_stream

        # Create agent
        agent = Agent(client=mock_client, agent_type="single_hop")

        # Create context with some history but NO user message yet
        context = RunContext(
            deps=None, messages=[Message.system("You are helpful"), Message.user("Previous question")]
        )

        user_prompt = "What is 2+2?"

        # Stream single hop
        events = []
        async for event in agent._stream_single_hop(user_prompt, context=context):
            events.append(event)

        # Verify user message was added to context
        assert len(context.messages) == 4  # system + previous user + new user + assistant response
        assert context.messages[2].role == MessageRole.USER
        assert context.messages[2].content == user_prompt

    @pytest.mark.asyncio
    async def test_stream_single_hop_doesnt_duplicate_message(self):
        """Test that _stream_single_hop doesn't duplicate if message already exists."""
        from unittest.mock import MagicMock
        from miiflow_llm.core.agent import Agent, RunContext
        from miiflow_llm.core.message import Message, MessageRole
        from miiflow_llm.core.client import StreamChunk

        # Create a mock client
        mock_client = MagicMock()

        # Setup mock streaming response
        async def mock_stream(*args, **kwargs):
            yield StreamChunk(content="4", delta="4", finish_reason="stop")

        mock_client.astream_chat = mock_stream

        # Create agent
        agent = Agent(client=mock_client, agent_type="single_hop")

        user_prompt = "What is 2+2?"

        # Create context with user message already present (simulating run() flow)
        context = RunContext(
            deps=None, messages=[Message.system("You are helpful"), Message.user(user_prompt)]
        )

        initial_message_count = len(context.messages)

        # Stream single hop
        events = []
        async for event in agent._stream_single_hop(user_prompt, context=context):
            events.append(event)

        # Verify message count increased by only 1 (the assistant response)
        # Should NOT add duplicate user message
        assert len(context.messages) == initial_message_count + 1
        # Last user message should still be the original one
        user_messages = [m for m in context.messages if m.role == MessageRole.USER]
        assert len(user_messages) == 1
        assert user_messages[0].content == user_prompt


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

        # Type annotations work even without deps_type/result_type params
        def create_typed_agent(client) -> Agent[MockDeps, str]:
            return Agent(client)

        agent_type = Agent[MockDeps, str]
        assert agent_type is not None