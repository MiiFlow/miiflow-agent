"""Clean interface for interacting with agents."""

import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from ..core import LLMClient, Message, MessageRole
from ..core.tools import FunctionTool, ToolRegistry
from ..core.tools.decorators import get_tool_from_function, is_tool
from ..core.agent import Agent, RunResult, AgentType, RunContext
from .context import AgentContext

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for creating an agent."""
    
    provider: str  # "openai", "anthropic", etc.
    model: str     # "gpt-4", "claude-3", etc.
    agent_type: AgentType  # SINGLE_HOP or REACT
    tools: List[FunctionTool] = None
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_iterations: int = 10
    
    def __post_init__(self):
        if self.tools is None:
            self.tools = []


class AgentClient:
    """Clean interface for interacting with agents."""
    
    def __init__(self, config: AgentConfig, agent: Agent):
        self.config = config
        self.agent = agent
        self.tool_registry = agent.tool_registry
    
    async def run(
        self, 
        prompt: str, 
        context: Optional[AgentContext] = None,
        message_history: Optional[List[Message]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Run agent with structured output (stateless)."""
        if context is None:
            context = AgentContext()
            
        # Pass message history directly to agent if provided
        result: RunResult = await self.agent.run(
            prompt,
            message_history=message_history,
            **kwargs
        )
        
        # Count tool calls from messages
        tool_calls_count = sum(len(msg.tool_calls) for msg in result.messages if msg.tool_calls)
        tool_messages_count = sum(1 for msg in result.messages if msg.role == MessageRole.TOOL)
        
        logger.debug(f"Final metrics - Individual tool calls: {tool_calls_count}, Tool result messages: {tool_messages_count}")
        
        return {
            "response": result.data,
            "reasoning_steps": len(result.all_messages),
            "tool_calls_made": tool_calls_count,
            "tool_results_received": tool_messages_count,
            "metadata": {
                "model": self.config.model,
                "provider": self.config.provider,
                "agent_type": self.config.agent_type.value
            }
        }
    
    def add_tool(self, tool: FunctionTool) -> None:
        """Add a tool to this agent instance."""
        self.agent.tool_registry.register(tool)
        self.agent._tools.append(tool)
    
    def list_tools(self) -> List[str]:
        """List tools registered with this agent."""
        return self.agent.tool_registry.list_tools()
    
    async def stream_react(
        self,
        prompt: str,
        context: Optional[AgentContext] = None,
        message_history: Optional[List[Message]] = None,
        **kwargs
    ):
        """Stream ReAct execution (stateless).
        
        Delegates to Agent.stream_react while maintaining clean interface.
        """
        if context is None:
            context = AgentContext()
            
        # Create a basic RunContext for streaming
        run_context = RunContext(
            deps=context,
            messages=message_history or []
        )
        
        try:
            async for event in self.agent.stream_react(prompt, run_context, **kwargs):
                yield event
        except Exception as e:
            # Emit error event with AgentClient context
            yield {
                "event": "error", 
                "data": {
                    "error": str(e), 
                    "error_type": type(e).__name__,
                    "context": "AgentClient.stream_react"
                }
            }
            raise
    
    async def stream_single_hop(
        self,
        prompt: str,
        context: Optional[AgentContext] = None,
        message_history: Optional[List[Message]] = None,
        **kwargs
    ):
        """Stream single-hop execution (stateless).
        
        Delegates to Agent.stream_single_hop while maintaining clean interface.
        """
        if context is None:
            context = AgentContext()
            
        try:
            async for event in self.agent.stream_single_hop(
                prompt,
                message_history=message_history,
                **kwargs
            ):
                yield event
        except Exception as e:
            # Emit error event with AgentClient context  
            yield {
                "event": "error",
                "data": {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "context": "AgentClient.stream_single_hop"
                }
            }
            raise


def create_agent(config: AgentConfig) -> AgentClient:
    """Factory function to create agents."""
    if not config.provider:
        raise ValueError("Provider is required")
    if not config.model:
        raise ValueError("Model is required")
        
    try:
        llm_client = LLMClient.create(
            provider=config.provider,
            model=config.model
        )
    except Exception as e:
        raise ValueError(f"Failed to create LLM client: {e}")
    
    agent = Agent(
        llm_client,
        agent_type=config.agent_type,
        system_prompt=config.system_prompt,
        temperature=config.temperature,
        max_iterations=config.max_iterations,
        tools=config.tools
    )
    
    return AgentClient(config, agent)
