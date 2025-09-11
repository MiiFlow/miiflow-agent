"""Unified Agent architecture with internal episodic memory."""

import asyncio
import time
import logging
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, Callable, Union, Protocol, AsyncIterator
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum

logger = logging.getLogger(__name__)

from .client import LLMClient
from .message import Message, MessageRole
from .tools import FunctionTool, ToolRegistry
from .exceptions import MiiflowLLMError, ErrorType

# Type variables for dependency injection and results
Deps = TypeVar('Deps')
Result = TypeVar('Result')


class AgentType(Enum):
    """Agent types based on reasoning approach."""
    SINGLE_HOP = "single_hop"      # Simple, direct response
    REACT = "react"                # ReAct with multi-hop reasoning


@dataclass
class RunResult(Generic[Result]):
    """Result from an agent run with metadata."""
    
    data: Result
    messages: List[Message]
    all_messages: List[Message] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.all_messages:
            self.all_messages = self.messages


class DatabaseService(Protocol):
    async def query(self, sql: str) -> List[Dict[str, Any]]: ...
    async def get_user_context(self, user_id: str) -> Dict[str, Any]: ...


class VectorStoreService(Protocol):
    async def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]: ...
    async def add_documents(self, documents: List[Dict[str, Any]]) -> None: ...


class ContextService(Protocol):
    async def retrieve_context(self, query: str, context_id: Optional[str] = None) -> Dict[str, Any]: ...
    async def store_context(self, context_id: str, context_data: Dict[str, Any]) -> None: ...


class SearchService(Protocol):
    async def search(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]: ...


@dataclass 
class RunContext(Generic[Deps]):
    """Context passed to tools and agent functions with dependency injection."""
    
    deps: Deps
    messages: List[Message] = field(default_factory=list)
    retry: int = 0
    user_id: Optional[str] = None
    thread_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def last_user_message(self) -> Optional[Message]:
        """Get the last user message."""
        for msg in reversed(self.messages):
            if msg.role == MessageRole.USER:
                return msg
        return None
    
    def last_agent_message(self) -> Optional[Message]:
        """Get the last agent message."""
        for msg in reversed(self.messages):
            if msg.role == MessageRole.ASSISTANT:
                return msg
        return None
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation for context."""
        if len(self.messages) <= 2:
            return "New conversation"
        
        user_messages = [msg.content for msg in self.messages if msg.role == MessageRole.USER]
        return f"Conversation with {len(user_messages)} user messages"
    
    def has_context(self, key: str) -> bool:
        """Check if context has a specific key."""
        return key in self.metadata


class Agent(Generic[Deps, Result]):
    """Unified Agent with internal episodic memory and dependency injection."""
    
    def __init__(
        self,
        client: LLMClient,
        *,
        agent_type: AgentType = AgentType.SINGLE_HOP,
        deps_type: Optional[Type[Deps]] = None,
        result_type: Optional[Type[Result]] = None,
        system_prompt: Optional[Union[str, Callable[[RunContext[Deps]], str]]] = None,
        retries: int = 1,
        max_iterations: int = 10,
        temperature: float = 0.7,
        tools: Optional[List[FunctionTool]] = None,
        max_stored_threads: int = 100,
        max_messages_per_thread: int = 1000
    ):
        self.client = client
        self.agent_type = agent_type
        self.deps_type = deps_type
        self.result_type = result_type or str
        self.system_prompt = system_prompt
        self.retries = retries
        self.max_iterations = max_iterations
        self.temperature = temperature
        
        # Internal episodic memory for conversation continuity
        self._thread_conversations: Dict[str, List[Message]] = {}
        self._thread_metadata: Dict[str, Dict[str, Any]] = {}
        self.max_stored_threads = max_stored_threads
        self.max_messages_per_thread = max_messages_per_thread
        
        # Share the tool registry with LLMClient for consistency
        self.tool_registry = self.client.tool_registry
        self._tools: List[FunctionTool] = []
        
        # Register provided tools
        if tools:
            for tool in tools:
                self.tool_registry.register(tool)
                self._tools.append(tool)
        
    
    def tool(
        self, 
        name_or_func=None, 
        description: Optional[str] = None, 
        *,
        name: Optional[str] = None
    ) -> Callable:
        """Decorator to register a tool with this agent.
        
        Supports multiple calling styles:
        
        @agent.tool  # Simple - uses function name
        async def search(ctx: RunContext, query: str) -> str:
            ...
        
        @agent.tool("custom_name")  # With custom name
        async def search_func(ctx: RunContext, query: str) -> str:
            ...
            
        @agent.tool("custom_name", "Custom description")  # With name and description
        async def search_func(ctx: RunContext, query: str) -> str:
            ...
            
        @agent.tool(name="custom_name", description="Custom description")  # With keywords
        async def search_func(ctx: RunContext, query: str) -> str:
            ...
        """
        
        def decorator(func: Callable) -> Callable:
            # Determine tool name - keyword takes precedence
            tool_name = name
            tool_desc = description
            
            if not tool_name:
                if isinstance(name_or_func, str):
                    tool_name = name_or_func
                elif name_or_func is not None and not callable(name_or_func):
                    tool_name = str(name_or_func)
            
            tool_instance = FunctionTool(func, tool_name, tool_desc)
            
            self.tool_registry.register(tool_instance)
            self._tools.append(tool_instance)
            
            logger.debug(f"Registered tool '{tool_instance.name}' with context pattern: {tool_instance.context_injection['pattern']}")
            
            return tool_instance
        
        # Handle direct decoration (@agent.tool)
        if callable(name_or_func):
            return decorator(name_or_func)
        else:
            return decorator
    
    async def run(
        self, 
        user_prompt: str, 
        *,
        deps: Optional[Deps] = None,
        message_history: Optional[List[Message]] = None,
        thread_id: Optional[str] = None
    ) -> RunResult[Result]:
        """Run the agent with dependency injection."""
        
        context = RunContext(
            deps=deps,
            messages=message_history or [],
            thread_id=thread_id
        )
        
        if thread_id:
            if thread_id in self._thread_conversations:
                previous_messages = self._thread_conversations[thread_id]
                context.messages.extend(previous_messages)
                context.metadata['restored_messages'] = len(previous_messages)
            elif deps and hasattr(deps, 'context_service'):
                try:
                    thread_context = await deps.context_service.retrieve_context(
                        query="", context_id=thread_id
                    )
                    if thread_context and 'messages' in thread_context:
                        previous_messages = [
                            Message(role=MessageRole(msg['role']), content=msg['content'])
                            for msg in thread_context['messages']
                        ]
                        context.messages.extend(previous_messages)
                        context.metadata['restored_messages'] = len(previous_messages)
                except Exception as e:
                    logger.warning(f"Could not load thread context: {e}")
        
        system_msg = None
        if self.system_prompt:
            if callable(self.system_prompt):
                system_content = self.system_prompt(context)
            else:
                system_content = self.system_prompt
            
            system_msg = Message(role=MessageRole.SYSTEM, content=system_content)
            context.messages.append(system_msg)
        
        user_msg = Message(role=MessageRole.USER, content=user_prompt)
        context.messages.append(user_msg)
        
        for attempt in range(self.retries):
            context.retry = attempt
            try:
                result = await self._execute_with_context(context)
                
                if thread_id:
                    messages_to_store = context.messages[-self.max_messages_per_thread:]
                    self._thread_conversations[thread_id] = messages_to_store
                    if len(self._thread_conversations) > self.max_stored_threads:
                        oldest_thread = next(iter(self._thread_conversations))
                        del self._thread_conversations[oldest_thread]
                
                return RunResult(
                    data=result,
                    messages=context.messages,
                    all_messages=context.messages.copy()
                )
                
            except Exception as e:
                if attempt == self.retries - 1:
                    raise MiiflowLLMError(f"Agent failed after {self.retries} retries: {e}", ErrorType.MODEL_ERROR)
                continue
        
        raise MiiflowLLMError("Agent execution failed", ErrorType.MODEL_ERROR)
    
    async def _execute_with_context(self, context: RunContext[Deps]) -> Result:
        """Route to appropriate execution based on agent type."""
        if self.agent_type == AgentType.SINGLE_HOP:
            return await self._execute_single_hop(context)
        else:  # AgentType.REACT
            return await self._execute_react(context)
    
    async def _execute_single_hop(self, context: RunContext[Deps]) -> Result:
        """Execute simple single-hop request-response (like LLM adapter pattern)."""
        response = await self.client.achat(
            messages=context.messages,
            tools=self._tools if self._tools else None,
            temperature=self.temperature
        )
        
        context.messages.append(response.message)
        
        if response.message.tool_calls:
            await self._execute_tool_calls(response.message.tool_calls, context)
            final_response = await self.client.achat(
                messages=context.messages,
                tools=None, 
                temperature=self.temperature
            )
            context.messages.append(final_response.message)
            return final_response.message.content
        
        return response.message.content
    
    async def _execute_react(self, context: RunContext[Deps]) -> Result:
        """Execute ReAct agent with multi-hop reasoning and persistent memory."""
        reasoning_steps = []
        
        for iteration in range(self.max_iterations):
            reasoning_step = {
                "step": iteration + 1,
                "timestamp": time.time(),
                "context_summary": context.get_conversation_summary(),
                "tools_available": len(self._tools),
                "reasoning": None,
                "action": None,
                "result": None
            }
            
            enhanced_messages = context.messages.copy()
            
            if iteration > 0:
                reasoning_summary = self._build_reasoning_summary(reasoning_steps)
                enhanced_messages.append(Message(
                    role=MessageRole.SYSTEM,
                    content=f"Previous reasoning steps: {reasoning_summary}"
                ))
            
            response = await self.client.achat(
                messages=enhanced_messages,
                tools=self._tools if self._tools else None,
                temperature=self.temperature
            )
            
            reasoning_step["reasoning"] = f"LLM response in step {iteration + 1}"
            context.messages.append(response.message)
            
            if response.message.tool_calls:
                reasoning_step["action"] = "tool_execution" 
                reasoning_step["tools_called"] = [
                    tc.function.name if hasattr(tc, 'function') else tc.get("function", {}).get("name") 
                    for tc in response.message.tool_calls
                ]
                
                await self._execute_tool_calls(response.message.tool_calls, context)
                reasoning_step["result"] = "tools_executed"
                reasoning_steps.append(reasoning_step)
                
                if iteration > 2:
                    recent_steps = reasoning_steps[-3:]
                    if all(step.get("action") == "tool_execution" for step in recent_steps):
                        logger.debug(f"Detected potential tool loop at iteration {iteration}")
                        context.messages.append(Message(
                            role=MessageRole.SYSTEM,
                            content="You have executed several tools successfully. Please provide your final answer based on the tool results. Do not call any more tools."
                        ))
                
                if iteration >= self.max_iterations - 2:
                    logger.debug(f"Near max iterations ({iteration}/{self.max_iterations}), forcing termination")
                    context.messages.append(Message(
                        role=MessageRole.SYSTEM,
                        content="This is your final opportunity to respond. Provide your best answer based on available information. Do not call any tools."
                    ))
                
                continue
            
            reasoning_step["action"] = "final_response"
            reasoning_step["result"] = response.message.content
            reasoning_steps.append(reasoning_step)
            
            if context.thread_id and hasattr(context.deps, 'context_service'):
                await self._save_conversation_context(context, reasoning_steps)
            
            if self.result_type == str:
                return response.message.content
            else:
                return response.message.content
        
        raise MiiflowLLMError(f"Agent exceeded maximum reasoning steps ({self.max_iterations})", ErrorType.MODEL_ERROR)
    
    def _build_reasoning_summary(self, reasoning_steps: List[Dict]) -> str:
        """Build a summary of previous reasoning steps."""
        if not reasoning_steps:
            return "No previous steps"
        
        summary_parts = []
        for step in reasoning_steps[-3:]:
            step_summary = f"Step {step['step']}: {step['action']}"
            if step.get('tools_called'):
                step_summary += f" (used tools: {', '.join(step['tools_called'])})"
            summary_parts.append(step_summary)
        
        return " → ".join(summary_parts)
    
    async def _save_conversation_context(self, context: RunContext[Deps], reasoning_steps: List[Dict]):
        """Save conversation context for persistent memory."""
        try:
            conversation_data = {
                "messages": [
                    {"role": msg.role.value, "content": msg.content} 
                    for msg in context.messages
                ],
                "reasoning_chain": reasoning_steps,
                "metadata": context.metadata,
                "timestamp": time.time()
            }
            
            if hasattr(context.deps, 'context_service'):
                await context.deps.context_service.store_context(context.thread_id, conversation_data)
        
        except Exception as e:
            logger.warning(f"Could not save conversation context: {e}")
    
    async def _execute_tool_calls(
        self, 
        tool_calls: List[Dict[str, Any]], 
        context: RunContext[Deps]
    ) -> None:
        """Execute tool calls with dependency injection."""
        logger.debug(f"About to execute {len(tool_calls)} tool calls")
        
        for i, tool_call in enumerate(tool_calls):
            logger.debug(f"Executing tool call {i+1}/{len(tool_calls)}")
            
            if hasattr(tool_call, 'function'):
                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments
                if isinstance(tool_args, str) and tool_args.strip():
                    import json
                    tool_args = json.loads(tool_args)
                elif not tool_args or (isinstance(tool_args, str) and not tool_args.strip()):
                    tool_args = {}
            else:
                tool_name = tool_call.get("function", {}).get("name")
                tool_args = tool_call.get("function", {}).get("arguments", {})
                if isinstance(tool_args, str) and tool_args.strip():
                    import json
                    tool_args = json.loads(tool_args)
                elif not tool_args or (isinstance(tool_args, str) and not tool_args.strip()):
                    tool_args = {}
            
            if tool_args is None:
                tool_args = {}
            elif not isinstance(tool_args, dict):
                logger.warning(f"Invalid tool_args type: {type(tool_args)}, converting to empty dict")
                tool_args = {}
            
            logger.debug(f"Tool '{tool_name}' with args: {tool_args}")
            
            tool = self.tool_registry.tools.get(tool_name)
            if tool and hasattr(tool, 'context_injection'):
                injection_pattern = tool.context_injection
                
                if injection_pattern['pattern'] == 'first_param':
                    logger.debug(f"Using Pydantic AI context injection for {tool_name}")
                    observation = await self.tool_registry.execute_safe_with_context(
                        tool_name, context, **tool_args
                    )
                else:
                    logger.debug(f"Plain function execution for {tool_name}")
                    observation = await self.tool_registry.execute_safe(tool_name, **tool_args)
            else:
                logger.debug(f"Plain function execution (no pattern detection) for {tool_name}")
                observation = await self.tool_registry.execute_safe(tool_name, **tool_args)
            
            logger.debug(f"Tool '{tool_name}' execution result: success={observation.success}, output='{observation.output}'")
            
            context.messages.append(Message(
                role=MessageRole.TOOL,
                content=str(observation.output) if observation.success else observation.error,
                tool_call_id=tool_call.id if hasattr(tool_call, 'id') else tool_call.get("id")
            ))
    
    def _create_react_loop(
        self,
        max_steps: int = 10,
        max_budget: Optional[float] = None,
        max_time_seconds: Optional[float] = None,
        safety_profile: str = "balanced"
    ):
        """Create a configured ReAct loop."""
        from .react import ReActLoop, SafetyProfiles
        
        # Select safety profile
        if safety_profile == "conservative":
            safety_manager = SafetyProfiles.conservative()
        elif safety_profile == "balanced":
            safety_manager = SafetyProfiles.balanced()
        elif safety_profile == "permissive":
            safety_manager = SafetyProfiles.permissive()
        else:
            # Custom safety manager with provided parameters
            from .react.safety import SafetyManager
            safety_manager = SafetyManager(
                max_steps=max_steps,
                max_budget=max_budget,
                max_time_seconds=max_time_seconds
            )
        
        return ReActLoop(
            agent=self,
            safety_manager=safety_manager
        )
    
    async def run_react(
        self,
        query: str,
        context: RunContext,
        max_steps: int = 10,
        max_budget: Optional[float] = None,
        max_time_seconds: Optional[float] = None,
        safety_profile: str = "balanced"
    ):
        """Run agent in ReAct mode with structured reasoning.
        """
        react_loop = self._create_react_loop(max_steps, max_budget, max_time_seconds, safety_profile)
        return await react_loop.execute(query, context, stream_events=False)
    
    async def stream_react(
        self,
        query: str,
        context: RunContext,
        max_steps: int = 10,
        max_budget: Optional[float] = None,
        max_time_seconds: Optional[float] = None,
        safety_profile: str = "balanced"
    ):
        """Run agent in ReAct mode with streaming events.
            ReActEvent objects for each reasoning step
        """
        react_loop = self._create_react_loop(max_steps, max_budget, max_time_seconds, safety_profile)
        async for event in react_loop.stream_execute(query, context):
            yield event
    
    async def stream_single_hop(
        self,
        user_prompt: str,
        *,
        deps: Optional[Deps] = None,
        message_history: Optional[List[Message]] = None,
        thread_id: Optional[str] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream single-hop execution with real-time events."""
        
        context = RunContext(
            deps=deps,
            messages=message_history or [],
            thread_id=thread_id
        )
        
        if thread_id:
            if thread_id in self._thread_conversations:
                previous_messages = self._thread_conversations[thread_id]
                context.messages.extend(previous_messages)
                context.metadata['restored_messages'] = len(previous_messages)
            elif deps and hasattr(deps, 'context_service'):
                try:
                    thread_context = await deps.context_service.retrieve_context(
                        query="", context_id=thread_id
                    )
                    if thread_context and 'messages' in thread_context:
                        previous_messages = [
                            Message(role=MessageRole(msg['role']), content=msg['content'])
                            for msg in thread_context['messages']
                        ]
                        context.messages.extend(previous_messages)
                        context.metadata['restored_messages'] = len(previous_messages)
                except Exception as e:
                    logger.warning(f"Could not load thread context: {e}")
        
        if self.system_prompt:
            if callable(self.system_prompt):
                system_content = self.system_prompt(context)
            else:
                system_content = self.system_prompt
            
            system_msg = Message(role=MessageRole.SYSTEM, content=system_content)
            context.messages.append(system_msg)
        
        user_msg = Message(role=MessageRole.USER, content=user_prompt)
        context.messages.append(user_msg)
        
        yield {
            "event": "execution_start",
            "data": {
                "prompt": user_prompt,
                "context_length": len(context.messages),
                "tools_available": len(self._tools)
            }
        }
        
        try:
            yield {"event": "llm_start", "data": {}}
            
            buffer = ""
            final_tool_calls = None
            
            async for chunk in self.client.astream_chat(
                messages=context.messages,
                tools=self._tools if self._tools else None,
                temperature=self.temperature
            ):
                if chunk.delta:
                    buffer += chunk.delta
                    yield {
                        "event": "llm_chunk",
                        "data": {"delta": chunk.delta, "content": buffer}
                    }
                
                if chunk.tool_calls and chunk.finish_reason:
                    final_tool_calls = chunk.tool_calls
                
                if chunk.finish_reason:
                    break
            
            response_message = Message(
                role=MessageRole.ASSISTANT,
                content=buffer,
                tool_calls=final_tool_calls
            )
            context.messages.append(response_message)
            
            if final_tool_calls:
                yield {
                    "event": "tools_start",
                    "data": {"tool_count": len(final_tool_calls)}
                }
                
                await self._execute_tool_calls(final_tool_calls, context)
                
                yield {"event": "tools_complete", "data": {}}
                
                final_response = await self.client.achat(
                    messages=context.messages,
                    tools=None,
                    temperature=self.temperature
                )
                context.messages.append(final_response.message)
                result = final_response.message.content
            else:
                result = buffer
            
            if thread_id:
                messages_to_store = context.messages[-self.max_messages_per_thread:]
                self._thread_conversations[thread_id] = messages_to_store
                if len(self._thread_conversations) > self.max_stored_threads:
                    oldest_thread = next(iter(self._thread_conversations))
                    del self._thread_conversations[oldest_thread]
            
            yield {
                "event": "execution_complete",
                "data": {"result": result}
            }
            
        except Exception as e:
            yield {
                "event": "error",
                "data": {"error": str(e), "error_type": type(e).__name__}
            }
            raise


class ExampleDeps:
    """Example dependency container showing the abstracted approach.
    
    Applications should create their own dependency containers that implement
    the service protocols. The agent doesn't need to know implementation details.
    
    Example implementations:
    
    # Database-based implementation (miiflow-web style)
    class MiiflowWebDeps:
        def __init__(self, db_pool, vector_client, redis_client):
            self.context_service = DatabaseContextService(db_pool)  # DB-based
            self.search_service = VectorSearchService(vector_client)  # Vector DB
            self.database = PostgreSQLService(db_pool)  # Direct DB access
            
    # Simple file-based implementation  
    class SimpleAppDeps:
        def __init__(self):
            self.context_service = FileContextService("./contexts/")  # File-based  
            self.search_service = InMemorySearchService()  # In-memory
            
    # Hybrid implementation
    class HybridDeps:
        def __init__(self):
            self.context_service = RedisContextService()  # Redis for speed
            self.search_service = ElasticsearchService()  # Full-text search
            self.database = RESTAPIService("api.example.com")  # External API
            
    The agent works with any of these through the protocol interfaces.
    """
    pass
