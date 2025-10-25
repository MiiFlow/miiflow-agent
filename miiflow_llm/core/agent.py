"""Unified Agent architecture focused on LLM reasoning (stateless)."""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, Generic, List, Optional, Type, TypeVar, Union

logger = logging.getLogger(__name__)

from .client import LLMClient
from .exceptions import ErrorType, MiiflowLLMError
from .message import Message, MessageRole
from .tools import FunctionTool, ToolRegistry

Deps = TypeVar("Deps")
Result = TypeVar("Result")


class AgentType(Enum):
    SINGLE_HOP = "single_hop"  # Simple, direct response
    REACT = "react"  # ReAct with multi-hop reasoning


@dataclass
class RunResult(Generic[Result]):
    data: Result
    messages: List[Message]
    all_messages: List[Message] = field(default_factory=list)

    def __post_init__(self):
        if not self.all_messages:
            self.all_messages = self.messages


@dataclass
class RunContext(Generic[Deps]):
    """Context passed to tools and agent functions (stateless)."""

    deps: Deps
    messages: List[Message] = field(default_factory=list)
    retry: int = 0
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
        """Get a summary of the current conversation context."""
        if len(self.messages) <= 2:
            return "New conversation"

        user_messages = [msg.content for msg in self.messages if msg.role == MessageRole.USER]
        return f"Conversation with {len(user_messages)} user messages"


class Agent(Generic[Deps, Result]):
    """Unified Agent focused on LLM reasoning (stateless)."""

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
        use_native_tool_calling: bool = True,
        json_schema: Optional[Dict[str, Any]] = None,
    ):
        self.client = client
        self.agent_type = agent_type
        self.deps_type = deps_type
        self.result_type = result_type or str
        self.system_prompt = system_prompt
        self.retries = retries
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.use_native_tool_calling = use_native_tool_calling
        self.json_schema = json_schema

        # Share the tool registry with LLMClient for consistency
        self.tool_registry = self.client.tool_registry
        self._tools: List[FunctionTool] = []

        # Register provided tools
        if tools:
            for tool in tools:
                self.tool_registry.register(tool)
                self._tools.append(tool)

    def add_tool(self, func: Callable) -> None:
        """Add a tool function (decorated with global @tool) to this agent.

        Usage:
        from miiflow_llm.core.tools import tool

        @tool("search", "Search the web")
        def search_web(query: str) -> str:
            return search_results

        agent.add_tool(search_web)
        """
        from .tools.decorators import get_tool_from_function

        tool_instance = get_tool_from_function(func)
        if not tool_instance:
            raise ValueError(f"Function {func.__name__} is not decorated with @tool")

        self.tool_registry.register(tool_instance)
        self._tools.append(tool_instance)

        logger.debug(f"Added tool '{tool_instance.name}' to agent")

    async def run(
        self,
        user_prompt: str,
        *,
        deps: Optional[Deps] = None,
        message_history: Optional[List[Message]] = None,
    ) -> RunResult[Result]:
        """Run the agent with dependency injection (stateless)."""

        context = RunContext(deps=deps, messages=message_history or [])

        # Add system prompt if provided
        if self.system_prompt:
            if callable(self.system_prompt):
                system_content = self.system_prompt(context)
            else:
                system_content = self.system_prompt

            system_msg = Message(role=MessageRole.SYSTEM, content=system_content)
            context.messages.append(system_msg)

        # Add user message
        user_msg = Message(role=MessageRole.USER, content=user_prompt)
        context.messages.append(user_msg)

        # Execute with retries
        for attempt in range(self.retries):
            context.retry = attempt
            try:
                result = await self._execute_with_context(context)

                return RunResult(
                    data=result, messages=context.messages, all_messages=context.messages.copy()
                )

            except Exception as e:
                if attempt == self.retries - 1:
                    raise MiiflowLLMError(
                        f"Agent failed after {self.retries} retries: {e}", ErrorType.MODEL_ERROR
                    )
                continue

        raise MiiflowLLMError("Agent execution failed", ErrorType.MODEL_ERROR)

    async def _execute_with_context(self, context: RunContext[Deps]) -> Result:
        """Route to appropriate execution based on agent type."""
        if self.agent_type == AgentType.SINGLE_HOP:
            # Extract user prompt from context messages
            user_prompt = ""
            for msg in reversed(context.messages):
                if msg.role == MessageRole.USER:
                    user_prompt = msg.content
                    break

            final_answer = None
            async for event in self.stream_single_hop(user_prompt, context=context):
                if isinstance(event, dict) and event.get("event") == "execution_complete":
                    final_answer = event.get("data", {}).get("result", "")
                    break

            if self.result_type == str:
                return final_answer or "No final answer received"
            else:
                return final_answer or "No final answer received"
        else:  # AgentType.REACT
            # Extract user prompt from context messages
            user_prompt = ""
            for msg in reversed(context.messages):
                if msg.role == MessageRole.USER:
                    user_prompt = msg.content
                    break

            final_answer = None
            async for event in self.stream_react(
                user_prompt, context, max_steps=self.max_iterations
            ):
                if event.event_type.value == "final_answer":
                    final_answer = event.data.get("answer", "")
                    break

            if self.result_type == str:
                return final_answer or "No final answer received"
            else:
                return final_answer or "No final answer received"

    async def _execute_tool_calls(
        self, tool_calls: List[Dict[str, Any]], context: RunContext[Deps]
    ) -> None:
        """Execute tool calls with dependency injection."""
        logger.debug(f"About to execute {len(tool_calls)} tool calls")

        for i, tool_call in enumerate(tool_calls):
            logger.debug(f"Executing tool call {i+1}/{len(tool_calls)}")

            # Extract tool name and arguments
            if hasattr(tool_call, "function"):
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
                logger.warning(
                    f"Invalid tool_args type: {type(tool_args)}, converting to empty dict"
                )
                tool_args = {}

            logger.debug(f"Tool '{tool_name}' with args: {tool_args}")

            # Execute tool with context injection if needed
            tool = self.tool_registry.tools.get(tool_name)
            if tool and hasattr(tool, "context_injection"):
                injection_pattern = tool.context_injection

                if injection_pattern["pattern"] == "first_param":
                    logger.debug(f"Using context injection for {tool_name}")
                    observation = await self.tool_registry.execute_safe_with_context(
                        tool_name, context, **tool_args
                    )
                else:
                    logger.debug(f"Plain function execution for {tool_name}")
                    observation = await self.tool_registry.execute_safe(tool_name, **tool_args)
            else:
                logger.debug(f"Plain function execution (no pattern detection) for {tool_name}")
                observation = await self.tool_registry.execute_safe(tool_name, **tool_args)

            logger.debug(
                f"Tool '{tool_name}' execution result: success={observation.success}, output='{observation.output}'"
            )

            # Add tool result message
            context.messages.append(
                Message(
                    role=MessageRole.TOOL,
                    content=str(observation.output) if observation.success else observation.error,
                    tool_call_id=tool_call.id if hasattr(tool_call, "id") else tool_call.get("id"),
                )
            )

    async def stream_react(
        self,
        query: str,
        context: RunContext,
        max_steps: int = 10,
        max_budget: Optional[float] = None,
        max_time_seconds: Optional[float] = None,
    ):
        """Run agent in ReAct mode with streaming events."""
        from .react import ReActFactory

        orchestrator = ReActFactory.create_orchestrator(
            agent=self,
            max_steps=max_steps,
            max_budget=max_budget,
            max_time_seconds=max_time_seconds,
            use_native_tools=self.use_native_tool_calling,
        )

        # Real-time streaming setup
        event_queue = asyncio.Queue()

        def real_time_stream(event):
            """Stream events immediately as they're published."""
            try:
                event_queue.put_nowait(event)
            except asyncio.QueueFull:
                import logging

                logging.getLogger(__name__).warning("Event queue full, dropping event")

        orchestrator.event_bus.subscribe(real_time_stream)
        execution_task = asyncio.create_task(orchestrator.execute(query, context))

        try:
            while not execution_task.done():
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=0.1)
                    yield event
                    event_queue.task_done()
                except asyncio.TimeoutError:
                    continue

            while not event_queue.empty():
                try:
                    event = event_queue.get_nowait()
                    yield event
                    event_queue.task_done()
                except asyncio.QueueEmpty:
                    break
            await execution_task

        finally:
            orchestrator.event_bus.unsubscribe(real_time_stream)

    async def stream_single_hop(
        self, user_prompt: str, *, context: RunContext[Deps]
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream single-hop execution - uses context from run() (no duplication)."""

        # Context is provided by run() - no setup duplication!

        yield {
            "event": "execution_start",
            "data": {
                "prompt": user_prompt,
                "context_length": len(context.messages),
                "tools_available": len(self._tools),
            },
        }

        try:
            yield {"event": "llm_start", "data": {}}

            buffer = ""
            final_tool_calls = None
            has_tool_calls = False

            # Stream LLM response
            async for chunk in self.client.astream_chat(
                messages=context.messages,
                tools=self._tools if self._tools else None,
                temperature=self.temperature,
                json_schema=self.json_schema,
            ):
                if chunk.delta:
                    buffer += chunk.delta
                    yield {"event": "llm_chunk", "data": {"delta": chunk.delta, "content": buffer}}

                # Check if we have tool calls
                if chunk.tool_calls:
                    has_tool_calls = True

                if chunk.finish_reason:
                    break

            # If we had tool calls, get them properly by making a non-streaming call
            if has_tool_calls:
                response = await self.client.achat(
                    messages=context.messages,
                    tools=self._tools if self._tools else None,
                    temperature=self.temperature,
                    json_schema=self.json_schema,
                )
                final_tool_calls = response.message.tool_calls

            response_message = Message(
                role=MessageRole.ASSISTANT, content=buffer, tool_calls=final_tool_calls
            )
            context.messages.append(response_message)

            # Handle tool calls if present
            if final_tool_calls:
                yield {"event": "tools_start", "data": {"tool_count": len(final_tool_calls)}}

                await self._execute_tool_calls(final_tool_calls, context)

                yield {"event": "tools_complete", "data": {}}
                final_response = await self.client.achat(
                    messages=context.messages,
                    tools=None,
                    temperature=self.temperature,
                    json_schema=self.json_schema,
                )
                context.messages.append(final_response.message)
                result = final_response.message.content
            else:
                result = buffer

            yield {"event": "execution_complete", "data": {"result": result}}

        except Exception as e:
            yield {"event": "error", "data": {"error": str(e), "error_type": type(e).__name__}}
            raise
