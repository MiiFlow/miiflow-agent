"""Unified Agent architecture focused on LLM reasoning (stateless)."""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, Generic, List, Optional, TypeVar, Union

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
    PLAN_AND_EXECUTE = "plan_and_execute"  # Plan then execute for complex multi-step tasks


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
        system_prompt: Optional[Union[str, Callable[[RunContext[Deps]], str]]] = None,
        retries: int = 1,
        max_iterations: int = 10,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[FunctionTool]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
    ):
        self.client = client
        self.agent_type = agent_type
        self.system_prompt = system_prompt
        self.retries = retries
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
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

        # Add system prompt if provided - INSERT at the beginning, not append
        if self.system_prompt:
            if callable(self.system_prompt):
                system_content = self.system_prompt(context)
            else:
                system_content = self.system_prompt

            system_msg = Message(role=MessageRole.SYSTEM, content=system_content)
            context.messages.insert(0, system_msg)

        # Add user message only if provided and not empty
        if user_prompt and user_prompt.strip():
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

    async def _execute_with_context(self, context: RunContext[Deps]) -> str:
        """Route to appropriate execution based on agent type."""
        # Extract user prompt from context messages
        user_prompt = ""
        for msg in reversed(context.messages):
            if msg.role == MessageRole.USER:
                user_prompt = msg.content
                break

        final_answer = None

        if self.agent_type == AgentType.SINGLE_HOP:
            async for event in self._stream_single_hop(user_prompt, context=context):
                if isinstance(event, dict) and event.get("event") == "execution_complete":
                    final_answer = event.get("data", {}).get("result", "")
                    break

        elif self.agent_type == AgentType.REACT:
            async for event in self._stream_react(
                user_prompt, context, max_steps=self.max_iterations
            ):
                if event.event_type.value == "final_answer":
                    final_answer = event.data.get("answer", "")
                    break

        elif self.agent_type == AgentType.PLAN_AND_EXECUTE:
            async for event in self._stream_plan_execute(
                user_prompt, context, max_replans=self.max_iterations // 5
            ):
                if hasattr(event, "event_type") and event.event_type.value == "final_answer":
                    final_answer = event.data.get("answer", "")
                    break

        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")

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

    async def stream(
        self,
        query: str,
        context: RunContext,
        *,
        agent_type: Optional[AgentType] = None,
        max_steps: int = 10,
        max_budget: Optional[float] = None,
        max_time_seconds: Optional[float] = None,
        max_replans: int = 2,
        existing_plan=None,
    ) -> AsyncIterator[Any]:
        """Unified streaming method that dispatches based on agent_type.

        Args:
            query: User's query/goal
            context: Run context with messages and deps
            agent_type: Optional override for agent type. Defaults to self.agent_type
            max_steps: Maximum ReAct steps (for REACT type)
            max_budget: Optional budget limit (for REACT type)
            max_time_seconds: Optional time limit (for REACT type)
            max_replans: Maximum replanning attempts (for PLAN_AND_EXECUTE type)
            existing_plan: Optional pre-generated plan (for PLAN_AND_EXECUTE type)

        Yields:
            Streaming events specific to the agent type
        """
        effective_type = agent_type or self.agent_type

        if effective_type == AgentType.SINGLE_HOP:
            async for event in self._stream_single_hop(query, context=context):
                yield event
        elif effective_type == AgentType.REACT:
            async for event in self._stream_react(
                query, context, max_steps, max_budget, max_time_seconds
            ):
                yield event
        elif effective_type == AgentType.PLAN_AND_EXECUTE:
            async for event in self._stream_plan_execute(
                query, context, max_replans, existing_plan
            ):
                yield event
        else:
            raise ValueError(f"Unknown agent type: {effective_type}")

    async def _stream_react(
        self,
        query: str,
        context: RunContext,
        max_steps: int = 10,
        max_budget: Optional[float] = None,
        max_time_seconds: Optional[float] = None,
    ):
        """Internal: Run agent in ReAct mode with streaming events."""
        from .react import ReActFactory

        orchestrator = ReActFactory.create_orchestrator(
            agent=self,
            max_steps=max_steps,
            max_budget=max_budget,
            max_time_seconds=max_time_seconds,
            use_native_tools=True,
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

    async def _stream_plan_execute(
        self,
        query: str,
        context: RunContext,
        max_replans: int = 2,
        existing_plan=None,
    ):
        """Internal: Run agent in Plan and Execute mode with streaming events.

        Args:
            query: User's query/goal
            context: Run context with messages and deps
            max_replans: Maximum number of replanning attempts
            existing_plan: Optional pre-generated Plan object from combined routing step.
                          If provided, skips initial plan generation (saves ~2-5s)
        """
        from .react import ReActFactory
        from .react.events import EventBus
        from .react.plan_execute_orchestrator import PlanAndExecuteOrchestrator
        from .react.safety import SafetyManager
        from .react.tool_executor import AgentToolExecutor

        # Create dependencies
        tool_executor = AgentToolExecutor(self)
        event_bus = EventBus()
        safety_manager = SafetyManager(max_steps=999)  # High limit for Plan & Execute

        # Create ReAct orchestrator for subtask execution (composition pattern)
        react_orchestrator = ReActFactory.create_orchestrator(
            agent=self,
            max_steps=10,  # Each subtask gets up to 10 ReAct steps
            use_native_tools=True,
        )

        # Create Plan and Execute orchestrator
        orchestrator = PlanAndExecuteOrchestrator(
            tool_executor=tool_executor,
            event_bus=event_bus,
            safety_manager=safety_manager,
            subtask_orchestrator=react_orchestrator,
            max_replans=max_replans,
            use_react_for_subtasks=True,  # Use ReAct for complex subtasks
        )

        # Real-time streaming setup
        event_queue = asyncio.Queue()

        def real_time_stream(event):
            """Stream events immediately as they're published."""
            try:
                event_queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("Event queue full, dropping event")

        event_bus.subscribe(real_time_stream)
        # NEW: Pass existing_plan if provided (saves ~2-5s by skipping plan generation)
        execution_task = asyncio.create_task(
            orchestrator.execute(query, context, existing_plan=existing_plan)
        )

        try:
            while not execution_task.done():
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=0.1)
                    yield event
                    event_queue.task_done()
                except asyncio.TimeoutError:
                    continue

            # Drain remaining events
            while not event_queue.empty():
                try:
                    event = event_queue.get_nowait()
                    yield event
                    event_queue.task_done()
                except asyncio.QueueEmpty:
                    break

            await execution_task

        finally:
            event_bus.unsubscribe(real_time_stream)

    def _prepare_messages_for_llm(self, messages: List[Message]) -> List[Message]:
        """Prepare messages for LLM by filtering out mid-conversation SYSTEM messages.

        Claude requires alternating USER/ASSISTANT messages. SYSTEM messages in the middle
        of the conversation (e.g., tool execution context) break this pattern and cause errors.

        This function keeps only the first SYSTEM message (assistant instructions) and filters
        out all subsequent SYSTEM messages.
        """
        if not messages:
            return messages

        result = []
        first_system_seen = False

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                if not first_system_seen:
                    # Keep the first SYSTEM message (assistant instructions)
                    result.append(msg)
                    first_system_seen = True
                # Skip all other SYSTEM messages (tool execution context, etc.)
            else:
                # Keep all USER and ASSISTANT messages
                result.append(msg)

        return result

    async def _stream_single_hop(
        self, user_prompt: str, *, context: RunContext[Deps]
    ) -> AsyncIterator[Dict[str, Any]]:
        """Internal: Stream single-hop execution - uses context from run() (no duplication)."""

        # Add user message to context if not already present
        # This handles cases where stream_single_hop is called directly (not from run())
        if not context.messages or context.messages[-1].content != user_prompt:
            user_msg = Message(role=MessageRole.USER, content=user_prompt)
            context.messages.append(user_msg)

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

            # Prepare messages for LLM (filter out mid-conversation SYSTEM messages)
            llm_messages = self._prepare_messages_for_llm(context.messages)

            # Stream LLM response
            async for chunk in self.client.astream_chat(
                messages=llm_messages,
                tools=self._tools if self._tools else None,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
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
                    messages=llm_messages,
                    tools=self._tools if self._tools else None,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
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
                # Re-filter messages after tool execution (tool results added to context.messages)
                llm_messages = self._prepare_messages_for_llm(context.messages)
                final_response = await self.client.achat(
                    messages=llm_messages,
                    tools=None,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
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
