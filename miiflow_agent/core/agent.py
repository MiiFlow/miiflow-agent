"""Unified Agent architecture focused on LLM reasoning (stateless)."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    TypeVar,
    Union,
)

if TYPE_CHECKING:
    from .config import AgentConfig
    from .subagent import SubAgent
    from .tools.mcp import NativeMCPServerConfig
    from .react.dispatch import DispatchCounter
    from .react.events.bus import EventBus

logger = logging.getLogger(__name__)

from .client import LLMClient
from .exceptions import ErrorType, MiiflowLLMError
from .message import Message, MessageRole
from .tools import FunctionTool, ToolRegistry


def _content_text(content: Any) -> str:
    """Return the textual representation of message content.

    Handles plain-string content as well as multimodal block lists (e.g.
    ``[TextBlock, ImageBlock, TextBlock]``). Used to dedupe a query against an
    existing user message in chat history regardless of content shape — a
    naive ``content == query`` check fails for multimodal because list != str.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            text = getattr(block, "text", None)
            if text is None and isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
            if text:
                parts.append(text)
        return " ".join(parts)
    return ""

Deps = TypeVar("Deps")
Result = TypeVar("Result")


class _NullCM:
    """No-op context manager used when an outer ToolSearch session is active."""

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class AgentType(Enum):
    """Agent execution path.

    Post unified-ReAct migration there are only two real paths:

    - ``SINGLE_HOP``: one LLM call (plus an optional finalization call
      after tools fire once). Used for chat-only assistants with zero
      tools or when ``json_schema`` structured output is required.
    - ``REACT``: the unified ReAct loop. Planning and multi-agent
      fan-out are emergent — the model self-escalates via
      ``enter_plan_mode`` / ``exit_plan_mode`` and parallel
      ``dispatch_assistant`` tool calls. The legacy
      ``PLAN_AND_EXECUTE`` / ``PARALLEL_PLAN`` / ``MULTI_AGENT`` types
      were removed in the migration; if you want their behavior, you
      want REACT.
    """

    SINGLE_HOP = "single_hop"
    REACT = "react"


@dataclass
class RunResult(Generic[Result]):
    data: Result
    messages: List[Message]
    all_messages: List[Message] = field(default_factory=list)

    def __post_init__(self):
        if not self.all_messages:
            self.all_messages = self.messages


@dataclass
class RunState:
    """Per-run state owned and provisioned by the orchestrator.

    Tools read these via ``ctx.run_state.*``; the orchestrator (and only the
    orchestrator) writes them. Distinct from ``ctx.deps``, which is the
    adapter-supplied bag of caller context (org id, thread id, user id, …).

    Splitting these surfaces gives every consumer an IDE-checkable contract
    for what the framework provisions per-run and prevents adapters from
    accidentally clobbering framework state with a string-keyed write.

    Note: this is the migration target for the legacy ``ctx.deps["event_bus"]``
    /``ctx.deps["step_number"]`` /``ctx.deps["dispatch_counter"]`` /
    ``ctx.deps["media_store"]`` magic-key pattern. During the transition the
    orchestrator dual-writes both surfaces — see
    ``react/orchestrator.py:execute`` and ``_execute_tool``. Readers should
    prefer ``ctx.run_state``; the ``ctx.deps`` entries will be removed once
    every caller is migrated.
    """

    # The orchestrator's event bus. Tools that need to publish back to the
    # parent stream (notably ``dispatch_assistant`` forwarding subagent
    # lifecycle events) read this.
    event_bus: Optional["EventBus"] = None

    # The current ReAct step number. Tools that emit events tied to the step
    # (dispatch lifecycle, observability hooks) read this.
    step_number: int = 0

    # Per-turn dispatch budget counter. Shared across concurrent dispatches in
    # the same parent turn so per-handle and per-turn caps are accurate under
    # parallel batches.
    dispatch_counter: Optional["DispatchCounter"] = None

    # Per-step media reference store. Tools that resolve ``media_ref:<id>``
    # inputs to URLs (image editors, creative analyzers) read this.
    media_store: Dict[str, str] = field(default_factory=dict)

    # Current permission mode for this run. ``"default"`` lets every tool
    # execute. ``"plan"`` short-circuits non-read-only tools with a
    # synthetic "blocked — plan mode active" tool result so the model has
    # to call ``exit_plan_mode`` before resuming side-effectful work.
    # ``"bypass"`` disables the plan-mode gate entirely (used by the
    # Django adapter when a user has already approved a high-trust run).
    # Mutated by the ``enter_plan_mode`` / ``exit_plan_mode`` deferred
    # tools; read by ``AgentToolExecutor.execute_tool``.
    permission_mode: str = "default"

    # File paths the model has read this run, mapped to the file mtime at
    # read time. The opt-in ``file_edit`` / ``file_write`` tools in
    # ``core/tools/coding`` enforce read-before-edit + on-disk-mtime-match
    # by consulting this map. Empty when the coding kit isn't registered.
    read_files: Dict[str, float] = field(default_factory=dict)


@dataclass
class RunContext(Generic[Deps]):
    """Context passed to tools and agent functions (stateless)."""

    deps: Deps
    messages: List[Message] = field(default_factory=list)
    retry: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    cancel_event: Optional[asyncio.Event] = None
    run_state: RunState = field(default_factory=RunState)

    @property
    def is_cancelled(self) -> bool:
        return self.cancel_event is not None and self.cancel_event.is_set()

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
        client: Optional[LLMClient] = None,
        *,
        config: Optional["AgentConfig"] = None,
        agent_type: AgentType = AgentType.SINGLE_HOP,
        system_prompt: Optional[Union[str, Callable[[RunContext[Deps]], str]]] = None,
        retries: int = 1,
        max_iterations: int = 10,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[FunctionTool]] = None,
        sub_agents: Optional[List["SubAgent"]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        context_compression: bool = True,
        max_context_tokens: Optional[int] = None,
        enable_plan_mode: bool = False,
    ):
        # Two construction modes:
        #   1) ``Agent(config=AgentConfig(...))`` — canonical, used by Stage 2+
        #      callers that need ``sub_agents`` or want their config audited
        #      as a single dataclass.
        #   2) ``Agent(client, agent_type=..., tools=..., ...)`` — legacy
        #      kwargs form. We build an internal AgentConfig so the rest of
        #      ``__init__`` only consumes one shape.
        if config is not None:
            if client is not None and client is not config.client:
                raise ValueError(
                    "Pass `client` either positionally OR via config, not both."
                )
            if any(
                v is not None
                for v in (system_prompt, tools, sub_agents, json_schema, max_tokens, max_context_tokens)
            ) or enable_plan_mode:
                raise ValueError(
                    "When passing `config=`, supply all knobs via the "
                    "AgentConfig — keyword args alongside config are not "
                    "merged (would be ambiguous about which wins)."
                )
            effective = config
        else:
            if client is None:
                raise ValueError("Agent requires a `client` or a `config=AgentConfig(...)`.")
            from .config import AgentConfig as _AgentConfig

            effective = _AgentConfig(
                client=client,
                agent_type=agent_type,
                system_prompt=system_prompt,
                retries=retries,
                max_iterations=max_iterations,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=list(tools) if tools else [],
                sub_agents=list(sub_agents) if sub_agents else [],
                json_schema=json_schema,
                context_compression=context_compression,
                max_context_tokens=max_context_tokens,
                enable_plan_mode=enable_plan_mode,
            )

        self.config = effective
        self.client = effective.client
        self.agent_type = effective.agent_type
        self.system_prompt = effective.system_prompt
        self.retries = effective.retries
        self.max_iterations = effective.max_iterations
        self.temperature = effective.temperature
        self.max_tokens = effective.max_tokens
        self.json_schema = effective.json_schema
        self.enable_plan_mode = getattr(effective, "enable_plan_mode", False)

        # Context compression: enabled for the multi-step ReAct path
        # (SINGLE_HOP has at most two LLM calls and doesn't need it).
        self._context_compressor = None
        if effective.context_compression and effective.agent_type == AgentType.REACT:
            from .context_compression import ContextCompressor
            self._context_compressor = ContextCompressor(
                client=effective.client,
                max_context_tokens=effective.max_context_tokens,
            )

        # Share the tool registry with LLMClient for consistency
        self.tool_registry = self.client.tool_registry
        # Calibrate ToolSearch threshold to the bound provider's grammar
        # limit. Anthropic's tool-use compiler is the strict one and 400s
        # at ~13-15 mid-complexity tools; OpenAI / Gemini handle the same
        # load without issue. This used to require per-adapter tuning
        # (Django was overriding the threshold for every Adlyse-family
        # assistant); now the framework picks the right default based on
        # the provider attached to the client. Explicit overrides at
        # registry construction still win — see
        # ToolRegistry.calibrate_for_provider docstring.
        #
        # Note: self.client is the LLMClient wrapper; the underlying
        # provider client (Anthropic/OpenAI/Gemini) hangs off
        # self.client.client and is where provider_name lives.
        _provider_client = getattr(self.client, "client", None)
        self.tool_registry.calibrate_for_provider(
            getattr(_provider_client, "provider_name", None)
        )
        self._tools: List[FunctionTool] = []
        self._sub_agents: List["SubAgent"] = []

        # Register provided tools
        for tool in effective.tools:
            self.tool_registry.register(tool)
            self._tools.append(tool)

        # Register provided sub-agents. When at least one is present we
        # also synthesize a `dispatch_assistant`-shaped FunctionTool that
        # routes calls through the framework's dispatch lifecycle
        # (guardrails, counter, event bubbling). Stage 1's parallel tool
        # batch executor will then gather multiple dispatcher calls
        # automatically — that's how this Stage 2 design unlocks parallel
        # dispatch without orchestrator surgery.
        for sub in effective.sub_agents:
            self._sub_agents.append(sub)

        if self._sub_agents:
            from .react.dispatch import (
                DISPATCH_TOOL_NAME,
                make_subagent_dispatcher_tool,
            )

            existing_tool_names = {t.name for t in self._tools}
            if DISPATCH_TOOL_NAME not in existing_tool_names:
                # `parent_assistant_id` defaults to a synthetic id because
                # the framework Agent doesn't (and shouldn't) know about
                # Django Assistant rows. The Django adapter in Stage 3
                # passes the real Assistant.id via AgentConfig.
                parent_id = effective.parent_assistant_id or "framework_agent"
                dispatcher = make_subagent_dispatcher_tool(
                    self._sub_agents,
                    parent_assistant_id=parent_id,
                )
                self.tool_registry.register(dispatcher)
                self._tools.append(dispatcher)

        # Plan-mode wiring: register `enter_plan_mode` / `exit_plan_mode`
        # so the model can self-escalate into a plan-only gate. The pair
        # is registered AFTER sub-agents so a user-supplied tool of the
        # same name still wins; conflict-detection happens in
        # `register_plan_mode_tools` itself (idempotent + skip-on-exists).
        if self.enable_plan_mode:
            from .tools.plan_mode import register_plan_mode_tools

            registered = register_plan_mode_tools(self.tool_registry)
            existing_names = {t.name for t in self._tools}
            for plan_tool in registered:
                if plan_tool.name not in existing_names:
                    self._tools.append(plan_tool)

    def add_tool(self, func: Callable) -> None:
        """Add a tool function (decorated with global @tool) to this agent.

        Usage:
        from miiflow_agent.core.tools import tool

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

    def register_native_mcp_server(self, config: "NativeMCPServerConfig") -> None:
        """Register an MCP server for native provider-side execution.

        Native MCP servers are handled directly by the LLM provider (Anthropic, OpenAI)
        rather than requiring client-side connection management.

        Usage:
            from miiflow_agent.core.tools.mcp import NativeMCPServerConfig

            config = NativeMCPServerConfig(
                name="my-mcp-server",
                url="https://example.com/mcp",
                authorization_token="Bearer token123"
            )
            agent.register_native_mcp_server(config)

        Args:
            config: NativeMCPServerConfig with server URL and auth details
        """
        from .tools.mcp import NativeMCPServerConfig as ConfigType

        if not isinstance(config, ConfigType):
            raise TypeError(f"Expected NativeMCPServerConfig, got {type(config)}")

        self.tool_registry.register_native_mcp_server(config)
        logger.info(f"Registered native MCP server: {config.name} -> {config.url}")

    @property
    def sub_agents(self) -> List["SubAgent"]:
        """SubAgent instances this agent can dispatch to.

        Mutable through `add_sub_agent()`; mostly read-only for callers.
        """
        return self._sub_agents

    def add_sub_agent(self, sub_agent: "SubAgent") -> None:
        """Register an additional SubAgent post-construction.

        Useful for Django callers that discover sub-assistants from the
        ORM after the parent Agent has been constructed. The framework's
        dispatch tool synthesis reads from `self._sub_agents` at
        orchestrator boundary time, so adding here before `stream()` is
        sufficient.
        """
        # No-collision invariant — mirrors AgentConfig.__post_init__.
        existing_handles = {s.handle for s in self._sub_agents}
        if sub_agent.handle in existing_handles:
            raise ValueError(
                f"Sub-agent '{sub_agent.handle}' already registered on this agent."
            )
        tool_names = {t.name for t in self._tools}
        if sub_agent.handle in tool_names:
            raise ValueError(
                f"Sub-agent handle '{sub_agent.handle}' collides with a "
                f"tool of the same name."
            )
        self._sub_agents.append(sub_agent)

    async def run(
        self,
        user_prompt: str,
        *,
        deps: Optional[Deps] = None,
        message_history: Optional[List[Message]] = None,
    ) -> RunResult[Result]:
        """Run the agent with dependency injection (stateless)."""
        from .callback_context import get_callback_context
        from .callbacks import CallbackEvent, CallbackEventType, get_global_registry
        from .tools.tool_search import tool_search_session

        # Open a per-run ToolSearch session so the registry can hide most tool
        # schemas behind the tool_search meta-tool when the catalog is large.
        # This is a no-op for small registries (see should_use_tool_search()).
        with tool_search_session():
            return await self._run_inner(user_prompt, deps=deps, message_history=message_history)

    async def _run_inner(
        self,
        user_prompt: str,
        *,
        deps: Optional[Deps] = None,
        message_history: Optional[List[Message]] = None,
    ) -> RunResult[Result]:
        from .callback_context import get_callback_context
        from .callbacks import CallbackEvent, CallbackEventType, get_global_registry

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

        # Get callback context and emit AGENT_RUN_START
        ctx = get_callback_context()
        start_event = CallbackEvent(
            event_type=CallbackEventType.AGENT_RUN_START,
            agent_type=self.agent_type.value,
            query=user_prompt,
            context=ctx,
        )
        await get_global_registry().emit(start_event)

        # Execute with retries
        success = False
        for attempt in range(self.retries):
            context.retry = attempt
            try:
                result = await self._execute_with_context(context)
                success = True

                # Emit AGENT_RUN_END on success
                end_event = CallbackEvent(
                    event_type=CallbackEventType.AGENT_RUN_END,
                    agent_type=self.agent_type.value,
                    query=user_prompt,
                    context=ctx,
                    success=True,
                )
                await get_global_registry().emit(end_event)

                return RunResult(
                    data=result, messages=context.messages, all_messages=context.messages.copy()
                )

            except Exception as e:
                if attempt == self.retries - 1:
                    # Emit AGENT_RUN_END on final failure
                    end_event = CallbackEvent(
                        event_type=CallbackEventType.AGENT_RUN_END,
                        agent_type=self.agent_type.value,
                        query=user_prompt,
                        context=ctx,
                        success=False,
                        error=e,
                        error_type=type(e).__name__,
                    )
                    await get_global_registry().emit(end_event)
                    raise MiiflowLLMError(
                        f"Agent failed after {self.retries} retries: {e}", ErrorType.MODEL_ERROR
                    )
                continue

        raise MiiflowLLMError("Agent execution failed", ErrorType.MODEL_ERROR)

    async def _execute_with_context(self, context: RunContext[Deps]) -> str:
        """Route to the unified ReAct loop (SINGLE_HOP keeps its own path).

        Plan-and-execute / parallel-plan / multi-agent are no longer
        separate orchestrators — they are emergent behaviors of the
        ReAct loop with `enter_plan_mode` and `dispatch_assistant`
        tools. The legacy ``AgentType`` enum values still pass through
        the constructor (so callers don't have to change config) but
        they all behave like REACT now.
        """
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
        else:
            # REACT, PLAN_AND_EXECUTE, PARALLEL_PLAN, MULTI_AGENT all
            # collapse to the same ReAct loop. The model decides whether
            # to plan, dispatch, or answer per turn via tool calls.
            async for event in self._stream_react(
                user_prompt, context, max_steps=self.max_iterations
            ):
                if event.event_type.value == "final_answer":
                    final_answer = event.data.get("answer", "")
                    break

        return final_answer or "No final answer received"

    async def _execute_tool_calls(
        self, tool_calls: List[Dict[str, Any]], context: RunContext[Deps]
    ) -> List[Dict[str, Any]]:
        """Execute tool calls with dependency injection.

        Returns:
            List of special result dicts (media/visualization) detected during execution.
            Each entry has {"type": "media"|"visualization", "data": ..., "tool_name": ...}.
        """
        from miiflow_agent.visualization.types import (
            is_media_result, extract_media_data,
            is_media_collection, extract_media_collection,
            is_visualization_result, extract_visualization_data,
        )
        from miiflow_agent.artifacts import is_artifact_result, extract_artifact_data

        special_results = []
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

            # Detect special results (media/visualization) before stringification
            if observation.success and observation.output is not None:
                if is_media_collection(observation.output):
                    media_items = extract_media_collection(observation.output) or []
                    for media_data in media_items:
                        special_results.append({
                            "type": "media",
                            "data": media_data,
                            "tool_name": tool_name,
                        })
                elif is_media_result(observation.output):
                    media_data = extract_media_data(observation.output)
                    if media_data:
                        special_results.append({
                            "type": "media",
                            "data": media_data,
                            "tool_name": tool_name,
                        })
                elif is_visualization_result(observation.output):
                    viz_data = extract_visualization_data(observation.output)
                    if viz_data:
                        special_results.append({
                            "type": "visualization",
                            "data": viz_data,
                            "tool_name": tool_name,
                        })
                elif is_artifact_result(observation.output):
                    artifact_data = extract_artifact_data(observation.output)
                    if artifact_data:
                        special_results.append({
                            "type": "artifact",
                            "data": artifact_data,
                            "tool_name": tool_name,
                        })

            # Add tool result message
            context.messages.append(
                Message(
                    role=MessageRole.TOOL,
                    content=str(observation.output) if observation.success else observation.error,
                    tool_call_id=tool_call.id if hasattr(tool_call, "id") else tool_call.get("id"),
                )
            )

        return special_results

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
        event_format: str = "react",
        thread_id: Optional[str] = None,
        message_id: Optional[str] = None,
    ) -> AsyncIterator[Any]:
        """Unified streaming method that dispatches based on agent_type.

        Args:
            query: User's query/goal
            context: Run context with messages and deps
            agent_type: Optional override for agent type. Defaults to self.agent_type
            max_steps: Maximum ReAct steps (for REACT type)
            max_budget: Optional budget limit (for REACT type)
            max_time_seconds: Optional time limit (for REACT type)
            max_replans: Kept for signature compat — ignored after the
                unified-ReAct migration (replanning is now a model-driven
                choice via repeated ``enter_plan_mode`` calls).
            existing_plan: Kept for signature compat — ignored.
            event_format: Event format - "react" for legacy events, "agui" for AG-UI protocol
            thread_id: Thread ID (required for agui format)
            message_id: Message ID (required for agui format)

        Yields:
            Streaming events specific to the agent type.
            In "react" mode: ReActEvent objects
            In "agui" mode: AG-UI protocol events (TextMessageContentEvent, etc.)
        """
        from .callback_context import get_callback_context
        from .callbacks import CallbackEvent, CallbackEventType, get_global_registry
        from .tools.tool_search import tool_search_session, is_session_active

        # Open a per-run ToolSearch session if none is active. We use a
        # ``with`` block on the synchronous context manager *inside* the async
        # generator body so the ContextVar token is set, used, and reset all
        # within a single task frame — important under ASGI where this
        # generator may be iterated by middleware.
        _own_session = not is_session_active()

        # Validate agui requirements
        if event_format == "agui" and (not thread_id or not message_id):
            raise ValueError("thread_id and message_id are required for agui event format")

        effective_type = agent_type or self.agent_type

        # Ensure context has the query as a USER message (similar to run()).
        # Compare on extracted text, not raw content, so multimodal messages
        # (list of blocks) aren't mistaken for a different message and re-
        # appended as a flat-string duplicate.
        has_user_message = any(
            msg.role == MessageRole.USER and _content_text(msg.content) == query
            for msg in context.messages
        )
        if not has_user_message and query and query.strip():
            context.messages.append(Message(role=MessageRole.USER, content=query))

        # Get callback context and emit AGENT_RUN_START
        ctx = get_callback_context()
        start_event = CallbackEvent(
            event_type=CallbackEventType.AGENT_RUN_START,
            agent_type=effective_type.value,
            query=query,
            context=ctx,
        )
        await get_global_registry().emit(start_event)

        # Import AG-UI factory if needed for lifecycle events
        agui_factory = None
        if event_format == "agui":
            from .react.events import AGUIEventFactory, AGUI_AVAILABLE
            if AGUI_AVAILABLE:
                agui_factory = AGUIEventFactory(thread_id, message_id)
                # Emit run_started event
                yield agui_factory.run_started()

        # Open the ToolSearch session inside a `with` block scoped to this
        # generator's body. The contextvar token is set and reset within the
        # same task frame that drives the generator, which keeps it safe under
        # ASGI middlewares that may rebind generator iteration to a child task.
        _session_cm = tool_search_session() if _own_session else _NullCM()
        with _session_cm:
            try:
                # SINGLE_HOP keeps its own one-shot LLM call (preserves
                # the json_schema response path). Every other agent_type
                # collapses to the unified ReAct loop — the model picks
                # its own behavior via plan-mode and dispatch_assistant
                # tool calls, no top-level mode selection needed.
                if effective_type == AgentType.SINGLE_HOP:
                    async for event in self._stream_single_hop(query, context=context):
                        yield event
                else:
                    async for event in self._stream_react(
                        query, context, max_steps, max_budget, max_time_seconds,
                        event_format=event_format, thread_id=thread_id, message_id=message_id
                    ):
                        yield event

                # Emit run_finished event for AG-UI mode
                if agui_factory:
                    yield agui_factory.run_finished()

                # Emit AGENT_RUN_END on success
                end_event = CallbackEvent(
                    event_type=CallbackEventType.AGENT_RUN_END,
                    agent_type=effective_type.value,
                    query=query,
                    context=ctx,
                    success=True,
                )
                await get_global_registry().emit(end_event)

            except Exception as e:
                # Emit run_error event for AG-UI mode
                if agui_factory:
                    yield agui_factory.run_error(str(e))

                # Emit AGENT_RUN_END on failure
                end_event = CallbackEvent(
                    event_type=CallbackEventType.AGENT_RUN_END,
                    agent_type=effective_type.value,
                    query=query,
                    context=ctx,
                    success=False,
                    error=e,
                    error_type=type(e).__name__,
                )
                await get_global_registry().emit(end_event)

                raise

    async def _stream_react(
        self,
        query: str,
        context: RunContext,
        max_steps: int = 10,
        max_budget: Optional[float] = None,
        max_time_seconds: Optional[float] = None,
        event_format: str = "react",
        thread_id: Optional[str] = None,
        message_id: Optional[str] = None,
    ):
        """Internal: Run agent in ReAct mode with streaming events."""
        from .react import ReActFactory

        # Create recovery manager (always enabled for ReAct agents)
        # Works with or without compressor — RETRY_WITH_GUIDANCE and SIMPLIFY_TOOLS
        # don't need one, only COMPRESS_AND_RETRY does (and it handles None gracefully)
        from .react.recovery import RecoveryManager
        recovery_manager = RecoveryManager(
            context_compressor=self._context_compressor,
        )

        orchestrator = ReActFactory.create_orchestrator(
            agent=self,
            max_steps=max_steps,
            max_budget=max_budget,
            max_time_seconds=max_time_seconds,
            event_format=event_format,
            thread_id=thread_id,
            message_id=message_id,
            recovery_manager=recovery_manager,
            context_compressor=self._context_compressor,
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

        # Add user message to context if not already present.
        # Compare on extracted text, not raw content, so multimodal messages
        # (list of blocks) aren't mistaken for a missing turn and re-appended
        # as a flat-string duplicate that doubles input tokens.
        last = context.messages[-1] if context.messages else None
        if not last or _content_text(last.content) != user_prompt:
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
            streamed_tool_calls: Dict[str, Dict[str, Any]] = {}

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

                # Accumulate tool calls from the stream. Each provider re-emits the
                # same id with progressively more complete arguments (Anthropic:
                # empty dict at start, parsed dict at stop; OpenAI: deltas growing
                # the args string), so last-write-wins by id yields the final call.
                if chunk.tool_calls:
                    for tc in chunk.tool_calls:
                        tc_id = tc.get("id") or f"_idx_{len(streamed_tool_calls)}"
                        streamed_tool_calls[tc_id] = tc

                if chunk.finish_reason:
                    break

            final_tool_calls = list(streamed_tool_calls.values()) or None

            response_message = Message(
                role=MessageRole.ASSISTANT, content=buffer, tool_calls=final_tool_calls
            )
            context.messages.append(response_message)

            # Handle tool calls if present
            if final_tool_calls:
                yield {"event": "tools_start", "data": {"tool_count": len(final_tool_calls)}}

                special_results = await self._execute_tool_calls(final_tool_calls, context)

                # Emit media/visualization events detected during tool execution
                # (single_hop doesn't go through ReAct orchestrator which normally handles this)
                from miiflow_agent.core.react.events.bus import EventFactory
                for sr in special_results:
                    if sr["type"] == "media":
                        yield EventFactory.media(0, sr["data"], sr["tool_name"])
                    elif sr["type"] == "visualization":
                        yield EventFactory.visualization(0, sr["data"], sr["tool_name"])
                    elif sr["type"] == "artifact":
                        yield EventFactory.artifact(0, sr["data"], sr["tool_name"])

                yield {"event": "tools_complete", "data": {}}
                # Re-filter messages after tool execution (tool results added to context.messages)
                llm_messages = self._prepare_messages_for_llm(context.messages)

                # Stream the post-tool LLM response
                post_tool_buffer = ""
                async for chunk in self.client.astream_chat(
                    messages=llm_messages,
                    tools=None,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    json_schema=self.json_schema,
                ):
                    if chunk.delta:
                        post_tool_buffer += chunk.delta
                        yield {"event": "llm_chunk", "data": {"delta": chunk.delta, "content": post_tool_buffer}}

                    if chunk.finish_reason:
                        break

                final_message = Message(role=MessageRole.ASSISTANT, content=post_tool_buffer)
                context.messages.append(final_message)
                result = post_tool_buffer
            else:
                result = buffer

            yield {"event": "execution_complete", "data": {"result": result}}

        except Exception as e:
            yield {"event": "error", "data": {"error": str(e), "error_type": type(e).__name__}}
            raise
