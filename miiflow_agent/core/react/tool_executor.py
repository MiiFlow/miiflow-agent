"""Clean tool executor adapter."""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..callbacks import CallbackEvent, CallbackEventType, get_global_registry
from ..callback_context import get_callback_context
from ..tools import ToolResult

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """One tool invocation in a (possibly parallel) batch.

    `tool_call_id` is the provider-supplied id (e.g. Anthropic's `toolu_*`)
    that must round-trip onto the matching `tool_result` block in the
    assistant transcript — required for the Anthropic API to accept the
    next turn (tool_use/tool_result pairing invariant).
    """

    tool_call_id: str
    name: str
    inputs: Dict[str, Any]


class AgentToolExecutor:
    """Tool execution adapter following Django Manager pattern."""

    def __init__(self, agent, tool_filter=None):
        self.agent = agent
        self._tool_registry = agent.tool_registry
        self._client = agent.client
        self.tool_filter = tool_filter  # Optional ToolFilter for narrowing available tools

    async def execute_tool(self, tool_name: str, inputs: dict, context=None) -> ToolResult:
        """Execute tool with context injection if context is provided.

        Emits PRE_TOOL_USE callback before execution (can block via ToolApprovalRequired).
        Emits a TOOL_EXECUTED callback event after execution for billing/tracking.
        """
        # Check tool filter before execution
        if self.tool_filter and not self.tool_filter.is_allowed(tool_name):
            return ToolResult(
                name=tool_name,
                input=inputs,
                output=None,
                error=f"Tool '{tool_name}' is not available in the current execution context.",
                success=False,
            )

        # Plan-mode gate: when the run is in plan mode, only tools that
        # declare `is_read_only=True` execute. Anything else gets a
        # synthetic refusal so the model self-corrects by either calling
        # a read-only alternative or exiting plan mode. We check before
        # the PRE_TOOL_USE callback to avoid spending an approval prompt
        # on a tool we're about to refuse anyway.
        blocked_result = self._plan_mode_block_if_needed(tool_name, inputs, context)
        if blocked_result is not None:
            return blocked_result

        # Emit PRE_TOOL_USE callback - allows blocking tool execution (e.g. for approval)
        await self._emit_pre_tool_use_callback(tool_name, inputs)

        start_time = time.time()

        # Respect per-tool needs_context: even if the caller passes a
        # context defensively (e.g. the batch path forwards one to every
        # call), only inject it for tools that actually expect it.
        # Without this guard, context-less tools would crash on the
        # `execute_safe_with_context` path. Safe no-op for the existing
        # single-tool callers since they already gate externally before
        # passing context — those keep working unchanged.
        if context is not None and not self.tool_needs_context(tool_name):
            context = None

        if context is not None:
            result = await self._tool_registry.execute_safe_with_context(tool_name, context, **inputs)
        else:
            result = await self._tool_registry.execute_safe(tool_name, **inputs)

        execution_time_ms = (time.time() - start_time) * 1000

        # Emit TOOL_EXECUTED callback for billing/tracking
        await self._emit_tool_executed_callback(
            tool_name=tool_name,
            inputs=inputs,
            result=result,
            execution_time_ms=execution_time_ms,
        )

        return result

    def is_batch_parallelizable(self, tool_calls: List[ToolCall]) -> bool:
        """Return True iff every tool in the batch can run in parallel.

        Applies the "all-or-nothing" rule: if ANY tool is
        ``parallelizable=False`` OR ``require_approval=True``, the whole
        batch runs serially. This is strictly safer than partial parallel
        (one mutation interleaved with reads has surprising ordering
        semantics) and side-steps the multi-approval-modal UX problem.
        """
        for tc in tool_calls:
            schema = self._get_tool_schema_obj(tc.name)
            if schema is None:
                # Unknown tool — be conservative; serial fallback lets the
                # error message land in a deterministic order.
                return False
            if not getattr(schema, "parallelizable", False):
                return False
            if getattr(schema, "require_approval", False):
                return False
        return True

    async def execute_many(
        self,
        tool_calls: List[ToolCall],
        context=None,
    ) -> List[ToolResult]:
        """Execute a batch of tool calls; return results in input order.

        If the batch is fully parallelizable (every tool ``parallelizable=True``
        and ``require_approval=False``), runs concurrently via
        ``asyncio.gather(return_exceptions=True)``. Otherwise falls back to
        sequential execution in the input order — same behavior as today's
        single-tool-per-step orchestrator path, just iterated.

        Each ``ToolResult`` independently carries ``success=False`` + ``error``
        for failures; raw exceptions from ``asyncio.gather`` are wrapped into
        ``ToolResult`` so one bad call never tanks the whole batch.

        Callbacks (PRE_TOOL_USE, TOOL_EXECUTED) fire per-tool via
        ``execute_tool``; under parallel mode they emit concurrently — order
        of TOOL_EXECUTED is completion-order, not input-order. Callers that
        need strict ordering should not mark their tools parallelizable.

        ``ToolApprovalRequired`` only fires in serial mode (approval-required
        tools force the batch serial via ``is_batch_parallelizable``). When
        raised, it propagates up to the orchestrator which pauses the run
        for user approval — same as today.
        """
        if not tool_calls:
            return []

        if self.is_batch_parallelizable(tool_calls):
            return await self._execute_parallel(tool_calls, context)
        return await self._execute_serial(tool_calls, context)

    async def _execute_serial(
        self,
        tool_calls: List[ToolCall],
        context,
    ) -> List[ToolResult]:
        """Run the batch one-at-a-time in input order. Same semantics as
        the single-tool path; failures don't stop subsequent tools.
        """
        results: List[ToolResult] = []
        for tc in tool_calls:
            results.append(await self.execute_tool(tc.name, tc.inputs, context=context))
        return results

    async def _execute_parallel(
        self,
        tool_calls: List[ToolCall],
        context,
    ) -> List[ToolResult]:
        """Run the batch via ``asyncio.gather``. Each tool's success/failure
        is independently captured. Raw exceptions get wrapped into
        ``ToolResult(success=False)`` so the caller always gets a result
        per input — same length, same order.

        Context isolation invariant: each tool call must run in its own
        copy of the caller's ContextVar Context so sibling tools cannot
        observe or clobber each other's contextvar state (notably the
        ToolSearch ``_enabled_tools`` set and any per-call mutations on
        ``ctx.deps``-adjacent contextvars). ``asyncio.gather`` on bare
        coroutines achieves this indirectly via ``ensure_future`` →
        ``loop.create_task``, but that's a behavior of the gather
        implementation, not a guarantee at this seam. We wrap each
        coroutine in an explicit ``asyncio.create_task`` so the
        per-branch Context copy is a property of THIS function — future
        refactors that swap ``gather`` for a TaskGroup, a manual loop, or
        any other scheduling primitive can't accidentally regress the
        isolation. See the dispatch_assistant flow (Django
        ``AssistantInvoker``) for the parent-most consumer of this
        invariant.
        """
        tasks = [
            asyncio.create_task(
                self.execute_tool(tc.name, tc.inputs, context=context),
                name=f"tool:{tc.name}:{tc.tool_call_id}",
            )
            for tc in tool_calls
        ]
        raw = await asyncio.gather(*tasks, return_exceptions=True)

        results: List[ToolResult] = []
        for tc, item in zip(tool_calls, raw):
            if isinstance(item, BaseException):
                logger.exception(
                    "Parallel tool '%s' raised an unhandled exception",
                    tc.name,
                    exc_info=item,
                )
                results.append(
                    ToolResult(
                        name=tc.name,
                        input=tc.inputs,
                        output=None,
                        error=f"Tool execution error: {item}",
                        success=False,
                    )
                )
            else:
                results.append(item)
        return results

    def _plan_mode_block_if_needed(
        self,
        tool_name: str,
        inputs: dict,
        context,
    ) -> Optional[ToolResult]:
        """Return a synthetic blocked-tool result iff this call should be
        refused for being non-read-only while plan mode is active.

        Logic mirrors Claude Code's plan-mode `canUseTool`: only tools
        declaring `is_read_only=True` execute while in plan mode. The
        run-state flag is set by `enter_plan_mode` and cleared by
        `exit_plan_mode` (both in `core/tools/plan_mode.py`).

        Returns `None` to pass through to normal execution. Emits a
        `TOOL_BLOCKED_BY_PLAN_MODE` event on the run's event bus when a
        block fires so the UI can surface the refusal as a hint.
        """
        run_state = getattr(context, "run_state", None) if context is not None else None
        if run_state is None:
            return None

        mode = getattr(run_state, "permission_mode", "default")
        if mode != "plan":
            return None

        schema = self._get_tool_schema_obj(tool_name)
        if schema is not None and getattr(schema, "is_read_only", False):
            return None

        # Fall through any tool with a known is_read_only=True flag; only
        # block tools we can prove are NOT read-only. Unknown tools are
        # treated as not-read-only (safer default — the model can still
        # exit plan mode and retry).
        message = (
            f"Tool '{tool_name}' is blocked while plan mode is active. "
            f"Plan mode only permits read-only tools (search/list/describe). "
            f"Finish investigating, then call `exit_plan_mode` with your plan "
            f"as markdown. Side-effectful tools will become available again."
        )

        bus = getattr(run_state, "event_bus", None)
        step_number = int(getattr(run_state, "step_number", 0) or 0)
        if bus is not None:
            # Best-effort fire-and-forget — never let the refusal pathway
            # itself raise; the synthetic ToolResult is the contract.
            try:
                from .enums import ReActEventType
                from .react_events import ReActEvent

                # Don't await: execute_tool is async but the event bus
                # publish is also async. We're inside an async function so
                # we can schedule a task that completes alongside the
                # return. This keeps the refusal synchronous from the
                # caller's perspective.
                import asyncio as _asyncio

                _asyncio.create_task(
                    bus.publish(
                        ReActEvent(
                            event_type=ReActEventType.TOOL_BLOCKED_BY_PLAN_MODE,
                            step_number=step_number,
                            data={"tool_name": tool_name, "inputs": inputs},
                        )
                    ),
                    name=f"plan-mode-block:{tool_name}",
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("plan-mode block event emission skipped: %s", exc)

        return ToolResult(
            name=tool_name,
            input=inputs,
            output=None,
            error=message,
            success=False,
            metadata={"error_type": "blocked_by_plan_mode"},
        )

    def _get_tool_schema_obj(self, tool_name: str):
        """Return the underlying ``ToolSchema`` instance (not the dict).

        Needed for reading the ``parallelizable`` / ``require_approval``
        booleans, which the dict-shaped ``get_tool_schema`` doesn't expose.
        """
        if tool_name == self._tool_search_meta_name():
            meta = self._tool_registry.get_tool_search_tool()
            return getattr(meta, "schema", None)
        tool = self._lookup_any_tool(tool_name)
        return getattr(tool, "schema", None) if tool else None

    async def _emit_pre_tool_use_callback(
        self,
        tool_name: str,
        inputs: dict,
    ) -> None:
        """Emit a PRE_TOOL_USE callback event before tool execution.

        If a callback sets event.blocked = True, raises ToolApprovalRequired
        to signal that the tool requires user approval.
        """
        callback_context = get_callback_context()

        event = CallbackEvent(
            event_type=CallbackEventType.PRE_TOOL_USE,
            tool_name=tool_name,
            tool_inputs=inputs,
            context=callback_context,
        )

        try:
            registry = get_global_registry()
            await registry.emit(event)
        except Exception as e:
            logger.warning(f"Failed to emit PRE_TOOL_USE callback: {e}")
            return

        if event.blocked:
            from .exceptions import ToolApprovalRequired

            raise ToolApprovalRequired(
                tool_name=tool_name,
                tool_inputs=inputs,
                reason=event.block_reason,
            )

    async def _emit_tool_executed_callback(
        self,
        tool_name: str,
        inputs: dict,
        result: ToolResult,
        execution_time_ms: float,
    ) -> None:
        """Emit a TOOL_EXECUTED callback event."""
        # Get current callback context if set
        callback_context = get_callback_context()

        event = CallbackEvent(
            event_type=CallbackEventType.TOOL_EXECUTED,
            tool_name=tool_name,
            tool_inputs=inputs,
            tool_output=result.output if result else None,
            tool_execution_time_ms=execution_time_ms,
            success=result.success if result else False,
            context=callback_context,
        )

        try:
            registry = get_global_registry()
            await registry.emit(event)
        except Exception as e:
            logger.warning(f"Failed to emit TOOL_EXECUTED callback: {e}")

    async def execute_without_tools(self, messages: List, temperature: float = None):
        """Execute LLM call with tools temporarily disabled."""
        saved_state = self._save_tool_state()

        try:
            self._disable_all_tools()
            return await self._client.achat(
                messages=messages,
                temperature=temperature or self.agent.temperature,
                max_tokens=self.agent.max_tokens,
            )
        finally:
            self._restore_tool_state(saved_state)

    async def stream_without_tools(self, messages: List, temperature: float = None):
        """Stream LLM call with tools temporarily disabled."""
        saved_state = self._save_tool_state()

        try:
            self._disable_all_tools()
            async for chunk in self._client.astream_chat(
                messages=messages,
                temperature=temperature or self.agent.temperature,
                max_tokens=self.agent.max_tokens,
            ):
                yield chunk
        finally:
            self._restore_tool_state(saved_state)

    async def execute_with_tools(self, messages: List, temperature: float = None):
        """Execute LLM call WITH native tools enabled."""
        tools = self._build_native_tool_schemas()

        # Use LLMClient with pre-formatted tools via _formatted_tools parameter
        # This ensures callbacks fire while avoiding tool re-formatting
        return await self._client.achat(
            messages=messages,
            _formatted_tools=tools,
            temperature=temperature or self.agent.temperature,
            max_tokens=self.agent.max_tokens,
        )

    async def stream_with_tools(self, messages: List, temperature: float = None):
        """Stream LLM call WITH native tools enabled."""
        tools = self._build_native_tool_schemas()

        # Use LLMClient with pre-formatted tools via _formatted_tools parameter
        # This ensures callbacks fire while avoiding tool re-formatting
        async for chunk in self._client.astream_chat(
            messages=messages,
            _formatted_tools=tools,
            temperature=temperature or self.agent.temperature,
            max_tokens=self.agent.max_tokens,
        ):
            yield chunk

    def _build_native_tool_schemas(self) -> List:
        """Build tool schemas in native provider format.

        Converts universal schemas to provider-specific format
        (OpenAI, Anthropic, Gemini, etc.)

        When a ToolSearch session is active and the registry is large enough
        to warrant it, only the meta-tool plus always_load tools plus the set
        the model has discovered via ``tool_search`` are included. Otherwise
        every registered function tool is included (legacy behaviour).
        """
        from ..tools import FunctionTool
        from ..tools.tool_search import get_enabled_tool_names, is_session_active

        # Determine which tools to expose this turn.
        use_tool_search = (
            is_session_active() and self._tool_registry.should_use_tool_search()
        )
        if use_tool_search:
            visible = set(self._tool_registry.get_always_load_names())
            enabled = get_enabled_tool_names()
            if enabled:
                visible.update(enabled)
            # The meta-tool itself is always exposed under tool_search.
            tool_names = [name for name in self.list_tools() if name in visible]
        else:
            tool_names = self.list_tools()

        native_schemas: List = []

        for tool_name in tool_names:
            tool = self._lookup_any_tool(tool_name)
            if not tool:
                continue

            # Get universal schema from tool. FunctionTool exposes `.schema`;
            # MCPTool/HTTPTool also expose `.schema`. Use it directly for any
            # tool that has it so MCP/HTTP tools aren't silently dropped from
            # the LLM's tool surface when surfaced via tool_search.
            tool_schema_obj = getattr(tool, "schema", None)
            if tool_schema_obj is not None:
                universal_schema = tool_schema_obj.to_universal_schema()
            else:
                universal_schema = self.get_tool_schema(tool_name)

            # Filter out context parameters from schema
            # (context is injected, not exposed to LLM)
            filtered_schema = self._filter_context_params(tool_name, universal_schema)

            # Inject __description parameter for LLM to provide human-readable descriptions
            filtered_schema = self._inject_description_param(filtered_schema)

            # Convert to provider-specific format
            provider_schema = self._client.client.convert_schema_to_provider_format(filtered_schema)
            native_schemas.append(provider_schema)

        # Append the tool_search meta-tool when ToolSearch is active so the
        # model can discover hidden tools by name/keyword.
        if use_tool_search:
            meta_tool = self._tool_registry.get_tool_search_tool()
            meta_universal = meta_tool.schema.to_universal_schema()
            meta_filtered = self._inject_description_param(meta_universal)
            native_schemas.append(
                self._client.client.convert_schema_to_provider_format(meta_filtered)
            )

        # Apply tool filter if configured
        if self.tool_filter:
            native_schemas = self.tool_filter.filter_schemas(native_schemas)

        logger.debug(
            f"Built {len(native_schemas)} native tool schemas for provider "
            f"{self._client.client.provider_name} (tool_search={use_tool_search})"
        )

        # Debug: Log the actual schemas being sent
        import json

        logger.debug(
            f"Tool schemas being sent to provider:\n{json.dumps(native_schemas, indent=2, default=str)}"
        )

        return native_schemas

    def _filter_context_params(self, tool_name: str, schema: dict) -> dict:
        """Remove context parameters from schema (they're injected, not LLM-provided)."""
        tool = self._tool_registry.tools.get(tool_name)
        if not tool or not hasattr(tool, "context_injection"):
            return schema

        context_pattern = tool.context_injection.get("pattern", "none")

        # If tool has context injection, remove the context parameter from schema
        if context_pattern == "first_param":
            # First param is context - already handled by FunctionTool schema generation
            # FunctionTool.schema already excludes first param if it's context
            return schema

        elif context_pattern == "keyword":
            # Remove 'context' or 'ctx' keyword parameter from properties
            filtered_schema = schema.copy()
            if "parameters" in filtered_schema and "properties" in filtered_schema["parameters"]:
                properties = filtered_schema["parameters"]["properties"].copy()
                properties.pop("context", None)
                properties.pop("ctx", None)
                filtered_schema["parameters"]["properties"] = properties

                # Also remove from required list
                if "required" in filtered_schema["parameters"]:
                    required = [
                        r
                        for r in filtered_schema["parameters"]["required"]
                        if r not in ("context", "ctx")
                    ]
                    filtered_schema["parameters"]["required"] = required

            return filtered_schema

        return schema

    def _inject_description_param(self, schema: dict) -> dict:
        """Inject __description parameter into tool schema (required).

        Asks the LLM to attach a short, verb-led imperative phrase to each tool
        call (e.g. 'Search the web for Tesla news'). The phrase is shown in the
        chat UI as both a status label and — when approval is required — the
        question the user is consenting to.

        Required, not optional. An earlier iteration kept this optional because
        a vague schema description led models to either emit perfunctory
        labels ("Calling tool") or skip the field entirely, leaving the UI to
        fall back to bare tool names like ``tool_search``. The schema
        description below pins ``__description`` to the *specific* call's
        action with per-tool examples, which makes "describe what THIS call
        does" the path of least resistance — and making it required ensures
        every call gets a user-readable label rather than a raw tool name.

        Cost: ~10-30 extra output tokens per tool call. Acceptable in
        exchange for consistent UX; flag if you observe max_tokens
        truncation tied to verbose-arg tools.
        """
        import copy

        schema = copy.deepcopy(schema)
        if "parameters" not in schema:
            schema["parameters"] = {"type": "object", "properties": {}, "required": []}

        # Ensure properties dict exists
        if "properties" not in schema["parameters"]:
            schema["parameters"]["properties"] = {}

        # Add __description as a *required* property. Format: short verb-led
        # action phrase that a non-technical user can read as either a status
        # ("Search the web for X") or a question ("Search the web for X?" with
        # consent chips beneath). Avoid gerunds ("Searching") and tool-name
        # jargon ("search_web") — those read as a debugger, not as the
        # assistant communicating with the user.
        #
        # Critical: __description must describe THIS specific tool call's
        # action, not an overall plan or what comes next. Models otherwise
        # tend to write a plan-level summary ("Pull this month's campaign
        # performance") on every call along the way, which mislabels early
        # discovery/lookup steps in the UI.
        schema["parameters"]["properties"]["__description"] = {
            "type": "string",
            "description": (
                "Required. Short imperative phrase describing THIS specific "
                "tool call — not your overall plan, not the next step, not "
                "the user's broader request. The phrase appears as a status "
                "label in the UI while this exact call runs.\n\n"
                "Match the phrase to what THIS call does, using the tool's "
                "actual arguments. For example, with `search_memory`, write "
                "'Search memory for past campaign preferences' — not 'Pull "
                "campaign performance'. With `list_all_ad_accounts`, write "
                "'List connected ad accounts' — not 'Compare Google and "
                "Meta spend'. With `google_ads_query` for a rollup, write "
                "'Pull account-level rollup for April 2026' — not "
                "'Summarize this month'.\n\n"
                "Format rules: verb-led imperative ('Search', 'List', "
                "'Pull', 'Send', 'Look up'). No gerunds ('Searching'). No "
                "tool-name jargon ('search_web'). No ellipses or trailing "
                "punctuation. Never repeat the same description across "
                "consecutive calls."
            ),
        }

        # Mark __description as required so every tool call carries a
        # user-readable label. Append to the existing required list rather
        # than replacing it so other required fields aren't lost.
        required = list(schema["parameters"].get("required") or [])
        if "__description" not in required:
            required.append("__description")
        schema["parameters"]["required"] = required

        return schema

    def _tool_search_meta_name(self) -> Optional[str]:
        """Name of the off-registry tool_search meta-tool, if it has been built."""
        meta = getattr(self._tool_registry, "_tool_search_tool", None)
        return meta.name if meta is not None else None

    def list_tools(self) -> List[str]:
        tools = self._tool_registry.list_tools()
        if self.tool_filter:
            return self.tool_filter.filter_tool_names(tools)
        return tools

    def _lookup_any_tool(self, tool_name: str):
        """Return the registered tool object across function/http/mcp dicts, or None.

        The registry partitions tools into three separate dicts; lookups that
        only consult ``tools`` silently miss every MCP/HTTP tool. Callers that
        just need "is this name registered, and give me the object" should use
        this single helper instead of touching the dicts directly.
        """
        reg = self._tool_registry
        return (
            reg.tools.get(tool_name)
            or reg.http_tools.get(tool_name)
            or reg.mcp_tools.get(tool_name)
        )

    def has_tool(self, tool_name: str) -> bool:
        # The tool_search meta-tool lives off-registry; recognize it when
        # the orchestrator validates an LLM-issued tool call against the
        # executor's view of available tools.
        if tool_name == self._tool_search_meta_name():
            return True
        if self._lookup_any_tool(tool_name) is None:
            return False
        if self.tool_filter and not self.tool_filter.is_allowed(tool_name):
            return False
        return True

    def get_tool_schema(self, tool_name: str) -> dict:
        if tool_name == self._tool_search_meta_name():
            return self._tool_registry.get_tool_search_tool().schema.to_universal_schema()
        tool = self._lookup_any_tool(tool_name)
        return tool.schema.to_universal_schema() if tool else {}

    def tool_needs_context(self, tool_name: str) -> bool:
        """Check if a tool requires context injection."""
        # The meta-tool runs without a context (it operates on the registry,
        # not on user data).
        if tool_name == self._tool_search_meta_name():
            return False
        tool = self._lookup_any_tool(tool_name)
        if not tool:
            return False
        # Check if tool has context_injection attribute and if pattern is not 'none'
        if hasattr(tool, "context_injection"):
            pattern = tool.context_injection.get("pattern", "none")
            return pattern in ("first_param", "keyword")
        return False

    def build_tools_description(self) -> str:
        """Format all tools for system prompt."""
        if not self.list_tools():
            return "No tools available."

        descriptions = []
        for tool_name in sorted(self.list_tools()):
            schema = self.get_tool_schema(tool_name)
            tool_desc = self._format_tool_description(tool_name, schema)
            descriptions.append(tool_desc)

        tools_text = "\n".join(descriptions)

        return tools_text

    def _save_tool_state(self) -> dict:
        return {
            "tools": dict(self._client.tool_registry.tools),
            "http_tools": dict(self._client.tool_registry.http_tools),
        }

    def _restore_tool_state(self, state: dict) -> None:
        self._client.tool_registry.tools = state["tools"]
        self._client.tool_registry.http_tools = state["http_tools"]

    def _disable_all_tools(self) -> None:
        self._client.tool_registry.tools = {}
        self._client.tool_registry.http_tools = {}

    def _format_tool_description(self, tool_name: str, schema: dict) -> str:
        """Format tool description for system prompt with detailed parameter information."""
        desc = schema.get("description", "No description available")
        params = schema.get("parameters", {}).get("properties", {})

        if not params:
            return f"- {tool_name}(): {desc}"

        # Format parameters with type and enum information
        param_descriptions = []
        for param_name, param_schema in params.items():
            param_type = param_schema.get("type", "any")

            # If enum exists, show the allowed values
            if "enum" in param_schema and param_schema["enum"]:
                enum_values = param_schema["enum"]
                if len(enum_values) <= 5:  # Show all values if reasonable
                    enum_str = "|".join(f'"{v}"' for v in enum_values)
                    param_descriptions.append(f"{param_name}: {param_type}({enum_str})")
                else:  # Just indicate there are allowed values
                    param_descriptions.append(f"{param_name}: {param_type}(allowed values defined)")
            else:
                param_descriptions.append(f"{param_name}: {param_type}")

        params_str = ", ".join(param_descriptions)
        return f"- {tool_name}({params_str}): {desc}"
