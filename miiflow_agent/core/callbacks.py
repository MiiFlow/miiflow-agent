"""Global callback system for LLM operations.

This module provides a callback registry that allows users to register listeners
for LLM events like token consumption, errors, and agent lifecycle events.

Usage:
    from miiflow_agent import on_post_call, CallbackEvent, CallbackContext, callback_context

    # Register a callback
    @on_post_call
    async def track_usage(event: CallbackEvent):
        print(f"Used {event.tokens.total_tokens} tokens")

    # Set context for billing
    ctx = CallbackContext(organization_id="org_123", agent_node_run_id="run_456")
    with callback_context(ctx):
        response = await client.achat(messages)
        # Callback fires with context attached
"""

import asyncio
import contextlib
import contextvars
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from threading import Lock
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from .metrics import TokenCount

logger = logging.getLogger(__name__)


class CallbackEventType(Enum):
    """Types of callback events."""

    POST_CALL = "post_call"  # After successful LLM call (with usage data)
    ON_ERROR = "on_error"  # On LLM call error
    AGENT_RUN_START = "agent_run_start"  # When agent execution begins
    AGENT_RUN_END = "agent_run_end"  # When agent execution completes
    TOOL_EXECUTED = "tool_executed"  # After a tool is executed (with tool info)
    PRE_TOOL_USE = "pre_tool_use"  # Before a tool is executed (can block/modify)
    POST_TOOL_USE = "post_tool_use"  # After a tool is executed (can transform result)


@dataclass
class CallbackContext:
    """Context passed through LLM calls to callbacks.

    Allows passing arbitrary metadata that will be available in callbacks.
    This is typically set using the callback_context context manager.
    """

    organization_id: Optional[str] = None
    agent_node_run_id: Optional[str] = None
    assistant_id: Optional[str] = None
    thread_id: Optional[str] = None
    user_id: Optional[str] = None
    # Arbitrary metadata dict for extensibility
    metadata: Dict[str, Any] = field(default_factory=dict)

    def with_metadata(self, **kwargs) -> "CallbackContext":
        """Create a copy with additional metadata."""
        new_metadata = {**self.metadata, **kwargs}
        return CallbackContext(
            organization_id=self.organization_id,
            agent_node_run_id=self.agent_node_run_id,
            assistant_id=self.assistant_id,
            thread_id=self.thread_id,
            user_id=self.user_id,
            metadata=new_metadata,
        )


@dataclass
class CallbackEvent:
    """Event data passed to callbacks."""

    event_type: CallbackEventType
    timestamp: datetime = field(default_factory=datetime.now)

    # LLM call information
    provider: Optional[str] = None
    model: Optional[str] = None

    # Usage information (for POST_CALL)
    tokens: Optional[TokenCount] = None
    latency_ms: Optional[float] = None

    # Streaming latency decomposition (for POST_CALL / ON_ERROR on the
    # astream path). latency_ms alone cannot tell "our request assembly is
    # slow" from "the provider queued us" from "the answer was long":
    #   request_build_ms — message normalization + tool formatting, local
    #   ttft_ms          — last stream-open → first chunk (network + server;
    #                      excludes transport-retry backoff)
    #   stream_ms        — first chunk → stream close (token generation)
    #   transport_retries — stream-open attempts that failed before a chunk
    request_build_ms: Optional[float] = None
    ttft_ms: Optional[float] = None
    stream_ms: Optional[float] = None
    transport_retries: int = 0

    # Short order-sensitive digest of the formatted tools array sent with the
    # call (None when tool-less). The tools array is the FIRST prompt-cache
    # tier, so two consecutive calls whose fingerprints differ have, by
    # construction, busted every cache tier — this is the diagnostic for
    # "cache missed although the previous turn was minutes ago". Computed
    # before provider-side additions (native MCP toolsets, search tool), so
    # an unchanged fingerprint with a cache miss points at those instead.
    tools_fingerprint: Optional[str] = None

    # Error information (for ON_ERROR)
    error: Optional[Exception] = None
    error_type: Optional[str] = None

    # Agent information (for AGENT_RUN_START, AGENT_RUN_END)
    agent_type: Optional[str] = None
    query: Optional[str] = None

    # Tool execution information (for TOOL_EXECUTED, PRE_TOOL_USE, POST_TOOL_USE)
    tool_name: Optional[str] = None
    tool_inputs: Optional[Dict[str, Any]] = None
    tool_output: Optional[Any] = None
    tool_execution_time_ms: Optional[float] = None
    tool_id: Optional[str] = None

    # Context from caller
    context: Optional[CallbackContext] = None

    # Success flag
    success: bool = True

    # PRE_TOOL_USE: set to True in callback to block tool execution
    blocked: bool = False
    block_reason: Optional[str] = None

    # PRE_TOOL_USE: set to reject the call's INPUTS as invalid. Unlike
    # ``blocked`` (which raises ToolApprovalRequired → a user approval modal),
    # a validation error returns a FAILED ToolResult so the model fixes the
    # inputs and retries — the human is never asked to approve a call that
    # would bounce off the API anyway.
    validation_error: Optional[str] = None

    # PRE_TOOL_USE: set this in callback to REPLACE the tool inputs before
    # execution (e.g. a user approved a tool but edited its arguments). The
    # executor reads ``inputs_override`` back after emitting the PRE event and
    # runs the tool with those values instead of the model's original inputs.
    inputs_override: Optional[Dict[str, Any]] = None
    inputs_overridden: bool = False

    # POST_TOOL_USE: set this in callback to replace the tool output
    transformed_output: Optional[Any] = None
    output_transformed: bool = False


# Type alias for callback functions
CallbackFn = Callable[[CallbackEvent], Union[None, Awaitable[None]]]


class CallbackRegistry:
    """Registry for managing LLM callbacks.

    Supports both sync and async callbacks. Callbacks are invoked in the order
    they were registered. Errors in callbacks are logged but don't affect the
    LLM call.

    Usage:
        from miiflow_agent import callbacks

        @callbacks.on_post_call
        async def track_usage(event: CallbackEvent):
            print(f"Used {event.tokens.total_tokens} tokens")

        # Or register programmatically
        callbacks.register(CallbackEventType.POST_CALL, my_callback)

        # Unregister when done
        callbacks.unregister(CallbackEventType.POST_CALL, my_callback)
    """

    def __init__(self, parent: Optional["CallbackRegistry"] = None):
        self._callbacks: Dict[CallbackEventType, List[CallbackFn]] = {
            event_type: [] for event_type in CallbackEventType
        }
        self._lock = Lock()
        # A scoped registry chains to its parent (the enclosing scope, or the
        # global one): emissions reach the parent's listeners too, resolved
        # at emit time so late registrations on the parent are still seen.
        # Registration and clear() only ever touch THIS registry — a scoped
        # clear cannot wipe process-wide billing/telemetry callbacks.
        self._parent = parent
        # Event types this scope has SHADOWED from its parent chain via
        # clear(). Shadowing is how a nested run replaces an inherited
        # policy instead of stacking on top of it: a dispatched child that
        # clears PRE_TOOL_USE and registers its own approval gate must not
        # have the parent's gate fire for its tools — while its POST_CALL
        # events still reach the parent's token tracking (child LLM cost
        # bills to the parent's turn). Meaningless without a parent.
        self._shadowed: set = set()

    def register(self, event_type: CallbackEventType, callback: CallbackFn) -> None:
        """Register a callback for an event type."""
        with self._lock:
            if callback not in self._callbacks[event_type]:
                self._callbacks[event_type].append(callback)
                logger.debug(f"Registered callback {callback.__name__} for {event_type.value}")

    def unregister(self, event_type: CallbackEventType, callback: CallbackFn) -> bool:
        """Unregister a callback. Returns True if it was registered."""
        with self._lock:
            if callback in self._callbacks[event_type]:
                self._callbacks[event_type].remove(callback)
                logger.debug(f"Unregistered callback {callback.__name__} for {event_type.value}")
                return True
            return False

    def clear(self, event_type: Optional[CallbackEventType] = None) -> None:
        """Clear callbacks for one event type, or all of them.

        On a parented (scoped) registry, clearing also SHADOWS the cleared
        type(s): the parent chain stops contributing listeners for them.
        "Clear then register my own" therefore means the same thing inside a
        scope that it always meant on the global registry — replace the
        policy — instead of leaving the inherited one active underneath.
        """
        with self._lock:
            if event_type:
                self._callbacks[event_type] = []
                if self._parent is not None:
                    self._shadowed.add(event_type)
            else:
                for et in CallbackEventType:
                    self._callbacks[et] = []
                if self._parent is not None:
                    self._shadowed.update(CallbackEventType)

    def get_callbacks(self, event_type: CallbackEventType) -> List[CallbackFn]:
        """Get all registered callbacks for an event type.

        Parent listeners run first (process-wide policy before run-local
        additions), then this registry's own. Types this scope shadowed via
        clear() skip the parent chain entirely.
        """
        with self._lock:
            own = self._callbacks[event_type].copy()
            shadowed = event_type in self._shadowed
        inherited: List[CallbackFn] = (
            self._parent.get_callbacks(event_type)
            if self._parent is not None and not shadowed
            else []
        )
        return inherited + [cb for cb in own if cb not in inherited]

    @contextlib.contextmanager
    def isolated_scope(self):
        """Snapshot the registry, run, then restore.

        Use this around a NESTED agent run (e.g. dispatch_assistant invoking
        a sub-assistant) so callback churn inside the nested run cannot leak
        out. Specifically prevents two failure modes:

          - Inner code calling `clear()` (typical of ToolCollector setup)
            wiping the outer caller's approval / billing callbacks.
          - Inner code registering callbacks that linger after the nested
            run completes and pollute the outer caller's continuation.

        The snapshot is shallow — callback functions themselves are not
        copied — but the per-event lists are restored exactly as they were
        at scope entry. Thread-safe via the registry's internal lock.

        Usage:
            with get_global_registry().isolated_scope():
                async for event in nested_run():
                    yield event
        """
        with self._lock:
            snapshot = {
                event_type: list(callbacks)
                for event_type, callbacks in self._callbacks.items()
            }
        try:
            yield
        finally:
            with self._lock:
                self._callbacks = {
                    event_type: list(callbacks)
                    for event_type, callbacks in snapshot.items()
                }

    async def emit(self, event: CallbackEvent) -> None:
        """Emit an event to all registered callbacks.

        Callbacks are invoked in order. Errors are logged but don't propagate.
        Async callbacks are awaited; sync callbacks are called directly.
        """
        callbacks = self.get_callbacks(event.event_type)

        for callback in callbacks:
            try:
                result = callback(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(
                    f"Error in callback {callback.__name__} for {event.event_type.value}: {e}",
                    exc_info=True,
                )

    def emit_sync(self, event: CallbackEvent) -> None:
        """Emit an event synchronously (for sync LLM calls).

        Async callbacks are run via asyncio.run() if no event loop is running,
        or scheduled on the existing loop. Scheduled tasks are held by strong
        reference until done — a bare ``create_task`` result can be garbage
        collected mid-flight, which silently dropped billing events emitted
        from streaming ``finally`` blocks. (A task still in flight when the
        loop itself shuts down remains best-effort; callers on the async path
        should prefer ``await emit()``.)
        """
        callbacks = self.get_callbacks(event.event_type)

        for callback in callbacks:
            try:
                result = callback(event)
                if asyncio.iscoroutine(result):
                    try:
                        loop = asyncio.get_running_loop()
                        task = loop.create_task(result)
                        _background_emit_tasks.add(task)
                        task.add_done_callback(_reap_background_emit_task)
                    except RuntimeError:
                        # No running loop, run synchronously
                        asyncio.run(result)
            except Exception as e:
                logger.error(
                    f"Error in callback {callback.__name__} for {event.event_type.value}: {e}",
                    exc_info=True,
                )


# Strong references to tasks spawned by emit_sync — without these, a task
# whose only reference is the create_task return value can be GC'd before it
# runs (documented CPython behavior), losing the event.
_background_emit_tasks: set = set()


def _reap_background_emit_task(task: "asyncio.Task") -> None:
    _background_emit_tasks.discard(task)
    if not task.cancelled() and task.exception() is not None:
        logger.error("Background callback task failed", exc_info=task.exception())


# Global registry instance
_global_registry = CallbackRegistry()

# Run-scoped registry override. Set via scoped_callbacks(); resolved by
# get_active_registry() at every emission site. ContextVar-based, so
# concurrent ASGI requests (and parallel tool branches, which run in their
# own Context copies) each see only their own scope — registering a
# run-local approval gate no longer makes it visible to every other request
# in the process.
_active_registry_var: "contextvars.ContextVar[Optional[CallbackRegistry]]" = (
    contextvars.ContextVar("miiflow_active_callback_registry", default=None)
)


def get_active_registry() -> CallbackRegistry:
    """The registry emission sites should use: the innermost scoped registry
    when one is active on this Context, else the global one."""
    return _active_registry_var.get() or _global_registry


@contextlib.contextmanager
def scoped_callbacks(registry: Optional[CallbackRegistry] = None):
    """Activate a run-scoped callback registry on the current Context.

    Yields a registry whose registrations are visible only inside this scope
    (and in tasks forked from it), while emissions still reach the enclosing
    listeners via the parent chain. This is the concurrency-safe alternative
    to registering run-specific callbacks on the global registry and relying
    on isolated_scope() — which protects nesting, not concurrent requests.

    The parent is the *currently active* registry, not the global one, so
    scopes nest: a dispatched child that enters its own scope inherits the
    parent turn's listeners (token tracking keeps billing child LLM calls to
    the parent turn) and can shadow the ones it replaces via ``clear()``
    (the parent's approval gate must not fire for the child's tools).

        with scoped_callbacks() as cbs:
            cbs.register(CallbackEventType.PRE_TOOL_USE, approval_gate)
            result = await agent.run(query)
    """
    scoped = registry or CallbackRegistry(parent=get_active_registry())
    token = _active_registry_var.set(scoped)
    try:
        yield scoped
    finally:
        _active_registry_var.reset(token)


async def scoped_callbacks_stream(source, registry: Optional[CallbackRegistry] = None):
    """Drive async generator ``source`` with a callback scope active during
    each advancement.

    The safe form of ``with scoped_callbacks(): async for ... yield`` inside
    an async generator. That shape sets the ContextVar during the first
    ``__anext__`` and resets it during a later one — and contextvars are
    per-TASK, not per-generator, so a stream advanced from a different task
    (a pump task, an ASGI server handing off the response body) resets a
    token created in another Context (ValueError) or leaves the scope active
    on the wrong task. Same failure shape as the 2026-08-04 orphaned-span
    incident; see ``observability.traced_stream``, whose mechanics this
    mirrors: the registry is activated and deactivated INSIDE each
    advancement, in the same call frame, so nothing straddles a suspension.

    Registrations made while the body runs land on the scoped registry;
    between yields (and in the consumer's frames) the previous registry is
    active. Tasks forked while the body runs snapshot the scope via their
    Context copy. Pass ``registry`` to activate a caller-owned registry —
    e.g. one the caller registered listeners on before the stream started.
    """
    scoped = (
        registry
        if registry is not None
        else CallbackRegistry(parent=get_active_registry())
    )
    iterator = source.__aiter__()
    try:
        while True:
            token = _active_registry_var.set(scoped)
            try:
                item = await iterator.__anext__()
            except StopAsyncIteration:
                break
            finally:
                # Same call frame as the set(), so this can never run in a
                # different task than the one that set it.
                _active_registry_var.reset(token)
            yield item
    finally:
        # Deterministic close of the inner generator (its finally blocks may
        # hold real resources — usage sessions, unsubscribes) even when the
        # consumer abandons this wrapper early. The scope must be ACTIVE for
        # the close: inner finally blocks emit real events — notably
        # LLMClient.astream_chat's POST_CALL/ON_ERROR with the partial usage
        # of an interrupted call — and a scope-less aclose would route them
        # to the global registry, past every run-scoped listener (billing).
        aclose = getattr(iterator, "aclose", None)
        if aclose is not None:
            token = _active_registry_var.set(scoped)
            try:
                await aclose()
            except Exception:  # noqa: BLE001 — already unwinding
                pass
            finally:
                _active_registry_var.reset(token)


# Convenience functions for global registry
def register(event_type: CallbackEventType, callback: CallbackFn) -> None:
    """Register a callback with the global registry."""
    _global_registry.register(event_type, callback)


def unregister(event_type: CallbackEventType, callback: CallbackFn) -> bool:
    """Unregister a callback from the global registry."""
    return _global_registry.unregister(event_type, callback)


def clear(event_type: Optional[CallbackEventType] = None) -> None:
    """Clear callbacks from the global registry."""
    _global_registry.clear(event_type)


def get_global_registry() -> CallbackRegistry:
    """Get the global callback registry."""
    return _global_registry


# Decorator factories for registering callbacks
def on_post_call(callback: CallbackFn) -> CallbackFn:
    """Decorator to register a POST_CALL callback."""
    register(CallbackEventType.POST_CALL, callback)
    return callback


def on_error(callback: CallbackFn) -> CallbackFn:
    """Decorator to register an ON_ERROR callback."""
    register(CallbackEventType.ON_ERROR, callback)
    return callback


def on_agent_run_start(callback: CallbackFn) -> CallbackFn:
    """Decorator to register an AGENT_RUN_START callback."""
    register(CallbackEventType.AGENT_RUN_START, callback)
    return callback


def on_agent_run_end(callback: CallbackFn) -> CallbackFn:
    """Decorator to register an AGENT_RUN_END callback."""
    register(CallbackEventType.AGENT_RUN_END, callback)
    return callback


def on_tool_executed(callback: CallbackFn) -> CallbackFn:
    """Decorator to register a TOOL_EXECUTED callback."""
    register(CallbackEventType.TOOL_EXECUTED, callback)
    return callback


def on_pre_tool_use(callback: CallbackFn) -> CallbackFn:
    """Decorator to register a PRE_TOOL_USE callback.

    PRE_TOOL_USE callbacks fire before a tool is executed. They can:
    - Inspect tool_name and tool_inputs
    - Block execution by setting event.blocked = True and event.block_reason
    - Modify tool_inputs before execution

    Example:
        @on_pre_tool_use
        async def approve_dangerous_tools(event: CallbackEvent):
            if event.tool_name in DANGEROUS_TOOLS:
                event.blocked = True
                event.block_reason = "Tool requires approval"
    """
    register(CallbackEventType.PRE_TOOL_USE, callback)
    return callback


def on_post_tool_use(callback: CallbackFn) -> CallbackFn:
    """Decorator to register a POST_TOOL_USE callback.

    POST_TOOL_USE callbacks fire after a tool has executed. They can:
    - Inspect tool_output
    - Transform the output by setting event.transformed_output and event.output_transformed = True
    - Enrich output with additional metadata (e.g., source references for citations)

    Example:
        @on_post_tool_use
        async def enrich_kb_results(event: CallbackEvent):
            if event.tool_name.startswith("retrieve_data_from_"):
                event.transformed_output = add_source_metadata(event.tool_output)
                event.output_transformed = True
    """
    register(CallbackEventType.POST_TOOL_USE, callback)
    return callback
