"""Global callback system for LLM operations.

This module provides a callback registry that allows users to register listeners
for LLM events like token consumption, errors, and agent lifecycle events.

Usage:
    from miiflow_llm import on_post_call, CallbackEvent, CallbackContext, callback_context

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

    # Error information (for ON_ERROR)
    error: Optional[Exception] = None
    error_type: Optional[str] = None

    # Agent information (for AGENT_RUN_START, AGENT_RUN_END)
    agent_type: Optional[str] = None
    query: Optional[str] = None

    # Context from caller
    context: Optional[CallbackContext] = None

    # Success flag
    success: bool = True


# Type alias for callback functions
CallbackFn = Callable[[CallbackEvent], Union[None, Awaitable[None]]]


class CallbackRegistry:
    """Registry for managing LLM callbacks.

    Supports both sync and async callbacks. Callbacks are invoked in the order
    they were registered. Errors in callbacks are logged but don't affect the
    LLM call.

    Usage:
        from miiflow_llm import callbacks

        @callbacks.on_post_call
        async def track_usage(event: CallbackEvent):
            print(f"Used {event.tokens.total_tokens} tokens")

        # Or register programmatically
        callbacks.register(CallbackEventType.POST_CALL, my_callback)

        # Unregister when done
        callbacks.unregister(CallbackEventType.POST_CALL, my_callback)
    """

    def __init__(self):
        self._callbacks: Dict[CallbackEventType, List[CallbackFn]] = {
            event_type: [] for event_type in CallbackEventType
        }
        self._lock = Lock()

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
        """Clear all callbacks, or callbacks for a specific event type."""
        with self._lock:
            if event_type:
                self._callbacks[event_type] = []
            else:
                for et in CallbackEventType:
                    self._callbacks[et] = []

    def get_callbacks(self, event_type: CallbackEventType) -> List[CallbackFn]:
        """Get all registered callbacks for an event type."""
        with self._lock:
            return self._callbacks[event_type].copy()

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
        or scheduled on the existing loop.
        """
        callbacks = self.get_callbacks(event.event_type)

        for callback in callbacks:
            try:
                result = callback(event)
                if asyncio.iscoroutine(result):
                    try:
                        loop = asyncio.get_running_loop()
                        # If there's a running loop, create a task
                        loop.create_task(result)
                    except RuntimeError:
                        # No running loop, run synchronously
                        asyncio.run(result)
            except Exception as e:
                logger.error(
                    f"Error in callback {callback.__name__} for {event.event_type.value}: {e}",
                    exc_info=True,
                )


# Global registry instance
_global_registry = CallbackRegistry()


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
