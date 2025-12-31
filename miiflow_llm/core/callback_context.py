"""Context variable management for callback context.

This module provides a way to pass context (like organization_id, agent_node_run_id)
through LLM calls to callbacks without changing method signatures.

Usage:
    from miiflow_llm import CallbackContext, callback_context

    ctx = CallbackContext(
        organization_id="org_123",
        agent_node_run_id="run_456",
        metadata={"owner_type": "platform"}
    )

    # Use context manager - all LLM calls within will have this context
    with callback_context(ctx):
        response = await client.achat(messages)
        # Callbacks automatically fire with the context

    # Or set/get manually
    token = set_callback_context(ctx)
    try:
        response = await client.achat(messages)
    finally:
        reset_callback_context(token)
"""

import contextvars
from contextlib import contextmanager
from typing import Optional

from .callbacks import CallbackContext

# Context variable for passing callback context through calls
_callback_context: contextvars.ContextVar[Optional[CallbackContext]] = contextvars.ContextVar(
    "callback_context", default=None
)


def get_callback_context() -> Optional[CallbackContext]:
    """Get the current callback context.

    Returns:
        The current CallbackContext if set, None otherwise.
    """
    return _callback_context.get()


def set_callback_context(context: CallbackContext) -> contextvars.Token:
    """Set the callback context.

    Args:
        context: The CallbackContext to set.

    Returns:
        A token that can be used to reset the context later.
    """
    return _callback_context.set(context)


def reset_callback_context(token: contextvars.Token) -> None:
    """Reset the callback context using the token from set_callback_context.

    Args:
        token: The token returned from set_callback_context.
    """
    _callback_context.reset(token)


@contextmanager
def callback_context(context: CallbackContext):
    """Context manager for setting callback context.

    All LLM calls made within this context manager will have the context
    available to callbacks.

    Usage:
        ctx = CallbackContext(organization_id="org_123", agent_node_run_id="run_456")
        with callback_context(ctx):
            response = await client.achat(messages)
            # Callbacks will receive the context

    Args:
        context: The CallbackContext to set for the duration of the block.

    Yields:
        The context that was set.
    """
    token = set_callback_context(context)
    try:
        yield context
    finally:
        reset_callback_context(token)
