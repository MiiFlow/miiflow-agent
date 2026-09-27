"""What a provider client actually sent, for the call's POST_CALL event.

The request a provider client puts on the wire is not the request the caller
built: the client adds native MCP connectors, a tool-search tool, deferral
flags, cache breakpoints, and on some retries strips deferral again. Those
additions are billed as prompt tokens and none of them is visible in the
caller's messages or tool list — so a prompt that is 200K tokens larger than
anything recorded cannot be explained after the fact.

`ModelClient` opens a slot per call (`open_slot`), the provider client writes
the shape of its final request into it (`record`), and the POST_CALL event
carries it (`CallbackEvent.wire_shape`). The slot is a plain dict held in a
ContextVar: provider streams may be advanced from a copied context (an
inactivity guard's task), and mutating the shared dict — rather than
re-setting the variable — is what stays visible to the caller either way.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Dict, Optional

_SLOT: ContextVar[Optional[Dict[str, Any]]] = ContextVar("miiflow_wire_shape", default=None)


def open_slot() -> Dict[str, Any]:
    """Start recording for one provider call; returns the dict to read after."""
    slot: Dict[str, Any] = {}
    _SLOT.set(slot)
    return slot


def record(**fields: Any) -> None:
    """Merge `fields` into the current call's slot (no-op outside a call)."""
    slot = _SLOT.get()
    if slot is not None:
        slot.update(fields)
