"""Trace context management for observability."""

import contextvars
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from uuid import uuid4


@dataclass
class TraceContext:
    """Context for trace information."""

    trace_id: str = field(default_factory=lambda: str(uuid4()))
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def child_context(self) -> "TraceContext":
        """Create a child context."""
        return TraceContext(
            trace_id=self.trace_id,
            parent_span_id=self.span_id,
            metadata=self.metadata.copy()
        )

    def with_span(self, span_id: str) -> "TraceContext":
        """Create a copy with a new span ID."""
        return TraceContext(
            trace_id=self.trace_id,
            span_id=span_id,
            parent_span_id=self.parent_span_id,
            metadata=self.metadata.copy()
        )

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the context."""
        self.metadata[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "metadata": self.metadata
        }


# Context variable for the current trace context
_trace_context: contextvars.ContextVar[Optional[TraceContext]] = contextvars.ContextVar(
    "trace_context", default=None
)


def get_current_trace_context() -> Optional[TraceContext]:
    """Get the current trace context."""
    return _trace_context.get()


def set_trace_context(context: TraceContext) -> None:
    """Set the trace context."""
    _trace_context.set(context)


def clear_trace_context() -> None:
    """Clear the trace context."""
    _trace_context.set(None)