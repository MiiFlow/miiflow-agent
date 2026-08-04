"""Observability and tracing for miiflow-agent with Phoenix integration."""

from .config import ObservabilityConfig
from .context import TraceContext, get_current_trace_context
from .auto_instrumentation import (
    enable_phoenix_tracing,
    check_instrumentation_status,
    setup_openinference_instrumentation,
    setup_opentelemetry_tracing,
    launch_local_phoenix,
    uninstrument_all,
)
from .logging import get_logger, configure_structured_logging
from .spans import agent_span, set_span_attribute, set_span_output, traced_stream

__all__ = [
    # Core configuration
    "ObservabilityConfig",
    # Context management
    "TraceContext",
    "get_current_trace_context",
    # Phoenix tracing setup
    "enable_phoenix_tracing",
    "setup_opentelemetry_tracing",
    "launch_local_phoenix",
    # Instrumentation management
    "setup_openinference_instrumentation",
    "check_instrumentation_status",
    "uninstrument_all",
    # Agent spans
    "agent_span",
    "set_span_attribute",
    "set_span_output",
    "traced_stream",
    # Logging utilities
    "get_logger",
    "configure_structured_logging",
]
