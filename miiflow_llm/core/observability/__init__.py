"""Observability and tracing for miiflow-llm with Phoenix integration."""

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
    # Logging utilities
    "get_logger",
    "configure_structured_logging",
]
