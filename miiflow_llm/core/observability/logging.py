"""Simplified structured logging with trace correlation."""

import logging
import sys
import structlog
from typing import Any, Dict, Optional
from contextvars import ContextVar

from .config import ObservabilityConfig
from .context import get_current_trace_context


# Context variable for additional logging context
_logging_context: ContextVar[Dict[str, Any]] = ContextVar("logging_context", default={})


def get_logging_context() -> Dict[str, Any]:
    """Get the current logging context."""
    return _logging_context.get({})


def set_logging_context(context: Dict[str, Any]) -> None:
    """Set the logging context."""
    _logging_context.set(context)


def add_to_logging_context(key: str, value: Any) -> None:
    """Add a key-value pair to the logging context."""
    context = get_logging_context().copy()
    context[key] = value
    set_logging_context(context)


def clear_logging_context() -> None:
    """Clear the logging context."""
    _logging_context.set({})


def add_trace_correlation(logger, method_name, event_dict):
    """Add trace correlation to log events."""
    # Get current trace context
    trace_context = get_current_trace_context()
    if trace_context:
        event_dict["trace_id"] = trace_context.trace_id
        event_dict["span_id"] = trace_context.span_id
        if trace_context.parent_span_id:
            event_dict["parent_span_id"] = trace_context.parent_span_id

    # Add logging context
    logging_context = get_logging_context()
    if logging_context:
        event_dict.update(logging_context)

    return event_dict


def add_miiflow_context(logger, method_name, event_dict):
    """Add miiflow-specific context to log events."""
    event_dict["component"] = "miiflow-llm"
    
    # Try to get version dynamically
    try:
        import miiflow_llm
        event_dict["version"] = getattr(miiflow_llm, "__version__", "unknown")
    except (ImportError, AttributeError):
        event_dict["version"] = "unknown"
    
    return event_dict


def configure_structured_logging(
    config: Optional[ObservabilityConfig] = None,
    force_configuration: bool = False
) -> bool:
    """Configure structured logging for miiflow-llm.
    
    Args:
        config: Observability configuration
        force_configuration: If True, configure even if already configured
        
    Returns:
        True if configuration was successful
    """
    if config is None:
        config = ObservabilityConfig.from_env()

    if not config.structured_logging:
        # Use basic logging
        if force_configuration or not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        return True

    try:
        # Check if structlog is already configured
        if not force_configuration and structlog.is_configured():
            return True

        # Configure structlog
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            add_miiflow_context,
            add_trace_correlation,
        ]

        # Add appropriate renderer based on environment
        if sys.stderr.isatty():
            # Development: use colorized console output
            processors.append(structlog.dev.ConsoleRenderer())
        else:
            # Production: use JSON output
            processors.append(structlog.processors.JSONRenderer())

        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        # Configure standard library logging
        if force_configuration or not logging.getLogger().handlers:
            logging.basicConfig(
                format="%(message)s",
                stream=sys.stdout,
                level=logging.INFO,
            )

        return True

    except Exception as e:
        # Fallback to basic logging if structured logging fails
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        logging.getLogger(__name__).warning(f"Failed to configure structured logging: {e}")
        return False


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger with trace correlation.
    
    This function ensures logging is configured before returning a logger.
    """
    # Ensure logging is configured (non-intrusive)
    configure_structured_logging()
    
    logger = structlog.get_logger(name)
    
    # Add trace context if available
    trace_context = get_current_trace_context()
    if trace_context:
        logger = logger.bind(
            trace_id=trace_context.trace_id,
            span_id=trace_context.span_id,
        )
    
    return logger
