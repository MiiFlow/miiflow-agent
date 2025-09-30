"""OpenInference auto-instrumentation setup for Phoenix compatibility."""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def setup_openinference_instrumentation() -> Dict[str, bool]:
    """Setup OpenInference auto-instrumentation for supported providers.

    Returns:
        Dict indicating which instrumentations were successfully setup.
    """
    instrumentation_status = {
        "openai": False,
        "anthropic": False,
    }

    # Setup OpenAI instrumentation
    try:
        from openinference.instrumentation.openai import OpenAIInstrumentor

        # Check if already instrumented
        if not OpenAIInstrumentor().is_instrumented_by_opentelemetry:
            OpenAIInstrumentor().instrument()
            logger.info("OpenAI auto-instrumentation enabled")
        else:
            logger.debug("OpenAI already instrumented")

        instrumentation_status["openai"] = True

    except ImportError:
        logger.debug("OpenInference OpenAI instrumentation not available")
    except Exception as e:
        logger.warning(f"Failed to setup OpenAI instrumentation: {e}")

    # Setup Anthropic instrumentation
    try:
        from openinference.instrumentation.anthropic import AnthropicInstrumentor

        # Check if already instrumented
        if not AnthropicInstrumentor().is_instrumented_by_opentelemetry:
            AnthropicInstrumentor().instrument()
            logger.info("Anthropic auto-instrumentation enabled")
        else:
            logger.debug("Anthropic already instrumented")

        instrumentation_status["anthropic"] = True

    except ImportError:
        logger.debug("OpenInference Anthropic instrumentation not available")
    except Exception as e:
        logger.warning(f"Failed to setup Anthropic instrumentation: {e}")

    return instrumentation_status


def setup_opentelemetry_tracing(endpoint: str = "http://localhost:6006") -> bool:
    """Setup OpenTelemetry tracing to send traces to Phoenix endpoint.

    Args:
        endpoint: Phoenix server endpoint (local or remote)

    Returns:
        True if setup was successful
    """
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk import trace as trace_sdk
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        # Check if tracer provider is already set
        current_provider = trace.get_tracer_provider()
        if hasattr(current_provider, 'add_span_processor'):
            logger.debug("OpenTelemetry tracer provider already configured")
            return True

        # Configure OpenTelemetry to send traces to Phoenix
        tracer_provider = trace_sdk.TracerProvider()

        # Add OTLP exporter for Phoenix
        otlp_exporter = OTLPSpanExporter(
            endpoint=f"{endpoint}/v1/traces"
        )
        tracer_provider.add_span_processor(
            BatchSpanProcessor(otlp_exporter)
        )

        # Set global tracer provider
        trace.set_tracer_provider(tracer_provider)

        logger.info(f"OpenTelemetry configured to send traces to {endpoint}")
        return True

    except ImportError as e:
        logger.warning(f"OpenTelemetry dependencies not available: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to setup OpenTelemetry tracing: {e}")
        return False


def launch_local_phoenix(port: int = 6006) -> Optional[Any]:
    """Launch a local Phoenix session for development.
    
    This should only be used in development environments.

    Args:
        port: Port for Phoenix UI

    Returns:
        Phoenix session object or None if setup failed
    """
    try:
        import phoenix as px
        
        # Launch Phoenix app locally
        session = px.launch_app(port=port)
        logger.info(f"Local Phoenix session started: {session.url}")
        return session

    except ImportError as e:
        logger.warning(f"Phoenix dependencies not available: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to launch local Phoenix: {e}")
        return None


def uninstrument_all() -> None:
    """Uninstrument all OpenInference instrumentations."""
    try:
        from openinference.instrumentation.openai import OpenAIInstrumentor
        OpenAIInstrumentor().uninstrument()
        logger.info("OpenAI instrumentation removed")
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Failed to uninstrument OpenAI: {e}")

    try:
        from openinference.instrumentation.anthropic import AnthropicInstrumentor
        AnthropicInstrumentor().uninstrument()
        logger.info("Anthropic instrumentation removed")
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Failed to uninstrument Anthropic: {e}")


def check_instrumentation_status() -> Dict[str, Dict[str, Any]]:
    """Check the status of all available instrumentations.

    Returns:
        Dictionary with instrumentation status and metadata
    """
    status = {}

    # Check OpenAI instrumentation
    try:
        from openinference.instrumentation.openai import OpenAIInstrumentor
        instrumentor = OpenAIInstrumentor()
        status["openai"] = {
            "available": True,
            "instrumented": instrumentor.is_instrumented_by_opentelemetry,
            "version": getattr(instrumentor, "__version__", "unknown")
        }
    except ImportError:
        status["openai"] = {
            "available": False,
            "instrumented": False,
            "error": "Package not installed"
        }

    # Check Anthropic instrumentation
    try:
        from openinference.instrumentation.anthropic import AnthropicInstrumentor
        instrumentor = AnthropicInstrumentor()
        status["anthropic"] = {
            "available": True,
            "instrumented": instrumentor.is_instrumented_by_opentelemetry,
            "version": getattr(instrumentor, "__version__", "unknown")
        }
    except ImportError:
        status["anthropic"] = {
            "available": False,
            "instrumented": False,
            "error": "Package not installed"
        }

    return status


# Main convenience function for Phoenix tracing setup
def enable_phoenix_tracing(endpoint: str = "http://localhost:6006", launch_local: bool = False) -> bool:
    """Setup complete Phoenix tracing with OpenTelemetry and auto-instrumentation.

    Args:
        endpoint: Phoenix server endpoint
        launch_local: Whether to launch a local Phoenix session (development only)

    Returns:
        True if setup was successful
    """
    success = True
    
    # Launch local Phoenix if requested (development only)
    if launch_local:
        session = launch_local_phoenix()
        if session:
            # Update endpoint to use local session URL
            from urllib.parse import urlparse
            parsed = urlparse(session.url)
            endpoint = f"{parsed.scheme}://{parsed.netloc}"
        else:
            logger.warning("Failed to launch local Phoenix, using provided endpoint")
    
    # Setup OpenTelemetry tracing
    if not setup_opentelemetry_tracing(endpoint):
        success = False
        logger.error("Failed to setup OpenTelemetry tracing")
    
    # Setup auto-instrumentation
    instrumentation_status = setup_openinference_instrumentation()
    enabled_instrumentations = [
        provider for provider, enabled in instrumentation_status.items() if enabled
    ]
    
    if enabled_instrumentations:
        logger.info(f"Auto-instrumentation enabled for: {', '.join(enabled_instrumentations)}")
    else:
        logger.warning("No auto-instrumentations were successfully enabled")
        success = False
    
    return success


if __name__ == "__main__":
    # CLI interface for testing
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "status":
            status = check_instrumentation_status()
            print("OpenInference Instrumentation Status:")
            for provider, info in status.items():
                print(f"  {provider}: {info}")

        elif command == "enable":
            endpoint = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:6006"
            success = enable_phoenix_tracing(endpoint)
            print(f"Phoenix tracing enabled: {success}")

        elif command == "uninstrument":
            uninstrument_all()
            print("All instrumentations removed")

        else:
            print("Available commands: status, enable [endpoint], uninstrument")
    else:
        # Default: show status
        status = check_instrumentation_status()
        print("OpenInference Instrumentation Status:")
        for provider, info in status.items():
            available = "✓" if info["available"] else "✗"
            instrumented = "✓" if info.get("instrumented", False) else "✗"
            print(f"  {provider}: Available {available}, Instrumented {instrumented}")
