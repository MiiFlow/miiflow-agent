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
        "google_genai": False,
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

    # Setup Google GenAI instrumentation
    try:
        from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor

        # Check if already instrumented
        if not GoogleGenAIInstrumentor().is_instrumented_by_opentelemetry:
            GoogleGenAIInstrumentor().instrument()
            logger.info("Google GenAI auto-instrumentation enabled")
        else:
            logger.debug("Google GenAI already instrumented")

        instrumentation_status["google_genai"] = True

    except ImportError:
        logger.debug("OpenInference Google GenAI instrumentation not available")
    except Exception as e:
        logger.warning(f"Failed to setup Google GenAI instrumentation: {e}")

    return instrumentation_status


def setup_opentelemetry_tracing(config: Optional["ObservabilityConfig"] = None) -> bool:
    """Setup OpenTelemetry tracing to send traces to Phoenix endpoint.

    Args:
        config: ObservabilityConfig instance. If None, loads from environment.

    Returns:
        True if setup was successful
    """
    # Load config from environment if not provided
    if config is None:
        from .config import ObservabilityConfig
        config = ObservabilityConfig.from_env()

    if not config.phoenix_endpoint:
        logger.warning("No Phoenix endpoint configured")
        return False

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

        # Build authentication headers for Phoenix Cloud
        headers = {}
        if config.phoenix_api_key:
            # Phoenix Cloud uses Bearer token authentication
            headers["Authorization"] = f"Bearer {config.phoenix_api_key}"
            logger.debug("Using Phoenix Cloud Bearer token authentication")
        elif config.phoenix_client_headers:
            for pair in config.phoenix_client_headers.split(","):
                if "=" in pair:
                    key, value = pair.strip().split("=", 1)
                    headers[key.strip()] = value.strip()
            logger.debug("Using Phoenix Cloud custom headers authentication")

        # Add OTLP exporter for Phoenix
        exporter_kwargs = {"endpoint": f"{config.phoenix_endpoint}/v1/traces"}
        if headers:
            exporter_kwargs["headers"] = headers

        otlp_exporter = OTLPSpanExporter(**exporter_kwargs)
        tracer_provider.add_span_processor(
            BatchSpanProcessor(otlp_exporter)
        )

        # Set global tracer provider
        trace.set_tracer_provider(tracer_provider)

        deployment_type = "Phoenix Cloud" if config.is_phoenix_cloud() else "local Phoenix"
        logger.info(f"OpenTelemetry configured to send traces to {deployment_type} at {config.phoenix_endpoint}")
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

    try:
        from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
        GoogleGenAIInstrumentor().uninstrument()
        logger.info("Google GenAI instrumentation removed")
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Failed to uninstrument Google GenAI: {e}")


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

    # Check Google GenAI instrumentation
    try:
        from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
        instrumentor = GoogleGenAIInstrumentor()
        status["google_genai"] = {
            "available": True,
            "instrumented": instrumentor.is_instrumented_by_opentelemetry,
            "version": getattr(instrumentor, "__version__", "unknown")
        }
    except ImportError:
        status["google_genai"] = {
            "available": False,
            "instrumented": False,
            "error": "Package not installed"
        }

    return status


# Main convenience function for Phoenix tracing setup
def enable_phoenix_tracing(
    config: Optional["ObservabilityConfig"] = None,
    endpoint: Optional[str] = None,
    launch_local: bool = False
) -> bool:
    """Setup complete Phoenix tracing with OpenTelemetry and auto-instrumentation.

    Args:
        config: ObservabilityConfig instance. If None, loads from environment.
        endpoint: (Deprecated) Phoenix server endpoint. Use config parameter instead.
        launch_local: Whether to launch a local Phoenix session (development only)

    Returns:
        True if setup was successful

    Examples:
        # Auto-detect from environment (recommended)
        enable_phoenix_tracing()

        # Local Phoenix with factory method
        config = ObservabilityConfig.for_local()
        enable_phoenix_tracing(config)

        # Phoenix Cloud with factory method
        config = ObservabilityConfig.for_cloud(
            api_key="your-api-key",
            endpoint="https://your-space.phoenix.arize.com"
        )
        enable_phoenix_tracing(config)

        # Backwards compatible (deprecated)
        enable_phoenix_tracing(endpoint="http://localhost:6006")
    """
    from .config import ObservabilityConfig

    success = True

    # Load config if not provided
    if config is None:
        if endpoint:
            # Backwards compatibility: create config from endpoint parameter
            logger.warning(
                "Passing 'endpoint' parameter is deprecated. "
                "Use ObservabilityConfig.for_local() or for_cloud() instead."
            )
            config = ObservabilityConfig(
                phoenix_enabled=True,
                phoenix_endpoint=endpoint,
                phoenix_project_name="miiflow-agent",
                structured_logging=True,
            )
        else:
            # Load from environment
            config = ObservabilityConfig.from_env()

    # Launch local Phoenix if requested (development only)
    if launch_local:
        session = launch_local_phoenix()
        if session:
            # Update config endpoint to use local session URL
            from urllib.parse import urlparse
            parsed = urlparse(session.url)
            config.phoenix_endpoint = f"{parsed.scheme}://{parsed.netloc}"
        else:
            logger.warning("Failed to launch local Phoenix, using configured endpoint")

    # Setup OpenTelemetry tracing
    if not setup_opentelemetry_tracing(config):
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
