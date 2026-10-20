"""Tests for cleaned-up observability features."""

import importlib.metadata

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from miiflow_agent.core.observability.config import ObservabilityConfig
from miiflow_agent.core.observability.context import TraceContext, get_current_trace_context, set_trace_context
from miiflow_agent.core.observability.auto_instrumentation import (
    setup_openinference_instrumentation,
    setup_opentelemetry_tracing,
    check_instrumentation_status,
    enable_phoenix_tracing,
)
from miiflow_agent.core.observability.logging import get_logger, configure_structured_logging


class TestObservabilityConfig:
    """Test observability configuration."""

    def test_config_from_env_disabled(self, monkeypatch):
        """Test configuration when observability is disabled."""
        monkeypatch.setenv("PHOENIX_ENABLED", "false")
        monkeypatch.setenv("STRUCTURED_LOGGING", "false")

        config = ObservabilityConfig.from_env()

        assert not config.phoenix_enabled
        assert not config.structured_logging

    def test_config_from_env_enabled(self, monkeypatch):
        """Test configuration when observability is enabled."""
        monkeypatch.delenv("PHOENIX_COLLECTOR_ENDPOINT", raising=False)
        monkeypatch.delenv("PHOENIX_API_KEY", raising=False)
        monkeypatch.setenv("PHOENIX_ENABLED", "true")
        monkeypatch.setenv("PHOENIX_ENDPOINT", "http://localhost:6006")
        monkeypatch.setenv("PHOENIX_PROJECT_NAME", "test-project")
        monkeypatch.setenv("STRUCTURED_LOGGING", "true")

        config = ObservabilityConfig.from_env()

        assert config.phoenix_enabled
        assert config.phoenix_endpoint == "http://localhost:6006"
        assert config.phoenix_project_name == "test-project"
        assert config.structured_logging
        assert not config.is_phoenix_cloud()

    def test_config_from_env_phoenix_cloud(self, monkeypatch):
        """Test configuration for Phoenix Cloud from environment."""
        monkeypatch.setenv("PHOENIX_ENABLED", "true")
        monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "https://my-space.phoenix.arize.com")
        monkeypatch.setenv("PHOENIX_API_KEY", "test-api-key")
        monkeypatch.setenv("PHOENIX_PROJECT_NAME", "cloud-project")

        config = ObservabilityConfig.from_env()

        assert config.phoenix_enabled
        assert config.phoenix_endpoint == "https://my-space.phoenix.arize.com"
        assert config.phoenix_api_key == "test-api-key"
        assert config.phoenix_project_name == "cloud-project"
        assert config.is_phoenix_cloud()

    def test_config_from_env_old_cloud_instance(self, monkeypatch):
        """Test configuration for old Phoenix Cloud instance with client headers."""
        monkeypatch.setenv("PHOENIX_ENABLED", "true")
        monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "https://my-space.phoenix.arize.com")
        monkeypatch.setenv("PHOENIX_API_KEY", "test-api-key")
        monkeypatch.setenv("PHOENIX_CLIENT_HEADERS", "api_key=test-api-key")

        config = ObservabilityConfig.from_env()

        assert config.phoenix_enabled
        assert config.phoenix_api_key == "test-api-key"
        assert config.phoenix_client_headers == "api_key=test-api-key"
        assert config.is_phoenix_cloud()

    def test_config_factory_for_local(self):
        """Test factory method for local Phoenix."""
        config = ObservabilityConfig.for_local("my-project")

        assert config.phoenix_enabled
        assert config.phoenix_endpoint == "http://localhost:6006"
        assert config.phoenix_project_name == "my-project"
        assert not config.is_phoenix_cloud()

    def test_config_factory_for_cloud(self):
        """Test factory method for Phoenix Cloud."""
        config = ObservabilityConfig.for_cloud(
            api_key="cloud-api-key",
            endpoint="https://cloud.phoenix.arize.com",
            project_name="cloud-project"
        )

        assert config.phoenix_enabled
        assert config.phoenix_endpoint == "https://cloud.phoenix.arize.com"
        assert config.phoenix_api_key == "cloud-api-key"
        assert config.phoenix_project_name == "cloud-project"
        assert config.is_phoenix_cloud()

    def test_config_factory_for_cloud_with_headers(self):
        """Test factory method for old Phoenix Cloud with client headers."""
        config = ObservabilityConfig.for_cloud(
            api_key="cloud-api-key",
            endpoint="https://cloud.phoenix.arize.com",
            client_headers="api_key=cloud-api-key"
        )

        assert config.phoenix_enabled
        assert config.phoenix_api_key == "cloud-api-key"
        assert config.phoenix_client_headers == "api_key=cloud-api-key"
        assert config.is_phoenix_cloud()

    def test_config_validation_invalid_endpoint(self):
        """Test configuration validation with invalid endpoint."""
        with pytest.raises(ValueError):
            ObservabilityConfig(
                phoenix_enabled=True,
                phoenix_endpoint="invalid-url"
            )

    def test_config_validation_valid(self):
        """Test configuration validation with valid settings."""
        config = ObservabilityConfig(
            phoenix_enabled=True,
            phoenix_endpoint="http://localhost:6006"
        )
        assert config.is_valid()

    def test_config_validation_phoenix_disabled(self):
        """Test configuration validation when Phoenix is disabled."""
        config = ObservabilityConfig(
            phoenix_enabled=False,
            phoenix_endpoint=None
        )
        assert config.is_valid()


class TestTraceContext:
    """Test trace context management."""

    def test_trace_context_creation(self):
        """Test creating a trace context."""
        context = TraceContext()

        assert context.trace_id is not None
        assert context.span_id is None
        assert context.parent_span_id is None
        assert isinstance(context.metadata, dict)

    def test_trace_context_child(self):
        """Test creating a child context."""
        parent = TraceContext()
        parent.span_id = "parent-span"

        child = parent.child_context()

        assert child.trace_id == parent.trace_id
        assert child.parent_span_id == parent.span_id
        assert child.span_id is None  # New span ID will be set later

    def test_trace_context_with_span(self):
        """Test creating context with span ID."""
        context = TraceContext()
        new_context = context.with_span("new-span-id")

        assert new_context.trace_id == context.trace_id
        assert new_context.span_id == "new-span-id"

    def test_trace_context_metadata(self):
        """Test adding metadata to context."""
        context = TraceContext()
        context.add_metadata("key", "value")

        assert context.metadata["key"] == "value"

    def test_context_variables(self):
        """Test context variable management."""
        context = TraceContext()

        # Initially no context
        assert get_current_trace_context() is None

        # Set context
        set_trace_context(context)
        assert get_current_trace_context() == context


def _attaching_instrumentor():
    """A mock that behaves like BaseInstrumentor: instrument() attaches it."""
    instrumentor = Mock()
    instrumentor.is_instrumented_by_opentelemetry = False
    instrumentor.instrumentation_dependencies.return_value = ()

    def _instrument(**_kwargs):
        instrumentor.is_instrumented_by_opentelemetry = True

    instrumentor.instrument.side_effect = _instrument
    return instrumentor


class TestAutoInstrumentation:
    """Test auto-instrumentation functionality."""

    def test_an_instrumentor_that_declines_to_attach_is_not_reported_enabled(self, caplog):
        # BaseInstrumentor.instrument() returns without raising on a
        # DependencyConflict and leaves the instrumentor unattached.
        declining = Mock()
        declining.is_instrumented_by_opentelemetry = False
        declining.instrumentation_dependencies.return_value = ()
        mock_module = Mock()
        mock_module.AnthropicInstrumentor.return_value = declining

        with patch.dict('sys.modules', {'openinference.instrumentation.anthropic': mock_module}):
            result = setup_openinference_instrumentation()

        assert result["anthropic"] is False
        assert "declined to attach" in caplog.text

    def test_an_import_error_inside_instrument_is_a_failure_not_a_missing_package(self, caplog):
        # anthropic 1.x vs instrumentor 0.1.20: instrument() itself imports
        # anthropic.resources.completions and raises ModuleNotFoundError.
        broken = Mock()
        broken.is_instrumented_by_opentelemetry = False
        broken.instrumentation_dependencies.return_value = ()
        broken.instrument.side_effect = ModuleNotFoundError(
            "No module named 'anthropic.resources.completions'"
        )
        mock_module = Mock()
        mock_module.AnthropicInstrumentor.return_value = broken

        with patch.dict('sys.modules', {'openinference.instrumentation.anthropic': mock_module}):
            with caplog.at_level("WARNING"):
                result = setup_openinference_instrumentation()

        assert result["anthropic"] is False
        assert "Failed to setup Anthropic instrumentation" in caplog.text

    @pytest.mark.parametrize(
        "requirement, found, level",
        [
            ("an-sdk-nobody-installed >= 1.0", None, "DEBUG"),
            ("pytest >= 999.0", f'("pytest {importlib.metadata.version("pytest")}")', "WARNING"),
        ],
    )
    def test_a_dependency_conflict_is_skipped_and_only_a_real_one_warns(
        self, caplog, requirement, found, level
    ):
        instrumentor = _attaching_instrumentor()
        instrumentor.instrumentation_dependencies.return_value = (requirement,)
        mock_module = Mock()
        mock_module.AnthropicInstrumentor.return_value = instrumentor

        with patch.dict('sys.modules', {'openinference.instrumentation.anthropic': mock_module}):
            with caplog.at_level("DEBUG"):
                result = setup_openinference_instrumentation()

        assert result["anthropic"] is False
        instrumentor.instrument.assert_not_called()
        record = next(r for r in caplog.records if "Anthropic instrumentation skipped" in r.getMessage())
        assert record.levelname == level
        if found:
            assert found in record.getMessage()

    @pytest.mark.parametrize(
        "module_path, class_name, sdk",
        [
            ("openinference.instrumentation.openai", "OpenAIInstrumentor", "openai"),
            ("openinference.instrumentation.anthropic", "AnthropicInstrumentor", "anthropic"),
            ("openinference.instrumentation.google_genai", "GoogleGenAIInstrumentor", "google.genai"),
        ],
    )
    def test_the_installed_instrumentor_attaches_to_the_installed_sdk(self, module_path, class_name, sdk):
        """Unmocked: catches an SDK bump the locked instrumentor cannot wrap."""
        pytest.importorskip(sdk)
        module = pytest.importorskip(module_path)
        from opentelemetry.sdk.trace import TracerProvider

        instrumentor = getattr(module, class_name)()
        if instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()
        try:
            instrumentor.instrument(tracer_provider=TracerProvider())
            assert instrumentor.is_instrumented_by_opentelemetry
        finally:
            if instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.uninstrument()

    def test_setup_openinference_instrumentation_openai(self):
        """Test OpenAI instrumentation setup."""
        # Mock the OpenAI instrumentor module
        mock_instrumentor = _attaching_instrumentor()

        mock_module = Mock()
        mock_module.OpenAIInstrumentor.return_value = mock_instrumentor

        with patch.dict('sys.modules', {'openinference.instrumentation.openai': mock_module}):
            result = setup_openinference_instrumentation()

            assert result["openai"] is True
            mock_instrumentor.instrument.assert_called_once()

    def test_setup_openinference_instrumentation_anthropic(self):
        """Test Anthropic instrumentation setup."""
        # Mock the Anthropic instrumentor module
        mock_instrumentor = _attaching_instrumentor()

        mock_module = Mock()
        mock_module.AnthropicInstrumentor.return_value = mock_instrumentor

        with patch.dict('sys.modules', {'openinference.instrumentation.anthropic': mock_module}):
            result = setup_openinference_instrumentation()

            assert result["anthropic"] is True
            mock_instrumentor.instrument.assert_called_once()

    def test_setup_openinference_instrumentation_google_genai(self):
        """Test Google GenAI instrumentation setup."""
        # Mock the Google GenAI instrumentor module
        mock_instrumentor = _attaching_instrumentor()

        mock_module = Mock()
        mock_module.GoogleGenAIInstrumentor.return_value = mock_instrumentor

        with patch.dict('sys.modules', {'openinference.instrumentation.google_genai': mock_module}):
            result = setup_openinference_instrumentation()

            assert result["google_genai"] is True
            mock_instrumentor.instrument.assert_called_once()

    def test_check_instrumentation_status(self):
        """Test checking instrumentation status."""
        status = check_instrumentation_status()

        assert "openai" in status
        assert "anthropic" in status
        assert "google_genai" in status
        assert "available" in status["openai"]
        assert "instrumented" in status["openai"]
        assert "available" in status["google_genai"]
        assert "instrumented" in status["google_genai"]

    @patch('miiflow_agent.core.observability.auto_instrumentation.setup_opentelemetry_tracing')
    @patch('miiflow_agent.core.observability.auto_instrumentation.setup_openinference_instrumentation')
    def test_enable_phoenix_tracing(self, mock_instrumentation, mock_otel):
        """Test enabling Phoenix tracing."""
        mock_otel.return_value = True
        mock_instrumentation.return_value = {"openai": True, "anthropic": True, "google_genai": True}

        result = enable_phoenix_tracing()

        assert result is True
        mock_otel.assert_called_once()
        mock_instrumentation.assert_called_once()


class TestLogging:
    """Test logging functionality."""

    def test_configure_structured_logging(self):
        """Test structured logging configuration."""
        config = ObservabilityConfig(structured_logging=True)
        
        result = configure_structured_logging(config, force_configuration=True)
        
        assert result is True

    def test_get_logger(self):
        """Test getting a logger."""
        logger = get_logger("test_module")
        
        assert logger is not None
        # Should be able to log without errors
        logger.info("Test message", test_key="test_value")


class TestIntegration:
    """Integration tests for cleaned-up observability."""

    def test_observability_graceful_degradation(self):
        """Test that observability degrades gracefully when dependencies are missing."""
        # Mock missing dependencies
        with patch.dict('sys.modules', {'phoenix': None}):
            config = ObservabilityConfig(phoenix_enabled=True, phoenix_endpoint="http://localhost:6006")
            # Should not raise an error even with missing dependencies
            assert config.phoenix_enabled

    @patch('miiflow_agent.core.observability.auto_instrumentation.setup_opentelemetry_tracing')
    def test_simplified_phoenix_setup(self, mock_otel_setup):
        """Test the simplified Phoenix setup process."""
        mock_otel_setup.return_value = True
        
        # Should work with minimal configuration
        result = enable_phoenix_tracing("http://localhost:6006")
        
        # Should not fail even if some components are missing
        assert isinstance(result, bool)
