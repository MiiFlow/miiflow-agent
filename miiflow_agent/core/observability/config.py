"""Configuration for observability features."""

import os
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse


@dataclass
class ObservabilityConfig:
    """Configuration for observability features."""

    phoenix_enabled: bool = False
    phoenix_endpoint: Optional[str] = None
    phoenix_project_name: str = "miiflow-agent"
    phoenix_api_key: Optional[str] = None
    phoenix_client_headers: Optional[str] = None
    structured_logging: bool = True

    # ── Arize AX ─────────────────────────────────────────────────────────
    # A DIFFERENT backend from Phoenix, not a cloud flavour of it: AX
    # collects at otlp.arize.com and authenticates with `space_id`+`api_key`
    # headers, where Phoenix Cloud uses `Authorization: Bearer`. Both speak
    # OTLP and both accept OpenInference spans, so the instrumentation is
    # shared and only the exporter differs.
    arize_space_id: Optional[str] = None
    arize_api_key: Optional[str] = None
    arize_endpoint: Optional[str] = None
    arize_project_name: str = "miiflow-agent"

    @classmethod
    def from_env(cls) -> "ObservabilityConfig":
        """Create configuration from environment variables.

        Environment variables:
            PHOENIX_ENABLED: Enable Phoenix tracing (true/false)
            PHOENIX_ENDPOINT: Phoenix server endpoint URL (local)
            PHOENIX_COLLECTOR_ENDPOINT: Phoenix Cloud collector endpoint (cloud)
            PHOENIX_API_KEY: Phoenix Cloud API key (for cloud instances)
            PHOENIX_CLIENT_HEADERS: Custom headers for authentication (for old cloud instances)
            PHOENIX_PROJECT_NAME: Project name for Phoenix traces
            STRUCTURED_LOGGING: Enable structured logging (true/false)
        """
        phoenix_enabled = os.getenv("PHOENIX_ENABLED", "false").lower() == "true"

        # Support both PHOENIX_COLLECTOR_ENDPOINT (cloud) and PHOENIX_ENDPOINT (local)
        # PHOENIX_COLLECTOR_ENDPOINT takes precedence for cloud deployments
        phoenix_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT") or os.getenv("PHOENIX_ENDPOINT")

        # Default to local Phoenix if enabled but no endpoint specified
        if phoenix_enabled and not phoenix_endpoint:
            phoenix_endpoint = "http://localhost:6006"

        # Phoenix Cloud authentication
        phoenix_api_key = os.getenv("PHOENIX_API_KEY")
        phoenix_client_headers = os.getenv("PHOENIX_CLIENT_HEADERS")

        return cls(
            phoenix_enabled=phoenix_enabled,
            phoenix_endpoint=phoenix_endpoint,
            phoenix_project_name=os.getenv("PHOENIX_PROJECT_NAME", "miiflow-agent"),
            phoenix_api_key=phoenix_api_key,
            phoenix_client_headers=phoenix_client_headers,
            structured_logging=os.getenv("STRUCTURED_LOGGING", "true").lower() == "true",
            arize_space_id=os.getenv("ARIZE_SPACE_ID"),
            arize_api_key=os.getenv("ARIZE_API_KEY"),
            # Arize's own onboarding hands out ".../v1"; the OTLP traces path
            # is ".../v1/traces". `arize_traces_url` normalises whichever form
            # is set, so pasting the value from the console just works.
            arize_endpoint=os.getenv("ARIZE_OTLP_ENDPOINT") or os.getenv(
                "ARIZE_COLLECTOR_ENDPOINT"
            ),
            arize_project_name=os.getenv("ARIZE_PROJECT_NAME", "miiflow-agent"),
        )

    @property
    def arize_enabled(self) -> bool:
        """Arize AX needs no separate on/off flag — credentials ARE the flag.

        Deliberate: the Phoenix path is gated by `PHOENIX_ENABLED`, and a
        second flag to forget would mean setting three env vars correctly and
        still getting silence. Nothing is exported unless both a space id and
        an api key are present.
        """
        return bool(self.arize_space_id and self.arize_api_key)

    @property
    def arize_traces_url(self) -> str:
        """Full OTLP traces URL, tolerant of how the endpoint was written.

        Accepts ``https://otlp.arize.com``, ``.../v1`` (what the console
        shows) or ``.../v1/traces``, and always yields ``.../v1/traces``.
        The Phoenix path's blind ``f"{endpoint}/v1/traces"`` would turn the
        console value into ``/v1/v1/traces`` and drop every span.
        """
        base = (self.arize_endpoint or "https://otlp.arize.com").rstrip("/")
        if base.endswith("/v1/traces"):
            return base
        if base.endswith("/v1"):
            return f"{base}/traces"
        return f"{base}/v1/traces"

    @classmethod
    def for_local(cls, project_name: str = "miiflow-agent") -> "ObservabilityConfig":
        """Factory method for local Phoenix deployment.

        Args:
            project_name: Project name for Phoenix traces

        Returns:
            Configuration for local Phoenix instance
        """
        return cls(
            phoenix_enabled=True,
            phoenix_endpoint="http://localhost:6006",
            phoenix_project_name=project_name,
            structured_logging=True,
        )

    @classmethod
    def for_cloud(
        cls,
        api_key: str,
        endpoint: str,
        project_name: str = "miiflow-agent",
        client_headers: Optional[str] = None,
    ) -> "ObservabilityConfig":
        """Factory method for Phoenix Cloud deployment.

        Args:
            api_key: Phoenix Cloud API key
            endpoint: Phoenix Cloud collector endpoint (e.g., https://your-space.phoenix.arize.com)
            project_name: Project name for Phoenix traces
            client_headers: Custom headers for old cloud instances (created before June 24, 2025)

        Returns:
            Configuration for Phoenix Cloud instance
        """
        return cls(
            phoenix_enabled=True,
            phoenix_endpoint=endpoint,
            phoenix_project_name=project_name,
            phoenix_api_key=api_key,
            phoenix_client_headers=client_headers,
            structured_logging=True,
        )

    def is_phoenix_cloud(self) -> bool:
        """Check if using Phoenix Cloud (vs local Phoenix).

        Returns:
            True if configured for Phoenix Cloud
        """
        return bool(self.phoenix_api_key)

    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        if self.phoenix_enabled and not self.phoenix_endpoint:
            return False

        if self.phoenix_endpoint:
            try:
                parsed = urlparse(self.phoenix_endpoint)
                return bool(parsed.scheme and parsed.netloc)
            except Exception:
                return False

        return True

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.is_valid():
            raise ValueError(
                f"Invalid observability configuration. "
                f"Phoenix enabled: {self.phoenix_enabled}, "
                f"endpoint: {self.phoenix_endpoint}"
            )
