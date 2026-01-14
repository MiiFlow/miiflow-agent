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
        )

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
