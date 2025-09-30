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
    phoenix_project_name: str = "miiflow-llm"
    structured_logging: bool = True

    @classmethod
    def from_env(cls) -> "ObservabilityConfig":
        """Create configuration from environment variables.
        
        Environment variables:
            PHOENIX_ENABLED: Enable Phoenix tracing (true/false)
            PHOENIX_ENDPOINT: Phoenix server endpoint URL
            PHOENIX_PROJECT_NAME: Project name for Phoenix traces
            STRUCTURED_LOGGING: Enable structured logging (true/false)
        """
        phoenix_enabled = os.getenv("PHOENIX_ENABLED", "false").lower() == "true"

        # Default Phoenix endpoint for local development
        default_endpoint = "http://localhost:6006" if phoenix_enabled else None
        phoenix_endpoint = os.getenv("PHOENIX_ENDPOINT", default_endpoint)

        return cls(
            phoenix_enabled=phoenix_enabled,
            phoenix_endpoint=phoenix_endpoint,
            phoenix_project_name=os.getenv("PHOENIX_PROJECT_NAME", "miiflow-llm"),
            structured_logging=os.getenv("STRUCTURED_LOGGING", "true").lower() == "true",
        )

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
