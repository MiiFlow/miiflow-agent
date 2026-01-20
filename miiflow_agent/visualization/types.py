"""
Type definitions for the visualization system.

This module provides generic dataclasses for defining visualization results.
The SDK does NOT define specific visualization types - that's an application concern.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import uuid


@dataclass
class VisualizationConfig:
    """
    Generic configuration options for visualizations.

    This is a flexible container for configuration settings that
    gets passed through to the frontend. The SDK does not prescribe
    specific config options - applications define what configs their
    frontend supports.

    Usage:
        config = VisualizationConfig(
            height=400,
            colors=["#3B82F6", "#10B981"],
            custom_option="value",
        )

    Note: All kwargs are stored and serialized. Common options include:
        - height: int - Height in pixels
        - width: str - Width (e.g., "100%", "500px")
        - colors: List[str] - Custom color palette
        - And any visualization-specific options defined by the application
    """

    def __init__(self, **kwargs: Any):
        """Initialize with arbitrary configuration options."""
        self._config = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        # Convert snake_case to camelCase for JavaScript convention
        result = {}
        for key, value in self._config.items():
            if value is not None:
                # Convert snake_case to camelCase
                parts = key.split("_")
                camel_key = parts[0] + "".join(p.capitalize() for p in parts[1:])
                result[camel_key] = value
        return result

    def __getattr__(self, name: str) -> Any:
        """Allow attribute access to config values."""
        if name.startswith("_"):
            raise AttributeError(name)
        return self._config.get(name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Allow attribute setting of config values."""
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self._config[name] = value


@dataclass
class VisualizationResult:
    """
    Result object returned by visualization tools.

    This is the marker object that tells the streaming handler to emit
    a visualization event to the frontend. The SDK provides this as a
    generic container - specific visualization types are defined by
    the application.

    Attributes:
        type: The visualization type (application-defined, e.g., "chart", "table")
        data: The visualization data (structure depends on type)
        title: Optional title for the visualization
        description: Optional description text
        config: Optional configuration settings
        id: Unique identifier for the visualization (auto-generated if not provided)

    Example:
        # Create a visualization (type and data structure defined by application)
        result = VisualizationResult(
            type="chart",
            data={
                "chartType": "line",
                "series": [
                    {"name": "Revenue", "data": [{"x": "Jan", "y": 100}]}
                ],
            },
            title="Monthly Revenue",
        )

        # The result is serialized and detected by the streaming handler
        # which emits a visualization SSE event to the frontend
    """
    type: str  # Application-defined type (e.g., "chart", "table", "custom_viz")
    data: Dict[str, Any]
    title: Optional[str] = None
    description: Optional[str] = None
    config: Optional[VisualizationConfig] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        The __visualization__ key is a marker that tells the streaming handler
        this is a visualization result that should be emitted as a separate event.
        """
        return {
            "__visualization__": True,  # Marker for streaming handler detection
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "description": self.description,
            "data": self.data,
            "config": self.config.to_dict() if self.config else {},
        }

    def __str__(self) -> str:
        """String representation for tool observation output."""
        # Return marker in text stream for positioning
        # The streaming handler will replace this with the actual visualization
        return f"[VIZ:{self.id}]"

    def __repr__(self) -> str:
        return f"VisualizationResult(type={self.type!r}, id={self.id!r}, title={self.title!r})"
