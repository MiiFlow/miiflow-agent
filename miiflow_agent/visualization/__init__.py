"""
Visualization module for miiflow-agent.

This module provides GENERIC PRIMITIVES for creating visualization tools.
The SDK does NOT define specific visualization types - that's an application concern.

The SDK provides:
- VisualizationResult: Generic container for visualization data
- VisualizationConfig: Generic configuration container
- visualization_tool: Decorator for creating visualization tools
- VisualizationRegistry: Registry for application-defined visualization types

Usage (in APPLICATION code):
    from miiflow_agent.visualization import (
        VisualizationResult,
        VisualizationConfig,
        visualization_tool,
        VisualizationRegistry,
    )

    # Define a visualization tool in your application
    @visualization_tool(viz_type="chart")
    def render_revenue_chart(data: list) -> VisualizationResult:
        return VisualizationResult(
            type="chart",
            data={"chartType": "line", "series": data},
            title="Revenue Chart",
        )

    # Optionally register the type schema for validation
    VisualizationRegistry.register_type(
        viz_type="chart",
        data_schema={
            "type": "object",
            "properties": {"chartType": {"type": "string"}, "series": {"type": "array"}},
            "required": ["chartType", "series"],
        },
    )
"""

from .types import (
    VisualizationConfig,
    VisualizationResult,
)
from .registry import (
    VisualizationRegistry,
    visualization_tool,
    is_visualization_tool,
    get_visualization_type,
)

__all__ = [
    # Types
    "VisualizationConfig",
    "VisualizationResult",
    # Registry
    "VisualizationRegistry",
    "visualization_tool",
    "is_visualization_tool",
    "get_visualization_type",
]
