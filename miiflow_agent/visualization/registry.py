"""
Visualization registry and decorator for registering visualization tools.

This module provides:
1. A registry for tracking visualization type schemas (generic, application-defined)
2. A decorator for creating visualization tools that return VisualizationResult

The SDK provides only generic primitives - specific visualization types
(chart, table, etc.) are defined by the application layer.
"""

import logging
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar

from ..core.tools.decorators import tool
from .types import VisualizationResult

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., VisualizationResult])


class VisualizationRegistry:
    """
    Registry for application-defined visualization types.

    Applications call this to register what visualization types their
    frontend supports. The SDK itself does NOT define specific types -
    it's purely a registration mechanism for applications.

    Example (in APPLICATION code, not SDK):
        # Register a custom visualization type
        VisualizationRegistry.register_type(
            viz_type="org_chart",
            data_schema={
                "type": "object",
                "properties": {
                    "nodes": {"type": "array"},
                    "edges": {"type": "array"},
                },
                "required": ["nodes"],
            },
        )
    """

    _types: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register_type(
        cls,
        viz_type: str,
        data_schema: Dict[str, Any],
        config_schema: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
    ) -> None:
        """
        Register a visualization type with its JSON schema.

        Applications call this to define what visualization types their
        frontend supports. The SDK itself does NOT call this.

        Args:
            viz_type: Type identifier (e.g., "chart", "table", "my_custom_viz")
            data_schema: JSON Schema for the visualization data
            config_schema: Optional JSON Schema for configuration
            description: Optional description of the visualization type
        """
        cls._types[viz_type] = {
            "data_schema": data_schema,
            "config_schema": config_schema or {},
            "description": description,
        }
        logger.debug(f"Registered visualization type: {viz_type}")

    @classmethod
    def get_type_schema(cls, viz_type: str) -> Optional[Dict[str, Any]]:
        """Get schema for a registered visualization type."""
        return cls._types.get(viz_type)

    @classmethod
    def get_all_types(cls) -> List[str]:
        """Get list of all registered visualization types."""
        return list(cls._types.keys())

    @classmethod
    def get_all_schemas(cls) -> Dict[str, Dict[str, Any]]:
        """Get all registered type schemas."""
        return cls._types.copy()

    @classmethod
    def is_registered(cls, viz_type: str) -> bool:
        """Check if a visualization type is registered."""
        return viz_type in cls._types

    @classmethod
    def clear(cls) -> None:
        """Clear all registered types (useful for testing)."""
        cls._types.clear()


def visualization_tool(
    func: Optional[F] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    viz_type: str,
    tags: Optional[List[str]] = None,
) -> F:
    """
    Decorator to mark a function as a visualization tool.

    The decorated function should return a VisualizationResult.
    The viz_type parameter specifies what visualization type this tool produces
    (must match a type the frontend can render).

    Args:
        func: The function to decorate
        name: Tool name (defaults to function name)
        description: Tool description (defaults to docstring)
        viz_type: Visualization type this tool produces (REQUIRED)
        tags: Optional tags for categorization

    Example (in APPLICATION code, not SDK):
        @visualization_tool(viz_type="chart")
        def render_sales_chart(
            product: str,
            data: List[Dict[str, Any]],
        ) -> VisualizationResult:
            '''Render a sales chart for a product.'''
            return VisualizationResult(
                type="chart",
                data={"chartType": "bar", "series": [{"name": product, "data": data}]},
            )
    """
    def decorator(fn: F) -> F:
        tool_name = name or fn.__name__
        tool_description = description
        if not tool_description and fn.__doc__:
            tool_description = fn.__doc__.strip().split("\n")[0]

        # Add visualization tags
        all_tags = list(tags or [])
        all_tags.append("visualization")
        all_tags.append(f"viz:{viz_type}")

        # Apply the standard tool decorator
        decorated = tool(
            name=tool_name,
            description=tool_description,
            tags=all_tags,
        )(fn)

        # Mark as visualization tool
        decorated._is_visualization_tool = True
        decorated._viz_type = viz_type

        return decorated

    if func is not None:
        # Decorator used without arguments - but viz_type is required
        raise TypeError("visualization_tool requires viz_type argument")
    else:
        # Decorator used with arguments: @visualization_tool(viz_type="...")
        return decorator


def is_visualization_tool(func: Callable) -> bool:
    """Check if a function is a visualization tool."""
    return getattr(func, "_is_visualization_tool", False)


def get_visualization_type(func: Callable) -> Optional[str]:
    """Get the visualization type of a tool, if it's a visualization tool."""
    return getattr(func, "_viz_type", None)
