"""Tool decorators for easy tool registration and definition."""

import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from functools import wraps

from .schemas import ParameterSchema, ToolSchema
from .types import FunctionType, ToolType
from .function import FunctionTool
from .schema_utils import get_fun_schema

logger = logging.getLogger(__name__)
F = TypeVar('F', bound=Callable[..., Any])


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    function_type: FunctionType = FunctionType.SYNC,
    tags: Optional[List[str]] = None,
    parameters: Optional[Dict[str, ParameterSchema]] = None,
    return_schema: Optional[Dict[str, Any]] = None,
    strict: bool = False
) -> Callable[[F], F]:
    """
    Decorator to mark a function as a tool with optional explicit schema.

    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to docstring first line)
        function_type: Sync/async function type
        tags: Tags for categorization
        parameters: Explicit parameter schemas (overrides reflection)
        return_schema: Explicit return type schema
        strict: Enable strict mode for type-safe function calling (for supported models)

    Example with automatic reflection:
        @tool(description="Add two numbers")
        def add(a: int, b: int) -> int:
            '''Add two numbers together.'''
            return a + b

    Example with explicit schemas:
        from miiflow_llm.core.tools import tool, ParameterSchema
        from miiflow_llm.core.tools.types import ParameterType

        @tool(
            description="Query CRM data with filters",
            parameters={
                "entity_type": ParameterSchema(
                    name="entity_type",
                    type=ParameterType.STRING,
                    description="Type of entity to query",
                    required=True,
                    enum=["account", "contact", "event", "signal"]
                ),
                "limit": ParameterSchema(
                    name="limit",
                    type=ParameterType.INTEGER,
                    description="Maximum results to return",
                    required=False,
                    default=20,
                    minimum=1,
                    maximum=100
                )
            }
        )
        def query_crm_data(ctx, entity_type: str, limit: int = 20):
            ...
    """
    def decorator(func: F) -> F:
        tool_name = name or func.__name__

        tool_description = description
        if not tool_description and func.__doc__:
            tool_description = func.__doc__.strip().split('\n')[0]

        # Use explicit parameters if provided, otherwise use reflection
        if parameters is not None:
            # User provided explicit schema - use it directly
            param_schemas = parameters
            logger.debug(f"Using explicit schema for {tool_name} with {len(parameters)} parameters")
        else:
            # Fall back to reflection-based schema generation
            try:
                from .types import ParameterType
                schema_dict = get_fun_schema(func)

                param_schemas = {}
                if 'parameters' in schema_dict and 'properties' in schema_dict['parameters']:
                    for param_name, param_info in schema_dict['parameters']['properties'].items():
                        # Convert string type to ParameterType enum
                        type_str = param_info.get('type', 'string')
                        param_type = ParameterType(type_str) if isinstance(type_str, str) else type_str

                        param_schemas[param_name] = ParameterSchema(
                            name=param_name,
                            type=param_type,
                            description=param_info.get('description', ''),
                            required=param_name in schema_dict['parameters'].get('required', []),
                            default=param_info.get('default'),
                            enum=param_info.get('enum')
                        )
                logger.debug(f"Generated schema via reflection for {tool_name} with {len(param_schemas)} parameters")
            except Exception as e:
                logger.warning(f"Schema reflection failed for {tool_name}, using empty schema: {e}")
                param_schemas = {}

        # Create ToolSchema
        schema = ToolSchema(
            name=tool_name,
            description=tool_description or f"Function {tool_name}",
            tool_type=ToolType.FUNCTION,
            parameters=param_schemas
        )

        # Add return schema if provided
        if return_schema:
            schema.metadata['return_schema'] = return_schema

        if tags:
            schema.metadata['tags'] = tags

        # Add strict mode flag
        if strict:
            schema.metadata['strict'] = True

        # Attach schema to function BEFORE creating FunctionTool
        # so FunctionTool can use the explicit schema
        func._tool_schema = schema  # type: ignore
        func._is_tool = True  # type: ignore

        # Create FunctionTool (will use func._tool_schema if available)
        function_tool = FunctionTool(func, tool_name, tool_description)

        # Attach function tool to function
        func._function_tool = function_tool  # type: ignore

        return func

    return decorator


def http_tool(
    url: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    parameters: Optional[Dict[str, ParameterSchema]] = None
) -> ToolSchema:
    """
    Create an HTTP tool schema.
    Example:
        weather_tool = http_tool(
            url="https://api.weather.com/v1/current",
            name="get_weather",
            description="Get current weather",
            parameters={
                "city": ParameterSchema(
                    type="string",
                    description="City name",
                    required=True
                )
            }
        )
    """
    if not name:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        path_parts = [p for p in parsed.path.split('/') if p]
        if path_parts:
            name = '_'.join(path_parts).replace('-', '_').lower()
        else:
            name = parsed.netloc.replace('.', '_').replace('-', '_').lower()
    
    return ToolSchema(
        name=name,
        description=description or f"{method} request to {url}",
        tool_type=ToolType.HTTP_API,
        url=url,
        method=method,
        headers=headers or {},
        parameters=parameters or {}
    )


def get_tool_from_function(func: Callable) -> Optional[FunctionTool]:
    
    return getattr(func, '_function_tool', None)


def is_tool(func: Callable) -> bool:
    
    return getattr(func, '_is_tool', False)


def get_tool_schema(func: Callable) -> Optional[ToolSchema]:
    return getattr(func, '_tool_schema', None)


def auto_register_tools(module, registry, prefix: str = "") -> int:
    """
    Automatically register all tools from a module.
    Example:
        import my_tools_module
        from miiflow_llm.core.tools import ToolRegistry
        
        registry = ToolRegistry()
        count = auto_register_tools(my_tools_module, registry)
        print(f"Registered {count} tools")
    """
    registered_count = 0
    
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        
        if attr_name.startswith('_') or not callable(attr):
            continue
            
        if is_tool(attr):
            tool = get_tool_from_function(attr)
            if tool:
                if prefix:
                    tool.schema.name = f"{prefix}_{tool.schema.name}"
                
                registry.register(tool)
                registered_count += 1
    
    return registered_count


# Backward compatibility aliases
function_tool = tool  # Old name for the decorator
create_tool = tool    # Alternative name
