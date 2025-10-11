"""Tool decorators for easy tool registration and definition."""

import inspect
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from functools import wraps

from .schemas import ParameterSchema, ToolSchema
from .types import FunctionType, ToolType
from .function import FunctionTool
from .schema_utils import get_fun_schema

F = TypeVar('F', bound=Callable[..., Any])


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    function_type: FunctionType = FunctionType.SYNC,
    tags: Optional[List[str]] = None
) -> Callable[[F], F]:
    """
    Decorator to mark a function as a tool and automatically generate its schema.

    Example:
        @tool(description="Add two numbers")
        def add(a: int, b: int) -> int:
            '''Add two numbers together.'''
            return a + b
        
        @tool(name="weather", function_type=FunctionType.POSITIONAL_ONLY)
        async def get_weather(city: str) -> str:
            '''Get weather for a city.'''
            return f"Weather in {city}: sunny"
    """
    def decorator(func: F) -> F:
        tool_name = name or func.__name__
        
        tool_description = description
        if not tool_description and func.__doc__:
            tool_description = func.__doc__.strip().split('\n')[0]
        
        try:
            schema_dict = get_fun_schema(func)
            
            parameters = {}
            if 'parameters' in schema_dict and 'properties' in schema_dict['parameters']:
                for param_name, param_info in schema_dict['parameters']['properties'].items():
                    parameters[param_name] = ParameterSchema(
                        name=param_name,
                        type=param_info.get('type', 'string'),
                        description=param_info.get('description', ''),
                        required=param_name in schema_dict['parameters'].get('required', []),
                        default=param_info.get('default')
                    )
            
            schema = ToolSchema(
                name=tool_name,
                description=tool_description or schema_dict.get('description', f"Function {tool_name}"),
                tool_type=ToolType.FUNCTION,
                parameters=parameters
            )
                
            if tags:
                schema.metadata['tags'] = tags
                
        except Exception as e:
            schema = ToolSchema(
                name=tool_name,
                description=tool_description or "No description available",
                tool_type=ToolType.FUNCTION,
                parameters={}
            )
        
        function_tool = FunctionTool(func, tool_name, tool_description)
        
        func._tool_schema = schema  # type: ignore
        func._function_tool = function_tool  # type: ignore
        func._is_tool = True  # type: ignore
        
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
