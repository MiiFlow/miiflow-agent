"""Tool calling system with registry, schema validation, and safe execution."""

import json
import inspect
import httpx
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, Union, get_type_hints
from enum import Enum
import asyncio
import logging

logger = logging.getLogger(__name__)


class ToolType(Enum):
    """Types of tools supported."""
    FUNCTION = "function"
    HTTP_API = "http_api"


@dataclass
class ParameterSchema:
    """Schema definition for a tool parameter."""
    name: str
    type: str  # "string", "number", "integer", "boolean", "array", "object"
    description: str
    required: bool = True
    enum: Optional[List[Any]] = None
    default: Optional[Any] = None
    
    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format."""
        schema = {
            "type": self.type,
            "description": self.description
        }
        
        if self.enum:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
            
        return schema


@dataclass 
class ToolSchema:
    """Complete schema definition for a tool."""
    name: str
    description: str
    tool_type: ToolType
    parameters: List[ParameterSchema] = field(default_factory=list)
    return_type: str = "string"
    return_description: str = "Tool execution result"
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object", 
                    "properties": properties,
                    "required": required
                }
            }
        }
    
    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic tool format."""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties, 
                "required": required
            }
        }


@dataclass
class Observation:
    """Structured result from tool execution."""
    tool_name: str
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata
        }


class BaseTool(ABC):
    """Abstract base class for all tools."""
    
    def __init__(self, schema: ToolSchema):
        self.schema = schema
    
    @abstractmethod
    async def execute(self, **kwargs) -> Observation:
        """Execute the tool with given parameters."""
        pass
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters against schema."""
        validated = {}
        
        # Check required parameters
        required_params = {p.name for p in self.schema.parameters if p.required}
        provided_params = set(parameters.keys())
        
        missing = required_params - provided_params
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")
        
        # Validate and convert parameter types
        param_lookup = {p.name: p for p in self.schema.parameters}
        
        for name, value in parameters.items():
            if name not in param_lookup:
                raise ValueError(f"Unknown parameter: {name}")
            
            param_schema = param_lookup[name]
            validated[name] = self._validate_parameter_type(value, param_schema)
        
        # Add defaults for missing optional parameters
        for param in self.schema.parameters:
            if not param.required and param.name not in validated and param.default is not None:
                validated[param.name] = param.default
        
        return validated
    
    def _validate_parameter_type(self, value: Any, param_schema: ParameterSchema) -> Any:
        """Validate and convert a single parameter."""
        if param_schema.enum and value not in param_schema.enum:
            raise ValueError(f"Parameter {param_schema.name} must be one of {param_schema.enum}")
        
        # Type conversion based on schema type
        if param_schema.type == "string":
            return str(value)
        elif param_schema.type == "integer":
            return int(value)
        elif param_schema.type == "number":
            return float(value)
        elif param_schema.type == "boolean":
            if isinstance(value, bool):
                return value
            return str(value).lower() in ("true", "1", "yes", "on")
        elif param_schema.type == "array":
            if not isinstance(value, list):
                raise ValueError(f"Parameter {param_schema.name} must be an array")
            return value
        elif param_schema.type == "object":
            if not isinstance(value, dict):
                raise ValueError(f"Parameter {param_schema.name} must be an object")
            return value
        else:
            return value


class FunctionTool(BaseTool):
    """Tool that wraps a Python function."""
    
    def __init__(self, func: Callable, schema: Optional[ToolSchema] = None):
        self.func = func
        
        # Auto-generate schema if not provided
        if schema is None:
            schema = self._generate_schema_from_function()
        
        super().__init__(schema)
    
    def _generate_schema_from_function(self) -> ToolSchema:
        """Generate tool schema from function signature and docstring."""
        sig = inspect.signature(self.func)
        type_hints = get_type_hints(self.func)
        
        # Parse docstring for parameter descriptions
        doc = inspect.getdoc(self.func) or ""
        param_descriptions = self._parse_docstring_params(doc)
        
        parameters = []
        for name, param in sig.parameters.items():
            # Skip self parameter
            if name == "self":
                continue
                
            param_type = self._python_type_to_json_type(type_hints.get(name, str))
            description = param_descriptions.get(name, f"Parameter {name}")
            required = param.default is inspect.Parameter.empty
            default = None if param.default is inspect.Parameter.empty else param.default
            
            parameters.append(ParameterSchema(
                name=name,
                type=param_type,
                description=description,
                required=required,
                default=default
            ))
        
        # Use decorator metadata if available
        tool_name = getattr(self.func, '_tool_name', self.func.__name__)
        tool_description = getattr(self.func, '_tool_description', None)
        
        # Extract description from docstring or use decorator description
        if tool_description:
            description = tool_description
        else:
            description = doc.split('\n')[0] if doc else f"Function {self.func.__name__}"
        
        return ToolSchema(
            name=tool_name,
            description=description,
            tool_type=ToolType.FUNCTION,
            parameters=parameters
        )
    
    def _parse_docstring_params(self, docstring: str) -> Dict[str, str]:
        """Parse parameter descriptions from docstring."""
        params = {}
        lines = docstring.split('\n')
        
        current_param = None
        for line in lines:
            line = line.strip()
            
            # Look for parameter documentation patterns
            if line.startswith("Args:") or line.startswith("Parameters:"):
                continue
            elif ":" in line and not line.startswith(" "):
                # New parameter: "param_name: description"
                parts = line.split(":", 1)
                if len(parts) == 2:
                    current_param = parts[0].strip()
                    params[current_param] = parts[1].strip()
            elif current_param and line.startswith(" "):
                # Continuation of previous parameter description
                params[current_param] += " " + line.strip()
        
        return params
    
    def _python_type_to_json_type(self, python_type: Type) -> str:
        """Convert Python type hints to JSON Schema types."""
        if python_type == str:
            return "string"
        elif python_type == int:
            return "integer" 
        elif python_type == float:
            return "number"
        elif python_type == bool:
            return "boolean"
        elif python_type == list or str(python_type).startswith("typing.List"):
            return "array"
        elif python_type == dict or str(python_type).startswith("typing.Dict"):
            return "object"
        else:
            return "string"  # Default fallback
    
    async def execute(self, **kwargs) -> Observation:
        """Execute the wrapped function."""
        try:
            validated_params = self.validate_parameters(kwargs)
            
            if asyncio.iscoroutinefunction(self.func):
                result = await self.func(**validated_params)
            else:
                result = self.func(**validated_params)
            
            return Observation(
                tool_name=self.schema.name,
                success=True,
                result=result,
                metadata={"execution_type": "function"}
            )
            
        except Exception as e:
            logger.error(f"Function tool {self.schema.name} execution failed: {e}")
            return Observation(
                tool_name=self.schema.name,
                success=False,
                error=str(e),
                metadata={"execution_type": "function", "error_type": type(e).__name__}
            )


class HTTPTool(BaseTool):
    """Tool that makes HTTP API calls."""
    
    def __init__(self, schema: ToolSchema, url: str, method: str = "GET", headers: Optional[Dict[str, str]] = None):
        super().__init__(schema)
        self.url = url
        self.method = method.upper()
        self.headers = headers or {}
    
    async def execute(self, **kwargs) -> Observation:
        """Execute HTTP API call."""
        try:
            validated_params = self.validate_parameters(kwargs)
            
            async with httpx.AsyncClient() as client:
                if self.method == "GET":
                    response = await client.get(
                        self.url,
                        params=validated_params,
                        headers=self.headers
                    )
                elif self.method == "POST":
                    response = await client.post(
                        self.url,
                        json=validated_params,
                        headers=self.headers
                    )
                else:
                    raise ValueError(f"Unsupported HTTP method: {self.method}")
                
                response.raise_for_status()
                
                try:
                    result = response.json()
                except json.JSONDecodeError:
                    result = response.text
                
                return Observation(
                    tool_name=self.schema.name,
                    success=True,
                    result=result,
                    metadata={
                        "execution_type": "http_api",
                        "status_code": response.status_code,
                        "url": self.url,
                        "method": self.method
                    }
                )
                
        except httpx.RequestError as e:
            logger.error(f"HTTP tool {self.schema.name} request failed: {e}")
            return Observation(
                tool_name=self.schema.name,
                success=False,
                error=f"Request failed: {str(e)}",
                metadata={"execution_type": "http_api", "error_type": "request_error"}
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP tool {self.schema.name} returned error status: {e}")
            return Observation(
                tool_name=self.schema.name,
                success=False,
                error=f"HTTP {e.response.status_code}: {e.response.text}",
                metadata={
                    "execution_type": "http_api", 
                    "error_type": "http_error",
                    "status_code": e.response.status_code
                }
            )
        except Exception as e:
            logger.error(f"HTTP tool {self.schema.name} execution failed: {e}")
            return Observation(
                tool_name=self.schema.name,
                success=False,
                error=str(e),
                metadata={"execution_type": "http_api", "error_type": type(e).__name__}
            )


class ToolRegistry:
    """Registry for managing and executing tools with security controls."""
    
    def __init__(self, allowlist: Optional[List[str]] = None):
        self.tools: Dict[str, BaseTool] = {}
        self.allowlist = set(allowlist) if allowlist else None
        
    def register_function(self, func: Callable, schema: Optional[ToolSchema] = None) -> ToolSchema:
        """Register a Python function as a tool."""
        tool = FunctionTool(func, schema)
        self._register_tool(tool)
        return tool.schema
    
    def register_http_tool(self, schema: ToolSchema, url: str, method: str = "GET", headers: Optional[Dict[str, str]] = None) -> ToolSchema:
        """Register an HTTP API as a tool."""
        tool = HTTPTool(schema, url, method, headers)
        self._register_tool(tool)
        return schema
    
    def _register_tool(self, tool: BaseTool) -> None:
        """Internal method to register a tool with validation."""
        if self.allowlist and tool.schema.name not in self.allowlist:
            raise ValueError(f"Tool {tool.schema.name} is not in allowlist")
        
        if tool.schema.name in self.tools:
            logger.warning(f"Overwriting existing tool: {tool.schema.name}")
        
        self.tools[tool.schema.name] = tool
        logger.info(f"Registered {tool.schema.tool_type.value} tool: {tool.schema.name}")
    
    def get_tool_schemas(self, format_type: str = "openai") -> List[Dict[str, Any]]:
        """Get all tool schemas in specified format."""
        schemas = []
        for tool in self.tools.values():
            if format_type == "openai":
                schemas.append(tool.schema.to_openai_format())
            elif format_type == "anthropic":
                schemas.append(tool.schema.to_anthropic_format())
            else:
                raise ValueError(f"Unsupported format type: {format_type}")
        return schemas
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Observation:
        """Execute a tool with given parameters."""
        if tool_name not in self.tools:
            return Observation(
                tool_name=tool_name,
                success=False,
                error=f"Tool '{tool_name}' not found in registry"
            )
        
        if self.allowlist and tool_name not in self.allowlist:
            return Observation(
                tool_name=tool_name,
                success=False,
                error=f"Tool '{tool_name}' is not in allowlist"
            )
        
        tool = self.tools[tool_name]
        return await tool.execute(**parameters)
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self.tools.keys())
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a tool."""
        if tool_name not in self.tools:
            return None
        
        tool = self.tools[tool_name]
        return {
            "name": tool.schema.name,
            "description": tool.schema.description,
            "type": tool.schema.tool_type.value,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default
                }
                for p in tool.schema.parameters
            ]
        }


# Decorator for easy function registration
def tool(name: Optional[str] = None, description: Optional[str] = None):
    """Decorator to mark a function as a tool."""
    def decorator(func: Callable) -> Callable:
        # Add metadata to function for later registration
        func._tool_name = name or func.__name__
        func._tool_description = description
        func._is_tool = True
        return func
    return decorator
