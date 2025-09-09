"""Type definitions and enums for the tools system."""

from enum import Enum
from typing import TypeVar

# Type variables for generic context and result types
ContextType = TypeVar('ContextType')
ResultType = TypeVar('ResultType')


class FunctionType(Enum):
    """Types of functions that can be tools."""
    SYNC = "sync"
    ASYNC = "async"
    SYNC_GENERATOR = "sync_generator"
    ASYNC_GENERATOR = "async_generator"


class ToolType(Enum):
    """Types of tools supported by the registry."""
    FUNCTION = "function"
    HTTP_API = "http_api"


class ParameterType(Enum):
    """JSON Schema parameter types for tool parameters."""
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    NULL = "null"
