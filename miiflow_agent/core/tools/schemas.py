"""Data classes for tool schemas and results."""

from typing import Any, Dict, List, Optional, Generic
from dataclasses import dataclass, field

from .types import ResultType, ToolType, ParameterType


@dataclass
class ParameterSchema:
    """Schema definition for tool parameters."""
    name: str
    type: ParameterType
    description: str
    required: bool = True
    default: Optional[Any] = None
    enum: Optional[List[Any]] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    pattern: Optional[str] = None
    # Support for nested schemas (arrays and objects)
    items: Optional[Dict[str, Any]] = None
    properties: Optional[Dict[str, Any]] = None
    additionalProperties: Optional[bool] = None
    
    def to_json_schema_property(self) -> Dict[str, Any]:
        """Convert to JSON Schema property format."""
        prop = {
            "type": self.type.value,
            "description": self.description
        }

        if self.default is not None:
            prop["default"] = self.default
        if self.enum is not None:
            prop["enum"] = self.enum
        if self.minimum is not None:
            prop["minimum"] = self.minimum
        if self.maximum is not None:
            prop["maximum"] = self.maximum
        if self.pattern is not None:
            prop["pattern"] = self.pattern

        # Add nested schema support for arrays and objects
        if self.items is not None:
            prop["items"] = self.items
        if self.properties is not None:
            prop["properties"] = self.properties
        if self.additionalProperties is not None:
            prop["additionalProperties"] = self.additionalProperties

        return prop


@dataclass
class ToolResult(Generic[ResultType]):
    """Production-grade tool execution result."""
    name: str
    input: Dict[str, Any]
    output: Optional[ResultType] = None
    error: Optional[str] = None
    success: bool = True
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_success(self) -> bool:
        return self.success and self.error is None


@dataclass
class ToolSchema:
    """Universal schema for tools (function and HTTP)."""
    name: str
    description: str
    tool_type: ToolType
    parameters: Dict[str, ParameterSchema] = field(default_factory=dict)
    
    # HTTP-specific fields
    url: Optional[str] = None
    method: str = "GET"
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0
    
    # Add metadata for arbitrary data
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        """Make ToolSchema hashable by using immutable fields only."""
        return hash((self.name, self.description, self.tool_type, self.url, self.method, self.timeout))
    
    def __eq__(self, other):
        """Equality based on name and tool_type."""
        if not isinstance(other, ToolSchema):
            return False
        return self.name == other.name and self.tool_type == other.tool_type
    
    def to_universal_schema(self) -> Dict[str, Any]:
        """Convert to universal JSON Schema format."""
        properties = {}
        required = []
        
        for param_name, param in self.parameters.items():
            properties[param_name] = param.to_json_schema_property()
            if param.required:
                required.append(param_name)
        
        schema = {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
        
        # Add returns field for consistency with function tools
        if self.tool_type == ToolType.HTTP_API:
            schema["returns"] = {
                "type": "object",
                "description": "HTTP API response data"
            }
        
        return schema
    
    def to_provider_format(self, provider: str) -> Dict[str, Any]:
        """Convert to provider-specific format.

        Note: This is a fallback method. Provider clients should use their own
        convert_schema_to_provider_format() method for proper name sanitization
        and mapping support.
        """
        # Direct implementation to avoid import issues
        universal = self.to_universal_schema()

        provider = provider.lower()

        if provider in ["openai", "groq", "xai", "mistral", "ollama"]:
            # OpenAI format - name sanitization is handled by provider clients
            return {
                "type": "function",
                "function": universal
            }
        elif provider == "anthropic":
            # Anthropic format
            return {
                "name": universal["name"],
                "description": universal["description"],
                "input_schema": universal["parameters"]
            }
        elif provider in ["gemini", "google"]:
            # Gemini format
            return {
                "name": universal["name"],
                "description": universal["description"],
                "parameters": universal["parameters"]
            }
        else:
            # Default: return universal schema
            return universal


@dataclass
class PreparedCall(Generic[ResultType]):
    """Prepared tool call with validated context and inputs."""
    tool_name: str
    function: Any  # Callable - avoiding import issues
    context: Any   # ContextType - avoiding import issues
    validated_inputs: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        import time
        self.prepared_at = time.time()
