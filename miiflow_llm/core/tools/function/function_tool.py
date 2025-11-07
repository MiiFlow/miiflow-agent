"""Function tool with context injection support."""

import asyncio
import time
import logging
from typing import Any, Callable, Dict, Optional

from ..schemas import ToolResult
from ..types import FunctionType
from ..exceptions import ToolExecutionError
from ..schema_utils import detect_function_type, get_fun_schema
from .context_patterns import analyze_context_pattern, filter_context_from_schema

logger = logging.getLogger(__name__)


class FunctionTool:
    """Function tool with safe execution."""
    
    def __init__(self, fn: Callable, name: Optional[str] = None, description: Optional[str] = None):
        from ..schemas import ToolSchema, ParameterSchema
        from ..types import ToolType, ParameterType

        self.fn = fn
        self.function_type = detect_function_type(fn)
        self.context_injection = analyze_context_pattern(fn)

        # Check if explicit schema was set by decorator
        if hasattr(fn, '_tool_schema'):
            # Use explicit schema from decorator
            self.definition = fn._tool_schema
            logger.debug(f"Using explicit schema for FunctionTool '{self.definition.name}'")
        else:
            # Fall back to reflection-based schema generation
            schema_dict = get_fun_schema(fn)
            parameters = {}
            if 'parameters' in schema_dict and 'properties' in schema_dict['parameters']:
                for param_name, param_info in schema_dict['parameters']['properties'].items():
                    # Convert string type to ParameterType enum
                    type_str = param_info.get('type', 'string')
                    param_type = ParameterType(type_str) if isinstance(type_str, str) else type_str

                    parameters[param_name] = ParameterSchema(
                        name=param_name,
                        type=param_type,
                        description=param_info.get('description', f'Parameter {param_name}'),
                        required=param_name in schema_dict['parameters'].get('required', []),
                        default=param_info.get('default')
                    )
            self.definition = ToolSchema(
                name=name or schema_dict.get('name', fn.__name__),
                description=description or schema_dict.get('description', f'Function {fn.__name__}'),
                tool_type=ToolType.FUNCTION,
                parameters=parameters
            )
            logger.debug(f"Generated schema via reflection for FunctionTool '{self.definition.name}'")

        self._filter_context_parameters()
    
    @property
    def schema(self) -> Any:
        """Schema property for backward compatibility."""
        return self.definition
    
    @property
    def name(self) -> str:
        """Get function name from definition."""
        return self.definition.name
        
    @property
    def description(self) -> str:
        """Get function description from definition."""
        return self.definition.description
    
    def to_provider_format(self, provider: str) -> Dict[str, Any]:
        """Convert to provider-specific format."""
        return self.definition.to_provider_format(provider)
    
    def validate_inputs(self, **kwargs) -> Dict[str, Any]:
        """Validate inputs against schema."""
        parameters = self.definition.parameters
        validated = {}
        
        for param_name, param_schema in parameters.items():
            if param_schema.required and param_name not in kwargs:
                raise ToolExecutionError(f"Missing required parameter: {param_name}")
            if param_name in kwargs:
                validated[param_name] = kwargs[param_name]
        
        for param_name, value in kwargs.items():
            if param_name not in parameters and param_name != 'context':
                logger.warning(f"Unknown parameter '{param_name}' for tool '{self.name}'")
            elif param_name != 'context': 
                validated[param_name] = value
        
        return validated

    async def acall(self, **kwargs) -> ToolResult:
        """Async execute the function with safe error handling."""
        start_time = time.time()

        try:
            validated_inputs = self.validate_inputs(**kwargs)

            if self.function_type == FunctionType.ASYNC:
                result = await self.fn(**validated_inputs)
            elif self.function_type == FunctionType.SYNC:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.fn(**validated_inputs)
                )
            elif self.function_type == FunctionType.ASYNC_GENERATOR:
                results = []
                async for item in self.fn(**validated_inputs):
                    results.append(item)
                result = results
            elif self.function_type == FunctionType.SYNC_GENERATOR:
                def run_sync_gen():
                    return list(self.fn(**validated_inputs))
                result = await asyncio.get_event_loop().run_in_executor(None, run_sync_gen)
            else:
                raise ToolExecutionError(f"Unsupported function type: {self.function_type}")

            execution_time = time.time() - start_time

            return ToolResult(
                name=self.name,
                input=validated_inputs,
                output=result,
                success=True,
                execution_time=execution_time,
                metadata={"function_type": self.function_type.value}
            )

        except ToolExecutionError as e:
            # Parameter validation or execution errors - provide detailed feedback
            execution_time = time.time() - start_time
            error_type = type(e).__name__
            error_details = str(e)

            # Build detailed error message with parameter information
            error_msg = f"Tool '{self.name}' execution failed ({error_type}): {error_details}"

            # Add parameter schema information if validation failed
            if "required parameter" in error_details.lower() or "missing" in error_details.lower():
                required_params = [name for name, schema in self.definition.parameters.items() if schema.required]
                optional_params = [name for name, schema in self.definition.parameters.items() if not schema.required]
                error_msg += f"\n\nExpected parameters:"
                if required_params:
                    error_msg += f"\n  Required: {', '.join(required_params)}"
                if optional_params:
                    error_msg += f"\n  Optional: {', '.join(optional_params)}"
                error_msg += f"\n  Provided: {', '.join(kwargs.keys())}"

            logger.debug(error_msg, exc_info=True)
            logger.error(error_msg)

            return ToolResult(
                name=self.name,
                input=kwargs,
                output=None,
                error=error_msg,
                success=False,
                execution_time=execution_time,
                metadata={"function_type": self.function_type.value, "error_type": error_type}
            )

        except TypeError as e:
            # Type errors often indicate parameter mismatches
            execution_time = time.time() - start_time
            error_details = str(e)

            # Build helpful error message with type information
            error_msg = f"Tool '{self.name}' parameter type error: {error_details}"
            error_msg += "\n\nParameter types expected:"
            for param_name, param_schema in self.definition.parameters.items():
                param_type = param_schema.type.value if hasattr(param_schema.type, 'value') else str(param_schema.type)
                required_str = " (required)" if param_schema.required else " (optional)"
                default_str = f", default={param_schema.default}" if param_schema.default is not None else ""
                error_msg += f"\n  - {param_name}: {param_type}{required_str}{default_str}"

            if kwargs:
                error_msg += "\n\nParameters provided:"
                for param_name, value in kwargs.items():
                    value_type = type(value).__name__
                    error_msg += f"\n  - {param_name}: {value_type} = {repr(value)[:100]}"

            logger.debug(error_msg, exc_info=True)
            logger.error(error_msg)

            return ToolResult(
                name=self.name,
                input=kwargs,
                output=None,
                error=error_msg,
                success=False,
                execution_time=execution_time,
                metadata={"function_type": self.function_type.value, "error_type": "TypeError"}
            )

        except ValueError as e:
            # Value errors often indicate invalid parameter values
            execution_time = time.time() - start_time
            error_details = str(e)

            error_msg = f"Tool '{self.name}' parameter value error: {error_details}"
            error_msg += "\n\nPlease check that all parameter values are valid and within expected ranges."

            # Add enum information if available
            for param_name, param_schema in self.definition.parameters.items():
                if param_schema.enum and param_name in kwargs:
                    error_msg += f"\n  - {param_name} must be one of: {', '.join(map(str, param_schema.enum))}"
                if hasattr(param_schema, 'minimum') and param_schema.minimum is not None and param_name in kwargs:
                    error_msg += f"\n  - {param_name} minimum: {param_schema.minimum}"
                if hasattr(param_schema, 'maximum') and param_schema.maximum is not None and param_name in kwargs:
                    error_msg += f"\n  - {param_name} maximum: {param_schema.maximum}"

            logger.debug(error_msg, exc_info=True)
            logger.error(error_msg)

            return ToolResult(
                name=self.name,
                input=kwargs,
                output=None,
                error=error_msg,
                success=False,
                execution_time=execution_time,
                metadata={"function_type": self.function_type.value, "error_type": "ValueError"}
            )

        except Exception as e:
            # Generic exception handling with detailed information
            execution_time = time.time() - start_time
            error_type = type(e).__name__
            error_details = str(e)

            # Build comprehensive error message
            error_msg = f"Tool '{self.name}' failed with {error_type}: {error_details}"

            # Include module and class info if available
            if hasattr(e, '__module__'):
                error_msg += f"\n  Exception module: {e.__module__}"

            # Add context about what was attempted
            if kwargs:
                error_msg += f"\n  Called with {len(kwargs)} parameter(s): {', '.join(kwargs.keys())}"

            logger.debug(error_msg, exc_info=True)
            logger.error(error_msg)

            return ToolResult(
                name=self.name,
                input=kwargs,
                output=None,
                error=error_msg,
                success=False,
                execution_time=execution_time,
                metadata={"function_type": self.function_type.value, "error_type": error_type}
            )
    
    def _filter_context_parameters(self):
        """Filter context parameters from the definition."""
        if self.context_injection['pattern'] in ('first_param', 'keyword'):
            param_name = self.context_injection['param_name']
            if param_name in self.definition.parameters:
                del self.definition.parameters[param_name]
                logger.debug(f"Filtered context parameter '{param_name}' from schema for tool '{self.name}'")
    
    def call_sync(self, **kwargs) -> ToolResult:
        """Synchronous call - runs async version in event loop."""
        return asyncio.run(self.acall(**kwargs))
