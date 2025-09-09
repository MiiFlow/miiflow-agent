"""Schema generation utilities and provider conversions."""

import inspect
from typing import Any, Callable, Dict, List, Union, get_type_hints

from .types import FunctionType, ParameterType


def detect_function_type(fn: Callable) -> FunctionType:
    """Detect function type automatically."""
    if inspect.isasyncgenfunction(fn):
        return FunctionType.ASYNC_GENERATOR
    elif inspect.isgeneratorfunction(fn):
        return FunctionType.SYNC_GENERATOR
    elif inspect.iscoroutinefunction(fn):
        return FunctionType.ASYNC
    else:
        return FunctionType.SYNC


def get_fun_schema(fn: Callable) -> Dict[str, Any]:
    """Generate comprehensive schema from function signature.
    
    Extracts:
    - Function name and docstring
    - Parameter types and descriptions 
    - Return type hints
    - Default values and requirements
    """
    sig = inspect.signature(fn)
    type_hints = get_type_hints(fn)
    doc = inspect.getdoc(fn) or ""
    
    param_docs = _parse_docstring_params(doc)
    
    properties = {}
    required = []
    
    for param_name, param in sig.parameters.items():
        if param_name in ['context', 'self']:
            continue
            
        param_type = type_hints.get(param_name, str)
        json_type = _python_type_to_json_type(param_type)
        
        param_description = param_docs.get(param_name, f"Parameter {param_name}")
        
        properties[param_name] = {
            "type": json_type.value,
            "description": param_description
        }
        
        if hasattr(param_type, '__members__'):
            properties[param_name]["enum"] = list(param_type.__members__.keys())
        
        if param.default is inspect.Parameter.empty:
            required.append(param_name)
        else:
            properties[param_name]["default"] = param.default
    
    return_type = type_hints.get('return', Any)
    
    schema = {
        "name": fn.__name__,
        "description": doc.split('\n')[0] if doc else f"Function {fn.__name__}",
        "parameters": {
            "type": "object", 
            "properties": properties,
            "required": required
        }
    }
    
    if return_type != Any:
        schema["returns"] = {
            "type": _python_type_to_json_type(return_type).value,
            "description": f"Returns {return_type.__name__}"
        }
    
    return schema




def _parse_docstring_params(docstring: str) -> Dict[str, str]:
    """Parse parameter descriptions from docstring."""
    param_docs = {}
    lines = docstring.split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith('Args:') or line.startswith('Parameters:'):
            continue
        if ':' in line and line.count(':') == 1:
            parts = line.split(':', 1)
            if len(parts) == 2:
                param_name = parts[0].strip().replace('-', '').strip()
                description = parts[1].strip()
                param_docs[param_name] = description
    
    return param_docs


def _python_type_to_json_type(python_type: type) -> ParameterType:
    """Convert Python type to JSON schema type."""
    type_mapping = {
        str: ParameterType.STRING,
        int: ParameterType.INTEGER,
        float: ParameterType.NUMBER, 
        bool: ParameterType.BOOLEAN,
        list: ParameterType.ARRAY,
        dict: ParameterType.OBJECT,
        type(None): ParameterType.NULL
    }
    
    origin = getattr(python_type, '__origin__', None)
    if origin is Union:
        args = getattr(python_type, '__args__', ())
        if len(args) == 2 and type(None) in args:
            non_none_type = next(arg for arg in args if arg is not type(None))
            return type_mapping.get(non_none_type, ParameterType.STRING)
    
    if origin in (list, List):
        return ParameterType.ARRAY
    if origin in (dict, Dict):
        return ParameterType.OBJECT
    
    return type_mapping.get(python_type, ParameterType.STRING)
