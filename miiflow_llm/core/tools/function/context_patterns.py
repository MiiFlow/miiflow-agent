"""Context injection pattern analysis for function tools."""

import inspect
import logging
from enum import Enum
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)


class ContextPattern(Enum):
    """Context injection patterns for function tools."""
    NONE = "none"
    FIRST_PARAM = "first_param"  # Pydantic AI style: fn(ctx, ...)
    KEYWORD = "keyword"  # Current style: fn(..., context=None)


def detect_context_pattern(fn: Callable) -> ContextPattern:
    """Detect the context injection pattern for a function."""
    result = analyze_context_pattern(fn)
    return ContextPattern(result['pattern'])


def filter_context_params(fn_schema: Dict[str, Any], fn: Callable) -> Dict[str, Any]:
    """Filter context parameters from function schema."""
    context_info = analyze_context_pattern(fn)
    filter_context_from_schema(fn_schema, context_info)
    return fn_schema


def analyze_context_pattern(fn: Callable) -> Dict[str, Any]:
    """Analyze how this function expects context injection.
    
    Supports two patterns:
    1. Pydantic AI style: context as first parameter (ctx, context)
    2. Current style: context as keyword parameter anywhere in signature
    """
    sig = inspect.signature(fn)
    params = list(sig.parameters.items())
    
    if not params:
        return {'pattern': 'none'}
    
    # Check for first parameter pattern (Pydantic AI style)
    first_param_name, first_param = params[0]
    if first_param_name in ('ctx', 'context'):
        return {
            'pattern': 'first_param',
            'param_name': first_param_name,
            'param_index': 0
        }
    
    # Check for keyword pattern (current style)
    for i, (param_name, param) in enumerate(params):
        if param_name == 'context' and i > 0:  # Not first param, but named context
            return {
                'pattern': 'keyword',
                'param_name': param_name,
                'param_index': i
            }
    
    return {'pattern': 'none'}


def filter_context_from_schema(fn_schema: Dict[str, Any], context_injection: Dict[str, Any]) -> None:
    """Remove context parameters from the function schema sent to LLM."""
    if context_injection['pattern'] == 'none':
        return
        
    param_name = context_injection['param_name']
    
    if ('parameters' in fn_schema and 
        'properties' in fn_schema['parameters'] and
        param_name in fn_schema['parameters']['properties']):
        del fn_schema['parameters']['properties'][param_name]
        logger.debug(f"Filtered context parameter '{param_name}' from schema")
    
    if ('parameters' in fn_schema and
        'required' in fn_schema['parameters'] and 
        param_name in fn_schema['parameters']['required']):
        fn_schema['parameters']['required'].remove(param_name)
        logger.debug(f"Removed '{param_name}' from required parameters")
