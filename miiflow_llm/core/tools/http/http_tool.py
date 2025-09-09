"""HTTP/REST API tool with proxy support."""

import asyncio
import json
import time
import logging
import httpx
from typing import Any, Dict

from ..schemas import ToolResult, ToolSchema
from ..types import ToolType
from ..exceptions import ToolExecutionError
from .proxy_utils import get_proxy_config, should_use_proxy

logger = logging.getLogger(__name__)


class HTTPTool:
    """HTTP/REST API tool that executes API calls from schema."""
    
    def __init__(self, schema: ToolSchema):
        if schema.tool_type != ToolType.HTTP_API:
            raise ValueError(f"Schema must be HTTP_API type, got {schema.tool_type}")
        if not schema.url:
            raise ValueError("HTTP tool schema must have URL")
            
        self.schema = schema
        self.name = schema.name
        
    async def execute(self, **kwargs) -> ToolResult:
        """Execute HTTP API call with proxy support."""
        start_time = time.time()
        
        try:
            validated_params = self._validate_parameters(kwargs)
            
            url = self.schema.url
            method = self.schema.method.upper()
            headers = self.schema.headers.copy()
            timeout = self.schema.timeout
            proxy_config = None
            if should_use_proxy(url):
                proxy_config = get_proxy_config()
            
            if proxy_config:
                logger.debug(f"Proxy detected via environment variables for {self.name}: {proxy_config}")
            else:
                logger.debug(f"No proxy configuration for {self.name}")
            
            # Configure httpx client - httpx automatically reads HTTP_PROXY/HTTPS_PROXY env vars
            client_kwargs = {
                "timeout": httpx.Timeout(timeout, connect=timeout/2),  
                "verify": True  
            }
            
            # httpx automatically detects proxy environment variables (HTTP_PROXY, HTTPS_PROXY, NO_PROXY)
            
            async with httpx.AsyncClient(**client_kwargs) as client:
                if method == "GET":
                    response = await client.get(
                        url, 
                        params=validated_params,
                        headers=headers
                    )
                elif method == "POST":
                    response = await client.post(
                        url,
                        json=validated_params,
                        headers=headers
                    )
                elif method == "PUT":
                    response = await client.put(
                        url,
                        json=validated_params,
                        headers=headers
                    )
                elif method == "DELETE":
                    response = await client.delete(
                        url,
                        params=validated_params,
                        headers=headers
                    )
                else:
                    raise ToolExecutionError(f"Unsupported HTTP method: {method}")
                
                response.raise_for_status()
                try:
                    result = response.json()
                except json.JSONDecodeError:
                    result = response.text
                
                execution_time = time.time() - start_time
                
                return ToolResult(
                    name=self.name,
                    input=validated_params,
                    output=result,
                    success=True,
                    execution_time=execution_time,
                    metadata={
                        "status_code": response.status_code,
                        "url": str(response.url),
                        "method": method
                    }
                )
                
        except httpx.ProxyError as e:
            execution_time = time.time() - start_time
            error_msg = f"Proxy error for HTTP tool '{self.name}': {str(e)}"
            logger.error(error_msg)
            logger.debug("Check proxy settings: HTTP_PROXY, HTTPS_PROXY, NO_PROXY")
            
            return ToolResult(
                name=self.name,
                input=kwargs,
                output=None,
                error=error_msg,
                success=False,
                execution_time=execution_time,
                metadata={
                    "error_type": "ProxyError",
                    "suggestion": "Verify proxy configuration in environment variables"
                }
            )
        except httpx.ConnectTimeout as e:
            execution_time = time.time() - start_time
            error_msg = f"Connection timeout for HTTP tool '{self.name}': {str(e)}"
            logger.error(error_msg)
            
            return ToolResult(
                name=self.name,
                input=kwargs,
                output=None,
                error=error_msg,
                success=False,
                execution_time=execution_time,
                metadata={
                    "error_type": "ConnectTimeout",
                    "suggestion": "Check network connectivity or increase timeout"
                }
            )
        except httpx.ReadTimeout as e:
            execution_time = time.time() - start_time
            error_msg = f"Read timeout for HTTP tool '{self.name}': {str(e)}"
            logger.error(error_msg)
            
            return ToolResult(
                name=self.name,
                input=kwargs,
                output=None,
                error=error_msg,
                success=False,
                execution_time=execution_time,
                metadata={
                    "error_type": "ReadTimeout",
                    "suggestion": "Server took too long to respond, try again or increase timeout"
                }
            )
        except httpx.HTTPStatusError as e:
            execution_time = time.time() - start_time
            error_msg = f"HTTP {e.response.status_code} error for tool '{self.name}': {e.response.text}"
            logger.error(error_msg)
            
            return ToolResult(
                name=self.name,
                input=kwargs,
                output=None,
                error=error_msg,
                success=False,
                execution_time=execution_time,
                metadata={
                    "error_type": "HTTPStatusError",
                    "status_code": e.response.status_code,
                    "response_text": e.response.text
                }
            )
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"HTTP tool '{self.name}' failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return ToolResult(
                name=self.name,
                input=kwargs,
                output=None,
                error=error_msg,
                success=False,
                execution_time=execution_time,
                metadata={"error_type": type(e).__name__}
            )
    
    def _validate_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters against schema."""
        validated = {}
        
        for param_name, param_schema in self.schema.parameters.items():
            
            if param_schema.required and param_name not in params:
                raise ToolExecutionError(f"Missing required parameter: {param_name}")
            
            if param_name in params:
                value = params[param_name]
                
                # Type validation (basic)
                if param_schema.type == "integer":
                    try:
                        value = int(value)
                    except (ValueError, TypeError):
                        raise ToolExecutionError(f"Parameter {param_name} must be integer")
                elif param_schema.type == "number":
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        raise ToolExecutionError(f"Parameter {param_name} must be number")
                elif param_schema.type == "boolean":
                    if isinstance(value, str):
                        value = value.lower() in ("true", "1", "yes", "on")
                    else:
                        value = bool(value)
                
                # Enum validation
                if param_schema.enum and value not in param_schema.enum:
                    raise ToolExecutionError(f"Parameter {param_name} must be one of {param_schema.enum}")
                
                # Range validation
                if param_schema.minimum is not None and value < param_schema.minimum:
                    raise ToolExecutionError(f"Parameter {param_name} must be >= {param_schema.minimum}")
                if param_schema.maximum is not None and value > param_schema.maximum:
                    raise ToolExecutionError(f"Parameter {param_name} must be <= {param_schema.maximum}")
                
                validated[param_name] = value
            elif param_schema.default is not None:
                validated[param_name] = param_schema.default
        
        return validated
