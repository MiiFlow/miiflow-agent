"""Tool registry for managing function and HTTP tools."""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from .exceptions import ToolPreparationError
from .function import FunctionTool
from .http import HTTPTool
from .schemas import ToolResult, ToolSchema
from .types import ToolType

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Tool registry with allowlist validation and safe execution."""

    def __init__(self, allowlist: Optional[List[str]] = None, enable_logging: bool = True):
        self.tools: Dict[str, FunctionTool] = {}
        self.http_tools: Dict[str, HTTPTool] = {}
        self.allowlist = set(allowlist) if allowlist else None
        self.enable_logging = enable_logging
        self.execution_stats: Dict[str, Dict[str, Any]] = {}

    def register(self, tool) -> None:
        """Register a function tool with allowlist validation."""
        if hasattr(tool, "_function_tool"):
            tool = tool._function_tool

        if not isinstance(tool, FunctionTool):
            raise TypeError(f"Expected FunctionTool or decorated function, got {type(tool)}")

        if hasattr(tool.schema, "name"):
            tool_name = tool.schema.name
        else:
            tool_name = tool.schema.get("name", tool.name)

        if self.allowlist and tool_name not in self.allowlist:
            raise ToolPreparationError(f"Tool '{tool_name}' not in allowlist: {self.allowlist}")

        self.tools[tool_name] = tool
        self.execution_stats[tool_name] = {
            "calls": 0,
            "successes": 0,
            "failures": 0,
            "total_time": 0.0,
        }

        if self.enable_logging:
            logger.info(f"Registered function tool: {tool_name}")

    def register_http_tool(self, schema: ToolSchema) -> None:
        """Register an HTTP/REST API tool with schema."""
        if schema.tool_type != ToolType.HTTP_API:
            raise ValueError(f"Expected HTTP_API tool type, got {schema.tool_type}")

        if self.allowlist and schema.name not in self.allowlist:
            raise ToolPreparationError(
                f"HTTP tool '{schema.name}' not in allowlist: {self.allowlist}"
            )

        http_tool = HTTPTool(schema)
        self.http_tools[schema.name] = http_tool
        self.execution_stats[schema.name] = {
            "calls": 0,
            "successes": 0,
            "failures": 0,
            "total_time": 0.0,
        }

        if self.enable_logging:
            logger.info(f"Registered HTTP tool: {schema.name} -> {schema.url}")

    def get(self, name: str) -> Optional[FunctionTool]:
        """Get a function tool by name."""
        return self.tools.get(name)

    def get_http_tool(self, name: str) -> Optional[HTTPTool]:
        """Get an HTTP tool by name."""
        return self.http_tools.get(name)

    def list_tools(self) -> List[str]:
        """List all registered tool names (function and HTTP)."""
        return list(self.tools.keys()) + list(self.http_tools.keys())

    def get_schemas(self, provider: str, client=None) -> List[Dict[str, Any]]:
        """Get all tool schemas in provider format."""
        schemas = []

        for tool in self.tools.values():
            if client and hasattr(client, "convert_schema_to_provider_format"):
                universal_schema = tool.definition.to_universal_schema()
                schemas.append(client.convert_schema_to_provider_format(universal_schema))
            else:
                schemas.append(tool.to_provider_format(provider))

        for http_tool in self.http_tools.values():
            if client and hasattr(client, "convert_schema_to_provider_format"):
                universal_schema = http_tool.schema.to_universal_schema()
                schemas.append(client.convert_schema_to_provider_format(universal_schema))
            else:
                schemas.append(http_tool.schema.to_provider_format(provider))

        return schemas

    def validate_tool_call(self, name: str, **kwargs) -> bool:
        """Validate a tool call against schema and allowlist."""
        if name not in self.tools and name not in self.http_tools:
            return False

        if self.allowlist and name not in self.allowlist:
            return False

        try:
            if name in self.tools:
                tool = self.tools[name]
                tool.validate_inputs(**kwargs)
            else:
                http_tool = self.http_tools[name]
                http_tool._validate_parameters(kwargs)
            return True
        except Exception:
            return False

    async def execute_safe(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool with comprehensive error handling and stats tracking."""
        if tool_name in self.execution_stats:
            self.execution_stats[tool_name]["calls"] += 1

        function_tool = self.get(tool_name)
        http_tool = self.get_http_tool(tool_name)

        if not function_tool and not http_tool:
            all_tools = list(self.tools.keys()) + list(self.http_tools.keys())
            error_msg = f"Tool '{tool_name}' not found. Available: {all_tools}"
            if self.enable_logging:
                logger.error(error_msg)

            return ToolResult(
                name=tool_name,
                input=kwargs,
                output=None,
                error=error_msg,
                success=False,
                metadata={"error_type": "tool_not_found"},
            )

        if self.allowlist and tool_name not in self.allowlist:
            error_msg = f"Tool '{tool_name}' not in allowlist: {sorted(self.allowlist)}"
            if self.enable_logging:
                logger.error(error_msg)

            return ToolResult(
                name=tool_name,
                input=kwargs,
                output=None,
                error=error_msg,
                success=False,
                metadata={"error_type": "allowlist_violation"},
            )

        try:
            if function_tool:
                result = await function_tool.acall(**kwargs)
            else:
                result = await http_tool.execute(**kwargs)

            if tool_name in self.execution_stats:
                stats = self.execution_stats[tool_name]
                stats["total_time"] += result.execution_time
                if result.success:
                    stats["successes"] += 1
                else:
                    stats["failures"] += 1

            return result

        except Exception as e:
            error_msg = f"Registry error executing '{tool_name}': {str(e)}"
            if self.enable_logging:
                logger.debug(error_msg, exc_info=True)
            logger.error(error_msg)

            if tool_name in self.execution_stats:
                self.execution_stats[tool_name]["failures"] += 1

            return ToolResult(
                name=tool_name,
                input=kwargs,
                output=None,
                error=error_msg,
                success=False,
                metadata={"error_type": "registry_error", "original_error": str(e)},
            )

    async def execute_safe_with_context(self, tool_name: str, context: Any, **kwargs) -> ToolResult:
        """Execute tool with context as first parameter (Pydantic AI pattern)."""
        if tool_name not in self.tools:
            available_tools = list(self.tools.keys()) + list(self.http_tools.keys())
            return ToolResult(
                name=tool_name,
                input=kwargs,
                success=False,
                error=f"Tool '{tool_name}' not found. Available: {available_tools}",
            )

        if tool_name in self.execution_stats:
            self.execution_stats[tool_name]["calls"] += 1

        tool = self.tools[tool_name]
        start_time = time.time()

        try:
            if hasattr(tool, "fn"):
                if asyncio.iscoroutinefunction(tool.fn):
                    result = await tool.fn(context, **kwargs)
                else:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: tool.fn(context, **kwargs)
                    )
            else:
                kwargs["context"] = context
                return await self.execute_safe(tool_name, **kwargs)

            execution_time = time.time() - start_time

            if tool_name in self.execution_stats:
                stats = self.execution_stats[tool_name]
                stats["total_time"] += execution_time
                stats["successes"] += 1

            return ToolResult(
                name=tool_name,
                input={"context": "<RunContext>", **kwargs},
                output=result,
                success=True,
                execution_time=execution_time,
                metadata={"execution_pattern": "first_param"},
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Tool '{tool_name}' failed: {str(e)}"
            logger.error(error_msg)

            if tool_name in self.execution_stats:
                self.execution_stats[tool_name]["failures"] += 1

            return ToolResult(
                name=tool_name,
                input={"context": "<RunContext>", **kwargs},
                output=None,
                error=error_msg,
                success=False,
                execution_time=execution_time,
                metadata={"execution_pattern": "first_param", "error_type": type(e).__name__},
            )

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get execution statistics for all tools."""
        stats = {}
        for tool_name, raw_stats in self.execution_stats.items():
            calls = raw_stats["calls"]
            successes = raw_stats["successes"]
            failures = raw_stats["failures"]
            total_time = raw_stats["total_time"]

            stats[tool_name] = {
                "calls": calls,
                "successes": successes,
                "failures": failures,
                "success_rate": successes / calls if calls > 0 else 0.0,
                "avg_time": total_time / calls if calls > 0 else 0.0,
                "total_time": total_time,
            }

        return stats

    def reset_stats(self) -> None:
        """Reset all execution statistics."""
        for tool_name in self.execution_stats:
            self.execution_stats[tool_name] = {
                "calls": 0,
                "successes": 0,
                "failures": 0,
                "total_time": 0.0,
            }
