"""Tool registry for managing function, HTTP, and MCP tools."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .exceptions import ToolPreparationError
from .function import FunctionTool
from .http import HTTPTool
from .schemas import ToolResult, ToolSchema
from .types import ToolType

if TYPE_CHECKING:
    from .mcp import MCPTool, MCPToolManager, NativeMCPServerConfig

logger = logging.getLogger(__name__)


def _sanitize_tool_name(name: str) -> str:
    """Sanitize tool name to match provider patterns (e.g., OpenAI's ^[a-zA-Z0-9_-]+$)."""
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized[:64]


class ToolRegistry:
    """Tool registry with allowlist validation and safe execution.

    Supports three types of tools:
    - FunctionTool: Python function wrappers
    - HTTPTool: REST API wrappers
    - MCPTool: Model Context Protocol server tools
    """

    def __init__(self, allowlist: Optional[List[str]] = None, enable_logging: bool = True):
        self.tools: Dict[str, FunctionTool] = {}
        self.http_tools: Dict[str, HTTPTool] = {}
        self.mcp_tools: Dict[str, MCPTool] = {}
        self.mcp_manager: Optional[MCPToolManager] = None
        self.allowlist = set(allowlist) if allowlist else None
        self.enable_logging = enable_logging
        self.execution_stats: Dict[str, Dict[str, Any]] = {}
        # Map sanitized names back to original names for provider compatibility
        self._sanitized_to_original: Dict[str, str] = {}
        # Native MCP servers (for provider-side execution)
        self._native_mcp_servers: List[NativeMCPServerConfig] = []

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

        # Register sanitized name mapping for OpenAI compatibility
        sanitized_name = _sanitize_tool_name(tool_name)
        if sanitized_name != tool_name:
            self._sanitized_to_original[sanitized_name] = tool_name

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

        # Register sanitized name mapping for OpenAI compatibility
        sanitized_name = _sanitize_tool_name(schema.name)
        if sanitized_name != schema.name:
            self._sanitized_to_original[sanitized_name] = schema.name

        if self.enable_logging:
            logger.info(f"Registered HTTP tool: {schema.name} -> {schema.url}")

    def _resolve_name(self, name: str) -> str:
        """Resolve a potentially sanitized name back to the original name."""
        return self._sanitized_to_original.get(name, name)

    def get(self, name: str) -> Optional[FunctionTool]:
        """Get a function tool by name (supports sanitized names from OpenAI)."""
        resolved_name = self._resolve_name(name)
        return self.tools.get(resolved_name)

    def get_http_tool(self, name: str) -> Optional[HTTPTool]:
        """Get an HTTP tool by name (supports sanitized names from OpenAI)."""
        resolved_name = self._resolve_name(name)
        return self.http_tools.get(resolved_name)

    def register_mcp_tool(self, tool: MCPTool) -> None:
        """Register an MCP tool with allowlist validation.

        Args:
            tool: MCPTool instance to register

        Raises:
            ToolPreparationError: If tool not in allowlist
        """
        if self.allowlist and tool.name not in self.allowlist:
            raise ToolPreparationError(
                f"MCP tool '{tool.name}' not in allowlist: {self.allowlist}"
            )

        self.mcp_tools[tool.name] = tool
        self.execution_stats[tool.name] = {
            "calls": 0,
            "successes": 0,
            "failures": 0,
            "total_time": 0.0,
        }

        # Register sanitized name mapping for OpenAI compatibility
        sanitized_name = _sanitize_tool_name(tool.name)
        if sanitized_name != tool.name:
            self._sanitized_to_original[sanitized_name] = tool.name

        if self.enable_logging:
            logger.info(
                f"Registered MCP tool: {tool.name} "
                f"(server: {tool.server_name}, original: {tool.original_name})"
            )

    def register_mcp_manager(self, manager: MCPToolManager) -> None:
        """Register all tools from an MCPToolManager.

        Args:
            manager: Connected MCPToolManager with discovered tools
        """
        self.mcp_manager = manager
        for tool in manager.get_all_tools():
            self.register_mcp_tool(tool)

        if self.enable_logging:
            logger.info(
                f"Registered {len(manager.get_all_tools())} MCP tools from manager"
            )

    def get_mcp_tool(self, name: str) -> Optional[MCPTool]:
        """Get an MCP tool by name (supports sanitized names from OpenAI).

        Args:
            name: Tool name (namespaced or sanitized)

        Returns:
            MCPTool instance or None if not found
        """
        resolved_name = self._resolve_name(name)
        return self.mcp_tools.get(resolved_name)

    def register_native_mcp_server(self, config: NativeMCPServerConfig) -> None:
        """Register an MCP server for native provider-side execution.

        Native MCP servers are handled directly by the LLM provider (Anthropic, OpenAI)
        rather than requiring client-side connection management.

        Args:
            config: NativeMCPServerConfig with server URL and auth details
        """
        self._native_mcp_servers.append(config)
        if self.enable_logging:
            logger.info(f"Registered native MCP server: {config.name} -> {config.url}")

    def get_native_mcp_configs(self) -> List[NativeMCPServerConfig]:
        """Get all registered native MCP server configurations.

        Returns:
            List of NativeMCPServerConfig instances
        """
        return self._native_mcp_servers

    def has_native_mcp_servers(self) -> bool:
        """Check if any native MCP servers are registered.

        Returns:
            True if at least one native MCP server is registered
        """
        return len(self._native_mcp_servers) > 0

    def clear_native_mcp_servers(self) -> None:
        """Remove all registered native MCP servers."""
        self._native_mcp_servers.clear()
        if self.enable_logging:
            logger.info("Cleared all native MCP servers")

    def list_tools(self) -> List[str]:
        """List all registered tool names (function, HTTP, and MCP)."""
        return (
            list(self.tools.keys())
            + list(self.http_tools.keys())
            + list(self.mcp_tools.keys())
        )

    def get_schemas(self, provider: str, client=None) -> List[Dict[str, Any]]:
        """Get all tool schemas in provider format.

        Args:
            provider: Provider name (openai, anthropic, gemini, etc.)
            client: Optional client with convert_schema_to_provider_format method

        Returns:
            List of provider-formatted tool schemas
        """
        schemas = []

        # Function tools
        for tool in self.tools.values():
            if client and hasattr(client, "convert_schema_to_provider_format"):
                universal_schema = tool.definition.to_universal_schema()
                schemas.append(client.convert_schema_to_provider_format(universal_schema))
            else:
                schemas.append(tool.to_provider_format(provider))

        # HTTP tools
        for http_tool in self.http_tools.values():
            if client and hasattr(client, "convert_schema_to_provider_format"):
                universal_schema = http_tool.schema.to_universal_schema()
                schemas.append(client.convert_schema_to_provider_format(universal_schema))
            else:
                schemas.append(http_tool.schema.to_provider_format(provider))

        # MCP tools
        for mcp_tool in self.mcp_tools.values():
            if client and hasattr(client, "convert_schema_to_provider_format"):
                universal_schema = mcp_tool.schema.to_universal_schema()
                schemas.append(client.convert_schema_to_provider_format(universal_schema))
            else:
                schemas.append(mcp_tool.to_provider_format(provider))

        return schemas

    def validate_tool_call(self, name: str, **kwargs) -> bool:
        """Validate a tool call against schema and allowlist."""
        # Resolve sanitized name back to original (for OpenAI compatibility)
        resolved_name = self._resolve_name(name)

        if (
            resolved_name not in self.tools
            and resolved_name not in self.http_tools
            and resolved_name not in self.mcp_tools
        ):
            return False

        if self.allowlist and resolved_name not in self.allowlist:
            return False

        try:
            if resolved_name in self.tools:
                tool = self.tools[resolved_name]
                tool.validate_inputs(**kwargs)
            elif resolved_name in self.http_tools:
                http_tool = self.http_tools[resolved_name]
                http_tool._validate_parameters(kwargs)
            elif resolved_name in self.mcp_tools:
                mcp_tool = self.mcp_tools[resolved_name]
                mcp_tool.validate_inputs(**kwargs)
            return True
        except Exception:
            return False

    async def execute_safe(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool with comprehensive error handling and stats tracking.

        Supports function tools, HTTP tools, and MCP tools.
        """
        # Resolve sanitized name back to original (for OpenAI compatibility)
        resolved_name = self._resolve_name(tool_name)

        if resolved_name in self.execution_stats:
            self.execution_stats[resolved_name]["calls"] += 1

        function_tool = self.get(tool_name)
        http_tool = self.get_http_tool(tool_name)
        mcp_tool = self.get_mcp_tool(tool_name)

        if not function_tool and not http_tool and not mcp_tool:
            all_tools = self.list_tools()
            error_msg = f"Tool '{tool_name}' not found. Available: {all_tools}"
            if self.enable_logging:
                logger.error(error_msg)

            return ToolResult(
                name=resolved_name,
                input=kwargs,
                output=None,
                error=error_msg,
                success=False,
                metadata={"error_type": "tool_not_found"},
            )

        if self.allowlist and resolved_name not in self.allowlist:
            error_msg = f"Tool '{resolved_name}' not in allowlist: {sorted(self.allowlist)}"
            if self.enable_logging:
                logger.error(error_msg)

            return ToolResult(
                name=resolved_name,
                input=kwargs,
                output=None,
                error=error_msg,
                success=False,
                metadata={"error_type": "allowlist_violation"},
            )

        try:
            if function_tool:
                result = await function_tool.acall(**kwargs)
            elif http_tool:
                result = await http_tool.execute(**kwargs)
            elif mcp_tool:
                result = await mcp_tool.execute(**kwargs)

            if resolved_name in self.execution_stats:
                stats = self.execution_stats[resolved_name]
                stats["total_time"] += result.execution_time
                if result.success:
                    stats["successes"] += 1
                else:
                    stats["failures"] += 1

            return result

        except Exception as e:
            error_msg = f"Registry error executing '{resolved_name}': {str(e)}"
            if self.enable_logging:
                logger.debug(error_msg, exc_info=True)
            logger.error(error_msg)

            if resolved_name in self.execution_stats:
                self.execution_stats[resolved_name]["failures"] += 1

            return ToolResult(
                name=resolved_name,
                input=kwargs,
                output=None,
                error=error_msg,
                success=False,
                metadata={"error_type": "registry_error", "original_error": str(e)},
            )

    async def execute_safe_with_context(self, tool_name: str, context: Any, **kwargs) -> ToolResult:
        """Execute tool with context as first parameter (Pydantic AI pattern)."""
        # Resolve sanitized name back to original (for OpenAI compatibility)
        resolved_name = self._resolve_name(tool_name)

        if resolved_name not in self.tools:
            available_tools = list(self.tools.keys()) + list(self.http_tools.keys())
            return ToolResult(
                name=resolved_name,
                input=kwargs,
                success=False,
                error=f"Tool '{tool_name}' not found. Available: {available_tools}",
            )

        if resolved_name in self.execution_stats:
            self.execution_stats[resolved_name]["calls"] += 1

        tool = self.tools[resolved_name]
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

            if resolved_name in self.execution_stats:
                stats = self.execution_stats[resolved_name]
                stats["total_time"] += execution_time
                stats["successes"] += 1

            return ToolResult(
                name=resolved_name,
                input={"context": "<RunContext>", **kwargs},
                output=result,
                success=True,
                execution_time=execution_time,
                metadata={"execution_pattern": "first_param"},
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Tool '{resolved_name}' failed: {str(e)}"
            logger.error(error_msg)

            if resolved_name in self.execution_stats:
                self.execution_stats[resolved_name]["failures"] += 1

            return ToolResult(
                name=resolved_name,
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
