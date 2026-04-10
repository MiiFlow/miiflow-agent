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

    def __init__(
        self,
        allowlist: Optional[List[str]] = None,
        enable_logging: bool = True,
        *,
        tool_search_enabled: bool = True,
        tool_search_threshold: Optional[int] = None,
    ):
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

        # ToolSearch: provider-agnostic deferred-tool discovery.
        from . import tool_search as _tool_search_mod

        self.tool_search_enabled = tool_search_enabled
        self.tool_search_threshold = (
            tool_search_threshold
            if tool_search_threshold is not None
            else _tool_search_mod.DEFAULT_TOOL_SEARCH_THRESHOLD
        )
        # Cached lowercase searchable text per tool name (built lazily).
        self._search_index: Dict[str, str] = {}
        # The built-in tool_search FunctionTool, lazily built on first need.
        self._tool_search_tool: Optional[FunctionTool] = None

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

        self._index_tool_for_search(
            tool_name,
            getattr(tool.definition, "description", "") or "",
            getattr(tool.definition, "metadata", {}) or {},
        )

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

        self._index_tool_for_search(schema.name, schema.description or "", schema.metadata or {})

        if self.enable_logging:
            logger.info(f"Registered HTTP tool: {schema.name} -> {schema.url}")

    def _index_tool_for_search(
        self,
        tool_name: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Cache the lowercase searchable text for a tool."""
        from . import tool_search as _tool_search_mod

        self._search_index[tool_name] = _tool_search_mod.build_searchable_text(
            tool_name, description or "", metadata or {}
        ).lower()

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

        self._index_tool_for_search(
            tool.name,
            getattr(tool, "description", "") or getattr(getattr(tool, "schema", None), "description", "") or "",
            getattr(getattr(tool, "schema", None), "metadata", {}) or {},
        )

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

    # ------------------------------------------------------------------
    # ToolSearch (provider-agnostic deferred-tool discovery)
    # ------------------------------------------------------------------

    def _iter_all_tool_entries(self):
        """Yield (name, description, metadata) for every registered tool."""
        for name, t in self.tools.items():
            md = getattr(getattr(t, "definition", None), "metadata", {}) or {}
            desc = getattr(getattr(t, "definition", None), "description", "") or ""
            yield name, desc, md
        for name, t in self.http_tools.items():
            md = getattr(getattr(t, "schema", None), "metadata", {}) or {}
            desc = getattr(getattr(t, "schema", None), "description", "") or ""
            yield name, desc, md
        for name, t in self.mcp_tools.items():
            md = getattr(getattr(t, "schema", None), "metadata", {}) or {}
            desc = getattr(t, "description", "") or getattr(getattr(t, "schema", None), "description", "") or ""
            yield name, desc, md

    def total_tool_count(self) -> int:
        """Number of registered tools across all kinds (excluding the meta-tool)."""
        return len(self.tools) + len(self.http_tools) + len(self.mcp_tools)

    def should_use_tool_search(self, threshold: Optional[int] = None) -> bool:
        """Whether ToolSearch should activate for the next LLM call.

        Activates when ``tool_search_enabled`` is True and the registered tool
        count exceeds ``threshold`` (or ``self.tool_search_threshold``).
        """
        if not self.tool_search_enabled:
            return False
        limit = threshold if threshold is not None else self.tool_search_threshold
        return self.total_tool_count() > limit

    def get_always_load_names(self) -> List[str]:
        """Names of tools flagged ``always_load`` in their schema metadata."""
        names: List[str] = []
        for name, _desc, md in self._iter_all_tool_entries():
            if md.get("always_load"):
                names.append(name)
        return names

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Lexically rank registered tools against ``query``.

        Returns up to ``max_results`` entries shaped as
        ``{"name", "description", "parameters"}`` (universal JSON schema).
        The built-in ``tool_search`` meta-tool is excluded from results.
        """
        from . import tool_search as _tool_search_mod

        query_tokens = _tool_search_mod._tokenize(query or "")
        if not query_tokens:
            return []

        scored: List[tuple] = []
        for name, _desc, _md in self._iter_all_tool_entries():
            if name == _tool_search_mod.TOOL_SEARCH_TOOL_NAME:
                continue
            text = self._search_index.get(name)
            if text is None:
                # Late-indexed tool (registered before this build); skip silently.
                continue
            score = _tool_search_mod.score_tool(query_tokens, text)
            if score > 0:
                scored.append((score, name))

        scored.sort(key=lambda x: x[0], reverse=True)

        results: List[Dict[str, Any]] = []
        for _score, name in scored[:max_results]:
            schema = self._get_universal_schema(name)
            if schema is not None:
                results.append(schema)
        return results

    def _get_universal_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """Return the universal-format schema for any registered tool by name."""
        if name in self.tools:
            return self.tools[name].definition.to_universal_schema()
        if name in self.http_tools:
            return self.http_tools[name].schema.to_universal_schema()
        if name in self.mcp_tools:
            return self.mcp_tools[name].schema.to_universal_schema()
        return None

    def get_tool_search_tool(self) -> "FunctionTool":
        """Return (and lazily build) the ``tool_search`` meta-tool.

        The tool is *not* inserted into ``self.tools``: it lives on its own
        attribute so it never leaks into ``list_tools()``, ``get_schemas()``,
        or any code that iterates the registry. ``execute_safe`` and
        ``get_filtered_schemas`` route to it explicitly.
        """
        from . import tool_search as _tool_search_mod

        if self._tool_search_tool is None:
            self._tool_search_tool = _tool_search_mod.build_tool_search_tool(self)
            self.execution_stats[self._tool_search_tool.name] = {
                "calls": 0,
                "successes": 0,
                "failures": 0,
                "total_time": 0.0,
            }
        return self._tool_search_tool

    def get_filtered_schemas(
        self,
        provider: str,
        client=None,
        enabled_names: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        """Return schemas for the next LLM turn under ToolSearch.

        Always includes: the ``tool_search`` meta-tool, every tool flagged
        ``always_load``, and any tools the model has already discovered (via
        ``enabled_names``). All other tools are hidden until ``tool_search``
        surfaces them.
        """
        # Build the meta-tool (lives off-registry so it never leaks into
        # legacy schema callers).
        meta_tool = self.get_tool_search_tool()

        visible: set = set(self.get_always_load_names())
        if enabled_names:
            visible.update(enabled_names)

        schemas: List[Dict[str, Any]] = []

        def _format(universal_or_tool, formatter_obj) -> Dict[str, Any]:
            if client and hasattr(client, "convert_schema_to_provider_format"):
                return client.convert_schema_to_provider_format(
                    formatter_obj.to_universal_schema()
                )
            return formatter_obj.to_provider_format(provider)

        # Always include the meta-tool first.
        schemas.append(_format(meta_tool, meta_tool.definition))

        for name, tool in self.tools.items():
            if name not in visible:
                continue
            schemas.append(_format(tool, tool.definition))
        for name, http_tool in self.http_tools.items():
            if name not in visible:
                continue
            schemas.append(_format(http_tool, http_tool.schema))
        for name, mcp_tool in self.mcp_tools.items():
            if name not in visible:
                continue
            if client and hasattr(client, "convert_schema_to_provider_format"):
                schemas.append(
                    client.convert_schema_to_provider_format(mcp_tool.schema.to_universal_schema())
                )
            else:
                schemas.append(mcp_tool.to_provider_format(provider))

        return schemas

    def validate_tool_call(self, name: str, **kwargs) -> bool:
        """Validate a tool call against schema and allowlist."""
        # Resolve sanitized name back to original (for OpenAI compatibility)
        resolved_name = self._resolve_name(name)

        # The off-registry tool_search meta-tool is always valid when active.
        if (
            self._tool_search_tool is not None
            and resolved_name == self._tool_search_tool.name
        ):
            return True

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

        # Route the off-registry tool_search meta-tool, if active.
        if (
            self._tool_search_tool is not None
            and resolved_name == self._tool_search_tool.name
        ):
            try:
                result = await self._tool_search_tool.acall(**kwargs)
                stats = self.execution_stats.get(resolved_name)
                if stats is not None:
                    stats["total_time"] += result.execution_time
                    if result.success:
                        stats["successes"] += 1
                    else:
                        stats["failures"] += 1
                return result
            except Exception as e:
                if resolved_name in self.execution_stats:
                    self.execution_stats[resolved_name]["failures"] += 1
                return ToolResult(
                    name=resolved_name,
                    input=kwargs,
                    output=None,
                    error=f"tool_search execution failed: {e}",
                    success=False,
                    metadata={"error_type": "tool_search_error"},
                )

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

        # Validate kwargs against the tool's schema - reject unknown parameters
        # so the LLM can correct its tool call
        if hasattr(tool, "definition") and hasattr(tool.definition, "parameters"):
            valid_params = set(tool.definition.parameters.keys())
            unknown_params = set(kwargs.keys()) - valid_params
            if unknown_params:
                error_msg = (
                    f"Tool '{resolved_name}' received unknown parameter(s): {sorted(unknown_params)}. "
                    f"Valid parameters are: {sorted(valid_params)}"
                )
                logger.warning(error_msg)
                return ToolResult(
                    name=resolved_name,
                    input=kwargs,
                    output=None,
                    error=error_msg,
                    success=False,
                    execution_time=0.0,
                    metadata={"error_type": "invalid_parameters", "unknown_params": list(unknown_params)},
                )

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
