"""MCP Tool wrapper class."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Dict, Optional

from ..schemas import ParameterSchema, ToolResult, ToolSchema
from ..types import ParameterType, ToolType
from ..exceptions import MCPToolError

if TYPE_CHECKING:
    from .mcp_connection import MCPServerConnection

logger = logging.getLogger(__name__)


class MCPTool:
    """Wrapper for an MCP server tool.

    Similar to FunctionTool and HTTPTool, but delegates execution
    to an MCP server connection. Tools are namespaced with the server
    name to prevent collisions between tools from different servers.

    Attributes:
        name: Namespaced tool name ({server}__{tool})
        original_name: Original tool name from the MCP server
        schema: Internal ToolSchema representation
    """

    def __init__(
        self,
        mcp_tool_definition: Any,
        connection: MCPServerConnection,
        server_name: str,
    ):
        """Initialize MCPTool from MCP tool definition.

        Args:
            mcp_tool_definition: Tool definition from session.list_tools()
            connection: Active MCP server connection
            server_name: Name of the MCP server (for namespacing)
        """
        self._mcp_definition = mcp_tool_definition
        self._connection = connection
        self._server_name = server_name

        # Build internal schema from MCP definition
        self.schema = self._build_schema()
        self.name = self.schema.name

    @property
    def original_name(self) -> str:
        """Original tool name as defined by the MCP server."""
        return self._mcp_definition.name

    @property
    def namespaced_name(self) -> str:
        """Tool name with server namespace prefix."""
        return f"{self._server_name}__{self._mcp_definition.name}"

    @property
    def server_name(self) -> str:
        """Name of the MCP server this tool belongs to."""
        return self._server_name

    @property
    def description(self) -> str:
        """Tool description."""
        return self.schema.description

    @property
    def definition(self) -> ToolSchema:
        """Alias for schema property (compatibility with FunctionTool)."""
        return self.schema

    def _build_schema(self) -> ToolSchema:
        """Convert MCP tool definition to internal ToolSchema.

        Maps MCP's JSON Schema format to our internal ParameterSchema format.
        """
        mcp_def = self._mcp_definition

        # Parse input schema from MCP format (JSON Schema)
        parameters = {}
        input_schema = getattr(mcp_def, "inputSchema", {}) or {}

        if isinstance(input_schema, dict):
            properties = input_schema.get("properties", {})
            required = input_schema.get("required", [])

            for param_name, param_def in properties.items():
                param_type_str = param_def.get("type", "string")

                # Map JSON Schema type to ParameterType
                type_mapping = {
                    "string": ParameterType.STRING,
                    "integer": ParameterType.INTEGER,
                    "number": ParameterType.NUMBER,
                    "boolean": ParameterType.BOOLEAN,
                    "array": ParameterType.ARRAY,
                    "object": ParameterType.OBJECT,
                    "null": ParameterType.NULL,
                }

                # Handle type arrays (e.g., ["string", "null"])
                if isinstance(param_type_str, list):
                    # Use first non-null type
                    for t in param_type_str:
                        if t != "null":
                            param_type_str = t
                            break
                    else:
                        param_type_str = "string"

                param_type = type_mapping.get(param_type_str, ParameterType.STRING)

                parameters[param_name] = ParameterSchema(
                    name=param_name,
                    type=param_type,
                    description=param_def.get("description", ""),
                    required=param_name in required,
                    default=param_def.get("default"),
                    enum=param_def.get("enum"),
                    items=param_def.get("items"),
                    properties=param_def.get("properties"),
                    minimum=param_def.get("minimum"),
                    maximum=param_def.get("maximum"),
                    pattern=param_def.get("pattern"),
                )

        # Get description, defaulting to a generic one
        description = getattr(mcp_def, "description", None)
        if not description:
            description = f"MCP tool: {mcp_def.name}"

        return ToolSchema(
            name=self.namespaced_name,
            description=description,
            tool_type=ToolType.MCP,
            parameters=parameters,
            metadata={
                "mcp_server": self._server_name,
                "original_name": mcp_def.name,
            },
        )

    async def execute(self, **kwargs) -> ToolResult:
        """Execute the MCP tool via the server connection.

        Args:
            **kwargs: Tool arguments

        Returns:
            ToolResult with output or error information
        """
        start_time = time.time()

        try:
            # Call tool using original name (server doesn't know about namespacing)
            result = await self._connection.call_tool(
                self.original_name, arguments=kwargs
            )

            execution_time = time.time() - start_time

            # Extract content from MCP result
            output = self._extract_output(result)

            return ToolResult(
                name=self.name,
                input=kwargs,
                output=output,
                success=True,
                execution_time=execution_time,
                metadata={
                    "mcp_server": self._server_name,
                    "original_name": self.original_name,
                },
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"MCP tool '{self.name}' failed: {str(e)}"
            logger.error(error_msg)

            return ToolResult(
                name=self.name,
                input=kwargs,
                output=None,
                error=error_msg,
                success=False,
                execution_time=execution_time,
                metadata={
                    "mcp_server": self._server_name,
                    "original_name": self.original_name,
                    "error_type": type(e).__name__,
                },
            )

    async def acall(self, **kwargs) -> ToolResult:
        """Async execute (alias for execute, for compatibility with FunctionTool)."""
        return await self.execute(**kwargs)

    def _extract_output(self, mcp_result: Any) -> Any:
        """Extract output from MCP call_tool result.

        MCP results can contain:
        - structuredContent: Parsed structured data
        - content: List of content blocks (text, image, resource)

        Args:
            mcp_result: Raw result from MCP call_tool

        Returns:
            Extracted output value
        """
        # Prefer structured content if available
        if hasattr(mcp_result, "structuredContent") and mcp_result.structuredContent:
            return mcp_result.structuredContent

        # Otherwise extract from content blocks
        if hasattr(mcp_result, "content") and mcp_result.content:
            outputs = []

            for block in mcp_result.content:
                # Handle different content types
                block_type = getattr(block, "type", None)

                if block_type == "text" or hasattr(block, "text"):
                    outputs.append(getattr(block, "text", str(block)))
                elif block_type == "image":
                    outputs.append(
                        {
                            "type": "image",
                            "data": getattr(block, "data", None),
                            "mimeType": getattr(block, "mimeType", None),
                        }
                    )
                elif block_type == "resource":
                    outputs.append(
                        {
                            "type": "resource",
                            "resource": str(getattr(block, "resource", block)),
                        }
                    )
                else:
                    # Fallback for unknown types
                    outputs.append(str(block))

            # Return single item if only one, otherwise list
            if len(outputs) == 1:
                return outputs[0]
            return outputs

        return None

    def to_provider_format(self, provider: str) -> Dict[str, Any]:
        """Convert to provider-specific format.

        Args:
            provider: Provider name (openai, anthropic, gemini, etc.)

        Returns:
            Provider-formatted tool schema
        """
        return self.schema.to_provider_format(provider)

    def validate_inputs(self, **kwargs) -> Dict[str, Any]:
        """Validate inputs against schema.

        Args:
            **kwargs: Input arguments to validate

        Returns:
            Validated inputs dict

        Raises:
            MCPToolError: If validation fails
        """
        parameters = self.schema.parameters
        validated = {}

        # Check required parameters
        for param_name, param_schema in parameters.items():
            if param_schema.required and param_name not in kwargs:
                raise MCPToolError(
                    f"Missing required parameter '{param_name}' for MCP tool '{self.name}'"
                )

        # Copy provided parameters
        for param_name, value in kwargs.items():
            if param_name in parameters:
                validated[param_name] = value
            else:
                # MCP tools might accept additional parameters not in schema
                logger.debug(
                    f"Unknown parameter '{param_name}' for MCP tool '{self.name}'"
                )
                validated[param_name] = value

        return validated

    def __repr__(self) -> str:
        return (
            f"MCPTool(name='{self.name}', "
            f"server='{self._server_name}', "
            f"original_name='{self.original_name}')"
        )
