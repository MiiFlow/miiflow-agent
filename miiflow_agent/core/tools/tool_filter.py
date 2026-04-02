"""Tool pool narrowing for restricting tool availability by role or mode.

Different agent roles should have access to different tool subsets:
- Researchers get read-only tools
- Implementers get all tools
- Analyzers get read + analysis tools

Inspired by Claude Code's progressive tool narrowing pattern where
coordinators, workers, and background agents get different tool sets.
"""

import logging
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class ToolFilter:
    """Restricts tool availability based on allowlist/denylist rules.

    Tools can be filtered by:
    - allowed_tools: Only these tools are available (allowlist)
    - denied_tools: These tools are excluded (denylist)

    If both are set, allowed_tools takes precedence (only allowed tools
    that are not denied are available).
    """

    def __init__(
        self,
        allowed_tools: Optional[List[str]] = None,
        denied_tools: Optional[List[str]] = None,
    ):
        """Initialize tool filter.

        Args:
            allowed_tools: Allowlist of tool names. None means all tools allowed.
            denied_tools: Denylist of tool names. None means no tools denied.
        """
        self._allowed: Optional[Set[str]] = set(allowed_tools) if allowed_tools is not None else None
        self._denied: Set[str] = set(denied_tools) if denied_tools else set()

    def is_allowed(self, tool_name: str) -> bool:
        """Check if a specific tool is allowed.

        Args:
            tool_name: Name of the tool to check.

        Returns:
            True if the tool is allowed by this filter.
        """
        if tool_name in self._denied:
            return False
        if self._allowed is not None:
            return tool_name in self._allowed
        return True

    def filter_schemas(self, schemas: List[Dict]) -> List[Dict]:
        """Filter tool schemas before passing to LLM.

        Args:
            schemas: List of tool schema dicts (provider-specific format).

        Returns:
            Filtered list with only allowed tools.
        """
        if self._allowed is None and not self._denied:
            return schemas

        filtered = []
        for schema in schemas:
            # Extract tool name from various schema formats
            name = self._extract_tool_name(schema)
            if name and self.is_allowed(name):
                filtered.append(schema)

        if len(filtered) != len(schemas):
            logger.debug(
                f"ToolFilter: {len(schemas)} -> {len(filtered)} tools "
                f"(denied={len(self._denied)}, "
                f"allowlist={'yes' if self._allowed is not None else 'no'})"
            )

        return filtered

    def filter_tool_names(self, tool_names: List[str]) -> List[str]:
        """Filter a list of tool names.

        Args:
            tool_names: List of tool names.

        Returns:
            Filtered list with only allowed tools.
        """
        return [name for name in tool_names if self.is_allowed(name)]

    def add_denied(self, tool_name: str) -> None:
        """Add a tool to the deny list at runtime.

        Args:
            tool_name: Name of the tool to deny.
        """
        self._denied.add(tool_name)

    def _extract_tool_name(self, schema: Dict) -> Optional[str]:
        """Extract tool name from a provider-specific schema dict."""
        # OpenAI format: {"type": "function", "function": {"name": "..."}}
        if "function" in schema and isinstance(schema["function"], dict):
            return schema["function"].get("name")
        # Anthropic format: {"name": "...", "description": "..."}
        if "name" in schema:
            return schema["name"]
        # Universal format: {"name": "...", "parameters": {...}}
        return schema.get("name")

    @classmethod
    def for_role(cls, role: str, available_tools: Optional[List[str]] = None) -> "ToolFilter":
        """Create a filter appropriate for a subagent role.

        Predefined role-based filters:
        - "researcher": Read-only tools (search, fetch, read)
        - "analyzer": Read + analysis tools
        - "implementer": All tools
        - "summarizer": No tools (text generation only)

        For unknown roles, returns an unrestricted filter.

        Args:
            role: Agent role name.
            available_tools: List of all available tool names (for context).

        Returns:
            ToolFilter configured for the role.
        """
        role_lower = role.lower()

        if role_lower == "summarizer":
            # Summarizers don't need any tools
            return cls(allowed_tools=[])

        if role_lower == "researcher":
            # Researchers get read-only tools
            read_patterns = {"search", "fetch", "read", "get", "list", "find", "lookup", "query"}
            if available_tools:
                allowed = [
                    t for t in available_tools
                    if any(p in t.lower() for p in read_patterns)
                ]
                return cls(allowed_tools=allowed) if allowed else cls()
            return cls()

        if role_lower == "analyzer":
            # Analyzers get read + analysis but not write/execute
            write_patterns = {"write", "create", "delete", "update", "execute", "run", "bash"}
            if available_tools:
                denied = [
                    t for t in available_tools
                    if any(p in t.lower() for p in write_patterns)
                ]
                return cls(denied_tools=denied) if denied else cls()
            return cls()

        # "implementer", "coder", or unknown roles get all tools
        return cls()

    def __repr__(self) -> str:
        allowed_str = f"allowed={sorted(self._allowed)}" if self._allowed is not None else "allowed=*"
        denied_str = f"denied={sorted(self._denied)}" if self._denied else ""
        parts = [allowed_str]
        if denied_str:
            parts.append(denied_str)
        return f"ToolFilter({', '.join(parts)})"
