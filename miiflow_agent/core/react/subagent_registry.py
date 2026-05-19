"""Dynamic subagent configuration and registry.

This module provides a flexible system for defining and managing
specialized subagents that can be dynamically spawned during agent
execution.

Key features:
- ``DynamicSubAgentConfig``: declarative configuration with model /
  tools / prompt scoping
- ``SubAgentRegistry``: central registry for available subagent types

Default templates (``explorer`` / ``researcher`` / ``implementer`` /
``reviewer`` / ``planner``) used to ship as auto-registered defaults
but were removed because they referenced tool names the SDK doesn't
own. Equivalent templates now live in
``packages/miiflow-agent/examples/subagents.py`` as documentation —
copy them into your own setup code and adapt the ``tools`` list to the
tools you actually register.

To turn registered configs into a working ``dispatch_assistant`` tool,
see ``ConfiguredSubAgent`` and ``make_registry_dispatcher_tool`` in
``miiflow_agent.core.react.configured_subagent``.

Based on patterns from:
- Claude Agent SDK's AgentDefinition pattern
- Google ADK's specialized agent delegation
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DynamicSubAgentConfig:
    """Enhanced configuration for a specialized subagent.

    This extends the basic SubAgentConfig with additional capabilities:
    - Model selection per subagent (haiku/sonnet/opus)
    - Tool scoping (limit which tools this agent can use)
    - Specialized system prompts
    - Nesting control (can this agent spawn subagents)

    Attributes:
        name: Unique identifier for this subagent type
        description: When to use this agent (helps lead agent decide)
        system_prompt: Specialized instructions for this agent
        tools: List of allowed tool names (None = all tools)
        model: Model override (haiku/sonnet/opus, None = use default)
        max_steps: Maximum reasoning steps for this agent
        timeout_seconds: Maximum execution time
        can_spawn_subagents: Whether this agent can dispatch to sub-agents
        output_schema: Optional JSON schema for structured output
        priority: Priority for selection when multiple agents match
    """
    name: str
    description: str
    system_prompt: str
    tools: Optional[List[str]] = None  # None means all available tools
    model: Optional[str] = None  # None means use default model
    max_steps: int = 25
    timeout_seconds: float = 360.0
    can_spawn_subagents: bool = False
    output_schema: Optional[Dict[str, Any]] = None
    priority: int = 0

    # Metadata for logging and debugging
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "tools": self.tools,
            "model": self.model,
            "max_steps": self.max_steps,
            "timeout_seconds": self.timeout_seconds,
            "can_spawn_subagents": self.can_spawn_subagents,
            "output_schema": self.output_schema,
            "priority": self.priority,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DynamicSubAgentConfig":
        """Create config from dictionary."""
        return cls(
            name=data["name"],
            description=data["description"],
            system_prompt=data["system_prompt"],
            tools=data.get("tools"),
            model=data.get("model"),
            max_steps=data.get("max_steps", 25),
            timeout_seconds=data.get("timeout_seconds", 360.0),
            can_spawn_subagents=data.get("can_spawn_subagents", False),
            output_schema=data.get("output_schema"),
            priority=data.get("priority", 0),
            metadata=data.get("metadata", {}),
        )

    def copy(self, **overrides) -> "DynamicSubAgentConfig":
        """Create a copy with optional overrides."""
        data = self.to_dict()
        data.update(overrides)
        return DynamicSubAgentConfig.from_dict(data)


class SubAgentRegistry:
    """Central registry for available subagent types.

    The registry maintains a collection of DynamicSubAgentConfig
    instances that can be looked up by name or matched to tasks. Ships
    empty — register the configs your application needs. See
    ``examples/subagents.py`` for adaptable templates.

    Usage:
        registry = SubAgentRegistry()

        # Register a custom agent
        registry.register(DynamicSubAgentConfig(
            name="explorer",
            description="Explore the codebase to find relevant files.",
            system_prompt="You are a codebase exploration specialist...",
            tools=["file_read"],
            max_steps=5,
        ))

        # Look it up
        explorer = registry.get("explorer")

        # Find agents suitable for a task
        agents = registry.find_by_task("search for files containing 'error'")
    """

    def __init__(self, register_defaults: bool = False):
        """Initialize the registry.

        Args:
            register_defaults: Deprecated. The default subagent templates
                that this flag used to register referenced tool names
                (``Glob``/``Grep``/``Read``/``Edit``/``Write``/``Bash``/
                ``WebSearch``/``WebFetch``) that this SDK does not ship
                — registering them produced configs that looked real
                but couldn't dispatch. Passing ``True`` now emits a
                ``DeprecationWarning`` and is otherwise a no-op. See
                ``examples/subagents.py`` for adaptable template
                configs.
        """
        self._agents: Dict[str, DynamicSubAgentConfig] = {}

        if register_defaults:
            warnings.warn(
                "SubAgentRegistry(register_defaults=True) is deprecated "
                "and no longer registers any configs. The previous "
                "templates referenced tool names the SDK does not own; "
                "copy from examples/subagents.py and register the "
                "configs you actually want.",
                DeprecationWarning,
                stacklevel=2,
            )

    def register(self, config: DynamicSubAgentConfig) -> None:
        """Register a subagent configuration.

        Args:
            config: The subagent configuration to register
        """
        if config.name in self._agents:
            logger.warning(f"Overwriting existing subagent: {config.name}")

        self._agents[config.name] = config
        logger.debug(f"Registered subagent: {config.name}")

    def unregister(self, name: str) -> bool:
        """Remove a subagent from the registry.

        Args:
            name: Name of the subagent to remove

        Returns:
            True if removed, False if not found
        """
        if name in self._agents:
            del self._agents[name]
            logger.debug(f"Unregistered subagent: {name}")
            return True
        return False

    def get(self, name: str) -> Optional[DynamicSubAgentConfig]:
        """Get a subagent configuration by name.

        Args:
            name: Name of the subagent

        Returns:
            The configuration if found, None otherwise
        """
        return self._agents.get(name)

    def get_all(self) -> List[DynamicSubAgentConfig]:
        """Get all registered subagent configurations.

        Returns:
            List of all configurations, sorted by priority (descending)
        """
        return sorted(
            self._agents.values(),
            key=lambda a: a.priority,
            reverse=True
        )

    def find_by_task(
        self,
        task_description: str,
        required_tools: Optional[List[str]] = None,
        max_results: int = 3,
    ) -> List[DynamicSubAgentConfig]:
        """Find subagents suitable for a task.

        Uses heuristics to match task description to agent descriptions.

        Args:
            task_description: Description of the task
            required_tools: Optional list of tools the agent must have
            max_results: Maximum number of results to return

        Returns:
            List of matching configurations, sorted by relevance
        """
        task_lower = task_description.lower()
        matches = []

        # Keyword matching patterns for each agent type
        TASK_KEYWORDS = {
            "explorer": ["find", "search", "locate", "look for", "where is", "files", "codebase", "explore"],
            "researcher": ["research", "documentation", "docs", "web", "internet", "information", "learn about"],
            "implementer": ["implement", "write", "create", "build", "add", "fix", "modify", "change", "update"],
            "reviewer": ["review", "check", "analyze", "security", "bugs", "issues", "quality"],
            "planner": ["plan", "design", "architect", "break down", "strategy", "approach"],
        }

        for agent in self._agents.values():
            score = 0

            # Check required tools
            if required_tools:
                if agent.tools is None:
                    # Agent has all tools, so it's compatible
                    score += 1
                elif all(t in agent.tools for t in required_tools):
                    score += 2
                else:
                    # Missing required tools, skip this agent
                    continue

            # Check keyword matches
            keywords = TASK_KEYWORDS.get(agent.name, [])
            for keyword in keywords:
                if keyword in task_lower:
                    score += 3

            # Check if description matches
            desc_words = agent.description.lower().split()
            for word in task_lower.split():
                if len(word) > 3 and word in desc_words:
                    score += 1

            # Add priority bonus
            score += agent.priority * 0.1

            if score > 0:
                matches.append((score, agent))

        # Sort by score and return top matches
        matches.sort(key=lambda x: x[0], reverse=True)
        return [agent for _, agent in matches[:max_results]]

    def find_by_tools(self, required_tools: List[str]) -> List[DynamicSubAgentConfig]:
        """Find subagents that have all the required tools.

        Args:
            required_tools: List of tool names the agent must support

        Returns:
            List of matching configurations
        """
        matches = []
        for agent in self._agents.values():
            if agent.tools is None:
                # None means all tools are available
                matches.append(agent)
            elif all(t in agent.tools for t in required_tools):
                matches.append(agent)

        return sorted(matches, key=lambda a: a.priority, reverse=True)

    @property
    def names(self) -> List[str]:
        """Get list of all registered subagent names."""
        return list(self._agents.keys())

    def __len__(self) -> int:
        """Number of registered subagents."""
        return len(self._agents)

    def __contains__(self, name: str) -> bool:
        """Check if a subagent is registered."""
        return name in self._agents


# Global registry instance (lazily initialized)
_global_registry: Optional[SubAgentRegistry] = None


def get_global_registry() -> SubAgentRegistry:
    """Get the global subagent registry.

    Returns:
        The global SubAgentRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = SubAgentRegistry()
    return _global_registry


def reset_global_registry() -> None:
    """Reset the global registry (useful for testing)."""
    global _global_registry
    _global_registry = None
