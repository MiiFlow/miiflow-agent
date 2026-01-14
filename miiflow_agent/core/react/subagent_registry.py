"""Dynamic subagent configuration and registry.

This module provides a flexible system for defining and managing specialized subagents
that can be dynamically spawned during agent execution.

Key features:
- DynamicSubAgentConfig: Enhanced configuration with model/tools/prompt scoping
- SubAgentRegistry: Central registry for available subagent types
- Default subagent types: explorer, researcher, implementer, reviewer
- Support for custom subagent registration

Based on patterns from:
- Claude Agent SDK's AgentDefinition pattern
- Google ADK's specialized agent delegation
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Default prompts for specialized subagents
EXPLORER_PROMPT = """You are a codebase exploration specialist.

Your role is to efficiently find and understand code:
- Use Glob to find files by pattern
- Use Grep to search for specific code patterns
- Use Read to examine file contents

Be thorough but focused. Report what you find clearly and concisely.
Do not make changes to any files - observation only."""

RESEARCHER_PROMPT = """You are a research specialist focused on gathering information.

Your role is to find accurate, up-to-date information:
- Use WebSearch to find relevant sources
- Use WebFetch to read page content
- Cross-reference multiple sources for accuracy

Cite your sources and note the date of information when relevant.
Focus on authoritative sources and official documentation."""

IMPLEMENTER_PROMPT = """You are a code implementation specialist.

Your role is to write clean, correct code:
- Read existing code to understand conventions
- Make focused, minimal changes
- Follow project patterns and style guides
- Test changes when possible using Bash

Write code that is:
- Well-structured and readable
- Consistent with existing codebase
- Properly handling edge cases"""

REVIEWER_PROMPT = """You are a code review specialist.

Your role is to analyze code for:
- Bugs and logic errors
- Security vulnerabilities
- Performance issues
- Code style and best practices

Be thorough but constructive. Explain issues clearly and suggest improvements.
Do not make changes - only report findings."""

PLANNER_PROMPT = """You are a planning and architecture specialist.

Your role is to:
- Break down complex tasks into manageable steps
- Identify dependencies between tasks
- Consider trade-offs between approaches
- Plan for error handling and edge cases

Provide clear, actionable plans with specific steps."""


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
        can_spawn_subagents: Whether this agent can use TaskTool
        output_schema: Optional JSON schema for structured output
        priority: Priority for selection when multiple agents match
    """
    name: str
    description: str
    system_prompt: str
    tools: Optional[List[str]] = None  # None means all available tools
    model: Optional[str] = None  # None means use default model
    max_steps: int = 10
    timeout_seconds: float = 120.0
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
            max_steps=data.get("max_steps", 10),
            timeout_seconds=data.get("timeout_seconds", 120.0),
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

    The registry maintains a collection of DynamicSubAgentConfig instances
    that can be looked up by name or matched to tasks.

    Usage:
        registry = SubAgentRegistry()

        # Get a specific agent
        explorer = registry.get("explorer")

        # Find agents suitable for a task
        agents = registry.find_by_task("search for files containing 'error'")

        # Register a custom agent
        registry.register(DynamicSubAgentConfig(
            name="custom_agent",
            description="Custom specialized agent",
            system_prompt="You are a custom agent...",
            tools=["Read", "Write"],
        ))
    """

    def __init__(self, register_defaults: bool = True):
        """Initialize the registry.

        Args:
            register_defaults: Whether to register default subagent types
        """
        self._agents: Dict[str, DynamicSubAgentConfig] = {}

        if register_defaults:
            self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default subagent types."""

        # Explorer: Fast codebase navigation
        self.register(DynamicSubAgentConfig(
            name="explorer",
            description="Explore codebase to find relevant files, understand patterns, and trace code paths. Use for file discovery, code navigation, and understanding project structure.",
            system_prompt=EXPLORER_PROMPT,
            tools=["Glob", "Grep", "Read"],
            model="haiku",  # Fast model for exploration
            max_steps=5,
            timeout_seconds=60.0,
            priority=10,
        ))

        # Researcher: Web search and information gathering
        self.register(DynamicSubAgentConfig(
            name="researcher",
            description="Search the web and gather information on topics. Use for documentation lookups, finding solutions, and researching best practices.",
            system_prompt=RESEARCHER_PROMPT,
            tools=["WebSearch", "WebFetch"],
            model="haiku",  # Fast for web lookups
            max_steps=5,
            timeout_seconds=60.0,
            priority=5,
        ))

        # Implementer: Code writing and modification
        self.register(DynamicSubAgentConfig(
            name="implementer",
            description="Write and modify code. Use for implementing features, fixing bugs, and making code changes.",
            system_prompt=IMPLEMENTER_PROMPT,
            tools=["Read", "Edit", "Write", "Bash", "Glob"],
            model="sonnet",  # Better model for code generation
            max_steps=15,
            timeout_seconds=180.0,
            priority=8,
        ))

        # Reviewer: Code review and analysis
        self.register(DynamicSubAgentConfig(
            name="reviewer",
            description="Review code for bugs, security issues, and best practices. Use for code quality analysis and finding issues.",
            system_prompt=REVIEWER_PROMPT,
            tools=["Read", "Glob", "Grep"],
            model="sonnet",  # Need good reasoning for review
            max_steps=10,
            timeout_seconds=120.0,
            priority=7,
        ))

        # Planner: Architecture and planning
        self.register(DynamicSubAgentConfig(
            name="planner",
            description="Plan implementation approaches and break down complex tasks. Use for architecture decisions and multi-step planning.",
            system_prompt=PLANNER_PROMPT,
            tools=["Read", "Glob", "Grep"],
            model="sonnet",  # Need good reasoning for planning
            max_steps=8,
            timeout_seconds=90.0,
            can_spawn_subagents=True,  # Planner can delegate
            priority=9,
        ))

        logger.info(f"Registered {len(self._agents)} default subagent types")

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
