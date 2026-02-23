"""Tests for dynamic subagent components.

This module tests:
- SubAgentRegistry: Subagent registration and lookup
- DynamicSubAgentConfig: Configuration dataclass
- ModelSelector: Model selection based on task and complexity
- ComplexityDetector: Task complexity detection
- TaskTool: Hierarchical agent spawning (basic structure tests)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from miiflow_agent.core.react.subagent_registry import (
    DynamicSubAgentConfig,
    SubAgentRegistry,
    get_global_registry,
    reset_global_registry,
    EXPLORER_PROMPT,
    RESEARCHER_PROMPT,
    IMPLEMENTER_PROMPT,
    REVIEWER_PROMPT,
    PLANNER_PROMPT,
)
from miiflow_agent.core.react.model_selector import (
    ModelSelector,
    ModelTier,
    TaskComplexity,
    ComplexityDetector,
    ModelCapabilities,
    MODEL_PROFILES,
    select_model_for_task,
    detect_complexity,
)
from miiflow_agent.core.react.task_tool import (
    TaskTool,
    TaskToolResult,
    TaskToolConfig,
    create_task_tool,
    MAX_NESTING_DEPTH,
    DEFAULT_SUBAGENT_TIMEOUT,
)


class TestDynamicSubAgentConfig:
    """Test DynamicSubAgentConfig dataclass."""

    def test_basic_creation(self):
        """Test creating a basic config."""
        config = DynamicSubAgentConfig(
            name="test_agent",
            description="A test agent",
            system_prompt="You are a test agent.",
        )

        assert config.name == "test_agent"
        assert config.description == "A test agent"
        assert config.system_prompt == "You are a test agent."
        assert config.tools is None  # Default
        assert config.model is None  # Default
        assert config.max_steps == 10  # Default
        assert config.timeout_seconds == 360.0  # Default
        assert config.can_spawn_subagents is False  # Default

    def test_full_creation(self):
        """Test creating a config with all parameters."""
        config = DynamicSubAgentConfig(
            name="full_agent",
            description="A fully configured agent",
            system_prompt="You are fully configured.",
            tools=["Read", "Write", "Bash"],
            model="sonnet",
            max_steps=20,
            timeout_seconds=300.0,
            can_spawn_subagents=True,
            output_schema={"type": "object"},
            priority=5,
            metadata={"custom": "value"},
        )

        assert config.name == "full_agent"
        assert config.tools == ["Read", "Write", "Bash"]
        assert config.model == "sonnet"
        assert config.max_steps == 20
        assert config.timeout_seconds == 300.0
        assert config.can_spawn_subagents is True
        assert config.output_schema == {"type": "object"}
        assert config.priority == 5
        assert config.metadata == {"custom": "value"}

    def test_to_dict(self):
        """Test serialization to dictionary."""
        config = DynamicSubAgentConfig(
            name="test",
            description="Test",
            system_prompt="Prompt",
            tools=["Read"],
            model="haiku",
        )

        data = config.to_dict()

        assert data["name"] == "test"
        assert data["description"] == "Test"
        assert data["system_prompt"] == "Prompt"
        assert data["tools"] == ["Read"]
        assert data["model"] == "haiku"
        assert data["max_steps"] == 10
        assert data["can_spawn_subagents"] is False

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "name": "from_dict_agent",
            "description": "Created from dict",
            "system_prompt": "From dict prompt",
            "tools": ["Grep", "Glob"],
            "model": "opus",
            "max_steps": 15,
            "priority": 3,
        }

        config = DynamicSubAgentConfig.from_dict(data)

        assert config.name == "from_dict_agent"
        assert config.tools == ["Grep", "Glob"]
        assert config.model == "opus"
        assert config.max_steps == 15
        assert config.priority == 3

    def test_copy_with_overrides(self):
        """Test creating a copy with overrides."""
        original = DynamicSubAgentConfig(
            name="original",
            description="Original config",
            system_prompt="Original prompt",
            model="haiku",
            can_spawn_subagents=True,
        )

        copy = original.copy(
            model="sonnet",
            can_spawn_subagents=False,
            max_steps=5,
        )

        # Original unchanged
        assert original.model == "haiku"
        assert original.can_spawn_subagents is True
        assert original.max_steps == 10

        # Copy has overrides
        assert copy.name == "original"  # Preserved
        assert copy.model == "sonnet"  # Overridden
        assert copy.can_spawn_subagents is False  # Overridden
        assert copy.max_steps == 5  # Overridden


class TestSubAgentRegistry:
    """Test SubAgentRegistry functionality."""

    def test_default_registration(self):
        """Test that default agents are registered."""
        registry = SubAgentRegistry(register_defaults=True)

        assert "explorer" in registry
        assert "researcher" in registry
        assert "implementer" in registry
        assert "reviewer" in registry
        assert "planner" in registry
        assert len(registry) == 5

    def test_no_default_registration(self):
        """Test registry without defaults."""
        registry = SubAgentRegistry(register_defaults=False)

        assert len(registry) == 0
        assert "explorer" not in registry

    def test_register_custom_agent(self):
        """Test registering a custom agent."""
        registry = SubAgentRegistry(register_defaults=False)

        custom_config = DynamicSubAgentConfig(
            name="custom",
            description="Custom agent",
            system_prompt="Custom prompt",
        )

        registry.register(custom_config)

        assert "custom" in registry
        assert registry.get("custom") == custom_config

    def test_get_existing_agent(self):
        """Test getting an existing agent."""
        registry = SubAgentRegistry()

        explorer = registry.get("explorer")

        assert explorer is not None
        assert explorer.name == "explorer"
        assert explorer.model == "haiku"  # Explorer uses haiku
        assert "Glob" in explorer.tools
        assert "Grep" in explorer.tools
        assert "Read" in explorer.tools

    def test_get_nonexistent_agent(self):
        """Test getting a non-existent agent returns None."""
        registry = SubAgentRegistry()

        result = registry.get("nonexistent")

        assert result is None

    def test_unregister_agent(self):
        """Test unregistering an agent."""
        registry = SubAgentRegistry()

        assert "explorer" in registry
        result = registry.unregister("explorer")

        assert result is True
        assert "explorer" not in registry

    def test_unregister_nonexistent(self):
        """Test unregistering non-existent agent returns False."""
        registry = SubAgentRegistry()

        result = registry.unregister("nonexistent")

        assert result is False

    def test_get_all_sorted_by_priority(self):
        """Test get_all returns agents sorted by priority."""
        registry = SubAgentRegistry()

        all_agents = registry.get_all()

        # Check that they're sorted by priority (descending)
        priorities = [a.priority for a in all_agents]
        assert priorities == sorted(priorities, reverse=True)

    def test_names_property(self):
        """Test names property returns all agent names."""
        registry = SubAgentRegistry()

        names = registry.names

        assert isinstance(names, list)
        assert "explorer" in names
        assert "researcher" in names
        assert len(names) == 5

    def test_find_by_task_exploration(self):
        """Test finding agents for exploration task."""
        registry = SubAgentRegistry()

        agents = registry.find_by_task("find all Python files in the codebase")

        # Explorer should be a top match
        agent_names = [a.name for a in agents]
        assert "explorer" in agent_names

    def test_find_by_task_research(self):
        """Test finding agents for research task."""
        registry = SubAgentRegistry()

        agents = registry.find_by_task("research best practices for React hooks")

        agent_names = [a.name for a in agents]
        assert "researcher" in agent_names

    def test_find_by_task_implementation(self):
        """Test finding agents for implementation task."""
        registry = SubAgentRegistry()

        agents = registry.find_by_task("implement a new authentication feature")

        agent_names = [a.name for a in agents]
        assert "implementer" in agent_names

    def test_find_by_task_review(self):
        """Test finding agents for review task."""
        registry = SubAgentRegistry()

        agents = registry.find_by_task("review this code for security issues")

        agent_names = [a.name for a in agents]
        assert "reviewer" in agent_names

    def test_find_by_task_planning(self):
        """Test finding agents for planning task."""
        registry = SubAgentRegistry()

        agents = registry.find_by_task("plan the architecture for this system")

        agent_names = [a.name for a in agents]
        assert "planner" in agent_names

    def test_find_by_task_with_required_tools(self):
        """Test finding agents with required tools."""
        registry = SubAgentRegistry()

        agents = registry.find_by_task(
            "search for errors",
            required_tools=["Grep"],
        )

        # All returned agents should have Grep
        for agent in agents:
            assert agent.tools is None or "Grep" in agent.tools

    def test_find_by_tools(self):
        """Test finding agents by required tools."""
        registry = SubAgentRegistry()

        agents = registry.find_by_tools(["Read", "Edit"])

        # Implementer should be in the results
        agent_names = [a.name for a in agents]
        assert "implementer" in agent_names


class TestGlobalRegistry:
    """Test global registry functions."""

    def teardown_method(self):
        """Reset global registry after each test."""
        reset_global_registry()

    def test_get_global_registry_creates_instance(self):
        """Test that get_global_registry creates a registry."""
        registry = get_global_registry()

        assert registry is not None
        assert isinstance(registry, SubAgentRegistry)
        assert "explorer" in registry

    def test_get_global_registry_returns_same_instance(self):
        """Test that get_global_registry returns the same instance."""
        registry1 = get_global_registry()
        registry2 = get_global_registry()

        assert registry1 is registry2

    def test_reset_global_registry(self):
        """Test that reset_global_registry clears the instance."""
        registry1 = get_global_registry()
        reset_global_registry()
        registry2 = get_global_registry()

        assert registry1 is not registry2


class TestModelTier:
    """Test ModelTier enum."""

    def test_tier_values(self):
        """Test tier enum values."""
        assert ModelTier.HAIKU.value == "haiku"
        assert ModelTier.SONNET.value == "sonnet"
        assert ModelTier.OPUS.value == "opus"


class TestTaskComplexity:
    """Test TaskComplexity enum."""

    def test_complexity_values(self):
        """Test complexity enum values."""
        assert TaskComplexity.SIMPLE.value == "simple"
        assert TaskComplexity.MEDIUM.value == "medium"
        assert TaskComplexity.COMPLEX.value == "complex"
        assert TaskComplexity.ULTRATHINK.value == "ultrathink"


class TestModelProfiles:
    """Test MODEL_PROFILES configuration."""

    def test_all_tiers_have_profiles(self):
        """Test that all model tiers have profiles."""
        for tier in ModelTier:
            assert tier in MODEL_PROFILES

    def test_haiku_profile(self):
        """Test Haiku profile characteristics."""
        profile = MODEL_PROFILES[ModelTier.HAIKU]

        assert profile.speed == "fast"
        assert profile.cost_multiplier < 1.0  # Cheaper than baseline
        assert "exploration" in profile.best_for
        assert "search" in profile.best_for

    def test_sonnet_profile(self):
        """Test Sonnet profile characteristics."""
        profile = MODEL_PROFILES[ModelTier.SONNET]

        assert profile.speed == "medium"
        assert profile.cost_multiplier == 1.0  # Baseline
        assert "coding" in profile.best_for
        assert "analysis" in profile.best_for

    def test_opus_profile(self):
        """Test Opus profile characteristics."""
        profile = MODEL_PROFILES[ModelTier.OPUS]

        assert profile.speed == "slow"
        assert profile.cost_multiplier > 1.0  # More expensive
        assert "architecture" in profile.best_for
        assert "deep_analysis" in profile.best_for


class TestModelSelector:
    """Test ModelSelector functionality."""

    def test_default_model(self):
        """Test default model selection."""
        selector = ModelSelector()

        # Unknown task type should return default
        result = selector.select(task_type="unknown_task")

        assert result == "sonnet"  # Default

    def test_select_for_exploration(self):
        """Test model selection for exploration tasks."""
        selector = ModelSelector()

        result = selector.select(task_type="exploration")

        assert result == "haiku"

    def test_select_for_coding(self):
        """Test model selection for coding tasks."""
        selector = ModelSelector()

        result = selector.select(task_type="coding")

        assert result == "sonnet"

    def test_select_for_architecture(self):
        """Test model selection for architecture tasks."""
        selector = ModelSelector()

        result = selector.select(task_type="architecture")

        assert result == "opus"

    def test_complexity_adjustment_simple(self):
        """Test complexity adjustment for simple tasks."""
        selector = ModelSelector()

        result = selector.select(
            task_type="coding",  # Normally sonnet
            complexity=TaskComplexity.SIMPLE,
        )

        assert result == "haiku"  # Downgraded

    def test_complexity_adjustment_complex(self):
        """Test complexity adjustment for complex tasks."""
        selector = ModelSelector()

        result = selector.select(
            task_type="coding",  # Normally sonnet
            complexity=TaskComplexity.COMPLEX,
        )

        assert result == "opus"  # Upgraded

    def test_complexity_adjustment_ultrathink(self):
        """Test ultrathink always uses opus."""
        selector = ModelSelector()

        result = selector.select(
            task_type="exploration",  # Normally haiku
            complexity=TaskComplexity.ULTRATHINK,
        )

        assert result == "opus"

    def test_cost_constraint(self):
        """Test cost constraint limits model selection."""
        selector = ModelSelector()

        result = selector.select(
            task_type="architecture",  # Normally opus
            max_cost_multiplier=1.0,  # Limit to sonnet's cost
        )

        assert result in ["haiku", "sonnet"]  # Can't be opus

    def test_require_fast(self):
        """Test require_fast forces haiku."""
        selector = ModelSelector()

        result = selector.select(
            task_type="coding",
            require_fast=True,
        )

        assert result == "haiku"

    def test_select_for_task_convenience(self):
        """Test select_for_task convenience method."""
        selector = ModelSelector()

        result = selector.select_for_task("exploration")

        assert result == "haiku"

    def test_get_capabilities(self):
        """Test getting capabilities for a model."""
        selector = ModelSelector()

        caps = selector.get_capabilities("sonnet")

        assert caps is not None
        assert caps.tier == ModelTier.SONNET
        assert caps.cost_multiplier == 1.0

    def test_get_capabilities_unknown(self):
        """Test getting capabilities for unknown model."""
        selector = ModelSelector()

        caps = selector.get_capabilities("unknown")

        assert caps is None

    def test_estimate_cost(self):
        """Test cost estimation."""
        selector = ModelSelector()

        cost_haiku = selector.estimate_cost("haiku", 1000)
        cost_sonnet = selector.estimate_cost("sonnet", 1000)
        cost_opus = selector.estimate_cost("opus", 1000)

        # Haiku should be cheapest, opus most expensive
        assert cost_haiku < cost_sonnet < cost_opus

    def test_recommend_for_budget(self):
        """Test model recommendation within budget."""
        selector = ModelSelector()

        # Very small budget should recommend haiku
        result = selector.recommend_for_budget(
            budget_usd=0.001,
            estimated_tokens=1000,
        )

        assert result == "haiku"


class TestComplexityDetector:
    """Test ComplexityDetector functionality."""

    def test_detect_ultrathink_keywords(self):
        """Test detection of ultrathink keywords."""
        detector = ComplexityDetector()

        prompts = [
            "deeply analyze this codebase",
            "comprehensive review of all possibilities",
            "ultrathink about this problem",
        ]

        for prompt in prompts:
            result = detector.detect(prompt)
            assert result == TaskComplexity.ULTRATHINK, f"Failed for: {prompt}"

    def test_detect_complex_keywords(self):
        """Test detection of complex keywords."""
        detector = ComplexityDetector()

        prompts = [
            "analyze the architecture",
            "design a new system",
            "refactor this module",
        ]

        for prompt in prompts:
            result = detector.detect(prompt)
            assert result == TaskComplexity.COMPLEX, f"Failed for: {prompt}"

    def test_detect_simple_keywords(self):
        """Test detection of simple keywords."""
        detector = ComplexityDetector()

        prompts = [
            "simple question about Python",
            "just lookup the value",
            "what is the meaning of this",
        ]

        for prompt in prompts:
            result = detector.detect(prompt)
            assert result == TaskComplexity.SIMPLE, f"Failed for: {prompt}"

    def test_detect_medium_default(self):
        """Test that medium is the default when no keywords match but context is moderate."""
        detector = ComplexityDetector()

        # Use a phrase without any complexity indicator keywords
        # Also provide moderate file/tool counts to avoid the simple heuristic
        result = detector.detect("process the data and return results", file_count=2, tool_count=2)

        assert result == TaskComplexity.MEDIUM

    def test_detect_with_context_heuristics(self):
        """Test detection with file/tool count heuristics."""
        detector = ComplexityDetector()

        # Many files should be complex
        result = detector.detect("process these", file_count=15)
        assert result == TaskComplexity.COMPLEX

        # Few files/tools should be simple
        result = detector.detect("process these", file_count=1, tool_count=1)
        assert result == TaskComplexity.SIMPLE


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_select_model_for_task(self):
        """Test select_model_for_task function."""
        result = select_model_for_task("exploration")
        assert result == "haiku"

        result = select_model_for_task("coding", "complex")
        assert result == "opus"

    def test_detect_complexity(self):
        """Test detect_complexity function."""
        result = detect_complexity("deeply analyze everything")
        assert result == "ultrathink"

        result = detect_complexity("simple lookup")
        assert result == "simple"


class TestTaskToolResult:
    """Test TaskToolResult dataclass."""

    def test_successful_result(self):
        """Test creating a successful result."""
        result = TaskToolResult(
            subagent_name="explorer_1",
            subagent_type="explorer",
            success=True,
            result="Found 10 files",
            execution_time=1.5,
            tokens_used=500,
            cost=0.01,
        )

        assert result.success is True
        assert result.result == "Found 10 files"
        assert result.error is None

    def test_failed_result(self):
        """Test creating a failed result."""
        result = TaskToolResult(
            subagent_name="explorer_1",
            subagent_type="explorer",
            success=False,
            error="Timeout exceeded",
        )

        assert result.success is False
        assert result.result is None
        assert result.error == "Timeout exceeded"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = TaskToolResult(
            subagent_name="test",
            subagent_type="explorer",
            success=True,
            result="OK",
        )

        data = result.to_dict()

        assert data["subagent_name"] == "test"
        assert data["subagent_type"] == "explorer"
        assert data["success"] is True
        assert data["result"] == "OK"


class TestTaskToolConfig:
    """Test TaskToolConfig dataclass."""

    def test_defaults(self):
        """Test default configuration values."""
        config = TaskToolConfig()

        assert config.max_nesting_depth == MAX_NESTING_DEPTH
        assert config.default_timeout == DEFAULT_SUBAGENT_TIMEOUT
        assert config.allow_parallel is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TaskToolConfig(
            max_nesting_depth=2,
            default_timeout=60.0,
            allow_parallel=False,
        )

        assert config.max_nesting_depth == 2
        assert config.default_timeout == 60.0
        assert config.allow_parallel is False


class TestTaskTool:
    """Test TaskTool functionality."""

    def test_creation_with_defaults(self):
        """Test TaskTool creation with defaults."""
        tool = TaskTool()

        assert tool.name == "Task"
        assert tool.registry is not None
        assert tool.model_selector is not None

    def test_creation_with_custom_registry(self):
        """Test TaskTool creation with custom registry."""
        registry = SubAgentRegistry(register_defaults=False)
        registry.register(DynamicSubAgentConfig(
            name="custom",
            description="Custom",
            system_prompt="Prompt",
        ))

        tool = TaskTool(registry=registry)

        assert "custom" in tool.registry.names
        assert "explorer" not in tool.registry.names

    def test_parameters_schema(self):
        """Test parameters JSON schema."""
        tool = TaskTool()

        params = tool.parameters

        assert params["type"] == "object"
        assert "subagent_type" in params["properties"]
        assert "prompt" in params["properties"]
        assert "subagent_type" in params["required"]
        assert "prompt" in params["required"]

    def test_parameters_include_available_types(self):
        """Test that parameters include available subagent types."""
        tool = TaskTool()

        params = tool.parameters

        enum_values = params["properties"]["subagent_type"]["enum"]
        assert "explorer" in enum_values
        assert "researcher" in enum_values

    def test_to_dict(self):
        """Test tool definition dictionary."""
        tool = TaskTool()

        data = tool.to_dict()

        assert data["name"] == "Task"
        assert "description" in data
        assert "parameters" in data


class TestCreateTaskTool:
    """Test create_task_tool factory function."""

    def test_creates_task_tool(self):
        """Test factory creates TaskTool."""
        tool = create_task_tool()

        assert isinstance(tool, TaskTool)
        assert tool.registry is not None

    def test_with_custom_registry(self):
        """Test factory with custom registry."""
        registry = SubAgentRegistry(register_defaults=False)
        # Register at least one agent so registry is not falsy (len > 0)
        registry.register(DynamicSubAgentConfig(
            name="custom",
            description="Custom agent",
            system_prompt="Custom prompt",
        ))
        tool = create_task_tool(registry=registry)

        assert tool.registry is registry


class TestDefaultSubAgentPrompts:
    """Test that default subagent prompts are properly defined."""

    def test_explorer_prompt(self):
        """Test explorer prompt is defined."""
        assert EXPLORER_PROMPT is not None
        assert "exploration" in EXPLORER_PROMPT.lower() or "find" in EXPLORER_PROMPT.lower()

    def test_researcher_prompt(self):
        """Test researcher prompt is defined."""
        assert RESEARCHER_PROMPT is not None
        assert "research" in RESEARCHER_PROMPT.lower() or "information" in RESEARCHER_PROMPT.lower()

    def test_implementer_prompt(self):
        """Test implementer prompt is defined."""
        assert IMPLEMENTER_PROMPT is not None
        assert "implement" in IMPLEMENTER_PROMPT.lower() or "code" in IMPLEMENTER_PROMPT.lower()

    def test_reviewer_prompt(self):
        """Test reviewer prompt is defined."""
        assert REVIEWER_PROMPT is not None
        assert "review" in REVIEWER_PROMPT.lower()

    def test_planner_prompt(self):
        """Test planner prompt is defined."""
        assert PLANNER_PROMPT is not None
        assert "plan" in PLANNER_PROMPT.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
