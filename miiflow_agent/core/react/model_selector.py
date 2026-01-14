"""Model selection for subagent execution.

This module provides intelligent model selection based on task type,
complexity, and budget constraints.

Key features:
- Model capability profiles (haiku/sonnet/opus)
- Task-based model selection
- Complexity-aware selection
- Budget-constrained selection
- Cost estimation

Based on patterns from:
- Claude Agent SDK's model selection per agent
- Cost-optimized agent architectures
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ModelTier(str, Enum):
    """Model tiers for selection."""
    HAIKU = "haiku"
    SONNET = "sonnet"
    OPUS = "opus"


class TaskComplexity(str, Enum):
    """Task complexity levels."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    ULTRATHINK = "ultrathink"


@dataclass
class ModelCapabilities:
    """Capabilities and characteristics of a model tier."""
    tier: ModelTier
    speed: str  # fast, medium, slow
    cost_multiplier: float  # relative cost (1.0 = baseline)
    best_for: List[str]  # task types this model excels at
    context_window: int  # max tokens
    max_output: int  # max output tokens

    # Quality characteristics
    reasoning_quality: float  # 0-1 scale
    coding_quality: float  # 0-1 scale
    instruction_following: float  # 0-1 scale


# Model capability profiles
MODEL_PROFILES: Dict[ModelTier, ModelCapabilities] = {
    ModelTier.HAIKU: ModelCapabilities(
        tier=ModelTier.HAIKU,
        speed="fast",
        cost_multiplier=0.2,  # ~5x cheaper than sonnet
        best_for=[
            "exploration", "search", "lookup", "simple_tasks",
            "file_discovery", "pattern_matching", "quick_answers",
        ],
        context_window=200000,
        max_output=64000,
        reasoning_quality=0.7,
        coding_quality=0.75,
        instruction_following=0.85,
    ),
    ModelTier.SONNET: ModelCapabilities(
        tier=ModelTier.SONNET,
        speed="medium",
        cost_multiplier=1.0,  # baseline
        best_for=[
            "coding", "analysis", "complex_reasoning", "review",
            "implementation", "debugging", "refactoring",
        ],
        context_window=200000,
        max_output=64000,
        reasoning_quality=0.9,
        coding_quality=0.95,
        instruction_following=0.95,
    ),
    ModelTier.OPUS: ModelCapabilities(
        tier=ModelTier.OPUS,
        speed="slow",
        cost_multiplier=5.0,  # ~5x more expensive than sonnet
        best_for=[
            "architecture", "deep_analysis", "ultrathink", "complex_planning",
            "novel_problem_solving", "research_synthesis",
        ],
        context_window=200000,
        max_output=32000,
        reasoning_quality=1.0,
        coding_quality=0.98,
        instruction_following=0.98,
    ),
}


class ModelSelector:
    """Intelligent model selection for subagent tasks.

    Selects the optimal model based on:
    - Task type and complexity
    - Budget constraints
    - Required capabilities
    - Speed requirements

    Usage:
        selector = ModelSelector()

        # Select based on task type
        model = selector.select_for_task("exploration")  # Returns "haiku"

        # Select with complexity
        model = selector.select(
            task_type="coding",
            complexity=TaskComplexity.COMPLEX
        )  # Returns "opus"

        # Select with budget constraint
        model = selector.select(
            task_type="coding",
            max_cost_multiplier=1.0
        )  # Returns "sonnet" (won't pick opus due to cost)
    """

    def __init__(
        self,
        default_model: ModelTier = ModelTier.SONNET,
        profiles: Optional[Dict[ModelTier, ModelCapabilities]] = None,
    ):
        """Initialize the model selector.

        Args:
            default_model: Default model when no better match is found
            profiles: Custom model profiles (uses defaults if not provided)
        """
        self.default_model = default_model
        self.profiles = profiles or MODEL_PROFILES

        # Build reverse lookup: task type -> recommended model
        self._task_model_map: Dict[str, ModelTier] = {}
        for tier, capabilities in self.profiles.items():
            for task_type in capabilities.best_for:
                self._task_model_map[task_type] = tier

    def select(
        self,
        task_type: Optional[str] = None,
        complexity: TaskComplexity = TaskComplexity.MEDIUM,
        max_cost_multiplier: Optional[float] = None,
        require_fast: bool = False,
        min_reasoning_quality: Optional[float] = None,
        min_coding_quality: Optional[float] = None,
    ) -> str:
        """Select the optimal model based on criteria.

        Args:
            task_type: Type of task (e.g., "coding", "exploration")
            complexity: Task complexity level
            max_cost_multiplier: Maximum acceptable cost multiplier
            require_fast: Whether fast response is required
            min_reasoning_quality: Minimum reasoning quality (0-1)
            min_coding_quality: Minimum coding quality (0-1)

        Returns:
            Model tier name as string (e.g., "haiku", "sonnet", "opus")
        """
        # Get candidate models based on task type
        if task_type and task_type.lower() in self._task_model_map:
            task_model = self._task_model_map[task_type.lower()]
        else:
            task_model = self.default_model

        # Adjust based on complexity
        selected = self._adjust_for_complexity(task_model, complexity)

        # Apply constraints
        selected = self._apply_constraints(
            selected,
            max_cost_multiplier=max_cost_multiplier,
            require_fast=require_fast,
            min_reasoning_quality=min_reasoning_quality,
            min_coding_quality=min_coding_quality,
        )

        logger.debug(
            f"Selected model {selected.value} for task_type={task_type}, "
            f"complexity={complexity.value}"
        )

        return selected.value

    def select_for_task(self, task_type: str) -> str:
        """Simple selection based on task type only.

        Args:
            task_type: Type of task

        Returns:
            Model tier name as string
        """
        return self.select(task_type=task_type)

    def _adjust_for_complexity(
        self,
        base_model: ModelTier,
        complexity: TaskComplexity,
    ) -> ModelTier:
        """Adjust model selection based on complexity.

        Args:
            base_model: Initial model selection
            complexity: Task complexity

        Returns:
            Adjusted model tier
        """
        tier_order = [ModelTier.HAIKU, ModelTier.SONNET, ModelTier.OPUS]
        base_index = tier_order.index(base_model)

        if complexity == TaskComplexity.SIMPLE:
            # Can downgrade by 1 tier
            new_index = max(0, base_index - 1)
        elif complexity == TaskComplexity.COMPLEX:
            # Upgrade by 1 tier
            new_index = min(len(tier_order) - 1, base_index + 1)
        elif complexity == TaskComplexity.ULTRATHINK:
            # Always use opus for ultrathink
            new_index = len(tier_order) - 1
        else:
            # Medium complexity, use base
            new_index = base_index

        return tier_order[new_index]

    def _apply_constraints(
        self,
        selected: ModelTier,
        max_cost_multiplier: Optional[float] = None,
        require_fast: bool = False,
        min_reasoning_quality: Optional[float] = None,
        min_coding_quality: Optional[float] = None,
    ) -> ModelTier:
        """Apply constraints to model selection.

        Args:
            selected: Currently selected model
            max_cost_multiplier: Maximum acceptable cost
            require_fast: Whether fast response is required
            min_reasoning_quality: Minimum reasoning quality
            min_coding_quality: Minimum coding quality

        Returns:
            Constrained model tier
        """
        tier_order = [ModelTier.HAIKU, ModelTier.SONNET, ModelTier.OPUS]
        selected_index = tier_order.index(selected)

        # Check cost constraint
        if max_cost_multiplier is not None:
            for i in range(selected_index, -1, -1):
                if self.profiles[tier_order[i]].cost_multiplier <= max_cost_multiplier:
                    selected_index = i
                    break
            else:
                # All models exceed budget, use cheapest
                selected_index = 0

        # Check speed constraint
        if require_fast:
            # Haiku is the only "fast" model
            if self.profiles[tier_order[selected_index]].speed != "fast":
                selected_index = 0  # Force haiku

        # Check quality constraints
        if min_reasoning_quality is not None:
            for i in range(selected_index, len(tier_order)):
                if self.profiles[tier_order[i]].reasoning_quality >= min_reasoning_quality:
                    selected_index = i
                    break

        if min_coding_quality is not None:
            for i in range(selected_index, len(tier_order)):
                if self.profiles[tier_order[i]].coding_quality >= min_coding_quality:
                    selected_index = i
                    break

        return tier_order[selected_index]

    def get_capabilities(self, model: str) -> Optional[ModelCapabilities]:
        """Get capabilities for a model tier.

        Args:
            model: Model tier name

        Returns:
            ModelCapabilities if found, None otherwise
        """
        try:
            tier = ModelTier(model.lower())
            return self.profiles.get(tier)
        except ValueError:
            return None

    def estimate_cost(
        self,
        model: str,
        estimated_tokens: int,
        base_cost_per_1k: float = 0.003,
    ) -> float:
        """Estimate cost for a model run.

        Args:
            model: Model tier name
            estimated_tokens: Estimated total tokens
            base_cost_per_1k: Base cost per 1K tokens

        Returns:
            Estimated cost in USD
        """
        capabilities = self.get_capabilities(model)
        if not capabilities:
            return 0.0

        return (estimated_tokens / 1000) * base_cost_per_1k * capabilities.cost_multiplier

    def recommend_for_budget(
        self,
        budget_usd: float,
        estimated_tokens: int,
        base_cost_per_1k: float = 0.003,
    ) -> str:
        """Recommend the best model that fits within budget.

        Args:
            budget_usd: Maximum budget in USD
            estimated_tokens: Estimated total tokens
            base_cost_per_1k: Base cost per 1K tokens

        Returns:
            Best model tier name that fits budget
        """
        tier_order = [ModelTier.OPUS, ModelTier.SONNET, ModelTier.HAIKU]

        for tier in tier_order:
            estimated_cost = self.estimate_cost(
                tier.value, estimated_tokens, base_cost_per_1k
            )
            if estimated_cost <= budget_usd:
                return tier.value

        # Budget too low even for haiku
        return ModelTier.HAIKU.value


@dataclass
class ComplexityDetector:
    """Detect task complexity from prompt and context.

    Uses keyword matching and heuristics to determine task complexity.

    Usage:
        detector = ComplexityDetector()
        complexity = detector.detect("deeply analyze this codebase")
        # Returns TaskComplexity.COMPLEX
    """

    # Keyword indicators for each complexity level
    COMPLEXITY_INDICATORS: Dict[TaskComplexity, List[str]] = field(default_factory=lambda: {
        TaskComplexity.ULTRATHINK: [
            "deeply analyze", "comprehensive review", "all possibilities",
            "ultrathink", "thorough investigation", "consider everything",
            "exhaustive", "complete analysis", "every aspect",
        ],
        TaskComplexity.COMPLEX: [
            "analyze", "architect", "design", "refactor",
            "complex", "investigate", "research thoroughly",
            "debug", "optimize", "security review",
        ],
        TaskComplexity.MEDIUM: [
            "explain", "implement", "fix", "update",
            "modify", "create", "write", "add",
        ],
        TaskComplexity.SIMPLE: [
            "simple", "quick", "just", "only",
            "what is", "how to", "lookup", "find",
        ],
    })

    def detect(
        self,
        prompt: str,
        tool_count: int = 0,
        file_count: int = 0,
    ) -> TaskComplexity:
        """Detect complexity level from prompt and context.

        Args:
            prompt: The task prompt
            tool_count: Number of tools available
            file_count: Number of files involved

        Returns:
            Detected TaskComplexity level
        """
        prompt_lower = prompt.lower()

        # Check explicit indicators (in order of priority)
        for level in [
            TaskComplexity.ULTRATHINK,
            TaskComplexity.COMPLEX,
            TaskComplexity.SIMPLE,
            TaskComplexity.MEDIUM,
        ]:
            indicators = self.COMPLEXITY_INDICATORS.get(level, [])
            if any(ind in prompt_lower for ind in indicators):
                return level

        # Heuristic based on context
        if file_count > 10 or tool_count > 5:
            return TaskComplexity.COMPLEX
        elif file_count > 3 or tool_count > 2:
            return TaskComplexity.MEDIUM
        elif file_count <= 1 and tool_count <= 1:
            return TaskComplexity.SIMPLE

        return TaskComplexity.MEDIUM


# Convenience functions
def select_model_for_task(task_type: str, complexity: str = "medium") -> str:
    """Convenience function to select a model.

    Args:
        task_type: Type of task
        complexity: Complexity level string

    Returns:
        Model tier name
    """
    selector = ModelSelector()
    try:
        complexity_enum = TaskComplexity(complexity.lower())
    except ValueError:
        complexity_enum = TaskComplexity.MEDIUM

    return selector.select(task_type=task_type, complexity=complexity_enum)


def detect_complexity(prompt: str) -> str:
    """Convenience function to detect complexity.

    Args:
        prompt: The task prompt

    Returns:
        Complexity level string
    """
    detector = ComplexityDetector()
    return detector.detect(prompt).value
