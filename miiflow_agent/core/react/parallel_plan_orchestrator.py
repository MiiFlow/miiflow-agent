"""Parallel Plan orchestrator for wave-based parallel subtask execution.

DEPRECATED: Use PlanAndExecuteOrchestrator(parallel_execution=True) instead.

This module is maintained for backward compatibility only. New code should use
PlanAndExecuteOrchestrator with the parallel_execution flag set to True.
"""

import logging
import warnings
from typing import Any, Dict, Optional

from .events import EventBus
from .orchestrator import ReActOrchestrator
from .plan_execute_orchestrator import PlanAndExecuteOrchestrator
from .safety import SafetyManager
from .tool_executor import AgentToolExecutor

logger = logging.getLogger(__name__)


class ParallelPlanOrchestrator(PlanAndExecuteOrchestrator):
    """DEPRECATED: Parallel Plan orchestrator with wave-based execution.

    .. deprecated::
        Use ``PlanAndExecuteOrchestrator(parallel_execution=True)`` instead.

    This class is maintained for backward compatibility. It now simply wraps
    PlanAndExecuteOrchestrator with parallel_execution=True.

    All parallel execution logic has been consolidated into PlanAndExecuteOrchestrator.
    """

    def __init__(
        self,
        tool_executor: AgentToolExecutor,
        event_bus: EventBus,
        safety_manager: SafetyManager,
        subtask_orchestrator: Optional[ReActOrchestrator] = None,
        max_replans: int = 2,
        subtask_timeout_seconds: float = 360.0,
        max_parallel_subtasks: int = 5,
    ):
        """Initialize Parallel Plan orchestrator.

        .. deprecated::
            Use ``PlanAndExecuteOrchestrator(parallel_execution=True)`` instead.

        Args:
            tool_executor: Tool execution adapter
            event_bus: Event bus for streaming events
            safety_manager: Safety condition checker
            subtask_orchestrator: ReAct orchestrator for subtask execution
            max_replans: Maximum number of re-planning attempts
            subtask_timeout_seconds: Timeout for each subtask execution
            max_parallel_subtasks: Maximum subtasks to run in parallel per wave
        """
        warnings.warn(
            "ParallelPlanOrchestrator is deprecated. "
            "Use PlanAndExecuteOrchestrator(parallel_execution=True) instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        super().__init__(
            tool_executor=tool_executor,
            event_bus=event_bus,
            safety_manager=safety_manager,
            subtask_orchestrator=subtask_orchestrator,
            max_replans=max_replans,
            subtask_timeout_seconds=subtask_timeout_seconds,
            parallel_execution=True,  # Enable parallel execution
            max_parallel_subtasks=max_parallel_subtasks,
        )

        logger.info(
            "ParallelPlanOrchestrator is deprecated. "
            "Using PlanAndExecuteOrchestrator with parallel_execution=True."
        )

    def get_current_status(self) -> Dict[str, Any]:
        """Get current execution status."""
        status = super().get_current_status()
        # Override agent_type for backward compatibility
        status["agent_type"] = "parallel_plan_orchestrator"
        return status
