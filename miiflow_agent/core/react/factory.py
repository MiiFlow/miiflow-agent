"""Simple factory for ReAct components."""

import logging
from typing import Optional, Literal

from .events import EventBus, EventFormat
from .orchestrator import ReActOrchestrator
from .safety import SafetyManager
from .tool_executor import AgentToolExecutor

logger = logging.getLogger(__name__)


class ReActFactory:
    """Simple factory for creating ReAct orchestrators."""

    @staticmethod
    def create_orchestrator(
        agent,
        max_steps: int = 25,
        max_budget: Optional[float] = None,
        max_time_seconds: Optional[float] = None,
        event_format: EventFormat = "react",
        thread_id: Optional[str] = None,
        message_id: Optional[str] = None,
        recovery_manager=None,
        context_compressor=None,
        tool_filter=None,
    ) -> ReActOrchestrator:
        """Create ReAct orchestrator with clean dependency injection.

        Args:
            agent: The agent instance
            max_steps: Maximum number of reasoning steps
            max_budget: Optional budget limit
            max_time_seconds: Optional time limit in seconds
            event_format: Event format - "react" for legacy, "agui" for AG-UI protocol
            thread_id: Thread ID (required for agui format)
            message_id: Message ID (required for agui format)
            recovery_manager: Optional RecoveryManager for graduated error recovery
            context_compressor: Optional ContextCompressor for context management
            tool_filter: Optional ToolFilter for restricting tool availability

        Returns:
            ReActOrchestrator instance
        """
        return ReActOrchestrator(
            tool_executor=AgentToolExecutor(agent, tool_filter=tool_filter),
            event_bus=EventBus(
                event_format=event_format,
                thread_id=thread_id,
                message_id=message_id,
            ),
            safety_manager=SafetyManager(
                max_steps=max_steps, max_budget=max_budget, max_time_seconds=max_time_seconds
            ),
            recovery_manager=recovery_manager,
            context_compressor=context_compressor,
            tool_filter=tool_filter,
        )
